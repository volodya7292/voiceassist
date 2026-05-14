"""Streaming web UI for voiceassist — Pipecat-powered.

Same Pipecat Pipeline as `bot.py`, but with `FastAPIWebsocketTransport`
in place of `LocalAudioTransport`. Each browser tab opens one WebSocket;
the server spins up an independent Pipeline per connection.

Wire protocol (JsonPcmSerializer):
  - Binary frames: raw int16 little-endian PCM, mono.
      * Browser -> server: 16 kHz mic stream.
      * Server -> browser: 24 kHz TTS output.
  - Text frames: JSON control messages emitted by Pipecat:
      {"type": "user_speaking",  "value": true|false}
      {"type": "bot_speaking",   "value": true|false}
      {"type": "transcript",     "text": "..."}
      {"type": "reply_chunk",    "text": "..."}    # sentence
      {"type": "interruption"}                     # barge-in fired

Barge-in is automatic: VADProcessor sits in the pipeline; when it detects
the user speaking, Pipecat emits InterruptionFrame, which cancels in-flight
TTS, flushes the output transport's queue, and tells the browser to drop
its scheduled audio playback.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # noqa: E402 — must run before tts_backends reads env

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

import httpx

from system_prompt import load_system_prompt
from tts_backends import VOICE_ALIASES, build_tts

WHISPER_URL = os.environ.get("WHISPER_URL", "http://127.0.0.1:8000/v1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
STT_MODEL = os.environ.get("STT_MODEL", "small")
TTS_BACKEND = os.environ.get("TTS_BACKEND", "supertonic").lower()
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
WS_SAMPLE_RATE = 16000

# Supertonic exposes 5 female voice styles (the bot persona is female).
SUPERTONIC_F_VOICES = ("F1", "F2", "F3", "F4", "F5")

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT") or load_system_prompt()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [webui] %(message)s")
log = logging.getLogger("webui")

WEBUI_DIR = Path(__file__).parent
STATIC_DIR = WEBUI_DIR / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Preload the heavy Pipecat models on server start.

    Without this, the first WebSocket connection blocks for ~100-200ms while
    SileroVADAnalyzer and LocalSmartTurnAnalyzerV3 each cold-load their ONNX
    models from disk. With this, both model files are in OS page cache and
    their ONNX runtimes have been touched before the user clicks Start.
    """

    def _warm_in_thread() -> None:
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
            LocalSmartTurnAnalyzerV3,
        )

        SileroVADAnalyzer(params=VADParams(stop_secs=0.5))
        LocalSmartTurnAnalyzerV3()

    log.info("preloading pipecat VAD + smart-turn models...")
    t0 = time.perf_counter()
    await asyncio.to_thread(_warm_in_thread)
    log.info(f"  preloaded in {time.perf_counter() - t0:.2f}s")
    yield


app = FastAPI(title="voiceassist webui", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class JsonPcmSerializer(FrameSerializer):
    """Custom serializer: raw PCM bytes for audio, JSON for control frames.

    Client may send JSON control messages on the text channel; currently the
    only one understood is `{"type":"set_voice","voice":"..."}`, which mutates
    the bound TTS service in place so the next synthesized sentence uses the
    new voice (no pipeline restart).

    See module docstring for the wire format.
    """

    def __init__(self, tts_service=None) -> None:
        super().__init__()
        self._in_sample_rate = WS_SAMPLE_RATE
        self._tts_service = tts_service

    async def setup(self, frame: StartFrame) -> None:
        if frame.audio_in_sample_rate:
            self._in_sample_rate = frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        # Audio frames -> binary
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio

        # Control frames -> JSON text
        if isinstance(frame, TranscriptionFrame):
            return json.dumps(
                {"type": "transcript", "text": frame.text}, ensure_ascii=False
            )
        if isinstance(frame, TTSTextFrame):
            return json.dumps(
                {"type": "reply_chunk", "text": frame.text}, ensure_ascii=False
            )
        if isinstance(frame, UserStartedSpeakingFrame):
            return json.dumps({"type": "user_speaking", "value": True})
        if isinstance(frame, UserStoppedSpeakingFrame):
            return json.dumps({"type": "user_speaking", "value": False})
        if isinstance(frame, BotStartedSpeakingFrame):
            return json.dumps({"type": "bot_speaking", "value": True})
        if isinstance(frame, BotStoppedSpeakingFrame):
            return json.dumps({"type": "bot_speaking", "value": False})
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"type": "interruption"})

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)):
            return InputAudioRawFrame(
                audio=bytes(data),
                sample_rate=self._in_sample_rate,
                num_channels=1,
            )
        # Text frame: JSON control message.
        if isinstance(data, str):
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                return None
            if msg.get("type") == "set_voice":
                new_voice = (msg.get("voice") or "").strip()
                setter = getattr(self._tts_service, "set_voice", None)
                if new_voice and setter is not None:
                    setter(new_voice)
                elif new_voice:
                    log.warning(
                        f"set_voice request ignored: {type(self._tts_service).__name__} "
                        f"doesn't support live voice change"
                    )
        return None


def build_pipeline(websocket: WebSocket, voice: str | None = None) -> PipelineTask:
    # Build TTS first so the serializer can route set_voice messages to it.
    tts = build_tts(sample_rate=TTS_SAMPLE_RATE, voice=voice)

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=WS_SAMPLE_RATE,
            audio_out_sample_rate=TTS_SAMPLE_RATE,
            add_wav_header=False,
            serializer=JsonPcmSerializer(tts_service=tts),
        ),
    )

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.7,
                start_secs=0.15,
                stop_secs=0.5,
                min_volume=0.6,
            )
        )
    )

    stt = OpenAISTTService(
        api_key="not-needed",
        base_url=WHISPER_URL,
        settings=OpenAISTTService.Settings(model=STT_MODEL, language=Language.RU),
    )

    llm = OpenAILLMService(
        api_key="ollama",
        base_url=OLLAMA_URL,
        settings=OpenAILLMService.Settings(model=LLM_MODEL),
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    aggregators = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=WS_SAMPLE_RATE,
            audio_out_sample_rate=TTS_SAMPLE_RATE,
        ),
    )

    @transport.event_handler("on_client_disconnected")
    async def _on_disc(_t, _ws):  # noqa: ARG001
        log.info("client disconnected, cancelling pipeline")
        await task.cancel()

    return task


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "llm_model": LLM_MODEL,
        "stt_model": STT_MODEL,
        "tts_backend": TTS_BACKEND,
        "tts_sample_rate": TTS_SAMPLE_RATE,
        "ws_sample_rate": WS_SAMPLE_RATE,
    }


@app.get("/api/voices")
async def voices():
    """Return the list of voice names available for the TTS selector.

    For Supertonic (default): the 5 female voice styles F1-F5.
    For intelliscrape: upstream /voices endpoint, with static alias-map fallback.
    """
    if TTS_BACKEND in {"supertonic", "supertonic-3", "s3"}:
        return {"voices": list(SUPERTONIC_F_VOICES)}

    base = os.environ.get("INTELLISCRAPE_TTS_URL", "https://tts.intelliscrape.com").rstrip("/")
    token = os.environ.get("INTELLISCRAPE_TTS_TOKEN", "")
    if token:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(
                    f"{base}/voices",
                    headers={"authorization": f"Bearer {token}"},
                )
                if r.status_code == 200:
                    data = r.json()
                    # Normalize: support [str], {"voices":[...]}, or [{"name":...}]
                    if isinstance(data, list):
                        names = [
                            v if isinstance(v, str) else v.get("name") or v.get("id")
                            for v in data
                        ]
                    elif isinstance(data, dict) and "voices" in data:
                        raw = data["voices"]
                        names = [
                            v if isinstance(v, str) else v.get("name") or v.get("id")
                            for v in raw
                        ]
                    else:
                        names = []
                    names = [n for n in names if n]
                    if names:
                        return {"voices": sorted(set(names))}
        except Exception as e:
            log.warning(f"/api/voices: upstream fetch failed: {e}")
    # Fallback: the static alias keys (these are the personas, not API voices)
    return {"voices": sorted(set(VOICE_ALIASES.values()))}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    voice = websocket.query_params.get("voice") or None
    log.info(f"client connected (voice={voice or 'default'})")
    task = build_pipeline(websocket, voice=voice)
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    except Exception:
        log.exception("pipeline error")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        log.info("session ended")
