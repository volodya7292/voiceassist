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
from pathlib import Path

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

from silero_tts_service import SileroTTSService

WHISPER_URL = os.environ.get("WHISPER_URL", "http://127.0.0.1:8000/v1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
STT_MODEL = os.environ.get("STT_MODEL", "small")
TTS_VOICE = os.environ.get("TTS_VOICE", "kseniya")
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
WS_SAMPLE_RATE = 16000

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "Ты дружелюбный голосовой помощник. "
    "Отвечай по-русски одним или двумя короткими предложениями. "
    "Только обычный текст, без markdown, без эмодзи, без специальных символов.",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [webui] %(message)s")
log = logging.getLogger("webui")

WEBUI_DIR = Path(__file__).parent
STATIC_DIR = WEBUI_DIR / "static"

app = FastAPI(title="voiceassist webui")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class JsonPcmSerializer(FrameSerializer):
    """Custom serializer: raw PCM bytes for audio, JSON for control frames.

    See module docstring for the wire format.
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_sample_rate = WS_SAMPLE_RATE

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
        # Browser doesn't send text frames at the moment.
        return None


def build_pipeline(websocket: WebSocket) -> PipelineTask:
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=WS_SAMPLE_RATE,
            audio_out_sample_rate=TTS_SAMPLE_RATE,
            add_wav_header=False,
            serializer=JsonPcmSerializer(),
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

    tts = SileroTTSService(voice=TTS_VOICE, sample_rate=TTS_SAMPLE_RATE)

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
        "tts_voice": TTS_VOICE,
        "tts_sample_rate": TTS_SAMPLE_RATE,
        "ws_sample_rate": WS_SAMPLE_RATE,
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    log.info("client connected")
    task = build_pipeline(websocket)
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
