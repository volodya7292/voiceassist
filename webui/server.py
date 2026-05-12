"""Push-to-talk web UI for voiceassist.

Single endpoint POST /api/turn: browser sends a WAV blob + JSON-encoded
conversation history, server runs the same Whisper -> Ollama -> Silero
pipeline as bot.py and returns WAV audio with transcript / reply / updated
history in response headers.

Run via `webui/run.sh` (or directly: `uv run uvicorn webui.server:app
--host 127.0.0.1 --port 8888`).
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

WHISPER_URL = os.environ.get("WHISPER_URL", "http://127.0.0.1:8000/v1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "ru")
TTS_VOICE = os.environ.get("TTS_VOICE", "kseniya")
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))

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

# Silero loaded lazily on first request, then reused. Serialized by lock to
# mirror the bot's behavior.
_silero_model = None
_silero_lock = asyncio.Lock()


async def _get_silero():
    global _silero_model
    if _silero_model is None:
        from silero_tts_service import (
            DEFAULT_CACHE,
            DEFAULT_MODEL_URL,
            _ensure_model,
            _resolve_device,
        )

        local = _ensure_model(DEFAULT_MODEL_URL, DEFAULT_CACHE)
        device = _resolve_device(None)
        log.info("loading silero from %s on %s", local, device)
        model = await asyncio.to_thread(
            lambda: torch.package.PackageImporter(str(local)).load_pickle(
                "tts_models", "model"
            )
        )
        model.to(torch.device(device))
        _silero_model = model
        log.info("silero loaded")
    return _silero_model


def _synth_sync(model, text: str) -> bytes:
    audio = model.apply_tts(
        text=text,
        speaker=TTS_VOICE,
        sample_rate=TTS_SAMPLE_RATE,
        put_accent=True,
        put_yo=True,
    )
    arr = np.asarray(audio.detach().cpu().numpy(), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, TTS_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"ok": True, "llm_model": LLM_MODEL, "tts_voice": TTS_VOICE}


@app.post("/api/turn")
async def turn(
    audio: UploadFile = File(...),
    history: str = Form(default="[]"),
):
    try:
        messages = json.loads(history)
        if not isinstance(messages, list):
            messages = []
    except (TypeError, json.JSONDecodeError):
        messages = []

    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")

    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1. STT
        files = {
            "file": (
                audio.filename or "audio.wav",
                audio_bytes,
                audio.content_type or "audio/wav",
            )
        }
        data = {"language": STT_LANGUAGE}
        r = await client.post(
            f"{WHISPER_URL}/audio/transcriptions", files=files, data=data
        )
        r.raise_for_status()
        transcript = r.json().get("text", "").strip()
        if not transcript:
            raise HTTPException(status_code=400, detail="empty transcript")

        # 2. LLM
        messages.append({"role": "user", "content": transcript})
        r = await client.post(
            f"{OLLAMA_URL}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": False,
            },
        )
        r.raise_for_status()
        reply = (
            r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        ).strip()
        if not reply:
            reply = "Извините, я не понял."
        messages.append({"role": "assistant", "content": reply})

    # 3. TTS
    model = await _get_silero()
    async with _silero_lock:
        wav_bytes = await asyncio.to_thread(_synth_sync, model, reply)

    log.info(
        "turn: in=%d bytes out=%d bytes  transcript=%r  reply=%r",
        len(audio_bytes), len(wav_bytes), transcript[:60], reply[:60],
    )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Transcript": quote(transcript),
            "X-Reply": quote(reply),
            "X-History": quote(json.dumps(messages, ensure_ascii=False)),
        },
    )
