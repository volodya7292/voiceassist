"""OpenAI-compatible /v1/audio/transcriptions backed by mlx-whisper.

Single-worker FastAPI app. Model weights are lazily downloaded from
Hugging Face on first request and cached under ~/.cache/huggingface.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from typing import Optional

import mlx_whisper
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

MODEL = os.environ.get(
    "WHISPER_MODEL_REPO",
    "mlx-community/whisper-large-v3-turbo",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [whisper] %(message)s")
log = logging.getLogger("whisper")

app = FastAPI()

# mlx-whisper is not thread-safe; serialize requests.
_inference_lock = asyncio.Lock()


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL}


def _transcribe_sync(path: str, language: Optional[str]) -> dict:
    kwargs = {"path_or_hf_repo": MODEL}
    if language:
        kwargs["language"] = language
    return mlx_whisper.transcribe(path, **kwargs)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="distil-large-v3"),
    language: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: Optional[float] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
):
    del model, temperature, prompt  # accepted for OpenAI compat, ignored

    data = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        async with _inference_lock:
            t0 = time.perf_counter()
            result = await asyncio.to_thread(_transcribe_sync, tmp_path, language)
            dt = time.perf_counter() - t0
        text = (result.get("text") or "").strip()
        log.info("transcribed %d bytes -> %d chars in %.2fs", len(data), len(text), dt)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if response_format == "text":
        return text

    return JSONResponse({"text": text})
