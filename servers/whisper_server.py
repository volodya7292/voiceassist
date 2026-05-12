"""OpenAI-compatible /v1/audio/transcriptions.

Backend is picked at import time by platform:

  * Apple Silicon:  mlx-whisper (Metal). Single-worker (MLX serializes).
  * Everything else: faster-whisper (CTranslate2 + CUDA when available).
    Loads a pool of N WhisperModel instances; concurrent requests run in
    parallel up to the pool size.

Environment:
  WHISPER_MODEL          short name or HF path (default: "small")
  WHISPER_MODEL_REPO     Apple-only MLX repo (default: "mlx-community/whisper-large-v3-turbo")
  WHISPER_DEVICE         faster-whisper device ("cuda" | "cpu" | "auto"; default "auto")
  WHISPER_COMPUTE        faster-whisper compute type (default: "float16" on cuda, "int8" on cpu)
  WHISPER_WORKERS        faster-whisper model-pool size (default: 2)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [whisper] %(message)s")
log = logging.getLogger("whisper")

BACKEND = "mlx" if sys.platform == "darwin" else "faster"
app = FastAPI()


if BACKEND == "mlx":
    # ----- Apple Silicon: MLX backend ----------------------------------------
    import mlx_whisper

    MODEL = os.environ.get(
        "WHISPER_MODEL_REPO",
        "mlx-community/whisper-large-v3-turbo",
    )
    WORKERS = 1
    _mlx_lock = asyncio.Lock()

    def _transcribe_sync(path: str, language: Optional[str]) -> str:
        kwargs: dict[str, Any] = {"path_or_hf_repo": MODEL}
        if language:
            kwargs["language"] = language
        result = mlx_whisper.transcribe(path, **kwargs)
        return (result.get("text") or "").strip()

    async def transcribe(path: str, language: Optional[str]) -> str:
        async with _mlx_lock:
            return await asyncio.to_thread(_transcribe_sync, path, language)

else:
    # ----- Linux / Windows: faster-whisper with a model pool -----------------
    from faster_whisper import WhisperModel

    MODEL = os.environ.get("WHISPER_MODEL", "small")
    DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
    COMPUTE = os.environ.get(
        "WHISPER_COMPUTE",
        "float16" if DEVICE != "cpu" else "int8",
    )
    WORKERS = int(os.environ.get("WHISPER_WORKERS", "2"))

    _pool: Queue = Queue()
    _executor: Optional[ThreadPoolExecutor] = None

    @app.on_event("startup")
    async def _load() -> None:
        global _executor
        log.info(
            "loading %d workers: model=%s device=%s compute=%s",
            WORKERS, MODEL, DEVICE, COMPUTE,
        )
        for i in range(WORKERS):
            t0 = time.perf_counter()
            m = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
            log.info("  worker %d loaded in %.1fs", i + 1, time.perf_counter() - t0)
            _pool.put(m)
        _executor = ThreadPoolExecutor(
            max_workers=WORKERS,
            thread_name_prefix="whisper",
        )
        log.info("ready: %d workers", WORKERS)

    def _transcribe_sync(path: str, language: Optional[str]) -> str:
        model = _pool.get()
        try:
            segments, _info = model.transcribe(path, language=language)
            return " ".join(seg.text for seg in segments).strip()
        finally:
            _pool.put(model)

    async def transcribe(path: str, language: Optional[str]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, _transcribe_sync, path, language)


# ----- shared HTTP surface ---------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "backend": BACKEND,
        "model": MODEL,
        "workers": WORKERS,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
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
        t0 = time.perf_counter()
        text = await transcribe(tmp_path, language)
        dt = time.perf_counter() - t0
        log.info("transcribed %d bytes -> %d chars in %.2fs", len(data), len(text), dt)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if response_format == "text":
        return text
    return JSONResponse({"text": text})
