"""In-process Pipecat TTS service backed by Silero v5_4_ru.

The Silero TTS model is a torch.package archive. We download it on first use
into a local cache and load it in the constructor. Device selection:

  * SILERO_DEVICE env var, if set ("cpu", "cuda", "cuda:0", "mps", ...)
  * else "cuda" when torch.cuda.is_available()
  * else "cpu"

On Apple Silicon CPU is plenty (RTF ~0.01); on a CUDA host with a recent GPU
this gets a few-x faster but is rarely the bottleneck.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

DEFAULT_MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_4_ru.pt"
DEFAULT_CACHE = Path(os.environ.get("SILERO_CACHE", "~/.cache/silero")).expanduser()
DEFAULT_SAMPLE_RATE = 24000  # Silero supports 8000 / 24000 / 48000


def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    env = os.environ.get("SILERO_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ensure_model(model_url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / Path(model_url).name
    if not local.is_file():
        logger.info(f"silero: downloading {model_url} -> {local}")
        torch.hub.download_url_to_file(model_url, str(local))
    return local


class SileroTTSService(TTSService):
    """Local Russian TTS via Silero v5 models."""

    def __init__(
        self,
        *,
        voice: str = "kseniya",
        model_url: str = DEFAULT_MODEL_URL,
        cache_dir: Path | str = DEFAULT_CACHE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        device: str | None = None,
        put_accent: bool = True,
        put_yo: bool = True,
        chunk_ms: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model="silero_v5_ru", voice=voice, language="ru"),
            **kwargs,
        )
        self._model_url = model_url
        self._cache_dir = Path(cache_dir).expanduser()
        self._device = _resolve_device(device)
        self._put_accent = put_accent
        self._put_yo = put_yo
        self._chunk_samples = max(1, int(sample_rate * chunk_ms / 1000))
        self._model = None
        self._inference_lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    def _load_model_sync(self):
        local = _ensure_model(self._model_url, self._cache_dir)
        logger.info(f"silero: loading {local} on {self._device}")
        model = torch.package.PackageImporter(str(local)).load_pickle("tts_models", "model")
        model.to(torch.device(self._device))
        logger.info(
            f"silero: loaded on {self._device}; "
            f"speakers={model.speakers[:8]}{'...' if len(model.speakers) > 8 else ''}"
        )
        return model

    async def _ensure_loaded(self):
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model_sync)

    def _synthesize_sync(self, text: str) -> bytes:
        audio = self._model.apply_tts(
            text=text,
            speaker=self._settings.voice,
            sample_rate=self.sample_rate,
            put_accent=self._put_accent,
            put_yo=self._put_yo,
        )
        arr = np.asarray(audio.detach().cpu().numpy(), dtype=np.float32)
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype("<i2").tobytes()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = (text or "").strip()
        if not text:
            return
        logger.debug(f"silero: synthesizing [{text}]")

        try:
            await self._ensure_loaded()
            async with self._inference_lock:
                pcm = await asyncio.to_thread(self._synthesize_sync, text)
        except (ValueError, RuntimeError) as e:
            logger.exception(f"silero: synthesis error: {e}")
            yield ErrorFrame(error=f"silero: {e}")
            return

        await self.start_tts_usage_metrics(text)

        bytes_per_chunk = self._chunk_samples * 2  # int16
        first = True
        for start in range(0, len(pcm), bytes_per_chunk):
            chunk = pcm[start : start + bytes_per_chunk]
            if not chunk:
                continue
            if first:
                await self.stop_ttfb_metrics()
                first = False
            yield TTSAudioRawFrame(
                audio=chunk,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )
