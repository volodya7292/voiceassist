"""In-process Pipecat TTS service backed by Piper.

Piper is an ONNX-based neural TTS. We use the `piper-tts` PyPI package and
ship a single voice file (`<voice>.onnx` + `.onnx.json`) per language/speaker.

Voice naming follows the rhasspy/piper-voices convention,
e.g. `ru_RU-irina-medium`. The file is auto-downloaded on first use into
`~/.cache/piper` (override with `PIPER_CACHE`).

Device: Piper's Python package runs on ONNX Runtime. CPU is fast enough on
Apple Silicon (RTF << 1 for the "medium" quality). CUDA can be enabled via
`PIPER_USE_CUDA=1` if the host has a working onnxruntime-gpu install.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

DEFAULT_VOICE = "ru_RU-irina-medium"
DEFAULT_CACHE = Path(os.environ.get("PIPER_CACHE", "~/.cache/piper")).expanduser()
DEFAULT_SAMPLE_RATE = 22050  # Piper "medium" quality voices output 22050 Hz


def _resolve_use_cuda(requested: bool | None) -> bool:
    if requested is not None:
        return requested
    return os.environ.get("PIPER_USE_CUDA", "0") not in ("0", "", "false", "False")


def _ensure_model(voice: str, cache_dir: Path) -> Path:
    """Make sure <voice>.onnx and <voice>.onnx.json are present in cache_dir.

    Returns the path to the .onnx file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_dir / f"{voice}.onnx"
    json_path = cache_dir / f"{voice}.onnx.json"
    if not onnx_path.is_file() or not json_path.is_file():
        from piper.download_voices import download_voice

        logger.info(f"piper: downloading voice {voice} -> {cache_dir}")
        download_voice(voice, download_dir=cache_dir)
    return onnx_path


class PiperTTSService(TTSService):
    """Local TTS via Piper (one voice file per language/speaker)."""

    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        cache_dir: Path | str = DEFAULT_CACHE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        use_cuda: bool | None = None,
        chunk_ms: int = 100,
        **kwargs,
    ) -> None:
        # Piper's sample rate is fixed by the voice file; pass it through to
        # Pipecat so the output transport can resample if it differs.
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model="piper", voice=voice, language=None),
            **kwargs,
        )
        self._voice = voice
        self._cache_dir = Path(cache_dir).expanduser()
        self._use_cuda = _resolve_use_cuda(use_cuda)
        self._chunk_samples = max(1, int(sample_rate * chunk_ms / 1000))
        self._voice_obj = None
        self._inference_lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    def _load_voice_sync(self):
        from piper import PiperVoice

        onnx_path = _ensure_model(self._voice, self._cache_dir)
        logger.info(f"piper: loading {onnx_path} (cuda={self._use_cuda})")
        voice = PiperVoice.load(onnx_path, use_cuda=self._use_cuda)
        logger.info(f"piper: loaded {self._voice}, sample_rate={voice.config.sample_rate}")
        return voice

    async def _ensure_loaded(self):
        if self._voice_obj is None:
            self._voice_obj = await asyncio.to_thread(self._load_voice_sync)

    def _synthesize_sync(self, text: str) -> bytes:
        # PiperVoice.synthesize() returns an iterator of AudioChunk; for our
        # short utterances we collect into one bytes buffer and let the async
        # side re-chunk for steady playback pacing.
        parts: list[bytes] = []
        for chunk in self._voice_obj.synthesize(text):
            parts.append(chunk.audio_int16_bytes)
        return b"".join(parts)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = (text or "").strip()
        if not text:
            return
        logger.debug(f"piper: synthesizing [{text}]")

        try:
            await self._ensure_loaded()
            async with self._inference_lock:
                pcm = await asyncio.to_thread(self._synthesize_sync, text)
        except (ValueError, RuntimeError) as e:
            logger.exception(f"piper: synthesis error: {e}")
            yield ErrorFrame(error=f"piper: {e}")
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
