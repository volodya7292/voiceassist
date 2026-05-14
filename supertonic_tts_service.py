"""In-process Pipecat TTS service backed by Supertonic v3.

Supertonic is a 99M-param multilingual neural TTS that runs on ONNX
Runtime. On first synth the model (~400 MB) is downloaded from Hugging
Face into the package's asset cache (~/.cache/supertonic by default).
CPU is fine on Apple Silicon; ONNX Runtime auto-selects available
execution providers (CoreML/CUDA if its EP wheel is installed).

Voices: 10 fixed styles (F1-F5 + M1-M5). Expressive tags `<laugh>`,
`<breath>`, `<sigh>` are honored inline in the input text.

Sample rate is fixed by the model (44.1 kHz); Pipecat's output transport
resamples to its `audio_out_sample_rate` automatically, so callers don't
need to know the native rate.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

DEFAULT_VOICE = "F5"
DEFAULT_LANG = "ru"
NATIVE_SAMPLE_RATE = 44100  # fixed by the supertonic-3 vocoder
SUPPORTED_VOICES = ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5")


class SupertonicTTSService(TTSService):
    """Local TTS via Supertonic v3 (ONNX). Multilingual, expressive tags."""

    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        lang: str = DEFAULT_LANG,
        sample_rate: int | None = None,
        total_steps: int = 8,
        speed: float = 1.05,
        chunk_ms: int = 100,
        **kwargs,
    ) -> None:
        # The vocoder is fixed at 44.1 kHz; tell Pipecat the real rate so its
        # output transport resamples cleanly to audio_out_sample_rate. Any
        # caller-supplied sample_rate is ignored (the build_tts helper passes
        # the transport's target rate, which is the wrong layer to plumb here).
        if sample_rate is not None and sample_rate != NATIVE_SAMPLE_RATE:
            logger.info(
                f"supertonic: ignoring sample_rate={sample_rate}; "
                f"using native {NATIVE_SAMPLE_RATE} (resampled downstream)"
            )
        super().__init__(
            sample_rate=NATIVE_SAMPLE_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model="supertonic-3", voice=voice, language=lang),
            **kwargs,
        )
        if voice not in SUPPORTED_VOICES:
            logger.warning(
                f"supertonic: unknown voice {voice!r}; falling back to {DEFAULT_VOICE}"
            )
            voice = DEFAULT_VOICE
        self._voice = voice
        self._lang = lang
        self._total_steps = total_steps
        self._speed = speed
        self._chunk_samples = max(1, int(NATIVE_SAMPLE_RATE * chunk_ms / 1000))

        self._tts = None
        self._style = None
        self._loaded_voice: str | None = None
        self._inference_lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    def set_voice(self, voice: str) -> None:
        """Swap the active voice. Takes effect on the next synthesis call."""
        voice = (voice or "").strip()
        if not voice or voice == self._voice:
            return
        if voice not in SUPPORTED_VOICES:
            logger.warning(f"supertonic: unknown voice {voice!r}; ignoring")
            return
        logger.info(f"supertonic: voice {self._voice} -> {voice}")
        self._voice = voice

    def _load_sync(self) -> None:
        from supertonic import TTS  # heavyweight import; ~400 MB on first run

        logger.info("supertonic: loading model (auto_download=True)")
        self._tts = TTS(auto_download=True)
        sr = getattr(self._tts, "sample_rate", None)
        if isinstance(sr, int) and sr > 0 and sr != self.sample_rate:
            logger.warning(
                f"supertonic: model sr={sr} != configured {self.sample_rate}; "
                f"Pipecat output transport will resample"
            )

    def _ensure_style_sync(self) -> None:
        if self._loaded_voice == self._voice and self._style is not None:
            return
        logger.info(f"supertonic: loading voice style {self._voice}")
        self._style = self._tts.get_voice_style(self._voice)
        self._loaded_voice = self._voice

    async def _ensure_loaded(self) -> None:
        if self._tts is None:
            await asyncio.to_thread(self._load_sync)
        await asyncio.to_thread(self._ensure_style_sync)

    def _synthesize_sync(self, text: str) -> bytes:
        wav, _duration = self._tts.synthesize(
            text,
            voice_style=self._style,
            total_steps=self._total_steps,
            speed=self._speed,
            lang=self._lang,
        )
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]  # [batch, T] -> first batch
        elif arr.ndim != 1:
            arr = arr.reshape(-1)
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype("<i2").tobytes()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = (text or "").strip()
        if not text:
            return
        logger.debug(f"supertonic: synthesizing [{text}] voice={self._voice}")

        try:
            await self._ensure_loaded()
            async with self._inference_lock:
                pcm = await asyncio.to_thread(self._synthesize_sync, text)
        except (RuntimeError, ValueError) as e:
            logger.exception(f"supertonic: synthesis error: {e}")
            yield ErrorFrame(error=f"supertonic: {e}")
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
