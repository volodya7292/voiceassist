"""In-process Pipecat TTS service backed by Qwen3-TTS (Alibaba).

Voice-cloning TTS — speech timbre comes from a short reference WAV plus its
transcript. The model and tokenizer auto-download on first use from Hugging
Face. CUDA-only (the model expects a GPU); the service raises at startup
if `torch.cuda.is_available()` is False, so don't enable this backend on
hosts without a GPU.

Defaults: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, language `Russian`, bf16, SDPA
attention (no flash-attn build dependency).

Reference voice: any short clean recording with its exact transcript.
We ship `Danchenko.wav` + `Danchenko.txt` in the repo root.

Wire shape:
  - `model.generate_voice_clone(text, language, ref_audio, ref_text)`
    returns `(wavs, sr)` where `wavs[0]` is a 1-D numpy array (float32 or
    int16 depending on transformers version) and `sr` is the model's
    output sample rate.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_LANGUAGE = "Russian"
DEFAULT_SAMPLE_RATE = 24000  # Qwen3-TTS-12Hz emits 24 kHz audio (12 Hz token rate)

# Maps short ISO codes to the strings Qwen3-TTS expects.
LANGUAGE_NAMES = {
    "en": "English",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def _resolve_language(lang: str) -> str:
    """Allow either ISO codes ('ru') or full names ('Russian')."""
    if not lang:
        return DEFAULT_LANGUAGE
    if lang in LANGUAGE_NAMES.values():
        return lang
    return LANGUAGE_NAMES.get(lang.lower(), lang)


class QwenTTSService(TTSService):
    """Voice-cloning TTS via Qwen3-TTS (CUDA only)."""

    def __init__(
        self,
        *,
        reference_audio_path: str | Path,
        reference_text: str,
        model_name: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        attn_implementation: str = "sdpa",
        chunk_ms: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model=model_name, voice="cloned", language=None),
            **kwargs,
        )
        ref_path = Path(reference_audio_path).expanduser()
        if not ref_path.is_file():
            raise FileNotFoundError(
                f"qwen3-tts reference audio not found: {ref_path}. "
                "Set QWEN_REF_AUDIO_PATH to a valid file or pass it explicitly."
            )
        ref_text = (reference_text or "").strip()
        if not ref_text:
            raise ValueError(
                "qwen3-tts requires the reference clip's transcript. "
                "Set QWEN_REF_TEXT or pass reference_text."
            )
        self._reference_audio_path = str(ref_path)
        self._reference_text = ref_text
        self._model_name = model_name
        self._language = _resolve_language(language)
        self._attn_implementation = attn_implementation
        self._chunk_samples = max(1, int(sample_rate * chunk_ms / 1000))
        self._model = None
        self._inference_lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    def _load_model_sync(self):
        import torch
        from qwen_tts import Qwen3TTSModel

        if not torch.cuda.is_available():
            raise RuntimeError(
                "qwen3-tts requires CUDA; torch.cuda.is_available() returned False. "
                "Switch to a CUDA host or use a different TTS_BACKEND."
            )

        logger.info(f"qwen3-tts: loading {self._model_name} (cuda, bf16, attn={self._attn_implementation})")
        model = Qwen3TTSModel.from_pretrained(
            self._model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation=self._attn_implementation,
        )
        logger.info(
            f"qwen3-tts: loaded; ref={Path(self._reference_audio_path).name}, "
            f"lang={self._language}"
        )
        return model

    async def _ensure_loaded(self):
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model_sync)

    def _synthesize_sync(self, text: str) -> bytes:
        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=self._language,
            ref_audio=self._reference_audio_path,
            ref_text=self._reference_text,
        )
        if sr != self.sample_rate:
            logger.warning(
                f"qwen3-tts: model returned sr={sr}, configured={self.sample_rate}; "
                f"Pipecat will resample downstream"
            )
        arr = np.asarray(wavs[0]).reshape(-1)
        if arr.dtype != np.int16:
            arr = np.clip(arr.astype(np.float32), -1.0, 1.0)
            arr = (arr * 32767.0).astype("<i2")
        return arr.tobytes()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = (text or "").strip()
        if not text:
            return
        logger.debug(f"qwen3-tts: synthesizing [{text}]")

        try:
            await self._ensure_loaded()
            async with self._inference_lock:
                pcm = await asyncio.to_thread(self._synthesize_sync, text)
        except (ValueError, RuntimeError) as e:
            logger.exception(f"qwen3-tts: synthesis error: {e}")
            yield ErrorFrame(error=f"qwen3-tts: {e}")
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
