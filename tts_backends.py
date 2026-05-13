"""TTS backend selector.

Picks between in-process backends by platform:
  - darwin                 -> Piper (ONNX, CPU/Metal, in-process)
  - linux / win (CUDA)     -> Qwen3-TTS (voice cloning, GPU, in-process)

Override the default with TTS_BACKEND=piper or TTS_BACKEND=qwen.
"""
from __future__ import annotations

import os
import sys

from pipecat.services.tts_service import TTSService


def _default_backend() -> str:
    return "piper" if sys.platform == "darwin" else "qwen"


def build_tts(*, sample_rate: int) -> TTSService:
    backend = os.environ.get("TTS_BACKEND", _default_backend()).lower()

    if backend == "piper":
        from piper_tts_service import PiperTTSService

        return PiperTTSService(
            voice=os.environ.get("TTS_VOICE", "ru_RU-irina-medium"),
            sample_rate=sample_rate,
        )

    if backend in {"qwen", "qwen3", "qwen-tts"}:
        from qwen_tts_service import (
            DEFAULT_MODEL,
            DEFAULT_LANGUAGE,
            QwenTTSService,
        )

        ref_audio = os.environ.get("QWEN_REF_AUDIO_PATH", "Danchenko.wav")
        ref_text = os.environ.get("QWEN_REF_TEXT")
        if not ref_text:
            ref_txt_path = os.environ.get("QWEN_REF_TEXT_PATH", "Danchenko.txt")
            try:
                with open(ref_txt_path, encoding="utf-8") as f:
                    ref_text = f.read().strip()
            except FileNotFoundError:
                ref_text = ""

        return QwenTTSService(
            reference_audio_path=ref_audio,
            reference_text=ref_text,
            model_name=os.environ.get("QWEN_MODEL", DEFAULT_MODEL),
            language=os.environ.get("TTS_LANG", DEFAULT_LANGUAGE),
            sample_rate=sample_rate,
            attn_implementation=os.environ.get("QWEN_ATTN", "sdpa"),
        )

    raise ValueError(
        f"unknown TTS_BACKEND={backend!r}; supported: piper, qwen"
    )
