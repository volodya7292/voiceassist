"""TTS backend selector.

Default: supertonic (local Supertonic v3 ONNX; multilingual + expressive
tags `<laugh>`/`<breath>`/`<sigh>`). Other backends:
  - `TTS_BACKEND=intelliscrape` — remote API (works on any host).
  - `TTS_BACKEND=piper` — offline fallback via local Piper ONNX.
"""
from __future__ import annotations

import os

from pipecat.services.tts_service import TTSService

# Friendly persona names -> the intelliscrape API's voice name.
VOICE_ALIASES = {
    "serhii": "serhii",
    "yurii": "tryus",
    "olena": "danchenko",
    "volodymyr": "andrienko",
}

# Supertonic voice style IDs (5 female + 5 male). Default: F5 (soft, soothing).
SUPERTONIC_VOICES = ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5")
SUPERTONIC_DEFAULT_VOICE = "F5"


def _resolve_voice(name: str) -> str:
    return VOICE_ALIASES.get(name.lower(), name)


def build_tts(*, sample_rate: int, voice: str | None = None) -> TTSService:
    """Build a TTS service. `voice` overrides the env-supplied default."""
    backend = os.environ.get("TTS_BACKEND", "supertonic").lower()

    if backend in {"supertonic", "supertonic-3", "s3"}:
        from supertonic_tts_service import SupertonicTTSService

        raw = (voice or os.environ.get("TTS_VOICE", SUPERTONIC_DEFAULT_VOICE)).strip()
        if raw not in SUPERTONIC_VOICES:
            raw = SUPERTONIC_DEFAULT_VOICE
        return SupertonicTTSService(voice=raw, sample_rate=sample_rate)

    if backend in {"intelliscrape", "api", "remote"}:
        from intelliscrape_tts_service import (
            DEFAULT_VOICE,
            IntelliscrapeTTSService,
        )

        raw = voice or os.environ.get("TTS_VOICE", DEFAULT_VOICE)
        return IntelliscrapeTTSService(voice=_resolve_voice(raw), sample_rate=sample_rate)

    if backend == "piper":
        from piper_tts_service import PiperTTSService

        return PiperTTSService(
            voice=voice or os.environ.get("TTS_VOICE", "ru_RU-irina-medium"),
            sample_rate=sample_rate,
        )

    raise ValueError(
        f"unknown TTS_BACKEND={backend!r}; supported: supertonic, intelliscrape, piper"
    )
