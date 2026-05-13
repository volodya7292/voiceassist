"""TTS backend selector.

Default: intelliscrape (remote API, works on any host). Set
`TTS_BACKEND=piper` for offline fallback via the local Piper ONNX model.
"""
from __future__ import annotations

import os

from pipecat.services.tts_service import TTSService

# Friendly persona names -> the API's voice name (surname-based clones).
# Lets the user write TTS_VOICE=olena instead of TTS_VOICE=danchenko.
VOICE_ALIASES = {
    "serhii": "serhii",
    "yurii": "tryus",
    "olena": "danchenko",
    "volodymyr": "andrienko",
}


def _resolve_voice(name: str) -> str:
    return VOICE_ALIASES.get(name.lower(), name)


def build_tts(*, sample_rate: int) -> TTSService:
    backend = os.environ.get("TTS_BACKEND", "intelliscrape").lower()

    if backend in {"intelliscrape", "api", "remote"}:
        from intelliscrape_tts_service import (
            DEFAULT_VOICE,
            IntelliscrapeTTSService,
        )

        voice = _resolve_voice(os.environ.get("TTS_VOICE", DEFAULT_VOICE))
        return IntelliscrapeTTSService(voice=voice, sample_rate=sample_rate)

    if backend == "piper":
        from piper_tts_service import PiperTTSService

        return PiperTTSService(
            voice=os.environ.get("TTS_VOICE", "ru_RU-irina-medium"),
            sample_rate=sample_rate,
        )

    raise ValueError(
        f"unknown TTS_BACKEND={backend!r}; supported: intelliscrape, piper"
    )
