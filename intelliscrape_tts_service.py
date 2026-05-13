"""Pipecat TTS service backed by the intelliscrape custom API.

Wire shape (per the team's existing client):
    POST https://tts.intelliscrape.com/tts
    Headers: authorization: Bearer <INTELLISCRAPE_TTS_TOKEN>
    Body: {"voice": <name>, "text": <utterance>}
    Response: opus bytes (audio/opus, OGG container).

Registered voices on the API: andrienko, danchenko, serhii, tryus.
Default voice: `danchenko` to match the Danchenko reference we've been using.

The opus payload is decoded to int16 PCM in-process via soundfile (libsndfile
reads OPUS-in-OGG natively) and yielded as TTSAudioRawFrames at the model's
native sample rate.
"""
from __future__ import annotations

import io
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import numpy as np
import soundfile as sf
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

DEFAULT_BASE_URL = os.environ.get(
    "INTELLISCRAPE_TTS_URL", "https://tts.intelliscrape.com"
)
DEFAULT_VOICE = "serhii"
DEFAULT_SAMPLE_RATE = 24000  # OPUS opus on this API decodes to 24 kHz mono


class IntelliscrapeTTSService(TTSService):
    """Remote TTS via tts.intelliscrape.com — works on any host (no GPU)."""

    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_ms: int = 100,
        request_timeout_s: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model="intelliscrape", voice=voice, language=None),
            **kwargs,
        )
        token = api_key or os.environ.get("INTELLISCRAPE_TTS_TOKEN", "")
        if not token:
            raise RuntimeError(
                "INTELLISCRAPE_TTS_TOKEN is not set. Put it in .env at the "
                "repo root (the run scripts load .env automatically)."
            )
        self._voice = voice
        self._token = token
        self._url = base_url.rstrip("/") + "/tts"
        self._chunk_samples = max(1, int(sample_rate * chunk_ms / 1000))
        self._request_timeout_s = request_timeout_s
        self._client: httpx.AsyncClient | None = None

    def can_generate_metrics(self) -> bool:
        return True

    def set_voice(self, voice: str) -> None:
        """Swap the active voice. Takes effect on the next synthesis call."""
        voice = (voice or "").strip()
        if not voice or voice == self._voice:
            return
        logger.info(f"intelliscrape: voice {self._voice} -> {voice}")
        self._voice = voice

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._request_timeout_s)
        return self._client

    def _decode_opus_to_int16(self, opus_bytes: bytes) -> bytes:
        """Decode an OPUS-in-OGG payload to int16 PCM bytes at self.sample_rate."""
        arr, sr = sf.read(io.BytesIO(opus_bytes), dtype="float32", always_2d=False)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)  # downmix to mono if stereo
        if sr != self.sample_rate:
            logger.warning(
                f"intelliscrape: decoded sr={sr} != configured {self.sample_rate}; "
                f"Pipecat output transport will resample"
            )
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype("<i2").tobytes()
        return pcm

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text = (text or "").strip()
        if not text:
            return
        logger.debug(f"intelliscrape: synthesizing [{text}] voice={self._voice}")

        client = await self._ensure_client()
        try:
            await self.start_tts_usage_metrics(text)
            resp = await client.post(
                self._url,
                headers={"authorization": f"Bearer {self._token}"},
                json={"voice": self._voice, "text": text},
            )
            if resp.status_code != 200:
                body = resp.text[:200] if resp.text else "(no body)"
                raise RuntimeError(
                    f"intelliscrape /tts returned {resp.status_code}: {body}"
                )
            pcm = self._decode_opus_to_int16(resp.content)
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception(f"intelliscrape: synthesis error: {e}")
            yield ErrorFrame(error=f"intelliscrape: {e}")
            return

        await self.stop_ttfb_metrics()

        bytes_per_chunk = self._chunk_samples * 2  # int16
        for start in range(0, len(pcm), bytes_per_chunk):
            chunk = pcm[start : start + bytes_per_chunk]
            if not chunk:
                continue
            yield TTSAudioRawFrame(
                audio=chunk,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )
