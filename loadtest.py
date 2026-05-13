"""Latency loadtest for the voiceassist pipeline.

Measures end-to-end latency from "user finishes speaking" (request submit) to
"bot starts speaking" (first TTS audio sample emitted) under different
concurrency levels.

The test bypasses Pipecat's audio I/O and calls the three pipeline components
directly so we can fan out N simultaneous "users" from one process:

  STT (HTTP Whisper :8000) -> LLM (HTTP Ollama :11434, streaming)
                           -> first-sentence boundary
                           -> TTS (Piper, in-process, serialized)

Pre-conditions: the Whisper server is running (servers/whisper_server.py) and
Ollama has gemma3:4b loaded. Run from the repo root via ``uv run python
loadtest.py``.

Notes on serialization:
  - Whisper server uses an asyncio lock (one transcribe at a time).
  - Piper is wrapped in a single asyncio lock to mirror the bot's behavior.
  - Ollama serializes by default unless OLLAMA_NUM_PARALLEL > 1; we don't
    change that here, so concurrency > 1 mostly exposes queueing at the
    component bottlenecks, which is exactly the metric we want.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Config

WHISPER_URL = os.environ.get("WHISPER_URL", "http://127.0.0.1:8000/v1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
# Wire-level model name. The native whisper_server uses the env-side
# WHISPER_MODEL (faster-whisper) or WHISPER_MODEL_REPO (mlx-whisper) to choose
# what to load; the request `model` field is ignored on both stacks, so this
# is purely cosmetic and matches the OpenAI shape.
STT_MODEL = os.environ.get("STT_MODEL", "small")
TTS_VOICE = os.environ.get("TTS_VOICE", "ru_RU-irina-medium")
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "22050"))

INPUTS_DIR = Path(
    os.environ.get("LOADTEST_INPUTS", Path(tempfile.gettempdir()) / "loadtest_inputs")
)
# Voice used to synthesize the *inputs*. Defaults to the same voice as the
# bot — content matters, not timbre.
INPUT_VOICE = os.environ.get("LOADTEST_INPUT_VOICE", TTS_VOICE)

PROMPTS_RU = [
    # ~100-char Russian prompts (15-18 words, ~6-8s of speech).
    "Расскажи, пожалуйста, как правильно заваривать чай, чтобы он получился ароматным и не слишком крепким.",
    "Объясни простыми словами, что такое искусственный интеллект и зачем он нужен в современной жизни сегодня.",
    "Подскажи, какие три самых известных достопримечательности стоит посмотреть обычному туристу в Москве.",
    "Я хочу научиться готовить итальянскую пасту, расскажи мне общий рецепт классического соуса карбонара.",
    "Помоги мне выбрать книгу для чтения вечером, посоветуй что-нибудь короткое, лёгкое и атмосферное.",
    "Какие интересные привычки помогают людям лучше высыпаться и просыпаться по утрам бодрыми и энергичными?",
    "Расскажи кратко, как устроена Солнечная система и почему Плутон перестали считать полноценной планетой.",
    "Дай совет начинающему программисту, который только что прочитал свой первый учебник по языку Python.",
]

SYSTEM_PROMPT_RU = (
    "Ты голосовой помощник. Отвечай по-русски, одним коротким предложением, "
    "обычным текстом без специальных символов."
)

# ---------------------------------------------------------------------------
# Input preparation

def _piper_synth_int16(voice_obj, text: str) -> tuple[np.ndarray, int]:
    """Run Piper synth and return (int16 PCM array, sample_rate)."""
    parts: list[np.ndarray] = []
    sr = 0
    for chunk in voice_obj.synthesize(text):
        parts.append(chunk.audio_int16_array)
        sr = chunk.sample_rate
    if not parts:
        return np.zeros(0, dtype=np.int16), sr or TTS_SAMPLE_RATE
    return np.concatenate(parts), sr


def prepare_inputs(input_voice_obj) -> list[Path]:
    """Synthesize Russian test WAVs with Piper (cached)."""
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    wavs: list[Path] = []
    for i, text in enumerate(PROMPTS_RU):
        wav = INPUTS_DIR / f"prompt_{i:02d}.wav"
        if not wav.exists():
            print(f"  generating {wav.name}: {text}")
            arr, sr = _piper_synth_int16(input_voice_obj, text)
            sf.write(wav, arr, sr, subtype="PCM_16")
        wavs.append(wav)
    return wavs


# ---------------------------------------------------------------------------
# Piper loading (reuses helpers from piper_tts_service)

def load_piper(voice: str):
    from piper import PiperVoice

    from piper_tts_service import DEFAULT_CACHE, _ensure_model, _resolve_use_cuda

    onnx_path = _ensure_model(voice, DEFAULT_CACHE)
    use_cuda = _resolve_use_cuda(None)
    print(f"  loading piper voice {voice} from {onnx_path} (cuda={use_cuda})")
    return PiperVoice.load(onnx_path, use_cuda=use_cuda)


def synthesize_sync(voice_obj, text: str) -> bytes:
    arr, _ = _piper_synth_int16(voice_obj, text)
    return arr.tobytes()


# ---------------------------------------------------------------------------
# Per-run worker

def first_sentence_boundary(text: str) -> int | None:
    """Return cut-index after the first sentence-ending punctuation, else None."""
    for i, ch in enumerate(text):
        if ch in ".!?…":
            j = i + 1
            while j < len(text) and text[j] in '"”»\')':
                j += 1
            return j
    return None


async def one_run(
    client: httpx.AsyncClient,
    model,
    tts_lock: asyncio.Lock,
    wav_path: Path,
    idx: int,
) -> dict:
    t0 = time.perf_counter()

    # 1. STT --------------------------------------------------------------
    audio_bytes = wav_path.read_bytes()
    r = await client.post(
        f"{WHISPER_URL}/audio/transcriptions",
        files={"file": (wav_path.name, audio_bytes, "audio/wav")},
        data={"model": STT_MODEL, "language": "ru"},
    )
    r.raise_for_status()
    transcript = r.json()["text"].strip()
    t_stt = time.perf_counter()

    # 2. LLM (stream until we have a sentence) ----------------------------
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_RU},
            {"role": "user", "content": transcript},
        ],
        "stream": True,
    }
    accumulated = ""
    t_first_token: float | None = None
    t_first_sentence: float | None = None
    first_sentence: str | None = None

    async with client.stream(
        "POST", f"{OLLAMA_URL}/chat/completions", json=body
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (
                chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                or ""
            )
            if not delta:
                continue
            if t_first_token is None:
                t_first_token = time.perf_counter()
            accumulated += delta
            end = first_sentence_boundary(accumulated)
            if end is not None:
                first_sentence = accumulated[:end].strip()
                t_first_sentence = time.perf_counter()
                break  # we have enough to start speaking

    if first_sentence is None:
        # Model didn't emit sentence-ending punctuation — use whole reply.
        first_sentence = accumulated.strip() or "Извините, я не понял."
        t_first_sentence = time.perf_counter()
    if t_first_token is None:
        t_first_token = t_first_sentence

    # Piper rejects empty input; guard.
    if not first_sentence.strip():
        first_sentence = "Хорошо."

    # 3. TTS (serialized, mirrors bot's in-process single-locked Piper) ---
    async with tts_lock:
        t_tts_wait_end = time.perf_counter()
        try:
            await asyncio.to_thread(synthesize_sync, model, first_sentence)
        except Exception as e:
            raise RuntimeError(
                f"piper synth failed for text={first_sentence!r}: "
                f"{type(e).__name__}: {e}"
            ) from e
        t_tts_end = time.perf_counter()

    return {
        "idx": idx,
        "transcript": transcript,
        "reply": first_sentence,
        # The realistic "first-audio" instant in production is right when Piper
        # *starts* synthesizing — TTS audio could be streamed out as it's produced.
        # Piper does stream per chunk; we report both the start and end.
        "e2e_first_audio_start_s": t_tts_wait_end - t0,
        "e2e_first_audio_ready_s": t_tts_end - t0,
        "stt_s": t_stt - t0,
        "llm_ttft_s": t_first_token - t_stt,
        "llm_to_sentence_s": t_first_sentence - t_first_token,
        "tts_queue_s": t_tts_wait_end - t_first_sentence,
        "tts_synth_s": t_tts_end - t_tts_wait_end,
    }


# ---------------------------------------------------------------------------
# Concurrency runner + reporting

async def run_concurrency_level(
    model,
    tts_lock: asyncio.Lock,
    wavs: list[Path],
    n: int,
    samples: int,
) -> list[dict]:
    async with httpx.AsyncClient(timeout=600.0) as client:
        results: list[dict] = []
        idx = 0
        while idx < samples:
            batch = min(n, samples - idx)
            tasks = [
                one_run(client, model, tts_lock, wavs[(idx + i) % len(wavs)], idx + i)
                for i in range(batch)
            ]
            outs = await asyncio.gather(*tasks, return_exceptions=True)
            for o in outs:
                if isinstance(o, Exception):
                    msg = str(o) or repr(o)
                    print(f"  ! error: {type(o).__name__}: {msg}")
                else:
                    results.append(o)
            idx += batch
        return results


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * p))))
    return xs[k]


def fmt_stat(xs: list[float]) -> str:
    if not xs:
        return "no data"
    return (
        f"mean={statistics.mean(xs):5.2f}s  "
        f"med={statistics.median(xs):5.2f}s  "
        f"p95={percentile(xs, 0.95):5.2f}s  "
        f"max={max(xs):5.2f}s"
    )


def summarize(label: str, results: list[dict]) -> None:
    if not results:
        print(f"\n=== concurrency {label}: no successful runs ===")
        return
    print(f"\n=== concurrency {label}  ({len(results)} runs) ===")
    rows = [
        ("e2e first-audio start",  "e2e_first_audio_start_s"),
        ("e2e first-audio ready",  "e2e_first_audio_ready_s"),
        ("  stt",                  "stt_s"),
        ("  llm ttft (post-stt)",  "llm_ttft_s"),
        ("  llm to 1st sentence",  "llm_to_sentence_s"),
        ("  tts queue wait",       "tts_queue_s"),
        ("  tts synth",            "tts_synth_s"),
    ]
    width = max(len(label) for label, _ in rows)
    for name, key in rows:
        xs = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        print(f"  {name:<{width}}  {fmt_stat(xs)}")


def print_summary_table(all_results: dict[int, list[dict]]) -> None:
    """Compact side-by-side summary across all concurrency levels."""
    levels = sorted(all_results.keys())
    if not levels:
        return

    def col(n: int, key: str, agg) -> str:
        xs = [r[key] for r in all_results[n] if isinstance(r.get(key), (int, float))]
        if not xs:
            return "    -  "
        return f"{agg(xs):6.2f}s"

    print("\n=== summary: first-audio latency by concurrency ===")
    header = "metric                     " + "  ".join(f" c={n:<3d}    " for n in levels)
    print(header)
    print("-" * len(header))
    rows = [
        ("e2e first-audio (median)", "e2e_first_audio_start_s", statistics.median),
        ("e2e first-audio (p95)",    "e2e_first_audio_start_s", lambda xs: percentile(xs, 0.95)),
        ("  stt          (median)", "stt_s",                    statistics.median),
        ("  llm ttft     (median)", "llm_ttft_s",               statistics.median),
        ("  llm->sentence(median)", "llm_to_sentence_s",        statistics.median),
        ("  tts queue    (median)", "tts_queue_s",              statistics.median),
        ("  tts synth    (median)", "tts_synth_s",              statistics.median),
    ]
    for label, key, agg in rows:
        cells = "  ".join(col(n, key, agg) for n in levels)
        print(f"{label:<27}  {cells}")
    print()
    counts = "  ".join(f"   n={len(all_results[n]):<3d} " for n in levels)
    print(f"{'successful runs':<27}  {counts}")


async def amain(args: argparse.Namespace) -> None:
    print(f"=== loading piper voice ({TTS_VOICE}) ===")
    model = await asyncio.to_thread(load_piper, TTS_VOICE)

    if INPUT_VOICE == TTS_VOICE:
        input_voice_obj = model
    else:
        print(f"=== loading piper input voice ({INPUT_VOICE}) ===")
        input_voice_obj = await asyncio.to_thread(load_piper, INPUT_VOICE)

    print("=== preparing inputs ===")
    wavs = await asyncio.to_thread(prepare_inputs, input_voice_obj)
    print(f"  {len(wavs)} prompts ready in {INPUTS_DIR}")

    tts_lock = asyncio.Lock()

    print("=== warmup (concurrency=1, 1 sample) ===")
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            w = await one_run(client, model, tts_lock, wavs[0], -1)
            print(
                f"  warmup ok: e2e first-audio ready in {w['e2e_first_audio_ready_s']:.2f}s "
                f"(stt={w['stt_s']:.2f}s, llm_ttft={w['llm_ttft_s']:.2f}s, "
                f"tts_synth={w['tts_synth_s']:.2f}s)"
            )
        except Exception as e:
            print(f"  ! warmup failed: {type(e).__name__}: {e}")
            print(
                "    is the whisper server running on :8000 and ollama on :11434?"
            )
            return

    all_results: dict[int, list[dict]] = {}
    for n in args.concurrency:
        print(f"\n>>> running concurrency={n}, samples={args.samples_per_level}")
        results = await run_concurrency_level(
            model, tts_lock, wavs, n, args.samples_per_level
        )
        all_results[n] = results
        summarize(str(n), results)

    print_summary_table(all_results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--concurrency",
        default="1,2,4,8",
        help="Comma-separated concurrency levels (default: 1,2,4,8).",
    )
    ap.add_argument(
        "--samples-per-level",
        type=int,
        default=8,
        help="Total requests to run at each concurrency level (default: 8).",
    )
    args = ap.parse_args()
    args.concurrency = [int(x) for x in args.concurrency.split(",") if x.strip()]
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
