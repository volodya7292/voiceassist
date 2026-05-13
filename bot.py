"""Local voice assistant: mic -> Whisper -> Gemma -> TTS -> speakers.

Local services expected to be running:
  - Whisper:  http://127.0.0.1:8000  (servers/whisper_server.py)
  - Ollama:   http://127.0.0.1:11434 (ollama serve; model gemma3:4b)
  - TTS:      in-process. Piper on darwin, Qwen3-TTS on CUDA hosts (see
              tts_backends.py).
"""
from __future__ import annotations

import asyncio
import os

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from tts_backends import build_tts

WHISPER_URL = os.environ.get("WHISPER_URL", "http://127.0.0.1:8000/v1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1")

LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
STT_MODEL = os.environ.get("STT_MODEL", "small")
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
GREET = os.environ.get("GREET", "1") != "0"

SYSTEM_PROMPT = (
    "Ты дружелюбный голосовой помощник, который отвечает по-русски. "
    "Отвечай одним или двумя короткими предложениями. "
    "Выводи только обычный текст: без markdown, без списков, без эмодзи, "
    "без блоков кода и без специальных символов. Твой ответ будет произнесён вслух."
)


async def main() -> None:
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=TTS_SAMPLE_RATE,
        )
    )

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.6))
    )

    stt = OpenAISTTService(
        api_key="not-needed",
        base_url=WHISPER_URL,
        settings=OpenAISTTService.Settings(model=STT_MODEL, language=Language.RU),
    )

    llm = OpenAILLMService(
        api_key="ollama",
        base_url=OLLAMA_URL,
        settings=OpenAILLMService.Settings(model=LLM_MODEL),
    )

    tts = build_tts(sample_rate=TTS_SAMPLE_RATE)

    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
    )
    aggregators = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=TTS_SAMPLE_RATE,
        ),
    )

    if GREET:
        @task.event_handler("on_pipeline_started")
        async def _greet(task, frame):  # noqa: ARG001
            logger.info("pipeline started — sending greeting")
            await task.queue_frames([LLMRunFrame()])

    logger.info("audio in/out ready — start speaking (Ctrl+C to quit)")
    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
