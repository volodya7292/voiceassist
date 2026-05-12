# voiceassist

Local voice assistant for Apple Silicon Macs. Open-weight models, no cloud.

- **STT**: `whisper-large-v3-turbo` via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (multilingual; Russian works out of the box)
- **LLM**: `gemma3:4b` via [Ollama](https://ollama.com) (Metal-accelerated)
- **TTS**: Silero v5_4_ru (`kseniya`) — in-process PyTorch, CPU, RTF ≈ 0.01

Pipeline framework: [Pipecat](https://github.com/pipecat-ai/pipecat).
Audio I/O: your Mac's default mic and speakers (via PyAudio / PortAudio).

## Setup

```bash
brew install portaudio                  # required for pyaudio
uv sync                                 # creates .venv and installs all deps
ollama pull gemma3:4b                   # ~3 GB
```

The Whisper turbo (~1.6 GB) and Silero v5_4_ru (~140 MB) weights download
automatically on first use into `~/.cache/huggingface` and `~/.cache/silero`.

> **Note on Silero.** The Silero model is hosted on `models.silero.ai`. From
> some regions you may need a VPN to reach the CDN for the initial download;
> after the first run the weights are cached locally.

## Run

```bash
./run.sh
```

This boots the Whisper FastAPI shim on `:8000`, waits for it to be healthy,
ensures `gemma3:4b` is pulled, then launches `bot.py` (which loads Silero
in-process). Press Ctrl+C to quit; the script cleans up child processes.

## Layout

```
bot.py                       Pipecat pipeline (mic -> STT -> LLM -> TTS -> speakers)
silero_tts_service.py        Pipecat TTS service backed by Silero v5_4_ru
servers/whisper_server.py    OpenAI-compatible /v1/audio/transcriptions
run.sh                       Launcher
```

## Component smoke tests

```bash
# Whisper STT
curl -F "file=@some.wav" -F "language=ru" http://127.0.0.1:8000/v1/audio/transcriptions

# Ollama LLM
curl http://127.0.0.1:11434/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"gemma3:4b","messages":[{"role":"user","content":"Привет"}]}'
```

## Configuration

Override via environment variables (see `bot.py` for the full list):

| var                   | default                                  |
|-----------------------|------------------------------------------|
| `LLM_MODEL`           | `gemma3:4b`                              |
| `STT_MODEL`           | `whisper-large-v3-turbo`                 |
| `WHISPER_MODEL_REPO`  | `mlx-community/whisper-large-v3-turbo`   |
| `TTS_VOICE`           | `kseniya` (Silero); also `aidar`, `baya`, `xenia` |
| `TTS_SAMPLE_RATE`     | `24000` (Silero also supports 8000 / 48000) |
| `WHISPER_URL`         | `http://127.0.0.1:8000/v1`               |
| `OLLAMA_URL`          | `http://127.0.0.1:11434/v1`              |
| `GREET`               | `1` (set to `0` to skip the startup greeting) |
