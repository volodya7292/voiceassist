# voiceassist

Local voice assistant. Open-weight models, no cloud. Two supported hosts:

| host             | STT                                  | LLM (Ollama)        | TTS                                |
|------------------|--------------------------------------|---------------------|------------------------------------|
| Apple Silicon    | `whisper-large-v3-turbo` via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | `gemma3:4b` (Metal) | Silero v5_4_ru in-process, CPU     |
| Linux + NVIDIA   | `faster-whisper-large-v3-turbo` via [fedirz/faster-whisper-server](https://github.com/fedirz/faster-whisper-server) (Docker, CUDA) | `gemma3:4b` (Docker, CUDA) | Silero v5_4_ru in-process, CUDA |

Pipeline framework: [Pipecat](https://github.com/pipecat-ai/pipecat).
Audio I/O: host mic + speakers via PyAudio / PortAudio. The bot itself runs
on the host in both setups — only the model servers differ.

## Setup — Apple Silicon

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

```bash
./run.sh
```

This boots the Whisper FastAPI shim on `:8000`, waits for it to be healthy,
ensures `gemma3:4b` is pulled, then launches `bot.py` (which loads Silero
in-process). Press Ctrl+C to quit; the script cleans up child processes.

## Setup — Linux + NVIDIA (CUDA)

Prereqs on the host:

- NVIDIA driver + [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `docker compose` v2
- `apt install portaudio19-dev`  (for PyAudio)
- `uv`

```bash
uv sync               # PyTorch CUDA wheels resolved automatically on Linux
./cuda/run.sh
```

`cuda/run.sh` boots `cuda/docker-compose.yml` (Whisper STT + Ollama on the
GPU), waits for both to be healthy, pulls `gemma3:4b` if missing, then runs
`bot.py` on the host. Silero loads in-process and auto-detects the GPU; you
can override with `SILERO_DEVICE=cuda:1` or `SILERO_DEVICE=cpu`.

To tear down the containers afterwards: `docker compose -f cuda/docker-compose.yml down`.

## Layout

```
bot.py                       Pipecat pipeline (mic -> STT -> LLM -> TTS -> speakers)
silero_tts_service.py        Pipecat TTS service backed by Silero v5_4_ru (auto CPU/CUDA)
servers/whisper_server.py    Apple Silicon: OpenAI-compatible /v1/audio/transcriptions via mlx-whisper
loadtest.py                  Concurrency + latency benchmark
run.sh                       Apple Silicon launcher
cuda/docker-compose.yml      CUDA model services (Whisper + Ollama)
cuda/run.sh                  CUDA launcher
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
| `WHISPER_MODEL_REPO`  | `mlx-community/whisper-large-v3-turbo` (Mac) |
| `WHISPER_MODEL`       | `Systran/faster-whisper-large-v3-turbo` (CUDA, sets `WHISPER__MODEL` in the Docker image) |
| `TTS_VOICE`           | `kseniya` (Silero); also `aidar`, `baya`, `xenia` |
| `TTS_SAMPLE_RATE`     | `24000` (Silero also supports 8000 / 48000) |
| `SILERO_DEVICE`       | auto: `cuda` if available, else `cpu`    |
| `OLLAMA_KEEP_ALIVE`   | `5m` (Docker compose; longer = less cold-load latency at the cost of VRAM) |
| `WHISPER_URL`         | `http://127.0.0.1:8000/v1`               |
| `OLLAMA_URL`          | `http://127.0.0.1:11434/v1`              |
| `GREET`               | `1` (set to `0` to skip the startup greeting) |
