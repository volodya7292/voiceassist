# voiceassist

Local voice assistant. Open-weight models, no cloud. Two supported hosts:

| host             | STT                                  | LLM (Ollama)        | TTS                                |
|------------------|--------------------------------------|---------------------|------------------------------------|
| Apple Silicon    | `whisper-large-v3-turbo` via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (single worker, Metal) | `gemma3:4b` (Metal) | Silero v5_4_ru in-process, CPU     |
| Linux + NVIDIA   | `whisper-small` via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (native Python, **N-worker pool on CUDA**) | `gemma3:4b` (Docker, CUDA) | Silero v5_4_ru in-process, CUDA |

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

`cuda/run.sh` boots Ollama via `cuda/docker-compose.yml`, launches the native
`servers/whisper_server.py` (which loads `WHISPER_WORKERS` copies of
`faster-whisper` on the GPU), waits for both to be healthy, pulls `gemma3:4b`
if missing, then runs `bot.py` on the host. Silero loads in-process and
auto-detects the GPU; override with `SILERO_DEVICE=cuda:1` or `SILERO_DEVICE=cpu`.

To leave Ollama up between runs: `KEEP_COMPOSE=1 ./cuda/run.sh`. To tear it
down manually: `docker compose -f cuda/docker-compose.yml down`.

**STT concurrency.** `WHISPER_WORKERS=N` (default `2`) spawns N `WhisperModel`
instances pinned to the same GPU. Each `WhisperModel` for `small` consumes
~480 MB VRAM in fp16. On a 16 GB GPU sharing with `gemma3:4b` you can
comfortably run 4–6 workers.

## Layout

```
bot.py                       Pipecat pipeline (mic -> STT -> LLM -> TTS -> speakers)
silero_tts_service.py        Pipecat TTS service backed by Silero v5_4_ru (auto CPU/CUDA)
servers/whisper_server.py    OpenAI-compatible /v1/audio/transcriptions
                             (mlx-whisper on Mac, faster-whisper N-worker pool elsewhere)
loadtest.py                  Concurrency + latency benchmark
run.sh                       Apple Silicon launcher (CLI bot)
cuda/docker-compose.yml      Ollama on GPU (whisper now runs natively)
cuda/run.sh                  CUDA launcher (CLI bot)
webui/server.py              Push-to-talk web UI backend
webui/static/index.html      Single-page browser frontend
webui/run.sh                 Web UI launcher (serves http://127.0.0.1:8888)
```

## Web UI

Push-to-talk in a browser tab instead of the native mic. Same backend
services (Whisper STT, Ollama, Silero TTS in-process), so any host that can
run `bot.py` can run the web UI.

```bash
# Make sure Ollama is running (brew services start ollama on Mac, or
# docker compose -f cuda/docker-compose.yml up -d on Linux/Windows).
./webui/run.sh
# Open http://127.0.0.1:8888/ in a browser. Click the big button, speak,
# click again to send. The reply audio plays back when ready.
```

The browser captures mic audio via the Web Audio API, encodes a 16 kHz WAV
client-side, and POSTs `/api/turn`. The server runs STT → LLM → TTS and
returns the reply WAV with `X-Transcript`, `X-Reply`, and `X-History`
response headers (URL-encoded) so the UI can render the conversation.

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
| `STT_MODEL`           | `small` — wire-level only; the server picks the actual model from `WHISPER_MODEL` (faster-whisper) or `WHISPER_MODEL_REPO` (mlx-whisper) |
| `WHISPER_MODEL_REPO`  | `mlx-community/whisper-large-v3-turbo` — Mac MLX server's actual model |
| `WHISPER_MODEL`       | `small` — faster-whisper short name or full HF repo path |
| `WHISPER_WORKERS`     | `2` — faster-whisper model-pool size (concurrent transcriptions) |
| `WHISPER_DEVICE`      | `auto` — `cuda` / `cpu` / `auto` for faster-whisper |
| `WHISPER_COMPUTE`     | `float16` on cuda, `int8` on cpu |
| `TTS_VOICE`           | `kseniya` (Silero); also `aidar`, `baya`, `xenia` |
| `TTS_SAMPLE_RATE`     | `24000` (Silero also supports 8000 / 48000) |
| `SILERO_DEVICE`       | auto: `cuda` if available, else `cpu`    |
| `OLLAMA_KEEP_ALIVE`   | `5m` (Docker compose; longer = less cold-load latency at the cost of VRAM) |
| `WHISPER_URL`         | `http://127.0.0.1:8000/v1`               |
| `OLLAMA_URL`          | `http://127.0.0.1:11434/v1`              |
| `GREET`               | `1` (set to `0` to skip the startup greeting) |
