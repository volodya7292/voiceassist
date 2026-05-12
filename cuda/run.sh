#!/usr/bin/env bash
# CUDA launcher. Brings up Whisper + Ollama in Docker, runs bot.py on the
# host, and stops everything (compose stack included) when the script exits.
#
# Set KEEP_COMPOSE=1 to leave the containers running after the script exits.
#
# Host requirements:
#   Linux:   NVIDIA driver + nvidia-container-toolkit + docker compose v2 + portaudio19-dev
#   Windows: WSL2 backend OR Docker Desktop + recent Windows PyTorch CUDA wheels
#   Both:    uv, Python 3.11/3.12
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE_FILE=cuda/docker-compose.yml

cleanup() {
    if [ "${KEEP_COMPOSE:-0}" = "1" ]; then
        echo "[cuda] KEEP_COMPOSE=1 set, leaving containers running"
        return
    fi
    echo "[cuda] stopping docker compose stack..."
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[cuda] syncing project deps (torch, pyaudio, pipecat, ...)"
uv sync

# Fast-fail if a critical import is missing — better here than 30s later
# after docker compose spins everything up.
if ! uv run python -c "import torch, pyaudio, pipecat" 2>/dev/null; then
    echo "[cuda] critical imports failed — running diagnostic:" >&2
    uv run python -c "
import importlib, traceback
for mod in ('torch', 'pyaudio', 'pipecat'):
    try:
        importlib.import_module(mod)
        print(f'  OK   {mod}')
    except Exception as e:
        print(f'  FAIL {mod}: {type(e).__name__}: {e}')
" >&2
    exit 1
fi

echo "[cuda] starting whisper + ollama via docker compose..."
docker compose -f "$COMPOSE_FILE" up -d

echo "[cuda] waiting for whisper :8000/health ..."
for i in $(seq 1 180); do
    if curl -fs http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "[cuda]   whisper ready"
        break
    fi
    sleep 1
    if [ "$i" -eq 180 ]; then
        echo "[cuda] whisper did not become healthy in 180s" >&2
        docker compose -f "$COMPOSE_FILE" logs --tail=50 whisper >&2 || true
        exit 1
    fi
done

echo "[cuda] waiting for ollama :11434 ..."
for i in $(seq 1 60); do
    if curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "[cuda]   ollama ready"
        break
    fi
    sleep 1
    if [ "$i" -eq 60 ]; then
        echo "[cuda] ollama did not become healthy in 60s" >&2
        exit 1
    fi
done

echo "[cuda] ensuring gemma3:4b is pulled..."
if ! curl -fs http://127.0.0.1:11434/api/tags | grep -q '"gemma3:4b"'; then
    echo "[cuda]   pulling gemma3:4b (this may take a few minutes)..."
    docker compose -f "$COMPOSE_FILE" exec -T ollama ollama pull gemma3:4b
fi

# Point bot.py at the local services.
export WHISPER_URL="${WHISPER_URL:-http://127.0.0.1:8000/v1}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434/v1}"
# faster-whisper-server selects the model server-side via WHISPER__MODEL; the
# STT_MODEL field bot.py sends is just OpenAI shape compatibility / logging.
export STT_MODEL="${STT_MODEL:-Systran/faster-whisper-large-v3-turbo}"

echo "[cuda] launching bot.py (Silero TTS loads in-process on CUDA)"
uv run python bot.py
