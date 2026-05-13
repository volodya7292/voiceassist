#!/usr/bin/env bash
# CUDA launcher.
# Brings up Ollama in Docker + the native faster-whisper server, ensures
# gemma4:e2b is pulled, then runs bot.py on the host. On exit (clean or
# Ctrl+C) tears everything down. Set KEEP_COMPOSE=1 to leave Ollama up.
#
# Host requirements:
#   Linux:   NVIDIA driver + nvidia-container-toolkit + docker compose v2 + portaudio19-dev
#   Windows: Docker Desktop (WSL2 backend) + a recent NVIDIA driver
#   Both:    uv, Python 3.11/3.12
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE_FILE=cuda/docker-compose.yml
CHILD_PIDS=()

cleanup() {
    # Stop the native whisper server (and its uvicorn child).
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 0.5
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    if [ "${KEEP_COMPOSE:-0}" = "1" ]; then
        echo "[cuda] KEEP_COMPOSE=1 set, leaving ollama running"
        return
    fi
    echo "[cuda] stopping ollama compose stack..."
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[cuda] syncing project deps (torch, faster-whisper, pyaudio, ...)"
uv sync

if ! uv run python -c "import torch, faster_whisper, pyaudio, pipecat" 2>/dev/null; then
    echo "[cuda] critical imports failed — running diagnostic:" >&2
    uv run python -c "
import importlib
for mod in ('torch', 'faster_whisper', 'pyaudio', 'pipecat'):
    try:
        importlib.import_module(mod)
        print(f'  OK   {mod}')
    except Exception as e:
        print(f'  FAIL {mod}: {type(e).__name__}: {e}')
" >&2
    exit 1
fi

echo "[cuda] starting ollama via docker compose..."
docker compose -f "$COMPOSE_FILE" up -d

echo "[cuda] starting native whisper server on :8000 (faster-whisper, model=${WHISPER_MODEL:-small}, workers=${WHISPER_WORKERS:-2})"
uv run uvicorn servers.whisper_server:app \
    --host 127.0.0.1 --port 8000 --log-level warning &
CHILD_PIDS+=($!)

echo "[cuda] waiting for whisper :8000/health ..."
for i in $(seq 1 300); do
    if curl -fs http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "[cuda]   whisper ready"
        break
    fi
    sleep 1
    if [ "$i" -eq 300 ]; then
        echo "[cuda] whisper did not become healthy in 300s" >&2
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

echo "[cuda] ensuring gemma4:e2b is pulled..."
if ! curl -fs http://127.0.0.1:11434/api/tags | grep -q '"gemma4:e2b"'; then
    echo "[cuda]   pulling gemma4:e2b (this may take a few minutes)..."
    docker compose -f "$COMPOSE_FILE" exec -T ollama ollama pull gemma4:e2b
fi

export WHISPER_URL="${WHISPER_URL:-http://127.0.0.1:8000/v1}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434/v1}"

echo "[cuda] launching bot.py (Piper TTS loads in-process)"
uv run python bot.py
