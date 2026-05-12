#!/usr/bin/env bash
# Web UI launcher. Same model services as bot.py — needs the Whisper server
# on :8000 and Ollama on :11434. The web app loads Silero in-process and
# serves http://127.0.0.1:8888.
#
# Apple Silicon:   just run this; Whisper is started for you, Ollama must
#                  already be running (brew services start ollama).
# Linux/Windows:   start the CUDA stack first via `./cuda/run.sh` in another
#                  terminal (or `docker compose -f cuda/docker-compose.yml up -d`
#                  for just Ollama), then run this script.
set -euo pipefail
cd "$(dirname "$0")/.."

CHILD_PIDS=()
cleanup() {
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

echo "[webui] syncing project deps"
uv sync

# Start whisper if nothing's listening.
if ! curl -fs http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[webui] starting whisper server on :8000"
    uv run uvicorn servers.whisper_server:app \
        --host 127.0.0.1 --port 8000 --log-level warning &
    CHILD_PIDS+=($!)

    for i in $(seq 1 120); do
        if curl -fs http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
        sleep 1
        if [ "$i" -eq 120 ]; then
            echo "[webui] whisper did not become healthy in 120s" >&2
            exit 1
        fi
    done
fi

# Ensure Ollama is reachable.
if ! curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "[webui] ollama is not reachable at 127.0.0.1:11434" >&2
    echo "        start it natively (macOS: 'brew services start ollama')" >&2
    echo "        or via Docker: 'docker compose -f cuda/docker-compose.yml up -d'" >&2
    exit 1
fi

echo "[webui] open http://127.0.0.1:8888/"
exec uv run uvicorn webui.server:app --host 127.0.0.1 --port 8888 --log-level info
