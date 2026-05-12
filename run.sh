#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

trap 'kill 0' EXIT INT TERM

echo "[run] starting whisper server on :8000"
uv run uvicorn servers.whisper_server:app \
    --host 127.0.0.1 --port 8000 --log-level warning &

echo "[run] waiting for whisper to become healthy..."
for i in $(seq 1 120); do
    if curl -fs "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
        echo "[run]   :8000 ready"
        break
    fi
    sleep 1
    if [ "$i" -eq 120 ]; then
        echo "[run] :8000 did not become healthy in 120s" >&2
        exit 1
    fi
done

echo "[run] checking gemma3:4b is pulled..."
if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx 'gemma3:4b'; then
    echo "[run] pulling gemma3:4b..."
    ollama pull gemma3:4b
fi

echo "[run] launching bot.py (Silero TTS loads in-process)"
uv run python bot.py
