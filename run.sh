#!/usr/bin/env bash
# Apple Silicon launcher.
# Brings up the local Whisper FastAPI shim, ensures gemma3:4b is pulled in
# Ollama, then runs bot.py. When this script exits (cleanly or via Ctrl+C),
# every background process it started is torn down.
set -euo pipefail
cd "$(dirname "$0")"

CHILD_PIDS=()

cleanup() {
    # Stop tracked children and any process tree rooted at them.
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            # Grandchildren (e.g. uvicorn under `uv run`) first.
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
    # Belt-and-suspenders: anything still listening on :8000.
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:8000 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[run] syncing project deps"
uv sync --quiet

echo "[run] starting whisper server on :8000"
uv run uvicorn servers.whisper_server:app \
    --host 127.0.0.1 --port 8000 --log-level warning &
CHILD_PIDS+=($!)

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
