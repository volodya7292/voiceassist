#!/usr/bin/env bash
# Boot the CUDA model services (Whisper + Ollama in Docker), wait until both
# are healthy, ensure gemma3:4b is pulled, then launch bot.py on the host.
#
# Host requirements:
#   - NVIDIA driver + nvidia-container-toolkit
#   - docker compose v2
#   - Python 3.11/3.12, uv, PortAudio (apt: portaudio19-dev)
#   - PyTorch with CUDA wheels (auto-installed by `uv sync`)
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE_FILE=cuda/docker-compose.yml

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

# Make sure the bot points at the local services (run.sh on Mac uses the same defaults).
export WHISPER_URL="${WHISPER_URL:-http://127.0.0.1:8000/v1}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434/v1}"
# faster-whisper-server expects the HF model repo string in /v1/audio/transcriptions,
# but the model is selected server-side via WHISPER__MODEL — STT_MODEL here is only
# used for logging/OpenAI shape compatibility.
export STT_MODEL="${STT_MODEL:-Systran/faster-whisper-large-v3-turbo}"

echo "[cuda] launching bot.py (Silero TTS loads in-process on CUDA)"
exec uv run python bot.py
