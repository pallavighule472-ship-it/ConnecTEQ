#!/bin/sh
# Auto-restart worker if it crashes
while true; do
    echo "[start.sh] Starting worker..."
    python worker.py
    echo "[start.sh] Worker exited — restarting in 3s..."
    sleep 3
done &

uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
