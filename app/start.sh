#!/usr/bin/env sh
set -e

echo "[start] Running model warmup"
python -m app.warmup_models

echo "[start] Starting API"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
