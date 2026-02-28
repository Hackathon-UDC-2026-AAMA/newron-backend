#!/usr/bin/env sh
set -e

echo "[start] Running model warmup"
python -m app.warmup_models

echo "[start] Exporting OpenAPI to /app/docs/openapi.json"
if ! python -m app.export_openapi; then
	echo "[start] Warning: OpenAPI export failed, continuing startup"
fi

echo "[start] Starting API"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
