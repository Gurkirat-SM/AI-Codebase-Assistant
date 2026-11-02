#!/usr/bin/env bash
# Start the full stack: backend (uvicorn, :8000) + frontend (vite, :5173).
#
# Run: bash scripts/dev.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# Pick the Python interpreter (prefer the repo's virtualenv).
PY=""
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

echo "==> Starting backend (uvicorn, :8000) via $PY"
"$PY" -m uvicorn backend.app.main:app --reload --port 8000 &
BEND=$!

# Wait for the health endpoint to accept connections before starting the frontend.
for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

cd "${OLDPWD}/frontend" || exit 1
echo "==> Starting frontend (vite, :5173) — press Ctrl+C to stop"
npm run dev

# Clean up the backend process when the user stops the stack.
kill "$BEND" 2>/dev/null || true
wait "$BEND" 2>/dev/null || true
