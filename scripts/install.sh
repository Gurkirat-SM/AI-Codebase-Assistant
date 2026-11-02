#!/usr/bin/env bash
# Install Python + Node dependencies for the AI Codebase Assistant.
#
# Run: bash scripts/install.sh
set -euo pipefail

# Always operate from the repository root, no matter where the caller is.
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

echo "==> Installing Python dependencies with: $PY"
"$PY" -m pip install --upgrade -r requirements.txt

echo "==> Installing Node dependencies (frontend)"
cd frontend || exit 1
npm install

echo "==> Done. Run 'bash scripts/dev.sh' to start the stack."
