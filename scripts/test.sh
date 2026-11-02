#!/usr/bin/env bash
# Run the backend test suite.
#
# Run: bash scripts/test.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

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

echo "==> Running tests with $PY"
"$PY" -m pytest tests/ -v
