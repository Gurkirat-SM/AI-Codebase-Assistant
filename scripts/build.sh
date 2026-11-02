#!/usr/bin/env bash
# Build the frontend for production.
#
# Run: bash scripts/build.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

cd frontend || exit 1
npm run build
