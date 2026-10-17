#!/usr/bin/env bash
# Prepare by default. Pass --run to train five models sequentially.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
"${PYTHON:-.venv/bin/python}" models/terramind/cross_validate.py "$@"
