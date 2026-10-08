#!/usr/bin/env bash
set -Eeuo pipefail

ARCHITECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python"

cd "$ARCHITECT_DIR"
exec "$PYTHON" benchmark_minimal.py "$@"
