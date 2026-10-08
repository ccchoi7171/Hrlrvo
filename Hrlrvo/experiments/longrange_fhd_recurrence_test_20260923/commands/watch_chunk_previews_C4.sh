#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$ROOT/tmp"
cd "$ROOT"
exec nice -n 10 /home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python -u "$ROOT/scripts/watch_chunk_previews.py" --mode modified --end-chunk 4 --producer-pid 3759387
