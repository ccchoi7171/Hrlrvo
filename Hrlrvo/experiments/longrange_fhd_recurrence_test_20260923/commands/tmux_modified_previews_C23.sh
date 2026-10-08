#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$ROOT/tmp"
cd "$ROOT"
nice -n 10 /home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python -u "$ROOT/scripts/watch_chunk_previews.py" --mode modified --end-chunk 23 --producer-pid 361284 2>&1 | tee "$ROOT/logs/chunk_previews_modified_C23.log"
exit "${PIPESTATUS[0]}"
