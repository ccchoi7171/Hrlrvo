#!/usr/bin/env bash
set -euo pipefail
unset LINEART_SOURCE_ROOT
source /home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_global_prior_20260923/commands/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$PY" -u "$ROOT/scripts/watch_progress_videos.py" --mode strength_1p0_global_lineart_prior --end-chunk 23 --producer-pid "$(cat "$ROOT/report/producer.pid")"
