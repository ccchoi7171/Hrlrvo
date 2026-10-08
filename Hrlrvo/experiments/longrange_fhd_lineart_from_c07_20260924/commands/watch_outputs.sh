#!/usr/bin/env bash
set -euo pipefail
unset LINEART_SOURCE_ROOT
source /home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_from_c07_20260924/commands/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
case "${1:?previews or progress}" in
  previews) script=watch_chunk_previews.py ;;
  progress) script=watch_progress_videos.py ;;
  *) exit 2 ;;
esac
exec "$PY" -u "$ROOT/scripts/$script" --mode strength_1p0_global_lineart_from_c07 --end-chunk 23 --producer-pid "$(cat "$ROOT/report/producer.pid")"
