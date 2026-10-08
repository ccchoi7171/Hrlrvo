#!/usr/bin/env bash
# Record of CPU-only candidate commands. Existing depth candidate names intentionally refuse overwrite.
set -euo pipefail
source /home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_global_prior_20260923/commands/env.sh
export CUDA_VISIBLE_DEVICES=''
"$PY" "$ROOT/scripts/make_lineart_candidates.py"
"$PY" "$ROOT/scripts/make_depth_candidates.py" --model depth-anything/Depth-Anything-V2-Small-hf --name D1_v2_small_518 --size 518
"$PY" "$ROOT/scripts/make_depth_candidates.py" --model LiheYoung/depth-anything-small-hf --name D2_v1_small_518 --size 518
"$PY" "$ROOT/scripts/make_depth_candidates.py" --model depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf --name D3_v2_metric_indoor_518 --kind metric_indoor --size 518
"$PY" "$ROOT/scripts/make_depth_candidates.py" --model depth-anything/Depth-Anything-V2-Small-hf --name D4_v2_small_1036 --size 1036
"$PY" "$ROOT/scripts/make_depth_candidates.py" --model depth-anything/Depth-Anything-V2-Base-hf --name D5_v2_base_518 --size 518
"$PY" "$ROOT/scripts/make_prior_gallery.py"
