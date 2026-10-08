#!/usr/bin/env bash
set -euo pipefail
cd /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting
export PYTHONPATH="$PWD/main/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
base=outputs/cudnn_benchmark_validation/20260912_0637
while kill -0 3826431 2>/dev/null; do sleep 10; done
python -c "import json; assert json.load(open('$base/production_benchmark/result.json'))['status']=='PASS'"
# Fresh pre-change-main control, same exact sample/seed/initial weights.
python -u tools/cudnn_benchmark_validation/validate_training.py --output-dir "$base/fresh_false_control" --steps 1 --baseline-false > "$base/fresh_false_control.log" 2>&1
# Independent actual warmed full-model Kineto capture; no permanent hook.
python -u tools/cudnn_benchmark_validation/profile_warmed.py --output-dir "$base/warmed_profile" --steps 2 > "$base/warmed_profile.log" 2>&1
# New representative exact-shape kernel/reduction checks in separate processes.
python -u tools/cudnn_benchmark_validation/probe_kernels.py --output-dir "$base/kernels_false" --benchmark 0 > "$base/kernels_false.log" 2>&1
python -u tools/cudnn_benchmark_validation/probe_kernels.py --output-dir "$base/kernels_true" --benchmark 1 > "$base/kernels_true.log" 2>&1
printf 'CONTROL_PROFILE_KERNEL_PROBES_COMPLETE\n'
