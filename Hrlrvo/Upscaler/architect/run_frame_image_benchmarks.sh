#!/usr/bin/env bash
set -Eeuo pipefail

ARCHITECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python"

cd "$ARCHITECT_DIR"

# These two frame-independent baselines intentionally use FlashVSR's exact
# source ranges (micro=21, mini=29, start=0). They run serially on one GPU.
exec "$PYTHON" benchmark_minimal.py \
  --models Real-ESRGAN HAT \
  --modes micro mini \
  --start-frame 0 \
  --status-json "$ARCHITECT_DIR/../result/benchmark/run_status_frame_image.json" \
  "$@"
