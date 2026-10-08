#!/usr/bin/env bash
set -euo pipefail

main_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python}"

cd "$main_dir"
exec "$python_bin" "$main_dir/tools/seed_search_4k.py" \
  --continuous \
  --steps 30 \
  --preview-step 10 \
  --preview-frames 0 32 64 \
  --max-concurrent 2 \
  --seeds-per-video-per-round 2 \
  --start-seed 33333333 \
  --videos 01,02,03,04,05 \
  --resume \
  --negative-prompt "people, person, human, staff, audience, crowd, backstage, seats" \
  --checkpoint "/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/hrlrvo_main_train_v1/checkpoints/checkpoint-007500" \
  "$@"
