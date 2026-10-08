#!/usr/bin/env bash
set -euo pipefail

pipeline_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
main_dir="$pipeline_dir/../VideoOutpainting/main"
video_id="$(basename -- "$(dirname -- "${BASH_SOURCE[0]}")")"
python_bin="${PYTHON_BIN:-/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python}"
output_root="${SEEDSEARCH_OUTPUT_ROOT:-$pipeline_dir/data/02_seedSearch_4K_3840x2160}"
seed="${1:?Usage: $0 SEED [additional seed_search_4k.py options]}"
shift
exec "$python_bin" "$main_dir/tools/seed_search_4k.py" \
  --single-video "$video_id" --single-seed "$seed" --resume \
  --steps 30 --preview-step 10 --preview-frames 0 32 64 \
  --output-root "$output_root" "$@"
