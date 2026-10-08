#!/usr/bin/env bash
set -euo pipefail

pipeline_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-$pipeline_dir/../../miniconda3/envs/Hrlrvo/bin/python}"
input="${1:-$pipeline_dir/data/00_Input_8K_7680x4320/05_Input_8K.mp4}"
exec "$python_bin" "$pipeline_dir/src/00to01/preproc_8k_to_2k_24.py" \
  "$input" --output-dir "$pipeline_dir/data/01_PreProc_2K_1920x1080" --overwrite
