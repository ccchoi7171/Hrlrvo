#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/source/VideoOutpainting_test/main/src:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-core/src:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-trainer/src:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-pipelines/src"
export HF_HOME="$ROOT/tmp/hf_home"
export XDG_CACHE_HOME="$ROOT/tmp/xdg_cache"
export TMPDIR="$ROOT/tmp"
export TORCH_HOME="$ROOT/tmp/torch_home"
export TRITON_CACHE_DIR="$ROOT/tmp/triton_cache"
export CUDA_CACHE_PATH="$ROOT/tmp/cuda_cache"
cd "$ROOT"
exec /home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python -u "$ROOT/source/VideoOutpainting_test/main/scripts/run_fhd_chain.py" --mode control --end-chunk 4 --no-rebase-temporal-position --no-use-latent-carry --generated-prefix-strength 1.0 "$@"
