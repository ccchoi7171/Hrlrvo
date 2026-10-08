#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_global_prior_20260923
export PYTHONDONTWRITEBYTECODE=1
SOURCE_ROOT="${LINEART_SOURCE_ROOT:-$ROOT/source/VideoOutpainting_test/main/src}"
export PYTHONPATH="$SOURCE_ROOT:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-core/src:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-trainer/src:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-pipelines/src"
export HF_HOME="$ROOT/tmp/hf_home" XDG_CACHE_HOME="$ROOT/tmp/xdg_cache" TMPDIR="$ROOT/tmp"
export TORCH_HOME="$ROOT/tmp/torch_home" TRITON_CACHE_DIR="$ROOT/tmp/triton_cache" CUDA_CACHE_PATH="$ROOT/tmp/cuda_cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python
cd "$ROOT"
