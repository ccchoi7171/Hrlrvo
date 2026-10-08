#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=33333333
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
exec "${CERT_OUTPAINT_PYTHON:-/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python}" \
  "$ROOT/tools/run_crop01_seed33333333.py" "$@"
