#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/source/VideoOutpainting_test/main/src:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-core/src:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-trainer/src:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-pipelines/src"
cd "$ROOT"
exec /home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python tests/preflight.py
