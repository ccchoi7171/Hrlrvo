#!/usr/bin/env bash
set -euo pipefail
EXP=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fix_test_20260923
export PYTHONPATH="$EXP/source:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-core/src:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/packages/ltx-trainer/src:/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/main/src"
export PYTHONDONTWRITEBYTECODE=1
cd "$EXP"
exec /home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python -u scripts/run_window.py --window B --rebase-temporal-position --use-latent-carry --generated-prefix-strength 0.8 "$@"
