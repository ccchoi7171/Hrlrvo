#!/usr/bin/env bash
set -euo pipefail
EXP=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fix_test_20260923
export PYTHONDONTWRITEBYTECODE=1
exec /home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python "$EXP/scripts/compare_window.py" --window B
