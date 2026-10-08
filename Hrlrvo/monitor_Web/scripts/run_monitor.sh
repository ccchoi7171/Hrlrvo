#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/monitor_Web
PY=/home/work/ai-graphics_33/miniconda3/envs/monitor_web/bin/python
cd "$ROOT"
exec "$PY" -B app.py --runs-root /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs --host 0.0.0.0 --port 8050 "$@"
