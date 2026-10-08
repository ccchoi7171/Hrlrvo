#!/usr/bin/env bash
set -euo pipefail

source /home/work/ai-graphics_33/miniconda3/etc/profile.d/conda.sh
conda activate Hrlrvo

expected_python=/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python
if [[ "$(command -v python)" != "$expected_python" ]]; then
  echo "Wrong inference environment: $(command -v python)" >&2
  exit 1
fi

cd /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/main
exec "$expected_python" \
  /home/work/ai-graphics_33/Hrlrvo/pipeline/src/01to02_17f_seedsearch/orchestrate_seedsearch.py \
  --phase "${SEEDSEARCH_PHASE:-bootstrap}"
