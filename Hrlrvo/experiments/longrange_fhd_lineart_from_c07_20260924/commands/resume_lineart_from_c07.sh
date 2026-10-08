#!/usr/bin/env bash
set -euo pipefail
unset LINEART_SOURCE_ROOT
source /home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_from_c07_20260924/commands/env.sh
printf '%s\n' "$$" > "$ROOT/report/producer.pid"
exec "$PY" -u "$ROOT/source/VideoOutpainting_test/main/scripts/run_lineart_chain.py" \
  --mode strength_1p0_global_lineart_from_c07 --end-chunk 23 \
  --rebase-temporal-position --use-latent-carry --generated-prefix-strength 1.0 \
  --lineart-global-prior --lineart-bias-scale 0.5 --resume
