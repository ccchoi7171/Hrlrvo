#!/usr/bin/env bash
set -euo pipefail
source /home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_global_prior_20260923/commands/env.sh
exec "$PY" -u "$ROOT/source/VideoOutpainting_test/main/scripts/run_lineart_chain.py" --mode strength_1p0_global_lineart_prior --end-chunk 4 --rebase-temporal-position --use-latent-carry --generated-prefix-strength 1.0 --lineart-global-prior --lineart-bias-scale 0.5 --bootstrap-c0
