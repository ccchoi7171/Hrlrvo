#!/usr/bin/env bash
set -euo pipefail
export LINEART_SOURCE_ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_global_prior_20260923/patches/original_src
source /home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_global_prior_20260923/commands/env.sh
exec "$PY" -u "$ROOT/scripts/recover_c0.py" --mode c0_recovery --end-chunk 0 --rebase-temporal-position --use-latent-carry --generated-prefix-strength 1.0
