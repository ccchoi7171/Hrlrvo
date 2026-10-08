#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_global_prior_20260923
bash "$ROOT/commands/recover_c0.sh" 2>&1 | tee "$ROOT/logs/recover_c0.log"
status=${PIPESTATUS[0]}
echo "C0_RECOVERY_EXIT=$status" | tee -a "$ROOT/logs/recover_c0.log"
exit "$status"
