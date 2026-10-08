#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
"$ROOT/commands/finalize_strength1_C23.sh" 2>&1 | tee "$ROOT/logs/finalize_strength1_C23.log"
rc=${PIPESTATUS[0]}
echo "FINALIZE_STRENGTH1_EXIT=$rc" | tee -a "$ROOT/logs/finalize_strength1_C23.log"
exit "$rc"
