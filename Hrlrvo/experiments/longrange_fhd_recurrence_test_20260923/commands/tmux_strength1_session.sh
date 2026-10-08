#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
"$ROOT/commands/run_strength_1p0_C23.sh" 2>&1 | tee "$ROOT/logs/run_strength_1p0_C23.log"
rc=${PIPESTATUS[0]}
echo "STRENGTH_1P0_EXIT=$rc" | tee -a "$ROOT/logs/run_strength_1p0_C23.log"
exit "$rc"
