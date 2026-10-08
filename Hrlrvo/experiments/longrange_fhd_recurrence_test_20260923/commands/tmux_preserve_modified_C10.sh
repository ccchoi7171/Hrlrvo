#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
"$ROOT/commands/preserve_modified_C10.sh" 2>&1 | tee "$ROOT/logs/preserve_modified_C10.log"
rc=${PIPESTATUS[0]}
echo "PRESERVE_MODIFIED_EXIT=$rc" | tee -a "$ROOT/logs/preserve_modified_C10.log"
exit "$rc"
