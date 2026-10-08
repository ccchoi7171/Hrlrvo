#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
PRODUCER_PID=3759387
while kill -0 "$PRODUCER_PID" 2>/dev/null; do
  if grep -q '"event": "run_complete"' "$ROOT/logs/run_modified_C4.log"; then break; fi
  sleep 15
done
if ! grep -q '"event": "run_complete"' "$ROOT/logs/run_modified_C4.log" || [ ! -s "$ROOT/outputs/modified/C0_C4_257f_1920x1080.mp4" ]; then
  echo 'C0-C4 smoke did not finish cleanly; refusing to resume.' | tee -a "$ROOT/logs/run_modified_C23_resume.log"
  exit 1
fi
"$ROOT/commands/run_modified_C23.sh" --resume 2>&1 | tee "$ROOT/logs/run_modified_C23_resume.log"
rc=${PIPESTATUS[0]}
echo "MODIFIED_C23_EXIT=$rc" | tee -a "$ROOT/logs/run_modified_C23_resume.log"
exit "$rc"
