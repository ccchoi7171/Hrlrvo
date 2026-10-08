#!/usr/bin/env bash
set -Eeuo pipefail

ARCHITECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python"
STATUS="$ARCHITECT_DIR/../result/benchmark/run_status_dove.json"
benchmark_pid=""

forward_signal() {
  local signal_name="$1"
  if [[ -n "$benchmark_pid" ]] && kill -0 "$benchmark_pid" 2>/dev/null; then
    kill -s "$signal_name" "$benchmark_pid" 2>/dev/null || true
  fi
}
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM
trap 'forward_signal HUP' HUP

cd "$ARCHITECT_DIR"

# DOVE alone is run here. Existing valid FlashVSR/SeedVR2/Real-ESRGAN/HAT
# reports are reused when the common timing, quantitative, and qualitative
# comparison artifacts are regenerated.
setsid "$PYTHON" benchmark_minimal.py \
  --models DOVE \
  --modes micro mini \
  --start-frame 0 \
  --status-json "$STATUS" \
  "$@" &
benchmark_pid=$!

while true; do
  set +e
  wait "$benchmark_pid"
  code=$?
  set -e
  if ! kill -0 "$benchmark_pid" 2>/dev/null; then
    break
  fi
done
benchmark_pid=""

set +e
"$PYTHON" report_dove_completion.py --launcher-exit-code "$code"
report_code=$?
set -e
if [[ "$code" -eq 0 && "$report_code" -ne 0 ]]; then
  code=5
fi
exit "$code"
