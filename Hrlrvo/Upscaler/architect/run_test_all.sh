#!/usr/bin/env bash
set -Eeuo pipefail

ARCHITECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON_PYTHON="/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python"
PARALLEL_MODE="${PARALLEL_MODE:-auto}"
SCRATCH_ROOT="$(mktemp -d "/tmp/hrlrvo_upscaler_${$}_XXXXXX")"
RESULT_ROOT="$ARCHITECT_DIR/../result/test"
COMPLETION_JSON="$RESULT_ROOT/run_completion.json"

cleanup() {
  rm -rf -- "$SCRATCH_ROOT"
}
# orchestrate.py owns and reaps every subprocess group before returning. Only
# remove staged scratch on shell exit, after cancellation cleanup finishes.
trap cleanup EXIT

orchestrator_pid=""
forward_signal() {
  local signal_name="$1"
  if [[ -n "$orchestrator_pid" ]] && kill -0 "$orchestrator_pid" 2>/dev/null; then
    kill -s "$signal_name" "$orchestrator_pid" 2>/dev/null || true
  fi
}
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM
trap 'forward_signal HUP' HUP

if [[ ! -x "$COMMON_PYTHON" ]]; then
  printf 'Missing benchmark interpreter: %s\n' "$COMMON_PYTHON" >&2
  exit 2
fi

cd "$ARCHITECT_DIR"
mkdir -p "$RESULT_ROOT"
printf '{"status":"RUNNING","started_at":"%s","parallel_mode":"%s"}\n' \
  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$PARALLEL_MODE" > "$COMPLETION_JSON"

"$COMMON_PYTHON" orchestrate.py \
  --scratch-root "$SCRATCH_ROOT" \
  --parallel-mode "$PARALLEL_MODE" &
orchestrator_pid=$!

# A signal delivered specifically to this shell interrupts wait. Keep waiting
# after forwarding it so orchestrate.py can terminate/reap every model process
# before the EXIT trap removes staged files.
while true; do
  set +e
  wait "$orchestrator_pid"
  benchmark_status=$?
  set -e
  if ! kill -0 "$orchestrator_pid" 2>/dev/null; then
    break
  fi
done
orchestrator_pid=""

if [[ "$benchmark_status" -eq 0 ]]; then
  completion_status="PASS"
else
  completion_status="FAIL"
fi
printf '{"status":"%s","exit_code":%d,"completed_at":"%s","comparison_markdown":"%s"}\n' \
  "$completion_status" "$benchmark_status" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  "$RESULT_ROOT/comparison.md" > "$COMPLETION_JSON"

printf '\n========== Hrlrvo Upscaler benchmark %s (exit %d) ==========\n' \
  "$completion_status" "$benchmark_status"
if [[ -s "$RESULT_ROOT/comparison.md" ]]; then
  cat "$RESULT_ROOT/comparison.md"
else
  printf 'Comparison report was not produced. Inspect: %s\n' "$RESULT_ROOT/run_all.log"
fi
printf 'Completion state: %s\n' "$COMPLETION_JSON"
printf '============================================================\n'

exit "$benchmark_status"
