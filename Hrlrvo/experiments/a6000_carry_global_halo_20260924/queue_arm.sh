#!/usr/bin/env bash
# Do not terminate or change any foreign GPU process. Retry our arm only on an idle GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:?arm name required}"
case "$NAME" in 01_baseline) GPU=0;; 02_no_carry) GPU=1;; 03_global_boost) GPU=2;; 04_native_halo) GPU=3;; *) exit 2;; esac
ARM="$ROOT/arms/$NAME"
exec 8>"$ARM/queue.lock"
flock -n 8 || { echo 'Queue already exists'; exit 3; }
exec > >(tee -a "$ARM/logs/queue.log") 2>&1
PY=/home/oem/hrlrvo/conda_env/bin/python
ATTEMPTS=0
queue_state() {
 printf '{"state":"%s","gpu":%s,"automatic_retry_count":%s,"note":"Foreign processes are never terminated"}\n' "$1" "$GPU" "$ATTEMPTS" > "$ARM/queue_status.json"
}
while true; do
 STATE=$($PY - "$ARM" <<'PY'
import json,sys
from pathlib import Path
arm=Path(sys.argv[1]); p=arm/'status.json'
s=json.loads(p.read_text()) if p.exists() else {}
pid=s.get('pid')
if s.get('state')=='complete':print('complete')
elif pid and Path(f'/proc/{pid}').exists() and b'run_arm.py' in Path(f'/proc/{pid}/cmdline').read_bytes():print('active')
else:print('stopped')
PY
)
 if [ "$STATE" = complete ]; then queue_state complete; echo "$NAME already complete"; exit 0; fi
 if [ "$STATE" = active ]; then queue_state monitoring_existing_run; sleep 30; continue; fi
 if [ "$ATTEMPTS" -ge 3 ]; then queue_state retry_limit_reached; echo 'Retry limit reached; no blind restart'; exit 1; fi
 UUID=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader)
 GPU_PROCESSES=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
 if grep -Fx "$UUID" <<< "$GPU_PROCESSES" >/dev/null; then
  queue_state waiting_for_idle_gpu
  echo "$(date -u +%FT%TZ) $NAME waiting: GPU $GPU occupied; leaving other jobs untouched"
  sleep 60; continue
 fi
 # Require a second idle observation rather than jumping into a momentary gap.
 queue_state confirming_idle_gpu
 sleep 30
 GPU_PROCESSES=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
 if grep -Fx "$UUID" <<< "$GPU_PROCESSES" >/dev/null; then continue; fi
 ATTEMPTS=$((ATTEMPTS+1)); queue_state restarting_same_settings
 echo "$(date -u +%FT%TZ) retry $ATTEMPTS: $NAME, same seed/settings, GPU $GPU"
 set +e
 if [ -f "$ARM/chunks/run.json" ]; then bash "$ROOT/start_arm.sh" "$NAME" --resume; CODE=$?
 else bash "$ROOT/start_arm.sh" "$NAME"; CODE=$?; fi
 set -e
 if [ "$CODE" -eq 0 ]; then queue_state complete; exit 0; fi
 if [ "$CODE" -eq 4 ]; then sleep 60; continue; fi
 # A retry failing on a now-idle GPU is not assumed to be a contention problem.
 GPU_PROCESSES=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
 if ! grep -Fx "$UUID" <<< "$GPU_PROCESSES" >/dev/null; then
  queue_state failed_without_foreign_gpu_process
  echo "Exit $CODE with GPU idle; needs diagnosis, refusing a blind retry"; exit "$CODE"
 fi
 sleep 60
done
