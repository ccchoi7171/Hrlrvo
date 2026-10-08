#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REC="$ROOT/recovery"
OLD=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
mkdir -p "$REC/logs" "$REC/tmp"
exec 9>"$ROOT/run.lock"
flock -n 9 || { echo 'GPU0 experiment/recovery already active'; exit 3; }
for name in assets shared; do
  if [ ! -e "$REC/$name" ]; then ln -s "$OLD/$name" "$REC/$name"; fi
done
export PATH=/home/oem/hrlrvo/conda_env/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HRLRVO_ARM=01_baseline PYTHONDONTWRITEBYTECODE=1
export HRLRVO_EXPERIMENT=a6000_depth_c0_recovery
OFFICIAL="$OLD/shared/LTX2.3"
export PYTHONPATH="$REC/arms/01_baseline/source/main/src:$OFFICIAL/packages/ltx-core/src:$OFFICIAL/packages/ltx-trainer/src:$OFFICIAL/packages/ltx-pipelines/src"
export HF_HOME="$REC/tmp/hf_home" XDG_CACHE_HOME="$REC/tmp/xdg_cache" TMPDIR="$REC/tmp"
export TORCH_HOME="$REC/tmp/torch_home" TRITON_CACHE_DIR="$REC/tmp/triton_cache" CUDA_CACHE_PATH="$REC/tmp/cuda_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
CUDA_VISIBLE_DEVICES='' python - "$ROOT" <<'PY'
from pathlib import Path
import hashlib,json,sys
r=Path(sys.argv[1]);m=json.loads((r/'verification/recovery_snapshot.json').read_text())
for name,digest in m.items():
 if hashlib.sha256((r/'recovery'/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Recovery source drift: '+name)
print('C0 recovery frozen baseline source verified')
PY
uuid=$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if grep -Fx "$uuid" <<< "$GPU_PROCESSES" >/dev/null; then echo 'GPU0 occupied'; exit 4; fi
cd "$REC"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$REC/logs/recovery_${STAMP}.log"
ln -sfn "$(basename "$LOG")" "$REC/logs/latest.log"
python -u run_arm.py --end-chunk 0 > >(tee -a "$LOG") 2>&1 &
PID=$!
printf '{"state":"running","pid":%s,"gpu":0,"log":"%s"}\n' "$PID" "$LOG" > "$REC/status.json"
set +e
wait "$PID"; CODE=$?
STATE=failed; if [ "$CODE" -eq 0 ]; then STATE=complete; fi
printf '{"state":"%s","exit_code":%s,"gpu":0,"log":"%s"}\n' "$STATE" "$CODE" "$LOG" > "$REC/status.json"
echo "RECOVERY_EXIT=$CODE"
exit "$CODE"
