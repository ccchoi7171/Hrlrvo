#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_NAME="${1:?Usage: start_arm.sh 01_baseline [--resume]}"; shift
case "$ARM_NAME" in
  01_baseline) GPU=0;; 02_no_carry) GPU=1;; 03_global_boost) GPU=2;; 04_native_halo) GPU=3;; *) echo 'Unknown arm'; exit 2;;
esac
ARM="$ROOT/arms/$ARM_NAME"
mkdir -p "$ARM/logs" "$ARM/tmp" "$ARM/outputs"
exec 9>"$ARM/run.lock"
flock -n 9 || { echo 'This arm is already running'; exit 3; }
export PATH="/home/oem/hrlrvo/conda_env/bin:$PATH"
export CONDA_PREFIX=/home/oem/hrlrvo/conda_env
export CONDA_DEFAULT_ENV=hrlrvo_fhd
export CUDA_VISIBLE_DEVICES="$GPU" HRLRVO_ARM="$ARM_NAME" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ARM/source/main/src:$ROOT/shared/LTX2.3/packages/ltx-core/src:$ROOT/shared/LTX2.3/packages/ltx-trainer/src:$ROOT/shared/LTX2.3/packages/ltx-pipelines/src"
export HF_HOME="$ARM/tmp/hf_home" XDG_CACHE_HOME="$ARM/tmp/xdg_cache" TMPDIR="$ARM/tmp"
export TORCH_HOME="$ARM/tmp/torch_home" TRITON_CACHE_DIR="$ARM/tmp/triton_cache" CUDA_CACHE_PATH="$ARM/tmp/cuda_cache"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
cd "$ROOT"
# This check is CPU-only; a failure aborts before model loading.
CUDA_VISIBLE_DEVICES='' python preflight.py --arm "$ARM_NAME"
uuid=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader)
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if grep -Fx "$uuid" <<< "$GPU_PROCESSES" >/dev/null; then
  echo "GPU $GPU already has a compute process; refusing to interfere"; exit 4
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ARM/logs/inference_${STAMP}.log"
ln -sfn "$(basename "$LOG")" "$ARM/logs/latest.log"
echo "START arm=$ARM_NAME physical_gpu=$GPU base_seed=71634910 output=$ARM/outputs"
python -u run_arm.py "$@" > >(tee -a "$LOG") 2>&1 &
PRODUCER=$!
echo "$PRODUCER" > "$ARM/producer.pid"
printf '{"state":"running","pid":%s,"gpu":%s,"log":"%s"}\n' "$PRODUCER" "$GPU" "$LOG" > "$ARM/status.json"
python -u tools/watch_chunk_previews.py --arm "$ARM_NAME" --end-chunk 23 --producer-pid "$PRODUCER" > "$ARM/logs/previews_${STAMP}.log" 2>&1 &
PREVIEW=$!
set +e
wait "$PRODUCER"; CODE=$?
if [ "$CODE" -eq 0 ]; then
  wait "$PREVIEW"; PREVIEW_CODE=$?
  STATE=complete
else
  kill "$PREVIEW" 2>/dev/null || true
  wait "$PREVIEW" 2>/dev/null
  PREVIEW_CODE=1; STATE=failed
fi
printf '{"state":"%s","exit_code":%s,"preview_exit_code":%s,"gpu":%s,"log":"%s"}\n' "$STATE" "$CODE" "$PREVIEW_CODE" "$GPU" "$LOG" > "$ARM/status.json"
echo "FINISHED arm=$ARM_NAME state=$STATE inference_exit=$CODE preview_exit=$PREVIEW_CODE"
exit "$CODE"
