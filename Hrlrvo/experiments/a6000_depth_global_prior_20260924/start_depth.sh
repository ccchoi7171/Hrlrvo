#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=strength_1p0_global_depth_prior
mkdir -p "$ROOT/logs" "$ROOT/tmp"
exec 9>"$ROOT/run.lock"
flock -n 9 || { echo 'Depth experiment already running'; exit 3; }
export PATH="/home/oem/hrlrvo/conda_env/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 HRLRVO_EXPERIMENT=a6000_depth_global_prior_20260924
OFFICIAL=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/shared/LTX2.3
export PYTHONPATH="$ROOT/source/VideoOutpainting_test/main/src:$OFFICIAL/packages/ltx-core/src:$OFFICIAL/packages/ltx-trainer/src:$OFFICIAL/packages/ltx-pipelines/src"
export HF_HOME="$ROOT/tmp/hf_home" XDG_CACHE_HOME="$ROOT/tmp/xdg_cache" TMPDIR="$ROOT/tmp"
export TORCH_HOME="$ROOT/tmp/torch_home" TRITON_CACHE_DIR="$ROOT/tmp/triton_cache" CUDA_CACHE_PATH="$ROOT/tmp/cuda_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
cd "$ROOT"
CUDA_VISIBLE_DEVICES='' python scripts/preflight.py
uuid=$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if grep -Fx "$uuid" <<< "$GPU_PROCESSES" >/dev/null; then
  echo 'GPU 0 has a compute process; refusing to interfere'; exit 4
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/inference_${STAMP}.log"
ln -sfn "$(basename "$LOG")" "$ROOT/logs/latest.log"
if [ -f "$ROOT/chunks/$MODE/run.json" ]; then RESUME=--resume; else RESUME=--bootstrap-c0; fi
echo "START depth GPU=0 mode=$MODE continuation=$RESUME"
python -u run_depth_chain.py --mode "$MODE" --end-chunk 23 --rebase-temporal-position --use-latent-carry --generated-prefix-strength 1.0 --depth-global-prior --depth-bias-scale 0.5 "$RESUME" > >(tee -a "$LOG") 2>&1 &
PRODUCER=$!
echo "$PRODUCER" > "$ROOT/producer.pid"
printf '{"state":"running","pid":%s,"gpu":0,"log":"%s"}\n' "$PRODUCER" "$LOG" > "$ROOT/status.json"
CUDA_VISIBLE_DEVICES='' python -u scripts/watch_chunk_previews.py --mode "$MODE" --end-chunk 23 --producer-pid "$PRODUCER" > "$ROOT/logs/previews_${STAMP}.log" 2>&1 &
PREVIEW=$!
set +e
wait "$PRODUCER"; CODE=$?
if [ "$CODE" -eq 0 ]; then
  wait "$PREVIEW"; PREVIEW_CODE=$?
  STATE=complete
  if [ "$PREVIEW_CODE" -ne 0 ]; then STATE=preview_failed; CODE=$PREVIEW_CODE; fi
else
  kill "$PREVIEW" 2>/dev/null || true
  wait "$PREVIEW" 2>/dev/null
  PREVIEW_CODE=1; STATE=failed
fi
printf '{"state":"%s","exit_code":%s,"preview_exit_code":%s,"gpu":0,"log":"%s"}\n' "$STATE" "$CODE" "$PREVIEW_CODE" "$LOG" > "$ROOT/status.json"
echo "FINISHED depth state=$STATE inference_exit=$CODE preview_exit=$PREVIEW_CODE"
exit "$CODE"
