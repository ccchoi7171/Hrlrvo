#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pipeline=/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K
stage="$pipeline/data/03_Upscale_16K_15360x8640/01"
python=/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-FlashVSR/bin/python

command=(
  "$python" -I "$root/final_only/final_only.py"
  --video-id 01
  --input-4k "$pipeline/data/02_Outpaint_4K_3840x2160/01/final/01_Outpaint_4K.mp4"
  --original-8k "$pipeline/data/00_Input_8K_7680x4320/01_Input_8K.mp4"
  --model-root "$pipeline/src/models/flashvsr_v1_1"
  --flashvsr-reference "$pipeline/src/flashvsr/architect/FlashVSR_v1.1/upscale.py"
  --outpainting-config "$pipeline/src/video_outpainting/main/configs/inference_4k.yaml"
  --output "$stage/final/01_Certification_Final_16K_16fps.mp4"
  --raw-output "$stage/raw_flashvsr/01_FlashVSR_raw_16K_16fps.mp4"
  --final-manifest "$stage/metadata/final_manifest.json"
  --work-dir "$stage/segments"
  --stage-manifest "$stage/segments/flashvsr_stage_manifest.json"
  --stage-request "$stage/segments/flashvsr_stage_request.json"
  --expected-frames 1952
  --tile-size 320
  --tile-overlap 32
  --min-tile-size 128
  --sparse-ratio 2.0
  --local-range 11
  --seed 50374151
  --gpu-index 0
  --ffmpeg-threads 32
  --encoder libx265
  --renderer threaded
  --compose-workers 8
)

if [[ ${1:-} == --preflight-only ]]; then
  exec "${command[@]}" --preflight-only
fi
if [[ $# -ne 0 ]]; then
  printf 'Unsupported arguments: %s\n' "$*" >&2
  exit 2
fi
if [[ -d /proc/929284 ]] && grep -qa 'run_cert.py' /proc/929284/cmdline; then
  echo 'Refusing final-only start: original video01 run_cert process is still active' >&2
  exit 1
fi
exec flock -n "$pipeline/state/locks/upscale.lock" \
  flock -n "$pipeline/state/locks/video_01.lock" \
  "$python" -I "$root/tools/run_with_status.py" "${command[@]}"
