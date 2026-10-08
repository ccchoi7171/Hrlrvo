#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pipeline_dir="$(cd -- "$script_dir/../.." && pwd)"
repo_root="$(cd -- "$pipeline_dir/.." && pwd)"

video_id="02"
input_4k="$pipeline_dir/data/02_VideoOutPaint_4K_3840x2160/02_VideoOutPaint_4K.mp4"
original_8k="$pipeline_dir/data/00_Input_8K_7680x4320/02_Input_8K.mp4"
flashvsr_model="$repo_root/Upscaler/models/FlashVSR_v1.1"
flashvsr_reference="$repo_root/Upscaler/architect/FlashVSR_v1.1/upscale.py"
outpainting_config="$repo_root/VideoOutpainting/main/configs/inference_4k.yaml"
pipeline_source="$pipeline_dir/src/03to04/upscale_blend_16k.py"
output_root="$pipeline_dir/data/04_Final_16K_15360x8640"
final_output="$output_root/02_x4Upscale_16K.mp4"
final_manifest="$output_root/02_x4Upscale_16K.manifest.json"
work_cache="$output_root/.work/02"

python_bin="${PYTHON_BIN:-/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-FlashVSR/bin/python}"
gpu_index="${GPU_INDEX:-0}"
ffmpeg_threads="${FFMPEG_THREADS:-32}"
encoder="${ENCODER:-libx265}"
expected_frames=1952

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
if (( $# != 0 )); then printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2; fi
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg is not available"
command -v ffprobe >/dev/null 2>&1 || die "ffprobe is not available"
[[ -x "$python_bin" ]] || die "Python is not executable: $python_bin"
for required in "$pipeline_source" "$flashvsr_reference" "$outpainting_config" \
  "$flashvsr_model/checkpoint/diffusion_pytorch_model_streaming_dmd.safetensors" \
  "$flashvsr_model/checkpoint/LQ_proj_in.ckpt" "$flashvsr_model/checkpoint/TCDecoder.ckpt" \
  "$original_8k"; do
  [[ -f "$required" ]] || die "required file is missing: $required"
done

command=(
  "$python_bin" "$pipeline_source" --video-id "$video_id"
  --input-4k "$input_4k" --original-8k "$original_8k"
  --model-root "$flashvsr_model" --flashvsr-reference "$flashvsr_reference"
  --outpainting-config "$outpainting_config" --output "$final_output"
  --final-manifest "$final_manifest" --work-dir "$work_cache"
  --expected-frames "$expected_frames" --tile-size auto --tile-overlap auto
  --min-tile-size 128 --sparse-ratio 2.0 --local-range 11 --seed 0
  --gpu-index "$gpu_index" --ffmpeg-threads "$ffmpeg_threads" --encoder "$encoder"
)
if "$dry_run"; then command+=(--dry-run); fi

printf 'video_id=%s\ninput_4k=%s\ninput_4k_status=%s\noriginal_8k=%s\nflashvsr_model=%s\nflashvsr_reference=%s\nfinal_output=%s\nwork_cache=%s\n' \
  "$video_id" "$input_4k" "$([[ -f "$input_4k" ]] && echo present || echo missing)" \
  "$original_8k" "$flashvsr_model" "$flashvsr_reference" "$final_output" "$work_cache"
printf 'expected_input=3840x2160 24/1fps %dframes\n' "$expected_frames"
printf 'expected_output=15360x8640 16/1fps %dframes\n' "$expected_frames"
printf 'center=x[3840,11520) y[2160,6480) feather=64px inward_smoothstep\n'
printf 'command:'; printf ' %q' "${command[@]}"; printf '\n'
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "${command[@]}"
