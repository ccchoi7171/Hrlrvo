#!/usr/bin/env bash
set -euo pipefail

pipeline_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
video_outpainting_dir="$(cd -- "$pipeline_dir/../VideoOutpainting" && pwd)"
main_dir="$video_outpainting_dir/main"
video_id="01"
input_anchor="$pipeline_dir/data/01_PreProc_2K_1920x1080/01_PreProc_2K.mp4"
bootstrap_65f="$pipeline_dir/data/03_fullCheck_65f_30step/01/01_50374151_65f_30step.mp4"
config_path="$main_dir/configs/inference_4k.yaml"
checkpoint_path="$video_outpainting_dir/runs/hrlrvo_main_train_v1/checkpoints/checkpoint-007500"
output_root="$pipeline_dir/data/02_VideoOutPaint_4K_3840x2160"
final_output="$output_root/01_VideoOutPaint_4K.mp4"
chunk_dir="$output_root/chunk/01"
total_frames=1952
steps=30
negative_prompt="people, person, human, staff, audience, crowd, backstage, seats"
python_bin="${PYTHON_BIN:-/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python}"

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if (( $# != 0 )); then
  printf 'Usage: %s [--dry-run]\n' "$0" >&2
  exit 2
fi

bootstrap_basename="$(basename -- "$bootstrap_65f")"
if [[ "$bootstrap_basename" =~ ^([0-9]{2})_([0-9]+)_65f_30step\.mp4$ ]]; then
  bootstrap_video_id="${BASH_REMATCH[1]}"
  seed="${BASH_REMATCH[2]}"
else
  printf 'ERROR: invalid bootstrap basename: %s\n' "$bootstrap_basename" >&2
  exit 2
fi
if [[ "$bootstrap_video_id" != "$video_id" ]]; then
  printf 'ERROR: bootstrap video id %s does not match %s\n' "$bootstrap_video_id" "$video_id" >&2
  exit 2
fi

resume_args=()
if [[ -f "$chunk_dir/run.json" ]]; then
  resume_args+=(--resume)
elif [[ -d "$chunk_dir" ]] && [[ -n "$(find "$chunk_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'ERROR: nonempty chunk directory has no run.json: %s\n' "$chunk_dir" >&2
  exit 2
fi

command=(
  "$python_bin" -m ltx_outpaint.inference
  --config "$config_path"
  --anchor-video "$input_anchor"
  --output "$final_output"
  --mode normal_long_range
  --num-frames "$total_frames"
  --bootstrap-first-chunk "$bootstrap_65f"
  --checkpoint "$checkpoint_path"
  --seed "$seed"
  --steps "$steps"
  --negative-prompt "$negative_prompt"
  --segment-dir "$chunk_dir"
  --manifest-dir "$chunk_dir"
  --keep-intermediates
  --video-only-gpu-weights
  --vae-decode-streaming
  --no-ffn-token-chunking
  "${resume_args[@]}"
)

printf 'video_id=%s\ninput_anchor=%s\nbootstrap_65f=%s\nseed=%s\nfinal_output=%s\nchunk_dir=%s\nconfig=%s\nresume=%s\n' \
  "$video_id" "$input_anchor" "$bootstrap_65f" "$seed" "$final_output" \
  "$chunk_dir" "$config_path" "$([[ ${#resume_args[@]} -gt 0 ]] && echo true || echo false)"
printf 'command:'
printf ' %q' "${command[@]}"
printf '\n'

errors=0
for required_file in "$input_anchor" "$bootstrap_65f" "$config_path" \
  "$checkpoint_path/adapter_model.safetensors"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'ERROR: required file is missing: %s\n' "$required_file" >&2
    errors=1
  fi
done
if [[ ! -x "$python_bin" ]]; then
  printf 'ERROR: Python is not executable: %s\n' "$python_bin" >&2
  errors=1
fi
(( errors == 0 )) || exit 1
if "$dry_run"; then
  exit 0
fi

mkdir -p "$output_root"
export PYTHONPATH="$main_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$main_dir"
exec "${command[@]}"
