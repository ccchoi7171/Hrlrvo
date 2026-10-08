#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pipeline_dir="$(cd -- "$script_dir/../.." && pwd)"
video_id="$(basename -- "$script_dir")"
case "$video_id" in
  01|02|03|04|05) ;;
  *)
    printf '[ERROR] unsupported video_id derived from script path: %s\n' "$video_id" >&2
    exit 2
    ;;
esac

input_mp4="$pipeline_dir/data/02_VideoOutPaint_4K_3840x2160/${video_id}_VideoOutPaint_4K.mp4"
output_root="$pipeline_dir/data/03_VideoOutPaint_4K_16fps_3840x2160"
output_mp4="$output_root/${video_id}_VideoOutPaint_4K_16fps.mp4"
temporary_output="$output_root/.${video_id}_VideoOutPaint_4K_16fps.partial.$$.mp4"

expected_width=3840
expected_height=2160
expected_frames=1952
expected_input_fps="24/1"
target_output_fps="16/1"
expected_duration_ratio="1.5"

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi
if (( $# != 0 )); then
  printf 'Usage: %s [--dry-run]\n' "$0" >&2
  exit 2
fi

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg is not available"
command -v ffprobe >/dev/null 2>&1 || die "ffprobe is not available"

ffmpeg_command=(
  ffmpeg
  -hide_banner
  -y
  -i "$input_mp4"
  -map 0:v:0
  -an
  -vf "settb=expr=1/16,setpts=N"
  -fps_mode:v passthrough
  -enc_time_base:v 1:16
  -c:v libx264
  -preset medium
  -crf 0
  -pix_fmt yuv420p
  -movflags +faststart
  "$temporary_output"
)

printf 'video_id=%s\n' "$video_id"
printf 'input=%s\n' "$input_mp4"
printf 'output=%s\n' "$output_mp4"
printf 'expected_input_fps=%s\n' "$expected_input_fps"
printf 'target_output_fps=%s\n' "$target_output_fps"
printf 'expected_frame_count=%d\n' "$expected_frames"
printf 'frame_count_policy=preserve every decoded frame exactly once; retime PTS only\n'
printf 'timing_policy=set video timebase to 1/16 and assign PTS=N; fps filter and output -r are not used\n'
printf 'expected_duration_ratio=%s\n' "$expected_duration_ratio"
printf 'input_status=%s\n' "$([[ -f "$input_mp4" ]] && echo present || echo missing)"
printf 'command:'
printf ' %q' "${ffmpeg_command[@]}"
printf '\n'

if "$dry_run"; then
  exit 0
fi

[[ -f "$input_mp4" ]] || die "input MP4 does not exist: $input_mp4"
[[ "$input_mp4" != "$output_mp4" ]] || die "input and output paths must differ"

probe_stream() {
  ffprobe \
    -v error \
    -count_frames \
    -select_streams v:0 \
    -show_entries \
      stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames \
    -of default=noprint_wrappers=1 \
    "$1"
}

probe_duration() {
  ffprobe \
    -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$1"
}

probe_field() {
  local report="$1"
  local key="$2"
  awk -F= -v key="$key" '$1 == key { print substr($0, length(key) + 2); exit }' \
    <<<"$report"
}

frame_count_from_probe() {
  local report="$1"
  local counted declared
  counted="$(probe_field "$report" nb_read_frames)"
  declared="$(probe_field "$report" nb_frames)"
  if [[ "$counted" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$counted"
  elif [[ "$declared" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$declared"
  else
    return 1
  fi
}

input_probe="$(probe_stream "$input_mp4")"
input_width="$(probe_field "$input_probe" width)"
input_height="$(probe_field "$input_probe" height)"
input_avg_fps="$(probe_field "$input_probe" avg_frame_rate)"
input_r_fps="$(probe_field "$input_probe" r_frame_rate)"
input_frames="$(frame_count_from_probe "$input_probe")" \
  || die "could not determine input frame count"
input_duration="$(probe_duration "$input_mp4")"

[[ "$input_width" == "$expected_width" ]] \
  || die "input width is $input_width, expected $expected_width"
[[ "$input_height" == "$expected_height" ]] \
  || die "input height is $input_height, expected $expected_height"
[[ "$input_avg_fps" == "$expected_input_fps" ]] \
  || die "input avg_frame_rate is $input_avg_fps, expected $expected_input_fps"
[[ "$input_r_fps" == "$expected_input_fps" ]] \
  || die "input r_frame_rate is $input_r_fps, expected $expected_input_fps"
[[ "$input_frames" == "$expected_frames" ]] \
  || die "input has $input_frames frames, expected $expected_frames"
[[ "$input_duration" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "input duration is not numeric: $input_duration"

mkdir -p "$output_root"
cleanup() {
  rm -f -- "$temporary_output"
}
trap cleanup EXIT

"${ffmpeg_command[@]}"

output_probe="$(probe_stream "$temporary_output")"
output_width="$(probe_field "$output_probe" width)"
output_height="$(probe_field "$output_probe" height)"
output_avg_fps="$(probe_field "$output_probe" avg_frame_rate)"
output_r_fps="$(probe_field "$output_probe" r_frame_rate)"
output_frames="$(frame_count_from_probe "$output_probe")" \
  || die "could not determine output frame count"
output_duration="$(probe_duration "$temporary_output")"

[[ "$output_width" == "$input_width" ]] \
  || die "output width changed: $input_width -> $output_width"
[[ "$output_height" == "$input_height" ]] \
  || die "output height changed: $input_height -> $output_height"
[[ "$output_avg_fps" == "$target_output_fps" ]] \
  || die "output avg_frame_rate is $output_avg_fps, expected $target_output_fps"
[[ "$output_r_fps" == "$target_output_fps" ]] \
  || die "output r_frame_rate is $output_r_fps, expected $target_output_fps"
[[ "$output_frames" == "$input_frames" ]] \
  || die "frame count changed: $input_frames -> $output_frames"
[[ "$output_duration" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "output duration is not numeric: $output_duration"

if ! awk \
  -v input_duration="$input_duration" \
  -v output_duration="$output_duration" \
  'BEGIN {
    expected = input_duration * 1.5
    difference = output_duration - expected
    if (difference < 0) difference = -difference
    exit !(difference <= 0.05)
  }'; then
  die "output duration $output_duration is not 1.5x input duration $input_duration"
fi

mv -f -- "$temporary_output" "$output_mp4"
trap - EXIT

printf '[DONE] video_id=%s input_frames=%s output_frames=%s output_fps=%s duration=%s output=%s\n' \
  "$video_id" "$input_frames" "$output_frames" "$output_avg_fps" \
  "$output_duration" "$output_mp4"
