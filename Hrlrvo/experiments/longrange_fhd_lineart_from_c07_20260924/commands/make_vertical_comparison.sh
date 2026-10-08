#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_from_c07_20260924
TOP="$ROOT/outputs/strength_1p0_global_lineart_from_c07/progress/C00_C23_1169f.mp4"
BOTTOM=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923/outputs/strength_1p0/C0_C23_1169f_1920x1080.mp4
OUT="$ROOT/comparisons/lineart_vs_baseline_vertical_half_960x1080.mp4"
PARTIAL="$ROOT/comparisons/.lineart_vs_baseline_vertical_half_960x1080.partial.mp4"
FONT=/usr/share/fonts/truetype/nanum/NanumGothic.ttf
cd "$ROOT"
test ! -e "$OUT"
test ! -e "$PARTIAL"
sha256sum "$TOP" "$BOTTOM" > "$ROOT/comparisons/vertical_comparison_sources.sha256"
ffmpeg -hide_banner -nostdin -n \
  -threads 2 -i "$TOP" -threads 2 -i "$BOTTOM" \
  -filter_complex_threads 2 \
  -filter_complex "[0:v]setpts=PTS-STARTPTS,scale=960:540:flags=lanczos,setsar=1,drawtext=fontfile='$FONT':text='LineArt 있음':fontsize=28:fontcolor=white:x=16:y=16:box=1:boxcolor=black@0.75:boxborderw=8[top];[1:v]setpts=PTS-STARTPTS,scale=960:540:flags=lanczos,setsar=1,drawtext=fontfile='$FONT':text='LineArt 없음':fontsize=28:fontcolor=white:x=16:y=16:box=1:boxcolor=black@0.75:boxborderw=8[bottom];[top][bottom]vstack=inputs=2:shortest=1[v]" \
  -map '[v]' -an -c:v libx264 -threads 4 -preset fast -crf 21 \
  -maxrate 4M -bufsize 8M -profile:v high -level:v 4.0 -pix_fmt yuv420p \
  -r 24 -fps_mode cfr -frames:v 1169 -g 48 -movflags +faststart "$PARTIAL"
sha256sum -c "$ROOT/comparisons/vertical_comparison_sources.sha256"
mv -n "$PARTIAL" "$OUT"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,duration:format=size,bit_rate \
  -of json "$OUT" > "$ROOT/comparisons/vertical_comparison_ffprobe.json"
ffmpeg -hide_banner -nostdin -v error -threads 2 -i "$OUT" -f null -
ffmpeg -hide_banner -nostdin -v error -n -ss 30 -threads 2 -i "$OUT" -frames:v 1 -q:v 3 "$ROOT/comparisons/lineart_vs_baseline_vertical_30s.jpg"
printf '\nCOMPLETED: %s\n' "$OUT"
