#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
SRC=/home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/05_Input_8K.mp4
TMP="$ROOT/input/.05_anchor_960x540_24fps_1169f.partial.mkv"
OUT="$ROOT/input/05_anchor_960x540_24fps_1169f.mkv"
test ! -e "$OUT" && test ! -e "$TMP"
ffmpeg -nostdin -hide_banner -loglevel info -threads 8 -i "$SRC" -map 0:v:0 -an \
  -vf 'scale=960:540:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24' \
  -frames:v 1169 -fps_mode passthrough -c:v ffv1 -level 3 -pix_fmt yuv420p10le -threads 8 "$TMP"
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_read_frames,pix_fmt -of json "$TMP" > "$ROOT/input/anchor_ffprobe.pending.json"
/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python - "$ROOT/input/anchor_ffprobe.pending.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))['streams'][0]
assert (s['width'],s['height'],s['avg_frame_rate'],int(s['nb_read_frames']))==(960,540,'24/1',1169),s
PY
mv "$TMP" "$OUT"
mv "$ROOT/input/anchor_ffprobe.pending.json" "$ROOT/input/anchor_ffprobe.json"
sha256sum "$OUT" > "$ROOT/input/anchor_sha256.txt"
