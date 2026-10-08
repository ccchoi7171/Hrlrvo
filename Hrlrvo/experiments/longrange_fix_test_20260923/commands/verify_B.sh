#!/usr/bin/env bash
set -euo pipefail
EXP=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fix_test_20260923
ROOT="$EXP/modified_outputs/window_B"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt -of json "$ROOT/window_B_global_1008_1168.mp4" > "$EXP/report/modified_B_ffprobe.json"
stat -c '%n %s %Y %a' /home/work/ai-graphics_33/Hrlrvo/pipeline/data/02_VideoOutPaint_4K_3840x2160/05_VideoOutPaint_4K.mp4 /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/main/src/ltx_outpaint/inference.py > "$EXP/report/protected_file_stat_after.txt"
diff -u "$EXP/report/protected_file_stat_before.txt" "$EXP/report/protected_file_stat_after.txt"
sha256sum /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/main/src/ltx_outpaint/inference.py | grep '^55525cd3fad501caf57fc1928104240521390bc0a9b7faa063f7f04625340286'
git -C /home/work/ai-graphics_33/Hrlrvo status --short > "$EXP/report/production_git_status_after.txt"
cat "$EXP/report/modified_B_ffprobe.json"
