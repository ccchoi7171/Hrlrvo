#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$ROOT/tmp"
PY=/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python
cd "$ROOT"
for attempt in $(seq 1 720); do
  if [ -s "$ROOT/outputs/modified/C0_C23_manifest.json" ] && [ -s "$ROOT/outputs/strength_1p0/C0_C23_manifest.json" ]; then break; fi
  if grep -Eq 'MODIFIED_C23_EXIT=[1-9]|STRENGTH_1P0_EXIT=[1-9]|Traceback' "$ROOT/logs/run_modified_C23_resume.log" "$ROOT/logs/run_strength_1p0_C23.log" 2>/dev/null; then
    echo 'One branch failed before finalization'; exit 1
  fi
  sleep 60
done
[ -s "$ROOT/outputs/modified/C0_C23_manifest.json" ]
[ -s "$ROOT/outputs/strength_1p0/C0_C23_manifest.json" ]
for mode in modified strength_1p0; do
  "$PY" "$ROOT/scripts/compare_chain.py" --mode "$mode" --end-chunk 23
  "$PY" "$ROOT/scripts/boundary_metric.py" --mode "$mode" --end-chunk 23
  ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=codec_name,width,height,avg_frame_rate,duration,nb_read_frames -of json "$ROOT/outputs/$mode/C0_C23_1169f_1920x1080.mp4" > "$ROOT/report/ffprobe_${mode}_C23.json"
done
"$PY" "$ROOT/scripts/compare_strength_ab.py"
stat -c '%n %s %Y %a' /home/work/ai-graphics_33/Hrlrvo/pipeline/data/00_Input_8K_7680x4320/05_Input_8K.mp4 /home/work/ai-graphics_33/Hrlrvo/pipeline/data/02_VideoOutPaint_4K_3840x2160/05_VideoOutPaint_4K.mp4 /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/main/src/ltx_outpaint/inference.py > "$ROOT/report/protected_stat_after.txt"
git -C /home/work/ai-graphics_33/Hrlrvo status --short > "$ROOT/report/original_git_status_after.txt"
echo 'FINAL_DIAGNOSTICS_COMPLETE'
