#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$ROOT/tmp"
export PYTHONPATH="$ROOT/source/VideoOutpainting_test/main/src"
PY=/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python
cd "$ROOT"
for attempt in $(seq 1 120); do
  grep -q 'STOPPED_INFERIOR_0P8_AFTER_C10_COMMIT' "$ROOT/logs/stop_modified_after_C10.log" 2>/dev/null && break
  sleep 5
done
grep -q 'STOPPED_INFERIOR_0P8_AFTER_C10_COMMIT' "$ROOT/logs/stop_modified_after_C10.log"
"$PY" - <<'PY'
import json,pathlib
from ltx_outpaint.streaming_video import encode_segments
r=pathlib.Path('/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923')
records=[json.loads((r/'chunks/modified'/f'chunk_{i:06d}.json').read_text()) for i in range(11)]
assert [x['chunk_index'] for x in records]==list(range(11))
out=r/'outputs/modified/C0_C10_545f_1920x1080.mp4'
if not out.exists():
    assert encode_segments(records,r/'chunks/modified',out,fps=24,width=1920,height=1080,expected_frames=545)==545
(r/'report/modified_0p8_early_stop.json').write_text(json.dumps({'reason':'At shared C0-C7, generated sides had persistent ~8 MAE chunk-boundary jumps and visibly washed/warped geometry versus the 1.0 branch. User asked whether one branch could be stopped; 0.8 C10 was allowed to commit before termination.','last_committed_chunk':10,'frames':545,'duration_seconds':545/24,'mp4':str(out),'resume_available':True,'comparison':'strength_1p0 continues to C23'},indent=2)+'\n')
print('MODIFIED_PARTIAL_SAVED',out)
PY
"$PY" "$ROOT/scripts/boundary_metric.py" --mode modified --end-chunk 10
"$PY" "$ROOT/scripts/compare_chain.py" --mode modified --end-chunk 10
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=codec_name,width,height,avg_frame_rate,duration,nb_read_frames -of json "$ROOT/outputs/modified/C0_C10_545f_1920x1080.mp4" > "$ROOT/report/ffprobe_modified_C10.json"
echo 'MODIFIED_C10_PRESERVED'
