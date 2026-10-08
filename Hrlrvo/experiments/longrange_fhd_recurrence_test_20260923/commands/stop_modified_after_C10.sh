#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923
PID=361284
RECORD="$ROOT/chunks/modified/chunk_000010.json"
while [ ! -s "$RECORD" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then echo '0.8 producer stopped before C10 commit'; exit 1; fi
  sleep 2
done
ROOT="$ROOT" /home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python - <<'PY'
import json,os,pathlib
r=pathlib.Path(os.environ['ROOT']);d=json.loads((r/'chunks/modified/chunk_000010.json').read_text())
assert d['chunk_index']==10 and d['chunk_seed']==71634910+480
for p in (r/'chunks/modified'/d['segment_file'],r/'chunks/modified'/d['next_prefix_file'],r/'latent_states/modified'/d['carry_file']):
    assert p.is_file() and p.stat().st_size>0,p
print('C10_COMMIT_AND_RESUME_FILES_VERIFIED',d['new_output_frame_range'],flush=True)
PY
CMD=$(ps -p "$PID" -o args= || true)
case "$CMD" in
  *"run_fhd_chain.py --mode modified --end-chunk 23"*)
    kill -TERM "$PID"
    echo "STOPPED_INFERIOR_0P8_AFTER_C10_COMMIT pid=$PID utc=$(date -u +%FT%TZ)"
    ;;
  *) echo "PID command changed; refusing to signal: $CMD"; exit 1;;
esac
