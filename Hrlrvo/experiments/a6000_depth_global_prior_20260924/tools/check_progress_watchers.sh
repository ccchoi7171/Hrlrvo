#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/oem/hrlrvo/experiments/a6000_depth_global_prior_20260924
cd "$ROOT"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/oem/hrlrvo/conda_env/bin/python -u - <<'PY'
import json,sys,datetime
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
from run_progress_for_arm import arm_spec,validate_producer
from watch_progress_videos import validate_video
r=Path.cwd(); results=[]
registry=json.loads((r/'verification/progress_watchers_started.json').read_text())
for item in registry['watchers']:
    spec=arm_spec(item['arm']); progress=spec.root/'outputs'/spec.mode/'progress'
    status=spec.root/'status.json'
    state=json.loads(status.read_text()) if status.exists() else {}
    if state.get('state')=='running':validate_producer(state['pid'],spec)
    watcher_pids=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
            if b'scripts/run_progress_for_arm.py' in cmd and item['arm'].encode() in cmd:
                env=proc.joinpath('environ').read_bytes().split(b'\0')
                if b'CUDA_VISIBLE_DEVICES=' not in env:raise RuntimeError('Watcher is not CPU-isolated')
                watcher_pids.append(int(proc.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    if len(watcher_pids)!=1:raise RuntimeError(f'Expected one live watcher for {item["arm"]}: {watcher_pids}')
    exit_file=Path(item['log']).with_suffix('.exit')
    if exit_file.exists():raise RuntimeError(f'Watcher exited: {exit_file.read_text()}')
    latest=json.loads((progress/'latest.json').read_text()) if (progress/'latest.json').exists() else None
    if latest:
        meta=validate_video(Path(latest['path']),expected_frames=latest['frame_count'],width=1920,height=1080)
        latest['fresh_decode_verified']=True
    result={'arm':item['arm'],'watcher_pids':watcher_pids,'producer':state,
        'committed_chunks':len(list((spec.root/'chunks'/spec.mode).glob('chunk_*.json'))),
        'latest_cumulative':latest,'pane':item['pane']}
    results.append(result);print(json.dumps(result),flush=True)
report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'watchers':results}
(r/'verification/progress_health.json').write_text(json.dumps(report,indent=2)+'\n')
PY
