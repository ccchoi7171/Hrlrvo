#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/oem/hrlrvo/experiments/a6000_depth_global_prior_20260924
cd "$ROOT"
export PATH=/home/oem/hrlrvo/conda_env/bin:$PATH
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
python - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
root=Path.cwd()
with tarfile.open('progress_watchers.tar') as bundle:
    manifest=json.load(bundle.extractfile('verification/progress_deploy_manifest.json'))
    allowed=set(manifest)|{'verification/progress_deploy_manifest.json'}
    if {m.name for m in bundle.getmembers()} != allowed:
        raise RuntimeError('Unexpected progress archive members')
    for member in bundle.getmembers():
        if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
            raise RuntimeError('Unsafe archive member')
        content=bundle.extractfile(member).read()
        path=root/member.name
        if member.name in manifest and hashlib.sha256(content).hexdigest()!=manifest[member.name]:
            raise RuntimeError('Archive content hash mismatch')
        if path.exists() and (path.is_symlink() or path.read_bytes()!=content):
            raise RuntimeError(f'Refusing to overwrite existing content: {path}')
    for member in bundle.getmembers():
        path=root/member.name
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(bundle.extractfile(member).read())
print('PROGRESS_DEPLOY_HASHES_PASS')
PY
python -m pytest -q tests/test_progress_videos.py tests/test_progress_arm_wrapper.py | tee verification/progress_remote_tests.log
python - <<'PY'
from pathlib import Path
import datetime,json,shlex,subprocess
r=Path.cwd(); registry=r/'verification/progress_watchers_started.json'
if registry.exists():raise RuntimeError('Progress watcher registry exists; inspect before relaunching')
arms=['02_no_carry','03_global_boost','04_native_halo','depth']
started=[]
for arm in arms:
    log=r/'logs'/f'progress_{arm}.log'
    if log.exists():raise RuntimeError(f'Progress log exists; refusing overwrite: {log}')
    shell=(f"cd {shlex.quote(str(r))}; "
           "export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1; "
           f"echo 'START CPU cumulative watcher arm={arm} milestones=5,10,15,20,24'; "
           f"nice -n 10 /home/oem/hrlrvo/conda_env/bin/python -u scripts/run_progress_for_arm.py --arm {arm} "
           f"> >(tee {shlex.quote(str(log))}) 2>&1; rc=$?; "
           f"printf '%s\\n' \"$rc\" > {shlex.quote(str(log.with_suffix('.exit')))}; "
           "echo PROGRESS_WATCHER_EXIT=$rc; exec bash")
    pane=subprocess.check_output(['tmux','new-window','-d','-P','-F','#{pane_id}',
          '-t','hrlrvo','-n',f'progress_{arm}','bash -lc '+shlex.quote(shell)],text=True).strip()
    started.append({'arm':arm,'pane':pane,'log':str(log)})
    registry.write_text(json.dumps({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'milestone_counts':[5,10,15,20,24],'cpu_only':True,'watchers':started},indent=2)+'\n')
print(json.dumps(started,indent=2))
PY
