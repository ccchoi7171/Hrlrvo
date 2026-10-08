#!/usr/bin/env bash
# Safe one-shot transition; never retries/kills a foreign process.
set -euo pipefail
ROOT=/home/oem/hrlrvo/experiments/a6000_depth_global_prior_20260924
export PATH=/home/oem/hrlrvo/conda_env/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
python -u - <<'PY'
from pathlib import Path
import json,time,datetime
r=Path.cwd();deadline=time.monotonic()+3600
while time.monotonic()<deadline:
    s=json.loads((r/'recovery/status.json').read_text())
    state=s.get('state')
    if state=='complete' and s.get('exit_code')==0:
        print('RECOVERY_COMPLETED',flush=True);break
    if state=='failed':raise RuntimeError('C0 recovery failed; depth launch prohibited')
    p=Path('/proc')/str(s.get('pid'))
    if state=='running' and (not p.exists() or b'run_arm.py' not in (p/'cmdline').read_bytes()):
        raise RuntimeError('C0 recovery producer missing; depth launch prohibited')
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'waiting_for_exact_C0_recovery',flush=True)
    time.sleep(30)
else:raise TimeoutError('C0 recovery deadline reached; no depth launch')
PY
CUDA_VISIBLE_DEVICES='' python scripts/bootstrap_and_configure.py | tee logs/c0_bootstrap_verified.log
CUDA_VISIBLE_DEVICES='' python scripts/preflight.py | tee logs/depth_prelaunch.log
if [ -f "$ROOT/status.json" ]; then echo 'Depth run state exists; refusing duplicate launch'; exit 1; fi
pane=$(tmux new-window -d -P -F '#{pane_id}' -t hrlrvo -n exp_depth_gpu0 "bash -lc 'bash $ROOT/start_depth.sh; echo DEPTH_WRAPPER_EXIT=\$?; exec bash'")
echo "$pane" > "$ROOT/tmux_pane.txt"
echo "DEPTH_TMUX_PANE=$pane"
python -u - <<'PY'
from pathlib import Path
import datetime,json,subprocess,time
r=Path.cwd();deadline=time.monotonic()+900;ready_since=None
expected={'02_no_carry':1203845,'03_global_boost':1206052,'04_native_halo':1206190}
while time.monotonic()<deadline:
    status=r/'status.json'
    s=json.loads(status.read_text()) if status.exists() else {}
    if s.get('state') in ['failed','preview_failed']:raise RuntimeError('Depth producer failed; inspect latest.log')
    pid=s.get('pid');p=Path('/proc')/str(pid)
    alive=bool(pid and p.exists() and b'run_depth_chain.py' in (p/'cmdline').read_bytes())
    text=(r/'logs/latest.log').read_text(errors='replace') if (r/'logs/latest.log').exists() else ''
    events=[]
    for line in text.splitlines():
        try:
            event=json.loads(line)
            if isinstance(event,dict) and 'event' in event:events.append(event)
        except ValueError:pass
    rng=next((e for e in events if e['event']=='resume_rng_restored'),None)
    scoped=next((e for e in events if e['event']=='depth_prior_scope_entered'),None)
    ready=alive and rng and rng.get('cpu_cuda_rng_equal') and rng.get('first_generated_chunk')==1 and scoped and scoped.get('chunk_index')==1
    if any(e.get('event')=='chunk_started' and e.get('chunk_index')==0 for e in events):
        raise RuntimeError('Depth producer improperly regenerated C0')
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps({'pid':pid,'alive':alive,'last_event':events[-1]['event'] if events else None,'C1_scope_entered':bool(ready)}),flush=True)
    if ready:
        if ready_since is None:ready_since=time.monotonic()
        if time.monotonic()-ready_since>=90:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True)
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True)
            uuid=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            rows=[line.split(',') for line in apps.splitlines()[1:] if line.strip()]
            if not any(row[0].strip()==uuid and int(row[1].strip())==pid for row in rows):
                raise RuntimeError('Depth process not found on physical GPU0')
            other_alive={}
            for name,opid in expected.items():
                op=Path('/proc')/str(opid)
                ok=op.exists() and b'run_arm.py' in (op/'cmdline').read_bytes()
                other_alive[name]={'pid':opid,'alive':ok}
                if not ok:raise RuntimeError('Another experiment PID changed; inspect before reporting')
            report={'launch_verified':True,'completion':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'physical_gpu':0,'pid':pid,'rng_restore':rng,'prior_scope':scoped,'other_arms':other_alive,'gpu':gpu,'compute_apps':apps,'c0_exact_hash_gate':json.loads((r/'report/c0_recovery_verification.json').read_text())}
            (r/'verification/startup_health.json').write_text(json.dumps(report,indent=2)+'\n')
            print(gpu,apps,'DEPTH_C1_LAUNCH_VERIFIED',flush=True);break
    else:ready_since=None
    time.sleep(15)
else:raise TimeoutError('Depth startup deadline; do not blindly relaunch')
PY
