set -euo pipefail
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
cd "$ROOT"
/home/oem/hrlrvo/conda_env/bin/python -u - <<'PY'
from pathlib import Path
import json,subprocess,time,datetime
root=Path.cwd(); names=['01_baseline','02_no_carry','03_global_boost','04_native_halo']
deadline=time.monotonic()+1800; ready_since=None
while time.monotonic()<deadline:
 rows=[]
 for name in names:
  arm=root/'arms'/name
  status=json.loads((arm/'status.json').read_text()) if (arm/'status.json').exists() else {'state':'preflight'}
  log=(arm/'logs/latest.log').read_text(errors='replace') if (arm/'logs/latest.log').exists() else ''
  events=[]
  for line in log.splitlines():
   try:
    event=json.loads(line)
    if isinstance(event,dict) and 'event' in event:events.append(event)
   except (ValueError,TypeError):pass
  started=any(e['event']=='chunk_started' for e in events)
  alive=status.get('state')=='running' and Path('/proc/'+str(status.get('pid'))).exists()
  row={'arm':name,'state':status['state'],'pid':status.get('pid'),'alive':alive,'last_event':events[-1]['event'] if events else None,'chunk_started':started}
  rows.append(row)
  if status.get('state')=='failed' or (status.get('state')=='running' and not alive):
   print(log[-14000:],flush=True);raise RuntimeError('Producer failed: '+name)
 print(datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps(rows),flush=True)
 if all(r['alive'] and r['chunk_started'] for r in rows):
  if ready_since is None:ready_since=time.monotonic()
  if time.monotonic()-ready_since>=90:
   gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv'],text=True)
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True)
   app_pids={int(line.split(',')[1].strip()) for line in apps.splitlines()[1:] if line.strip()}
   if all(r['pid'] in app_pids for r in rows):
    report={'launch_verified':True,'completion':False,'halo_native_tokens':1,'rows':rows,'gpu':gpu,'compute_apps':apps,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (root/'verification/startup_health.json').write_text(json.dumps(report,indent=2)+'\n')
    print(gpu,apps,'ALL_FOUR_IN_DENOISING',flush=True);break
 else:ready_since=None
 time.sleep(30)
else:raise TimeoutError('Startup deadline reached; inspect existing processes, do not duplicate launch')
PY
