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
  status=json.loads((arm/'status.json').read_text()) if (arm/'status.json').exists() else {}
  queue=json.loads((arm/'queue_status.json').read_text()) if (arm/'queue_status.json').exists() else {}
  pid=status.get('pid'); proc=Path('/proc/'+str(pid))
  alive=bool(pid and proc.exists() and b'run_arm.py' in (proc/'cmdline').read_bytes())
  log=(arm/'logs/latest.log').read_text(errors='replace') if (arm/'logs/latest.log').exists() else ''
  events=[]
  for line in log.splitlines():
   try:
    e=json.loads(line)
    if isinstance(e,dict) and 'event' in e:events.append(e)
   except (ValueError,TypeError):pass
  started=alive and any(e['event']=='chunk_started' for e in events)
  row={'arm':name,'pid':pid if alive else None,'alive':alive,'queue':queue.get('state'),'last_event':events[-1]['event'] if events else None,'chunk_started':started}
  rows.append(row)
  if not alive and status.get('state')=='running':
   status.update(state='interrupted',note='Previous attempt stopped during queue guard repair; not an active inference process')
   (arm/'status.json').write_text(json.dumps(status)+'\n')
  if queue.get('state') in ['retry_limit_reached','failed_without_foreign_gpu_process']:
   print(log[-10000:],flush=True);raise RuntimeError('Automatic queue needs diagnosis: '+name)
 print(datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps(rows),flush=True)
 if all(r['chunk_started'] for r in rows):
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
else:raise TimeoutError('Startup deadline reached; leave compatible queues intact')
PY
