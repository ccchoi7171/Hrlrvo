set -eu
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
cd "$ROOT"
date -u
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
/home/oem/hrlrvo/conda_env/bin/python - <<'PY'
from pathlib import Path
import json
root=Path.cwd()
for arm in sorted((root/'arms').iterdir()):
 s=json.loads((arm/'status.json').read_text()) if (arm/'status.json').exists() else {}
 q=json.loads((arm/'queue_status.json').read_text()) if (arm/'queue_status.json').exists() else {}
 pid=s.get('pid'); path=Path('/proc/'+str(pid))
 alive=bool(pid and path.exists() and b'run_arm.py' in (path/'cmdline').read_bytes())
 text=(arm/'logs/latest.log').read_text(errors='replace') if (arm/'logs/latest.log').exists() else ''
 events=[]
 for line in text.splitlines():
  try:
   e=json.loads(line)
   if 'event' in e:events.append(e['event'])
  except (ValueError,TypeError):pass
 print(json.dumps({'arm':arm.name,'pid':pid,'alive':alive,'status':s.get('state'),'queue':q.get('state'),'last_event':events[-1] if events else None,'last_log':text.splitlines()[-2:]},ensure_ascii=False))
PY
ps -u oem -o pid,ppid,stat,etime,args | grep -E 'queue_arm|start_arm|run_arm.py|preflight.py' | grep -v grep || true
