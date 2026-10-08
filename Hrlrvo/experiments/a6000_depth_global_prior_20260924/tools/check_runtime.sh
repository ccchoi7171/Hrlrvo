set -eu
ROOT=/home/oem/hrlrvo/experiments/a6000_depth_global_prior_20260924
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
/home/oem/hrlrvo/conda_env/bin/python - <<'PY'
from pathlib import Path
import json,time
r=Path('/home/oem/hrlrvo/experiments/a6000_depth_global_prior_20260924')
for stage in ['recovery','depth']:
 root=r/'recovery' if stage=='recovery' else r
 p=root/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
 print('STAGE',stage,'STATUS',json.dumps(s))
 log=root/'logs/latest.log';text=log.read_text(errors='replace') if log.exists() else ''
 events=[]
 for line in text.splitlines():
  try:
   e=json.loads(line)
   if e.get('event') in ['chunk_started','chunk_committed','run_complete','resume_rng_restored','depth_prior_scope_entered']:events.append(e)
  except (ValueError,AttributeError):pass
 for e in events[-4:]:
  if 'metrics' in e:e['metrics']={k:v for k,v in e['metrics'].items() if k in ['denoise_elapsed_seconds','vae_decode_elapsed_seconds','depth_global_prior']}
  if 'time' in e:e['elapsed_sec']=round(time.time()-e['time'],1)
  print(json.dumps(e))
 if 'Traceback' in text:print('ERROR',text[-3500:])
old=Path('/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/arms')
for name in ['02_no_carry','03_global_boost','04_native_halo']:
 s=json.loads((old/name/'status.json').read_text());print('OTHER',name,json.dumps(s))
PY
