#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES='' /home/oem/hrlrvo/conda_env/bin/python -u - <<'PY'
from pathlib import Path
import json,time,datetime,statistics
new=Path('/home/oem/hrlrvo/experiments/a6000_depth_global_prior_20260924')
old=Path('/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/arms')
print('NOW',datetime.datetime.now(datetime.timezone.utc).isoformat())
for name,root in [('02_no_carry',old/'02_no_carry'),('03_global_boost',old/'03_global_boost'),('04_native_halo',old/'04_native_halo'),('C0_recovery',new/'recovery'),('depth',new)]:
    log=root/'logs/latest.log'
    events=[]
    if log.exists():
        for line in log.read_text(errors='replace').splitlines():
            try:
                event=json.loads(line)
                if isinstance(event,dict):events.append(event)
            except ValueError:pass
    starts={e['chunk_index']:e for e in events if e.get('event')=='chunk_started'}
    commits={e['chunk_index']:e for e in events if e.get('event')=='chunk_committed'}
    rows=[]
    for index,event in sorted(commits.items()):
        metrics=event.get('metrics',{})
        # Commit events have no wall timestamp; measure consecutive chunk starts.
        # This includes inference, validation, commit and next input preparation.
        elapsed=starts[index+1]['time']-starts[index]['time'] if index in starts and index+1 in starts else None
        row={'chunk_index':index,'elapsed_seconds':elapsed,
             'timings':{k:v for k,v in metrics.items() if 'elapsed' in k or 'time' in k or 'seconds' in k}}
        rows.append(row)
    values=[row['elapsed_seconds'] for row in rows if row['chunk_index']>0 and row['elapsed_seconds'] is not None]
    current=[{'chunk_index':i,'elapsed_seconds':time.time()-e['time']} for i,e in starts.items() if i not in commits]
    result={'arm':name,'timing_definition':'consecutive_chunk_start_interval_including_save_and_next_input_preparation','committed':rows,'mean_C1plus_seconds':statistics.mean(values) if values else None,'range_C1plus_seconds':[min(values),max(values)] if values else None,'current':current}
    print(json.dumps(result),flush=True)
PY
