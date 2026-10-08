"""Lossless segment temporal MAE; auxiliary continuity signal, not quality verdict."""
import argparse,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--mode',choices=('strength_1p0_global_lineart_prior','lineart_off_control'),default='strength_1p0_global_lineart_prior');p.add_argument('--end-chunk',type=int,required=True);a=p.parse_args()
seg=ROOT/'chunks'/a.mode
rows=[]
def mae(one,two,x0,x1):
    # Spatial subsampling only reduces CPU cost, not temporal sampling.
    aa=one[::4,x0:x1:4].astype(np.int16);bb=two[::4,x0:x1:4].astype(np.int16)
    return float(np.abs(aa-bb).mean())
for c in range(1,a.end_chunk+1):
    prev=np.load(seg/f'segment_{c-1:06d}.npy',mmap_mode='r',allow_pickle=False)
    now=np.load(seg/f'segment_{c:06d}.npy',mmap_mode='r',allow_pickle=False)
    regions={'left':(0,480),'center':(480,1440),'right':(1440,1920)}
    for reg,(x0,x1) in regions.items():
        before=mae(prev[-2],prev[-1],x0,x1)
        boundary=mae(prev[-1],now[0],x0,x1)
        after=mae(now[0],now[1],x0,x1)
        rows.append({'boundary':f'C{c-1}->C{c}','region':reg,'before_mae':before,'boundary_mae':boundary,'after_mae':after,'boundary_ratio':boundary/max((before+after)/2,0.01)})
out=ROOT/'report'/f'boundary_{a.mode}_C0_C{a.end_chunk}.tsv'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0],delimiter='\t');w.writeheader();w.writerows(rows)
print(json.dumps({'file':str(out),'rows':len(rows),'max_side_boundary_ratio':max(r['boundary_ratio'] for r in rows if r['region']!='center')}))
