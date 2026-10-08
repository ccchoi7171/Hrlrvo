"""CPU-only neighbouring-frame MAE for context; not a geometry quality metric."""
from pathlib import Path
import av, numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/home/work/ai-graphics_33/Hrlrvo/pipeline/data/02_VideoOutPaint_4K_3840x2160/05_VideoOutPaint_4K.mp4')
MOD=ROOT/'modified_outputs/window_B/window_B_global_1008_1168.mp4'
pairs=[(1023,1024),(1024,1025),(1025,1026),(1071,1072),(1072,1073),(1073,1074),(1119,1120),(1120,1121),(1121,1122)]
need={f for pair in pairs for f in pair}
def frames(path,offset):
    selected={}
    with av.open(str(path)) as c:
        for i,frame in enumerate(c.decode(video=0)):
            global_i=i+offset
            if global_i in need:
                selected[global_i]=frame.to_ndarray(format='rgb24')
                if len(selected)==len(need):break
    if len(selected)!=len(need):raise RuntimeError('Frame decode incomplete')
    return selected
rows=['video\tframes\tleft_mae\tcenter_mae\tright_mae']
for label,path,offset in [('baseline',BASE,0),('modified',MOD,1008)]:
    imgs=frames(path,offset)
    for f0,f1 in pairs:
        a,b=imgs[f0],imgs[f1]
        vals=[]
        for x0,x1 in [(0,960),(960,2880),(2880,3840)]:
            vals.append(float(np.abs(a[:,x0:x1].astype(np.int16)-b[:,x0:x1].astype(np.int16)).mean()))
        rows.append(f'{label}\t{f0}->{f1}\t'+'\t'.join(f'{v:.3f}' for v in vals))
path=ROOT/'report/window_B_temporal_mae.tsv'
path.write_text('\n'.join(rows)+'\n')
print('\n'.join(rows))
