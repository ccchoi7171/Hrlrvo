"""CPU-only pre-MP4 RGB24 temporal MAE at three chunk boundaries."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/home/work/ai-graphics_33/Hrlrvo/pipeline/data/02_VideoOutPaint_4K_3840x2160/chunk/05')
MOD=ROOT/'modified_outputs/window_B/segments'
rows=['video\tboundary\tphase\tleft_mae\tcenter_mae\tright_mae']
for name,root in [('baseline',BASE),('modified',MOD)]:
 for next_chunk in (21,22,23):
  prev=root/f'segment_{next_chunk-1:06d}.npy' if next_chunk>21 or name=='baseline' else root/'initial_prefix_rgb24.npy'
  nxt=root/f'segment_{next_chunk:06d}.npy'
  a=np.load(prev,mmap_mode='r',allow_pickle=False)
  b=np.load(nxt,mmap_mode='r',allow_pickle=False)
  for phase,one,two in [('before',a[-2],a[-1]),('boundary',a[-1],b[0]),('after',b[0],b[1])]:
   vals=[]
   for x0,x1 in ((0,960),(960,2880),(2880,3840)):
    vals.append(float(np.abs(one[:,x0:x1].astype(np.int16)-two[:,x0:x1].astype(np.int16)).mean()))
   rows.append(f'{name}\tC{next_chunk-1}->C{next_chunk}\t{phase}\t'+'\t'.join(f'{v:.3f}' for v in vals))
path=ROOT/'report/window_B_lossless_segment_continuity.tsv'
path.write_text('\n'.join(rows)+'\n')
print('\n'.join(rows))
