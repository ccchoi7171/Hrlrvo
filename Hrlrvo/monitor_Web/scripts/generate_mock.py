#!/usr/bin/env python3
"""Generate deterministic synthetic monitor input. Never reads the training project."""
from __future__ import annotations
import argparse,json,math,random
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'mock_data'

def generate(out: Path, steps: int=160):
    out.mkdir(parents=True,exist_ok=True); rng=random.Random(20260911); now=datetime.now(timezone.utc)
    records=[]
    for step in range(1,steps+1):
        progress=step/steps; wave=lambda p,a: a*math.sin(step*p)
        native=[]; adaptive=[]; local=[]; glob=[]
        for block in range(48):
            n=min(.82,max(.18,.54+.12*math.sin(block/6+step/31)+rng.uniform(-.018,.018)))
            l=min(.82,max(.18,.50+.10*math.cos(block/7-step/37)+rng.uniform(-.018,.018)))
            native.append(round(n,6)); adaptive.append(round(1-n,6)); local.append(round(l,6)); glob.append(round(1-l,6))
        loss=max(.055,1.4*math.exp(-3.0*progress)+.025+rng.gauss(0,.018))
        lr=1e-4*(1-progress*.92)
        rec={
          'timestamp':(now-timedelta(seconds=2*(steps-step))).isoformat(),'run_id':'mock_training_run','optimizer_step':step,
          'batches_seen':step*4,'samples_seen':step*4,'loss':round(loss,7),'learning_rate':lr,
          'grad_norm_global':round(.85+wave(.11,.18)+rng.uniform(-.04,.04),6),
          'grad_norm_op_lora':round(.32+wave(.08,.06)+rng.uniform(-.015,.015),6),
          'grad_norm_a_lora':round(.23+wave(.10,.05)+rng.uniform(-.012,.012),6),
          'grad_norm_g_lora':round(.19+wave(.07,.04)+rng.uniform(-.012,.012),6),
          'grad_norm_f_lora':round(.27+wave(.09,.05)+rng.uniform(-.012,.012),6),
          'grad_norm_c2':round(.13+wave(.13,.025)+rng.uniform(-.008,.008),6),
          'grad_norm_c4_shared':round(.115+wave(.12,.02)+rng.uniform(-.007,.007),6),
          'grad_norm_r2':round(.095+wave(.14,.018)+rng.uniform(-.006,.006),6),
          'grad_norm_r4':round(.082+wave(.15,.015)+rng.uniform(-.006,.006),6),
          'grad_norm_local_fusion':round(.052+wave(.16,.012)+rng.uniform(-.004,.004),6),
          'grad_norm_global_fusion':round(.061+wave(.15,.013)+rng.uniform(-.004,.004),6),
          'local_native_ratio_mean':sum(native)/48,'local_native_ratio_min':min(native),'local_native_ratio_max':max(native),
          'local_adaptive_ratio_mean':sum(adaptive)/48,'local_adaptive_ratio_min':min(adaptive),'local_adaptive_ratio_max':max(adaptive),
          'global_local_ratio_mean':sum(local)/48,'global_local_ratio_min':min(local),'global_local_ratio_max':max(local),
          'global_global_ratio_mean':sum(glob)/48,'global_global_ratio_min':min(glob),'global_global_ratio_max':max(glob),
          'local_native_ratio_by_block':native,'local_adaptive_ratio_by_block':adaptive,
          'global_local_ratio_by_block':local,'global_global_ratio_by_block':glob,
          'nan_gradient_count':0,'inf_gradient_count':0,'zero_gradient_trainable_groups':[],
          'frozen_parameter_grad_violation_count':0,'gradient_clipped':step%17==0,
          'grad_norm_before_clip':round(1.18+wave(.09,.15),6),'grad_norm_after_clip':1.0 if step%17==0 else round(1.18+wave(.09,.15),6),
        }
        if step==37: rec.pop('grad_norm_r4')  # demonstrate historical missing metric handling
        records.append(rec)
    with (out/'train_metrics.jsonl').open('w',encoding='utf-8') as f:
        for rec in records: f.write(json.dumps(rec,separators=(',',':'))+'\n')
    with (out/'train_console.log').open('w',encoding='utf-8') as f:
        for rec in records: f.write(f"{rec['timestamp']} INFO optimizer_step={rec['optimizer_step']} loss={rec['loss']:.6f} lr={rec['learning_rate']:.8f}\n")
    (out/'run_config.yaml').write_text('mode: mock\ntraining: false\nmonitor_only: true\n',encoding='utf-8')
    (out/'run_manifest.json').write_text(json.dumps({'run_id':'mock_training_run','synthetic':True,'steps':steps},indent=2)+'\n',encoding='utf-8')
    print(out/'train_metrics.jsonl')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=DEFAULT); p.add_argument('--steps',type=int,default=160); a=p.parse_args()
    if a.steps<1: p.error('--steps must be positive')
    generate(a.output.resolve(),a.steps)
if __name__=='__main__': main()
