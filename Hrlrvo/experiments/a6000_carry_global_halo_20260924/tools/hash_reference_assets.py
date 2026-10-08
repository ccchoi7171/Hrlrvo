from pathlib import Path
import hashlib,json,time
root=Path(__file__).resolve().parents[1]
prod=Path('/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting')
old=root.parent/'longrange_fhd_recurrence_test_20260923'
files={}
for base,prefix in [(prod/'models/LTX2.3/models/LTX-2.3','models/LTX2.3/models/LTX-2.3'),(prod/'models/LTX2.3/models/gemma-3-12b','models/LTX2.3/models/gemma-3-12b')]:
 for p in sorted(base.iterdir()):
  if p.is_file() and p.suffix in ('.safetensors','.json','.model'):files[f'{prefix}/{p.name}']=p
for name in ['adapter_model.safetensors','outpainting_manifest.json']:
 files['models/checkpoint-007500/'+name]=prod/'runs/hrlrvo_main_train_v1/checkpoints/checkpoint-007500'/name
files['input/05_anchor_960x540_24fps_1169f.mkv']=old/'input/05_anchor_960x540_24fps_1169f.mkv'
for pkg in ['ltx-core','ltx-trainer','ltx-pipelines']:
 base=prod/'models/LTX2.3/packages'/pkg/'src'
 for p in sorted(base.rglob('*.py')):files[f'models/LTX2.3/packages/{pkg}/src/{p.relative_to(base)}']=p
out={}
inventory=json.loads((root/'shared/portable_inventory.json').read_text())['files']
files={k:v for k,v in files.items() if not k.startswith('models/LTX2.3/models/') or k in inventory}
for rel,p in files.items():
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(16*1024*1024):h.update(b)
 out[rel]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 if p.stat().st_size>100000000: print(rel,h.hexdigest(),flush=True)
(root/'shared/reference_assets.json').write_text(json.dumps(out,indent=2)+'\n')
print('FILES',len(out),flush=True)
