"""Fail-closed CPU preflight. Does not load models onto CUDA or run inference."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import struct
import subprocess
import sys
from dataclasses import asdict

ROOT = Path(__file__).resolve().parent
NAMES = ['01_baseline','02_no_carry','03_global_boost','04_native_halo']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=NAMES,required=True)
    a=p.parse_args()
    arm=ROOT/'arms'/a.arm
    spec=json.loads((arm/'config.json').read_text())
    settings=json.loads((arm/'settings.json').read_text())
    reference=json.loads((ROOT/'shared/effective_strength_1p0_settings.json').read_text())
    reference['ltx_repo']=settings['ltx_repo']
    reference['provenance']['ltx_source_repo']=settings['provenance']['ltx_source_repo']
    for key in ['base_model_path','text_encoder_path','adapter_path']:
        reference['model'][key]=settings['model'][key]
    assert settings==reference, 'Non-path baseline setting changed'
    assert all((ROOT/'arms'/n/'settings.json').read_bytes()==(arm/'settings.json').read_bytes() for n in NAMES)
    assert spec['base_seed']==71634910 and spec['generated_prefix_strength']==1.0
    assert spec['rebase_temporal_position'] is True
    assert spec['use_latent_carry']==(a.arm!='02_no_carry')
    assert spec['early_global_boost']==(a.arm=='03_global_boost')
    assert spec['native_halo_radius_tokens']==(1 if a.arm=='04_native_halo' else 0)
    assert spec['physical_gpu']==NAMES.index(a.arm) and spec['start_policy']=='fresh_C0'
    source=arm/'source/main/src'
    manifest=json.loads((arm/'source_manifest.json').read_text())
    actual={str(f.relative_to(source)):digest(f) for f in source.rglob('*') if f.is_file() and '__pycache__' not in str(f) and f.suffix!='.pyc'}
    assert actual==manifest, 'Arm source changed since freeze'
    base=json.loads((ROOT/'shared/reference_source_hashes.json').read_text())
    permitted={'01_baseline':set(),'02_no_carry':{'ltx_outpaint/inference.py'},'03_global_boost':{'ltx_outpaint/inference.py'},'04_native_halo':{'ltx_outpaint/native_local_attention.py'}}
    assert actual.keys()==base.keys()
    assert {k for k in actual if actual[k]!=base[k]}==permitted[a.arm]
    for f in [ROOT/'run_arm.py',ROOT/'preflight.py',*source.rglob('*.py')]:
        ast.parse(f.read_text(),filename=str(f))
    assets=json.loads((ROOT/'verification/asset_verification.json').read_text())
    assert assets['passed'] and assets['manifest_sha256']==digest(ROOT/'shared/reference_assets.json')
    for relative,info in assets['files'].items():
        if relative.startswith('models/LTX2.3/packages/'):
            path=ROOT/'shared/LTX2.3/packages'/relative.removeprefix('models/LTX2.3/packages/')
        else:path=ROOT/'assets'/relative
        stat=path.stat()
        assert (stat.st_size,stat.st_mtime_ns)==(info['bytes'],info['mtime_ns']), 'Asset changed: '+relative
    repo=Path(settings['ltx_repo'])
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==settings['provenance']['ltx_source_revision']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no','--','packages'],text=True).strip()
    paths=[source,*[repo/'packages'/pkg/'src' for pkg in ['ltx-core','ltx-trainer','ltx-pipelines']]]
    sys.path[:0]=list(map(str,paths))
    from ltx_outpaint import inference as inf
    from ltx_outpaint.model import ArchitectureConfig
    import ltx_core
    import torch
    assert Path(inf.__file__).resolve().is_relative_to(source.resolve())
    assert Path(ltx_core.__file__).resolve().is_relative_to(repo.resolve())
    architecture=dict(settings['architecture'])
    architecture['k']=architecture.pop('local_partition_k')
    architecture['scale']=architecture.pop('compression_scale')
    checkpoint=ROOT/'shared/checkpoint-007500/adapter_model.safetensors'
    with checkpoint.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(length))
    metadata=header['__metadata__']
    assert json.loads(metadata['architecture'])==asdict(ArchitectureConfig(**architecture))
    assert json.loads(metadata['provenance'])==settings['provenance']
    assert Path(metadata['base_model_path']).resolve()==Path(settings['model']['base_model_path']).resolve()
    receipt=json.loads((checkpoint.parent/'relocation_receipt.json').read_text())
    assert receipt['tensor_bytes_identical'] and receipt['tensor_layout_identical']
    assert receipt['target_sha256']==digest(checkpoint)
    from ltx_trainer.model_loader import read_video_scale_factors
    from ltx_outpaint.streaming_video import probe_video
    sf=read_video_scale_factors(settings['model']['base_model_path'])
    assert (sf.time,sf.height,sf.width)==(8,32,32)
    meta=probe_video(ROOT/'assets/input/05_anchor_960x540_24fps_1169f.mkv')
    assert (meta.frame_count,meta.fps,meta.width,meta.height)==(1169,24.0,960,540)
    if a.arm=='03_global_boost':
        assert [inf._early_global_boost_scale(i) for i in range(11)]==[1.4]*10+[None]
    if a.arm=='04_native_halo':
        from ltx_outpaint.native_local_attention import NATIVE_LOCAL_HALO_RADIUS
        assert NATIVE_LOCAL_HALO_RADIUS==1
    versions={n:importlib.metadata.version(n) for n in ['torch','transformers','safetensors','av','numpy']}
    report={'passed':True,'arm':a.arm,'spec':spec,'versions':versions,'cuda_build':torch.version.cuda,
            'source_hashes':actual,'runner_sha256':digest(ROOT/'run_arm.py'),'settings_sha256':digest(arm/'settings.json'),
            'cuda_inference_performed':False,'tensor_payload_sha256':receipt['tensor_payload_sha256']}
    (ROOT/'verification'/f'preflight_{a.arm}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PREFLIGHT_PASSED',a.arm,json.dumps(versions),flush=True)

if __name__=='__main__':
    main()
