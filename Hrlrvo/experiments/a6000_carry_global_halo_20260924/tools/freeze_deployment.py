"""Freeze independent arm code and identical effective settings before transfer."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924'

def main():
    settings = json.loads((ROOT/'shared/effective_strength_1p0_settings.json').read_text())
    settings['ltx_repo'] = REMOTE+'/shared/LTX2.3'
    settings['provenance']['ltx_source_repo'] = settings['ltx_repo']
    models = '/home/oem/hrlrvo/portable_fhd/models/LTX2.3/models'
    settings['model']['base_model_path'] = models+'/LTX-2.3/ltx-2.3-22b-dev.safetensors'
    settings['model']['text_encoder_path'] = models+'/gemma-3-12b'
    settings['model']['adapter_path'] = REMOTE+'/shared/checkpoint-007500/adapter_model.safetensors'
    reference = json.loads((ROOT/'shared/reference_source_hashes.json').read_text())
    allowed = {'01_baseline':set(), '02_no_carry':{'ltx_outpaint/inference.py'},
               '03_global_boost':{'ltx_outpaint/inference.py'},
               '04_native_halo':{'ltx_outpaint/native_local_attention.py'}}
    changes = {}
    for name, permitted in allowed.items():
        arm = ROOT/'arms'/name
        (arm/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
        source = arm/'source/main/src'
        actual = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc'}
        assert actual.keys() == reference.keys(), name
        changed = {k for k in actual if actual[k]!=reference[k]}
        assert changed == permitted, (name,changed,permitted)
        (arm/'source_manifest.json').write_text(json.dumps(actual,indent=2)+'\n')
        changes[name] = sorted(changed)
    report={'source_changes':changes, 'common_settings_identical':True,
            'relocation_only_settings':['ltx_repo','provenance.ltx_source_repo','model.base_model_path','model.text_encoder_path','model.adapter_path'],
            'historical_reference':str(ROOT.parent/'longrange_fhd_recurrence_test_20260923/outputs/strength_1p0'),
            'fresh_C0':True, 'bitwise_cross_GPU_reproducibility':'not assumed; measured after generation'}
    (ROOT/'verification/source_isolation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
