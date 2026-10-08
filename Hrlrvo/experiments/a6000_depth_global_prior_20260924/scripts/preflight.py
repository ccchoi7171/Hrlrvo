"""Fail closed on source, baseline, reference, bootstrap, and sampling drift."""
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Preflight must remain CPU-only')
    source = ROOT/'source/VideoOutpainting_test/main/src'
    expected = read(ROOT/'verification/source_snapshot.json')['sha256']
    actual = {str(p.relative_to(source)): sha(p) for p in source.rglob('*.py')}
    if actual != expected:
        raise RuntimeError('H200 lineart inference module snapshot changed')
    baseline = Path('/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/arms/01_baseline')
    settings = read(ROOT/'configs/inference_settings.json')
    if settings != read(baseline/'effective_settings.json'):
        raise RuntimeError('Settings differ from A6000 baseline')
    manifest = read(baseline/'source_manifest.json')
    differences = [name for name, digest in manifest.items() if actual.get(name) != digest]
    if differences != ['ltx_outpaint/global_attention.py']:
        raise RuntimeError('Baseline source changed outside Global attention: '+str(differences))
    model_root = ROOT/'models/D5_v2_base_518'
    model = read(model_root/'asset_manifest.json')
    if model['revision'] != 'b1958afc87fb45a9e3746cb387596094de553ed8':
        raise RuntimeError('D5 revision changed')
    for name, digest in model['sha256'].items():
        if sha(model_root/name) != digest:
            raise RuntimeError('Depth weights changed: '+name)
    prior = read(ROOT/'configs/inference_depth_global.json')['depth_global_prior']
    reference = ROOT/prior['reference_map']
    if not reference.resolve().is_relative_to(ROOT/'reference') or sha(reference) != prior['reference_sha256']:
        raise RuntimeError('Reference file/hash invalid')
    if sha(ROOT/'reference/depth_reference.json') != prior['reference_extraction_sha256']:
        raise RuntimeError('Depth extraction receipt changed')
    values = np.load(reference, allow_pickle=False)
    if values.shape != (9, 15) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise RuntimeError('Reference shape/dtype invalid')
    if values.min() < 0 or values.max() > 1 or np.ptp(values) == 0:
        raise RuntimeError('Reference must be finite normalized and nonconstant')
    bootstrap = read(ROOT/'report/c0_recovery_verification.json')
    if bootstrap['status'] != 'PASS' or bootstrap['source_arm'] != str(baseline):
        raise RuntimeError('Wrong baseline bootstrap')
    for name, digest in bootstrap['files_sha256'].items():
        if sha(ROOT/'bootstrap/c0'/name) != digest:
            raise RuntimeError('C0 archive changed: '+name)
    if bootstrap['frame0_pixel_sha256'] != prior['c0_frame0_pixel_sha256']:
        raise RuntimeError('Reference C0 and recurrence C0 disagree')
    extraction = read(ROOT/'reference/depth_reference.json')
    if (extraction['source']['baseline_arm'] != str(baseline)
            or extraction['source']['frame_pixel_sha256'] != bootstrap['frame0_pixel_sha256']
            or extraction['source']['segment_pixel_sha256'] != bootstrap['segment_pixel_sha256']):
        raise RuntimeError('Depth reference was not extracted from the exact A6000 C0')
    if extraction['model']['revision'] != model['revision'] or extraction['model']['device'] != 'cpu':
        raise RuntimeError('D5 extraction model/device changed')
    if (extraction['model']['manifest_sha256'] != sha(model_root/'asset_manifest.json')
            or extraction['model']['asset_sha256'] != model['sha256']
            or extraction['model']['model_id'] != model['model_id']):
        raise RuntimeError('D5 extraction receipt refers to different model assets')
    numerics = extraction['numerics']
    if (numerics['actual_input_shape'], numerics['percentile_clip'], numerics['inversion'], numerics['pooling_divisor']) != (
            [1, 3, 518, 924], [0.5, 99.5], False, None):
        raise RuntimeError('D5 preprocessing or mapping changed')
    for name, artifact in extraction['artifacts'].items():
        if sha(ROOT/'reference'/name) != artifact['sha256']:
            raise RuntimeError('Depth artifact changed: '+name)
    sys.path.insert(0, str(ROOT/'scripts'))
    from prepare_depth_reference import pool_global_depth_strength
    normalized = np.load(ROOT/'reference/depth_inverse_normalized.npy', allow_pickle=False)
    if not np.array_equal(values, pool_global_depth_strength(normalized)):
        raise RuntimeError('Depth spatial mapping differs from verified float reference')
    # Import the exact frozen implementation and verify validation values without CUDA.
    sys.path.insert(0, str(source))
    from ltx_outpaint import inference as inf
    from ltx_outpaint.lineart_global_prior import LineArtGlobalPrior
    inf._install_official_sources(Path(settings['ltx_repo']))
    from ltx_core.components.schedulers import LTX2Scheduler
    import torch
    sigmas = tuple(float(x) for x in LTX2Scheduler().execute(steps=30)[:-1])
    controller = LineArtGlobalPrior(torch.from_numpy(values), sigmas, enabled=True, bias_scale=0.5)
    if [controller._lambda_for_step(step) for step in [0, 9, 10, 19, 20, 29]] != [.5, .5, .3, .3, .1, .1]:
        raise RuntimeError('Schedule mismatch')
    sampling = read(ROOT/'configs/baseline_05_run.readonly_copy.json')['request']
    if (sampling['seed'], sampling['inference_steps'], sampling['positive_prompt'], sampling['negative_prompt']) != (
            71634910, 30, '', 'people, person, human, staff, audience, crowd, backstage, seats'):
        raise RuntimeError('Baseline sampling mismatch')
    pinned = settings['sampling']
    if (pinned['cfg_scale'], pinned['stg_scale'], pinned['guidance_rescale'], pinned['stg_blocks']) != (3.0, 1.0, .7, [28]):
        raise RuntimeError('Baseline guidance mismatch')
    if settings['long_range'] != {'clip_frames': 65, 'final_chunk_policy': 'repeat_last', 'prefix_frames': 17, 'stride_frames': 48}:
        raise RuntimeError('Chunk geometry changed')
    report = {'passed': True, 'physical_gpu': 0, 'cuda_inference_performed': False,
              'frozen_module_count': len(actual), 'baseline_modified_modules': differences,
              'new_module': 'ltx_outpaint/lineart_global_prior.py (unchanged H200 implementation)',
              'reference_sha256': sha(reference), 'reference_min_max': [float(values.min()), float(values.max())],
              'seed': 71634910, 'strength': 1.0, 'latent_carry': True, 'temporal_rebase': True,
              'c0_source_arm': str(baseline), 'first_generated_chunk': 1,
              'runner_sha256': sha(ROOT/'run_depth_chain.py'), 'settings_sha256': sha(ROOT/'configs/inference_settings.json')}
    (ROOT/'verification/preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
