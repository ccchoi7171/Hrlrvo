"""Freeze ONLY A6000 baseline C0 and its exact recurrence/RNG state, CPU-only."""
from pathlib import Path
import hashlib
import json
import os
import shutil

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path('/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/arms/01_baseline')
APPROVED_C0 = {
    'segment_pixel_sha256': 'c427021db8c691806f17d18fa4b2bd89919abd54dbaa3ae161af0c6e67ea2ec6',
    'next_prefix_state_sha256': '1eb955cb01755ae008574fa5a1fbf6b801deb2413b5c5d55799488568d1452e4',
    'carry_sha256': '892a7f5d9a9b500300e6d994682ae670991019e95106d029b9e61d6ebf181d63',
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def checked_name(name):
    if not isinstance(name, str) or Path(name).name != name:
        raise RuntimeError('Unsafe committed filename')
    return name


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Bootstrap is CPU-only')
    torch.set_num_threads(4)
    request = read(BASELINE/'chunks/run.json')['request']
    expected = {'base_seed': 71634910, 'steps': 30, 'generated_prefix_strength': 1.0,
                'rebase_temporal_position': True, 'use_latent_carry': True,
                'visible_canvas': [1920, 1080], 'encoded_canvas': [1920, 1152]}
    for key, value in expected.items():
        if request.get(key) != value:
            raise RuntimeError('Baseline condition mismatch: '+key)
    if read(ROOT/'configs/inference_settings.json') != read(BASELINE/'effective_settings.json'):
        raise RuntimeError('Settings differ from actual A6000 baseline settings')
    record = read(BASELINE/'chunks/chunk_000000.json')
    for key, digest in APPROVED_C0.items():
        if record.get(key) != digest:
            raise RuntimeError('Approved A6000 C0 record changed: '+key)
    if (record['chunk_index'], record['absolute_start_frame'], record['chunk_seed'],
            record['output_appended_count']) != (0, 0, 71634910, 65):
        raise RuntimeError('Wrong source C0')
    segment_path = BASELINE/'chunks'/checked_name(record['segment_file'])
    segment = np.load(segment_path, mmap_mode='r', allow_pickle=False)
    if segment.dtype != np.uint8 or segment.shape != (65, 1080, 1920, 3):
        raise RuntimeError('C0 segment dtype/shape mismatch')
    digest = hashlib.sha256()
    for frame in segment:
        digest.update(frame.tobytes(order='C'))
    if digest.hexdigest() != record['segment_pixel_sha256']:
        raise RuntimeError('C0 lossless pixels corrupt')
    if segment_path.stat().st_size != record['segment_byte_size']:
        raise RuntimeError('C0 segment size mismatch')
    frame_digest = hashlib.sha256(segment[0].tobytes(order='C')).hexdigest()
    del segment
    prefix_path = BASELINE/'chunks'/checked_name(record['next_prefix_file'])
    carry_path = BASELINE/'latent_states'/checked_name(record['carry_file'])
    prefix_recovery = None
    if sha(prefix_path) != record['next_prefix_state_sha256']:
        # Baseline rotates two prefix/RNG slots; C2 replaced C0's slot. Never
        # reconstruct FP32 recurrence state from lossy RGB24 pixels or the carry.
        recovered = ROOT/'recovery/arms/01_baseline'
        replay = read(recovered/'chunks/chunk_000000.json')
        for key, digest in APPROVED_C0.items():
            if replay.get(key) != digest:
                raise RuntimeError('C0 exact replay differs from original A6000 C0: '+key)
        replay_segment = recovered/'chunks'/checked_name(replay['segment_file'])
        if sha(replay_segment) != sha(segment_path):
            raise RuntimeError('Replayed C0 segment bytes differ')
        replay_carry = recovered/'latent_states'/checked_name(replay['carry_file'])
        if sha(replay_carry) != record['carry_sha256']:
            raise RuntimeError('Replayed C0 carry differs')
        prefix_path = recovered/'chunks'/checked_name(replay['next_prefix_file'])
        prefix_recovery = {'reason': 'original two-slot prefix journal replaced C0 with C2',
                           'replayed_in_isolated_original_baseline_source': True,
                           'recovered_prefix_path': str(prefix_path),
                           'all_original_c0_hashes_match': True}
    if sha(prefix_path) != record['next_prefix_state_sha256'] or sha(carry_path) != record['carry_sha256']:
        raise RuntimeError('C0 prefix/RNG/carry checksum mismatch; refusing approximate continuation')
    state = torch.load(prefix_path, map_location='cpu', weights_only=True)
    carry = torch.load(carry_path, map_location='cpu', weights_only=True)
    if state['chunk_index'] != 0 or carry['chunk_index'] != 0:
        raise RuntimeError('C0 state chunk mismatch')
    if state['prefix'].dtype != torch.float32 or tuple(state['prefix'].shape) != (17, 3, 1080, 1920):
        raise RuntimeError('Wrong FP32 prefix')
    if carry['latent'].dtype != torch.bfloat16 or tuple(carry['latent'].shape) != (1, 128, 3, 36, 60):
        raise RuntimeError('Wrong latent carry')
    for key in ['cpu_rng_state', 'cuda_rng_state']:
        value = state.get(key)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise RuntimeError('Missing exact saved RNG: '+key)
    if state['cuda_device'] not in ['cuda', 'cuda:0']:
        raise RuntimeError('C0 RNG belongs to a different logical CUDA device')
    if not torch.isfinite(state['prefix']).all() or not torch.isfinite(carry['latent']).all():
        raise RuntimeError('Nonfinite C0 state')
    del state, carry
    archive = ROOT/'bootstrap/c0'
    archive.mkdir(parents=True, exist_ok=True)
    paths = [segment_path, prefix_path, carry_path, BASELINE/'chunks/chunk_000000.json',
             BASELINE/'chunks/model_ready.json', BASELINE/'outputs/chunk_previews/C00_full65.mp4']
    checks = {}
    for src in paths:
        digest = sha(src)
        dst = archive/src.name
        if dst.exists():
            if sha(dst) != digest:
                raise RuntimeError('Refusing to overwrite a different C0 archive: '+str(dst))
        else:
            partial = dst.with_name('.'+dst.name+'.partial')
            shutil.copy2(src, partial)
            if sha(partial) != digest:
                raise RuntimeError('Copy checksum mismatch')
            os.replace(partial, dst)
        checks[src.name] = digest
    source = Path(request['anchor'])
    target = ROOT/'input/05_anchor_960x540_24fps_1169f.mkv'
    if not target.exists():
        target.symlink_to(source.resolve())
    if target.resolve() != source.resolve() or sha(target) != request['anchor_sha256']:
        raise RuntimeError('Input anchor changed')
    report = {'status': 'PASS', 'source_arm': str(BASELINE), 'source_chunk': 0,
              'source_run_sha256': sha(BASELINE/'chunks/run.json'), 'files_sha256': checks,
              'segment_pixel_sha256': record['segment_pixel_sha256'],
              'frame0_pixel_sha256': frame_digest, 'same_rgb_prefix_carry_rng': True,
              'copied_baseline_chunks': [0], 'next_generated_chunk': 1,
              'prefix_recovery': prefix_recovery, 'approved_original_c0': APPROVED_C0}
    (ROOT/'report/c0_recovery_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    reference = ROOT/'reference/global_depth_strength.npy'
    strength = np.load(reference, allow_pickle=False)
    if strength.dtype != np.float32 or strength.shape != (9, 15) or not np.isfinite(strength).all():
        raise RuntimeError('Depth reference invalid')
    if strength.min() < 0 or strength.max() > 1 or np.ptp(strength) == 0:
        raise RuntimeError('Depth reference empty/constant/outside range')
    config = {'enabled': True, 'source': 'A6000_baseline_C0_frame0',
              'extractor': 'D5_v2_base_518', 'bias_mode': 'token_strength_outer_product',
              'bias_scale': 0.5, 'apply_branch': 'global',
              'schedule': [{'start_step': 0, 'end_step': 9, 'multiplier': 1.0},
                           {'start_step': 10, 'end_step': 19, 'multiplier': 0.6},
                           {'start_step': 20, 'end_step': 29, 'multiplier': 0.2}],
              'step_indexing': 'zero_based', 'reference_map': 'reference/global_depth_strength.npy',
              'reference_sha256': sha(reference), 'reference_extraction_sha256': sha(ROOT/'reference/depth_reference.json'),
              'query_scope': 'new_generated_only', 'keys': 'all_global_video_tokens',
              'known_output_preservation': 'restore original Global output after decompression on anchor/prefix',
              'c0_enabled': False, 'depth_orientation': 'relative_inverse_depth_white_near',
              'mapping': 'percentile_0.5_99.5_normalize_then_zero_pad_y36_then_mean32_then_mean4',
              'lineart_density_divisor_used': False, 'formula': 'B_ij=lambda_step*g_i*g_j',
              'c0_frame0_pixel_sha256': frame_digest,
              'semantic_limit': 'Depth salience outer product, not same-depth similarity or learned depth conditioning'}
    (ROOT/'configs/inference_depth_global.json').write_text(json.dumps({'depth_global_prior': config}, indent=2)+'\n')
    print(json.dumps({'event': 'c0_bootstrap_verified', **report}, indent=2), flush=True)


if __name__ == '__main__':
    main()
