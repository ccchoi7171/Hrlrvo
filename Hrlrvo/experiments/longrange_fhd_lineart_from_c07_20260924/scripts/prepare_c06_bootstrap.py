"""CPU-only baseline C0-C6 archive with exact carry and explicit RGB quantization."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from ltx_outpaint.streaming_video import _write_prefix_state, file_sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'longrange_fhd_recurrence_test_20260923'


def main():
    archive = ROOT / 'bootstrap/c0_c06'
    archive.mkdir(parents=True, exist_ok=True)
    if any(archive.iterdir()):
        raise RuntimeError('Bootstrap archive already contains files; never overwrite')
    segment_root = BASE / 'chunks/strength_1p0'
    latent_root = BASE / 'latent_states/strength_1p0'
    records = []
    sha_by_file = {}
    for index in range(7):
        record_path = segment_root / f'chunk_{index:06d}.json'
        record = json.loads(record_path.read_text())
        assert record['chunk_index'] == index
        assert record['chunk_seed'] == 71634910 + index * 48
        assert record['output_appended_count'] == (65 if index == 0 else 48)
        source = segment_root / record['segment_file']
        array = np.load(source, mmap_mode='r', allow_pickle=False)
        assert array.shape == (record['output_appended_count'], 1080, 1920, 3)
        assert array.dtype == np.uint8
        digest = hashlib.sha256()
        for frame in array:
            digest.update(frame.tobytes(order='C'))
        assert digest.hexdigest() == record['segment_pixel_sha256']
        del array
        carry_path = latent_root / record['carry_file']
        assert file_sha256(carry_path) == record['carry_sha256']
        payload = torch.load(carry_path, map_location='cpu', weights_only=True)
        carry = payload['latent']
        assert payload['chunk_index'] == index
        assert carry.shape == (1, 128, 3, 36, 60) and carry.dtype == torch.bfloat16
        assert carry.device.type == 'cpu' and bool(torch.isfinite(carry).all())
        for path in (source, carry_path):
            target = archive / path.name
            shutil.copy2(path, target)
            digest = file_sha256(target)
            assert digest == file_sha256(path)
            sha_by_file[target.name] = digest
        records.append(record)

    # The default generators are untouched by this deterministic VAE/forward
    # path. GaussianNoiser uses its own per-chunk-seeded CUDA generator.
    states = torch.load(ROOT / 'bootstrap/rng_reference_states.pt',
                        map_location='cpu', weights_only=True)
    for index in (22, 23):
        for key in ('cpu_rng_state', 'cuda_rng_state'):
            assert torch.equal(states[0][key], states[index][key])
        assert states[0]['cuda_device'] == states[index]['cuda_device']
    state = states[0]
    last_record = records[-1]
    array = np.load(archive / last_record['segment_file'], mmap_mode='r', allow_pickle=False)
    prefix = torch.from_numpy(np.array(array[-17:], copy=True)).permute(0, 3, 1, 2).float().div_(255)
    assert prefix.shape == (17, 3, 1080, 1920)
    assert bool(torch.isfinite(prefix).all()) and 0 <= float(prefix.min()) <= float(prefix.max()) <= 1
    assert torch.equal(prefix.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1),
                       torch.from_numpy(np.array(array[-17:], copy=True)))
    del array
    original_prefix_hash = last_record['next_prefix_state_sha256']
    prefix_path = archive / 'previous_prefix_0.pt'
    size, digest = _write_prefix_state(prefix_path, 6, prefix,
        cpu_rng_state=state['cpu_rng_state'], cuda_rng_state=state['cuda_rng_state'],
        cuda_device=state['cuda_device'])
    last_record.update(next_prefix_file=prefix_path.name,
                       next_prefix_byte_size=size, next_prefix_state_sha256=digest,
                       bootstrap_prefix_reconstructed_from='segment_000006.npy:last17:uint8/255',
                       original_next_prefix_state_sha256=original_prefix_hash,
                       bootstrap_fp32_prefix_bit_exact=False)
    sha_by_file[prefix_path.name] = digest
    for record in records:
        path = archive / f"chunk_{record['chunk_index']:06d}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        sha_by_file[path.name] = file_sha256(path)
    shutil.copy2(segment_root / 'model_ready.json', archive / 'model_ready.json')
    sha_by_file['model_ready.json'] = file_sha256(archive / 'model_ready.json')
    provenance = {
        'status': 'PASS', 'bootstrap_through': 6,
        'source_root': str(BASE), 'source_mode': 'strength_1p0',
        'source_run_sha256': file_sha256(segment_root / 'run.json'),
        'reused_chunks': list(range(7)), 'reused_output_frames': 353,
        'first_new_chunk': 7, 'first_new_output_frame': 353,
        'rgb_segments_bit_exact': True, 'latent_carry_exact': True,
        'rgb_prefix_recovery': 'uint8_lossless_segment_tail_div255',
        'bit_exact_fp32_prefix': False,
        'rgb_error_bound_normalized': 1 / 255,
        'rgb_error_note': 'Bound from stored truncation x*255->uint8; float32 arithmetic tolerance applies. Not an original FP32 resume.',
        'original_c06_prefix_state_sha256': original_prefix_hash,
        'reconstructed_c06_prefix_state_sha256': digest,
        'rng_recovery': 'C0/C22/C23 CPU and CUDA states byte-identical; default RNG unadvanced in deterministic forward',
        'rng_state_reference_chunks': [0, 22, 23],
        'file_sha256': sha_by_file,
    }
    (archive / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    (ROOT / 'report/c06_bootstrap_validation.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in provenance.items() if k != 'file_sha256'}, indent=2))


if __name__ == '__main__':
    with torch.inference_mode():
        main()
