"""Isolated, resumable three-chunk LongRange experiment; writes only below EXP."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import av, numpy as np, torch
from ltx_outpaint import inference as inf
from ltx_outpaint.streaming_video import SegmentedOutputStore, StreamingAnchorReader, file_sha256, probe_video, session_fingerprint

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/home/work/ai-graphics_33/Hrlrvo/pipeline/data/02_VideoOutPaint_4K_3840x2160/chunk/05')
WINDOWS = {'A': (14, 16), 'B': (21, 23)}

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.partial')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def save_carry(path, carry, index):
    tmp = path.with_name('.' + path.name + '.partial')
    with tmp.open('wb') as f:
        torch.save({'chunk_index': index, 'latent': carry}, f); f.flush(); os.fsync(f.fileno())
    digest = file_sha256(tmp)
    os.replace(tmp, path)
    return digest

def load_carry(path, record):
    if file_sha256(path) != record['latent_carry_sha256']:
        raise RuntimeError('Latent carry checksum mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    carry = payload['latent']
    if payload['chunk_index'] != record['chunk_index'] or carry.device.type != 'cpu' or carry.dtype != torch.bfloat16 or carry.ndim != 5 or not bool(torch.isfinite(carry).all()):
        raise RuntimeError('Latent carry representation mismatch')
    return carry

def seed_prefix(start_chunk, spec, segment_root):
    prior = json.loads((BASE / f'chunk_{start_chunk-1:06d}.json').read_text())
    source = BASE / prior['segment_file']
    segment = np.load(source, mmap_mode='r', allow_pickle=False)
    first = start_chunk * spec.stride_frames
    if prior['new_output_frame_range'][1] != first + spec.prefix_frames or segment.dtype != np.uint8:
        raise RuntimeError('Baseline seed lineage mismatch')
    seed_file = segment_root / 'initial_prefix_rgb24.npy'
    if not seed_file.exists():
        tmp = seed_file.with_name('.' + seed_file.name + '.partial')
        mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(spec.prefix_frames, *segment.shape[1:]))
        mm[:] = segment[-spec.prefix_frames:]; mm.flush(); del mm
        os.replace(tmp, seed_file)
    seed = np.load(seed_file, mmap_mode='r', allow_pickle=False)
    if seed.shape != (spec.prefix_frames, *segment.shape[1:]) or not np.array_equal(seed, segment[-spec.prefix_frames:]):
        raise RuntimeError('Seed copy differs from baseline lossless RGB segment')
    rgb = torch.from_numpy(np.array(seed)).float().div_(255.0).permute(0, 3, 1, 2)
    return rgb, {'source_segment': str(source), 'source_segment_pixel_sha256': prior['segment_pixel_sha256'], 'seed_file_sha256': file_sha256(seed_file), 'global_frame_range': [first, first+spec.prefix_frames], 'limitation': 'Baseline rolling FP32 prefix was overwritten; seed reconstructed from pre-MP4 uint8 RGB24 segment.'}

def encode_window(segment_root, records, destination, fps, width, height, prefix_frames):
    tmp = destination.with_name('.' + destination.stem + '.partial.mp4')
    count = 0
    try:
        with av.open(str(tmp), mode='w') as container:
            stream = container.add_stream('libx264', rate=int(fps)); stream.width = width; stream.height = height; stream.pix_fmt = 'yuv420p'; stream.options = {'crf': '18'}
            for path in [segment_root/'initial_prefix_rgb24.npy', *(segment_root/r['segment_file'] for r in records)]:
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                for rgb in array:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')): container.mux(packet)
                    count += 1
                del array
            for packet in stream.encode(): container.mux(packet)
        if count != prefix_frames + sum(r['output_appended_count'] for r in records):
            raise RuntimeError('Encoded frame count mismatch')
        os.replace(tmp, destination)
    except BaseException:
        tmp.unlink(missing_ok=True); raise
    return count

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--window', choices=WINDOWS, required=True)
    p.add_argument('--rebase-temporal-position', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--use-latent-carry', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--generated-prefix-strength', type=float, default=1.0)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not 0 <= a.generated_prefix_strength <= 1: p.error('strength must be in [0,1]')
    if ROOT.name != 'longrange_fix_test_20260923': raise RuntimeError('Wrong isolated experiment root')
    baseline = json.loads((BASE/'run.json').read_text())
    request = baseline['request']; settings = request['settings']
    spec = inf.LongRangeSpec(**settings['long_range'])
    start_chunk, end_chunk = WINDOWS[a.window]
    output_root = ROOT/'modified_outputs'/f'window_{a.window}'
    output_root.mkdir(parents=True, exist_ok=True)
    segment_root = output_root/'segments'
    anchor_path = Path(request['input_path']); checkpoint = Path(request['checkpoint_argument'])
    prompt, negative_prompt = request['positive_prompt'], request['negative_prompt']
    base_seed, steps = int(request['seed']), int(request['inference_steps'])
    if request['early_global_boost']: raise RuntimeError('Baseline Early Global Boost unexpectedly enabled')
    if settings['runtime']['dtype'] != 'bfloat16': raise RuntimeError('Expected baseline BF16 dtype')
    inf._install_official_sources(Path(settings['ltx_repo']))
    from ltx_trainer.model_loader import read_video_scale_factors
    meta = probe_video(anchor_path)
    if (meta.frame_count, meta.fps, meta.width, meta.height) != (request['input_frame_count'], request['input_fps'], request['input_width'], request['input_height']):
        raise RuntimeError('Anchor metadata changed from baseline')
    vae_path = Path(settings['model'].get('video_vae_path') or settings['model']['base_model_path'])
    sf = read_video_scale_factors(vae_path); spec.validate(sf.time)
    width, height = map(int, settings['video']['canvas'])
    boundary = inf._vae_boundary(height, width, sf.height, sf.width)
    anchor_rect = inf._center_anchor(boundary, meta.height, meta.width)
    records = [r for r in inf._iter_long_range_chunks(meta.frame_count, spec, temporal_factor=sf.time) if start_chunk <= r['chunk_index'] <= end_chunk]
    if len(records) != 3 or any(r['working_frame_count'] != spec.clip_frames for r in records): raise RuntimeError('Three full aligned chunks required')
    config = inf._make_validation_config(settings, prompt, boundary, spec.clip_frames, negative_prompt=negative_prompt)
    config.seed = base_seed; config.inference_steps = steps
    memory = inf._memory_optimization_settings(settings)
    write_json(ROOT/'configs'/'effective_baseline_settings.json', settings)
    identity_request = {'window': a.window, 'chunks': [start_chunk,end_chunk], 'baseline_fingerprint':baseline['request_fingerprint'], 'checkpoint':str(checkpoint), 'seed':base_seed, 'steps':steps, 'prompt':prompt, 'negative_prompt':negative_prompt, 'rebase_temporal_position':a.rebase_temporal_position, 'use_latent_carry':a.use_latent_carry, 'generated_prefix_strength':a.generated_prefix_strength, 'patch_source_sha256':file_sha256(ROOT/'source/ltx_outpaint/inference.py')}
    identity = {'request':identity_request, 'request_fingerprint':session_fingerprint(identity_request)}
    store = SegmentedOutputStore(segment_root); store.prepare(identity, resume=a.resume)
    completed = []
    for expected in records:
        found = store.load_record(expected, output_height=height, output_width=width, chunk_seed=base_seed+expected['absolute_start_frame'])
        if found is None: break
        if a.use_latent_carry: load_carry(segment_root/found['latent_carry_file'], found)
        completed.append(found)
    if any((segment_root/f"chunk_{r['chunk_index']:06d}.json").exists() for r in records[len(completed):]): raise RuntimeError('Resume journal gap')
    if not completed:
        previous_prefix, seed_meta = seed_prefix(start_chunk, spec, segment_root)
        write_json(output_root/'initial_seed.json', seed_meta)
        carry_input = None
    elif len(completed) < len(records):
        latest = completed[-1]
        resume = store.load_resume_state(latest['chunk_index'], prefix_frames=spec.prefix_frames, output_height=height, output_width=width)
        previous_prefix = resume.previous_prefix
        carry_input = load_carry(segment_root/latest['latent_carry_file'], latest) if a.use_latent_carry else None
    else:
        previous_prefix = carry_input = None
    if len(completed) < len(records):
        device_name = str(settings['runtime']['device'])
        if device_name.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        transformer, runner, encoder, adapter_report = inf._load_models(settings, config, boundary, checkpoint_path=checkpoint, memory_optimization=memory)
        store.record_model_ready(adapter_report)
        if completed: inf._restore_post_chunk_rng(resume.cpu_rng_state, resume.cuda_rng_state, resume.cuda_device, expected_device=device_name)
        print(json.dumps({'event':'model_ready','window':a.window,'completed':len(completed),'adapter':adapter_report}), flush=True)
        with StreamingAnchorReader(anchor_path, stride_frames=spec.stride_frames) as reader:
            for record in records[len(completed):]:
                index = int(record['chunk_index']); start = int(record['absolute_start_frame'])
                anchor = reader.read_chunk(record)
                if anchor.shape[0] != spec.clip_frames: raise RuntimeError('Partial selected chunk')
                carried = [] if a.use_latent_carry else None
                metrics = {}
                if device_name.startswith('cuda'): torch.cuda.reset_peak_memory_stats(torch.device(device_name))
                print(json.dumps({'event':'chunk_started','chunk_index':index,'global_start':start,'wall_time':time.time()}), flush=True)
                generated = inf.sample_chunk(transformer=transformer, runner=runner, vae_encoder=encoder, anchor_frames=anchor, previous_prefix=previous_prefix, boundary=boundary, anchor_rect=anchor_rect, absolute_start_frame=start, runtime_metrics=metrics, boundary_blend_width_pixels=inf._inference_boundary_blend_width(settings), allow_nonempty_text=True, vae_decode_streaming=bool(memory['vae_decode_streaming']), rebase_temporal_position=a.rebase_temporal_position, prefix_latent_override=carry_input, generated_prefix_strength=a.generated_prefix_strength, latent_carry_out=carried)
                cpu_rng, cuda_rng, cuda_device = inf._capture_post_chunk_rng(device_name)
                if device_name.startswith('cuda'):
                    metrics['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(torch.device(device_name))
                    metrics['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(torch.device(device_name))
                next_prefix = generated[-spec.prefix_frames:].clone(); emitted = generated[spec.prefix_frames:spec.clip_frames]
                if emitted.shape[0] != record['output_appended_count']: raise RuntimeError('Emission shape mismatch')
                record['runtime'] = metrics; record['emitted_tensor_range']=[spec.prefix_frames,spec.clip_frames]
                if a.use_latent_carry:
                    if carried is None or len(carried)!=1: raise RuntimeError('Expected one clean carry')
                    path = segment_root/f'carry_{index:06d}.pt'
                    record['latent_carry_file']=path.name; record['latent_carry_sha256']=save_carry(path,carried[0],index)
                    record['latent_carry_shape']=list(carried[0].shape); record['latent_carry_dtype']=str(carried[0].dtype)
                    carry_input = carried[0]
                else: carry_input = None
                committed = store.commit_chunk(record, emitted, next_prefix, chunk_seed=base_seed+start, cpu_rng_state=cpu_rng, cuda_rng_state=cuda_rng, cuda_device=cuda_device)
                completed.append(committed); previous_prefix=next_prefix
                del generated, emitted, anchor, carried
                print(json.dumps({'event':'chunk_committed','chunk_index':index,'global_range':record['new_output_frame_range'],'metrics':metrics}),flush=True)
    if len(completed)!=len(records): raise RuntimeError('Window incomplete')
    global_first=records[0]['prefix_output_frame_range'][0]; global_last=records[-1]['new_output_frame_range'][1]-1
    mp4=output_root/f'window_{a.window}_global_{global_first}_{global_last}.mp4'
    if not mp4.is_file():
        count=encode_window(segment_root, completed, mp4, meta.fps, width,height,spec.prefix_frames)
        write_json(output_root/'video_frame_map.json', {'mp4':str(mp4),'mp4_frame_count':count,'global_first_frame':global_first,'global_last_frame_inclusive':global_last,'mapping':'global_frame = mp4_local_frame + global_first_frame'})
    print(json.dumps({'event':'window_complete','window':a.window,'output':str(mp4)}),flush=True)

if __name__=='__main__':
    with torch.inference_mode(): main()
