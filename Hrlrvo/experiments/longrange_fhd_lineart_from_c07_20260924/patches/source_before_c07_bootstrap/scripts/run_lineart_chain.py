"""Independent FHD C0-first LongRange chain. Writes only inside experiment root."""
from __future__ import annotations
import argparse, json, os, sys, time, shutil
import numpy as np
from ltx_outpaint.lineart_global_prior import LineArtGlobalPrior
from ltx_outpaint import global_attention as global_module
from ltx_outpaint import lineart_global_prior as prior_module
from copy import deepcopy
from pathlib import Path
import torch
from ltx_outpaint import inference as inf
from ltx_outpaint.streaming_video import SegmentedOutputStore, StreamingAnchorReader, encode_segments, file_sha256, probe_video, session_fingerprint

ROOT=Path(__file__).resolve().parents[4]
COPY_SRC=ROOT/'source/VideoOutpainting_test/main/src'
ANCHOR=ROOT/'input/05_anchor_960x540_24fps_1169f.mkv'
BASELINE=ROOT/'configs/baseline_05_run.readonly_copy.json'
FRAMES=1169

def atomic_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.partial')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def save_carry(path, tensor, index):
    tmp=path.with_name('.'+path.name+'.partial')
    with tmp.open('wb') as f:
        torch.save({'chunk_index':index,'latent':tensor},f);f.flush();os.fsync(f.fileno())
    digest=file_sha256(tmp);os.replace(tmp,path)
    return digest

def load_carry(path, record, expected_shape):
    if not path.is_file() or file_sha256(path)!=record['carry_sha256']:
        raise RuntimeError('Committed carry missing or corrupt')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    value=payload['latent']
    if payload['chunk_index']!=record['chunk_index'] or value.device.type!='cpu' or value.dtype!=torch.bfloat16 or tuple(value.shape)!=expected_shape or not bool(torch.isfinite(value).all()):
        raise RuntimeError('Resume carry representation mismatch')
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('strength_1p0_global_lineart_prior','lineart_off_control'),required=True)
    p.add_argument('--end-chunk',type=int,choices=(4,19,23),required=True)
    p.add_argument('--rebase-temporal-position',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--use-latent-carry',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--generated-prefix-strength',type=float,default=1.0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--bootstrap-c0',action='store_true')
    p.add_argument('--lineart-global-prior',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--lineart-bias-scale',type=float,default=0.5)
    a=p.parse_args()
    if not 0<=a.generated_prefix_strength<=1:p.error('strength must be in [0,1]')
    if ROOT.name!='longrange_fhd_lineart_global_prior_20260923' or not Path(inf.__file__).resolve().is_relative_to(COPY_SRC):
        raise RuntimeError('Copied experiment source must be imported, never production source')
    print(json.dumps({'event':'import_proof','module_file':str(Path(inf.__file__).resolve()),'pythonpath':os.environ.get('PYTHONPATH'),'cwd':os.getcwd(),'experiment_root':str(ROOT)},sort_keys=True),flush=True)
    for module in (global_module, prior_module):
        if not Path(module.__file__).resolve().is_relative_to(COPY_SRC):
            raise RuntimeError('Line-art source isolation violated')
    if not (a.rebase_temporal_position and a.use_latent_carry and a.generated_prefix_strength == 1.0):
        raise RuntimeError('Line-art experiment must retain strength_1p0 recurrence parameters')
    if a.resume and a.bootstrap_c0:raise RuntimeError('Choose resume OR initial C0 bootstrap')
    if not a.resume and not a.bootstrap_c0:raise RuntimeError('Reuse validated exact C0; do not generate a new branch C0')
    if not ANCHOR.is_file():raise FileNotFoundError(ANCHOR)
    baseline=json.loads(BASELINE.read_text())['request']
    settings=deepcopy(baseline['settings'])
    settings['video']['canvas']=[1920,1080]
    settings['video']['expected_anchor']=[960,540]
    settings['video']['enforce_expected_anchor']=True
    if settings['runtime']['dtype']!='bfloat16' or settings['runtime']['device']!='cuda':raise RuntimeError('Baseline device/dtype changed')
    inf._install_official_sources(Path(settings['ltx_repo']))
    from ltx_trainer.model_loader import read_video_scale_factors
    meta=probe_video(ANCHOR)
    if (meta.frame_count,meta.fps,meta.width,meta.height)!=(FRAMES,24.0,960,540):raise RuntimeError(f'Anchor mismatch: {meta}')
    vae_path=Path(settings['model'].get('video_vae_path') or settings['model']['base_model_path'])
    sf=read_video_scale_factors(vae_path)
    spec=inf.LongRangeSpec(**settings['long_range']);spec.validate(sf.time)
    boundary=inf._vae_boundary(1080,1920,sf.height,sf.width,height_alignment_pixels=128)
    anchor_rect=inf._center_anchor(boundary,meta.height,meta.width)
    if (boundary.encoded_width,boundary.encoded_height,boundary.top,anchor_rect)!=(1920,1152,36,(306,480,846,1440)):
        raise RuntimeError('FHD proxy pad/anchor geometry changed')
    records=[r for r in inf._iter_long_range_chunks(FRAMES,spec,temporal_factor=sf.time) if r['chunk_index']<=a.end_chunk]
    if len(records)!=a.end_chunk+1 or any(r['working_frame_count']!=65 for r in records):raise RuntimeError('Chunk plan mismatch')
    config=inf._make_validation_config(settings,baseline['positive_prompt'],boundary,65,negative_prompt=baseline['negative_prompt'])
    config.seed=int(baseline['seed']);config.inference_steps=int(baseline['inference_steps'])
    checkpoint=Path(baseline['checkpoint_argument'])
    memory=inf._memory_optimization_settings(settings)
    outputs=ROOT/'outputs'/a.mode;segments=ROOT/'chunks'/a.mode;latent_root=ROOT/'latent_states'/a.mode
    outputs.mkdir(parents=True,exist_ok=True);latent_root.mkdir(parents=True,exist_ok=True)
    identity_request={'mode':a.mode,'anchor':str(ANCHOR),'anchor_sha256':file_sha256(ANCHOR),'baseline_checkpoint':str(checkpoint),'baseline_prompt':baseline['positive_prompt'],'baseline_negative_prompt':baseline['negative_prompt'],'base_seed':config.seed,'steps':config.inference_steps,'baseline_fingerprint':json.loads(BASELINE.read_text())['request_fingerprint'],'visible_canvas':[1920,1080],'encoded_canvas':[1920,1152],'rebase_temporal_position':a.rebase_temporal_position,'use_latent_carry':a.use_latent_carry,'generated_prefix_strength':a.generated_prefix_strength,'experiment_source_sha256':file_sha256(Path(inf.__file__))}
    from ltx_core.components.schedulers import LTX2Scheduler
    sigma_by_step=tuple(float(x) for x in LTX2Scheduler().execute(steps=config.inference_steps)[:-1])
    line_config=json.loads((ROOT/'configs/inference_lineart_global.json').read_text())['lineart_global_prior']
    reference_path=(ROOT/line_config['reference_map']).resolve()
    if not reference_path.is_relative_to(ROOT/'reference'):
        raise RuntimeError('Reference map must remain inside this experiment')
    if file_sha256(reference_path)!=line_config['reference_sha256']:
        raise RuntimeError('Selected reference map hash changed')
    line_config.update(enabled=a.lineart_global_prior,bias_scale=a.lineart_bias_scale)
    prior=LineArtGlobalPrior(torch.from_numpy(np.load(reference_path,allow_pickle=False)),sigma_by_step,
                            enabled=a.lineart_global_prior,bias_scale=a.lineart_bias_scale)
    identity_request['lineart_global_prior']={**line_config,'reference_sha256':file_sha256(reference_path),
        'global_attention_sha256':file_sha256(Path(global_module.__file__)),
        'prior_module_sha256':file_sha256(Path(prior_module.__file__))}
    atomic_json(ROOT/'configs'/f'effective_{a.mode}_lineart.json',identity_request['lineart_global_prior'])
    print(json.dumps({'event':'lineart_reference','path':str(reference_path),**identity_request['lineart_global_prior']},sort_keys=True),flush=True)
    store=SegmentedOutputStore(segments)
    store.prepare({'request':identity_request,'request_fingerprint':session_fingerprint(identity_request)},resume=a.resume)
    if a.bootstrap_c0:
        archive=ROOT/'bootstrap/c0'
        verification=json.loads((ROOT/'report/c0_recovery_verification.json').read_text())
        if verification['status']!='PASS':raise RuntimeError('C0 exact replay verification not PASS')
        source_record=json.loads((archive/'chunk_000000.json').read_text())
        for name in (source_record['segment_file'],source_record['next_prefix_file'],'model_ready.json'):
            shutil.copy2(archive/name,segments/name)
        shutil.copy2(archive/source_record['carry_file'],latent_root/source_record['carry_file'])
        preview_dir=outputs/'chunk_previews';preview_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy2(archive/'C00_full65.mp4',preview_dir/'C00_full65.mp4')
        atomic_json(segments/'bootstrap_provenance.json',{'source':str(archive),'verification':verification,
                    'c0_lineart_enabled':False,'same_rgb_prefix_carry_rng':True})
        atomic_json(segments/'chunk_000000.json',source_record)
    atomic_json(ROOT/'configs'/f'effective_{a.mode}_settings.json',settings)
    completed=[]
    carry_shape=(1,128,3,36,60)
    for expected in records:
        actual=store.load_record(expected,output_height=1080,output_width=1920,chunk_seed=config.seed+expected['absolute_start_frame'])
        if actual is None:break
        if a.use_latent_carry:load_carry(latent_root/actual['carry_file'],actual,carry_shape)
        completed.append(actual)
    if any((segments/f"chunk_{r['chunk_index']:06d}.json").exists() for r in records[len(completed):]):raise RuntimeError('Resume journal gap')
    if completed and len(completed)<len(records):
        latest=completed[-1]
        resume=store.load_resume_state(latest['chunk_index'],prefix_frames=17,output_height=1080,output_width=1920)
        previous_prefix=resume.previous_prefix
        carry_input=load_carry(latent_root/latest['carry_file'],latest,carry_shape) if a.use_latent_carry else None
    else:
        previous_prefix=None;carry_input=None
    if len(completed)<len(records):
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        transformer,runner,vae_encoder,adapter_report=inf._load_models(settings,config,boundary,checkpoint_path=checkpoint,memory_optimization=memory)
        store.record_model_ready(adapter_report)
        if completed:inf._restore_post_chunk_rng(resume.cpu_rng_state,resume.cuda_rng_state,resume.cuda_device,expected_device='cuda')
        print(json.dumps({'event':'model_ready','completed':len(completed),'adapter':adapter_report},sort_keys=True),flush=True)
        with StreamingAnchorReader(ANCHOR,stride_frames=48) as reader:
            for record in records[len(completed):]:
                index=int(record['chunk_index']);start=int(record['absolute_start_frame'])
                anchor=reader.read_chunk(record)
                if anchor.shape!=(65,3,540,960):raise RuntimeError('Anchor chunk shape mismatch')
                carried=[] if a.use_latent_carry else None
                metrics={};torch.cuda.reset_peak_memory_stats(torch.device('cuda'))
                print(json.dumps({'event':'chunk_started','chunk_index':index,'absolute_start_frame':start,'time':time.time()}),flush=True)
                with prior.chunk_scope(index,prefix_latent_frames=3):
                    generated=inf.sample_chunk(transformer=transformer,runner=runner,vae_encoder=vae_encoder,anchor_frames=anchor,previous_prefix=previous_prefix,boundary=boundary,anchor_rect=anchor_rect,absolute_start_frame=start,runtime_metrics=metrics,boundary_blend_width_pixels=inf._inference_boundary_blend_width(settings),allow_nonempty_text=True,vae_decode_streaming=bool(memory['vae_decode_streaming']),rebase_temporal_position=a.rebase_temporal_position,prefix_latent_override=carry_input,generated_prefix_strength=a.generated_prefix_strength,latent_carry_out=carried,latent_carry_frames=17 if a.use_latent_carry else None)
                    metrics['lineart_global_prior']=prior.summary()
                if generated.shape!=(65,3,1080,1920) or not bool(torch.isfinite(generated).all()):raise RuntimeError('Visible output invalid')
                if previous_prefix is not None and not torch.equal(generated[:17],previous_prefix):raise RuntimeError('Fixed RGB prefix changed')
                if not torch.equal(generated[:,:,540,960],anchor[:,:,270,480]):raise RuntimeError('Visible center anchor changed')
                cpu_rng,cuda_rng,cuda_device=inf._capture_post_chunk_rng('cuda')
                metrics['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(torch.device('cuda'))
                metrics['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved(torch.device('cuda'))
                next_prefix=generated[-17:].clone()
                offset=0 if index==0 else 17
                emitted=generated[offset:offset+record['output_appended_count']]
                if emitted.shape[0]!=(65 if index==0 else 48):raise RuntimeError('Output emission count mismatch')
                record['runtime']=metrics;record['emitted_tensor_range']=[offset,offset+len(emitted)]
                if a.use_latent_carry:
                    if carried is None or len(carried)!=1 or tuple(carried[0].shape)!=carry_shape:raise RuntimeError('Final clean latent carry invalid')
                    path=latent_root/f'carry_{index:06d}.pt'
                    record['carry_file']=path.name;record['carry_sha256']=save_carry(path,carried[0],index)
                    record['carry_shape']=list(carried[0].shape);record['carry_dtype']=str(carried[0].dtype)
                    carry_input=carried[0]
                else:carry_input=None
                committed=store.commit_chunk(record,emitted,next_prefix,chunk_seed=config.seed+start,cpu_rng_state=cpu_rng,cuda_rng_state=cuda_rng,cuda_device=cuda_device)
                completed.append(committed);previous_prefix=next_prefix
                print(json.dumps({'event':'chunk_committed','chunk_index':index,'range':record['new_output_frame_range'],'metrics':metrics},sort_keys=True),flush=True)
                del generated,emitted,anchor,carried
    if len(completed)!=len(records):raise RuntimeError('Incomplete requested chain')
    expected_frames=65+a.end_chunk*48
    output=outputs/f'C0_C{a.end_chunk}_{expected_frames}f_1920x1080.mp4'
    if not output.is_file():
        encoded=encode_segments(completed,segments,output,fps=24.0,width=1920,height=1080,expected_frames=expected_frames)
        if encoded!=expected_frames:raise RuntimeError('Stitch frame accounting mismatch')
    atomic_json(outputs/f'C0_C{a.end_chunk}_manifest.json',{'mp4':str(output),'frames':expected_frames,'fps':24,'visible_canvas':[1920,1080],'encoded_canvas':[1920,1152],'chunks':list(range(a.end_chunk+1)),'flags':identity_request})
    print(json.dumps({'event':'run_complete','last_chunk':a.end_chunk,'frames':expected_frames,'mp4':str(output)},sort_keys=True),flush=True)

if __name__=='__main__':
    with torch.inference_mode():main()
