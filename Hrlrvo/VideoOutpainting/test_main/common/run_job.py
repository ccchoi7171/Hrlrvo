"""Execute one isolated, fully described GPU experiment with native sampling."""
import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def _validate_reduced_long_control(job, settings, label):
    if job.get('backend') != 'native_portable' or not job.get('name', '').startswith('long_'):
        raise ValueError(f'{label} requires the measured A6000 LongRange backend')
    if job.get('steps') != 30 or not job.get('reuse_identical_cfg'):
        raise ValueError(f'{label} requires 30 steps and exact CFG reuse')
    if job.get('recurrence') != {'reduced_geometry': True}:
        raise ValueError(f'{label} cannot be combined with another recurrence intervention')
    if any(key != 'collect_branch_stats' or value for key, value in job.get('attention', {}).items()):
        raise ValueError(f'{label} requires unmodified attention and disabled branch statistics')
    if job.get('early_global_boost') or job.get('prompt', '') or settings['sampling'].get('negative_prompt', ''):
        raise ValueError(f'{label} requires the measured empty-prompt control')
    if settings['runtime']['dtype'] != 'bfloat16' or not settings['memory_optimization']['video_only_gpu_weights']:
        raise ValueError(f'{label} requires BF16 video-only inference')
    proof = Path(job.get('cfg_equivalence_evidence', 'missing'))
    evidence = json.loads(proof.read_text())
    native_sha = evidence.get('native_output_sha256', '')
    if not (evidence.get('status') == 'PASS' and evidence.get('output_bitwise_equal') is True
            and len(native_sha) == 64 and native_sha == evidence.get('reuse_output_sha256')
            and evidence.get('frames', 0) >= 142 and evidence.get('steps') == 30
            and evidence.get('reused_calls', 0) >= 90):
        raise ValueError(f'{label} requires the completed 142-frame exact CFG control')


def validate_global_memory_job(job, settings):
    spec = job.get('global_c4_memory')
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - {'alpha_total', 'start_step'}:
        raise ValueError('Invalid Global C4 memory configuration')
    if job.get('noise_correlation') is not None:
        raise ValueError('Global C4 memory and noise correlation must be tested separately')
    _validate_reduced_long_control(job, settings, 'Global C4 memory')
    return dict(spec)


def validate_noise_correlation_job(job, settings):
    spec = job.get('noise_correlation')
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {'rho'}:
        raise ValueError('Noise correlation requires an explicit rho')
    rho = spec['rho']
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError('Noise correlation rho must be finite and strictly between zero and one')
    if job.get('global_c4_memory') is not None or job.get('seed') != 33333333:
        raise ValueError('Noise correlation requires the fixed-seed control and no feature memory')
    _validate_reduced_long_control(job, settings, 'Noise correlation')
    return dict(spec)


def validate_late_stg_job(job, settings):
    spec = job.get('late_stg_schedule')
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {'start_step'} or type(spec['start_step']) is not int or spec['start_step'] != 20:
        raise ValueError('Late STG screen requires the preregistered start_step=20')
    backend = job.get('backend')
    name = job.get('name', '')
    long_screen_frames = {
        'long_late_stg0_161f_screen': 161,
        'long_late_stg0_1025f': 1025,
        'long_late_stg0_1025f_h200': 1025,
    }
    is_long_screen = name in long_screen_frames
    if backend not in ('native', 'native_portable'):
        raise ValueError('Late STG screen requires a measured native backend')
    if is_long_screen:
        expected_recurrence = {'reduced_geometry': True}
        if (type(job.get('frames')) is not int or job['frames'] != long_screen_frames[name]
                or job.get('reuse_identical_cfg', False) is not False
                or settings['video']['canvas'] != [1920, 1080]):
            raise ValueError('Long Late STG requires its exact reduced frame count and original CFG calls')
        if name == 'long_late_stg0_1025f' and backend != 'native_portable':
            raise ValueError('Full Long Late STG requires the measured A6000 backend')
        if name == 'long_late_stg0_1025f_h200' and backend != 'native':
            raise ValueError('H200 Full Long Late STG requires the measured native backend')
    elif name.startswith('motion_'):
        expected_recurrence = {} if backend == 'native' else {'reduced_geometry': True}
        if job.get('reuse_identical_cfg') is not True:
            raise ValueError('Motion Late STG screen requires exact CFG reuse')
    else:
        raise ValueError('Late STG screen requires the preregistered motion or LongRange screen')
    if (type(job.get('steps')) is not int or job['steps'] != 30
            or type(job.get('seed')) is not int or job['seed'] != 33333333):
        raise ValueError('Late STG screen requires the fixed seed and 30 steps')
    if job.get('recurrence', {}) != expected_recurrence:
        raise ValueError('Late STG screen cannot change native recurrence')
    if any(key != 'collect_branch_stats' or value for key, value in job.get('attention', {}).items()):
        raise ValueError('Late STG screen requires unmodified attention with branch statistics disabled')
    if (job.get('global_c4_memory') is not None or job.get('noise_correlation') is not None
            or job.get('early_global_boost') or job.get('prompt', '')):
        raise ValueError('Late STG screen must be isolated from other interventions')
    sampling = settings['sampling']
    if (settings['runtime']['dtype'] != 'bfloat16' or sampling.get('negative_prompt', '')
            or sampling['inference_steps'] != 30 or sampling['stg_scale'] != 1.0
            or sampling['cfg_scale'] != 3.0 or sampling['guidance_rescale'] != .7
            or sampling['stg_blocks'] != [28]):
        raise ValueError('Late STG screen requires the measured BF16 guidance settings')
    if not is_long_screen:
        proof = json.loads(Path(job.get('cfg_equivalence_evidence', 'missing')).read_text())
        native_sha = proof.get('native_output_sha256', '')
        if not (proof.get('status') == 'PASS' and proof.get('output_bitwise_equal') is True
                and len(native_sha) == 64 and native_sha == proof.get('reuse_output_sha256')
                and proof.get('frames', 0) >= 142 and proof.get('steps') == 30
                and proof.get('reused_calls', 0) >= 90):
            raise ValueError('Late STG screen requires a completed 142-frame exact CFG control')
        if backend == 'native' and (proof.get('resolution') != settings['video']['canvas']
                                   or proof.get('same_request_except_output_path') is not True
                                   or proof.get('same_22_production_source_hashes') is not True):
            raise ValueError('H200 Late STG screen requires the matching full-resolution native CFG proof')
    return dict(spec)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    job=json.loads(args.job.read_text())
    output=Path(job['output'])
    output.parent.mkdir(parents=True,exist_ok=True)
    status=output.parent/'job_status.json'
    if status.exists() and json.loads(status.read_text()).get('status')=='PASS':
        print('Already complete:',output,flush=True);return
    import yaml
    import torch
    if job.get('memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(float(job['memory_fraction']))
    import ltx_outpaint.inference as inference
    settings=yaml.safe_load(Path(job['config']).read_text())
    memory_spec=validate_global_memory_job(job,settings)
    noise_spec=validate_noise_correlation_job(job,settings)
    late_stg_spec=validate_late_stg_job(job,settings)
    inference._install_official_sources(Path(settings['ltx_repo']))
    from backend import install_block_streaming, install_relocation_checks
    from interventions import install as install_interventions, summarize
    from recurrence import install as install_recurrence
    backend=job.get('backend','cpu_block_streaming')
    if backend!='native':
        install_relocation_checks(inference,
            settings['provenance']['ltx_source_repo'], settings['ltx_repo'],
            original_base_model_path='/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3/models/LTX-2.3/ltx-2.3-22b-dev.safetensors',
            original_base_sha256=settings['provenance']['base_sha256'])
    recurrence_context=install_recurrence(inference,job.get('recurrence',{}))
    cfg_stats=None
    if job.get('reuse_identical_cfg'):
        from cfg_reuse import install as install_cfg_reuse
        import ltx_trainer.validation_runner as validation_runner
        cfg_stats=install_cfg_reuse(validation_runner)
    original_load=inference._load_models
    handles=[]
    memory_state={'handle':None}
    noise_state={'handle':None}
    late_stg_state={'handle':None}
    runtime_contexts=ExitStack()
    peaks={'allocated':0,'reserved':0}
    def capture_peaks():
        peaks['allocated']=max(peaks['allocated'],torch.cuda.max_memory_allocated())
        peaks['reserved']=max(peaks['reserved'],torch.cuda.max_memory_reserved())
    original_chunk=inference.sample_chunk
    def measured_chunk(**kwargs):
        if handles:
            handles[0].absolute_start_frame=kwargs['absolute_start_frame']
        noise=noise_state['handle']
        noise_begun=False
        try:
            if noise is not None:
                noise.begin_chunk(kwargs['absolute_start_frame'])
                noise_begun=True
            memory=memory_state['handle']
            if memory is not None:
                memory.begin_chunk(kwargs['absolute_start_frame'],kwargs['previous_prefix'],
                                   kwargs['boundary'],kwargs['anchor_rect'])
            decoded=original_chunk(**kwargs)
            if memory is not None:
                memory.end_chunk(decoded)
            return decoded
        finally:
            if noise_begun:
                noise.end_chunk()
            capture_peaks()
    inference.sample_chunk=measured_chunk
    def load(settings, *a, **kw):
        if backend in ('native','native_portable'):
            model,runner,encoder,report=original_load(settings,*a,**kw)
            handles.append(install_interventions(model,job.get('attention',{})))
            if memory_spec is not None:
                from global_c4_memory import install as install_memory
                memory_state['handle']=install_memory(
                    model,runner,inference_steps=job['steps'],**memory_spec,
                    provenance={'config_sha256':digest(job['config']),
                                'adapter_sha256':digest(Path(job['checkpoint'])/'adapter_model.safetensors'),
                                'cfg_evidence_sha256':digest(job['cfg_equivalence_evidence'])})
            if noise_spec is not None:
                from ltx_core.components.noisers import GaussianNoiser
                from noise_correlation import install_noise_correlation
                noise_state['handle']=install_noise_correlation(GaussianNoiser,**noise_spec)
            if late_stg_spec is not None:
                import ltx_trainer.validation_runner as validation_runner
                from late_stg_schedule import late_stg_schedule
                late_stg_state['handle']=runtime_contexts.enter_context(late_stg_schedule(
                    runner,validation_runner,enabled=True,inference_steps=job['steps'],
                    expected_stg_scale=settings['sampling']['stg_scale'],**late_stg_spec))
            capture_peaks()
            return model,runner,encoder,report
        cpu=copy.deepcopy(settings)
        cpu['runtime']['device']='cpu'
        # Weight-placement is recorded; sampling/model shape and YAML unchanged.
        cpu['memory_optimization']['video_only_gpu_weights']=False
        model,runner,encoder,report=original_load(cpu,*a,**kw)
        intervention=install_interventions(model,job.get('attention',{}))
        handles.append(intervention)
        handles.append(install_block_streaming(model,'cuda:0',expected_dtype=torch.bfloat16,
                                              video_only_streaming=True))
        print(json.dumps({'event':'streaming_installed',
                          'h2d_weight_bytes':handles[-1].h2d_weight_bytes}),flush=True)
        if next(model.parameters()).device.type!='cuda':
            raise RuntimeError('First parameter must identify the sampling device')
        capture_peaks()
        return model,runner,encoder,report
    inference._load_models=load
    class Progress:
        def __init__(self):self.step=0;self.started=time.monotonic()
        def advance_step(self):
            self.step+=1
            if self.step%5==0:
                print(json.dumps({'event':'denoising_progress','step':self.step,
                    'elapsed_seconds':time.monotonic()-self.started}),flush=True)
    inference._SamplingProgress=Progress
    started=time.monotonic()
    info={'status':'RUNNING','pid':os.getpid(),'job':job,'torch':torch.__version__,
          'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'gpu':torch.cuda.get_device_name(0),'backend':backend,
          'config_sha256':digest(job['config']),
          'experiment_source_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
          'dtype':settings['runtime']['dtype'],
          'settings':settings,
          'source_sha256':{str(p):digest(p) for p in Path(inference.__file__).parent.glob('*.py')}}
    status.write_text(json.dumps(info,indent=2))
    torch.cuda.reset_peak_memory_stats()
    try:
        inference.sample_long_range(settings,Path(job['input']),output,
            checkpoint_path=Path(job['checkpoint']),seed=job.get('seed',33333333),
            prompt=job.get('prompt',''),
            inference_steps=job.get('steps',30),num_frames=job.get('frames'),
            early_global_boost=job.get('early_global_boost',False),
            segment_dir=output.parent/'segments',keep_intermediates=True)
        info['status']='PASS'
        info['output_sha256']=digest(output)
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,avg_frame_rate,nb_frames','-of','json',str(output)],
            capture_output=True,text=True,check=True)
        info['output_metadata']=json.loads(probe.stdout)
        info['branch_stats']=summarize(handles[0])
        info['branch_stats_accumulator_dtype']=getattr(handles[0],'stats_dtype','unknown')
    except BaseException as e:
        info['status']='FAIL';info['error']=str(e)
        info['traceback']=traceback.format_exc()
        raise
    finally:
        # Restore the instance before any reporting operation can fail. The
        # handle keeps its completed activation records after context exit.
        try:
            runtime_contexts.close()
        except BaseException:
            info['status']='FAIL'
            info['restoration_error']=traceback.format_exc()
            status.write_text(json.dumps(info,indent=2))
            raise
        info['runtime_seconds']=time.monotonic()-started
        info['recurrence_events']=recurrence_context['events']
        info['cfg_reuse']=cfg_stats
        if memory_state['handle'] is not None:
            info['global_c4_memory']=memory_state['handle'].summary()
        if noise_state['handle'] is not None:
            info['noise_correlation']=noise_state['handle'].summary
        if late_stg_state['handle'] is not None:
            info['late_stg_schedule']=late_stg_state['handle'].summary()
            if job.get('cfg_equivalence_evidence'):
                info['cfg_equivalence_evidence_sha256']=digest(job['cfg_equivalence_evidence'])
        if handles and job.get('attention',{}).get('native_halo_motion_band'):
            info['motion_band_activation']=handles[0].motion_band
        if handles and job.get('attention',{}).get('native_anchor_bridge_alpha',0)>0:
            info['anchor_bridge_activation']=handles[0].anchor_bridge
        capture_peaks()
        info['peak_cuda_allocated_bytes']=peaks['allocated']
        info['peak_cuda_reserved_bytes']=peaks['reserved']
        status.write_text(json.dumps(info,indent=2))


if __name__=='__main__':main()
