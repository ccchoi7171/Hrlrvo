"""Assemble final evidence only after all requested real CLI jobs complete."""
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting')
OUT = ROOT / 'outputs/final_runtime_validation'
RUN = OUT / 'training_smoke/validated_run'
GROUPS = ['op_lora', 'a_lora', 'g_lora', 'f_lora', 'c2', 'c4_shared', 'r2', 'r4', 'local_fusion', 'global_fusion']

def read(path):
    return json.loads(path.read_text())

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

policy = read(OUT / 'resolved_policy.json')
initial = read(RUN / 'runtime_result_start_000000.json')
resumed = read(RUN / 'runtime_result_start_000005.json')
metrics = [json.loads(line) for line in (RUN / 'train_metrics.jsonl').read_text().splitlines() if line]
assert [x['optimizer_step'] for x in metrics] == list(range(1, 7))
assert (initial['status'], initial['initial_step'], initial['final_step']) == ('COMPLETE', 0, 5)
assert (resumed['status'], resumed['initial_step'], resumed['final_step']) == ('COMPLETE', 5, 6)
assert resumed['runtime_state']['restored_optimizer_state_parameters'] == 1728
assert resumed['runtime_state']['restored_scheduler_state']['last_epoch'] == 5
assert resumed['runtime_state']['restored_scheduler_state']['total_iters'] == 2000
for m in metrics:
    assert m['batches_seen'] == m['samples_seen'] == m['optimizer_step']
    assert all(math.isfinite(m[k]) for k in ('loss', 'sigma', 'grad_norm_global'))
    assert 0 <= m['sigma'] <= 1
    assert all(m[k] == 0 for k in ('nan_gradient_count', 'inf_gradient_count', 'frozen_parameter_grad_violation_count', 'frozen_connector_violation_count', 'nonzero_known_timestep_count'))
    assert m['zero_gradient_trainable_groups'] == []
    assert not m['fusion_observer_errors']
    assert len(m['local_native_ratio_by_block']) == len(m['global_local_ratio_by_block']) == 48
    assert all(m['frozen_representative_unchanged'].values())
    assert m['training_geometry']['latent_shape'] == [1, 128, 9, 68, 120]
    assert m['training_geometry']['anchor_rectangles'] == [[17, 30, 51, 90]]
    assert m['training_geometry']['anchor_latent_hw'] == [[34, 60]]
    assert sorted(m['training_geometry']['empty_connector_keys']) == ['audio_prompt_embeds', 'prompt_attention_mask', 'video_prompt_embeds']
    assert m['training_geometry']['audio_generation'] is False
    for group in GROUPS:
        d = m['gradient_group_details'][group]
        assert d['finite'] and d['nonzero'] and d['parameters_without_grad'] == 0
        assert m['parameter_updates'][group]

checkpoint = RUN / 'checkpoints/checkpoint-000005'
header = read(OUT / 'checkpoint_5_header_audit.json')
assert header['key_count'] == 1728 and header['total_elements'] == 280363008
for name in header['files']:
    assert (checkpoint / name).is_file()
checkpoint_manifest = read(checkpoint / 'outpainting_manifest.json')
source = read(OUT / 'source_change_check.json')
assert source['unexpected_changes'] == source.get('missing_preexisting', []) == []
assert source['pinned_source_packages_status'] == ''
assert source['architecture_config_parity'] == 'PASS'
assert '116 passed' in (OUT / 'cpu_tests.log').read_text()
assert 'All checks passed!' in (OUT / 'ruff.log').read_text()
assert read(OUT / 'runtime_scope_update.json')['requested_baseline_videos'] == 2
jobs = read(OUT / 'execution_retry_state.json')['jobs']
assert all(j['status'] == 'PASS' and j['exit_code'] == 0 for j in jobs)

videos = []
for number, expected in ((1, 94), (2, 72)):
    directory = OUT / f'pretrain_inference/video_{number:02d}'
    m = read(directory / 'inference_manifest.json')
    lineage = read(directory / 'chunk_lineage.json')
    assert m['adapter_load']['mode'] == 'initialized_adapters'
    assert m['architecture_version'] == 8 and m['inference_steps'] == 30
    assert m['chunk_count'] == len(lineage['chunks']) == 2
    assert m['requested_frames'] == m['input_total_frames'] == expected
    assert m['output_rgb_shape'] == [expected, 3, 2160, 3840]
    assert m['output_finite'] and m['prefix_gap_count'] == m['prefix_duplicate_count'] == 0
    assert m['fixed_positive_prompt'] == m['fixed_negative_prompt'] == ''
    assert lineage['chunks'][0]['new_output_frame_range'] == [0, 65]
    assert lineage['chunks'][1]['prefix_output_frame_range'] == [48, 65]
    assert lineage['chunks'][1]['new_output_frame_range'] == [65, expected]
    assert sum(c['output_appended_count'] for c in lineage['chunks']) == expected
    for c in lineage['chunks']:
        r = c['runtime']
        assert r['known_anchor_rect_yxyx'] == [17, 30, 51, 90]
        assert r['denoised_latent_finite'] and r['decoded_chunk_finite']
        assert r['full_generation_latent_shape'] == [1, 128, 9, 68, 120]
    videos.append({'video': number, 'manifest': m, 'lineage': lineage})

ci = read(OUT / 'checkpoint_load_inference/inference_manifest.json')
assert ci['adapter_load']['mode'] == 'checkpoint'
assert ci['adapter_load']['adapter_path'] == str(checkpoint / 'adapter_model.safetensors')
assert ci['architecture_version'] == 8 and ci['inference_steps'] == 30
assert ci['chunk_count'] == 1 and ci['output_rgb_shape'] == [65, 3, 2160, 3840]
assert ci['output_finite'] and ci['prefix_gap_count'] == ci['prefix_duplicate_count'] == 0
assert ci['adapter_load']['expected_key_count'] == ci['adapter_load']['loaded_key_count'] == 1728
assert all(ci['adapter_load'][k] == 0 for k in ('missing_key_count', 'unexpected_key_count', 'shape_mismatch_count'))
media = read(OUT / 'media_validation.json') + read(OUT / 'checkpoint_media_validation.json')
assert len(media) == 3 and all(x['full_decode_exit_code'] == 0 for x in media)
for entry, expected in zip(media, [94, 72, 65], strict=True):
    stream = entry['ffprobe']['streams'][0]
    assert (stream['width'], stream['height'], int(stream['nb_frames'])) == (3840, 2160, expected)

first5 = metrics[:5]
gradients = {g: {'min': min(m[f'grad_norm_{g}'] for m in first5), 'median': statistics.median(m[f'grad_norm_{g}'] for m in first5), 'max': max(m[f'grad_norm_{g}'] for m in first5), 'parameter_count': first5[0]['gradient_group_details'][g]['parameter_count'], 'finite_nonzero_steps': 5, 'update_steps': 5} for g in GROUPS}
counts = initial['parameter_counts']
assert counts['trainable_total'] == 280363008 and counts['frozen_base'] == 18988838144 and counts['total'] == 19269201152
peak_alloc = max(m['peak_allocated_gib'] for m in metrics)
peak_reserved = max(m['peak_reserved_gib'] for m in metrics)
summary = {'timestamp': datetime.now(timezone.utc).isoformat(), 'verdicts': {'training_runtime': 'PASS', 'inference_runtime': 'PASS', 'train_inference_checkpoint_compatibility': 'PASS', 'final': 'PASS — ready for production training'}, 'scope': {'baseline_video_count': 2, 'third_video': 'CANCELLED_USER / excluded from verdict', 'training_steps': 5, 'resume_additional_steps': 1, 'checkpoint_inference_chunks': 1, 'denoising_steps_per_chunk': 30}, 'policy': policy, 'parameter_counts': counts, 'training_steps': metrics, 'gradient_summary_first_five': gradients, 'checkpoint_step5': str(checkpoint), 'resume_runtime_state': resumed['runtime_state'], 'training_peak_allocated_gib': peak_alloc, 'training_peak_reserved_gib': peak_reserved, 'baseline_videos': videos, 'checkpoint_inference': ci, 'media_validation': media, 'source_validation': source, 'cpu_tests_passed': 116, 'resolved_observer_incident': 'RUNTIME_OBSERVER_INCIDENT.md', 'long_production_training_started': False}
(OUT / 'validation_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')

step_table = table(['Step', 'Loss', 'Sigma', 'LR used', 'Forward s', 'Backward+clip s', 'Optimizer s', 'Step s'], [[m['optimizer_step'], f"{m['loss']:.9f}", f"{m['sigma']:.9f}", f"{m['learning_rate']:.10g}", f"{m['forward_seconds']:.3f}", f"{m['backward_and_clip_seconds']:.3f}", f"{m['optimizer_seconds']:.3f}", f"{m['total_step_seconds']:.3f}"] for m in metrics])
grad_table = table(['Group', 'Parameters', 'Grad norm min / median / max (steps1–5)', 'Finite/nonzero', 'Updated'], [[g, f"{d['parameter_count']:,}", ' / '.join(f"{d[k]:.8g}" for k in ('min', 'median', 'max')), '5/5', '5/5'] for g, d in gradients.items()])
fusion_table = table(['Step', 'Local Native mean', 'Local Adaptive mean', 'Global Local mean', 'Global Global mean'], [[m['optimizer_step']] + [f"{m[k]:.10f}" for k in ('local_native_ratio_mean', 'local_adaptive_ratio_mean', 'global_local_ratio_mean', 'global_global_ratio_mean')] for m in metrics])
video_table = table(['Video', 'Input / output frames', 'Chunks × denoising steps', 'Gap / duplicate', 'Wall s', 'Peak allocated / reserved GiB', 'Verdict'], [[v['video'], f"{v['manifest']['input_total_frames']} / {v['manifest']['output_rgb_shape'][0]}", '2 × 30', '0 / 0', f"{v['manifest']['wall_time_seconds']:.3f}", f"{v['manifest']['peak_cuda_allocated_bytes']/2**30:.3f} / {v['manifest']['peak_cuda_reserved_bytes']/2**30:.3f}", 'PASS'] for v in videos])
lineage_table = table(['Video', 'Chunk', 'Anchor input range', 'Previous output prefix', 'New output range', 'Append count', 'Internal repeat-last padding'], [[v['video'], c['chunk_index'], c['anchor_source_frame_range'], c['prefix_output_frame_range'], c['new_output_frame_range'], c['output_appended_count'], c['repeat_last_padding_count']] for v in videos for c in v['lineage']['chunks']])
mem_table = table(['Step', 'After zero_grad allocated GiB', 'Reserved GiB', 'Lifecycle peak allocated GiB', 'Physical free GiB'], [[m['optimizer_step']] + [f"{m[k]:.3f}" for k in ('allocated_gib', 'reserved_gib', 'peak_allocated_gib', 'physical_free_gib')] for m in metrics])
count_table = table(['Category', 'Actual parameters'], [[k, f'{v:,}'] for k, v in counts.items()])
checkpoint_table = table(['Trainable category', 'Saved / loaded keys'], [[k, v] for k, v in ci['adapter_load']['keys_by_category'].items()])
outputs_table = table(['Artifact', 'Absolute path'], [['Training run', str(RUN)], ['Step5 full checkpoint', str(checkpoint)]] + [[f"Baseline video {v['video']}", v['manifest']['output_path']] for v in videos] + [['Checkpoint5 inference', ci['output_path']], ['CLI documentation', str(ROOT / 'docs/FINAL_TRAIN_INFERENCE_CLI.md')]])
report = f'''# Final Hrlrvo v8 runtime validation

Generated: {summary['timestamp']} (UTC).
Source root: `{ROOT}`. Conda: `Hrlrvo`.

## Final verdict and scope

- **TRAINING RUNTIME: PASS**
- **INFERENCE RUNTIME: PASS**
- **TRAIN→INFERENCE CHECKPOINT COMPATIBILITY: PASS**
- **FINAL IMPLEMENTATION VERDICT: PASS — ready for production training**

The user's final scope is **two baseline videos**, not three. Video3 was interrupted by SIGINT only to this validation's own PID983503 after the user's request; its partial console evidence is preserved and excluded from the verdict (`runtime_scope_update.json`). No further third-video generation was run.

Completed exactly five initial optimizer steps, one resumed step5→6, two initialized-adapter baseline videos (two recurrent chunks each, 30 denoising steps per chunk), and one real checkpoint5-loaded 65-frame/30-step chunk. This is execution/compatibility evidence, **not a quality, overfitting, or long-training stability claim**. No production long training was started.

## 1. Hardware and unchanged architecture

Single GPU0 NVIDIA H200, nvidia-smi total143,771 MiB (**140.401 GiB**); idle before each job, used0 MiB, free143,156 MiB (**139.801 GiB**). CUDA API reports usable total139.801 GiB. MIG disabled; no competing GPU compute processes at launch. Jobs ran sequentially. Other workloads/cache supervisors were not killed or reconfigured. Hardware and package captures: `hardware_before.txt`, `environment.txt`; each launcher also records pre/post GPU state and a low-frequency GPU trace.

Actual v8 48-block installed model, no dummy, no frame/resolution/rank reduction, no quantization or optimizer-state offload introduced. Native/Adaptive/Global hierarchy, shared full-cell C4, PMA, Target-only write, shared O/CA/FFN are unchanged. Core architecture/data/cache source hashes match `source_before.json`. Pinned LTX packages remain clean at `{source['pinned_revision']}`.

{count_table}

## 2. Exact production policy and loss

- Micro batch1 × gradient accumulation1 × world size1 = effective batch1.
- **Checkpoint every500 optimizer steps = 500×1×1×1 =500 processed samples.** Accumulation was not modified to match this number. Periodic500 execution was not itself run; native-loop unit/static evidence verifies the policy, while real forced checkpoints at5/6 exercise the same writer.
- Native torch AdamW; exactly one optimizer group containing280,363,008 trainable elements. Base LR **1e-4** for OP/A/G/F/C2/sharedC4/R2/R4/LocalFusion/GlobalFusion. No multiplier.
- Native LinearLR: start_factor1, end_factor0.1, horizon **2,000 optimizer steps**, unchanged by the smoke stop. After step k, next LR=`1e-4*(1-0.9*k/2000)`.
- Max grad norm1; BF16 production compute/frozen base, FP32 trainable master parameters/checkpoint; gradient checkpointingON; native SDPA backend priority CUDNN_ATTENTION→FLASH_ATTENTION→EFFICIENT_ATTENTION→MATH. No optimizer-state offload or8bit substitution.
- FIRST RGB[0:65) ↔ latent[0:9), real GT `[1,128,9,68,120]`; independent Anchor `[1,128,9,34,60]` placed y[17:51),x[30:90). All30,024 Anchor caches were READY at startup; these sampled training records used cache HITs, not synthetic tensors or new encodes.
- Prefix17 RGB/3latent, probability0.5. `known = spatial_anchor OR temporal_prefix`; known values clean, per-token timestep0, loss excluded. Step1 and resumed6 exercised prefix-conditioned batches; steps2–5 prefix-free.
- `OutpaintingStrategy.prepare_training_inputs()` retains native flow target `noise-clean_latent`; inherited `FlexibleStrategy.compute_loss()` uses MSE only on `NOT known`, normalized by active elements (`mean(mask*error²)/mean(mask)`), not the full canvas count. No branch-specific auxiliary GT loss.
- Fixed-empty positive/negative text; runtime condition keys include `video_prompt_embeds`, `audio_prompt_embeds`, `prompt_attention_mask`. Audio generationFalse. Frozen Text CA remains active; every step reaches the upstream trainables.

## 3. Training results

{step_table}

Steps1–5 are the requested smoke; step6 is a separate-process resume. Each step has finite sigma/loss/gradients, full canonical geometry, zero nonzero-known timestep entries. Different real records/noise/sigma were sampled, so this loss sequence is **not interpreted as a monotonic learning trend**. Record IDs and all tensor/known counts are in `training_smoke/validated_run/train_metrics.jsonl`.

Timing columns separate native forward, backward+clipping, optimizer update; the step duration includes synchronous health bookkeeping but not initial cache inventory/model startup. Initial process elapsed{jobs[0]['elapsed_seconds']:.3f}s, resumed process elapsed{jobs[1]['elapsed_seconds']:.3f}s; read-only provenance inventory takes substantial startup time.

### Gradient ownership and actual updates

{grad_table}

All1,728 trainable tensors have gradients; every group is finite/nonzero in all five steps and the resumed sixth. First-backward LoRA A=zero is compatible with zero-B initialization; group/B gradients are nonzero. NaN gradient count0, Inf count0, missing-gradient trainable tensors0, unclassified trainables0. Optimizer membership is exactly the unique requires_grad set, no duplicate parameter identity and no frozen parameter included.

Frozen transformer gradient violations **0** every step; frozen connector violations **0**. Tiny deterministic representative probes from base SA-Q, shared O, FFN and Text CA are unchanged after every optimizer step. This is a full frozen-gradient/membership audit plus representative numerical immutability probes, **not a full19B-parameter before/after checksum**. Trainable update evidence samples a nonzero-gradient location in every category; it does not claim every scalar changes each step.

### Actual fusion ratios (not gradients)

{fusion_table}

Gate is the unchanged `sigmoid(beta+gamma*vgate_msa)`. Local ratios are observed only at actual resolved-T write sites, including overlap-window visits; outside T, Native passes through and is not mislabeled as a50:50 fusion sample. Global ratios cover the canonical full-grid gate. Every row contains48 block means and spatial/channel extrema; recomputation hooks are excluded. No observer errors/nonfinite ratios. Initial ratios are exactly0.5.

At step5, Local Native min/max={first5[-1]['local_native_ratio_min']:.9f}/{first5[-1]['local_native_ratio_max']:.9f}; Global Local min/max={first5[-1]['global_local_ratio_min']:.9f}/{first5[-1]['global_local_ratio_max']:.9f}. Complement ratios are1-alpha. Fusion gradients and alphas are logged separately.

### Memory

{mem_table}

Lifecycle peak including resume: **allocated{peak_alloc:.3f} GiB**, **reserved{peak_reserved:.3f} GiB**. Initial five-step peak allocated{initial['peak_allocated_gib']:.3f} GiB. NoOOM. Reserved memory includes reusable allocator cache, not just live tensors; after zero_grad live allocation returns to~42.323 GiB. Nevertheless nvidia-smi/free memory is tight while that cache remains reserved: use an otherwise idle H200 and do not co-run precache/inference/training. This short test is not a guarantee against all future long-run memory behavior.

## 4. Step5 save and real Resume5→6

Checkpoint: `{checkpoint}`.

Files: `adapter_model.safetensors` (FP32 trainables only), `optimizer.bin`, `scheduler.bin`, `random_states_0.pkl`, `outpainting_manifest.json`. Frozen19B base is **not copied**. Safetensors1,121,680,096 bytes; Adam state2,244,368,675 bytes. Metadata includes v8architecture, base provenance, config fingerprint, READY-before-MISSING data order and counters.

{checkpoint_table}

Saved/loaded adapter tensors **1,728** / trainable elements **280,363,008**. Real resume restored1,728 optimizer state entries; scheduler `last_epoch=5`, `_step_count=6`, horizon2000; Accelerator step5, samples5. Next real step is6, samples6, LR{metrics[5]['learning_rate']:.10g} (exact continuation of step5 nextLR); after6 nextLR{metrics[5]['next_learning_rate']:.10g}. Optimizer groups, RNG restore and counters are captured in `runtime_result_start_000005.json` and console logs. A separate checkpoint6 was saved at the bounded stop; checkpoint5 remained unchanged for inference.

Native generic startup wording is not a reset signal; actual restored state/counters and metrics are the authority. Changed config/provenance/architecture/order fails closed. The500-step production interval is separate from explicit smoke `--stop-after-step5/6`.

## 5. Two real initialized-adapter recurrent inference baselines

Inputs are actual1920×1080 Anchor-only MP4s at24fps. No GT outer is used as conditioning. Both outputs are3840×2160/24fps. Empty text, native30-step CFG/STG/Euler pipeline and all48 v8blocks are retained. Full input lengths were processed; final partial chunks use the existing `repeat_last` internal policy, then discard padding on append.

{video_table}

{lineage_table}

Frame ranges are zero-based half-open. Recurrent prefix is the previous generated full-canvas output's last17RGBframes; source Anchor is retained, prefix remains clean/known, and only new frames are appended. First prefix is absent; second prefix[48:65) precedes new frames[65:...). Gap0, duplicate0. Internal extension to65frames is explicitly logged and does not create extra saved frames.

Every chunk reports full latent `[1,128,9,68,120]`, known mask `[1,9,68,120]`, Anchor rectangle `[17,30,51,90]`, finite denoised latent and decoded RGB. Known-mask count is18,360 tokens prefix-free and36,720 with3latent full-canvas prefix. Metadata/lineage verification plus CPU production geometry/noiser regressions support the conditioning contract; no subjective quality verdict is assigned.

All saved MP4s have been fully decoded using CPU ffmpeg with exit0, and ffprobe confirms94/72frames. See `media_validation.json`, per-video `inference_manifest.json`, `chunk_lineage.json`, console/launcher logs. Third-video partial evidence is deliberately excluded by user scope.

## 6. Training checkpoint → actual inference

The production inference entrypoint loaded **checkpoint5**, not checkpoint6 or fresh adapters. `inference_model_ready` and final manifest agree: expected1,728, loaded1,728, missing0, unexpected0, shape mismatch0, architecture8. Each adapter/C/R/fusion category is covered by the table above; strict loading is not a bare `torch.load()` probe.

Actual result: first Anchorvideo, **one65-frame full4K chunk ×30denoising steps**, finite latent/RGB,3840×2160at24fps; full CPU MP4decodePASS. Wall time **{ci['wall_time_seconds']:.3f}s**; peak allocated **{ci['peak_cuda_allocated_bytes']/2**30:.3f} GiB**, reserved **{ci['peak_cuda_reserved_bytes']/2**30:.3f} GiB**. Output: `{ci['output_path']}`. Evidence: finalmanifest, lineage, console, `checkpoint_media_validation.json`. This proves runtime compatibility, not trained quality after five updates.

## 7. Shared call path / changes / regression checks

TRAIN: `python -m ltx_outpaint.train <config>` → `load_training_config` → `OutpaintingTrainer`/native `LtxvTrainer.train` → `OutpaintingTrainer._training_step`/`LazyAnchorCache.complete_batch` → native embeddings processor (both streams) → `OutpaintingStrategy.prepare_training_inputs` → installed LTX transformer/common `MultiViewSelfAttention.forward` → native final masked-MSE → native backward/clip/AdamW/LinearLR.

INFERENCE: `python -m ltx_outpaint.inference --config...` → `sample_long_range`/`_load_models` → same `install_outpainting` plus strict `load_adapter` → `sample_chunk`/`_condition_latents`/`_known_mask` → native `ValidationRunner` denoiser → same transformer/common blocks → scheduler/known reinjection → native VAEdecode → prefix/new-frame assembly.

No model/attention/LoRA/geometry/PMA/C-R/fusion/data/cache source was changed. Deliberate runtime-only changes:
- `main/src/ltx_outpaint/train.py`: CLI/output/resume integration, config fingerprint/counters, observer installation, preserve scheduler kwargs against native pop mutation.
- New `training_runtime.py`, `health_logging.py`: fail-safe buffered metrics/console, on-device grouped norms/actual gate observations, tiny representative update probes, strict bounded stop after a completed optimizer update.
- `main/src/ltx_outpaint/inference.py`: explicit initialized-vs-checkpoint CLI, strict key report, manifests/finite/lineage observations. No duplicate denoising/model implementation.
- `main/configs/train_4k.yaml`: requested500-step interval and canonical production output root. `inference_4k.yaml`: corresponding future production adapter path; validated run uses explicit checkpoint override.
- New scoped runtime/health/inference CLI regression tests; serial validation orchestrator under `tools/final_runtime_validation/`; CLI documentation.

**116 CPU tests PASS**, RuffPASS, source compile/import/config parsePASS; architecture config parityPASS. Missing original files0, unexpected original-file modifications0; pinned `packages/` git statusclean. Snapshot/hash evidence: `source_before.json`, `source_change_check.json`; logs `cpu_tests.log`, `ruff.log`, `final_static_validation.log`. No global project git repository is available, so the per-file baseline hash audit is the change evidence.

## 8. Resolved incident and remaining limits

The first attempt failed **before any forward/backward/optimizer step** due solely to new observer FP32linspace endpoint rounding when sampling a67,108,864-element frozen FFNweight. The observer was corrected to integer index arithmetic and safe error telemetry; exact regression added. All failed artifacts are preserved outside `validated_run`; see `RUNTIME_OBSERVER_INCIDENT.md`. The successful5+1real steps above used the corrected observer. No architecture/config reduction hid the failure.

No unresolved runtime blocker in the requested two-video/5+1-step scope. Limits: no long-run training/tiny-overfit/quality conclusion; sampled frozen-value probes rather than full-base checksum; complete30024-record provenance inventory is an intentional expensive startup phase;500th periodic step not executed; all smoke records were AnchorHIT (no newMISS stress run); reservedGPUmemory precludes concurrent large workloads. No requirement is silently represented as a stronger test.

## 9. Output and CLI paths

{outputs_table}

Machine-readable aggregate: `{OUT / 'validation_summary.json'}`.
Independent read-only review: **PASS**, `independent_verification.md`. Final `hardware_after.txt` confirms no GPU compute process remains (used0 MiB/free143,156 MiB).
Exact reproducible job commands: `execution_retry_state.json` and original baseline logs/state. New/Resume/Pretraining/Checkpoint inference commands, conda/tmux and tails: `docs/FINAL_TRAIN_INFERENCE_CLI.md`. The only manual change for Resume/Inference is an existing complete checkpoint path; no future checkpoint is claimed to exist now.
'''
(OUT / 'FINAL_RUNTIME_VALIDATION_REPORT.md').write_text(report)
print(json.dumps({'verdict': summary['verdicts'], 'report': str(OUT / 'FINAL_RUNTIME_VALIDATION_REPORT.md'), 'train_peak_allocated_gib': peak_alloc, 'train_peak_reserved_gib': peak_reserved, 'checkpoint_inference_wall_seconds': ci['wall_time_seconds']}, indent=2))
