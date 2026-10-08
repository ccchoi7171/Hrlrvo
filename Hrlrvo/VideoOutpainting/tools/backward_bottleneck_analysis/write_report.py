"""Build the final human report from completed diagnostic artifacts (CPU only)."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'outputs/backward_bottleneck_analysis/20260911_1410'
PREFIX = 'outputs/backward_bottleneck_analysis/20260911_1410'


def read(name):
    return json.loads((A / name).read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def phases(j):
    return {x['stage']: x['seconds'] for x in j['phases']}


def step_sum(j):
    return sum(v for k, v in phases(j).items() if k not in ('cache_contract_setup', 'model_load', 'optimizer_setup', 'gradient_audit_OUTSIDE_TIMING_AND_PROFILER'))


base = read('full_baseline/result.json')
new = read('full_ncdhw/result.json')
bench = read('full_benchmark/result.json')
warm = read('benchmark_warm_training/warm_step_2/result.json')
assert warm['status'] == 'PASS'
if bench['status'] == 'RUNNING':
    raise SystemExit('Fresh benchmark run is not complete; do not finalize the report yet.')
bs, ns = read('full_baseline/scopes.json'), read('full_ncdhw/scopes.json')
blocks = read('block_table.json')
replacements = {}
labels = {
    'real_CPU_data_load': 'CPU real-cache read + collate',
    'anchor_cache_completion': 'Anchor cache HIT / completion',
    'H2D': 'H2D', 'empty_connector': 'Empty connector (cold)',
    'conditioning_noise_mask': 'Condition / known mask / noise / sigma',
    'forward': 'Model forward', 'loss': 'Loss construction',
    'backward_only': '**loss.backward() ONLY**',
    'clip_grad_norm_only': 'clip_grad_norm_ ONLY (max=1)',
    'optimizer_step': 'AdamW.step (state materialized)',
    'zero_grad': 'zero_grad', 'scheduler_step': 'LinearLR.step',
}
replacements['TIMING'] = table(['Phase', 'Original seconds', 'R2/R4 NCDHW-only seconds', 'Fresh benchmark=True seconds'], [[v, f'{phases(base)[k]:.6f}', f'{phases(new)[k]:.6f}', f'{phases(bench)[k]:.6f}' if k in phases(bench) else 'not completed'] for k, v in labels.items()] + [['**Measured phase sum**', f'**{step_sum(base):.6f}**', f'**{step_sum(new):.6f}**', f'**{step_sum(bench):.6f}**' if bench['status']=='PASS' else bench['status']]])
mem_rows=[]
for label, j in [('Original', base), ('R2/R4 NCDHW-only', new), ('Fresh benchmark=True', bench), ('Autotuning warm optimizer step2', warm)]:
    m = j.get('end', j.get('failure_memory', {}))
    mem_rows.append([label, j['status'], f"{m.get('peak_allocated_gib', 0):.3f}", f"{m.get('peak_reserved_gib', 0):.3f}", f"{m.get('free_gib', 0):.3f}", j.get('loss', ['N/A'])[0], j.get('global_grad_norm_before_clip', 'N/A')])
replacements['FULL_COMPARE']=table(['Experiment','Status','Peak allocated GiB','Peak reserved GiB','End physical free GiB','Loss','Pre-clip global grad norm'],mem_rows)
replacements['BENCH_RESULT'] = f"Fresh full-step result: **{bench['status']}**; forward **{phases(bench).get('forward', 0):.3f} s**, backward-only **{phases(bench).get('backward_only', 0):.3f} s**, completed phase sum **{step_sum(bench):.3f} s**." if bench['status']=='PASS' else f"Fresh full-step result: **{bench['status']}**. Error: `{bench.get('error')}`. Last completed phase: `{bench.get('phases', [{}])[-1].get('stage')}`. Do not extrapolate the fast warm microbenchmark to full training."
replacements['BLOCKS'] = table(['Block','K / windows / nonempty A','Forward s','Backward interval s','Replay s / count','NCDHW backward s','Autotune warm-step2 backward s'], [[b['block'], f"{b['K']} / {b['K']**2} / {12 if b['K']==4 else 24}", f"{b['forward']['Shared/block']:.4f}", f"{b['backward_s']:.4f}", f"{b['backward_only']['Shared/block']:.4f} / 1", f"{ns['block_backward'][str(b['block'])]['seconds']:.4f}", f"{read('benchmark_warm_training/warm_step_2/scopes.json')['block_backward'][str(b['block'])]['seconds']:.4f}"] for b in blocks])
subscope_keys=['Native/branch','Adaptive/branch','Global/branch','Adaptive/C2','Adaptive/C4','Adaptive/R2','Global/R4','LocalFusion/LocalFusion','GlobalFusion/GlobalFusion','TextCA/TextCA','FFN/FFN']
replacements['BLOCK_FORWARD_DETAIL']=table(['Block','Native','Adaptive incl.','Global incl.','A-C2','A-C4','A-R2','G-R4','Local gate','L-G gate','Text CA','FFN'], [[b['block']]+[f"{b['forward'].get(k,0):.4f}" for k in subscope_keys] for b in blocks])
aggregates=defaultdict(lambda:defaultdict(float))
for key, v in bs['details'].items():
    fields=key.split('/')
    aggregates['/'.join(fields[3:])][fields[1]]+=v['gpu_seconds']
replacements['BRANCH_FWD']=table(['Scope (inclusive; do not sum parent+child)','Forward s /48 blocks','Backward-phase forward replay s'], [[k,f"{v.get('forward',0):.6f}",f"{v.get('backward_only',0):.6f}"] for k,v in aggregates.items()])
atr=[]
for k in ('k4_off','k5_off'):
    j=read(f'block_replay/{k}/profiler_attribution.json')['scopes']
    attr=defaultdict(float)
    for key,v in j.items():
        if key.startswith('backward/'):
            scope='/'.join(key.split('/')[4:]) if '/HR/' in key else 'UNATTRIBUTED'
            attr[scope]+=v['self_cuda_us']/1e6
    atr.append(attr)
replacements['BRANCH_BWD_WORK']=table(['Nearest backward forward-source scope','K4 self CUDA work s','K5 self CUDA work s'], [[k, f'{atr[0].get(k,0):.6f}',f'{atr[1].get(k,0):.6f}'] for k in sorted(set(atr[0])|set(atr[1]),key=lambda k:atr[0].get(k,0),reverse=True)])
groups=['op_lora','a_lora','g_lora','f_lora','c2','c4_shared','r2','r4','local_fusion','global_fusion']
grad_rows=[]
for key in groups:
    g=base['gradient_health']['groups'][key]
    h=new['gradient_health']['groups'][key]
    grad_rows.append([key,f"{g['parameter_count']:,}",g['tensor_count'],f"{g['norm']:.9f}",f"{h['norm']:.9f}",g['missing_grad_tensors'],g['nonfinite_elements'],'yes' if g['norm']>0 else 'no'])
replacements['GRADS']=table(['Group','Parameters','Tensors with grad','Original norm','NCDHW norm','Missing grad tensors','NaN/Inf elements','Nonzero'],grad_rows)
group_suffix = {'op_lora':'LoRA_OP','a_lora':'LoRA_A','g_lora':'LoRA_G','f_lora':'LoRA_F','c2':'C2','c4_shared':'C4','r2':'R2','r4':'R4','local_fusion':'LocalFusion','global_fusion':'GlobalFusion'}
group_work=[]
for group in groups:
    costs=[sum(v for scope,v in a.items() if scope.split('/')[-1]==group_suffix[group]) for a in atr]
    count=base['gradient_health']['groups'][group]['parameter_count']
    per_block_million=count/48/1e6
    group_work.append([group,f'{count/280363008*100:.3f}%',f'{costs[0]:.6f}',f'{costs[1]:.6f}',f'{costs[0]*1000/per_block_million:.3f}'])
replacements['GROUP_COST']=table(['Group','Share of all trainable parameters','K4 backward CUDA work s','K5 backward CUDA work s','K4 CUDA work ms / million per-block parameters*'],group_work)

ops=[x for x in read('block_replay/k4_off/profiler_ops.json') if x['device_type']=='DeviceType.CPU' and x['name'].startswith('aten::')]
ops.sort(key=lambda x:x['self_cuda_us'],reverse=True)
replacements['OPS']=table(['Rank / ATen op','Count','Self CUDA work s','Inclusive CUDA work s','Self CPU s','Inclusive CPU s','CUDA memory net MiB*'], [[f"{i+1}. `{x['name']}`",x['count'],f"{x['self_cuda_us']/1e6:.6f}",f"{x['cuda_us']/1e6:.6f}",f"{x['self_cpu_us']/1e6:.6f}",f"{x['cpu_us']/1e6:.6f}",f"{x['cuda_memory']/2**20:.2f}"] for i,x in enumerate(ops[:40])])
checkpoint_rows=[]
for x in read('checkpoint_warm/summary.json'):
    fs=[p for p in x['phases'] if p['stage']=='forward'][1:]
    bws=[p for p in x['phases'] if p['stage']=='backward_only'][1:]
    checkpoint_rows.append([x['experiment'],f"{statistics.mean(p['seconds'] for p in fs):.6f}",f"{statistics.mean(p['seconds'] for p in bws):.6f}",f"{fs[-1]['after']['allocated_gib']:.3f}",f"{x['memory_before_audit']['peak_allocated_gib']:.3f}",x.get('grad_comparison_to_off',{}).get('relative_l2','reference')])
replacements['CKPT']=table(['Actual block0 experiment (3 warm repeats)','Forward mean s','Backward mean s','Allocated immediately after forward GiB','Peak allocated GiB','Grad relative L2 vs OFF'],checkpoint_rows)
replacements['GRAPH']=table(['Autograd node type','Count'],sorted(base['graph']['node_types'].items(),key=lambda x:x[1],reverse=True)[:18])
replacements['WARM_TIMING']=table(['Warmed actual optimizer step2 phase','Seconds'], [[labels.get(k,k).replace('(cold)', '(warm)'),f'{v:.6f}'] for k,v in phases(warm).items() if 'audit' not in k]+[['**Phase sum (audit excluded)**',f'**{step_sum(warm):.6f}**']])
replacements['WARM_GRADS']=table(['Group','Warm step2 grad norm','Missing gradient tensors','Nonfinite gradient elements'], [[k,f"{warm['gradient_health']['groups'][k]['norm']:.9f}",warm['gradient_health']['groups'][k]['missing_grad_tensors'],warm['gradient_health']['groups'][k]['nonfinite_elements']] for k in groups])
replacements['BENCH_GRAD'] = ('; '.join(f"{k}: norm={v['norm']:.7f}, nonfinite={v['nonfinite_elements']}, missing={v['missing_grad_tensors']}" for k,v in bench.get('gradient_health',{}).get('groups',{}).items()) + f". Frozen violations={bench.get('gradient_health',{}).get('frozen_gradient_violations','not measured')}.")
replacements['PREFIX']=PREFIX

text = r'''# Hrlrvo 4K Training Backward Bottleneck Analysis

작성 기준: 현재 production v8 source와 이번 독립 실행의 실제 측정. 서버에서 수집한 timestamp는 UTC를 기준으로 해석한다(중단 기록 `2026-09-11T14:10:11Z`; nvidia-smi의 현지 표시는 UTC+9).

**최종 핵심 결론: backward 병목은 Attention MATH fallback이나 중첩 checkpoint가 아니라, R4 및 일부 R2의 BF16 depthwise Conv3D weight-gradient에 선택된 cuDNN 실행 경로다.**

## 1. Executive Summary

- 실제 **48 blocks, 4K internal 3840×2176, 65 RGB / 9 latent frames, batch 1**, real GT/Anchor cache, BF16, checkpoint ON으로 optimizer step을 완주했다. 원래 경로에서 **forward 24.415 s, backward-only 376.698 s**, clipping **0.045 s**, AdamW **0.296 s**다. `backward_and_clip`에 섞인 logging 시간이 아니라 **backward 자체가 느리다**.
- **R4**: `SpatialDecompressor.refine`의 `groups=4096`, BF16, channels-last-3D activation이 cuDNN의 기본 heuristic(`benchmark=False`)에서 **4096개 channel/group별 wgrad kernel** 경로를 선택한다. 대표 block당 R4 wgrad의 실제 GPU interval union은 **약 5.87 s**다. 이 일이 **48번** 반복된다.
- **R2**: 같은 primitive의 shape-dependent algorithm 선택 문제가 추가된다. K4의 12개 R2 호출은 모두 느린 경로; K5의 24개 중 4개만 같은 경로다. 그래서 **window가 더 적은 K4가 오히려 느리다**. 단순 window 수/graph 크기 설명은 반증되었다.
- **Checkpoint replay는 정확히 48회, 각 block 1회, 총 23.814 s**다. backward의 **6.32%**로서 15배 격차의 주원인이 아니다. 이미 `use_reentrant=False`다.
- Native / Adaptive / Global / Text CA의 대표 실제 forward/backward trace는 모두 **cuDNN SDPA**다. **관측된 MATH fallback은 0**이다. “flash”라는 cuDNN 내부 kernel 문자열을 PyTorch `FLASH_ATTENTION` backend로 혼동하지 않았다.
- 같은 source·shape·sample·seed·sigma에서 **R2/R4 Conv3d 입력 storage만 NCDHW-contiguous로 만든 독립 실험**은 backward **70.360 s**, 측정 phase 합 **94.919 s**로 줄었다. 원래 phase 합 **402.838 s** 대비 **4.24×**, backward만 **5.35×**다. **production에는 적용하지 않았다.**
- 이 layout 변경은 convolution 수학/weight ownership을 유지하지만 **BF16 reduction 경로가 달라져 bitwise training equivalence가 아니다**. 특히 선택한 R4 dW의 FP64 reference 오차는 NCDHW가 선택한 PyTorch native CUDA depthwise 경로의 오차가 더 컸다. 속도만 보고 무위험으로 승인하지 않는다.
- **마지막 실제 warmed optimizer step:** production source는 그대로 두고 새 프로세스에서 `cudnn.benchmark=True`만 설정한 뒤 같은 process의 두 번째 실제 step을 측정했다. **forward16.165s / backward38.398s / phase합54.674s**, 모든 group gradient finite/nonzero, frozen위반0, peakallocated129.255GiB. **이 runtime-only 후보를 우선 권장**한다. 첫 search 비용(독립 cold run248.715s; warm-step 실험의 같은-process 첫step249.264s)과 구분하고, 단일 record/한 warmed step이라는 검증 범위를 유지한다.
- production PID를 사용자의 허용 범위에서 중단했으며 **production training은 재시작하지 않았다**. 별도 진단 프로세스만 실행했다. 최종 source hash 검증은 §17에 기록한다.

**Evidence boundary:** 전체 48-block의 wall time/재계산 횟수는 직접 측정. 세부 operator/kernel attribution은 실제 K4/K5 block trace이며, 전체 R4 약 281.6 s / R2 약 37.2 s는 그 trace와 48-block 반복 구조에서의 **외삽**이다. 두 종류의 숫자를 섞지 않는다.

## 2. Current Production Baseline

### 2.1 중단 전 상태와 보존

| Item | Recorded value |
|---|---|
| Exact production PID / PPID | 2856987 / 1994057 |
| Command | `python -u -m ltx_outpaint.train main/configs/train_4k.yaml --output-dir /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/hrlrvo_main_train_v1 --disable-progress-bars` |
| tmux | `hrlrvo_train`, pane `%340`; session 유지 |
| 마지막 official 출력 | Step **40/30000**, loss **0.4758**, LR **9.99e-05**, Time/Step **370.92 s** |
| 정확한 종료 시 in-flight step | **확인 불가**. status가 20-step 주기이므로 step 40이라고 단정하지 않음 |
| 중단 직전 GPU | NVIDIA H200, **143771 MiB total / 142053 MiB used**, utilization **100%**, MIG disabled |
| 중단 | exact cmdline 확인 후 해당 PID에 **SIGINT**, `2026-09-11T14:10:11.369913Z` |
| 종료 증거 | native trainer의 `accelerator.backward`에서 KeyboardInterrupt, PID 소멸 확인 (`14:23:07Z`에 확인; 실제 exit time이라고 단정하지 않음) |
| 보존 | tmux, run, dataset/cache, checkpoint/model 파일 보존. 다른 사용자의 process는 종료하지 않음 |

원시 기록: [production_stop.json](@@PREFIX@@/production_stop.json), `production_console_before_stop.txt`, `production_console_stopped.txt`.
500-step checkpoint 이전의 미저장 진행은 이 승인된 중단에서 복구되지 않는다. 이번 진단은 해당 run에 resume/write하지 않았다.

### 2.2 실제 소프트웨어 / 학습 조건

| Item | Value / source |
|---|---|
| Torch / CUDA runtime / cuDNN | **2.8.0+cu128 / 12.8 / 91002 (9.10.2)** |
| Driver | 580.126.20; nvidia-smi의 CUDA 13.0은 driver capability이며 Torch runtime 12.8과 구분 |
| Torch allocatable total / initial free | 139.800720 / 139.286255 GiB; nvidia-smi physical total과 API 기준이 다름 |
| Environment | `Hrlrvo`; `CUDA_VISIBLE_DEVICES=0`, OMP/MKL threads 8 |
| Model | pinned LTX-2.3 revision `598ab41247a77dbfe29b5186e915bcf4f9040ec7`, actual 48×D4096, 32 heads ×128 |
| Weights | BF16 base; existing FP32 adapter/C/R/Fusion master policy; BF16 autocast |
| Trainable / frozen / installed | **280,363,008 / 18,988,838,144 / 19,269,201,152** |
| Batch / accumulation / world | **1 / 1 / 1** |
| Optimizer / LR / scheduler | unchanged AdamW, **1e-4**, LinearLR start 1.0 → end 0.1 / 30,000 steps |
| Gradient clipping | unchanged `max_grad_norm=1.0` |
| Checkpoint | whole native block, `use_reentrant=False`, ON |
| Temporal | FIRST 65 RGB→9 latent; 17 RGB prefix→3 latent, probability .5 |
| Seed | **33333333** |
| Custom heavy logging | default OFF (`training_runtime.py::run_training`, `diagnostics_enabled=False`) |

Production source: `main/configs/train_4k.yaml`; `main/src/ltx_outpaint/train.py::OutpaintingTrainer`; `training_runtime.py:129-155`; native `models/LTX2.3/packages/ltx-trainer/src/ltx_trainer/trainer.py:217-245`.

### 2.3 실제 데이터 / exact repeat identity

- Record: `0002e965-b236-4ec5-846d-4d1b919cf3f0.pt`.
- GT: `/home/work/ai-graphics_33/Hrlrvo/cache_Data/train/latents/0002e965-b236-4ec5-846d-4d1b919cf3f0.pt`.
- Anchor: `/home/work/ai-graphics_33/Hrlrvo/cache_Data_Anchor_2K/train/latents/0002e965-b236-4ec5-846d-4d1b919cf3f0.pt`.
- GT `[1,128,9,68,120]`, independently encoded Anchor `[1,128,9,34,60]`; known rectangle **y[17:51), x[30:90)**.
- Actual sigma **0.9246503114700317**; this fixed sample draw enabled 3 latent-frame temporal prefix. Known count **36,720 / 73,440 tokens**.
- Noisy model input `[1,73440,128]` SHA256 **`81a74ca4c1df1ed2470c2a2e9a630283f1e143e996f3912c095fb16bfd3504ac`**. Repeated full baseline/candidate runs check the same input and sigma.
- Existing cache HIT only, `created_count == 0`, VAE encoder was not instantiated. No cache overwrite/regeneration.
- Adapter initial state is the same seeded **fresh initialization**, not production step-40 trained weights. Kernel/shape bottleneck is established for this actual production architecture; arbitrary later trained values/multiple different samples were not exhaustively tested.

## 3. Profiling Methodology

### 3.1 공통 production component만 호출

Diagnostic `tools/backward_bottleneck_analysis/profile_training.py` imports the actual `OutpaintingTrainer`, config parser, `OutpaintingCacheDataset`, `collate_outpainting`, and shared Anchor cache consumer. It fixes one existing record instead of running the production sampler for thousands of steps. Output directory only is redirected to diagnostic artifacts.

```text
actual config / GT + Anchor dataset
→ actual OutpaintingTrainer model construction / adapter installation
→ OutpaintingTrainer._training_step (train.py:538-560)
→ actual LTX Transformer / 48 native block checkpoints
→ unchanged masked flow-matching loss
→ accelerator.backward(loss.mean())
→ accelerator.clip_grad_norm_(actual trainable params, 1.0)
→ actual AdamW.step → zero_grad → LinearLR.step
```

Loss source is pinned `training_strategies/flexible.py::_initialize_noisy_target` (velocity `noise-latents`, line459) and `_compute_modality_loss` (line676). Hrlrvo `train.py::OutpaintingStrategy.prepare_training_inputs` resolves `known = spatial | temporal`, reinjects clean conditioning, assigns known timestep0 and removes known from loss mask. Masked MSE is normalized over active elements, not the full canvas. These paths and sampler assertions are unchanged.

Loss/noise/mask/CA/FFN are not copied into a dummy model. No production checkpoint is saved. The actual trainer's optimizer order is preserved; diagnostic gradient inspection occurs after measured backward and outside the profiler.

### 3.2 Timing / ownership rules

- Phase boundaries use CUDA synchronize + wall clock **only in isolated diagnostics**. Per-block/branch CUDA Event pairs are resolved after backward, without a synchronize for every small operator.
- Parameter gradients are aggregated by group only **after backward timing/profiling ends**. The measured audit cost is **0.247 s**, separately labeled; it is not inside 376.698 s.
- Initial model construction, cache-contract hashing, Python graph traversal, JSON export and profiler postprocessing are not training-step compute. `phase sum` means explicitly listed phases, **not total diagnostic process lifetime**.
- Block backward boundaries use lightweight tensor hooks because native block arguments are dataclasses. Block 0 has no input dx hook; its end is the enclosing backward boundary. The emitted `end_is_backward_boundary` flag is not used as evidence: end events were filled before exporter logic. Timings and this explicit block-0 caveat are authoritative.
- Forward method scopes reached while `phase=backward_only` are **checkpoint replay forward time**, not that branch's gradient-kernel time.
- Profiler ATen backward operators map through autograd sequence numbers / forward thread to their forward branch scopes. CUDA kernel `External id`/correlation maps to those CPU op records. Separate tables distinguish **self CUDA work sum**, **inclusive parent work**, **overlap-merged device interval union**, and synchronized **wall time**.

### 3.3 실행 목록 / controlled hypotheses

| Experiment | Purpose | Scope |
|---|---|---|
| `full_baseline` | backward/clip/AdamW separation,48-block counts | actual full 4K65f optimizer step |
| `full_profile` | CPU+CUDA Kineto attribution attempt | actual full step completed; export aborted as documented below |
| `block_replay/k4_off,k5_off` | kernel/branch attribution | actual single production K4/K5 block, full 4K spatial+9 latent hidden |
| `checkpoint_warm` | checkpoint / reentrant clean A/B | same actual block0, one warmup +3 unprofiled repeats per mode |
| `block_replay/k4_flash,k4_efficient` | backend fallback hypothesis | actual block; no mask changes |
| `decompressor_ab` | layout/dispatch causal isolation | actual production R2/R4 class,D4096,T9; seeded synthetic probe/cotangent, not a substitute for full real-cache step |
| `conv_precision` | separate floating-point risk | exact-size operators,8 selected channel dW vs CPU FP64 direct reference |
| `full_ncdhw` | root-cause intervention validation | actual full step, temporary input storage conversion only |
| `fresh_benchmark`, `full_benchmark` | cuDNN algorithm search/cache hypothesis | new processes, flag set before any convolution |

**Full-profile limitation:** CPU+CUDA profiler used `record_shapes=False`, `with_stack=False`, `profile_memory=True`. The GPU step completed: forward **26.733 s**, backward **382.158 s**, clip **.0559 s**, optimizer **.3082 s**. Backward overhead versus baseline was about **1.45%**. Then Python/Kineto event materialization in `key_averages()` grew to roughly **92 GiB RSS** and prolonged CPU sorting (~17 minutes observed). The completed GPU result was preserved in `full_profile/completed_gpu_step_before_export.json`; the isolated diagnostic exporter was interrupted/stopped after GPU work ended. The final `result.json` export exception is **not training OOM**. No full-48 operator trace is claimed. Attribution instead uses bounded actual K4/K5 traces; their profiled wall times contain cold-start/CPU overhead and are not production speed benchmarks.

This limitation itself is compatible with millions of group-wise convolution launches; it is not proof of training activation retention. PyTorch warns that shapes/stacks add profiling overhead; they were disabled from the outset, but `profile_memory=True` and the massive event count still produced this postprocessing failure; no shapes/stacks-enabled comparison was run. [Official profiler documentation](https://docs.pytorch.org/docs/2.8/profiler.html)

## 4. Forward vs Backward Timing Breakdown

@@TIMING@@

**Q1 answer: `loss.backward()` itself = 376.698085 seconds.** Forward/backward ratio **15.43×**. Clipping and optimizer together are only **0.341 s**, so retaining max-grad-norm and AdamW is not the speed problem.

### 4.1 실제 같은-process warmed optimizer step

@@WARM_TIMING@@

정확히 같은 record/sigma/noisy SHA/known mask를 다시 사용하고, 첫 optimizer update 이후의 weights와 materialized Adam state를 그대로 사용했다. 두 번째 loss **.3351958096**, preclip norm **.1881634742**, LR **9.9994e-5**. 이 loss는 fresh-init baseline과 다른 weights의 결과이므로 numerical-equality comparison으로 쓰지 않는다.

Backward/forward ratio는 **2.375×**로 정상적인 checkpointed graph 수준으로 줄었다. 원래 cold phase합402.838s와의 단순비는7.37×, 기존 production warm log370.92s와의 참조비는6.78×다. 서로 다른 weights/steps/cold-state 비교이므로 이를 엄격한 동일-state paired speedup이라고 주장하지 않는다. 핵심 causal 증거는 full48 측정, operator/kernel 변화, 독립 microprecision 및 동일입력 반복 결과의 결합이다.

Original startup only: cache-contract setup **43.229 s**, actual model load **69.206 s**, optimizer object setup **.00646 s**. The first empty connector call is included in the phase table; later warmed steps need not repeat its cold cost.

`_training_step` forward-side measured wrapper wall is **25.8353 s** versus its named inner phases about **25.7692 s**. The approximately **.0661 s** remainder is Python/diagnostic bookkeeping between named regions; it is not a hidden 350-second CPU validation. Graph traversal/export and process startup have no invented per-step timing. CUDA wait at phase boundaries is attributed to the phase that queued the work.

@@FULL_COMPARE@@

Original allocated peak is **127.228 GiB**; reserved before diagnostics after backward **135.408 GiB**, and lifecycle reserved after Adam state materialization **137.938 GiB**. `allocated` and `reserved` are not the same. The NCDHW experiment largely preserves allocated peak; its reserved peak drops ~**4.545 GiB**. Neither result establishes unlimited headroom for a different sample or a checkpoint-OFF full model.

## 5. Attention Backend Analysis

### 5.1 Actual chosen backend, not priority text

| Branch | Production input/mask facts | Observed forward | Observed backward | Evidence scope |
|---|---|---|---|---|
| Native Local | K4 Q/K/V `[1,4590,4096]`; mask None | `aten::_scaled_dot_product_cudnn_attention` | `aten::_scaled_dot_product_cudnn_attention_backward` | K4/K5 actual traces |
| Adaptive | target-only Q; K4 Q=1215, mixed KV=2592/2700/3348; mask None | same cuDNN op | same cuDNN backward | variable-length actual local calls |
| Global | Q/K/V=4590 tokens (`9×17×30`); mask None | same cuDNN op | same cuDNN backward | actual compressed grid |
| Frozen Text CA | Q=73440,K/V=1024; additive FP32 mask `[1,1,1,1024]` | same cuDNN op | same cuDNN backward for hidden-state gradients | structurally active actual CA |

SDPA uses `is_causal=False`, dropout 0. The outer wrapper may see Native Q as FP32 before autocast; the actually dispatched SDPA fused path runs under production BF16 autocast. PMA/geometry creates Q/K phases, not an arbitrary dense hierarchy boolean mask passed to SDPA in these paths.

Kernel examples:

```text
cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_kernel0_0
cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_knob_26_64x64x128_1x4x1_kernel0_0
```

The `flash`/`f16` substrings in generated cuDNN names do not override the ATen backend identity or prove a production FP16 policy. Attribution uses the ATen cuDNN operators plus source autocast policy.

**Q4 answer:** observed Native/Adaptive/Global/Text CA use **CUDNN_ATTENTION in forward and backward**. Observed MATH fallback **0**. Because full trace export did not finish, this is not an exhaustive per-call kernel census of all 48 blocks; common implementation/shape parity and representative replay traces support, but do not overstate, whole-model coverage.

### 5.2 Backend A/B result

- Forced **FLASH**: Native/Adaptive/Global forward reached CA, then rejected at `attention.py::Attention.forward → masked_attention_function` with **“Flash Attention does not support non-null attn_mask” / “No available kernel”** in this installed stack. No mask was removed, no CA bypass performed. This is a controlled unsupported-backend result, not baseline failure.
- Forced **EFFICIENT**: actual block forward/backward finite; **.4427 s / 7.7498 s**, gradient relative L2 vs reference **8.95e-4**. The profile/cold history differs, so do not declare it faster than cuDNN from a cold comparator. Warm current cuDNN block backward is **7.5691 s**. It does not remove the decompressor floor.
- Forced MATH full attention was not run: there is no observed fallback to explain, and allocating the full large attention matrix is an unnecessary risk. No external flash-attn install/FlexAttention rewrite was attempted.

Source: pinned `ltx_core/model/transformer/attention.py::PytorchAttention.__call__` (SDPA call around line98), `MultiViewSelfAttention` and branch implementations. The priority `CUDNN>FLASH>EFFICIENT>MATH` is only eligibility order; actual events above establish selection.

## 6. Gradient Checkpointing / Recomputation Analysis

Native `ltx_core/model/transformer/model.py::_process_transformer_blocks` (loop at435-470) wraps each whole block once with **`use_reentrant=False`**. `main/src/ltx_outpaint` adds no nested branch/window checkpoints. `preserve_rng_state` is default True; non-reentrant early-stop is default enabled. Trainable FFN adapters near the block tail require almost the complete block replay.

| Question | Finding |
|---|---|
| 48 block replay calls | **48**, exactly **1 per block** |
| Full replay CUDA Event sum | **23.81373 s** |
| Full backward | **376.69809 s** |
| Replay fraction | **6.32%** |
| Branch/window nested checkpoint | **none** in active Hrlrvo path |
| Accidental repeated replay | not observed |
| `use_reentrant=False` missing? | **already applied**, not a new optimization |

### 6.1 Warmed controlled A/B

Actual block0 at full spatial/temporal hidden `[1,73440,4096]`; one warmup then three timed repeats, no profiler per repeat. All 48 weights resident but only one actual block graph active. Same explicit hidden tensor/VJP objective and parameters; no optimizer update between A/B modes. K5 profiling uses block1 weights with the captured first-block input as a controlled fixture, not a claim about the real layer1 activation distribution.

@@CKPT@@

Non-reentrant adds **.268685 s / 3.55%** over no-checkpoint backward: approximately **one extra .295-s forward**, not twelve. Reentrant differs by only ~**.006 s**, with BF16/non-deterministic reduction differences. The reentrant test uses an explicit tensor input/output shim around native dataclass arguments; simply replacing False→True in production would not be equivalent.

The checkpoint-OFF experiment retains ~**92.529 GiB** after forward versus **49.805 GiB** with checkpointing: a ~42.7-GiB activation-residency difference for this one block fixture. Peak during checkpoint recompute can still be similar/higher for one isolated block; that does not negate the full-model memory need. **48-block full4K checkpoint OFF was deliberately not run.**

After the runtime-only autotune remedy, the actual warm step still replays **exactly48blocks once each**, totaling **15.0775s** (~39.3% of38.398s backward). Its relative share grows because the pathological gradient kernels are gone, not because checkpointing worsened. This now sets a secondary upper budget; eliminating all replay is not presumed memory-feasible.

**Q3 answer: No.** Eliminating every measured replay could save only ~23.8 s before any secondary effects, leaving ~352.9 s. `preserve_rng_state=False`, switching reentrance, or nested-checkpoint removal cannot explain/fix the observed main gap. PyTorch documents the nested-tensor and early-stop differences. [Official checkpoint API](https://docs.pytorch.org/docs/2.8/checkpoint.html)

## 7. Block-wise Runtime Analysis

`Backward interval` includes that block's replay and gradient work, not only its forward kernels. The sum of block intervals **376.6016 s** matches whole backward **376.6981 s** within **.0965 s**. In the following table `Replay` is a subset of backward, not additional time.

@@BLOCKS@@

- Slowest: block47 **24.819 s**, then block46 **12.062 s**. They are the **first two visited in reverse order**, consistent with cold/library/order initialization effects; this is an inference, not an isolated measurement of initialization cost. It is incorrect to diagnose the last layer as architecturally different from this one cold step.
- Stable blocks0–45: K4 mean **7.878 s**, K5 mean **6.892 s**, gap **.986 s**. K5 has more local windows yet is faster: kernel choice, not window count alone.
- Early0–15 / middle16–31 / late32–47 backward means **7.388 / 7.417 / 8.733 s**. Last group includes cold reverse-entry outliers.
- After isolated NCDHW remedy, warm K4/K5 means become about **.919 / 1.074 s**; more-window K5 is now moderately slower. This reversal independently supports the shape-dependent decompressor diagnosis.

### 7.1 48-block forward subscopes (seconds)

Adaptive and Global columns include their C/R children. Gate-only columns do not include all surrounding lerp/scatter/assembly work. Cold CPU launch gaps can appear between CUDA Event boundaries; these are elapsed scope envelopes, not kernel sums.

@@BLOCK_FORWARD_DETAIL@@

## 8. Branch-wise Runtime Analysis

### 8.1 Full forward and replay scopes

@@BRANCH_FWD@@

Parents and children overlap hierarchically; **do not add every row**. This table quantifies forward/recompute for all 48 blocks; backward-kernel attribution follows separately.

### Native

`native_local_attention.py` reads the current hidden local window and computes frozen base + OP QKV, native normalization/RoPE, SDPA. No A/G adapters and no branch-local shared O. 16 K4 or25 K5 windows; **984 local calls per full forward**. Its full forward scope is **3.340 s**, replay **2.894 s**. Representative Native backward SDPA/GEMM/RoPE work is small relative to decompressor wgrad; no slow MATH path found.

### Adaptive

`adaptive_local_attention.py` uses current hidden, resolved A>T>B, aligned support, full C2 promotion and shared pure-B C4 context, target-only Q, PMA, R2 and target-only scatter. **864 nonempty target windows** over48 blocks; 120 geometry-empty local calls exit before attention/R2. Read/support planning is CPU geometry work, while compressed reads/projections/restores are autograd operations. Full forward **13.210 s**, replay **12.750 s**: it is the largest **forward** branch, but not the largest **backward** branch.

R2 weight-gradient is the main measured Adaptive backward problem. C2/C4 conv, QKV, PMA, index/scatter have true costs, but the selected trace shows no comparable multi-second kernel floor from them. K4 R2 wgrad union **1.284 s**, K5 **.267 s** per representative block.

### Global

`global_attention.py::global_attention` calls per-block shared C4 → base+OP+G QKV attention → R4 whole-grid restore. Canonical C4 `68×120→17×30`; R4 bilinear/AA spatial upsampling + existing trainable DW Conv3D. Forward **2.526 s**, replay **2.971 s** for48 blocks, but **one R4 wgrad consumes ~5.87 s of actual kernel occupancy per block**.

**Q6 answer: Global, specifically R4, dominates backward.** Extrapolating representative identical-shape calls gives **~281.6 s (~74.8% of full backward)** of R4 wgrad **interval-union occupancy**. Adaptive R2 adds **~37.2 s (~9.9%)**. These percentages are a documented extrapolation from per-block union timings, not a completed full-trace exact branch partition. Native/CA/FFN CPU scheduling and remaining kernels share the residual; no false exact full-branch percentages are invented.

### Shared O / Text CA / FFN

Final Local–Global fusion → shared `attn1.to_out` exactly once → native SA residual → frozen Text CA with hidden-state autograd → shared FFN+F-LoRA. Full forward O **.253 s**, CA **1.333 s**, FFN **1.756 s**; replay approximately **.251 /1.290 /1.759 s**. Frozen weights omit parameter gradients but do not eliminate necessary input gradients. No higher-order graph or CA bypass.

### 8.2 Selected-block backward source attribution (CUDA work sums)

@@BRANCH_BWD_WORK@@

**Important:** R4's ~46.8 “self CUDA seconds” is **work summed over concurrent streams**, not a46.8-second block. The actual whole block is shorter. Use §9 union measurements for wall contributions. Successfully correlated CPU-op self CUDA work is assigned to its nearest annotated forward source; base/normalization/assembly work can remain in shared scopes. These values cannot uniquely assign cost to only a group's weight gradient rather than its input-gradient work. Sequence matching has **50 K4 /97 K5 ambiguous keys** plus an UNATTRIBUTED bucket, so narrow group totals are useful operator-source accounting, not a perfect unique decomposition.

## 9. C2/C4/R2/R4/Fusion Analysis

### 9.1 Exact operator and layout chain

`main/src/ltx_outpaint/compression.py::SpatialDecompressor`:

```text
line53: Conv3d(D,D,kernel=3,stride=1,padding=1,groups=D)
line64: tokens → reshape → permute NCHW (channel stride=1)
line67: F.interpolate(float32, bilinear, antialias=True)
line68: restore original dtype
line69: reshape / permute to logical N,C,T,H,W
line70: self.refine(volume)
```

D=4096; FP32 master kernel shape `[4096,1,3,3,3]` +4096 biases. **Only 114,688 parameters per decompressor per block**, but each R4 kernel touches full restored `[1,4096,9,68,120]`.

Measured strides:

```text
R4 logical [1,4096,9,68,120]
stride     [300810240,1,33423360,491520,4096]   # channels_last_3d
R2 logical [135,4096,9,2,2]
stride     [147456,1,16384,8192,4096]
```

This is storage layout, not a wrong semantic axis order. Source is not doing an invalid convolution. The **selected implementation is inefficient for these valid shapes**.

### 9.2 CUDA kernel proof and overlap-correct accounting

Kernel: `cudnn::cnn::convolveNd_wgrad_engine<__nv_bfloat16,3,...>`.
K4 has **13 calls ×4096 groups =53,248 wgrad kernels**. Associated layout transforms are also group-wise. Mapping uses autograd sequence→forward scope→convolution External ID→CUDA kernel intervals. Two missing External IDs were recovered uniquely through cudaLaunchKernel correlation.

| Representative trace | wgrad launches | Summed kernel durations | Overlap-merged GPU occupancy |
|---|---:|---:|---:|
| K4 Global R4 | 4096 | 46.676 s | **5.865 s** |
| K4 Adaptive R2 (12 calls) | 49152 | 8.839 s | **1.284 s** |
| **K4 R4+R2 wgrad** | **53248** | **55.515 s** | **7.149 s** |
| K5 Global R4 | 4096 | 46.818 s | **5.868 s** |
| K5 R2 pathological subset (4 of24 calls) | 16384 | .561 s | **.251 s** |
| K5 other20 R2 calls | 40 tensor-core wgrad kernels | .016 s | about **.016 s** |
| **K5 all R2 wgrad** | **16424** | **.577 s** | **.267 s** |

K4 backward kernel union total is **7.463 s**; R4/R2 all kernels union **7.189 s =96.3% timeline coverage of kernel-active time**. This coverage can overlap other kernels and is not an exclusive partition of GPU time. Eight overlapping CUDA streams explain why summed wgrad55.515s is only7.149s of device-wall occupancy. The profiled/cold K4 synchronized backward is11.601s; wgrad occupies61.6% of that wall. Remaining CPU/profiler gaps must not be reassigned to convolution GPU work.

K5 R2 target-cell counts21/24 choose the slow per-group engine;39/84 select a two-kernel tensor-core wgrad route. K4 count135 chooses the slow route. **This algorithm discontinuity explains the measured K4/K5 gap**: R2 union difference1.017s versus full-model stable block difference.986s.

Full-model estimate, explicitly extrapolated:

```text
R4 interval-union occupancy ≈24×(5.865+5.868)=281.6 s
R2 interval-union occupancy ≈24×(1.284+0.267)=37.2 s
combined≈318.8 s =84.6% of376.7 s
```

This is an occupancy extrapolation, not a guaranteed exclusive/removable time budget. It is consistent with the **direct full-model intervention saving306.34 backward seconds**. It need not equal the savings exactly: replay/CPU scheduling/newly exposed kernels and cold effects differ.

### 9.3 Why a layout change selects a different backend

Tagged PyTorch2.8 source distinguishes these paths:

1. `cudnn_conv_suggest_memory_format` selects ChannelsLast3d if input **or weight** suggests it.
2. `use_cudnn_depthwise` has an early channels-last cuDNN path. Contiguous specialized cuDNN eligibility is narrower (FP16/4-D), so this **BF16/5-D** NCDHW case reaches **native `CudaDepthwise3d`** instead.
3. The native CUDA implementation supports BF16. The independent trace records `aten::conv_depthwise3d`, `conv_depthwise3d_cuda_backward_input_kernel<c10::BFloat16,...>` and `conv_depthwise3d_cuda_backward_weight_kernel<c10::BFloat16,...>` directly (`conv_precision/R4_ncdhw_ops.json`); the backend switch is observed, not merely guessed from timing. The measured NCDHW and scoped-cuDNN-off results both use this alternative, not a quantized or CPU-offloaded operator.

References: [v2.8 ConvUtils.h](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/ConvUtils.h#L320-L360), [Convolution.cpp](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/Convolution.cpp#L423-L507), [native depthwise dispatch](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/Convolution.cpp#L1207-L1223), [BF16 native implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/DepthwiseConv3d.cu#L636-L700).

Do not generalize this to every Conv3d/cuDNN version. cuDNN's exact internal heuristic preference is not derived from public source. Weight memory-format conversion alone did **not** help here. Input-contiguous only selects the measured native route while the weight also does not suggest CL3d; changing both layouts can invalidate that inference.

### 9.4 C2/C4 and Fusion are not the hidden multi-minute cost

C2/shared C4 remain full-cell spatial depthwise Conv2D operations, same learned C4 Parameter identity for Adaptive/Global. K4 selected backward CUDA work C2~.029s, C4~.022s; K5~.051/.047s. Local gate~.00087s, global gate~.00221s in K4. These are not comparable to R4's per-channel wgrad kernels. Fusion lerp/index assembly has additional shared-scope costs; the gate-only number is not the entire assembly cost.

**Q7 answer:** yes—**R4 and some R2** are disproportionately expensive despite having far fewer parameters than LoRAs. C2/C4/Fusion and bilinear interpolation are measured secondary costs, not the multi-minute cause. R4 interpolation backward K4 work is contained in the overall13-call AA interpolation backward total of about.0345s; the DWConv3D wgrad dominates by orders of magnitude.

## 10. Gradient Health vs Runtime

Gradient collection is outside backward/profiler timing, with GPU aggregation then CPU read. All **1728 trainable tensors** have grad, all ten groups have finite/nonzero norm; **frozen gradient violations =0**. Initial zero-B LoRA naturally gives zero A gradients on some tensors; group connectivity is the correct interpretation.

@@GRADS@@

같은 group의 parameter 규모와 해당 연산 경로 backward 비용을 나란히 비교한다:

@@GROUP_COST@@

*분모는 해당 group의 **per-block** parameter 수(전체/48)다. 분자는 그 연산 경로의 dW뿐 아니라 dx/주변 cast를 포함한 **overlap-summed GPU work**, not wall latency. 이 값은 parameter 수만으로 runtime을 예측할 수 없다는 진단 지표이며 branch 가치/gradient 중요도/시간 절감률을 뜻하지 않는다. R4/R2 latency는 반드시 §9 union 값을 사용한다.

Global preclip norm **.196903184**, so the unchanged clip1.0 call does not scale this sample's gradients. AdamW state tensor element count **560,727,744** equals two moment tensors over280,363,008 parameters plus1728 step counters. Thus this was not a forward-only or backward-only “training success.” LR after the real optimizer/scheduler step is **9.999700000000001e-05**.

Runtime comparison by group is in §8.2. In particular, R4's **5.505M parameters** (~1.96% of trainables) account for the largest kernel-wall estimate, while A-LoRA's75.5M do not. “seconds per parameter” is not a physical efficiency metric because token count, restored grid, backward-input work and kernel dispatch differ; group norms/costs are shown together without making a pruning/value judgement.

**Q9 answer: Yes for this measured full sample.** All intended trainable groups receive finite/nonzero gradients; frozen base/CA parameters have no gradients. This is not a claim that every module has nonzero gradient at every possible prefix/geometry draw, nor a long-run numerical convergence guarantee.

Fresh benchmark gradient audit: @@BENCH_GRAD@@

Actual warmed second optimizer step (post-update weights):

@@WARM_GRADS@@

All1728 trainable tensors have gradients, **all10 group norms are finite/nonzero**, frozen violations0. This does not assert every individual tensor has a nonzero gradient. No long-run convergence/quality claim is made.

## 11. Autograd / Graph Duplication Investigation

### 11.1 Active-code findings

- No production `retain_graph=True`, `create_graph=True`, higher-order backward, or repeated optimizer backward on the same step.
- Whole-block checkpoint is single-layered; replay exactly1/block. No per-window nested checkpoint.
- Gate ContextVar resets in `finally` (`model.py::_forward_with_native_gate`) and the captured native gate is cleared before CA/FFN. The normal production path does not install logging observers or keep graph tensors in a metrics history.
- Local window result lists retain a **bounded16/25** tensors until assembly—part of the specified graph, not an accumulating cross-step cache.
- Full-grid read/scatter fan-out is real:984 Native windows and1440 total C2/C4 gathers across48 blocks. Many View/Slice/Index/Copy nodes do not by themselves prove duplicate computation.
- No experiment changed source lock/masks/loss, disabled hidden gradients through frozen modules, or detached a trainable branch.

### 11.2 Measured graph size and saved-tensor boundary

Full loss graph has **277,514 autograd nodes**. Key counts: View53070, ToCopy31369, Slice25904, UnsafeView20696, Mul20663, Mm17136, T17136, Transpose15264, ReshapeAlias10783, Add8856, CopySlices7584, Addmm5976, Permute5887.

The exact **7584 CopySlices =1896 custom SA calls×Q/K×2 split-RoPE half writes**. Source: pinned `rope.py::apply_split_rotary_emb` and `ops.py::PytorchPreAttention`. It is **not** evidence for7584 full-canvas Adaptive overwrites.

NCDHW candidate graph has **278,474 nodes (+960)** yet backward is5.35× faster. This causal counterexample rejects “node count alone causes370s.” Graph kernel choice matters much more here. Full block0, with no input dx boundary, still takes7.844s in baseline, so previous-block full-canvas gradient scatter alone cannot explain the floor.

One BF16 hidden grid is300,810,240 elements≈**.560GiB**. Forty-eight such checkpoint boundaries would nominally be~26.9GiB before aliasing/other tensors, **not a measured saved-activation total**. An exhaustive `saved_tensors_hooks` census was not run: checkpoint hooks would complicate attribution and add overhead after the primary cause was proven. Warm block residency difference42.7GiB (§6) provides a direct activation-memory observation. No exact per-branch saved/unpacked byte total is claimed.

**Q8 answer:** large fixed fan-out exists, but no accidental repeated checkpoint graph/unbounded per-step retention was found. Many-kernel cuDNN wgrad is the demonstrated cause; graph cleanup/compile remains secondary until measured after fixing dispatch.

## 12. Profiling Top CUDA Operators

Actual K4 block0, CPU+CUDA, forward+backward, profiler enabled. Sorted by ATen **self CUDA work sum**, not wall time. CUDA kernel event rows are not counted again as separate ATen operators. Top40:

@@OPS@@

`CUDA memory net` is the profiler operator's aggregated inclusive allocation/deallocation accounting, **not peak live memory**. CPU self/envelope and inclusive CUDA also cannot be added to self CUDA. For `convolution_backward`, **55.9s summed CUDA work** overlaps across8streams; §9's7.149s wgrad union is the right comparison to block latency.

Additional raw kernel evidence: BF16`convolveNd_wgrad_engine` **53248calls**, group-wise layout transforms, cuDNN SM90 SDPA forward/backward kernels. ATen mm/addmm/copy/AA-upsample/index operations are visible but do not explain the bulk of376.7s.

Artifacts: `block_replay/k4_off/{trace.json,profiler_ops.json,profiler_attribution.json}`, equivalent `k5_off` files; exact exported trace filename can be read from that directory. The full48 trace export failure is explicitly documented in §3; this table does **not** pretend to be full48 operator totals.

## 13. Root Causes Ranked

| Rank | Cause | Numeric evidence | Confidence / caveat |
|---|---|---|---|
| **1** | **Global R4 BF16 DWConv3D dW: channels-last→cuDNN4096-group engine** |5.865–5.868s wgrad union/block; ~281.6s/48 estimated; layout-only full-step intervention saves306s backward jointly withR2 | **High / causally tested**; exact full48 R4-only time is extrapolated |
| **2** | **Adaptive R2 shape-dependent same pathological dW dispatch** |K4 12slowcalls→1.284s union; K5 4/24slow→.267s; predicts measuredK4/K5 .986s gap | **High**; do not equate more windows with slower backward |
| **3** | **Ordinary whole-block replay + remaining autograd/kernel scheduling** |48replays23.814s=6.32%; no nesting; warmed A/B adds.269s/block | **Measured but normal**, not the source of12–15× asymmetry |

R4/R2 per-group launch/layout-conversion storm also consumes CPU submission time. It is coupled to causes1/2, **not a separate additive55-second/second-copy estimate**. Tiny optimizer/clipping costs, SDPA MATH fallback, frozen parameter audit and checkpoint duplication are rejected as primary explanations by measured evidence.

Interim static notes initially considered attention arithmetic and scatter fan-out. Kernel attribution falsified the attention-first ranking; **this final report supersedes those preliminary hypotheses**.

## 14. Optimization Experiments

All interventions were temporary in standalone processes. No persistent production optimization was applied.

### 14.1 Same primitive, shape and BF16: storage/backend micro A/B

| Operator / mode | Warm forward s | Warm backward s | Peak allocated GiB | Result |
|---|---:|---:|---:|---|
| R4 original channels-last input |.045555|**5.915762**|6.764|baseline |
| R4 weight CL3d only |.04545|5.91624|about6.764|no observed win |
| R4 input NCDHW-contiguous |**.015093**|**.031870**|3.469|~186× local backward speedup |
| R4 scoped cuDNN disabled |.01508|.033992|3.470|same native-depthwise alternative |
| R2 original (135query cells) |.003388|**.112855**|.459|baseline |
| R2 input NCDHW-contiguous |.002268|**.024262**|.325|~4.65× local backward speedup |
| R2 scoped cuDNN disabled |similar|.024374|similar|alternative, not additive |

These use actual `SpatialDecompressor`, current initialization and full logical sizes but a fixed random operator probe/VJP. They are not synthetic substitutes for the full-model result. The full actual real-cache layout experiment (§4) confirms that this local improvement matters end-to-end.

### 14.2 Numerical/learning-risk audit—important qualification

Identity-initialized micro forward outputs were bit-exact between original and NCDHW. dW is **not** bitwise equal: R4 relativeL2 difference about**.00558**, R2 about**.01424**. Some R2 original repeats themselves are nondeterministic. Full actual loss changes **.3444887698→.3445034325** (~4.26e-5 relative), global gradient norm **.1969032→.1962864**. All gradient groups remain healthy, but equal norms do not imply equal gradient vectors or long-run learning outcomes.

Independent CPU FP64 direct convolution-dW reference on8selected channels, using exact BF16 inputs/cotangents:

| Operator | Original relativeL2 error vsFP64 | NCDHW native relativeL2 error vsFP64 | Interpretation |
|---|---:|---:|---|
| R4 |**.0017375**|**.0053875**|NCDHW-selected **PyTorch native CUDA depthwise dW** was **less accurate on this probe**; original equals BF16-rounded FP64 reference, native does not |
| R2 |.012164 (repeat .013726)|**.0039076**|Native was more accurate on this probe; original reduction nondeterminism visible |

This is8channels/onefixture, not a universal accuracy ordering. A layout-only mathematically identical operator can change BF16 accumulation. **Do not silently apply and call it “no learning effect.”** Approval plus short controlled loss/gradient-comparison validation is warranted. No architecture/weight/rank/freeze workaround was used.

### 14.3 Fresh-process cuDNN benchmark A/B

The first same-process `benchmark=False→True` toggle showed5.916→5.921s and was **cache-confounded**, not proof that autotuning is useless. In PyTorch2.8 the cuDNN convolution cache key excludes `benchmark`; cache lookup precedes the heuristic/find choice. [v2.8 cuDNN cache and search](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cudnn/Conv_v8.cpp#L974-L1068)

A fresh process set `torch.backends.cudnn.benchmark=True` **before any convolution**, retaining default benchmark_limit10. Same R4 original-layout micro warm backward becomes **.059673s**, ~99× faster than5.915762s. Warm forward~.03075s. Cold search cost is real: firstforward~1.1455s, firstbackward~13.3934s, peakreserved~15.283GiB for the isolated module.

@@BENCH_RESULT@@

The timing concentration is consistent with cold autotuning/library work in reverse-entry blocks (their entire time is not proved to be search): b47 **133.042s**, b46 **45.380s**; together **178.423s**. Remaining b0–45 mean **K4 .721448s / K5 .884486s**. These within-step warmed block values are **not a full warm-step measurement**.

Independent warm operator traces preserve **cuDNN** while changing the plan: the warmed **R4 and135-cell R2 operator probes** each emit **2 tensor-core cuDNN xmma wgrad kernels**, not4096 `convolveNd_wgrad_engine` kernels. R4 named wgrad work **11.099ms**, R2 **1.083ms**. The conv remains trainable; only selected implementation changes.

The independent FP64-reference probe (`benchmark_precision/`) gives R4 relativeL2 dW error **.00173797** (default .00173752, NCDHW-selected PyTorch CUDA .00538747), R2 **.00168876** (default .012164, NCDHW-selected PyTorch CUDA .00390755). This supports lower sampled numerical risk than NCDHW for R4, but only8channels were checked. The earlier `fresh_benchmark` JSON zero-difference field was a self-reference and is **not used as cross-mode precision evidence**.

A separate `benchmark_warm_training` process reproduced the cold first step (**F31.5563/B216.0128s, phase합249.2641s**) and then performed a real second optimizer step in that same live process: **F16.164762s / B38.398138s / clip.016926s / AdamW.029813s / phase합54.674292s**. That is the authoritative warmed result, not the extrapolated .721/.884 block means. It used the same sample/sigma/input hash, but post-step1 weights and live Adam/scheduler state. Peakallocated **129.255GiB**, peakreserved **135.934GiB**, endphysicalfree **3.101GiB**. No OOM; all groups healthy. Existing shape-cache set is warmed; an unseen shape/layout may incur a new search.

The full actual step in§4 is the authoritative full-size feasibility test, not the fast micro number. Autotuning can incur searches for varyingR2query-count/layout/alignment shapes and needs workspace. A fresh process was essential; changing a flag after algorithm-cache warmup would not reproduce this comparison. Public API semantics and reproducibility tradeoffs: [PyTorch backends](https://docs.pytorch.org/docs/2.8/backends.html#torch.backends.cudnn.benchmark), [reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html).

### 14.4 Experiments deliberately not escalated

- No full48checkpointOFF OOM fishing; one-block warmed A/B already quantifies recompute cost.
- No MATH attention stress run when no baseline fallback exists.
- No FP8/quantization/offload/frame reduction/branch removal/rank reduction.
- No immediate torch.compile: production native-gate bridge currently checks/rejects compilation; dynamic local-token geometry and Python loops need a separate correctness/graph-break effort. Compile is not a one-line established solution here.
- Full runs addressed distinct questions (baseline, profiler, layout causality, fresh autotuning, and one same-process warmed optimizer step). The final warm-step check separates expensive first-search behavior from actual repeated training; it is not a blind parameter sweep.

## 15. Recommended Optimization Plan

**Analysis only. Recommendations are not applied.** “Low risk” below means low architecture/code-scope risk, not bitwise or long-run-convergence proof. The first choices are alternative ways to avoid the **same** slow convolution implementation; speedups are **not additive**.

@@RECOMMENDATIONS@@

Guardrails before any future approved change:

1. Preserve exact Conv3D k3/stride1/pad1/groups4096, learned weight/bias objects, all gradients, input values and temporal/spatial axes.
2. Keep R2 full2×2 restore then target mask/scatter; R4 whole-grid bilinearAA+refinement. Do not replace learned refinement by identity just because initialization is identity.
3. Compare same saved adapter/input/seed/sigma, finite loss and all group gradients; use a documented BF16 tolerance and inspect R4 dW, not only gradient norms. Check resume/state_dict/parameter identity.
4. Measure several warmed actual steps and varying production-valid R2 shapes, with no permanent profiler/health scan in the hot path. Monitor peak headroom and cold autotune workspace.
5. Only after eliminating this backend pathology, profile the remaining measured **38.40-second warm autotuned backward**, including15.08s replay (or the separate NCDHW70.36-second cold backward) to assess checkpoint/cache/dispatch/compile. Do not install FlashAttention or change masks in response to the wrong root cause.

Multi-GPU/DDP may improve later sample throughput but does not fix a5.9-second per-block local kernel. It is a later scale-out option, not this diagnosis.

## 16. Expected Training-Time Reduction

@@SCENARIOS@@

Days = `seconds_per_step×30000/86400`; assumes continuous single-GPU utilization and excludes VAE cache misses, checkpoint I/O, restarts, validation and unrelated GPU competition. New shapes/autotuning or trained states can alter timings. No fabricated ETA is a completion promise.

- Baseline measured phase-sum402.84s is a cold actual step; observed original production Time/Step370.92s is a separate warmed log metric.
- Isolated layout-only coldactual94.92s is **measured**. Its stable K4/K5 backward intervals sum~47.8s over48blocks; with~17–23s forward and small optimizer cost, an unmeasured warmed~65–75s scenario is plausible, **not yet a full multi-step measurement**.
- The runtime-only autotuned actual warm54.674s is now measured; its long-run multi-sample distribution is not. A medium/aggressive target must come from the remaining budget, not applying186× micro speedup to the whole model. There is no evidence for arbitrary10s/step fulltraining or elimination of native CA/FFN.

## 17. Risks / Things NOT To Change

### Preserved

- Production v8 hierarchy,OP/A/G/F, ranks/alphas, sharedC4 ownership, C2/R2/R4 primitives, target-only write, masks/prefix/geometry/PMA, fusion/sharedO/CA/FFN unchanged.
- Actual production optimizer/LR/scheduler, clipping1.0,BF16,accumulation,seed,checkpoint interval/format,trainable/frozen policy unchanged.
- Pinned source/model/rawdataset/GTcache/Anchorcache untouched; real-sample diagnostic refused a cache miss rather than encoding a new Anchor.
- Production run was not resumed or restarted. No monitoring process left attached to production.

### Measurement limitations

1. No completed full48operator trace; selected actual block traces plusfull48wall/recompute/candidateintervention establish the cause. Fulltrace Pythonexport aborted, not hidden.
2. Cold firstreverseblocks and profiled selected-blockCPU overhead inflate somewalls. Warmed checkpoint A/B is separate; not alltimingrows are commensurate.
3. Exactfull Native/Adaptive/Global backwardcriticalpath percentages unavailable. R4/R2 full percentages explicitlyextrapolated; selfCUDAwork tables explicitlyoverlap-summed.
4. Checkpoint saved-tensor byte census notrun; graphnodes/residency measured instead. No false claim ofzero graphgrowth fromone-stepmemoryalone.
5. Full tests used one real record, fresh initialization and an actual second updated step; no step40 weight resumption,long-runquality,orallrecordsvalidation. BF16numericaldifferences quantified, not dismissed.
6. Existingprofilingrawnotes includeearlierhypotheses; thisfinalreport uses latestkernelmeasurements, especiallytheR2shapealgorithmcounterexample.

### Files / verification

Only analysis scripts/artifacts and this report were added/edited. New scripts under `tools/backward_bottleneck_analysis/` are not imported by production. Analysis runtime monkeypatches/hooks/backend flags die with their isolated process; no source revert of production was needed.

@@FINAL_INTEGRITY@@

### Raw evidence index

Root: `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/@@PREFIX@@/`

- `full_baseline/`: result,per-phase memory/timing,48-blockscopes,exactsamplehash.
- `full_profile/completed_gpu_step_before_export.json`, `export_abort.json`: completedGPUstep andexportlimitation.
- `block_replay/k4_off/`, `k5_off/`: actualprofilerATen/kernel/sequenceattribution andChrome traces.
- `checkpoint_warm/`: warmedOFF/nonreentrant/reentrant3repeatA/B.
- `decompressor_ab/`, `conv_precision/`: actualprimitive layout andFP64dWrisk.
- `full_ncdhw/`: actualfull4K isolatedlayout-onlyoptimizerstep.
- `fresh_benchmark/`, `full_benchmark/`, `benchmark_precision/`, `benchmark_warm_training/warm_step_2/`: fresh algorithm search, independent precision, and actual same-process warm step.
- `production_stop.json`, `production_hashes_before.json`, `production_hashes_after_check.json`: process/sourcepreservationevidence.

## 18. Final Verdict

**BACKWARD BOTTLENECK: IDENTIFIED WITH REAL 4K EXECUTION AND KERNEL EVIDENCE.**

| User question | Final answer |
|---|---|
| Q1 Actual backward-only? | Original **376.698s**; runtime-only autotuned warm **38.398s**; clipping/AdamW measured separately |
| Q2 Top3 causes? | R4cuDNNdWdispatch; R2shape-dependentdWdispatch; normalcheckpointreplay/remainingautogradwork as secondary |
| Q3 Checkpoint alone explains15×? | **No**; exactly1replay/block,23.814s(6.32%) |
| Q4 Actualattentionbackend/MATH? | **cuDNN forward+backward** inall4representativepaths; observedMATH0 |
| Q5 Particularslowblock? | All48haveR4floor;47/46coldoutliers,stableK4>K5by.986s explainedbyR2algorithm |
| Q6 Mostexpensivebranch? | **Global R4**, ~281.6sfull extrapolated wgrad; AdaptiveR2~37.2s |
| Q7 C/R/Fusion surprise? | **R4/R2DWConv3DdW**, not C2/C4/Fusiongate |
| Q8 Accidentalgraphduplication? | Notfound; fixedlargegraph/normalreplay, no nested orretainedhigherordergraph |
| Q9 Gradientshealthy? | **All10groupsfinite/nonzero,1728gradtensors,frozenviolations0** onactualsample |
| Q10 Safestnextsteps? | AvoidpathologicalR4/R2kernelbyfreshautotuneorboundedlayoutdispatch; validateBF16dW/workspace; retaincheckpoint/SDPA/architecture |

**Production optimization applied: NO. Production restarted: NO.**
Measured preferred candidate: **fresh-process cuDNN autotuning**, actual warm phase sum **54.674s** without architecture changes; preserve its startup/workspace/non-bitwise caveats.

The next decision is the user's approval of a narrowly scoped convolution-backend remedy and its numerical validation—not a new architecture or a blind attention/checkpoint rewrite.
'''

# Final recommendations/scenarios depend on the completed fresh full-step result.
replacements['RECOMMENDATIONS'] = (A / 'recommendations_final.txt').read_text()
replacements['SCENARIOS'] = (A / 'scenarios_final.txt').read_text()
replacements['FINAL_INTEGRITY'] = (A / 'integrity_final.txt').read_text()
for key,value in replacements.items():
    text=text.replace('@@'+key+'@@',value)
if '@@' in text:
    raise RuntimeError('Unresolved report placeholder')
(ROOT/'BACKWARD_BOTTLENECK_ANALYSIS.md').write_text(text)
print('Wrote', ROOT/'BACKWARD_BOTTLENECK_ANALYSIS.md', len(text))
