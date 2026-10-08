# Static Gradient and Train–Inference Parity Audit

**Audit date:** 2026-09-12 UTC  
**Repository:** `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting`  
**Environment:** `Hrlrvo`, installed PyTorch 2.8.0+cu128  
**Overall verdict: PASS WITH CAVEATS**

## 1. Executive Summary

현재 소스에서 의도한 v8 trainable 10개 그룹 모두에 대해 **parameter → forward operation → hierarchical fusion/residual → final video prediction → masked loss** 경로를 확인했다. 활성 학습 경로의 의도치 않은 gradient 절단, dead trainable group, 잘못된 cross-block parameter sharing 또는 train/inference adapter key mismatch는 발견하지 않았다.

다만 **동일 architecture**와 **모든 추론 호출의 동일 실행/정밀도/입력 분포**는 다르다. 다음 차이를 숨기지 않고 조건부 판정한다.

1. Training adapter master는 FP32, inference는 설치 후 전체 transformer를 BF16으로 변환한다. Key/shape/parameter 의미는 같지만 FP32 checkpoint 값의 bitwise 보존이나 train/eval 수치 일치까지 뜻하지 않는다.
2. Inference `stg_scale=1.0`, `stg_blocks=[28]`는 **활성 STG**다. 일반 conditional pass는 같은 전체 hierarchy를 사용하지만 STG 보조 pass의 block 28은 의도된 attention bypass를 사용한다. `1.0`이 STG-off라는 해석은 틀리다.
3. Training FIRST clip/GT clean prefix와 inference recurrent generated-RGB prefix는 같은 known-mask contract를 사용하지만 prefix 내용과 temporal origin은 다르다.
4. Inference YAML은 `enforce_expected_anchor: false`다. 임의 입력에서도 학습의 1920×1080 Anchor geometry가 보장되는 것은 아니다.
5. 현재 inference YAML의 기본 `checkpoint-030000` 파일은 없다. 명시적인 존재하는 checkpoint가 필요하며 loader는 누락 시 fail-closed한다. Architecture/key 문제와 구분한다.

### Evidence discipline / 실제 수행 범위

- **Evidence (E):** 현재 source/AST, config parse, 실제 base safetensors **JSON header만** 읽기, 실제 공식 configurator와 공통 installer로 만든 **48-block meta 객체 2개**, 기존 step-5 adapter header, CPU pure geometry/mask/lineage checks.
- **Inference (I):** differentiable PyTorch 연산의 source/destination 관계를 따라 도출한 gradient 연결성. 이것은 이번 작업에서 gradient 값을 측정했다는 뜻이 아니다.
- **Unknown (U):** 이 감사로 새 loss/gradient 값, numerical stability, GPU kernel dispatch, runtime 성능, bitwise output equality를 입증하지 않는다.
- Git repository가 root에 없어 SHA256으로 보호한 source/config/test/pinned package 파일 **339개 전후 동일**. Production source/config, pinned LTX, 기존 run/checkpoint/cache 수정 없음.
- `CUDA_VISIBLE_DEVICES=''`; 종료 시 `torch.cuda.is_initialized() == False`.
- `nn.Module._call_impl`을 감사 subprocess에서 금지하도록 guard: **module forward 호출 0**. Backward도 금지 guard. Optimizer/Trainer 실행, VAE encoding, generation, profiler, checkpoint 생성, production 재시작 **0**.
- 컴파일은 소스 문자열 `compile(..., 'exec')`만 수행하여 `.pyc`도 만들지 않았다. Import/config parse PASS. Ruff `--select F,E9 --no-cache main/src/ltx_outpaint` PASS.
- Loader negative checks는 safetensors I/O를 **mock header + meta state**로 대체하고 실제 `load_adapter` 검증 분기를 호출했다. 실제 checkpoint payload를 읽거나 복원한 실험이 아니다.

### Source reference convention

아래 축약 prefix는 정확히 다음 repository-relative 경로를 뜻한다. 모든 `file:symbol:line` 인용은 현재 source 기준이다.

| Prefix | Exact path |
|---|---|
| `H/` | `main/src/ltx_outpaint/` |
| `C/` | `models/LTX2.3/packages/ltx-core/src/ltx_core/` |
| `T/` | `models/LTX2.3/packages/ltx-trainer/src/ltx_trainer/` |

`H/model.py:MultiViewSelfAttention.forward:233–296`처럼 symbol을 함께 기록한다. 과거 architecture 문서는 구현 사실의 근거로 사용하지 않았다.

## 2. Architecture Graph

### 2.1 실제 train call chain

```text
python -m ltx_outpaint.train
  H/train.py:main (967–1007)
    cudnn.benchmark = True (968)
    load_training_config / cache audit / Anchor inventory
    OutpaintingTrainer.__init__
      native LtxvTrainer.__init__
        _load_models → T/model_loader.load_transformer → LTXModelConfigurator → LTXModel
        _collect_trainable_params → custom _setup_lora → H/model.install_outpainting
    H/training_runtime.run_training(diagnostics_enabled=False)
      native LtxvTrainer.train
        custom OutpaintingTrainer._training_step
          LazyAnchorCache.complete_batch → send_to_device → constant empty condition
          native LtxvTrainer._training_step
            embeddings_processor.create_embeddings(video features, audio features, mask)
            OutpaintingStrategy.prepare_training_inputs
            LTXModel.forward(video=OutpaintingModality, audio=None, perturbations=None)
              OutpaintingPreprocessor.prepare → native preprocessor
              LTXModel._process_transformer_blocks
                checkpoint(block, video, audio, use_reentrant=False) × 48
                block._forward_with_native_gate → BasicAVTransformerBlock.forward
              native _process_output → video velocity prediction
            FlexibleStrategy.compute_loss → _compute_modality_loss
        accelerator.backward(loss.mean())
        accelerator.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step → zero_grad → scheduler.step
```

이 순서는 source 추적 결과이며 위 train 함수를 실제 호출하지 않았다. 기본 경로는 `H/training_runtime.py:run_training:129–144`, native backward/clip/optimizer는 `T/trainer.py:LtxvTrainer.train:225–243`이다. 10개 그룹은 **semantic/audit groups**이며 실제 AdamW가 10개 param-group를 구성한다는 뜻이 아니다.

### 2.2 실제 inference call chain

```text
H/inference.py:main → sample_long_range
  _load_models
    same T/model_loader.load_transformer → same LTXModelConfigurator/LTXModel
    same H/model.install_outpainting
    transformer.to(BF16) → same H/model.load_adapter → eval
  per recurrent chunk: sample_chunk [inference_mode]
    isolated Anchor encode (+ previous 17 RGB prefix encode)
    actual Anchor Rect → _known_mask / _canvas_geometry
    VideoConditionByMask → GaussianNoiser
    GeometryValidationRunner._Runner._modality_from_latent_state → attach_geometry
    native ValidationRunner._run_denoising
      X0Model(the same installed velocity transformer)
        LTXModel.forward → same 48 BasicAVTransformerBlock / MultiViewSelfAttention
      guidance / conditioning re-injection / Euler scheduler
    native decode → visible crop → known RGB composite → append only new frames
```

근거: `H/inference.py:GeometryValidationRunner.create:210–249`, `_load_models:326–410`, `sample_chunk:612–747`, `sample_long_range:750–903`; `T/validation_runner.py:_run_denoising:855–982`; `C/model/transformer/model.py:X0Model.forward:590–604`. X0 wrapper는 `sample − velocity × timestep` 변환이며 별도 학습 모델/head가 아니다.

### 2.3 한 block의 정확한 tensor 흐름

```text
block residual hidden X_i  [B, 73440, 4096]
  native AdaLN/RMSNorm → U_i  (current block hidden; GT/cache를 다시 읽지 않음)
    ├─ LOCAL, each original K4/K5 window:
    │    Native(U_i window): W_QKV + OP_QKV → native preattention/RoPE → SA core → N
    │    Adaptive(U_i): A>T>B / aligned support / C4-first promotion
    │      C2 + SAME per-block C4 → T-touching C2 Q, all C2/C4 KV
    │      W_QKV + OP_QKV + A_QKV → PMA + SA core
    │      R2 full footprint → T-only selection → Local-coordinate scatter → A_local
    │    where(M_T, lerp(A_local, N, local_gate), N)
    │    overlapping window index_add + coverage average → LocalFull
    └─ GLOBAL:
         SAME per-block C4 over full canonical 68×120 → 17×30
         W_QKV + OP_QKV + G_QKV → phase/RoPE + SA core
         R4 whole-grid bilinear + depthwise Conv3D → GlobalFull
  lerp(GlobalFull, LocalFull, global_gate)
  preserved native per-head gated-attention operation [frozen to_gate_logits]
  shared native attn1.to_out exactly ONCE: W_O + OP_O
  native SA residual + native vgate_msa
  frozen Text Cross-Attention (+ native CA AdaLN/gate) + residual
  shared native FFN: W + F-LoRA at two native Linears + residual
  X_(i+1)
```

중요: `X_i`가 raw residual hidden이라면 attention/C/R 입력은 **native pre-normalization을 거친 U_i**다. 현재 block hidden을 사용한다는 의미를 GT latent 직접 참조로 해석해서는 안 된다. 또한 실제 base metadata `apply_gated_attention=True`, `cross_attention_adaln=True`이므로 위 frozen native head gate/CA AdaLN을 생략한 개념도는 구현 전체가 아니다. 새 branch가 아니라 보존된 LTX 연산이다.

근거: `C/model/transformer/transformer.py:BasicAVTransformerBlock.forward:271–298,399–417`; `H/model.py:MultiViewSelfAttention.forward:247–296`; `C/model/transformer/ops.py:PytorchGatedAttention:94–106`.

## 3. Forward Connectivity Audit

| Stage | Actual source/symbol | Input → output / 실제 사용 확인 | Verdict |
|---|---|---|---|
| Native Local | `H/native_local_attention.py:native_local/attend:13–31` | Canonical window view + native PE subset; OP만 추가, A/G 없음; O 없음 | PASS |
| Adaptive Local | `H/adaptive_local_attention.py:adaptive_local/mixed_attention:13–93` | Current U_i를 C2/C4로 읽고 R2 후 original Local geometry로 복귀 | PASS |
| Local Fusion | `H/model.py:MultiViewSelfAttention.forward:258–276` | 두 map shape assert; Target에서만 lerp, 그 밖 Native 직접 반환 | PASS |
| LocalFull | `H/model.py:MultiViewSelfAttention.forward:278–286` | Canonical index_add / coverage count로 overlap 평균 | PASS |
| Global | `H/global_attention.py:global_attention:8–16` | Full grid C4 → OP+G SA → R4 반환값이 global_full에 사용 | PASS |
| L–G Fusion | `H/model.py:MultiViewSelfAttention.forward:287–289` | LocalFull/GlobalFull 모두 fused에 포함 | PASS |
| Native head gate + shared O | `H/model.py:MultiViewSelfAttention.forward:293–296` | Native head gating 후 `return self.to_out(fused)` 한 곳 | PASS |
| SA residual / Text CA | `C/model/transformer/transformer.py:forward:278–298` | SA 반환값을 native residual에 더하고 바로 attn2 호출 | PASS |
| FFN | 같은 `forward:399–417`; `H/model.py:install_outpainting:312–314` | shared ff 한 개, 두 Linear delta, FFN residual | PASS |
| Next block/final prediction | `C/model/transformer/model.py:_process_transformer_blocks/_process_output/forward:435–538` | 재할당된 video.x를 다음 block / frozen final head에 전달 | PASS |

### 3.1 Layout / Q / restore tracing

- `geometry.py:anchor_patch:71–85`는 전체 Anchor 중 현재 Local 크기의 **reference patch**를 선택한다. Intended bbox는 이 patch와 Local의 최소 union (`adaptive_layout:226–229`)이다. 전체 Anchor를 항상 bbox 안에 강제로 포함하지 않는다. Expanded support 안의 모든 실제 Anchor 픽셀에는 full Anchor rectangle로 A 우선권을 적용한다 (`resolve_adaptive_roles:171–178`).
- `align_adaptive_support:147–168`: 부족분 `(-extent)%4`; `start+end <= canvas_limit`이면 right/down, 아니면 left/up. 선호 방향의 가용분부터 쓰고 잔여만 반대편으로 확장한다. Origin을 global C4 lattice에 snap하지 않는다.
- `build_adaptive_multires_tokens:181–215`: bbox-local full4 parent, pure-B이면 C4 하나, A/T touch면 C2 네 개. B-only promoted subcell도 C2다.
- Anchor/Prefix known mask와 Adaptive write mask를 dense SA attention-mask로 혼동하지 않는다. 이 경로의 N/A/G attention callable에는 pairwise mask가 전달되지 않는다. 지원 restriction은 token 선택/Q 자격/최종 write로 구현되고, custom attn1은 비어 있지 않은 dense mask를 명시적으로 reject한다 (`H/model.py:forward:236–240`).
- Query 목록을 packed sequence 맨 앞에 배치하되 query 자격은 `TARGET in pixel_roles`이며 exclusive token label이 아니다. K/V는 전체 mixed sequence다. `mixed_attention:23–37`는 Q만 `:target_count`에 대해 projection한다. C4는 Q가 되지 않는다.
- `position.py:target_query_positions:52–84`: nominal P_k=origin/r_k+1/2, `(P_k−1/2)*(r_k/2)+1/2 = origin/2+1/2`. Q reference r_q=2; C2/C4 K의 r_k=2/4. Full-canvas origin과 native temporal bounds를 유지한다. Packed ordinal을 위치로 사용하지 않는다.
- `adaptive_layout:233–243`는 R2 full 2×2 결과의 `query*4+tap` → `(y−Local.y0)*Local.width+(x−Local.x0)` mapping을 **T인 tap만** 생성한다. 따라서 A/B와 Local 밖에 write하지 않는다.
- `adaptive_local:77–89`: `[B,Q,T,D] → [B*Q,T,1,D] → R2((1,1),(2,2)) → [B,T,Q*4,D] → T gather → [B,T,LocalH,LocalW,D]`. Read bbox 크기를 output Local 크기로 잘못 쓰지 않는다.
- `A T / T T`는 full C2 read + Q + full R2를 거친 후 T 세 tap만 선택된다. A-only C2는 KV-only, B/T mixed도 B output을 버린다.
- T empty이면 `geometry.py:adaptive_layout:223–225`에서 tokenization 자체를 건너뛰고 `adaptive_local:58–60`도 압축/QKV/R2를 호출하지 않는다. Zero map + false mask 때문에 Local은 Native 그대로다.

### 3.2 순수 geometry 검증 결과 (forward 없음)

| Partition | Local windows | Adaptive nonempty / early-exit | C2 spatial token range | C4 spatial token range | Target-Q spatial range |
|---|---:|---:|---:|---:|---:|
| K4 | 16 | 12 / 4 | 0–316 | 0–56 | 0–135 |
| K5 | 25 | 24 / 1 | 0–236 | 0–60 | 0–84 |

41 cases 전부: inside-canvas, support H/W multiple-of-4, A>T>B, read coverage exactly once, promotion4×C2, pure-B-only C4, full footprints, Q iff T-touching C2, write index exactly resolved T, unique write destinations PASS. Empty case는 support=None/Q=0이므로 H/W invariant를 허위 적용하지 않고 별도로 통과시켰다. Global `68×120→510 cells→17×30` PASS; `(67,120)/(68,119)/(5,5)` reject PASS.

K4 height=17,width=30; K5 height=14,width=24. K5 Y starts `0,14,27,41,54`, overlap rows `27,54`, X overlap 없음. K4 Y/X overlap 없음. 둘 모두 full canvas coverage, crop/padding 없음 (`geometry.py:partition_axis/local_windows:57–68`). 48 blocks는 K4 24개/K5 24개이다.

Unperturbed 48-block 1회 호출의 **정적 예상**은 Native windows 984개, nonempty Adaptive windows 864개, shared O/CA/FFN 각각 48회다. 이는 계수 계산이며 실제 forward counter 측정값이 아니다. Checkpoint replay/STG 별도 호출은 이 수에 포함하지 않는다.

## 4. Backward / Gradient Path Audit

### 4.1 공통 differentiable suffix

`parameter → branch context → local/global fusion → native head gate → (W_O + OP_O) → SA residual → Text CA residual → FFN residual → subsequent blocks → final output head → video_pred → loss`.

Frozen Linear/Conv weight는 **weight의 gradient 축적만 차단**한다. 입력 hidden에 대한 `dY/dX` 계산은 autograd-enabled forward에서 유지된다. Text CA를 `no_grad`로 감싼 코드가 없고, `C/model/transformer/transformer.py:forward:287–297`가 그대로 실행되므로 earlier adapter까지 gradient가 갈 수 있다. Final `norm_out/proj_out`도 `C/model/transformer/model.py:_process_output:472–490`에서 detach 없이 적용된다.

### 4.2 Local mask / fusion Jacobian

Actual formula (`H/model.py:forward:268–273`):

```python
local_weight = local_fusion(native_gate).to(native.dtype)
local = torch.where(write_mask, torch.lerp(adaptive, native, local_weight), native)
```

M_T=1이면 `L=(1-w)A+wN`: `∂L/∂A=1-w`, `∂L/∂N=w`, `∂L/∂w=N-A`.
M_T=0이면 `L=N`: Adaptive 및 Local gate의 해당 위치 contribution/gradient가 0이고 Native에는 identity 경로가 남는다. Bool mask 자체를 학습하지 않는 것은 의도이며 branch tensor gradient를 detach하지 않는다. 초기 w=.5일 때 Target에서 양쪽이 연결된다. Native=Adaptive이거나 gate가 극단값에 포화하면 gradient가 작거나 0일 수 있지만 이는 structural cut이 아니다.

Gate는 `sigmoid(beta+gamma*vgate_msa)` (`NativeGateFusion.forward:191–197`)이며 beta/gamma는 trainable. `vgate_msa`는 frozen native timestep/modulation 경로의 실제 결과를 사용한다. Frozen constant 입력이라고 해서 beta/gamma 학습이 끊기지 않는다.

### 4.3 Global fusion

`F=(1-w_g)GlobalFull+w_g LocalFull` (`H/model.py:forward:287–289`). 양쪽 context와 gate가 shared O 입력으로 연결된다. Global이 계산 후 버려지는 dead branch가 아니다. 여기서 전체 Anchor hidden까지 block별로 frozen시키는 설계는 아니다. Source preservation은 diffusion conditioning/loss와 최종 RGB composite의 계약이다.

### 4.4 Index / scatter / cast

- `index_select`, advanced gather, `reshape/permute`, `.float()`/`.to(dtype)`는 활성 tensor를 값으로 재생성하지 않는다. 선택된 source에 gradient가 연결된다.
- Packed `new_empty`는 처음엔 no-grad이지만 C2/C4 source로 **모든 destination을 한 번씩** `index_copy`하므로 결과 graph가 연결된다. 41-case coverage로 uninitialized token/누락/중복 destination을 배제했다.
- R2의 `index_copy`는 out-of-place이다. A/B output 삭제는 의도된 write ownership이고 A/B 입력을 통한 KV/context gradient는 여전히 가능하다.
- LocalFull의 `index_add_`는 fresh zero destination, count는 non-grad 별도 tensor다. Local source gradient는 누적과 평균을 통과한다. Source activation에 덮어쓰는 것이 아니다.
- `.to(BF16)`는 finite-precision rounding을 만들지만 detach가 아니다. 이 감사는 BF16에서 모든 gradient가 수치적으로 nonzero라고 주장하지 않는다.

## 5. Trainable Group Audit

### 5.0 전체 소유권 / actual meta count

실제 safetensors header의 `LTXModelConfigurator`를 사용했다. Dummy topology나 video-only 축소 model이 아니다. AV base의 비활성 audio weights도 존재하고 frozen count에 포함된다. Text encoder/VAE/connector의 외부 model parameter 수는 아래 **installed transformer total에 포함하지 않는다**.

| Group | Parameters | Parameter tensors | Train / inference parameter dtype |
|---|---:|---:|---|
| op_lora | 50,331,648 | 384 | FP32 / BF16 |
| a_lora | 75,497,472 | 288 | FP32 / BF16 |
| g_lora | 75,497,472 | 288 | FP32 / BF16 |
| f_lora | 62,914,560 | 192 | FP32 / BF16 |
| c2 | 983,040 | 96 | FP32 / BF16 |
| c4_shared | 3,342,336 | 96 | FP32 / BF16 |
| r2 | 5,505,024 | 96 | FP32 / BF16 |
| r4 | 5,505,024 | 96 | FP32 / BF16 |
| local_fusion | 393,216 | 96 | FP32 / BF16 |
| global_fusion | 393,216 | 96 | FP32 / BF16 |
| **Trainable total** | **280,363,008** | **1,728** | |
| Frozen transformer base | 18,988,838,144 | 4,186 | BF16 / BF16 |
| **Installed transformer total** | **19,269,201,152** | **5,914** | |

Meta train/inference modules: **6,032 each**, module-name/type mismatch 0, parameter-name/shape mismatch 0, state_dict key/shape mismatch 0. Original base Parameter identities and native O/FFN Linear object identities are preserved by installation. `named_parameters(remove_duplicate=False)` duplicate registration 0. Optimizer-selection list duplicate identities 0.

`H/model.py:install_outpainting:299–323` freezes base first, installs adapters/new modules, records `_outpainting_parameter_names`. `T/trainer.py:_collect_trainable_params:451–463` collects requires_grad; `_init_optimizer:704–722` sends that single flat list to AdamW. No per-group LR multiplier is introduced. `H/model.py:adapter_state_dict:342–344` and `load_adapter:347–370` cover all ten groups.

| Group | Forward used? | Loss-connected? | Gradient path structurally valid? | Optimizer member? | Saved? | Loaded in inference? | Verdict |
|---|---|---|---|---|---|---|---|
| OP | QKV in all views + final O | Yes | Yes | Yes | Yes | Same keys/objects role | PASS |
| A | Adaptive QKV | Yes, T-nonempty windows | Yes | Yes | Yes | Same | PASS |
| G | Global QKV | Yes | Yes | Yes | Yes | Same | PASS |
| F | Two shared FFN Linears | Yes | Yes | Yes | Yes | Same | PASS |
| C2 | Adaptive promoted full cells | Yes, T-nonempty windows | Yes | Yes | Yes | Same | PASS |
| shared C4 | Adaptive pure-B + Global | Yes, both branches can contribute | Yes | Once | Once | Same per-block owner | PASS |
| R2 | Target-query restore | Yes, retained T taps | Yes | Yes | Yes | Same | PASS |
| R4 | Whole Global grid restore | Yes | Yes | Yes | Yes | Same | PASS |
| Local Fusion | Target-only Local lerp | Yes, Target locations | Yes | Yes | Yes | Same | PASS |
| Global Fusion | Full-map Local–Global lerp | Yes | Yes | Yes | Yes | Same | PASS |

위 Yes는 source/meta 구조 판정이며 새 gradient 실측이나 실제 checkpoint payload 복원을 의미하지 않는다.

### OP-LoRA

- Creation: `H/model.py:MultiViewSelfAttention.__init__:214–223`; delta-only `LinearLoRA` (`H/lora.py:13–26`). A `[32,4096]`, B `[4096,32]` for each Q/K/V/O. Alpha/rank=1; bias/dropout 없음; FP32 master, Kaiming A/zero B.
- Native: `native_local_attention.attend:13–21` receives no specialization, so W+OP.
- Adaptive: `mixed_attention:25–30` explicitly uses same `attention.shared_outpainting_lora` plus A.
- Global: `global_attention:14` passes `attention.global_lora` to the same `attend`, yielding W+OP+G.
- O: native `to_out[0]` owns its delta and the original frozen weight; `lora.py:install_linear_lora:45–58` keeps Linear identity. Only `MultiViewSelfAttention.forward:296` invokes shared O.
- `project_with_loras:29–35` adds independently computed deltas of **the same x**, never sequential adapters. Base dtype is restored at return.
- OP Q/K/V objects are shared only within their block; every loop iteration constructs new objects. O delta is a separate projection delta in the same logical OP group, not a branch copy.
- Structural path: OP Q/K/V → attention → fusion, OP O → SA residual. No detachment.

### A-LoRA

- Creation: `H/model.py:__init__:219`; Q/K/V rank64, alpha64, A `[64,4096]`, B `[4096,64]`.
- Only `H/adaptive_local_attention.py:mixed_attention:25–38` accesses these projections; Native/Global never reference `adaptive_lora`.
- Q reads target-touching C2; K/V reads all cells. Attention output → R2 → T write → Local fusion preserves graph.
- Empty Target windows intentionally skip this branch. This is not an orphan parameter: same block's nonempty windows use the same A matrices.

### G-LoRA

- Creation: `H/model.py:__init__:220`; distinct rank64 QKV, not A aliases.
- Only `H/global_attention.py:global_attention:14` specializes `attend` with Global LoRA.
- C4 tokens → OP+G projections → Global SA → R4 → L–G fusion → output/loss.

### F-LoRA

- `H/model.py:install_outpainting:312–314`: only `ff.net.0.proj` and `ff.net.2`, rank32/alpha32. No extra branch-specific FFN.
- Actual FFN input Linear dimensions 4096→16384, output 16384→4096. Delta shapes respectively `[32,4096]/[16384,32]`, `[32,16384]/[4096,32]`.
- Original native Linear object/base weight retained; wrapper `H/lora.py:_forward_with_lora:38–42` calls `nn.Linear.forward` on self, avoiding recursive wrapper application.
- `C/model/transformer/feed_forward.py:FeedForward:6–15`, `gelu_approx.py:GELUApprox:4–10`: original GELU/shared FFN order; `transformer.py:forward:399–405`: FFN after CA and native FFN residual. No training bypass.

### C2

- `H/model.py:__init__:224–226`: separate per-block scale2 module.
- `H/compression.py:SpatialCompressor:12–44`: depthwise Conv2D kernel/stride2, padding0, groups4096; average-like init plus zero bias.
- Full current-hidden 2×2 canonical taps → gather/rearrange → actual `self.conv` → packed Adaptive sequence. Role A/T/promoted B does not change operator.
- No partial/masked convolution, area/valid-count multiplier, zero-filled input or fake padding. `valid_count` is a **validation property**, not normalization. 983,040 learned scalars including biases.

### shared C4

- Same ModuleDict `attention.compressors['4']` is used at `adaptive_local:64–72` and `global_attention:11–12`; no separately registered Adaptive C4.
- Meta identity checks for all 48 blocks confirm same object/weight/bias in both lookups, different from C2 and from every other block.
- Full4 only, Conv2D kernel=stride4/padding0/groups4096, same initialization and forward primitive. `geometry.py:global_cells/cells_for_rect:247–249,121–139` rejects nondivisible H/W before cells; compressor independently verifies exact in-canvas footprints.
- One autograd leaf can receive the sum of Adaptive-context and Global-path gradient contributions. It appears once in optimizer selection and checkpoint. 3,342,336 scalars.

### R2

- `H/model.py:__init__:227–229`: scale2 decompressor; **not shared with R4**.
- `adaptive_local:77–88`: Q cells treated as batch, T remains temporal, each query restored to complete2×2 via the same bilinear+DWConv3D primitive.
- `SpatialDecompressor.forward:58–70`: interpolation casts FP32 and back, then trainable refinement; no detach. Full time/spatial refinement remains differentiable.
- Afterwards only T taps selected/scattered. A/B read influence may affect target output; A/B **write** does not happen. R2's refiner gets loss through selected T taps; it does not have to reconstruct partial1×2/1×1 modes.

### R4

- Same `SpatialDecompressor` class but independent instance/parameters. Full Global `[B,T,510,D]` reshaped17×30, bilinear upsample68×120, depthwise Conv3D3³, then `[B,T,68,120,D]`.
- `H/global_attention.py:global_attention:15–16`; `H/compression.py:SpatialDecompressor:47–70`.
- This is **one fixed whole-grid restoration operation**, not 510 independent per-token decoders. R4 is never called on Adaptive C4 KV-only tokens. No partial/masked/edge mode selected.
- Initialization center tap1/off-center0/bias0 is installed in constructor; parameters remain trainable after installation. Init literal values are source-verified, not meta-value measured.

### Local Fusion

- Separate beta/gamma per block (`H/model.py:184–197,230`), initially zero by `.5` config, weight sigmoid(0)=.5.
- NativeGate ContextVar passes actual native `vgate_msa` without detach; forward uses beta/gamma and Target-masked lerp.
- Outside T: Native exact passthrough by expression, not `lerp(zero,Native,w)` attenuation. Overlapping assembly follows fusion, not a flat three-way fusion.

### Global Fusion

- Different beta/gamma instance (`H/model.py:231`), initializes Local/Global=.5/.5.
- `H/model.py:287–296`: gate is consumed in actual `fused` output → head gate → O. No dead/gated-off auxiliary output.
- Where Adaptive T is present, neutral nested contribution is N/A/G=.25/.25/.50 **before native head gating/O**. Where M_T=0, Local is Native, so the composition is N/G=.50/.50; .25/.25/.50 is not a universal Anchor-region mixture.

## 6. Block-to-Block Continuity

1. `LTXModel._process_transformer_blocks:435–468` assigns `(video,audio)` from each block to the next, without detach or tensor-from-values conversion.
2. `BasicAVTransformerBlock.forward:417` uses `dataclasses.replace(video, x=vx)`: object wrapping retains the original differentiable tensor reference.
3. `OutpaintingModality.split/attach_geometry:138–155` and `OutpaintingPreprocessor.prepare:164–181` only attach immutable geometry/positions; no hidden tensor detachment. Batch split is Tensor.split, not `.tolist()` reconstruction.
4. Only one checkpoint call site in active loop: **whole native block**, `use_reentrant=False`, enabled only while model.training. No additional checkpoint in Native/Adaptive/window loops. Default non-reentrant early-stop/RNG policy is not overridden.
5. Wrapper `_forward_with_native_gate:35–43` enters ContextVar, delegates to actual native block, and resets in `finally`, including checkpoint replay early-stop. `_capture_native_ada_values:46–54` obtains exactly native returned gate. Clearing ContextVar at attention entry discards a Python reference, not the tensor's autograd history.
6. `del local/weight/gate/...` statements free Python names after consumers hold graph dependencies; they are not detach operations.
7. Final frozen head normalizes/modulates/projects block47.x without no_grad. There is no frozen-to-frozen rule that cuts adapter-produced hidden gradients.

**Scope:** This proves structural continuity for the current eager unperturbed training path. It does not count runtime replay executions. `torch.compile` is explicitly rejected by the native gate wrapper; compile support is not claimed.

## 7. Loss Connectivity

Actual source:

- `H/train.py:OutpaintingStrategy.prepare_training_inputs:133–214` calls inherited noiser and returns video/audio=None, video_targets, video_loss_mask.
- `T/training_strategies/flexible.py:_initialize_noisy_target:449–462`:

```text
sigma = timestep_sampler.sample_for(latents)
noise = randn_like(clean_latents)
x_sigma = (1-sigma)*clean_latents + sigma*noise
v_target = noise - clean_latents
```

- Known spatial/temporal union replaces x_sigma with clean condition values; timestep0; loss_mask &= ~known. Sampling constants are non-trainable inputs, not detached adapter activations.
- Native `T/trainer.py:_training_step:393–408` passes **undetached video_pred** to strategy loss. Sigma detached for reporting at404 does not change loss.
- `FlexibleStrategy._compute_modality_loss:675–687` slices pred to targets (same full target length in this integration), squares error, multiplies broadcast mask, and computes:

```text
mean_{N,C}(M * (pred-target)^2) / max(mean_N(M), 1e-8)
= sum_{active tokens, C} error² / (active token count * C), if any active tokens
```

Thus known positions do not inflate the effective denominator. C=128 output latent channels, not D=4096 hidden channels. There is **no Native/Adaptive/Global/C/R auxiliary GT loss/head** in the active path; final velocity loss is the sole model objective. Loss value is not converted to CPU/NumPy before native backward (`T/trainer.py:train:232`). Post-backward official console detach is safe.

`loss_mask=0` excludes direct prediction supervision on known pixels, but those current-hidden positions may remain read context for generated targets and thereby receive activation gradients. Source protection does not mean no internal attention/hidden computation on Anchor/prefix.

## 8. Anchor / Prefix Contract

### 8.1 Spatial source

| Boundary | Training | Inference |
|---|---|---|
| Visible pixels | 3840×2160 requested | 3840×2160 requested |
| Internal VAE pixels | 3840×2176 | `_vae_boundary` symmetric top/bottom8 |
| Target latent | [B,128,9,68,120] | same full generation shape |
| Isolated Anchor latent | [B,128,9,34,60] cache | independent Anchor RGB encode + actual shape validation |
| Actual latent rectangle | [17:51),[30:90) from provenance/extent validation | same origin + **actual encoded H/W** |
| Spatial known convention | bool True=clean/loss-excluded | mask1=known; denoise_mask0=clean |
| Geometry | CanvasGeometry with spatial Rect | same CanvasGeometry with same Rect |

Training provenance and placement: `H/anchor_cache.py:LazyAnchorCache.complete_batch:621–648`; `H/train.py:_latent_anchor_rect:217–236`, `prepare_training_inputs:142–205`. Inference: `H/inference.py:_condition_latents:500–571`, `_known_mask:575–600`, `_canvas_geometry:603–609`. No stale end `[17:50)` is used in these active boundary paths.

Same rectangle result is consumed for inference placement, masks and Adaptive geometry. Training reads/validates the cache's actual extent before placing; no floor-computed end is reintroduced. Default source is an isolated clean Anchor, not GT outer conditioning. Training's full GT is used as velocity target and (when selected) known temporal prefix only.

### 8.2 Temporal source

- `data.py:OutpaintingCacheDataset.__getitem__:226–255`: FIRST `[0:65)` RGB / `[0:9)` latent, using official temporal mapping. No random future GT frames fill spatial cells.
- 17 RGB prefix → `(17−1)//8+1 = 3` latent slices. Train prefix probability=.5 independently sampled per batch element (`train.py:162–170`).
- Known = spatial Anchor OR temporal prefix. Prefix values override the intersection: train `torch.where(temporal, GT_latents, anchor_latents)`; inference first places Anchor then assigns previous-prefix latent to first3 slices (`inference.py:540–571`). Priority **A>T>B concerns spatial pixel ownership**, not a claim that isolated Anchor latent wins over temporal prefix values.
- Inference bootstrap has no prefix; subsequent chunks start at +48 RGB, take previous output's last17, append only new48. `_plan_long_range_chunks:140–196`, `sample_long_range:838–897`. Last short input repeats the last frame only for processing and appends only real remaining frames.
- Training prefix is teacher-forced clean GT latent. Inference prefix is previously decoded/generated full RGB (already Anchor-composited), then re-encoded. The conditional role is equivalent, the numerical latent/source distribution is not guaranteed identical.
- Inference adds absolute chunk time to native temporal position bounds (`sample_chunk:681`); training FIRST positions start at0 (`train.py:181–183`). **INFERENCE-ONLY intended recurrent offset**, no different PMA implementation. This exposes later absolute times and generated prefixes not present as such in FIRST-only training.

### 8.3 Known-mask static CPU evidence

| Prefix | Known tokens (of73,440) | Generated/loss tokens | Missing Anchor tokens | Row y=50 |
|---|---:|---:|---:|---|
| None | 18,360 | 55,080 | 0 | Known all9×60 |
| 17 RGB /3 latent | 36,720 | 36,720 | 0 | Known all9×60 |

`_known_mask` was called on CPU and compared exactly with an independent rectangular+prefix boolean union, not with a model/noiser. Corresponding active scalar loss elements: 7,050,240 /4,700,160. Pure lineage cases65,66,113,161,170 frames give1,2,2,3,4 chunks, gap=duplicate=0. No video encoding/generation ran.

### 8.4 Noiser / preservation source proof

- `C/conditioning/types/mask_cond.py:VideoConditionByMask.apply_to:36–49`: known1 → clean_latent assigned and denoise_mask0 at strength1.
- `C/components/noisers.py:GaussianNoiser.__call__:30–37`: lerp with denoise_mask0 selects clean_latent, so Anchor/prefix is not noise target.
- `T/validation_runner.py:_modality_from_latent_state:1261–1273`: token timestep=`denoise_mask*sigma`.
- `C/model/transformer/model.py:X0Model.forward:601–604`: known timestep0 converts velocity to unchanged input sample.
- `T/validation_runner.py:_run_denoising:954–977`: clean conditioning re-injection before Euler update. `H/inference.py:sample_chunk:717–724` additionally composites exact known RGB at output boundary.
- Adaptive A/T/B stays **spatial** in both train and inference: temporal prefix is not another 3D role map. Within known prefix timesteps outer spatial T may still have Adaptive hidden computation; mask/loss/noise contract preserves the known latent, not every block's internal hidden.

### 8.5 Fixed-empty Text CA / connector parity

Training `_load_models:291–324` freezes encoder/processor and stores video **and audio preconnector** features plus mask; `_training_step:538–555` supplies all three; native `trainer.py:381–387` calls `create_embeddings(video_features,audio_features,mask)`.

Inference `_cache_prompt_embeddings:319–363` is no-grad and calls `process_hidden_states`, which itself runs the same feature extractor then `create_embeddings` (`C/text_encoders/gemma/embeddings_processor.py:70–96,98–117`). Positive/negative are both empty; nonempty API/CLI prompt is explicitly rejected. Audio connector features are supplied, but generated audio Modality is None.

**Mask detail:** Train passes connector output binary mask to Text CA; inference uses `context_mask=None`. This is semantically all-valid for this actual connector, not an unexamined omission: official video connector uses **128 learnable registers**, substitutes padded features and returns a zero additive mask (`embeddings_connector.py:139–169`). `create_embeddings` converts it to all-one binary mask; native `_prepare_attention_mask:199–206` converts ones back to all-zero bias. Meta connector construction confirmed128 registers. This equivalence depends on this pinned connector contract, not arbitrary future no-register connectors.

Text CA itself is structurally active in both modes. Training `.requires_grad_(False)` on encoder/connector is explicit; inference text caching is protected by no_grad/inference_mode and unloaded after use. Do not confuse “no training of a temporary inference text encoder” with a claim that every temporary object's requires_grad flag was explicitly reset by its loader.

## 9. Train vs Inference Architecture Parity

Classification distinguishes module wiring from scheduler/autograd/sampling differences.

| Component | Train path | Inference path | Same module/code? | Same parameter semantics? | Classification / Risk |
|---|---|---|---|---|---|
| Base transformer/48 blocks | native load_transformer / LTXModelConfigurator | same factory/base metadata | Yes | Yes | SAME |
| Native | native_local/attend | same MultiViewSelfAttention call | Yes | W+OP | SAME for normal pass |
| Adaptive | adaptive_local/mixed_attention | same call | Yes | W+OP+A | SAME for normal pass |
| Global | global_attention | same call | Yes | W+OP+G | SAME for normal pass |
| OP/A/G | common installer | common installer then loader | Yes | Same per-block ownership/ranks | SAME; FP32→BF16 caveat |
| C2/shared C4 | common ModuleDict | same ModuleDict | Yes | same full-cell primitive/shared C4 | SAME |
| R2/R4 | common decompressor class | same class and call shapes | Yes | independent R2/R4 | SAME |
| Local Fusion | where(T,lerp(A,N,w),N) | same code | Yes | same beta/gamma/native gate | SAME |
| Global Fusion | lerp(G,L,w) | same code | Yes | same beta/gamma/native gate | SAME |
| shared O | after final fusion/head gate | same location | Yes | original Linear+OP-O | SAME |
| Text CA | native frozen attn2 + CA AdaLN | same native attn2 | Yes | unchanged base/no adapter | SAME; all-zero vs None mask representation |
| FFN/F | two native Linears+F | same | Yes | same shared FFN delta targets | SAME |
| Anchor | independent cache + checked actual extent | isolated RGB encode + actual extent | same downstream Rect/Geometry | same known role | CONFIG-DIFFERENCE input boundary; dimensions not enforced in inference |
| Prefix | probability.5;GT clean first3 | bootstrap then prior last17 encode | same downstream known role, different preparation | no new trainables | INFERENCE-ONLY recurrent source / TRAIN-ONLY teacher forcing |
| PMA/RoPE | same helpers/canonical positions | same helpers+absolute time offset | Yes | same constants/no learned positions | SAME algorithm; INFERENCE-ONLY temporal offset |
| Final output | velocity then masked MSE | same velocity then X0/guidance/Euler | same model/head | Yes | intended TRAIN-ONLY loss / INFERENCE-ONLY scheduler |
| Checkpoint replay | whole-block non-reentrant | no replay in eval | same block function | Yes | TRAIN-ONLY activation strategy |
| Autograd | enabled for trainable graph | inference_mode/no_grad | same installed architecture | Yes | intentional mode difference |
| STG | perturbations=None | block28 skip-self-attn auxiliary call | same model, **different selected control branch** | no new/absent tensors | INFERENCE-ONLY active guidance |

### 9.1 STG를 제외하고 모든 inference pass가 같다고 말하면 부정확하다

`sampling.stg_scale=1.0`, `stg_blocks=[28]`; `C/components/guiders.py:MultiModalGuider.calculate:259–267` explicitly adds `stg_scale*(cond−perturbed)`, and `do_perturbed_generation:279–281` is true when scale!=0. Thus **STG=1 is not neutral/off**.

`ValidationRunner._run_denoising:927–937` creates this auxiliary pass. In `H/model.py:MultiViewSelfAttention.forward:243–246`, `all_perturbed=True` uses base V+OP-V and bypasses local/adaptive/global attention and two fusions, then still runs preserved native head gate/shared O/residual/CA/FFN. Conditional and empty-negative CFG passes have perturbations=None and use the full architecture. This is configured guidance, not a silently different inference model or failed checkpoint install.

If “all inference invocations must execute every N/A/G branch identically to train” is an absolute acceptance criterion, **that stricter statement does not hold under the current STG config**. The audited contract that the main velocity-model path/installed parameters are the same is satisfied. No config was changed.

### 9.2 Kernel/numeric limits

Meta construction was CPU-hidden-GPU and therefore does not identify actual CUDA attention dispatch. Both entrypoints inherit the same native attention factory/ops rather than duplicate attention math. Train's all-zero CA mask versus inference None can also affect kernel choice even when semantic masking is equivalent. No performance or bitwise numerical parity claim follows from this static audit.

## 10. Parameter Key / State Dict Parity

### 10.1 Installation → optimizer → save → inference

1. `_setup_lora` in `H/train.py:326–333` overrides official PEFT setup; ArchitectureConfig is the single construction source. Top-level LoRA descriptor is not a second wrapper.
2. `install_outpainting:322` records all and only new requires_grad named tensors. Meta count1,728. Group classifier and registration audit found no old Native/attn2 adapter names.
3. `_setup_accelerator:343–360` saves adapter-only state via hook and clears base-weight serialization; `_save_checkpoint:654–735` writes those keys in FP32 plus architecture/base/source metadata. Optimizer/scheduler/RNG are separate resume state.
4. Inference `_load_models:397–409` installs same ArchitectureConfig before `load_adapter`; shapes/keys therefore match expected tensor destinations.
5. `load_adapter:352–369` requires matching base path, full architecture dict, source provenance, **exact key set**, shape and floating dtype; only then calls `load_state_dict(strict=False)`.
6. `strict=False` intentionally permits omitted frozen base tensors because this is adapter-only. It does **not** silently permit a missing/unexpected trainable: explicit prior set equality rejects it. Destination identity is unique.

### 10.2 Header evidence (actual pre-existing checkpoint)

Read JSON header only:

`outputs/final_runtime_validation/training_smoke/validated_run/checkpoints/checkpoint-000005/adapter_model.safetensors`

- Stored tensor keys1,728, all F32.
- Current train expected ↔ current inference expected ↔ stored key set: exact.
- Full architecture metadata (version8), configured base path, provenance: exact match.
- Missing=0; unexpected=0; shape mismatch=0; duplicate destination=0.
- Payload bytes were **not** loaded. Therefore saved↔loaded numerical equality and file payload finite status were not tested here.

| Category | Keys | Missing | Unexpected | Shape mismatch | Duplicate destination |
|---|---:|---:|---:|---:|---:|
| op_lora | 384 | 0 | 0 | 0 | 0 |
| a_lora | 288 | 0 | 0 | 0 | 0 |
| g_lora | 288 | 0 | 0 | 0 | 0 |
| f_lora | 192 | 0 | 0 | 0 | 0 |
| c2 | 96 | 0 | 0 | 0 | 0 |
| c4_shared | 96 | 0 | 0 | 0 | 0 |
| r2 | 96 | 0 | 0 | 0 | 0 |
| r4 | 96 | 0 | 0 | 0 | 0 |
| local_fusion | 96 | 0 | 0 | 0 | 0 |
| global_fusion | 96 | 0 | 0 | 0 | 0 |

Representative prefixes (`i=0…47`):

```text
transformer_blocks.i.attn1.shared_outpainting_lora.{q,k,v}.lora_{A,B}.weight
transformer_blocks.i.attn1.to_out.0.shared_outpainting_lora.lora_{A,B}.weight
transformer_blocks.i.attn1.adaptive_lora.{q,k,v}.lora_{A,B}.weight
transformer_blocks.i.attn1.global_lora.{q,k,v}.lora_{A,B}.weight
transformer_blocks.i.ff.net.0.proj.ffn_lora.lora_{A,B}.weight
transformer_blocks.i.ff.net.2.ffn_lora.lora_{A,B}.weight
transformer_blocks.i.attn1.compressors.{2,4}.conv.{weight,bias}
transformer_blocks.i.attn1.decompressors.{2,4}.refine.{weight,bias}
transformer_blocks.i.attn1.{local_fusion,global_fusion}.{beta,gamma}
```

Full key/shape listing is `outputs/static_gradient_parity_audit/20260912/adapter_key_shapes.json`.

### 10.3 Fail-closed mock/meta tests

실제 `load_adapter` 함수에서 safe_open/load_file만 mock한 결과, model forward 없이:

| Test | Result |
|---|---|
| Matching v8 metadata + matching meta-state | ACCEPT |
| v6 / v7 metadata → v8 | REJECT / REJECT |
| Wrong architecture field | REJECT |
| Wrong provenance | REJECT |
| Wrong base_model_path | REJECT |
| Missing trainable key | REJECT |
| Unexpected key | REJECT |
| Tensor shape mismatch | REJECT |
| Nonfloating tensor | REJECT |

No real or fake checkpoint file was created. Matching case loads **meta placeholders**, not real trained values. ArchitectureConfig constructor independently rejects v6/v7/partial policy.

### 10.4 Resume vs inference compatibility

`H/train.py:_load_checkpoint:362–427`, `_init_optimizer:429–453` perform additional resume checks: complete state files, architecture/provenance, config fingerprint, world size/type, seed, batch/accumulation, FIRST policy, counters/order. A model-compatible old smoke adapter does **not** imply that its full run can resume under changed current seed/30,000-step config. Inference does not need optimizer state. Do not confuse deliberate resume rejection with an adapter key mismatch.

### 10.5 Limits

- Inference casts installed modules BF16 **before** loader copies FP32 values. Key/shape match does not imply bitwise preserved FP32 adapter parameters. C/R/Fusion arithmetic also runs under inference BF16 policy; training FP32 master policy remains different.
- Adapter loader validates floating type/shape but has no explicit NaN/Inf payload scan. Header match alone cannot certify contents. This is a validation boundary, not evidence of corrupted checkpoint.
- Provenance comparison includes declared SHA/revision; this task did not rehash tens of GB base weights. Production `checkpoint_provenance` (`H/outpainting_adapter.py:164–185`) performs its own source/base validation when loading.
- Configured default `runs/hrlrvo_v8/checkpoints/checkpoint-030000/adapter_model.safetensors` **does not currently exist**. Loader raises; there is no silent fresh-initialization fallback. Existing step-5 header fit does not manufacture a production checkpoint.

## 11. Config Parity

Parsed `main/configs/train_4k.yaml` through actual `load_training_config` and inference YAML through its actual architecture alias normalization. All ArchitectureConfig fields match after `local_partition_k→k`, `compression_scale→scale` (`H/inference.py:_load_models:348–351`). Unknown fields cannot silently become architecture attributes; dataclass keyword construction rejects them.

| Field | Train | Inference | Classification |
|---|---|---|---|
| Architecture version | 8 | 8 | SAME |
| Base checkpoint/source | LTX-2.3-22b-dev / pinned revision598ab412… | same path/provenance | SAME |
| Blocks / hidden dim | base metadata48 /4096 | same metadata | SAME |
| OP QKVO | r32 α32 | r32 α32 | SAME |
| A /G QKV | independent r64 α64 each | same | SAME |
| F | shared FFN r32 α32 | same | SAME |
| Native LoRA /Text CA LoRA | absent /absent | absent /absent | SAME |
| C2/C4 | full2 /shared full4 | same | SAME |
| R2/R4 | per-query full2 /wholeGlobal-grid | same | SAME |
| K /Global scale | k=4 (block별 4+(i%2)) /scale=4 | aliases normalize to4 /4 | SAME |
| Adaptive roles/support/restore | A>T>B, center-expand, C4-first, T-only | same | SAME |
| PMA | pma_v2_nominal_grid | same | SAME |
| Fusion init | Local.5 /L-G.5, native-vgate | same | SAME |
| Refinement init | depthwise_identity | same | SAME |
| Visible/internal | [2160,3840] H/W + cache2176 | [3840,2160] W/H + VAE2176 | SAME normalized geometry |
| Clip/prefix/stride | 65 /17 /implicit generation48 | 65 /17 /48 | SAME temporal contract |
| Prefix use | Bernoulli.5, GT prefix | none bootstrap, then recurrent prefix | intended difference |
| Anchor | exact1080×1920 cache | expected1920×1080, enforcementFalse | POTENTIAL-RISK for other inputs |
| Positive/negative text | empty/empty | empty/empty | SAME |
| Audio generation | None | None | SAME |
| Parameter precision | frozenBF16/newFP32 masters, mixedBF16 | allBF16 at eval | CONFIG-DIFFERENCE |
| Grad checkpoint | ON in train | skipped in eval | TRAIN-ONLY |
| Seed | 33333333 | 42 | CONFIG-DIFFERENCE, not weight topology |
| Noise/scheduler | stable shifted-logit-normal /velocity loss | 30-stepLTX2Scheduler/Euler | intended train/inference difference |
| Guidance | no perturbation in loss call | CFG3/STG1 block28/rescale.7 | INFERENCE-ONLY |
| Optimizer | AdamW, common1e-4, LinearLR→1e-5 over30,000 | none | TRAIN-ONLY |
| Batch/accum/clip | 1 /1 /maxnorm1 | no optimizer | TRAIN-ONLY |
| Checkpoint cadence | 500 optimizer steps; FP32 adapter/full resume | load specified adapter | TRAIN-ONLY state vs inference weights |
| cuDNN benchmark | enabled in train.main | no corresponding explicit assignment | runtime backend policy difference only |

Main configs retain some comments/error strings calling the LoRA arrangement “v6” (`train_4k.yaml:lora` comment; `H/train.py:_validate_integration_config:826–840`; `H/model.py:__post_init__/parameter_counts`). Actual architecture_version and metadata are8. This is stale wording, not a silent old architecture load. No wording/source was changed in this audit.

## 12. Detach / In-place / Autograd Risk Audit

AST scan covers every `H/*.py` and relevant pinned transformer/trainer/conditioning/connector/noiser/scheduler source: **54 files**, **323 candidate sites** including initialization and Python dict/counter false positives. Every candidate is classified with symbol/line/snippet/rationale in `autograd_pattern_inventory.json`. This is supplemented by direct forward/loss call tracing; grep absence alone is not the proof.

| Site/family | Classification | Actual effect / why not a broken training path |
|---|---|---|
| `H/adaptive_local_attention.py:adaptive_local:61–92` tensor(indices), index_copy/index_select | SAFE | Indices from immutable integer metadata; packed/local/result source gradients retained; bool mask writes only metadata |
| `H/model.py:forward:284–285` index_add_ | SAFE | Fresh LocalFull accumulator + independent no-grad counts; not in-place modification of branch source |
| `H/model.py:NativeGateFusion.forward:197` sigmoid_ | SAFE | Fresh addcmul output, no consumer needing pre-sigmoid alias |
| `H/compression.py:forward:41–44` gather/index/cast | SAFE | Full current-hidden source, no existing tensor copied via torch.tensor |
| `H/compression.py:SpatialDecompressor.forward:64–70` casts/permute | SAFE | Differentiable interpolate/refine/casts; no host conversion |
| `H/lora.py:LinearLoRA.forward/project_with_loras:23–35` dtype casts/add | SAFE | Same x for independent deltas; base-result cast retains autograd |
| `H/position.py:target_query_positions/phase_rope:69–84,117–159` tensor/clone/assignment | SAFE | Canonical positions/frequencies, not learned activations; index_select owns edited output |
| `C/model/transformer/rope.py:apply_split_rotary_emb:66–84` addcmul_ | SAFE | Disjoint halves of fresh output; original Q/K views not overwritten |
| `H/model.py:gate ContextVar:35–54,241–257` reset/del | SAFE | Python reference lifetime, not tensor detach; finally also handles replay early-stop |
| `C/model/transformer/model.py:_process_transformer_blocks:461` checkpoint | SAFE | Single whole-block use_reentrantFalse call; no nested branch checkpoint |
| `H/train.py:_load_models:291–324` inference_mode/cpu | EXPECTED | Frozen text feature preparation only; not Text CA hidden forward |
| `H/train.py:prepare_training_inputs:159–179` mask assignment/where | SAFE | Bool source mask and input preparation. Prediction is created later and not detached |
| `H/anchor_cache.py:_tensor_digest/encode_prepared/complete_batch:65,473–500,621–648` detach/cpu/numpy/assignment | EXPECTED | Cache checksum or frozen VAE/source input pipeline; no trainable downstream graph exists yet |
| `H/data.py:__getitem__/collate:200–278` CPU loading/crop/dicts | EXPECTED | Cached input tensors, not model activation output; no trainable preprocessing model |
| `T/trainer.py:train/_training_step:276–288,402–408` detach/cpu | SAFE | Loss logging은 post-backward, sigma.detach는 _training_step 반환 전 reporting metadata 처리; output.loss itself remains on graph |
| `H/train.py:_save_checkpoint:699` detach/to_cpu | SAFE | Separate serialization snapshot; original trainable Parameter remains installed |
| `H/health_logging.py` observer/grad/snapshot code | EXPECTED, inactive | Opt-in-only; detached observations not returned as forward tensor |
| `H/training_runtime.py:run_training_with_health:186–390` scan/clone/wrapper code | EXPECTED, inactive | default diagnosticsFalse → direct native train; no hooks/observer graph references |
| `H/inference.py:sample_chunk/sample_long_range:612,750` inference_mode | EXPECTED | Inference intentionally has no backward; same underlying model source remains trainable in training context |
| `H/inference.py:681,722–747,875` position offset/composite/cpu/clone | EXPECTED | Pure inference constants/final export/recurrent prefix boundary; not training hidden detach |
| `T/validation_runner.py:_cache_prompt_embeddings` no_grad /cpu | EXPECTED | Frozen text conditioning cache, not a train-time Text CA bypass |
| `C/model/transformer/modality.py:split`, `transformer.py:forward:417` split/replace | SAFE | Tensor references/views preserved through dataclass wrappers |
| `C/model/transformer/transformer_args.py:_prepare_self_attention_mask:208–235` bias assignment | EXPECTED, inactive for SA | Current Modality.attention_mask isNone. Non-None dense SA masks are rejected by custom attn1; not secretly learned |
| `C/model/transformer/cudagraph_capture.py` static.copy_ | EXPECTED, inactive | No CudaGraphRunner installed; eager gate explicitly rejects compile |
| Other pinned intrinsic/reference/keyframe masks, Res2s, optimizer-offload branches | EXPECTED, inactive | Current custom OutpaintingStrategy bypasses extra condition preparation; infer uses mask/Euler; configured offloadFalse |
| `.data` hits | SAFE false positives | Config.data field accesses, not Tensor.data |
| nn.init.*_/requires_grad_(False) | SAFE | Constructor/freeze phase; deliberately no graph exists yet |

**No active hidden `.detach()/.clone().detach()/.cpu()/.numpy()/torch.tensor(hidden)/Tensor.data` cut found.** No explicit `retain_graph=True` or `create_graph=True` in the traced production train path. No additional custom autograd.Function implementation is introduced in Hrlrvo's common graph. Generic pinned optional implementations are not assumed active simply because a file exists.

The raw AST scan is not a formal proof of all native CUDA kernels; it resolves the requested source-level operations and flags control/config boundaries. **RISK** is reserved for the concrete parity caveats in §15, not assigned to every harmless tensor operation.

## 13. Dead / Duplicate / Shared Parameter Audit

### 13.1 Registration evidence

- All1,728 requires_grad tensors match `_outpainting_parameter_names` exactly; every key classified into one of the ten intended groups.
- Every original base parameter still refers to the same Python Parameter object and is frozen after installation; no frozen19B base copy added into an adapter wrapper.
- All5,914 named parameter identities are unique when `remove_duplicate=False`; duplicate optimizer identities0. Non-parameter stateless attention callables may be shared across blocks by native factory; that is not learned-weight sharing.
- No `.native_lora` and no `.attn2.*lora` name in either meta model.
- Per-block OP QKV reused across N/A/G by same attention object access; OP O and FFN LoRAs each own their distinct projection-specific delta.
- A/G are different modules; C2/C4 different; R2/R4 different; Local/Global gates different; no parameter IDs shared between blocks.
- Adaptive/Global C4 is the single desired intra-block alias at **call-site lookup**, with only one registered name. It is not two registered copies of the same weights.

### 13.2 Forward usage vs optimizer membership

Counts alone do not prove usage. §3–5 trace the actual referenced members to outputs, and the native optimizer source collects exactly those requires_grad tensors. Thus no group is merely installed/saved but never called. Native frozen AV audio branches remain structurally present and dormant with `audio=None`; they are not dead **trainables**.

### 13.3 Conditional zero is not dead

- Scratch 초기화에서 B=0인 각 실행된 LoRA의 최초 backward는 수학적으로 dA=0이고, dB는 upstream이 있으면 가능하다. Skip된 경로는 grad=None일 수도 있다. Nonzero-B checkpoint를 load/resume한 경우와 구분하며, 이는 연결 단절이 아니다.
- Completely known Local windows skip Adaptive C/R/A projections. **Local gate는 여전히 계산되지만**, false write_mask의 torch.where가 Native를 선택하므로 해당 위치에서 gate/Adaptive의 output contribution과 gradient가 0이다 (`H/model.py:forward:260–273`). Canonical41 cases include5 such windows (K4:4,K5:1); other windows in the same block consume the same parameters.
- A pure-B C4 context is absent in some Adaptive layouts. Shared C4 still has its full-canvas Global path.
- 선택되지 않은 R2 A/B output tap 자체에는 write loss gradient가 0이다. A/B 입력의 간접 영향은 **그보다 upstream의 full-cell compression 및 Adaptive Q/K/V read 경로**에서 발생한다. Discarded R2 output을 통과하여 context gradient가 생기는 것은 아니다.
- Fusion gradient depends on N−A or L−G and actual vgate; equal contexts or saturated gates may givezero.
- STG-selected auxiliary inference calls bypass some modules, but normal training always supplies perturbations=None and full graph.

## 14. benchmark=True Impact

Exact current startup change:

```python
# main/src/ltx_outpaint/train.py:main:967–969
def main() -> None:
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description=__doc__)
```

The assignment is at the start of main, before config/artifact preparation, cache/VAE/model construction and any forward. Importing `ltx_outpaint.train` alone does not call main; this static audit did not switch the flag or execute convolution. Source search found no later `cudnn.benchmark=False` assignment in the active Hrlrvo/pinned training path. `accelerate.utils.set_seed` calls deterministic algorithms only when its optional deterministic flag is explicitly true; Hrlrvo supplies seed only.

| Question | Static result |
|---|---|
| Module/branch definition changed by the flag? | No |
| Trainable/frozen ownership? | No |
| state_dict key/shape or checkpoint metadata? | No |
| Forward equations/loss/sigma/prefix/masks? | No |
| Inference architecture installation? | No |
| Conv algorithm selection/reduction order/workspace? | May differ; runtime backend policy, not graph topology |
| Bitwise training trajectory guaranteed unchanged? | **Not guaranteed**; different CUDA algorithm/reduction order may round differently |

No speedup/kernel claim was remeasured in this static audit. Current request is not permission to repeat the prior benchmark or restart production.

## 15. Risks / Remaining Caveats

| Priority | Finding / exact source | Type | Impact / interpretation |
|---|---|---|---|
| 1 | `H/inference.py:_load_models:405–407`: all parametersBF16 before load | CONFIG-DIFFERENCE | Same weights' conceptual mapping but FP32 precision not preserved. Do not claim exact FP32 saved↔inference values or bitwise train/eval results |
| 2 | `inference_4k.yaml:sampling.stg_scale=1.0`; `H/model.py:forward:243–246`; `C/components/guiders.py:264,279–281` | INFERENCE-ONLY | Active STG block28 bypass; ordinary model pass same, every auxiliary inference pass not identical. No silent branch loss |
| 3 | `inference_4k.yaml:video.enforce_expected_anchor=false`; `H/inference.py:sample_long_range:790–801` | POTENTIAL-RISK | Other Anchor sizes may change geometry or fail anchor_patch/global constraints. Standard2K Anchor case is matched; arbitrary-input parity is not certified |
| 4 | `H/train.py:prepare_training_inputs:175–182`; `H/inference.py:sample_chunk:548–571,681`; `sample_long_range:875` | Intended conditioning/distribution difference | GT teacher forcing vs generated/reencoded prefix, zero vs absolute temporal origin. Same known/t0/17f semantics; no quality/exposure guarantee |
| 5 | default inference adapter path currently absent | Runtime prerequisite | Explicit existing checkpoint needed. Not a model-key bug; no runtime executed or checkpoint fabricated |
| 6 | `H/model.py:load_adapter:367–370` checks shape/floating but not payload finite | Validation boundary | Header-compatible checkpoint could still contain bad numerical values. No such corruption observed or claimed; this audit reads headers only |
| 7 | `position.py:phase_rope:146–149` partial-cell comment; several v6-named error strings/comments | MINOR stale wording | Current active Global full4 /Adaptive target_query_rope rules do not execute legacy partial semantics. Could confuse future maintenance; no code changed |
| 8 | native gate wrapper `H/model.py:35–38` rejects compile | Explicit compatibility limit | Current eager/checkpoint path valid; not automatically compatible with future torch.compile |

**Actual source-level broken gradient BUG found: none.** These caveats must not be relabeled as measured failures. Conversely, static success must not be relabeled as a fresh runtime/numerical PASS. Proposed fixes/optimizations are outside scope and were not applied.

Additional limits: no real checkpoint tensor values read, no actual baseweight SHA recalculated, no encoder/VAE execution, no optimizer object/step, no dynamic autograd graph/saved-tensor inspection, no custom forward/gradient hooks installed in production. Python meta construction verifies shape/identity but cannot detect data-dependent runtime anomalies.

## 16. Final Verdict

### Q1–Q14 explicit answers

| Question | Answer |
|---|---|
| Q1. All intended trainables used in forward? | **Yes**, all10 groups, conditional on intended Target/normal unperturbed pass; §3/5/13 |
| Q2. Every group structurally loss-connected? | **Yes**, traced suffix to final masked velocity loss; not a claim of nonzero every individual tensor/step |
| Q3. Detach/no_grad/in-place/index/scatter cuts? | **No unintended cut found** in active training path; intentional write masking and frozen conditioning are distinguished |
| Q4. Global contributes to prediction? | **Yes**, R4→GlobalFusion→sharedO→residual→head/loss |
| Q5. Adaptive contributes? | **Yes**, R2→T-only scatter→LocalFusion→LocalFull→GlobalFusion→prediction |
| Q6. C2/C4/R2/R4 loss-connected? | **Yes**; same sharedC4 leaf receives both branches' possible contributions |
| Q7. Fusion parameters loss-connected? | **Yes**, beta/gamma consumed in differentiable sigmoid/lerp; Local only onT |
| Q8. All48 blocks continuous? | **Yes**, dataclass reference propagation and one non-reentrant whole-block checkpoint site |
| Q9. Train/inference install samev8? | **Yes**, same actual configurator/installer, full meta name/shape/type equality |
| Q10. Saved vs inference expected keys match? | **Yes**, existing step5 header1,728 keys, all mismatch/duplicate counts0; fail-closed guard tests PASS |
| Q11. Train/inference divergence? | **No duplicated architecture implementation**; **yes intentional execution differences**: eval/no-grad, checkpoint replay off, BF16cast, CFG/STG, sampling boundary. STG auxiliary block28 does not run all branches |
| Q12. Anchor/Prefix contract same? | **Yes for canonical 2K→4K geometry and17f semantics**, knownunion/t0/loss/injection consistent. Prefix values/absolute origin differ, arbitrary Anchor geometry not enforced |
| Q13. benchmark=True changes graph? | **No graph/architecture/ownership/key/loss change**; CUDA algorithm/numerical order may change |
| Q14. Actual BUG/RISK? | **No active graph-cut/dead-parameter/key BUG found**; §15 lists concrete precision/STG/input/prerequisite caveats |

### STATIC AUDIT VERDICT: **PASS WITH CAVEATS**

**Production changes made by this audit: none.** No training/inference/optimizer/profiler/GPU execution, no new checkpoint, no production restart. The prior single benchmark startup assignment remains untouched.

### Reproducible evidence files

All ancillary evidence is under:

`/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/outputs/static_gradient_parity_audit/20260912/`

- `check_static_contract.py`: guarded real48-block meta installation, enumeration, header comparison, pure geometry/mask/lineage/import/config/syntax checks.
- `check_loader_guards.py`: mock-header/meta-state calls of real loader rejection guards; no checkpoint file creation or payload read.
- `meta_and_geometry_results.json`, `adapter_key_shapes.json`, `loader_guard_results.json`.
- `autograd_pattern_inventory.json`: per-site file/symbol/line/classification/rationale.
- `source_hashes_before.json`, `source_hashes_after.json`:339protectedfilesunchanged.
- `static_check.log`, `loader_guards.log`, `ruff.log`.

Core deliverable:
`/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/STATIC_GRADIENT_AND_TRAIN_INFERENCE_PARITY_AUDIT.md`
