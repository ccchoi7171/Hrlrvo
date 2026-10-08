# Hrlrvo v8 학습 사양 — 최신 코드 정적 감사

## 0. 문서 범위와 판정 기준

이 문서는 **현재 작업 트리의 production 코드**를 직접 읽어 정리한 학습 사양이다. 과거 validation report나 README의 서술을 근거로 사용하지 않았다. 기준 구현은 다음과 같다.

- Hrlrvo 진입점: [`../../src/ltx_outpaint/train.py`](../../src/ltx_outpaint/train.py)
- 모델 설치/계층: [`../../src/ltx_outpaint/model.py`](../../src/ltx_outpaint/model.py)
- 데이터: [`../../src/ltx_outpaint/data.py`](../../src/ltx_outpaint/data.py)
- 독립 Anchor cache: [`../../src/ltx_outpaint/anchor_cache.py`](../../src/ltx_outpaint/anchor_cache.py)
- Local/Adaptive/Global: [`../../src/ltx_outpaint/native_local_attention.py`](../../src/ltx_outpaint/native_local_attention.py), [`../../src/ltx_outpaint/adaptive_local_attention.py`](../../src/ltx_outpaint/adaptive_local_attention.py), [`../../src/ltx_outpaint/global_attention.py`](../../src/ltx_outpaint/global_attention.py)
- 현재 config: [`../../configs/train_4k.yaml`](../../configs/train_4k.yaml)
- pinned native trainer: [`../../../models/LTX2.3/packages/ltx-trainer/src/ltx_trainer/trainer.py`](../../../models/LTX2.3/packages/ltx-trainer/src/ltx_trainer/trainer.py)
- pinned native strategy/loss: [`../../../models/LTX2.3/packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py`](../../../models/LTX2.3/packages/ltx-trainer/src/ltx_trainer/training_strategies/flexible.py)
- pinned native block: [`../../../models/LTX2.3/packages/ltx-core/src/ltx_core/model/transformer/transformer.py`](../../../models/LTX2.3/packages/ltx-core/src/ltx_core/model/transformer/transformer.py)

이번 감사에서는 model forward, backward, optimizer step, VAE encode, GPU 작업 및 테스트를 실행하지 않았다. 따라서 아래의 `PASS`는 **정적 코드 계약의 일치**를 뜻하며, 실제 장치 실행 성립 여부는 `INCONCLUSIVE (미실행)`로 구분한다.

## 1. 식별자와 고정 구성

| 항목 | 현재 값 | 코드 근거 | 정적 판정 |
|---|---:|---|---|
| architecture version | **8** | `ArchitectureConfig.architecture_version`, config `outpainting.architecture` | PASS |
| base | LTX-2.3 22B dev | `model.model_path` | PASS |
| pinned source revision | `598ab41247a77dbfe29b5186e915bcf4f9040ec7` | `outpainting.provenance` | PASS |
| latent canvas | `T=9, H=68, W=120` | 65-frame/2176×3840 VAE mapping 및 cache contract 검증 코드 | PASS |
| canonical token count | `9×68×120 = 73,440` | patch size 1 | PASS |
| hidden dimension | `D=4096` | pretrained LTX video pathway | PASS (checkpoint identity 전제) |
| Local partition | even block K=4, odd block K+1=5 | `MultiViewSelfAttention.partition_count = k + block_index % 2` | PASS |
| compression scale | S=4, promoted fine scale=2 | `ArchitectureConfig.scale`, Adaptive C4-first policy | PASS |
| text | positive empty string, external condition cache 없음 | `constant_prompt: ""`, `conditions_dir: null` | PASS |
| generated modality | video only | `training_strategy.video.is_generated=true`, `audio=None` validation | PASS |
| temporal sample | FIRST only: RGB `[0:65)`, latent `[0:9)` | dataset/cache request assertions | PASS |

`ArchitectureConfig.__post_init__()`는 version 8, C4-first/C2 promotion, Target-touching query-only, nominal-grid PMA, Target-masked Native passthrough, LoRA 구성, S=4를 fail-closed로 검증한다. 내부 오류 문자열 일부가 역사적 명칭인 “v6”를 사용하지만 실제 비교 값과 checkpoint identity는 version 8이다. 이는 메시지 표기상의 잔여 흔적이며 현재 parameter/layout 검증을 완화하지 않는다.

## 2. 데이터와 cache 계약

### 2.1 4K GT target

`main()` → `read_video_scale_factors()` → `audit_ltx_cache()`가 기존 cache를 읽기 전용으로 감사한다.

- root: `/home/work/ai-graphics_33/Hrlrvo/cache_Data/train`
- official record fields: `latents`, `num_frames`, `height`, `width`, `fps`
- full record tensor: `[C=128, T_full, H=68, W=120]`, BF16
- 학습 slice: `[128, 9, 68, 120]`
- `map_temporal_slice(0, 9, ..., temporal_scale=8)`로 RGB `[0:65)`와 대응
- cache 자체를 overwrite/re-cache하는 경로는 없다.

`OutpaintingCacheDataset.__getitem__()`은 무조건 `start=0`, `stop=9`를 사용한다. `temporal_sampling != "first"`는 생성자에서 거부한다.

### 2.2 독립 2K Anchor

Anchor는 GT latent 중앙 crop이 아니다. 호출 체인은 다음과 같다.

```text
OutpaintingCacheDataset.__getitem__
  -> LazyAnchorCache.make_request(record_id, latent_start=0)
  -> read_cached(request)                  # hit는 CPU read/검증만
collate_outpainting
OutpaintingTrainer._training_step
  -> LazyAnchorCache.complete_batch(batch) # owning rank
     -> miss이면 get_or_create(request)
        -> prepare_anchor(request)
           -> raw RGB 정확히 [0:65) decode
           -> source RGB center crop 1920×1080
           -> native reflect alignment
        -> encode_prepared([one item])      # frozen official VAE, inference_mode
        -> lock + fsync + atomic rename
     -> canonical full-canvas condition에 삽입
```

핵심 shape:

- raw crop: `[F=65, H=1080, W=1920, C=3]`
- normalized/padded VAE input: `[1, 3, 65, 1088, 1920]`, BF16
- independent Anchor cache latent: `[128, 9, 34, 60]`, BF16
- canonical placement rectangle: latent `[y=17:51, x=30:90]`
- `complete_batch()` 결과 `anchor_latents["latents"]`: `[B,128,9,68,120]`; Anchor support 밖은 0이나, 그 0은 conditioning mask로 known 처리되지 않는다.

`complete_batch()`는 GT `gt_record_id`, `rgb_start_frame`, `latent_start_index`와 Anchor request를 다시 비교한다. cache miss VAE encode는 DataLoader worker에서 금지되고 owning training process에서 동기적으로 한 sample씩 수행된다. raw RGB 전체 4K를 VAE에 입력하는 경로 또는 `crop(VAE(full GT))` fallback은 코드상 없다.

### 2.3 READY-before-MISSING 순서

`OutpaintingTrainer._init_dataloader()`는 startup 시 Anchor inventory를 `READY`와 `MISSING`으로 나눈다. `DeterministicEpochSampler`는 각 그룹을 동일 seed로 독립 shuffle한 뒤 READY 그룹을 먼저 연결한다. ordering contract와 index snapshot은 checkpoint manifest에 저장된다.

- 같은 seed/source/cache snapshot: 동일 순서
- resume 중 formerly-missing entry가 생성되어도 저장된 startup 순서를 유지
- 과거 READY가 사라지거나 provenance가 invalid해지면 resume 실패
- world size, batch size, accumulation, dataset fingerprint 차이도 resume 실패

## 3. 고정 empty-text conditioning

### 3.1 사전 connector feature 생성

`OutpaintingTrainer._load_models()`는 native transformer/embeddings processor를 로드한 후 다음을 수행한다.

1. persistent `self._embeddings_processor.eval().requires_grad_(False)`
2. `conditions_dir is None`, `constant_prompt == ""` 재검증
3. temporary frozen/eval Gemma text encoder로 `encode([""])`
4. temporary frozen/eval embeddings processor의 `feature_extractor(hidden_states, attention_mask, "left")`
5. **video와 audio preconnector feature를 모두** CPU에 저장

저장 key:

```text
video_prompt_embeds
 audio_prompt_embeds
prompt_attention_mask
```

여기서 `audio_prompt_embeds`는 audio diffusion branch를 활성화하지 않는다. pinned LTX AV embeddings connector의 `create_embeddings(video_features, audio_features, additive_mask)` 계약을 충족하기 위한 두 번째 feature stream이다.

### 3.2 매 batch 처리

`OutpaintingTrainer._training_step()`은 dataset이 `conditions`를 제공하면 즉시 거부한다. 이후 constant tensors를 `[B,...]`로 `unsqueeze(0).expand(...)`하고 native `LtxvTrainer._training_step()`에 넘긴다.

Native trainer는:

```text
convert_to_additive_mask(binary_mask)
  -> self._embeddings_processor.create_embeddings(video_features, audio_features, additive_mask)
  -> final video context + audio context + binary mask
```

을 수행한다. `OutpaintingStrategy`는 최종 **video context**와 mask만 video `Modality`에 넣고 `audio=None`을 반환한다. Text Cross-Attention module은 삭제되지 않으며 frozen parameter로 매 block 한 번 실행된다. Cross-Attention을 `no_grad`로 감싸지 않으므로 hidden-state gradient는 앞쪽 trainable SA adapter까지 전달될 수 있다.

## 4. 한 batch의 noise, known mask, timestep

### 4.1 Patchification

`OutpaintingStrategy.prepare_training_inputs()`:

- target `[B,128,9,68,120]` → patch size 1 → `latents [B,73440,128]`
- anchor canvas `[B,128,9,68,120]` → `[B,73440,128]`
- 두 shape가 다르면 실패

### 4.2 Flow-matching 초기 상태

Pinned `FlexibleStrategy._initialize_noisy_target()`를 그대로 호출한다. sample별 `σ_b`에 대해:

\[
\epsilon \sim \mathcal N(0,I),\qquad
x_{\sigma}=(1-\sigma)x_0+\sigma\epsilon,
\]

\[
v^{\star}=\epsilon-x_0.
\]

초기 `timesteps`는 `[B,N]`에서 sample별 σ를 모든 token에 broadcast하고, 초기 `loss_mask`는 모두 True다.

`shifted_logit_normal`은 Hrlrvo subclass `StableShiftedLogitNormalTimestepSampler`를 사용한다. native RNG draw 순서는 유지하면서 sigmoid 및 percentile stretch만 FP64로 계산하며, 결과를 원래 RNG dtype으로 돌린다. `video.sigma`와 `video.timesteps`는 transformer 직전 finite assertion을 통과해야 한다.

### 4.3 Spatial known mask

Anchor rectangle은 config pixel 끝점을 다시 floor 계산하지 않는다. 각 cache entry provenance의:

```text
canonical_latent_rect_yxyx = [17,30,51,90]
anchor_latent_hw = [34,60]
```

를 `_latent_anchor_rect()`가 상호 검증해 사용한다. spatial mask shape은 `[B,68,120]`, flatten/repeat 후 `[B,73440]`이며 모든 9 latent frame에서 `[17:51,30:90]`가 True다.

### 4.4 Prefix mask

- `prefix_frames=17`
- temporal factor=8
- `(17-1)/8+1 = 3` latent frames
- sample별 Bernoulli `use_prefix ~ Bernoulli(0.5)`
- prefix mask: 처음 `3×68×120` tokens True

최종 known mask:

\[
K = K_{anchor}\lor K_{prefix}.
\]

Prefix-free sample에서는 Anchor만 known이다. Prefix-conditioned sample에서는 처음 3 latent frame 전체 canvas가 known이고, 이후 frame에서는 Anchor만 known이다.

현재 4K latent geometry에서:

- 전체 tokens: 73,440
- Anchor known: `9×34×60 = 18,360`
- prefix-free unknown/loss tokens: `73,440-18,360 = 55,080`
- prefix-conditioned known union: `18,360 + 3×68×120 - 3×34×60 = 36,720`
- prefix-conditioned unknown/loss tokens: 36,720

### 4.5 Clean value, timestep, loss exclusion

```python
condition_values = where(prefix_mask, full_gt_latent, independent_anchor_canvas)
noisy = where(known, condition_values, noisy)
timesteps = where(known, 0, sampled_timestep)
loss_mask = loss_mask & ~known
```

따라서:

| 영역 | transformer input | token timestep | loss |
|---|---|---:|---|
| clean full-canvas prefix | GT clean latent | 0 | 제외 |
| 이후 Anchor | independent Anchor latent | 0 | 제외 |
| 이후 Outer | flow-noised GT | sampled σ | 포함 |
| prefix-free Anchor | independent Anchor latent | 0 | 제외 |
| prefix-free Outer | flow-noised GT | sampled σ | 포함 |

Prefix clean latent는 기존 full-sequence GT cache에서 가져온다. `LazyAnchorCache.__init__()`는 실제 checkpoint VAE config가 audited causal `CausalVideoAutoencoder`, `pixel_norm`, 허용된 causal encoder blocks인지 검사한다. 이 계약을 통과하지 못하면 시작이 차단된다.

## 5. Transformer 입력과 block 호출 체인

`ModelInputs.video`는 다음을 가진다.

```text
latent:       [B,73440,128]
sigma:        [B]
timesteps:    [B,73440]
positions:    [B,3,73440,2]
context:      empty-prompt video connector output
context_mask: connector binary mask
geometry:     CanvasGeometry(T=9,H=68,W=120,anchors per sample)
```

Native LTX preprocessor가 latent를 `D=4096` hidden으로 투영한다. `OutpaintingPreprocessor.prepare()`는 native positional embeddings를 `ViewContext`로 감싸되 canonical positions, sigma 및 geometry를 보존한다.

각 `BasicAVTransformerBlock`의 실제 호출 순서는 유지된다.

```text
get_ada_values(slice 0:3) -> vshift_msa, vscale_msa, vgate_msa
AdaNorm(input)
MultiViewSelfAttention.forward
  Local System
    K 또는 K+1 windows
    각 window:
      Native Local SA
      Adaptive Target-query-only SA
      Target write-mask Local Fusion
    coordinate index_add + count normalization -> LocalFull
  Global C4 compression -> Global SA -> R4 -> GlobalFull
  LocalFull/GlobalFull fusion
  shared attn1.to_out exactly once
native post-SA gate/residual
native text Cross-Attention exactly once
native FFN exactly once
next block
```

### 5.1 Native

Native window `L_j`는 원 canonical window를 직접 읽는다.

\[
Q_N=(W_Q+\Delta Q_{OUT})X_{L_j}
\]

K/V도 동일하다. Native-specific LoRA는 없다. Native output은 원 window의 `[B,T,H_j,W_j,D]`다.

### 5.2 Adaptive

Adaptive는 C4-first tokenization이다. pure Bridge C4 parent는 C4 하나로 유지하고, A/T pixel이 하나라도 닿은 parent는 네 개의 full C2로 승격한다. Target-touching C2가 packed sequence 앞에 오며:

```text
Q = Target-touching C2 only
K/V = all Anchor/Bridge/Target C2/C4 cells
```

\[
Q_A=(W_Q+\Delta Q_{OUT}+\Delta Q_A)X_T,
\]

\[
K_A,V_A=(W+\Delta OUT+\Delta A)X_{A+B+T}.
\]

PMA는 packed index가 아니라 canonical nominal cell origin을 Target query scale(S/2=2)에서 해석한다. 시간 좌표는 native LTX bounds를 그대로 쓴다. Adaptive output은 각 query C2를 R2로 복원한 뒤 T-owned tap만 원 Local 좌표에 쓴다. A/B는 K/V context일 뿐 write하지 않는다.

### 5.3 Local Fusion과 overlap assemble

`vgate_msa`는 native block이 실제 계산한 `[B,1,D]` 또는 `[B,N,D]` 신호다. block별 trainable vectors `γ_L,β_L∈R^D`:

\[
\alpha_L=\sigma(\beta_L+g_{msa}\odot\gamma_L).
\]

Target write-mask 내부:

\[
L_j^*=\alpha_L\odot Native_j+(1-\alpha_L)\odot Adaptive_j.
\]

Target write-mask 밖은 `torch.where(..., Native)`로 **Native exact passthrough**다. 이후 모든 `L_j*`를 canonical indices에 `index_add_`하고 count로 나눈다. non-divisible K4/K5 overlap은 uniform normalized blending이다.

현재 config의 Local 초기 native weight는 **0.5**이므로 `γ=0`, `β=logit(0.5)=0`, 초기 α=0.5다.

### 5.4 Global 및 두 번째 fusion

Global:

```text
[B,T,H,W,D]
 -> C4 explicit cells [B,T,N4,D]
 -> (W + ΔOUT + ΔG) Q/K/V SA
 -> rectangular R4 to [B,T,H,W,D]
```

Global geometry는 `H%4 == W%4 == 0`을 검증한 뒤 full4×4 cells만 만든다. Adaptive pure-B와 동일한 per-block `compressors["4"]`의 learned Conv2d를 호출하며 partial/valid-count normalization은 없다. R4는 기존 whole-grid bilinear upsampling + depthwise Conv3D다. LocalFull과 GlobalFull이 동일 canonical support가 된 뒤:

\[
\alpha_G=\sigma(\beta_G+g_{msa}\odot\gamma_G),
\]

\[
FusedSA=\alpha_G\odot LocalFull+(1-\alpha_G)\odot GlobalFull.
\]

현재 초기 LocalFull weight는 0.5다.

### 5.5 Shared O, Cross-Attention, FFN

- final fused tensor에 native `attn1.to_out[0]` + shared r32 O LoRA가 정확히 한 번 적용된다.
- native SA residual/gate는 pinned block `post_sa_function`이 소유한다.
- frozen native `attn2` Text Cross-Attention은 한 번 실행된다.
- frozen native FFN base의 `ff.net.0.proj`, `ff.net.2`에 shared r32 FFN LoRA delta가 설치되며 FFN 자체는 한 번 실행된다.
- branch별 O/CA/FFN은 없다.

## 6. Trainable/frozen ownership

`install_outpainting()`은 먼저 `model.requires_grad_(False)`를 호출하고, 새 delta/module만 trainable로 등록한다. `_collect_trainable_params()`는 최종적으로 `requires_grad=True`인 parameter object만 `_trainable_params`에 수집한다.

### 6.1 Trainable

| 범주 | 구성 | parameter 수 |
|---|---|---:|
| shared Outpainting LoRA | block별 Q/K/V/O, r32 α32 | 50,331,648 |
| Adaptive LoRA | block별 Q/K/V, r64 α64 | 75,497,472 |
| Global LoRA | block별 Q/K/V, r64 α64 | 75,497,472 |
| shared FFN LoRA | block별 `ff.net.0.proj`, `ff.net.2`, r32 α32 | 62,914,560 |
| compressors | block별 C2+C4 depthwise Conv2D | 4,325,376 |
| decompressors | block별 R2+R4 depthwise Conv3D | 11,010,048 |
| fusions | block별 Local/Global γ,β | 786,432 |
| **합계** |  | **280,363,008** |

LoRA는 bias 없이 `A: in→rank` Kaiming-uniform, `B: rank→out` zero initialization이며:

\[
\Delta W(x)=\frac{\alpha}{r}B(A(x)).
\]

현재 모든 rank/alpha pair에서 nominal scale은 1이다. base Linear object/weight는 복제하지 않는다.

### 6.2 Frozen

- pretrained SA Q/K/V/O base
- native AdaLN/modulation tables과 norms
- Text Cross-Attention Q/K/V/O 전체
- FFN base weights
- input/output native projections 및 기타 transformer base
- text encoder
- embeddings connector/processor parameters
- VAE

Frozen transformer base count는 18,988,838,144, installed total은 19,269,201,152다. [header 기반 정적 재계산](evidence/parameter_static_inspection.json)으로 확인했다. model을 materialize하지 않았으므로 parameter object/runtime count 검사는 미실행이다. 이 frozen count에는 native AV/audio machinery도 포함되며, video-only generation이 모든 audio parameter object를 삭제한다는 뜻은 아니다. `parameter_counts()`는 실제 설치 시 parameter name과 `requires_grad` ownership을 검사한다.

## 7. 최종 loss: 정확한 단일 목적함수

Transformer의 최종 video prediction `\hat v∈R^{B×N×128}` 하나만 loss에 들어간다. Native, Adaptive, Global, Local fusion, Global fusion에 대한 별도 auxiliary loss는 **없다**.

Pinned `FlexibleStrategy._compute_modality_loss()`는 target 길이에 맞춰 prediction 뒤쪽 N tokens를 선택하고 다음을 sample별 계산한다. `m_{bn}∈{0,1}`은 unknown/loss mask다.

\[
\mathcal L_b
=
\frac{
\operatorname{mean}_{n,c}\left[m_{bn}(\hat v_{bnc}-v^\star_{bnc})^2\right]
}{
\operatorname{mean}_{n}\left[m_{bn}\right]_{\min 10^{-8}}
}
=
\frac{1}{C\sum_n m_{bn}}
\sum_{n,c}m_{bn}(\hat v_{bnc}-(\epsilon-x_0)_{bnc})^2.
\]

Native trainer loop은 `output.loss.mean()`으로 batch 평균을 만든 뒤 Accelerator backward를 호출한다.

**명시적 부재:**

- Native branch loss 없음
- Adaptive branch loss 없음
- Global branch loss 없음
- fusion consistency loss 없음
- known Anchor reconstruction loss 없음
- clean prefix loss 없음
- audio loss 없음

모든 trainable branch는 최종 fused prediction의 동일 masked flow-matching loss를 통해 joint optimization된다.

## 8. Optimizer, scheduler, precision, distributed execution

현재 config와 pinned trainer 조합:

- optimizer: `torch.optim.AdamW(_trainable_params, lr=1e-4)`
  - 별도 override가 없으므로 PyTorch AdamW 기본 betas/eps/weight decay를 사용
- scheduler: `LinearLR`, 2,000 steps, factor 1.0 → 0.1
- mixed precision: BF16 Accelerator autocast
- gradient checkpointing: enabled, native transformer mechanism
- per-device batch: 1
- gradient accumulation: 1
- max grad norm: 1.0
- workers: 2, spawn, pinned memory/persistent workers enabled
- DDP/FSDP: native Accelerator preparation을 재사용

Loop 순서:

```text
complete lazy Anchor batch
-> send tensors to accelerator device
-> empty connector conditions construct
-> native connector
-> prepare_training_inputs
-> transformer
-> one final masked loss
-> accelerator.backward(loss.mean())
-> clip_grad_norm_(trainable params, 1.0)
-> AdamW.step
-> zero_grad
-> LinearLR.step
```

Frozen parameter는 `_trainable_params`에 들어가지 않는다. `_setup_lora()`가 official PEFT setup을 호출하지 않고 Hrlrvo `install_outpainting()`만 호출하므로 adapter가 이중 설치되지 않는다.

## 9. Checkpoint save/resume/inference handoff

### 9.1 저장

`OutpaintingTrainer._save_checkpoint()`는:

- `adapter_model.safetensors`: `_outpainting_parameter_names`에 속한 v8 trainables만 저장
- metadata: absolute base path, full `ArchitectureConfig`, pinned provenance
- Accelerator state: optimizer, scheduler, scaler(해당 시), dataloader, rank RNG
- `outpainting_manifest.json`: global step, full training/outpainting config, parameter counts, world/distributed type, seed, batch/accumulation, temporal sampling, dataset fingerprint, frozen READY/MISSING ordering

을 저장한다. frozen 약 19B base 전체를 adapter checkpoint에 복제하지 않는다.

### 9.2 load/resume

`load_adapter()`는 다음을 모두 strict 비교한다.

1. `base_model_path`
2. exact architecture dataclass serialization(version 8 포함)
3. base/source provenance
4. expected state key 집합의 exact equality
5. 각 tensor shape 및 floating dtype

Pre-v8 또는 다른 layout checkpoint는 architecture mismatch로 거부된다. standalone adapter file은 `checkpoints.no_resume=true`일 때만 weights-only start가 가능하다. 정확한 resume에는 optimizer/RNG/manifest와 scheduler/scaler 조건부 state가 모두 있어야 한다.

### 9.3 inference handoff

Inference는 동일 `install_outpainting()`으로 frozen base에 v8 module topology를 만든 후 `load_adapter()`를 호출한다. 따라서 training-side adapter-only save와 inference-side strict load가 같은 parameter identity를 사용한다.

## 10. 정적 감사 결론

| 범주 | 판정 | 이유 |
|---|---|---|
| GT/Anchor temporal correspondence | PASS | FIRST `[0:65)↔[0:9)`를 dataset/request 양쪽에서 assert |
| independent Anchor / GT leakage 차단 | PASS | raw crop→독립 VAE만 존재; GT latent crop fallback 없음 |
| known mask/timestep/loss | PASS | provenance `[17:51,30:90]`, prefix union, t=0, loss 제외가 한 함수에 연결 |
| fixed empty text dual connector | PASS | video/audio preconnector stream 보존 후 video-only Modality |
| hierarchical forward | PASS | Local per-window fusion→assemble→Global fusion→shared O→native CA→native FFN |
| single final loss / no branch loss | PASS | native compute_loss에 final `video_pred` 하나만 전달 |
| trainable/frozen selection | PASS | freeze-all 후 명시적 delta/module만 trainable; name audit 존재 |
| adapter-only checkpoint contract | PASS | metadata/key/shape/provenance strict load |
| 실제 장치 forward/backward/optimizer | INCONCLUSIVE | 이 문서 작업에서는 실행 금지 |
| 실제 cache miss VAE encode | INCONCLUSIVE | 이 문서 작업에서는 실행 금지 |

**TRAINING SPEC STATIC VERDICT: PASS**  
**RUNTIME EXECUTION VERDICT: INCONCLUSIVE (본 audit에서는 의도적으로 미실행)**
