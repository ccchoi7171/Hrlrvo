# Hrlrvo v8 추론 사양 — 최신 코드 정적 감사

## 0. 문서 범위

이 문서는 현재 [`../../src/ltx_outpaint/inference.py`](../../src/ltx_outpaint/inference.py)와 pinned LTX-2.3 [`ValidationRunner`](../../../models/LTX2.3/packages/ltx-trainer/src/ltx_trainer/validation_runner.py)의 실제 호출 경로를 기준으로 작성했다. 현재 config는 [`../../configs/inference_4k.yaml`](../../configs/inference_4k.yaml)이다.

관련 shared implementation:

- model topology/load: [`../../src/ltx_outpaint/model.py`](../../src/ltx_outpaint/model.py)
- geometry: [`../../src/ltx_outpaint/geometry.py`](../../src/ltx_outpaint/geometry.py)
- compression/reconstruction: [`../../src/ltx_outpaint/compression.py`](../../src/ltx_outpaint/compression.py)
- PMA/RoPE: [`../../src/ltx_outpaint/position.py`](../../src/ltx_outpaint/position.py)
- pinned mask conditioner: [`../../../models/LTX2.3/packages/ltx-core/src/ltx_core/conditioning/types/mask_cond.py`](../../../models/LTX2.3/packages/ltx-core/src/ltx_core/conditioning/types/mask_cond.py)

이번 문서 작성 중 실제 model load, VAE encode/decode, denoising, GPU 작업 및 output 생성은 실행하지 않았다. 따라서 실행 성공 여부는 별도로 `INCONCLUSIVE`다.

## 1. CLI와 입력 계약

Production CLI:

```bash
python -m ltx_outpaint.inference \
  --config configs/inference_4k.yaml \
  --anchor-video /path/to/anchor.mp4 \
  --output /path/to/output.mp4 \
  --prompt "" \
  [--num-frames N]
```

`--prompt`는 `choices=[""]`이고 `sample_long_range()`도 nonempty prompt를 재차 거부한다. config `sampling.negative_prompt` 역시 빈 문자열이어야 한다.

현재 기본값:

| 항목 | 값 |
|---|---:|
| requested canvas | 3840×2160 RGB |
| encoded VAE canvas | 3840×2176 RGB |
| input Anchor expected(default, 비강제) | 1920×1080 |
| FPS | 24 |
| chunk | 65 RGB frames |
| prefix/overlap | 17 RGB frames |
| stride/new frames | 48 RGB frames |
| VAE latent chunk | 9 frames |
| VAE latent prefix | 3 frames |
| steps | 30 |
| CFG | 3.0 |
| STG | 1.0, block 28 |
| guidance rescale | 0.7 |
| seed | 42 + absolute chunk start |
| dtype/device | BF16/CUDA |
| audio | disabled |

Input FPS mismatch는 tolerance 0.001을 넘으면 실패하며 resampling하지 않는다. Anchor spatial resize도 하지 않는다. `enforce_expected_anchor=false`이므로 1920×1080 외의 in-canvas Anchor도 geometry상 허용되지만, 선택된 adapter가 그러한 크기를 학습했는지는 별도의 품질 일반화 문제다.

## 2. 전체 call chain

```text
main
  -> _load_settings
     -> _install_official_sources(pinned repo)
  -> sample_long_range
     -> read_video(anchor_video)
     -> FPS/frame count 검사
     -> read_video_scale_factors
     -> _vae_boundary(requested 2160×3840 -> encoded 2176×3840)
     -> _center_anchor
     -> LongRangeSpec.validate
     -> _make_validation_config(empty positive/negative)
     -> _load_models
        -> GeometryValidationRunner.create
           -> native ValidationRunner.__init__
              -> empty text positive/negative embeddings cache
              -> decoder lifecycle 준비
        -> frozen VAE encoder load on CPU
        -> base transformer load
        -> install_outpainting(v8)
        -> strict load_adapter
        -> eval
     -> chunk loop
        -> sample_chunk
           -> _condition_latents
           -> _known_mask
           -> VideoConditionByMask.apply_to
           -> GaussianNoiser
           -> native ValidationRunner._run_denoising
           -> native VAE decode
           -> exact Anchor/prefix RGB composite
     -> concatenate only new frames
     -> save_video
```

## 3. Model 및 adapter load

`_load_models()`은 config architecture의 `local_partition_k`/`compression_scale` 이름을 `ArchitectureConfig.k`/`scale`로 정규화한 후 version 8 dataclass를 생성한다.

순서:

1. `checkpoint_provenance()`로 base SHA/source revision 확인
2. native `GeometryValidationRunner` 생성: text embeddings를 먼저 cache하고 Gemma unload
3. frozen VAE encoder를 CPU에 유지
4. LTX transformer를 지정 device/dtype으로 load
5. `install_outpainting(transformer, architecture, provenance)`
6. 전체 model을 inference dtype/device로 통일
7. `load_adapter(...)` strict load
8. `transformer.eval()`

`load_adapter()`는 base path, exact v8 architecture metadata, provenance, exact state key 집합, tensor shape/type을 검사한다. v7을 포함한 pre-v8 adapter나 일부 key만 가진 adapter는 silent partial load되지 않는다.

현재 config가 기대하는 adapter 경로:

```text
outputs/train_4k/checkpoints/checkpoint-002000/adapter_model.safetensors
```

실제 파일 존재와 완전성은 load 시 검사된다. 이 정적 audit에서는 해당 checkpoint를 열거나 model에 load하지 않았다.

## 4. Empty text와 native CFG/STG

### 4.1 Positive/negative

`_validate_empty_text_conditioning()`은:

```text
all sample.prompt == ""
validation_config.negative_prompt == ""
```

만 허용한다. CLI와 API 양쪽에서 검사된다.

Native `ValidationRunner._cache_prompt_embeddings()`는 empty positive와 empty negative를 Gemma에 각각 통과시키고 `EmbeddingsProcessor.process_hidden_states()`로 video/audio connector output을 cache한다. 추론은 video-only지만 AV connector contract 때문에 audio encoding도 cache된다.

### 4.2 실제 denoising context

`sample_chunk()`는 `runner._get_prompt_embeddings()`에서 `v_pos`, `v_neg`를 얻는다. `audio_state=None`이며 transformer에는 video modality만 공급된다.

CFG scale이 3.0이므로 native runner는 positive forward와 negative forward를 수행할 수 있다. 두 문자열이 모두 empty이므로 text 의미는 동일하지만, 코드는 native CFG 호출 구조를 유지한다. STG는 configured block 28에 self-attention skip perturbation을 구성한다. Hrlrvo `MultiViewSelfAttention`의 `all_perturbed` path도 shared Outpainting V/O 경로를 사용한다.

**공식 IC-LoRA reference token, green sentinel, caption prompt는 사용하지 않는다.** Spatial conditioning은 아래의 independent Anchor latent + binary mask로만 구성된다.

## 5. Pixel/VAE 경계와 독립 Anchor encode

### 5.1 Boundary

`_vae_boundary(2160,3840,sf.height,sf.width)`는 spatial factor에 맞춰:

```text
requested: 2160×3840
encoded:   2176×3840
top/bottom: 8/8
left/right: 0/0
```

의 `PixelBoundary`를 만든다. 이는 VAE pixel boundary 전용이다. Self-Attention token canvas는 encoded latent `68×120`이며 추가 feature-map padding을 하지 않는다. decode 후 top/bottom 8 pixel을 제거한다.

### 5.2 Center Anchor

1920×1080 input일 때 `_center_anchor()`:

```text
canonical encoded pixel rect:
y=[548:1628), x=[960:2880)
```

을 얻는다. Anchor content 자체는 resize하지 않는다.

### 5.3 `_condition_latents()`

Input:

```text
anchor:          [65,3,1080,1920]
previous_prefix: None 또는 [17,3,2160,3840]
latent_shape:    [1,128,9,68,120]
```

Anchor의 canonical origin을 VAE lattice에 맞추기 위해 leading/trailing reflect pad를 계산한다. 1920×1080 기본 Anchor는:

```text
reflect input: [1,3,65,1088,1920]
encoded Anchor latent: [1,128,9,34,60]
canonical latent rect: y=[17:51), x=[30:90)
```

이다. Anchor latent shape를 보고 임의로 중앙 삽입하지 않고, aligned encoded origin과 실제 output extent에서 rect를 만든다.

LongRange 후속 chunk의 `previous_prefix`는 full requested canvas RGB 17 frames다. `_reflect_pad_rgb()`로 `[1,3,17,2176,3840]`을 만들고 **별도로 VAE encode**하여 `[1,128,3,68,120]` prefix latent를 얻는다. condition canvas의 처음 3 latent frame 전체를 이 값으로 덮는다.

따라서 추론 조건은 clean future GT가 아니라:

- 현재 시점 input Anchor RGB의 독립 VAE latent
- 이전에 생성 완료된 clean output 17 RGB frames의 독립 VAE latent

뿐이다.

## 6. Known mask와 timestep/noise 의미

`_known_mask()`는 `_condition_latents()`가 반환한 **실제 encoded Anchor rect**를 그대로 받는다. RGB 끝점을 다시 floor 변환하지 않는다.

Mask shape:

```text
[B=1,T=9,H=68,W=120]
1 = known/clean condition
0 = generated/noised
```

첫 chunk:

- 모든 9 latent frame에서 Anchor `[17:51,30:90]` = 1
- Outer = 0
- prefix 없음

후속 chunk:

- Anchor = 1
- 처음 3 latent frame full canvas = 1
- 나머지 Outer = 0

`VideoConditionByMask(condition_latent, mask, strength=1.0).apply_to()`는 pinned native code에서:

```text
clean_latent = old_clean*(1-mask) + condition*mask
denoise_mask = old_denoise*(1-mask) + 0*mask
```

를 만든다. 이후 `GaussianNoiser`가 state를 초기화한다. Denoising step마다 native `_modality_from_latent_state()`가:

\[
token\_timesteps=denoise\_mask\times\sigma
\]

를 사용하므로 known token은 timestep 0, unknown token은 현재 sigma다.

각 step의 model output 뒤 `_post_process_latent()`가:

\[
denoised'=denoised\times denoise\_mask+clean\_latent\times(1-denoise\_mask)
\]

로 known latent를 재주입한다. 이 mask는 spatial compression density map이 아니며 Adaptive A/B/T token role과 별개다.

## 7. 실제 v8 block inference 경로

Training과 inference는 같은 `install_outpainting()`과 같은 adapter tensor를 사용한다. `OutpaintingPreprocessor`는 각 CFG/STG modality에 `CanvasGeometry(T,H,W,anchor_rect)`를 붙인다.

각 block:

```text
native AdaLN -> vgate_msa capture
LOCAL SYSTEM
  K4/K5 canonical Local windows
  per window:
    Native QKV = base + shared OUT r32
    Adaptive:
      C4-first / C2 promotion
      Q = Target-touching C2 only
      K/V = Anchor + Bridge + Target cells
      QKV = base + shared OUT r32 + Adaptive r64
      R2 full cell -> T-owned pixels만 write
    write-mask 내부 Native/Adaptive fusion
    write-mask 밖 Native exact passthrough
  coordinate assemble + overlap count normalization -> LocalFull
GLOBAL
  C4 full canvas compression
  QKV = base + shared OUT r32 + Global r64
  R4 -> GlobalFull
LocalFull/GlobalFull vgate fusion
shared native O + shared OUT,O r32 exactly once
native residual
frozen Text Cross-Attention exactly once
frozen base FFN + shared FFN r32 exactly once
```

### Fusion

\[
\alpha_L=\operatorname{sigmoid}(\beta_L+vgate_{msa}\odot\gamma_L),
\quad
\alpha_G=\operatorname{sigmoid}(\beta_G+vgate_{msa}\odot\gamma_G).
\]

현재 adapter topology의 initialization contract는 Local 0.5 Native/0.5 Adaptive, Global 0.5 LocalFull/0.5 GlobalFull이다. 추론에서는 checkpoint에서 학습된 γ/β를 load한다.

Adaptive write-mask 밖 Local output은 Native exact passthrough다. 이는 zero-valued Adaptive buffer를 blend해 Native를 감쇠하는 동작을 방지한다.

## 8. Native denoising sampler

Hrlrvo는 sampler를 새로 구현하지 않고 pinned `ValidationRunner._run_denoising()`을 직접 호출한다.

- schedule: `LTX2Scheduler().execute(steps=30)`
- integrator: `EulerDiffusionStep`
- model wrapper: `X0Model`
- positive model call
- 필요 시 CFG negative call
- 필요 시 STG perturbed call
- `MultiModalGuider.calculate(...)`
- known condition re-injection
- Euler state update

`sample_chunk()`는 video-only로 호출한다.

```text
video_state != None
video_clean != None
audio_state = None
audio_clean = None
audio contexts are not supplied to an audio modality
```

`state.positions[:,0]`에는 `absolute_start_frame/fps`를 더해 후속 chunk의 temporal RoPE 시간이 0으로 재시작하지 않도록 한다.

## 9. Decode와 pixel-exact observed output

Denoising 완료 후 native runner:

1. conditioning bookkeeping clear
2. unpatchify
3. tiled VAE decode
4. `[-1,1]→[0,1]` clamp

를 수행한다. Hrlrvo는 encoded canvas에서 requested 2160×3840 영역만 crop한다.

마지막으로 pixel output에서:

- Anchor rect를 원 input Anchor RGB로 정확히 덮어씀
- 후속 chunk의 첫 17 frames를 `previous_prefix` RGB로 정확히 덮어씀

Transformer hidden을 block마다 freeze하는 것이 아니라, 최종 pixel output에서 관측 영역을 exact composite하는 정책이다.

Output shape은 chunk마다 `[65,3,2160,3840]`이며 `sample_chunk()` 반환 시 CPU tensor다.

## 10. LongRange 65/17/48 조립

`LongRangeSpec` invariant:

```text
65 = 17 + 48
(65-1) % 8 = 0
(17-1) % 8 = 0
```

### 첫 chunk

```text
absolute range: [0:65)
previous_prefix: None
output append: generated[0:real_count]
```

이는 학습의 prefix-free sample에 대응한다.

### 두 번째 chunk

첫 chunk 마지막 17 frames `[48:65)`를 clean prefix로 사용한다.

```text
start=48
Anchor input range: [48:113)
prefix condition: absolute [48:65)
new output: generated[17:65] -> absolute [65:113)
```

### 이후

start가 48씩 증가한다. 매 chunk의 마지막 17 generated frames가 다음 chunk의 prefix가 되고, append할 때 prefix 17은 버리고 새 48만 연결한다. 따라서 의도된 경로에서는 prefix frame을 중복 append하지 않는다.

마지막 chunk가 65보다 짧으면 기본 `repeat_last`가 Anchor input의 마지막 RGB frame을 반복해 VAE-aligned chunk를 만든다. 실제 결과에서는 `new_real_count`만 append하고 최종적으로 requested length로 자른다. `final_chunk_policy=reject`를 선택하면 이러한 padding 대신 실패한다.

각 chunk seed:

\[
seed_{chunk}=42+absolute\_start\_frame.
\]

이는 chunk마다 deterministic하되 서로 다른 noise를 사용한다.

## 11. Training–Inference conditioning 대응

| 의미 | Training | Inference | 판정 |
|---|---|---|---|
| Anchor source | raw center crop 독립 VAE cache | input Anchor RGB 독립 VAE | PASS |
| Anchor latent rect | provenance actual extent `[17:51,30:90]` | actual encoded extent `[17:51,30:90]` | PASS |
| known polarity | True=clean/t=0/loss 제외 | mask 1=clean/denoise mask 0 | PASS |
| prefix | 확률 0.5, GT clean first 17 RGB/3 latent | prior generated clean 17 RGB 독립 encode/3 latent | 의미 대응 PASS |
| prefix-free | 확률 0.5 | first chunk | PASS |
| unknown Outer | sampled flow noise, loss 포함 | Gaussian initial noise, denoise | PASS |
| text positive | empty | empty | PASS |
| text negative | validation empty | empty | PASS |
| audio branch | None | None | PASS |
| architecture/adapter | v8 | exact v8 strict load | PASS |

학습 prefix의 clean 값은 causal audited GT cache이며, inference prefix는 이전 output RGB의 independent encoding이라는 데이터 출처 차이는 의도적이다. 두 경로 모두 known prefix timestep 0과 full-canvas support를 공유한다.

## 12. Checkpoint와 재현성

Inference는 frozen base를 config path에서 다시 load하고 adapter-only checkpoint를 엄격히 적용한다. 필요한 metadata:

- exact base path
- complete serialized `ArchitectureConfig` version 8
- base checkpoint SHA와 pinned source revision을 포함한 provenance
- 모든 expected trainable tensor key, 추가 key 없음
- exact tensor shapes

Runtime config의 seed, sampler steps, CFG/STG, FPS 및 chunk policy는 adapter metadata와 별개이며 inference YAML이 결정한다. 동일 input/config/device/kernel에서 seed 경로는 deterministic하게 구성되어 있으나 CUDA kernel 수준 exact reproducibility는 이 정적 감사에서 실행 검증하지 않았다.

## 13. 정적 감사 결론

| 범주 | 판정 | 근거 |
|---|---|---|
| fixed empty positive/negative text | PASS | CLI/API/ValidationConfig 삼중 guard |
| independent Anchor encode | PASS | Anchor RGB만 VAE input, actual extent에서 rect 생성 |
| 1920×1080 기본 geometry | PASS | encoded `[17:51,30:90]`, mask/CanvasGeometry 동일 rect |
| prefix condition | PASS | 17 RGB→3 latent, full canvas known, prior output source |
| known timestep/noise re-injection | PASS | native `VideoConditionByMask`, per-step post-process |
| train/inference architecture parity | PASS | same `install_outpainting` + strict `load_adapter` |
| hierarchy/shared O/CA/FFN | PASS | shared production model code 사용 |
| official scheduler/CFG/STG | PASS | native `_run_denoising` 직접 호출 |
| long-range frame ordering | PASS | first 65, 이후 17 overlap 제거 후 48 append |
| actual adapter file availability | INCONCLUSIVE (runtime prerequisite) | 기본 config의 adapter_path가 stat상 없음; fake checkpoint 생성/교체/모델 load 미실행 |
| actual 4K VAE/model execution | INCONCLUSIVE | 실행 금지 |
| output visual/temporal quality | INCONCLUSIVE | generation 미실행 |

**INFERENCE SPEC STATIC VERDICT: PASS**  
**PRODUCTION EXECUTION VERDICT: INCONCLUSIVE (본 audit에서는 의도적으로 미실행)**

현재 configured checkpoint의 부재는 `evidence/parameter_static_inspection.json`에서 filesystem stat으로 확인했다.
이것은 train/inference architecture mismatch가 아니라 실제 inference를 실행하기 전 필요한 artifact 상태다.
