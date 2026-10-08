# 10. 최종 Architecture Audit — v8 shared full-cell C4 반영

## 감사 요약

**PASS — architecture is fully specified and code-consistent**

최신 사용자 정의에 따라 C4는 **Adaptive/Global이 공유하는 per-block full4×4
learned primitive**이며, R4는 **whole Global compressed grid의 fixed bilinear +
depthwise Conv3D restoration**이다. 이전 감사의 Global partial C4 문제(M1)는
소스에서 제거했다. R4 per-token isolation을 요구했던 이전 판정(M2)은 과도한
판정 기준이었으므로 철회했다. **R4 구현은 변경하지 않았다.**

이 판정은 source/CPU-unit contract 판정이다. 실제 training/inference, GPU,
full-model forward/backward 또는 optimizer step 성공을 의미하지 않는다.

## 1. 기준 코드 / 작업 경계

- Project: `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/main`
- Conda: `Hrlrvo`; Python `/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python`
- Architecture/checkpoint metadata: **version8 유지**.
- Pinned LTX: `598ab41247a77dbfe29b5186e915bcf4f9040ec7`; read-only.
- 수정 production source: `geometry.py`, `compression.py`, `global_attention.py`만.
- tests와 문서/README는 변경된 계약 및 기존 잘못된 설명에 한정해 갱신했다.
- `model.py`, Adaptive/Native, LoRA, PMA, train/inference, data/cache, YAML은 변경하지 않았다.
- 현재 C4의 허용 full-cell 입력에서 weight/shape/key/init/operator 의미는 그대로다.
  비정상 geometry의 partial fallback만 제거했으므로 version을 임의 증가시키거나
  checkpoint를 변환하지 않았다. 기존 strict metadata/key/provenance load는 그대로다.

### 검증 경계

실행: CPU geometry/layout/coordinate 검사, 작은 full-cell Conv2d, 작은 두
`MultiViewSelfAttention` **construction only** ownership 검사, CPU AdamW parameter
selection(상태 비어 있음), compressor-only `torch.autograd.grad`, compile/import/config/Ruff.

**미실행:** LTX 모델/attention forward, training/inference pipeline, final loss backward,
optimizer step, GPU/CUDA, VAE encode/decode, generation. `torch.cuda.is_initialized()`는
False다. 기존 cache/precache/다른 process를 중단하거나 재구성하지 않았다.

원래 read-only 감사와 현재 수정의 증거를 분리한다. 원래 geometry/parameter/PNG
증거는 보존하고, 이번 실행/diff/hash는 `evidence/c4_unification/`에 기록한다.

## 2. 상세 문서 안내

| 문서 | 내용 |
|---|---|
| [01_TRAINING_SPEC.md](01_TRAINING_SPEC.md) | cache/dataset/known mask/noise/timestep/connector/model/final loss/optimizer/checkpoint |
| [02_INFERENCE_SPEC.md](02_INFERENCE_SPEC.md) | Anchor preparation/empty prompt/known state/native denoiser/long-range/decode/train parity |
| [03_NATIVE_LOCAL_BRANCH.md](03_NATIVE_LOCAL_BRANCH.md) | K4/K5, canonical window/RoPE, base+OP QKV, O 이전 core |
| [04_ADAPTIVE_LOCAL_BRANCH.md](04_ADAPTIVE_LOCAL_BRANCH.md) | A>T>B, alignment, full-cell promotion, Q/KV, PMA, per-cell R2, Target-only Local scatter |
| [05_GLOBAL_BRANCH.md](05_GLOBAL_BRANCH.md) | shared C4, Global QKV, R4, full-cell fail-closed 및 fixed Global-grid R4 |
| [06_FUSION_SPEC.md](06_FUSION_SPEC.md) | mask-gated Local fusion, normalized assemble, Local–Global fusion, native gate/O |
| [07_COMPRESSION_DECOMPRESSION_SPEC.md](07_COMPRESSION_DECOMPRESSION_SPEC.md) | C2/C4/R2/R4 primitive/ownership/initialization, active-vs-legacy path |
| [08_FULL_ARCHITECTURE_FLOW.md](08_FULL_ARCHITECTURE_FLOW.md) | one canonical hidden의 block-level 전체 흐름과48 blocks/denoising steps 구분 |
| [09_ADAPTIVE_GRID_GEOMETRY_AUDIT.md](09_ADAPTIVE_GRID_GEOMETRY_AUDIT.md) | 실제41 cases 표, coordinate projection, overlap/count 통계, PNG |
| 현재 문서 | 종합 PASS/FAIL/INCONCLUSIVE 및 evidence/변경 범위 |

## 3. Train / Inference의 실제 공통 call graph

### Training

```text
src/ltx_outpaint/train.py::main
  -> OutpaintingTrainer (pinned LtxvTrainer subclass).train / DataLoader
  -> OutpaintingCacheDataset + collate_outpainting
  -> OutpaintingTrainer._training_step
       -> LazyAnchorCache.complete_batch
       -> send_to_device + fixed-empty video/audio preconnector feature supply
       -> super()._training_step (pinned LtxvTrainer)
  -> EmbeddingsProcessor.create_embeddings(video_features, audio_features, mask)
  -> OutpaintingStrategy.prepare_training_inputs
  -> installed LTXModel.forward
  -> LTXModel._process_transformer_blocks
  -> 48 BasicAVTransformerBlock instances
  -> _forward_with_native_gate
  -> BasicAVTransformerBlock.forward
  -> MultiViewSelfAttention.forward
  -> native SA residual / frozen Text CA / shared FFN
  -> final video prediction
  -> native FlexibleStrategy.compute_loss (single masked flow loss)
```

### Inference

```text
src/ltx_outpaint/inference.py::main / sample_long_range
  -> _load_models
  -> same load_transformer + install_outpainting + load_adapter
  -> inference.sample_chunk (module-level function)
  -> GeometryValidationRunner (_Runner) + native ValidationRunner._run_denoising
  -> same installed LTXModel.forward
  -> same 48 BasicAVTransformerBlock instances / native-gate wrapper
  -> same MultiViewSelfAttention.forward
  -> same native residual / Text CA / FFN
  -> native Euler scheduler + known state post_process_latent
  -> native VAE decode / visible-boundary crop / source compositing
```

`train.py`나 `inference.py`에 별도 Adaptive tokenization/restoration/Fusion 사본은 없다.
두 entrypoint는 immutable `CanvasGeometry`를 공급하고 실제 block에서 공통 geometry
helper를 호출한다. 차이는 데이터 생성/conditioning 공급, training loss vs scheduler,
autocast/eval dtype policy와 CFG/STG이지 별도의 attention architecture가 아니다.

**정밀한 예외:** production inference는 official STG가 설정되어 있다. perturbation
pass의 `all_perturbed` block에서는 native V-bypass를 보존하므로 hierarchy를 건너뛴다.
이것은 별도 inference용 Adaptive 구현이 아니라 **동일 MultiViewSelfAttention의
명시적 guidance control path**다. 따라서 “모든 denoising model call에서48개 block의
모든 branch가 무조건 실행된다”는 주장은 하지 않는다. 정상 denoising pass는 위
공통 hierarchy를 사용하고, O는 bypass에서도 같은 마지막 단일 호출이다.

## 4. 요구 항목별 최종 판정

`PASS`는 static/선택한 CPU-unit 범위다. 전체 모델 runtime/VRAM 판정은 아니다.

| 항목 | 판정 | 핵심 근거/범위 |
|---|---|---|
| Training specification | PASS | FIRST cache→known/noisy→single final flow loss; 01 |
| Inference specification | PASS | native scheduler/known reinjection/empty text/common model; 02 |
| Train/Inference architecture parity | PASS | 동일 installer/native block/shared functions; 변경 없음 |
| Native Local branch | PASS | canonical Local; base+OP only; branch 내부 O 없음 |
| Adaptive Local branch | PASS | full C4/C2 promotion→Target Q→R2→Target write; 41 regression |
| Global branch | PASS | shared full-cell C4→base+OP+G SA→fixed-grid R4 |
| Local Fusion | PASS | `where(M_T,lerp(A,N,alpha_L),N)`; non-T exact Native |
| Local–Global Fusion | PASS | canonical LocalFull/GlobalFull의 `lerp(G,L,alpha_G)` |
| shared O exactly once | PASS | 최종 sole `self.to_out(fused)`; branch 내부 O 없음 |
| Cross-Attention position | PASS | native SA residual 뒤 shared frozen Text CA |
| FFN position | PASS | Text CA residual 뒤 native shared FFN+F-LoRA |
| C2 single-operation invariant | PASS | complete2×2, 단일 learned Conv2d |
| C4 single-operation invariant | PASS | complete4×4, 동일 per-block object/forward; partial path0 |
| R2 single-operation invariant | PASS | query-as-batch 1×1→2×2; 이후 Target mask; 변경 없음 |
| R4 single-operation invariant | PASS | whole Global grid→bilinear full canvas→DW Conv3D; byte-identical |
| A > T > B ownership | PASS | `resolve_adaptive_roles`; source 보호 |
| C4-first/C2-promotion | PASS | A/T touch parent→exact4 fullC2; pureB만 C4 |
| Target-query-only | PASS | T-touching C2만 Q; 모든 mixed token KV |
| Target-only Adaptive write | PASS | query tap→Local-relative scatter + fusion mask |
| resolved-T empty early exit | PASS | 41개 중5개, Q/cells/write 없음 |
| PMA mixed-resolution coordinates | PASS | canonical origins/nominal scale/native time; selected CPU regression |
| 4×4 geometry | PASS | 16 windows/layouts; full coverage |
| 5×5 geometry | PASS | 25 windows/layouts; minimal overlap/normalized assembly 유지 |
| 41-case visualization audit | PASS | 기존41 PNG+3 sheets; 새 geometry regression과 layout 불변 |
| Global invalid geometry fail-closed | PASS | cell 생성 이전 divisibility ValueError; no C4/SA/R4 execution |
| C4 parameter/optimizer/state ownership | PASS | 동일 module/weight; block간 독립; 중복0 |
| C4 two-consumer gradient sum | PASS | 작은 Conv-only autograd + differentiable shared source call sites |
| checkpoint/architecture metadata consistency | PASS | v8/35 normalized fields; parameter namespace/loader 변경 없음 |
| parameter category/count static inspection | PASS | 기존 header/formula, trainable280,363,008; parameter 변화0 |
| real trained checkpoint roundtrip | INCONCLUSIVE | 미실행; configured artifact는 별도 runtime prerequisite |
| real forward/backward/inference/VRAM | INCONCLUSIVE | 요청에 따라 미실행 |

## 5. 이전 FAIL의 처리

### M1 — Global partial C4: 수정 완료

- 기존 `geometry.global_cells/cells_for_rect`는 nondivisible edge를 `min(...)`으로
  줄였고 `SpatialCompressor.forward`가 valid kernel tap만 더한 뒤 `s²/valid_count`로
  보정했다. 이 경로는 실제로 도달 가능한 legacy fallback이었다.
- 현재 `cells_for_rect`는 H/W가 scale로 나누어떨어지지 않으면 cell 생성 **전에**
  ValueError를 낸다. 허용된 cell의 indices/bounds는 언제나 full s×s다.
- `SpatialCompressor.forward`는 full count/taps/bounds/current-canvas indices를
  검증하고 `[B·T·N,D,s,s] → self.conv` 한 경로만 실행한다. masked/partial
  einsum, area normalization, input zero-fill, padding, remainder drop은 없다.
- Global compressed shape는 guard 이후 `(H//4,W//4)`다. `ceil` fallback은 없다.
- `global_cells(68,120,4)`: full510/partial0 →17×30.
- `global_cells(17,20,4)`: 명시적 ValueError, partial token 생성0.
- `CanvasGeometry`나 standalone Adaptive의 기존 작은 geometry는 재설계하지 않았다.
  거부 지점은 **Global cell producer**이며 Global compressor/attention/R4 이전이다.

### M2 — R4 per-token 요구: 판정 철회, 구현 변경 없음

최신 최종 정의는 다음과 같다.

> R4 is one fixed Global-grid restoration operation: bilinear spatial upsampling followed by the existing depthwise Conv3D refinement. It is not required to decode each C4 token independently.

`SpatialDecompressor` 전체 class body가 변경 전과 byte-identical하다. Global은
전체17×30 lattice를68×120으로 bilinear(`align_corners=False, antialias=True`)
upsample한 뒤 DW Conv3D refinement한다. 이웃 token mixing은 의도된 연산이다.
R2처럼 token별 isolation을 새로 만들지 않았고, Adaptive C4에 R4를 추가하지 않았다.

원래 FAIL 보고서/summary는
[10_FINAL_ARCHITECTURE_AUDIT.before.txt](evidence/c4_unification/10_FINAL_ARCHITECTURE_AUDIT.before.txt),
[audit_summary.before.json](evidence/c4_unification/audit_summary.before.json)에 보존한다.
`geometry_results.json`의 Global partial 반례도 수정 전 evidence이며 현재 결과로
덮어쓰지 않았다. **현재 미해결 C4/R4 mismatch는 없다.**

## 6. 불일치가 아닌 차이/주의사항

1. **version8 vs v7:** 최신 Adaptive 변경을 반영한 정당한 checkpoint identity이며
   train/inference35개 normalized architecture field가 정확히 같다.
2. **YAML aliases:** Train은 `k/scale`, Inference는 `local_partition_k/compression_scale`이다.
   `inference._load_models`가 명시적으로 rename한 뒤 같은 `ArchitectureConfig`를 만든다.
   raw YAML key equality가 아니라 이 실제 parser 의미를 비교했다. unknown legacy
   architecture key를 silent-ignore하는 필터는 없다.
3. **known extent:** actual Anchor `[17:51,30:90)`를 geometry/mask/placement에 쓴다.
   cache 안 old `[17:50)` metadata key는 active mask source가 아니다.
4. **RGB vs latent footprint:**1080 RGB content와 native padded1088 footprint가 다르다.
   시각화는 native coordinate helper+visible boundary crop을 사용했다.
5. **spatial compression vs temporal mixing:** C2/C4는 T를 줄이지 않는다. R2/R4의
   depthwise3D kernel은 T 길이를 유지하면서 temporal neighbors를 혼합할 수 있다.
6. **source preservation:** Adaptive write가 Anchor를 바꾸지 않는 것과 모든 DiT hidden
   layer의 Anchor feature가 변하지 않는 것은 다른 주장이다. Native/Global/CA/FFN은
   known 위치 hidden도 문맥 처리하고 diffusion state는 known reinjection/t=0로 보존한다.
7. **dtype:** Training FP32 adapter masters+BF16 autocast와 Inference eval BF16 cast는
   같은 parameter topology이며 bitwise 동일 intermediate를 의미하지 않는다.
8. **legacy 설명:** `Cell.valid_count/nominal_count/support_ratio`는 descriptor의
   수량/진단 정보다. 현재 C2/C4에서는 count=s²이며 feature normalization에
   사용되지 않는다. 과거 partial counterexample log를 active implementation으로
   해석하지 않는다. position/LoRA/attention의 별도 알고리즘 변경도 하지 않았다.

## 7. Parameter ownership / count

실제 checkpoint의 **JSON header만** 읽어 hidden dimension4096, FFN inner16384,
block48 및 tensor shape/key를 확인했다. 전체 모델/메타모델을 만들거나 base tensor를
GPU/CPU에 materialize하여 count하지 않았다. 후속 ownership unit은 D=8의 두
attention container만 construction하며 전체 parameter count를 대체하지 않는다.

| category | 정적 count | ownership |
|---|---:|---|
| shared OP QKVO r32 | 50,331,648 | block별; QKV는 N/A/G 공유, O는 최종1회 |
| Adaptive QKV r64 | 75,497,472 | block별; Local별 duplication 없음 |
| Global QKV r64 | 75,497,472 | block별 |
| shared FFN r32 | 62,914,560 | native FFN 두 Linear |
| C2/C4 | 4,325,376 | density별/block별; C4는 Adaptive+Global 공유 |
| R2/R4 | 11,010,048 | 서로 독립 instance/block별 |
| Local Fusion | 393,216 | gamma/beta/block별 |
| Local–Global Fusion | 393,216 | gamma/beta/block별 |
| **Trainable total** | **280,363,008** | adapter-only save set |
| Frozen transformer base | **18,988,838,144** | base SA/CA/FFN/modulation/output 및 native AV machinery |
| **Installed total** | **19,269,201,152** | frozen+trainable |

Header frozen count식:
`21,005,004,544 - 1,612,546,304 - 403,620,096 = 18,988,838,144`.
뺀 두 항은 installed transformer 밖의 video/audio text embedding connector다.
이 count를 “audio machinery를 모두 제거한 순수 video-only parameter subset”이라고
부르면 부정확하다. generation modality가 `audio=None`이라는 것과 구별한다.

`install_outpainting`의 freeze-all 후 새 delta/C/R/Fusion 설치,
`_outpainting_parameter_names`, `_setup_lora`의 trainable collection 및
`parameter_counts()` whitelist를 source로 대조했다. 실제 final training loss gradients/optimizer parameter updates는 실행하지 않았다.

후속 C4 unit은 같은 `compressors["4"]`를 두 consumer에서 호출하여 output equality와
`grad(L_A+L_G)=grad(L_A)+grad(L_G)`를 float64 `rtol=atol=1e-12`로 확인했다.
이 toy loss는 production branch loss를 새로 도입한 것이 아니다. 실제 production은
기존 **하나의 final loss**가 Local/Global fusion을 통해 동일 C4 parameter에 연결된다.

두 작은 block의 `named_parameters(remove_duplicate=False)`/state_dict 및 CPU
AdamW selection에서 duplicate0, C4 weight/bias 각각1회, C2와 C4 별도 object,
block간 C4 별도 object를 검사했다. CPU optimizer의 state는 비어 있고 step은 하지 않았다.

증거: [parameter_static_inspection.json](evidence/parameter_static_inspection.json).

## 8. Geometry / visualization 결과 요약 (동일 layout regression 재확인)

- K4:16 windows; K5:25 windows; 총41개 exact layout.
- nonempty36, early-exit5(K4=4,K5=1).
- read-support 외부/canvas 외부 index0.
- Adaptive partial C2/C4/C1=0/0/0.
- role overlap/hole=0/0.
- token read-support drop/duplication=0/0.
- Target write outside/drop/duplication=0/0/0.
- 최대 spatial KV372, Q135;9 latent frames에서는3348/1215.
- K4 canonical Local overlap0.
- K5 row27/54,240 positions가2번 read되며 count-normalized assembly.
- PNG **41개**, contact sheets **3개**.
- actual PMA coordinate helper3-case 최대 spatial error **0.0**, temporal exact.

전체표와통계: [09 문서](09_ADAPTIVE_GRID_GEOMETRY_AUDIT.md).
전체한눈보기: [all_41_cases.png](adaptive_grid_visualizations/all_41_cases.png).

## 9. Checkpoint compatibility와 runtime prerequisite

`model.adapter_state_dict()`는 `_outpainting_parameter_names`만 저장한다.
`OutpaintingTrainer._save_checkpoint()`가 architecture/provenance를 함께 기록하고,
Inference도 동일 `load_adapter()`를 쓴다.

`load_adapter`의 `strict=False`는 frozen base 전체를 adapter 파일에 요구하지 않기
위한 것이다. 그 앞에서 adapter expected-key set과 실제 key set을 **exact equality**로
검사하므로 missing/unexpected adapter tensor를 허용하는 silent partial load가 아니다.
base path, architecture, provenance, shape/type도 각각 reject한다.

- **static save/load contract: PASS**.
- config version5/6/7 rejection은 작은 config dataclass validation으로 확인.
- actual checkpoint roundtrip/key mapping 실행: **미실행/INCONCLUSIVE**. 이 작업에서
  missing/unexpected tensor count를 런타임0으로 측정했다고 주장하지 않는다.
- configured inference adapter:
  `outputs/train_4k/checkpoints/checkpoint-002000/adapter_model.safetensors`
  는 **최초 감사 stat상 없었다**. 학습 후 생성해야 할 artifact이며 architecture bug로 판정하지 않는다.
  이번 C4 작업은 이 경로/파일에 개입하지 않으며 fake adapter 생성, 다른
  checkpoint로 path 변경, 자동 변환을 하지 않았다.

## 10. 최신 검증 / 변경 범위

| 검사 | 결과 | 증거 |
|---|---|---|
| Selected CPU units | **62 PASS** | [cpu_unit_tests.log](evidence/c4_unification/cpu_unit_tests.log) |
| Full learned C2/C4 cell Conv arithmetic/order | PASS, float64 atol/rtol1e-12 | 위 로그 / new unit file |
| Canonical C4 average init | exact equality;510 cells;17×30 | 위 로그 |
| Invalid geometry/direct malformed cell | explicit ValueError, Conv 이전 | 위 로그 |
| 41 Adaptive layouts | PASS; partial/drop/duplication0 | new regression + 09 기존표 |
| Same C4 ownership/CPU selection/gradient sum | PASS | new unit file |
| Compile/import/YAML | PASS,22 Python files | [compile_and_config.log](evidence/c4_unification/compile_and_config.log) |
| Normalized Train/Inference config | PASS,35 fields/version8 | selected existing config unit |
| Ruff | PASS | [ruff.log](evidence/c4_unification/ruff.log) |
| Scope/parameter topology/R4 preservation | PASS | [verification_summary.json](evidence/c4_unification/verification_summary.json) |
| Documentation/PNG/link integrity | PASS,10 MD/41 PNG+3 sheets | [artifact_validation.json](evidence/artifact_validation.json) |

`main`은 Git worktree가 아니므로 git-clean이라고 주장하지 않는다. C4 변경 시작 시
source SHA256 snapshot과 세 source 파일의 before-text를 보존했다. 실제 diff와
hash 비교는 [production_changes.diff](evidence/c4_unification/production_changes.diff),
[source_before.json](evidence/c4_unification/source_before.json)에 대응한다.

의도한 production 수정은3파일뿐이다. 나머지 model/LoRA/Adaptive/Native/PMA,
train/inference, configs, cache orchestration은 source hash로 불변을 확인한다.
R2/R4 class body와 C2/C4 constructor/init도 byte-identical이다. Pinned source205개는
기존 snapshot과 동일하다. cache/data content를 재생성/수정하지 않았으며 실행 중인
precache/process에 어떠한 명령도 보내지 않았다.

원래 `source_immutability.json`의 source modifications0은 **이전 read-only 감사**의
기록이다. 현재 명시적 C4 bugfix까지 source modifications0이라고 주장하지 않는다.
전체 pytest에는 model forward/backward test가 있으므로 실행하지 않았다.

## 11. 종료 상태

- C4 operator: **single shared learned full4×4 primitive**.
- Adaptive C4: pure-B parent만 사용; 기존 C4-first/C2 promotion 유지.
- Global C4: full-canvas complete4×4 cells만; invalid H/W fail-closed.
- R4: 기존 fixed whole-grid bilinear + depthwise refinement 유지.
- Native/Adaptive/Global hierarchy, LoRA, masked Local fusion, shared O/CA/FFN,
  conditioning/cache/temporal policy: 변경 없음.
- 실제 training/inference/GPU/optimizer step: 미실행.

### ARCHITECTURE VERDICT

**PASS — architecture is fully specified and code-consistent**

최신 사용자 C4/R4 정의를 기준으로 한 static/CPU-unit 판정이다. 전체 production
training/inference 성능 또는 장기 품질 검증을 완료했다는 의미는 아니다.
