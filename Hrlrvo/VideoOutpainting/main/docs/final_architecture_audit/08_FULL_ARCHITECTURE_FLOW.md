# 08. 전체 Architecture 실행 흐름 감사

## 판정

**HIERARCHY/CALL-ORDER: PASS** — architecture metadata v8에서 canonical diffusion
hidden은 하나이며, top-level Self-Attention hierarchy는 `Local System + Global`
이다. Local은 window별 `Native + Adaptive -> masked Local Fusion`, 이후
coordinate assemble/overlap normalization을 수행한다. 그 다음에만 Global과
fusion하고 shared head gate/`W_O`를 한 번 거쳐 native SA residual, frozen Text
CA, shared native FFN 순서로 진행한다.

**CURRENT ARCHITECTURE: PASS (static/CPU-unit scope)** — Global의 legacy partial
C4 경로를 제거했다. Adaptive/Global은 동일한 full-cell C4 primitive를 공유하고,
nondivisible Global H/W는 geometry 단계에서 거부된다. R4는 최종 사용자 정의에
따라 fixed whole-Global-grid bilinear + depthwise Conv3D로 판정하며 변경하지 않았다.
과거 per-token R4 정의에 따른 FAIL은 철회한다. 자세한 근거는
[05_GLOBAL_BRANCH.md](./05_GLOBAL_BRANCH.md)와
[07_COMPRESSION_DECOMPRESSION_SPEC.md](./07_COMPRESSION_DECOMPRESSION_SPEC.md)에
기록한다.

이 문서는 현재 source call graph를 감사한 결과다. production 22B GPU
forward/backward, numerical quality, VRAM 적합성을 PASS로 주장하지 않는다.

## 설치 범위

`install_outpainting()`의 동작:

1. `model.requires_grad_(False)`로 pretrained LTX 전체를 freeze한다.
2. 각 block이 정확히 pinned `BasicAVTransformerBlock`인지 확인한다.
3. 기존 video `block.attn1`의 Q/K/V/O, Q/K norm, head gate, attention callable을
   동일 객체로 재사용하는 `MultiViewSelfAttention`으로 교체한다.
4. shared Outpainting/Adaptive/Global LoRA, C/R, 두 fusion을 block별로 설치한다.
5. 기존 shared `ff.net.0.proj`, `ff.net.2` Linear 객체에 FFN LoRA만 설치한다.
6. native block forward와 `get_ada_values`에 instance-bound instrumentation을
   붙이되 class identity와 native execution body는 유지한다.
7. video preprocessor를 geometry-aware `OutpaintingPreprocessor`로 감싼다.

base Transformer, base QKV/O, Text CA, base FFN을 view별로 복제하지 않는다.
근거: [`model.py:299-323`](../../src/ltx_outpaint/model.py).

## canonical 입력과 immutable context

native block이 받는 video hidden은 `vx:[B,N,D]`, `N=T*H*W`다.
`OutpaintingPreprocessor`는 native preprocessor 결과의 positional embeddings를
다음 `ViewContext`로 교체한다.

```text
geometry  = CanvasGeometry(T,H,W,anchors)
positions = native LTX [B,3,N,2] bounds
sigma     = [B]
native_pe = native LTX RoPE tuple
rope      = native theta/max-pos/head-layout settings
```

geometry는 checkpoint recomputation argument 안에 immutable하게 전달된다.
batch마다 Anchor 하나 또는 sample별 Anchor `B`개를 허용한다. token 수와
`T*H*W`가 다르면 fail closed한다. audio modality는 이 entrypoint에서 받지
않는다. 근거: [`model.py:138-181`](../../src/ltx_outpaint/model.py).

## block 단위 exact call order

native `BasicAVTransformerBlock.forward()`와 replacement `attn1`을 합친 실제
video 경로는 다음과 같다.

```text
canonical block state vx [B,N,D]
  -> native get_ada_values(slice(0,3))
       vshift_msa, vscale_msa, vgate_msa
  -> native AdaLN/RMSNorm
       x = norm_vx [B,N,D]
  -> MultiViewSelfAttention(x, ViewContext)
       ├─ Local windows: even K4 / odd K5
       │    └─ per L_j
       │         ├─ Native Local SA
       │         ├─ Adaptive Target-query-only SA
       │         └─ Target-masked Local Fusion
       ├─ fused windows coordinate assemble + overlap normalization
       │    -> LocalFull [B,T,H,W,D]
       ├─ Global: C4 -> Global SA -> R4
       │    -> GlobalFull [B,T,H,W,D]
       ├─ Local–Global Fusion
       ├─ optional STG blend / native per-head gate
       └─ shared W_O + Delta_OUT,O exactly once
            -> vx_msa_out [B,N,D]
  -> native post_sa_function(vx, vx_msa_out, vgate_msa)
       -> SA residual + normalized state
  -> native Text Cross-Attention exactly once
       -> residual
  -> native FFN exactly once
       -> FFN residual
  -> next block canonical vx [B,N,D]
```

중요하게 `MultiViewSelfAttention`에 들어가는 `x`는 native normalized SA input이지만,
block-level canonical state는 원래 `vx` 하나다. 세 view가 독립 diffusion state나
세 Transformer branch를 만들지 않는다.

Pinned native 순서 근거:
[`transformer.py:271-297`](../../../models/LTX2.3/packages/ltx-core/src/ltx_core/model/transformer/transformer.py),
[`transformer.py:399-417`](../../../models/LTX2.3/packages/ltx-core/src/ltx_core/model/transformer/transformer.py).

## view별 Q/K/V와 shared post path

```text
Native:
  W_QKV + Delta_OUT,QKV

Adaptive:
  W_QKV + Delta_OUT,QKV + Delta_A,QKV
  Q = Target-touching C2 only
  K/V = all emitted C2/C4

Global:
  W_QKV + Delta_OUT,QKV + Delta_G,QKV

after both fusions:
  W_O + Delta_OUT,O                    # one call
  native SA residual                   # one call
  frozen native Text CA                # one call
  frozen native FFN + Delta_FFN        # one call
```

모든 delta는 같은 입력에 대한 additive residual이다. Native-specific LoRA,
Text-CA LoRA, branch-specific O/FFN은 없다. base parameters와 native modulation은
frozen이고 다음만 trainable하다.

* shared Outpainting Q/K/V/O LoRA r32;
* Adaptive Q/K/V LoRA r64;
* Global Q/K/V LoRA r64;
* shared FFN two Linear LoRA r32;
* C2/C4 compressors, R2/R4 decompressors;
* Local/Global fusion gamma·beta.

## Local System

K와 S는 독립이다. K4/K5는 Local window 수를 정하고, S=4는 Adaptive/Global
spatial density를 정한다.

1. Native는 해당 Local의 canonical hidden과 native positions만 읽는다.
2. Adaptive는 geometry-selected Anchor patch와 Target Local의 bbox를 canvas 안
   C4 support로 최소 확장한다.
3. A > T > B role resolution 후 pure Bridge C4는 유지하고 나머지 C4 parent는
   네 C2로 promote한다.
4. Target-touching C2 Q만 계산하고 모든 emitted C2/C4를 K/V로 사용한다.
5. query별 R2 결과에서 Target tap만 Local map에 쓴다.
6. `M_T` 안은 learned Local fusion, 밖은 exact Native passthrough다.
7. 모든 fused Local을 canonical 좌표에 더하고 overlap count로 나눈다.

세부사항은 [03_NATIVE_LOCAL_BRANCH.md](./03_NATIVE_LOCAL_BRANCH.md),
[04_ADAPTIVE_LOCAL_BRANCH.md](./04_ADAPTIVE_LOCAL_BRANCH.md),
[06_FUSION_SPEC.md](./06_FUSION_SPEC.md)를 참조한다.

## Global과 shared path 경계

Global은 Local System과 동급인 유일한 full-canvas peer다. configured C4
cell sequence에서 Global QKV를 계산하고 현재 `R4`가 반환한
`[B,T,H,W,D]` tensor를 LocalFull과 fusion한다. `global_cells`가 full4×4만
허용하므로 compressed shape은 정확히 `(H//4,W//4)`다. Adaptive와 Global의
C4 operator/weights는 같고, branch-specific LoRA와 attention에서만 해석이 다르다.
R4는 cell별 독립 복원이 아니라 full compressed grid bilinear resize와 full-canvas
Conv3D를 수행한다. 이것이 R4의 최종 single-operation 정의다. 상세 판정은
[05_GLOBAL_BRANCH.md](./05_GLOBAL_BRANCH.md),
[07_COMPRESSION_DECOMPRESSION_SPEC.md](./07_COMPRESSION_DECOMPRESSION_SPEC.md)에 맡긴다.

## Text Cross-Attention과 FFN

Self-Attention 반환 후 pinned block은 `post_sa_function`으로 native residual과
normalization을 수행하고, 그 normalized hidden을 `attn2`에 한 번 전달한다.
Text CA는 module에서 제거되지 않으며 base Q/K/V/O와 CA-only modulation은
frozen이다. `torch.no_grad()`로 감싸지 않으므로 CA parameter gradient는 없어도
hidden-state gradient는 앞선 trainable Self-Attention adapters로 흐를 수 있다.

이후 native `get_ada_values(slice(3,6))`, AdaLN, `self.ff`, `vgate_mlp` residual이
한 번 실행된다. base FFN/GELU/Sequential은 그대로이고 `ff.net.0.proj`와
`ff.net.2`에 shared FFN LoRA만 additive하게 설치된다. 따라서 FFN 횟수는
branch 수와 무관하게 block당 1회다.

## STG/control-flow 예외

일반 호출만 보고 “모든 invocation에서 세 view를 실행한다”고 단정하면
틀리다.

* `all_perturbed=True`: Local/Global hierarchy와 두 fusion을 생략하고
  shared `W_V + Delta_OUT,V` bypass를 사용한다.
* `perturbation_mask is not None`: 정상 hierarchy 결과와 같은 V bypass를
  elementwise blend한다.
* 이후 optional native per-head gate와 shared O는 여전히 한 번 실행된다.

이것은 pinned Attention의 official STG V-bypass control flow를 보존한 것이다.
다만 Hrlrvo에서는 shared Outpainting V/O delta가 포함된다. Adaptive/Global
branch LoRA를 STG bypass에 추가하지 않는다. native head gate는 fusion gate인
`vgate_msa`와 별도다.

## fail-closed 경계

다음 조건은 즉시 예외다.

* architecture metadata가 v8 exact contract와 다름;
* `attn1` 이외 호출인데 `context`/`k_pe`가 제공됨;
* dense pairwise SA mask가 제공됨;
* immutable `ViewContext`가 없음;
* captured native gate가 없거나 shape가 `[B,1,D]`/`[B,N,D]`가 아님;
* Adaptive cell이 complete C2/C4가 아님;
* Adaptive result/write-mask가 Local canonical shape와 다름;
* `torch.compile` capture가 시도됨.

official latent conditioning mask/timestep과 dense pairwise attention mask를
혼동하지 않는다.

## 정적 architecture invariant 감사

| invariant | 판정 | source 근거 |
|---|---|---|
| canonical diffusion hidden 하나 | PASS | native block `vx` 하나에 SA residual |
| top-level Local System + Global | PASS | window loop 완료 후 단일 Global call |
| per-window Native + Adaptive fusion | PASS | 동일 loop 내부 `torch.where` |
| non-Target Local exact Native | PASS | false branch가 `native` 객체 |
| all Local 후 assemble | PASS | `fused_windows` 수집 후 별도 loop |
| normalized overlap | PASS | `local_full / count` |
| Global 반환 tensor 후 fusion call order/shape | PASS | 양쪽 `[B,T,H,W,D]` |
| shared full-cell C4 / fixed Global-grid R4 | PASS | full-cell guard와 단일 shared Conv2d; R4 whole-grid 구현 유지 |
| shared native head gate 최대 1회 | PASS | fusion/STG 뒤 단일 call site |
| shared `W_O` 정확히 1회 | PASS | `return self.to_out(fused)` 한 곳 |
| native SA residual 1회 | PASS | pinned `post_sa_function` 한 곳 |
| native Text CA 1회 | PASS | pinned `_apply_text_cross_attention` 한 곳 |
| native FFN 1회 | PASS | pinned `self.ff` 한 곳 |
| base/CA frozen, hidden autograd 유지 | PASS | global freeze 후 no `no_grad` wrapper |
| production 22B GPU forward/backward | 미수행 | read-only source audit 범위 밖 |

실행 기반 41-geometry 결과와 validation 범위는
[09_ADAPTIVE_GRID_GEOMETRY_AUDIT.md](./09_ADAPTIVE_GRID_GEOMETRY_AUDIT.md)를 참조한다.
