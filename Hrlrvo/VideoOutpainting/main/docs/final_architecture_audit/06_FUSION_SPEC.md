# 06. Fusion Specification 감사

## 판정

**SOURCE-STATIC PASS** — Local Fusion과 Local–Global Fusion은 서로 다른
두 단계이며, 둘 다 각 native block이 이미 계산한 `vgate_msa`를 재사용한다.
Local Fusion은 Adaptive write mask 안에서만 동작하고 그 밖은 exact Native
passthrough다. 최종 shared head gate와 `W_O`는 두 fusion 뒤 정확히 한 번이다.
이 판정은 source inspection이며 runtime 수치/GPU 검증을 뜻하지 않는다.

## native gate capture

pinned `BasicAVTransformerBlock.forward()`는 Self-Attention 전에 다음을
계산한다.

```text
(vshift_msa, vscale_msa, vgate_msa)
    = get_ada_values(scale_shift_table, B, video.timesteps, slice(0,3))
norm_vx = ada_zero(vx, vscale_msa, vshift_msa)
vx_msa_out = attn1(norm_vx, ...)
vx = post_sa(vx, vx_msa_out, vgate_msa)
```

Hrlrvo는 block class와 native block 연산 구현을 보존한다. 다만 instance의
`forward` binding은 `_forward_with_native_gate` wrapper로 교체하며, 이 wrapper가
`BasicAVTransformerBlock.forward(self, video, audio)`를 직접 호출하고 `finally`에서
ContextVar를 reset한다. instance-bound `get_ada_values()`가 정확히 `self.scale_shift_table`과 `slice(0,3)`인 호출의
세 번째 값만 `ContextVar`에 저장한다. replacement `attn1`이 이를 읽고 즉시
scope 값을 비워 CA/FFN까지 tensor를 보존하지 않는다. native forward는 같은
`vgate_msa`를 원래 SA residual에 다시 사용한다.

근거: [`model.py:30-54`](../../src/ltx_outpaint/model.py), pinned
[`transformer.py:191-200`](../../../models/LTX2.3/packages/ltx-core/src/ltx_core/model/transformer/transformer.py),
[`transformer.py:271-297`](../../../models/LTX2.3/packages/ltx-core/src/ltx_core/model/transformer/transformer.py).

## gate module과 shape

각 block은 서로 독립적인 `local_fusion`, `global_fusion`을 하나씩 가진다.

```text
alpha = sigmoid(beta + gamma * vgate_msa)
gamma:[D] = 0
beta:[D]  = logit(initial_weight)
```

현재 두 initial weight는 0.5이므로 beta도 0이고 초기 alpha는 모든 유한
token/channel에서 0.5다. gamma/beta는 trainable FP32 parameter다.

native gate 허용 shape:

```text
[B,1,D]          # token축 broadcast
[B,T*H*W,D]      # canonical token별 gate
```

`expand()` 후 `[B,T,H,W,D]`로 view/reshape하며 `repeat()`로 복제하지 않는다.
branch feature를 압축하더라도 gate는 canonical resolution 그대로 유지한다.
`NativeGateFusion.forward()`은 별도 timestep encoder나 D×D MLP 없이
`torch.addcmul`과 sigmoid만 사용한다.

## Fusion 1: per-window masked Local Fusion

window `L_j`에 대해 Native와 Adaptive는 모두
`[B,T,H_j,W_j,D]`를 반환한다. Adaptive가 함께 반환하는
`M_T:[B,1,H_j,W_j,1]`는 resolved Target write support다.

```text
alpha_L,j = sigmoid(beta_L + gamma_L * vgate_msa[L_j])

if M_T(p):
    L_j*(p) = alpha_L,j(p) * Native_j(p)
              + (1 - alpha_L,j(p)) * Adaptive_j(p)
else:
    L_j*(p) = Native_j(p)          # exact passthrough
```

실제 코드는 다음과 동치다.

```python
torch.where(M_T, torch.lerp(Adaptive, Native, alpha_L), Native)
```

따라서 Adaptive result의 unwritten zero가 Anchor/Bridge/empty-Target Local의
Native feature를 0.5로 감쇠하지 않는다. `M_T`는 T/D로 broadcast할 뿐 source
hidden 전체를 clamp/freeze/detach하지 않는다. Target support 안에서는 기존
channel-wise, token/timestep-conditioned fusion 식이 그대로다.

근거: [`model.py:258-276`](../../src/ltx_outpaint/model.py),
[`adaptive_local_attention.py:41-93`](../../src/ltx_outpaint/adaptive_local_attention.py).

## Local assemble

모든 `L_j*`를 계산한 뒤 각 window의 canonical flat indices에 `index_add_`한다.
별도 `count` tensor에도 동일 indices로 1을 더한 뒤 나눈다.

```text
LocalFull(p) = sum_j L_j*(p) / count(p)
shape        = [B,T,H,W,D]
```

이 단계는 uniform normalized overlap blend다. Adaptive/Native를 따로 assemble한
뒤 fusion하지 않으며, 이미 per-window fusion된 결과만 assemble한다.

## Fusion 2: full-canvas Local–Global Fusion

Global branch가 canonical resolution으로 복원한
`GlobalFull:[B,T,H,W,D]`와 `LocalFull`만 fusion한다.

```text
alpha_G = sigmoid(beta_G + gamma_G * vgate_msa)
FusedSA = alpha_G * LocalFull + (1 - alpha_G) * GlobalFull
```

실제 코드는 `torch.lerp(GlobalFull, LocalFull, alpha_G)`다. 초기 alpha가 0.5인
경우:

* resolved Target: Local 내부가 0.5 Native + 0.5 Adaptive이므로 최종 기여는
  0.25 Native + 0.25 Adaptive + 0.50 Global이다.
* Local mask 밖: Local은 exact Native이므로 최종 기여는
  0.50 Native + 0.50 Global이다.

이는 위치에 따라 정의된 **2단 hierarchical fusion**이며 flat 3-way fusion이
아니다. Global branch 내부의 compression/reconstruction 판정은
[05_GLOBAL_BRANCH.md](./05_GLOBAL_BRANCH.md)와
[07_COMPRESSION_DECOMPRESSION_SPEC.md](./07_COMPRESSION_DECOMPRESSION_SPEC.md)를 참조한다.

## STG blend와 native head gate

정상 hierarchy 후 `perturbation_mask`가 있으면 다음 blend가 추가된다.

```text
V_bypass = W_V(x) + Delta_OUT,V(x)
FusedSA   = FusedSA * perturbation_mask
            + V_bypass * (1 - perturbation_mask)
```

`all_perturbed=True`면 Native/Adaptive/Global과 두 fusion을 모두 건너뛰고
`V_bypass`에서 시작한다. 둘 다 official STG의 V-bypass/control-flow 의미를
유지하지만 shared Outpainting V delta는 포함한다. Adaptive/Global-specific V
delta는 bypass에 포함하지 않는다.

그 뒤 `to_gate_logits`가 존재할 때에만 native `gated_attention_function(x,
fused,self)`을 한 번 적용한다. 이 per-head gate는 `vgate_msa` 기반 channel
fusion과 별개이며, 각 branch에 복제되지 않는다.

## shared `W_O` exactly once

마지막 호출은 항상 다음 하나다.

```text
self.to_out(fused)
  = [W_O(fused) + Delta_OUT,O(fused)] -> native Identity
```

base `to_out` Sequential과 그 첫 Linear 객체는 pretrained 객체 그대로이며
frozen이다. `Delta_OUT,O`만 trainable하다. Native/Adaptive/Global에는 별도 O가
없다. 반환 후 pinned native block이 `post_sa_function(...,vgate_msa)`으로 SA
residual을 한 번 수행한다.

## parameter 소유권

한 block 기준:

| parameter | shape | 상태 |
|---|---:|---|
| Local gamma, beta | 각 `[D]` | trainable |
| Global gamma, beta | 각 `[D]` | trainable |
| native `scale_shift_table`, AdaLN | native shape | frozen |
| native `vgate_msa` | activation | parameter 아님 |
| base `W_O`/bias | native shape | frozen |
| shared Outpainting O LoRA r32 | low-rank | trainable |
| optional `to_gate_logits` | native shape | frozen |

## 정적 감사 표

| 검사 | 판정 | 근거 |
|---|---|---|
| Local Fusion이 per-window | PASS | window loop 내부에서 fusion 후 append |
| non-Target Local이 exact Native | PASS | `torch.where(mask,...,native)` |
| Local assemble이 fusion 이후 | PASS | `fused_windows`만 `index_add_` |
| overlap normalization | PASS | canonical sum / count |
| Global Fusion이 full canvas | PASS | `reshape_as(grid)` 양쪽 shape |
| `vgate_msa` 재사용 | PASS | exact native Ada slice capture |
| extra timestep MLP 없음 | PASS | gamma/beta affine만 존재 |
| head gate 1회 | PASS | fusion/STG 뒤 단일 call site |
| shared `W_O` 1회 | PASS | forward의 유일한 `self.to_out` |
| production runtime/GPU 수치 검증 | 미수행 | 이번 문서는 read-only source audit |
