# 03. Native Local Branch 감사

## 판정

**SOURCE-STATIC PASS** — architecture metadata v8의 Native Local은 각 Local
window만 읽고, frozen LTX Q/K/V에 **공유 Outpainting Q/K/V LoRA**만 더한다.
Native-specific LoRA, branch별 `W_O`, branch별 Cross-Attention, branch별 FFN은
없다. 이 판정은 source inspection 결과이며 production GPU 실행 증거가
아니다.

## 진입점과 소유권

| 항목 | 실제 구현 | 소유권 |
|---|---|---|
| block별 분할 수 | `config.k + block_index % 2` | 비학습 geometry |
| Local window 생성 | `geometry.local_windows()` | 비학습 geometry |
| Native 실행 | `native_local_attention.native_local()` | 함수, parameter 없음 |
| base Q/K/V | 기존 `attn1.to_q/to_k/to_v` 객체 | pretrained, frozen |
| task delta | `attn1.shared_outpainting_lora.{q,k,v}` | trainable, 모든 view가 공유 |
| Native-specific delta | 없음 | 해당 parameter 0개 |
| Q/K normalization·RoPE | native `preattention_function` | pretrained/native, frozen |
| output projection | Native 안에서는 호출하지 않음 | 최종 fusion 뒤 공유 1회 |

근거: [`model.py:200-230`](../../src/ltx_outpaint/model.py),
[`native_local_attention.py:13-31`](../../src/ltx_outpaint/native_local_attention.py),
[`lora.py:29-35`](../../src/ltx_outpaint/lora.py).

## K4/K5 partition

canonical latent를 `X:[B,T,H,W,D]`라 하면 block index가 0부터 시작할 때:

```text
P_i = K + (i mod 2)
even block: P_i = 4 -> 4 x 4 = 16 windows
odd  block: P_i = 5 -> 5 x 5 = 25 windows
```

한 축 길이 `L`, partition 수 `P`에서:

```text
window_size = ceil(L / P)
span        = L - window_size
start_j     = floor((j * span + floor((P - 1) / 2)) / (P - 1))
end_j       = start_j + window_size
```

`P=1`은 `[0,L)` 하나다. 첫 window는 0에서 시작하고 마지막 window는 L에서
끝난다. 나누어떨어지지 않을 때 발생하는 overlap은 start에 균등 분산되며,
실제 hidden을 crop/resize/pad하지 않는다. 구현은
[`geometry.py:57-72`](../../src/ltx_outpaint/geometry.py)에 있다.

## 입력·출력 shape와 좌표

각 `Rect(y0,x0,y1,x1)` window에 대해:

```text
입력 grid          [B,T,H,W,D]
window view        [B,T,H_j,W_j,D]
attention sequence [B,T*H_j*W_j,D]
Native 출력        [B,T,H_j,W_j,D]
```

sequence index는 position으로 새로 만들지 않는다. `t*H*W + y*W + x`로
canonical flattened indices를 만들고, 그 indices로 LTX native RoPE tensor를
선택한다. 따라서 window 내부 순서는 `[T,H_j,W_j]` flatten이지만 position은
원래 full canvas의 `(t,y,x)`를 유지한다. 근거:
[`native_local_attention.py:24-31`](../../src/ltx_outpaint/native_local_attention.py).

## Q/K/V 식

한 Native sequence `x_N`에 대한 실제 projection은 다음과 같다.

```text
Q_N = W_Q(x_N) + Delta_OUT,Q(x_N)
K_N = W_K(x_N) + Delta_OUT,K(x_N)
V_N = W_V(x_N) + Delta_OUT,V(x_N)
```

`project_with_loras()`는 base projection과 각 delta를 **같은 입력**에
독립적으로 적용한 뒤 더한다. adapter를 직렬로 연결하지 않는다. base
projection 결과 dtype으로 반환한다. 공유 LoRA는 FP32 master parameter,
Kaiming-uniform A, zero B, bias 없음, `alpha/rank` scaling이다. Native에는
Adaptive/Global LoRA가 전달되지 않는다.

Q/K는 기존 `preattention_function`을 통해 native RMSNorm과 native RoPE를
거친다. 이후 선택된 native `attention_function`이 window 전체 temporal·spatial
sequence에 self-attention을 수행한다. Target, Anchor, generation mask를 이유로
Native sequence를 내부 분리하지 않는다.

## `W_O`가 Native에 없는 이유

`native_local()`은 attention head 결합 결과까지만 반환한다. `W_O`는 Local
Fusion, overlap-normalized Local assembly, Global Fusion, 선택적 STG blend와
native head gate를 지난 뒤 `MultiViewSelfAttention.forward()` 끝에서 정확히
한 번 호출된다. 따라서 Native window마다 `W_O`가 실행되지 않는다. 자세한
순서는 [06_FUSION_SPEC.md](./06_FUSION_SPEC.md)와
[08_FULL_ARCHITECTURE_FLOW.md](./08_FULL_ARCHITECTURE_FLOW.md)를 따른다.

## STG와 head-gate 예외

`all_perturbed=True`인 official STG bypass 호출에서는 Native/Adaptive/Global
hierarchy 전체를 실행하지 않는다. 이때 출력 후보는
`W_V(x)+Delta_OUT,V(x)`이며, 이후 optional native per-head gate와 공유
`W_O+Delta_OUT,O`는 그대로 한 번 실행된다. `perturbation_mask`만 주어진
일반 호출은 hierarchy를 먼저 계산한 다음 그 결과를 같은 shared-V bypass와
blend한다. Adaptive/Global-specific V LoRA는 bypass에 사용되지 않는다.

`to_gate_logits`가 존재하면 head gate는 Native branch 안이 아니라 최종 fused
attention tensor에 한 번 적용된다. 이것은 timestep-conditioned
`vgate_msa`/fusion gate와 다른 native attention head gate다. 근거:
[`model.py:233-296`](../../src/ltx_outpaint/model.py), pinned native
[`attention.py:512-579`](../../../models/LTX2.3/packages/ltx-core/src/ltx_core/model/transformer/attention.py).

## 정적 invariant 감사

| 검사 | 판정 | 근거 |
|---|---|---|
| even K4 / odd K5 | PASS | `partition_count = k + block_index % 2` |
| window 밖 hidden read 없음 | PASS | 입력 slice와 RoPE indices 모두 동일 Rect |
| base Q/K/V 객체 복제 없음 | PASS | native 객체를 replacement가 직접 재사용 |
| base Q/K/V frozen | PASS | 설치 전 `model.requires_grad_(False)` |
| Native-specific LoRA 없음 | PASS | shared LoRA만 `attend(..., lora=None)`에 적용 |
| Adaptive/Global LoRA 유입 없음 | PASS | Native 호출이 branch adapter를 전달하지 않음 |
| Native 내부 `W_O` 호출 없음 | PASS | `native_local()` 반환 전에 `to_out` 없음 |
| production runtime/GPU 수치 검증 | 미수행 | 이번 문서는 read-only source audit |

K4/K5 및 41-geometry 실행 증거는
[09_ADAPTIVE_GRID_GEOMETRY_AUDIT.md](./09_ADAPTIVE_GRID_GEOMETRY_AUDIT.md)에
분리한다.
