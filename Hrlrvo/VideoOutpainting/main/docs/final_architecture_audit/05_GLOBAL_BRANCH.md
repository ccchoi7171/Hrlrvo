# 05. Global Branch 정적 아키텍처 감사

## 0. 범위와 authoritative 계약

이 문서는 production source의 다음 경로를 정적으로 추적한다.

- `src/ltx_outpaint/global_attention.py`
- `src/ltx_outpaint/geometry.py::global_cells`, `cells_for_rect`
- `src/ltx_outpaint/compression.py::SpatialCompressor`, `SpatialDecompressor`
- `src/ltx_outpaint/native_local_attention.py::attend`
- `src/ltx_outpaint/model.py::MultiViewSelfAttention`

모델/attention forward, GPU, 실제 학습/inference 또는 optimizer step은 실행하지 않았다. 후속 CPU unit 검사는 작은 compressor 연산과 module/optimizer parameter-selection 검사만 수행했다(§10).

Global branch의 최종 계약은 다음과 같다.

1. Global C4와 Adaptive pure-Bridge C4는 각 DiT block의 기존 `compressors["4"]` **동일 module object**를 호출한다.
2. 두 caller는 동일 weight, bias, 초기화와 동일 `SpatialCompressor.forward()`를 사용한다. Global 전용 C4 operator나 parameter는 없다.
3. C4 입력은 항상 실제 canonical hidden의 complete `4×4` footprint다. zero padding, partial cell, 다른 cell에서의 borrowing, valid-count normalization과 alternate fallback은 없다.
4. Global `H` 또는 `W`가 4로 나누어떨어지지 않으면 geometry가 token을 만들기 전에 `ValueError`로 fail-closed한다.
5. R4는 **전체 Global C4 lattice**를 하나의 fixed rectangular grid로 해석하여 whole-grid bilinear resize 후 depthwise Conv3D refinement를 수행한다. R4는 per-token 독립 복원 연산이 아니며, per-token one-to-one 복원은 판정 기준이 아니다.
6. Global은 Local System과 동급인 유일한 full-canvas branch다. Global 내부에는 `W_O`, Cross-Attention 또는 FFN 복제가 없다.

## 1. Symbol과 shape

Global compression scale을 `S=4`로 둔다. 허용되는 geometry는 `H % S == 0`이고 `W % S == 0`인 경우뿐이다.

| Symbol | 의미 | Shape / 값 |
|---|---|---|
| `grid` | canonical block hidden | `[B,T,H,W,D]` |
| `cells` | row-major full C4 cells | 길이 `N = (H//4)·(W//4)` |
| `compressed` | C4 Global token grid | `[B,T,N,D]` |
| `(Gh,Gw)` | fixed compressed grid shape | `(H//4,W//4)` |
| `pe` | Global C4 positional encoding | LTX attention이 소비하는 RoPE tuple |
| `output` | Global SA output | flatten 시 `[B,T·N,D]`, restore 전 `[B,T,N,D]` |
| `GlobalFull` | R4 결과 | `[B,T,H,W,D]` |
| `LocalFull` | Local window fusion/assemble 결과 | `[B,T,H,W,D]` |
| `alpha_G` | native `vgate_msa` 기반 channel gate | `[B,T,H,W,D]`에 broadcast 가능 |
| `fused` | shared O 직전 결과 | `[B,T,H,W,D] → [B,T·H·W,D]` |

`ceil(H/4)` 또는 `ceil(W/4)`는 이 계약에 등장하지 않는다. 나머지가 있는 canvas는 작은 edge cell을 만들지 않고 거절된다.

## 2. End-to-end 실행 도식

```text
canonical grid [B,T,H,W,D]
        │
        ├─ geometry guard: H%4 == 0 and W%4 == 0
        │        └─ false → ValueError, no Global cells/tokens
        │
        ├─ global_cells(H,W,4)
        │        └─ N=(H//4)(W//4), every cell has 16 real indices
        │
        ├─ SAME block.compressors["4"] used by Adaptive pure-B
        │        └─ one depthwise Conv2d forward over full 4×4 cells
        │
        ├─ Global Q/K/V attention + Global Q/K/V LoRA
        │        └─ [B,T,N,D] ↔ [B,T·N,D]
        │
        ├─ block.decompressors["4"]
        │        ├─ fixed grid [H//4,W//4]
        │        ├─ whole-grid bilinear resize → [H,W]
        │        └─ full-volume depthwise Conv3D refinement
        │
        └─ GlobalFull [B,T,H,W,D]
                 │
                 ├─ lerp with LocalFull using alpha_G
                 └─ shared/native W_O exactly once
```

## 3. Per-block C4 ownership과 공유

각 `MultiViewSelfAttention` block은 scale별 compressor/decompressor를 한 벌만 소유한다.

```text
Block i
 ├─ compressors["2"] = C2_i
 │    └─ Adaptive promoted full C2 cells
 ├─ compressors["4"] = C4_i
 │    ├─ Adaptive pure-Bridge full C4 cells
 │    └─ Global full-grid C4 cells
 ├─ decompressors["2"] = R2_i
 │    └─ Adaptive Target-touching C2 query restore
 └─ decompressors["4"] = R4_i
      └─ Global fixed-grid restore
```

중요한 identity는 “같은 수식의 별도 module”이 아니라 **같은 `C4_i` object**라는 점이다.

- state namespace: `transformer_blocks.i.attn1.compressors.4.conv.{weight,bias}`
- Adaptive와 Global forward가 읽는 parameter storage: 동일
- 초기화: channel별 16 tap `1/16`, bias `0`
- gradient ownership: 동일 parameter에 두 caller의 사용이 누적됨
- Global-only C4 state key: 없음
- window별 C4 copy: 없음

Adaptive pure-B C4는 K/V context로 사용되며 R4로 복원되지 않는다. R4의 production caller는 Global branch다.

## 4. Global geometry: complete cells only

`global_cells(H,W,4)`는 full canvas `Rect(0,0,H,W)`를 `cells_for_rect()`에 넘긴다. geometry contract는 각 축의 extent가 scale로 정확히 나누어지는지를 먼저 검증한다.

허용된 경우 cell origin은 다음과 같다.

```text
y origins = 0,4,8,...,H-4
x origins = 0,4,8,...,W-4

for every cell:
    bounds      = (y,x,y+4,x+4)
    indices     = exactly 16 canonical flat indices
    offsets     = (0,1,...,15)
    valid_count = nominal_count = 16
```

`cells_for_rect()` 자체가 incomplete extent를 거절하므로 `global_cells()` 이외의 직접 caller도 같은 full-cell invariant를 우회할 수 없다. `global_cells()`도 full-canvas scale divisibility를 명시적으로 보장하는 geometry boundary다.

### 4.1 허용 예: 68×120

```text
H = 68 = 17·4
W = 120 = 30·4
Gh,Gw = 17,30
N = 510 tokens/frame
T = 9이면 Global attention sequence 길이 = 4,590
```

모든 token은 정확히 하나의 actual `4×4` hidden footprint를 읽는다.

### 4.2 거절 예: 17×120

```text
H = 17, W = 120, S = 4
H % S != 0
→ ValueError
```

이 경우 y=16에서 `1×4` cell을 만들지 않는다. 따라서 다음 항목도 발생하지 않는다.

- edge partial C4 token
- implicit/explicit zero padding
- adjacent cell hidden borrowing
- partial offset tuple
- `16/valid_count` rescaling
- 이름만 C4인 다른 operator

이것은 crop 또는 보정이 아니라 **unsupported geometry rejection**이다. 호출자는 shape를 조용히 바꾸지 않으며 invalid input은 attention과 R4에 도달하지 않는다.

## 5. C4의 유일한 forward operator

`SpatialCompressor(D,4)`는 다음 depthwise Conv2d 하나를 소유한다.

```text
Conv2d(
    in_channels=D,
    out_channels=D,
    kernel_size=4,
    stride=4,
    padding=0,
    groups=D,
    bias=True,
)
```

`forward(grid,cells)`는 각 cell의 scale, actual index count, full kernel offsets를 검증한다. 검증된 cell들을 gather하여 다음처럼 실행한다.

```text
[B,T,N,16,D]
 → permute/reshape [B·T·N,D,4,4]
 → the one self.conv call
 → reshape [B,T,N,D]
```

cell은 batch axis로 접히므로 서로 다른 cell의 hidden은 convolution support에서 섞이지 않는다. Adaptive와 Global의 차이는 caller와 token 배열뿐이며 C4 module, weight와 forward operator는 완전히 같다.

Incomplete/malformed cell은 `ValueError`다. 과거의 valid-tap `einsum`, represented-area normalization, offset별 alternate bucket path는 이 정의에 존재하지 않는다.

## 6. Global attention과 position

`global_attention()`은 compressed `[B,T,N,D]`를 `[B,T·N,D]`로 flatten하여 `attend()`에 전달한다. `phase_rope(context,cells,4,dtype)`는 canonical `Cell.indices`와 `Cell.bounds`를 사용하므로 packed sequence 번호를 공간 좌표로 오인하지 않는다.

Global Q/K/V는 다음과 같다.

```text
Q_G = W_Q(X_G) + ΔQ_OUT(X_G) + ΔQ_G(X_G)
K_G = W_K(X_G) + ΔK_OUT(X_G) + ΔK_G(X_G)
V_G = W_V(X_G) + ΔV_OUT(X_G) + ΔV_G(X_G)
```

- `W_Q/W_K/W_V`: frozen shared native projection
- `Δ*_OUT`: shared Outpainting Q/K/V LoRA
- `Δ*_G`: Global-specific Q/K/V LoRA
- Global-specific `W_O`: 없음

## 7. R4: fixed Global-grid 복원 정의

R4의 input unit은 개별 C4 token이 아니라 **시간 frame마다 구성된 전체 rectangular Global token grid**다.

```text
tokens           = [B,T,N,D]
compressed_shape = (H//4,W//4) = (Gh,Gw)
target_shape     = (H,W) = (4Gh,4Gw)
N                = Gh·Gw
```

`SpatialDecompressor.forward()`의 고정 연산은 다음과 같다.

```text
[B,T,N,D]
 → reshape full Global lattice [B·T,D,Gh,Gw]
 → bilinear resize of the WHOLE lattice to [B·T,D,H,W]
      mode="bilinear", align_corners=False, antialias=True
 → reshape volume [B,D,T,H,W]
 → depthwise Conv3d(kernel=3,stride=1,padding=1,groups=D)
 → GlobalFull [B,T,H,W,D]
```

이 연산에서 bilinear interpolation은 이웃 Global token을 결합할 수 있고 Conv3D는 공간 및 시간 이웃을 refinement할 수 있다. 이는 결함이나 fallback이 아니라 R4의 의도된 fixed Global-grid 정의다.

따라서 다음 요구는 R4에 적용하지 않는다.

- one C4 token을 다른 token과 독립적으로 정확히 한 4×4 block에 복제할 것
- cell 경계를 지나 공간 정보를 섞지 않을 것
- R2와 같은 cell-as-batch isolation을 사용할 것

R4는 arbitrary target resize API로 평가하지 않는다. C4 geometry가 divisibility를 보장하므로 production Global caller는 항상 `(Gh,Gw) → (4Gh,4Gw)` fixed-grid 복원만 요청한다. `n == Gh·Gw` 검증은 rectangular lattice shape를 보장한다.

Conv3D는 center tap 1, 나머지 0, bias 0으로 초기화된다. 따라서 초기 refinement는 bilinear 결과에 identity지만, 모든 spatial/temporal tap은 학습 가능하다.

## 8. Local–Global fusion과 shared 후단

R4 이후 `GlobalFull`과 assembled `LocalFull`은 모두 `[B,T,H,W,D]`다. `MultiViewSelfAttention.forward()`는 다음을 수행한다.

```text
alpha_G = sigmoid(gamma_G * native_vgate_msa + beta_G)
fused   = lerp(GlobalFull, LocalFull, alpha_G)
output  = shared_to_out(fused)
```

`torch.lerp(global_full,local_full,alpha_G)`는 `(1-alpha_G)·GlobalFull + alpha_G·LocalFull`이다. Global branch 내부에는 O projection이 없고, shared/native `attn1.to_out`은 두 fusion 이후 정확히 한 번 호출된다. 이후 native block이 SA residual, Text Cross-Attention과 shared FFN을 소유한다.

## 9. 정적 판정

| 항목 | 정적 판정 | 근거 |
|---|---:|---|
| Global은 유일한 full-canvas peer | PASS | `MultiViewSelfAttention.forward()`의 `local_full`/`global_full` fusion |
| Adaptive와 Global의 C4 object/weights 공유 | PASS | block별 단일 `compressors["4"]` |
| C4 forward operator 통일 | PASS | full-cell gather 후 단일 depthwise Conv2d |
| Global C4 actual full 4×4 only | PASS | geometry divisibility + compressor full-cell validation |
| partial/valid normalization path | 없음 | incomplete cell은 `ValueError` |
| invalid Global H/W fail-closed | PASS | cells/token/attention 생성 전 geometry rejection |
| R4 fixed whole-Global-grid 복원 | PASS | integer grid shape, whole-grid bilinear + depthwise Conv3D |
| Global 내부 O/CA/FFN 복제 | 없음 | shared 후단만 실행 |
| shared O exactly once | PASS | fusion 뒤 `self.to_out(fused)` |

### Overall Global Branch verdict

**PASS by static code contract.**

Global C4는 Adaptive pure-B C4와 module/weight/forward가 동일하고 actual full `4×4`만 소비한다. non-divisible geometry는 partial 의미로 축소되지 않고 fail-closed한다. R4는 개별 token 복원이 아니라 fixed Global lattice 전체에 대한 bilinear + depthwise Conv3D 복원으로 판정하며, 이 정의와 production call shape가 일치한다.

## 10. Evidence 상태

변경 전 source snapshot과 구현 계획은 `docs/final_architecture_audit/evidence/c4_unification/`에 보존한다.

- 선택한 CPU tests: **62 PASS**, [cpu_unit_tests.log](evidence/c4_unification/cpu_unit_tests.log).
- canonical full C4 `68×120 → 17×30`, malformed/nondivisible fail-closed, 41 Adaptive layouts 재검증 PASS.
- 작은 두 block을 construction하여 C4 object/state key/optimizer selection 중복0을 검사했다. 전체 LTX 모델을 만들거나 attention forward를 실행하지 않았다.
- 두 CPU compressor consumer의 `autograd.grad(L_A+L_G) == grad(L_A)+grad(L_G)` PASS. 이는 primitive 수준의 gradient ownership 검증이지 실제 final training loss backward 결과는 아니다.
- production attention/LoRA/R4 source 보존 및 Ruff/compile: [verification_summary.json](evidence/c4_unification/verification_summary.json).
