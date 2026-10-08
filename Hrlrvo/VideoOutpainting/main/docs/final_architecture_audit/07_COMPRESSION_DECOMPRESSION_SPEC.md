# 07. Compression / Decompression 정적 명세 감사

## 0. 목적과 최종 판정 기준

이 문서는 production C2, C4, R2, R4의 symbol, shape, parameter ownership, operator, 초기화와 경계 조건을 정적으로 명세한다.

```text
C2: full actual 2×2 canonical hidden → one token
C4: full actual 4×4 canonical hidden → one token
    Adaptive pure-B와 Global이 same per-block module/weights/forward를 사용
R2: each Target-touching C2 query → isolated full 2×2 restore, then T-only write
R4: entire fixed Global C4 lattice → whole-grid bilinear resize
    → full-volume depthwise Conv3D refinement
```

R2와 R4는 서로 다른 restore topology를 갖는다. R2는 cell-as-batch 복원이다. R4는 개별 C4 token의 독립 4×4 복원이 아니며, **whole Global grid가 연산 단위**다.

모델/attention forward, GPU, 실제 학습/inference와 optimizer step은 실행하지 않았다. 작은 CPU compressor/geometry/unit 검증의 범위와 증거는 §14에 따로 기록한다.

## 1. Module inventory와 ownership

각 `MultiViewSelfAttention` block은 `scale=4`일 때 다음 module object를 정확히 한 벌 생성한다.

| Module | State namespace | Production consumer | 공유 범위 |
|---|---|---|---|
| C2 | `attn1.compressors.2.conv` | Adaptive promoted full C2 | block별 1개, 모든 window/role 공유 |
| C4 | `attn1.compressors.4.conv` | Adaptive pure-B C4 + Global C4 | block별 1개, 두 branch와 모든 window 공유 |
| R2 | `attn1.decompressors.2.refine` | Adaptive Target-touching C2 output | block별 1개, 모든 query/window 공유 |
| R4 | `attn1.decompressors.4.refine` | Global output grid | block별 1개 |

별도 Anchor, Bridge, Target 또는 Global compressor는 없다. 특히 Adaptive C4와 Global C4는 유사한 두 module이 아니라 동일 `compressors["4"]` object를 호출한다.

```text
Block i
 ├─ C2_i ← every Adaptive scale-2 cell
 ├─ C4_i ← Adaptive pure-B scale-4 cells AND Global scale-4 cells
 ├─ R2_i ← each Adaptive Target-touching C2 query, cell-as-batch
 └─ R4_i ← one full Global compressed lattice
```

### 1.1 Parameter 수와 초기화

hidden dimension을 `D`, compressor scale을 `s`라 하면:

- `Cs` depthwise Conv2d weight: `D×1×s×s`
- `Cs` bias: `D`
- `C2 = D(4+1) = 5D`
- `C4 = D(16+1) = 17D`
- `Rs` depthwise Conv3d weight: `D×1×3×3×3 = 27D`
- `Rs` bias: `D`
- `R2 = R4 = 28D`

초기화는 다음과 같다.

```text
C2 taps = 1/4,  bias = 0
C4 taps = 1/16, bias = 0
R2/R4 Conv3D center tap = 1, other taps = 0, bias = 0
```

C2/C4 및 R2/R4는 서로 다른 parameter object다. 공유 identity는 **C4의 두 consumer 사이**에만 해당한다.

## 2. Cell descriptor와 universal full-cell invariant

`geometry.Cell`은 한 compressed token의 canonical support를 보존한다.

| Field | 의미 |
|---|---|
| `scale` | density class, production에서는 2 또는 4 |
| `role` | Adaptive pixel role bit union; Global은 0 |
| `y,x` | canonical top-left |
| `indices` | 실제 읽는 canonical spatial flat indices |
| `offsets` | depthwise kernel tap offsets |
| `bounds` | half-open bounds `(y0,x0,y1,x1)` |
| `pixel_roles` | Adaptive cell 각 tap의 A/T/B owner |
| `valid_count` | 실제 `indices` 수 |
| `nominal_count` | `scale²` |

모든 compressor input cell은 다음 invariant를 만족해야 한다.

```text
valid_count == nominal_count == scale²
offsets == (0,1,...,scale²-1)
bounds == (y,x,y+scale,x+scale)
indices are the exact row-major in-canvas positions implied by (y,x,bounds)
```

Adaptive와 Global 모두 이 규칙을 따른다. “Global만 partial edge cell을 허용”하는 예외는 없다.

## 3. Geometry producer의 fail-closed 규칙

### 3.1 Adaptive

Adaptive support는 canvas 안에서 4배수 extent로 확장되고 full C4 parent lattice로 분할된다. pure Bridge parent는 full C4 하나이며, A 또는 T가 한 pixel이라도 있는 parent는 네 개의 full C2 child로 승격된다.

- C2: 4 actual indices, offsets `(0,1,2,3)`
- C4: 16 actual indices, offsets `(0,...,15)`
- support를 full cell로 만들 수 없음: `ValueError`
- `adaptive_local()` compressor call 전에도 full-cell invariant 재검증

### 3.2 Global

Global은 `Rect(0,0,H,W)`의 full C4 lattice만 허용한다.

```text
required: H % 4 == 0 and W % 4 == 0
Gh = H//4
Gw = W//4
N  = Gh·Gw
```

`cells_for_rect()`는 전달된 rectangle의 height/width가 scale로 나누어떨어지지 않으면 `ValueError`를 낸다. `global_cells()`는 full canvas rectangle을 이 helper에 전달하므로 non-divisible Global geometry는 token construction 전에 종료된다. `CanvasGeometry` 자체와 pure Adaptive/unit subcanvas geometry API는 변경하지 않는다.

따라서 마지막 row/column을 `1×4`, `4×3` 또는 `1×3` C4라고 부르는 path가 없다.

### 3.3 Rejected behavior

다음은 C2와 C4 모두에서 금지된다.

- missing position을 zero로 채워 nominal cell 제조
- adjacent cell 또는 canvas 밖에서 hidden borrowing
- edge cell crop
- partial `indices`/`offsets` 허용
- valid-tap weighted `einsum`
- `scale²/valid_count` normalization
- partial shape별 fallback operator

Malformed cell은 계산 결과를 만들지 않고 `ValueError`로 종료한다.

## 4. SpatialCompressor: C2/C4 공통 단일 forward

`SpatialCompressor(D,s)`가 소유하는 유일한 operator는 다음 depthwise Conv2d다.

```text
Conv2d(
    in_channels=D,
    out_channels=D,
    kernel_size=s,
    stride=s,
    padding=0,
    groups=D,
    bias=True,
)
```

`forward()`는 cell scale, tap count/order, bounds와 bounds가 의미하는 canonical row-major indices를 검증한다. 그 뒤 shape는 다음과 같이 흐른다.

```text
grid [B,T,H,W,D]
 + N full cells, each s² canonical indices
 → gather [B,T,N,s²,D]
 → [B·T·N,D,s,s]
 → one self.conv call
 → [B,T,N,D]
```

여러 cell은 batch axis로 접힐 뿐, convolution은 한 cell의 `s×s` actual footprint만 본다. C2/C4의 차이는 `s`와 parameter object뿐이며 partial용 second operator는 없다.

## 5. C2 operator와 의미

Adaptive C4 parent가 A 또는 T에 닿으면 네 개 full C2로 전부 승격된다. B-only child도 promoted parent 안에서는 C2를 사용한다.

```text
one actual [D,2,2] footprint
 → C2 depthwise Conv2d once
 → one [D] token
```

초기에는 channel별 2×2 average다. 학습 후 tap weight는 변하지만 support 크기와 operator topology는 변하지 않는다.

### C2 정적 판정

- full actual 2×2: **PASS**
- padding/borrowing: **없음**
- partial alternate path: **없음**
- one full footprint → one token: **PASS**
- 기존 C2 promotion semantics: **변경 없음**

## 6. C4 operator 통일

> The per-block C4 compressor is a single shared learned spatial primitive used by both Adaptive and Global branches. It always maps a complete 4×4 current-hidden spatial cell to one coarse token. Partial, masked, or valid-count-normalized C4 operations are not part of the architecture.

### 6.1 Adaptive pure-B C4

16 pixel이 모두 Bridge인 full C4 parent는 C4 token 하나가 된다. support alignment와 runtime validation이 16 actual indices를 보장한다.

### 6.2 Global C4

Global은 `H//4 × W//4` row-major full cell lattice를 만든다. non-divisible H/W는 geometry에서 거절되므로 compressor에 partial Global cell이 전달되지 않는다.

### 6.3 Same object, same weights, same forward

두 경로는 다음 동일 호출 의미를 갖는다.

```text
Adaptive pure-B ─┐
                 ├─ block.compressors["4"]
Global full grid ┘      ├─ same conv.weight / conv.bias
                       ├─ same full-cell validation
                       └─ same [D,4,4] → depthwise Conv2d → [D]
```

branch 이름에 따른 조건 분기, Global 전용 weight, Adaptive 전용 base C4 또는 partial normalization은 없다.

### C4 정적 판정

| Path | Actual support | Operator | 판정 |
|---|---|---|---:|
| Adaptive pure-B | full 4×4 | shared C4 Conv2d | PASS |
| Global divisible H/W | full 4×4 only | same shared C4 Conv2d | PASS |
| Global non-divisible H/W | token 생성 없음 | geometry `ValueError` | PASS, fail-closed |
| malformed direct compressor cell | output 없음 | `ValueError` | PASS, fail-closed |

## 7. R2: isolated per-query full-cell restore

Target이 한 pixel이라도 있는 promoted C2가 Adaptive Q row다. A/B-only C2와 pure-B C4는 K/V context만 제공한다.

Attention output:

```text
[B,T,Q,D], where every Q is a full Target-touching C2 cell
```

Adaptive restore path는 query cell을 batch axis로 격리한다.

```text
[B,T,Q,D]
 → permute/reshape [B·Q,T,1,D]
 → R2(compressed_shape=(1,1), target_shape=(2,2))
 → [B·Q,T,2,2,D]
 → [B,T,Q·4,D]
```

R2 내부는 one-token bilinear `1×1→2×2` 후 depthwise Conv3D를 수행한다. 서로 다른 query cell은 R2 spatial grid에서 이웃이 아니므로 cell 간 공간 mixing이 없다. 시간축 `T`는 유지되며 Conv3D의 temporal neighbor mixing은 학습 가능하다.

R2는 full 2×2를 먼저 복원한다. 그 뒤 `write_source_indices`가 resolved Target owner tap만 선택해 original Local Target 좌표에 scatter한다. A/B tap은 read context에는 참여하지만 Adaptive output owner가 아니다.

### R2 정적 판정

- one C2 query → full 2×2 restore: **PASS**
- query cell 간 spatial isolation: **PASS**
- temporal Conv3D 유지: **PASS**
- restore 후 T-only write: **PASS**
- odd/remainder Target tap drop: **없음 by write-map contract**

## 8. R4: fixed whole-Global-grid restore

> R4 is one fixed Global-grid restoration operation: bilinear spatial upsampling followed by the existing depthwise Conv3D refinement. It is not required to decode each C4 token independently.

Global attention output 전체가 R4에 한 번 전달된다.

```text
H % 4 == 0, W % 4 == 0
Gh,Gw            = H//4,W//4
tokens           = [B,T,Gh·Gw,D]
compressed_shape = (Gh,Gw)
target_shape     = (4Gh,4Gw)
```

R4의 정확한 operator는 다음이다.

```text
[B,T,Gh·Gw,D]
 → [B·T,D,Gh,Gw]                 # whole compressed image
 → bilinear resize to [B·T,D,H,W]
      align_corners=False, antialias=True
 → [B,D,T,H,W]
 → depthwise Conv3d(k=3,s=1,p=1,groups=D)
 → [B,T,H,W,D]
```

`n == Gh·Gw` 검증이 token count와 rectangular lattice shape를 묶는다. `global_attention()`은 compressed shape를 integer division으로 구성하므로 ceil-based remainder grid가 없다.

### 8.1 R4와 R2의 차이

| 항목 | R2 | R4 |
|---|---|---|
| 연산 단위 | 개별 query cell | 전체 Global lattice |
| compressed spatial shape | 항상 `(1,1)` per cell | `(H//4,W//4)` |
| target shape | `(2,2)` per cell | full `(H,W)` |
| cell 간 spatial mixing | 없음 | bilinear/Conv3D에서 가능 |
| temporal Conv3D | 유지 | 유지 |

R4의 이웃 token 결합은 partial-cell 보정이나 evidence borrowing이 아니다. 모든 input token은 이미 full actual C4에서 생성되었고, 그 후 **Global reconstruction operator**가 full lattice를 공동 처리한다.

### 8.2 철회된 판정 기준

다음 기준은 R4 contract가 아니므로 FAIL 근거로 사용하지 않는다.

```text
one C4 token → an exclusive corresponding 4×4 block
no cross-cell interpolation or refinement
```

R4의 authoritative 정의는 기존 whole-grid bilinear + depthwise Conv3D다. valid production shape는 항상 정확한 4배 Global grid이므로 arbitrary non-divisible target 복원도 평가 대상이 아니다.

### R4 정적 판정

**PASS** — fixed `(H//4,W//4) → (H,W)` Global-grid 복원 정의와 구현이 일치한다.

## 9. Temporal semantics

Compression은 spatial-only다.

```text
C2/C4: [B,T,H,W,D] → [B,T,N,D]
```

`T`는 축소되지 않는다. R2/R4의 depthwise Conv3D는 `kernel=(3,3,3)`, `stride=1`, `padding=1`이므로 T 길이를 보존하면서 `t-1/t/t+1`을 학습 가능하게 혼합한다.

- C2/C4 temporal compression: 없음
- R2: per-cell temporal sequence 유지
- R4: full Global spatial/temporal volume refinement

## 10. Dtype와 initialization

### Compressor

- C2/C4는 설치 시 생성되는 trainable depthwise Conv2d다.
- 새 module은 base device로 이동하며 training master dtype 정책은 설치 코드와 runtime autocast가 관리한다.
- full cell은 실제 `self.conv` 경로 하나만 사용한다.
- partial tap weight cast나 manual `einsum` 경로가 없다.
- result는 input grid의 output buffer dtype으로 반환된다.

### Decompressor

- bilinear interpolation은 입력을 FP32로 올린 후 token dtype으로 되돌린다.
- depthwise Conv3D compute dtype은 runtime autocast/model dtype 정책을 따른다.
- R2/R4 center-identity 초기화는 interpolation 결과를 초기 상태에서 그대로 통과시킨다.
- 초기화는 topology를 바꾸지 않는다. R4는 초기 상태에서도 whole-grid bilinear 연산이다.

## 11. Parameter/checkpoint identity

C/R state namespace는 다음과 같다.

```text
transformer_blocks.i.attn1.compressors.2.conv.{weight,bias}
transformer_blocks.i.attn1.compressors.4.conv.{weight,bias}
transformer_blocks.i.attn1.decompressors.2.refine.{weight,bias}
transformer_blocks.i.attn1.decompressors.4.refine.{weight,bias}
```

C4 통일은 module을 추가, 복제 또는 rename하지 않는다. 따라서 valid full-cell input에 대한 parameter namespace, tensor shape, 초기화와 forward arithmetic identity는 유지된다. 바뀌는 것은 invalid partial geometry를 처리하던 의미뿐이며, 이제 그 geometry는 결과 대신 `ValueError`를 낸다.

## 12. Shape와 경계 조건 판정표

| 조건 | C2 | C4 | R2 | R4 |
|---|---|---|---|---|
| Adaptive promoted parent | full 2×2 | 해당 없음 | per-query full 2×2 | 해당 없음 |
| Adaptive pure-B parent | 해당 없음 | shared full 4×4 | 호출 안 함 | 호출 안 함 |
| Global 68×120 | 해당 없음 | `17×30` full C4 lattice | 해당 없음 | `17×30 → 68×120` whole-grid restore |
| Global 17×120 | 해당 없음 | geometry `ValueError`, no token | 해당 없음 | 호출되지 않음 |
| malformed partial cell 직접 전달 | compressor `ValueError` | compressor `ValueError` | 해당 없음 | 해당 없음 |
| T-empty Local | Local C/R 호출 없음 | Local C4 호출 없음 | 호출 없음 | Global은 독립적으로 실행 |
| batch >1 | tensor batch 보존 | tensor batch 보존 | `B·Q` folding, cell isolation | batch별 whole-grid 처리 |
| temporal T | 보존 | 보존 | 길이 보존 + temporal mix | 길이 보존 + temporal mix |

## 13. 최종 판정

### C2

**PASS.** 기존 C2 semantics는 변경되지 않았다. Adaptive C2는 actual full 2×2이며 depthwise Conv2d 한 번으로 token 하나가 된다.

### C4

**PASS.** Adaptive pure-B와 Global은 동일 block의 동일 `compressors["4"]` module/weights/forward를 사용하고 actual full 4×4만 소비한다. non-divisible geometry와 malformed partial cell은 fail-closed한다.

### R2

**PASS.** Target-touching query마다 기존 R2가 full 2×2를 cell-isolated 방식으로 복원하고 이후 T owner tap만 write한다.

### R4

**PASS.** 기존 whole-Global-grid bilinear + depthwise Conv3D를 유지하며, integer compressed grid에서 full canonical canvas를 복원한다. per-token exclusive 4×4는 요구사항이 아니다.

### Overall Compression/Decompression verdict

**PASS by static code contract.**

C2/R2는 변경 없이 유지된다. C4는 두 consumer가 단일 module/weight/forward를 공유하는 full-cell-only 연산이며, partial/valid normalization은 완전히 제거된다. R4는 fixed whole-Global-grid reconstruction이라는 정확한 정의로 판정한다.

## 14. Evidence 상태

`docs/final_architecture_audit/evidence/c4_unification/`에 변경 전 source snapshot, 계획과 후속 evidence를 보존한다.

- **62 CPU tests PASS**: [cpu_unit_tests.log](evidence/c4_unification/cpu_unit_tests.log).
- 실제 learned full-cell C2/C4 Conv2d와 개별 patch reference 비교: float64 `rtol=atol=1e-12` PASS. token order, B/T/feature shape 유지.
- canonical average-init C4 결과: `torch.equal`, `68×120 → 17×30`, partial/drop/duplication0.
- same-C4 consumer output/gradient-sum 비교 및 두 block의 parameter identity/state keys/CPU optimizer selection PASS. optimizer step/state allocation은 하지 않았다.
- R2/R4 class body는 변경 전과 byte-identical임을 확인했다. 이 작업에서는 R4 forward를 실행하거나 다른 decoder로 변경하지 않았다.
- 전체 LTX/attention forward, final diffusion loss backward, CUDA 실행은 하지 않았다. 전체 training 성능/VRAM/quality를 unit PASS로 주장하지 않는다.
- compile/import/config/Ruff와 변경 범위: [verification_summary.json](evidence/c4_unification/verification_summary.json).
