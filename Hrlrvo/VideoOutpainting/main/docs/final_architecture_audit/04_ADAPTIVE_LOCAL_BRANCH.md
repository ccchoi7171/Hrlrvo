# 04. Adaptive Local Branch 감사

## 판정

**SOURCE-STATIC PASS** — v8 Adaptive Local은 C4-first/C2-promotion layout에서
Target-touching C2의 Q만 계산하고, 모든 C2/C4 token을 K/V context로 사용한다.
각 query는 기존 `R2`로 완전한 2x2 footprint를 복원하지만 resolved Target
pixel만 Local 좌표에 기록한다. Anchor/Bridge write는 0이다. 이 판정은 source
inspection이며 production GPU 실행 증거가 아니다.

## 호출 경로

```text
adaptive_local(attention, grid, target_window, ViewContext)
  -> batch의 동일 Anchor geometry를 bucket
  -> adaptive_layout(anchor, target, H, W, S=4)
  -> C2/C4 full-cell compression
  -> mixed_attention(): Q_T-only, K/V_all
  -> query별 R2: [B,T,1,D] -> [B,T,2,2,D]
  -> resolved Target tap만 Local 좌표로 scatter
  -> (adaptive_result, write_mask)
```

주요 symbol은 [`adaptive_local_attention.py`](../../src/ltx_outpaint/adaptive_local_attention.py),
[`geometry.py:143-241`](../../src/ltx_outpaint/geometry.py),
[`position.py:52-96`](../../src/ltx_outpaint/position.py)에 있다.

## 1. receptive field와 C4 정렬

현재 target Local `T_j`와 geometry rule로 선택한 동일 크기 Anchor patch
`A_patch,j`의 최소 bbox를 먼저 만든다.

```text
raw_bbox = MinBBox(A_patch,j, T_j)
```

Anchor patch는 Anchor 안에 clamp된다. target이 Anchor 안에 완전히 포함되면
generation Target이 없으므로 bbox/patch/token을 만들기 전에 empty layout을
반환한다. 그 외에는 각 bbox 축 길이를 4의 배수가 되도록 **canvas 안에서만**
최소 확장한다. 확장 방향은 canvas center 쪽을 우선하며 정확한 center tie는
high coordinate(right/down) 쪽이다. lattice origin은 global C4 grid로 snap하지
않고 확장된 support의 실제 `(y0,x0)`다. 공간 padding, resize, crop은 없다.

근거: [`geometry.py:75-87`](../../src/ltx_outpaint/geometry.py),
[`geometry.py:154-174`](../../src/ltx_outpaint/geometry.py),
[`geometry.py:218-241`](../../src/ltx_outpaint/geometry.py).

## 2. role resolution: A > T > B

확장된 support의 모든 canonical pixel에 정확히 하나의 role을 준다.

```text
if pixel in full observed Anchor: ANCHOR
else if pixel in target Local:    TARGET
else:                             BRIDGE
```

여기서 Anchor role은 초기 bbox를 정한 selected patch만이 아니라 expanded
support와 **전체 observed Anchor rectangle의 교집합**이다. Anchor와 Target이
겹치면 Anchor가 우선하므로 해당 pixel은 Adaptive write Target이 아니다.
role은 generation/timestep mask와 별개의 Adaptive tokenization metadata다.

## 3. C4-first/C2 promotion과 read support

확장 support를 완전한 4x4 parent로 순회한다.

| parent 상태 | emitted token |
|---|---|
| 16 pixel 모두 Bridge | full C4 1개 |
| Anchor 또는 Target pixel 하나라도 존재 | full C2 4개 |

따라서 Adaptive에 partial cell은 없다. 각 emitted cell은 `scale²`개의 실제
canonical index와 `offsets=0..scale²-1`를 가져야 하며, runtime도 이 조건을
assert한다.

중요하게 **read support와 write support는 다르다**.

* C2/C4 feature read: 해당 full cell의 4/16 hidden 전체.
* query eligibility: C2의 네 pixel 중 resolved Target이 하나라도 존재.
* write support: query C2에서 resolved Target로 표시된 tap만.

즉 mixed-role C2는 Anchor/Bridge hidden까지 포함하여 하나의 query/K/V feature를
만들 수 있다. 그러나 복원 후 Anchor/Bridge pixel에는 쓰지 않는다. 이것은
과거 field-fragment/partial-support compression이 아니다.

## 4. packed tensor shape와 Target-query-only attention

한 Anchor bucket의 batch를 `B_a`, 전체 cell 수를 `N`, query cell 수를 `N_q`,
hidden dimension을 `D`라 하자. `build_adaptive_multires_tokens()`는 query cell을
앞에 배치한다.

```text
compressed tokens  [B_a,T,N,D]
all K/V input      [B_a,T*N,D]
Target Q input     [B_a,T*N_q,D]
attention output   [B_a,T*N_q,D]
restored query set [B_a,T,N_q,D]
```

실제 projection은 같은 입력에 대한 additive residual이다.

```text
Q_A = W_Q(x_query) + Delta_OUT,Q(x_query) + Delta_A,Q(x_query)
K_A = W_K(x_all)   + Delta_OUT,K(x_all)   + Delta_A,K(x_all)
V_A = W_V(x_all)   + Delta_OUT,V(x_all)   + Delta_A,V(x_all)
```

Anchor/Bridge Q는 계산하지 않는다. query C2 자신도 `x_all`에 남으므로 K/V self
context를 유지한다. 이것은 별도 reference/text Cross-Attention이 아니라 같은
Adaptive self-attention의 query row 제한이다. dense pairwise mask나 fabricated
sequence padding을 쓰지 않는다.

## 5. PMA와 temporal position

Adaptive query scale은 항상 `r_q=S/2=2`다. cell canonical origin을 `o`, key
scale을 `r_k`라 하면 코드가 만드는 nominal key position과 PMA는 다음과 같다.

```text
P_k       = o / r_k + 1/2
P_k^PMA   = (P_k - 1/2) * (r_k / r_q) + 1/2
          = o / r_q + 1/2
r_q       = 2
r_k       = 2 (promoted C2) or 4 (pure-Bridge C4)
```

x/y를 독립 적용한다. C2 query와 C2 key는 native nominal C2 phase를 사용하고,
C4 key는 `4/2` 비율로 query-native scale에 해석된다. semantic mixture, Target
coverage, centroid, packed index는 position을 바꾸지 않는다. 두 spatial
`[start,end)` endpoint에는 같은 phase point를 기록하여 LTX midpoint 추출이 그
point를 읽게 한다.

Temporal axis는 `context.positions`에서 canonical representative indices를
선택한 값을 그대로 둔다. spatial C2/C4 scale을 temporal seconds/bounds에
적용하지 않는다. 최종 RoPE frequency 생성, head layout, theta/max-position은
pinned LTX native `precompute_freqs_cis()`를 그대로 사용한다.

## 6. R2와 Target-only writeback

query들은 spatially dense rectangle이 아니므로 query index를 H/W 좌표처럼
배치하지 않는다. 각 query를 독립 batch item으로 바꾼다.

```text
[B_a,T,N_q,D]
 -> [B_a*N_q,T,1,D]
 -> R2((1,1) -> (2,2))
 -> [B_a,T,N_q*4,D]
```

`R2`는 bilinear interpolation 후 depthwise Conv3D다. 각 query의 완전한 2x2
결과 중 `layout.write_source_indices`가 선택한 Target tap만
`write_target_indices`로 `T_j`의 flattened 좌표에 `index_copy`한다.
결과와 mask는 다음 shape다.

```text
adaptive_result [B,T,H_j,W_j,D]
write_mask      [B,1,H_j,W_j,1]  # T와 D에 broadcast
```

Anchor/Bridge output contribution은 0이다. 이 zero는 fake input/padding token이
아니라 반환 map의 unwritten value다. Local Fusion은 mask가 false인 위치에서
Adaptive zero를 섞지 않고 Native를 그대로 통과시킨다. 자세한 식은
[06_FUSION_SPEC.md](./06_FUSION_SPEC.md)를 따른다.

## 7. empty Target early exit

target Local 전체가 Anchor 안이면 `adaptive_layout()`은 다음을 반환한다.

```text
rectangle=None, cells=(), target_count=0,
write_source_indices=(), write_target_indices=()
```

`adaptive_local()`은 그 bucket에서 compression, QKV, attention, R2를 실행하지
않고 zero result/false mask를 유지한다. 실제 존재하지 않는 bbox나 token을
만들지 않는다. 이 early exit는 canonical source hidden을 detach/freeze하지
않는다. 해당 Local의 Native path와 full-canvas Global path, 이후 native block
연산은 계속 미분 가능하게 실행된다.

## 8. parameter 소유권

| module | 상태 | 공유 범위 |
|---|---|---|
| base Q/K/V, Q/K RMSNorm | frozen | native `attn1` 객체 재사용 |
| shared Outpainting Q/K/V LoRA r32 | trainable | Native/Adaptive/Global 공용 |
| Adaptive Q/K/V LoRA r64 | trainable | block당 1벌, 모든 Local window 공유 |
| C2 compressor | trainable | block당 1개, 모든 promoted C2 공유 |
| C4 compressor | trainable | block당 1개, pure-Bridge Adaptive와 Global 공유 |
| R2 decompressor | trainable | block당 1개, Adaptive query restore |
| `W_O` | Adaptive 안에서 미호출 | 최종 shared path |

## 정적 invariant 감사

| 검사 | 판정 | source 근거 |
|---|---|---|
| C4-first/C2 promotion | PASS | `build_adaptive_multires_tokens()` |
| full C2/C4 real-cell read | PASS | `valid_count/offsets` runtime assertion |
| Q는 Target-touching C2만 | PASS | queries-first + `target_count` slice |
| 모든 emitted token이 K/V | PASS | K/V 입력이 packed 전체 `x` |
| Target query도 K/V 포함 | PASS | query cell을 packed에서 제거하지 않음 |
| Anchor/Bridge write 0 | PASS | write indices가 `role == TARGET`만 선택 |
| empty Target에서 fabricated token 없음 | PASS | layout early return + bucket `continue` |
| Local 밖 write 없음 | PASS | destination은 target-relative index만 생성 |
| production GPU/full-model 수치 검증 | 미수행 | 이번 문서는 read-only source audit |

41-geometry의 실행 기반 coverage/read/write 표와 grid visualization은
[09_ADAPTIVE_GRID_GEOMETRY_AUDIT.md](./09_ADAPTIVE_GRID_GEOMETRY_AUDIT.md)에
분리한다. Global의 full-cell-only C4 및 fixed Global-grid R4 판정은 각각
[05_GLOBAL_BRANCH.md](./05_GLOBAL_BRANCH.md),
[07_COMPRESSION_DECOMPRESSION_SPEC.md](./07_COMPRESSION_DECOMPRESSION_SPEC.md)를 따른다.
