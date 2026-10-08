# 09. Adaptive grid geometry 전수 감사 — 실제 helper로 계산한 41 cases

## 1. 범위·판정·재현 증거

**ADAPTIVE GEOMETRY: PASS — K4 16개 + K5 25개, 총 41개.**

이번 계산은 현재 `src/ltx_outpaint/geometry.py`의 `local_windows`, `anchor_patch`,
`adaptive_layout`, `align_adaptive_support`, `resolve_adaptive_roles`,
`build_adaptive_multires_tokens`를 직접 import하여 수행했다. Layout에 들어 있는
`Cell.indices`, `offsets`, `pixel_roles`, `write_source_indices`,
`write_target_indices`를 검사했다. 그림의 window/support/token 위치를 수작업으로
배치하지 않았다.

- 현재 architecture metadata: **8**. v7 LoRA/Fusion hierarchy를 보존하면서 Adaptive full-cell promotion/write semantics를 version 8로 식별한다.
- **model instance 생성 0; model forward/backward 0; GPU/CUDA context 초기화 0.**
- R2/Attention을 실행하지 않았다. R2 이후 위치를 의미하는 좌표라벨 CPU tensor와 실제 layout의 write indices로 **scatter 주소 대응만** 검증했다.
- Train/Eval layout 비교는 mode를 받지 않는 동일 pure geometry helper의 결정성 비교다. full Train/Eval model output 비교가 아니다.
- Global C4는 후속 수정에서 full-cell-only/fail-closed로 통일했다. R4는 최종 정의인 fixed Global-grid restoration으로 재판정했다. 본문의 41 Adaptive layouts/PNG는 변경되지 않았으며 새 CPU regression에서도 동일하다(§9).

증거:

- [geometry_results.json](evidence/geometry_results.json): 모든 원시 bbox/count/통계/위반 수.
- [geometry_run.log](evidence/geometry_run.log): 완료된 최종 실행 출력.
- [geometry_audit_script.txt](evidence/geometry_audit_script.txt): `/tmp`에서 실행한 one-off Python의 재현용 사본. production source/test에 코드를 추가하지 않았다.
- [pma_coordinate_results.json](evidence/pma_coordinate_results.json): 별도 작은 CPU 좌표 helper 3-case 확인; Attention/RoPE 적용 forward 아님.

실행 환경은 `Hrlrvo` Python, `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONPATH=src`다. 개별 geometry PNG는 41개이며, contact sheet 3개는 이 수에 포함하지 않는다.

## 2. 실제 coordinate source of truth

### 2.1 입력과 native scale

사용 record ID:
`0002e965-b236-4ec5-846d-4d1b919cf3f0.pt`.

읽은 파일은 다음이며 **read-only CPU mmap**으로 shape/provenance만 확인했다.

- GT: `/home/work/ai-graphics_33/Hrlrvo/cache_Data/train/latents/0002e965-b236-4ec5-846d-4d1b919cf3f0.pt`
- Anchor: `/home/work/ai-graphics_33/Hrlrvo/cache_Data_Anchor_2K/train/latents/0002e965-b236-4ec5-846d-4d1b919cf3f0.pt`
- GT contract: `/home/work/ai-graphics_33/Hrlrvo/cache_Data/train/cache_contract.json`

GT full cache `[128,11,68,120]`, Anchor `[128,9,34,60]`, 둘 다 BF16다.
학습 slice는 GT 첫 9 latent이고 source 시간은 RGB `[0:65)`다.

`ltx_trainer.model_loader.read_video_scale_factors()`는 설정된 pretrained checkpoint의
**header metadata만** 읽어 `(time=8,height=32,width=32)`를 반환했다. VAE/model을 load하지
않았다. `(65-1)//8+1=9`를 확인했다.

### 2.2 내부 canvas와 visible canvas

`inference._vae_boundary()`를 실제 inference YAML의 `video.canvas`와 위 native
scale에 적용했다. GT cache contract의 padding과 대조했다.

| 좌표계 | H×W / 범위 |
|---|---|
| 원본/표시 RGB | 2160×3840 |
| native aligned RGB | 2176×3840 |
| GT top/bottom padding | 8 / 8 RGB rows |
| canonical latent | 68×120 |
| exact source RGB Anchor content | y `[540:1620)`, x `[960:2880)`; 1080×1920 |
| aligned canvas상의 RGB content | y `[548:1628)`, x `[960:2880)` |
| Anchor independent VAE input | 1088×1920; reflect top/bottom 4/4 |
| aligned encoded Anchor envelope | y `[544:1632)`, x `[960:2880)` |
| actual Anchor latent extent | y `[17:51)`, x `[30:90)`; 34×60 |
| 그 latent footprint를 visible RGB로 투영 | y `[536:1624)`, x `[960:2880)` |

**1080-row content와 1088-row encoded footprint는 다른 것**이다. 그림의 orange/red
Anchor는 Adaptive가 실제 사용하는 **latent known extent**를 나타내므로 표시상
1088-row footprint가 된다. 원본 content crop을 1088로 바꾼 것이 아니다.

변환은 단순히 그림 좌표에 임의 상수 32를 곱한 것이 아니다.

1. canonical rectangle을 `[B,3,N,2]` latent-bound tensor로 구성.
2. pinned `ltx_core.components.patchifiers.get_pixel_coords(..., actual_scale_factors, causal_fix=True)` 호출.
3. production decode의 crop과 같은 `PixelBoundary.top/left`를 차감.
4. **표시에서만** requested visible canvas와 clip. 내부 geometry는 그대로 보존.

Temporal causal correction은 native helper에 남아 있으며 공간 그림은 한 temporal
slice를 표시한다. 위쪽/아래쪽 token이 visible crop에서 일부 잘려 보일 수 있지만,
실제 compressed cell은 internal canonical canvas 안의 full cell이다.

### 2.3 provenance 주의사항

Anchor의 실제 `canonical_latent_rect_yxyx=[17,30,51,90]`와 tensor 34×60을 사용했다.
cache 안에는 과거 `known_latent_rect_yxyx=[17,30,50,90]`도 남아 있지만 **현재 active
mask/placement는 이 legacy key를 읽지 않는다**. `train._latent_anchor_rect()`는
actual extent를 계산·대조하고 `LazyAnchorCache.complete_batch()`도 canonical rect를
사용한다. 시각화에서도 legacy floor-end를 쓰지 않았다. cache는 수정하지 않았다.

GT contract hash는 파일 whitespace의 raw-byte hash가 아니라 production
`anchor_cache._canonical_json(contract)`의 SHA256이다. 해당 hash와 VAE SHA를
Anchor provenance/config와 대조했다. GT contract 안 과거 preprocessing label은
현재 읽은 cache provenance의 문자열일 뿐 과거 architecture를 가져온 근거가 아니다.

## 3. K4/K5 Local partition

`geometry.partition_axis(length,count)`:

```text
size = ceil(length/count)
start[j] = (j*(length-size) + (count-1)//2) // (count-1)
window[j] = [start[j], start[j]+size)
```

row-major Cartesian product가 `local_windows(H,W,k)`다. `MultiViewSelfAttention`
block index가 짝수면 K4, 홀수면 K5다. 총 48 blocks에서 각각 24 blocks다.

| partition | H axis intervals | W axis intervals | Local H×W | hole | overlap positions | extra reads | max multiplicity |
|---|---|---|---:|---:|---:|---:|---:|
| K4 | [0:17), [17:34), [34:51), [51:68) | [0:30), [30:60), [60:90), [90:120) | 17×30 | 0 | 0 | 0 | 1 |
| K5 | [0:14), [14:28), [27:41), [41:55), [54:68) | [0:24), [24:48), [48:72), [72:96), [96:120) | 14×24 | 0 | 240 | 240 | 2 |

K4는 68과120이 모두4로 나뉘므로 overlap이 없다. K5의 H축은 68/5가 정수가
아니므로 각 window 높이14, row **27과54**에서 최소 overlap이다. W축은24씩
정확히 나뉜다. 240 spatial positions가 두 Local에 포함되고, 별도의 hole이나
crop/resize/padding은 없다. 이 중복 read는 의도된 Local overlap이며 Adaptive
한 read-support 내부 token duplication 0과 혼동하면 안 된다.

`MultiViewSelfAttention.forward()`는 이미 Native+Adaptive fusion된 Local 결과를
canonical `index_add_`한 후 count로 나눈다. count가 모든 위치에서1이상임을 위
geometry로 확인했다. source-level 식은 06 문서에 있다.

### 3.1 전체 Native Local bbox

모든 표의 bbox 순서는 **y 먼저, x 다음**, half-open이다.

| partition | Local index | raw Local bbox | H×W |
|---|---:|---|---:|
| K4 | 00 | `[0:17, 0:30)` | 17×30 |
| K4 | 01 | `[0:17, 30:60)` | 17×30 |
| K4 | 02 | `[0:17, 60:90)` | 17×30 |
| K4 | 03 | `[0:17, 90:120)` | 17×30 |
| K4 | 04 | `[17:34, 0:30)` | 17×30 |
| K4 | 05 | `[17:34, 30:60)` | 17×30 |
| K4 | 06 | `[17:34, 60:90)` | 17×30 |
| K4 | 07 | `[17:34, 90:120)` | 17×30 |
| K4 | 08 | `[34:51, 0:30)` | 17×30 |
| K4 | 09 | `[34:51, 30:60)` | 17×30 |
| K4 | 10 | `[34:51, 60:90)` | 17×30 |
| K4 | 11 | `[34:51, 90:120)` | 17×30 |
| K4 | 12 | `[51:68, 0:30)` | 17×30 |
| K4 | 13 | `[51:68, 30:60)` | 17×30 |
| K4 | 14 | `[51:68, 60:90)` | 17×30 |
| K4 | 15 | `[51:68, 90:120)` | 17×30 |
| K5 | 00 | `[0:14, 0:24)` | 14×24 |
| K5 | 01 | `[0:14, 24:48)` | 14×24 |
| K5 | 02 | `[0:14, 48:72)` | 14×24 |
| K5 | 03 | `[0:14, 72:96)` | 14×24 |
| K5 | 04 | `[0:14, 96:120)` | 14×24 |
| K5 | 05 | `[14:28, 0:24)` | 14×24 |
| K5 | 06 | `[14:28, 24:48)` | 14×24 |
| K5 | 07 | `[14:28, 48:72)` | 14×24 |
| K5 | 08 | `[14:28, 72:96)` | 14×24 |
| K5 | 09 | `[14:28, 96:120)` | 14×24 |
| K5 | 10 | `[27:41, 0:24)` | 14×24 |
| K5 | 11 | `[27:41, 24:48)` | 14×24 |
| K5 | 12 | `[27:41, 48:72)` | 14×24 |
| K5 | 13 | `[27:41, 72:96)` | 14×24 |
| K5 | 14 | `[27:41, 96:120)` | 14×24 |
| K5 | 15 | `[41:55, 0:24)` | 14×24 |
| K5 | 16 | `[41:55, 24:48)` | 14×24 |
| K5 | 17 | `[41:55, 48:72)` | 14×24 |
| K5 | 18 | `[41:55, 72:96)` | 14×24 |
| K5 | 19 | `[41:55, 96:120)` | 14×24 |
| K5 | 20 | `[54:68, 0:24)` | 14×24 |
| K5 | 21 | `[54:68, 24:48)` | 14×24 |
| K5 | 22 | `[54:68, 48:72)` | 14×24 |
| K5 | 23 | `[54:68, 72:96)` | 14×24 |
| K5 | 24 | `[54:68, 96:120)` | 14×24 |

## 4. 각 Adaptive layout의 계산 정의

1. `resolved T = Local \ Anchor`. Anchor가 Local 전체를 포함하면 **즉시 종료**.
2. nonempty case만 `anchor_patch(anchor,local)`로 Local과 같은 H/W의 참고 patch를 선택.
3. `intended = MinBBox(selected_anchor_patch, raw_Local)`.
4. `align_adaptive_support()`가 extent 부족분 `(-extent)%4`만 추가한다.
   - `start+end <= canvas_extent`이면 positive 방향(오른쪽/아래); tie 포함.
   - 아니면 negative 방향(왼쪽/위).
   - 가능한 만큼 선호 방향에 넣고 remainder만 반대 방향.
   - origin은 global 4-grid로 snap하지 않는다.
5. expanded support 전체에서 full Anchor > raw Local > Bridge로 role을 결정.
6. pure-B parent만 C4; A/T가 한 tap이라도 있는 parent는 full C2 네 개.
7. T-touching C2를 packed 앞쪽에 배치. 모든 cell은 K/V에 남긴다.
8. `write_source_indices`는 query별 full2×2 tap 중 T만, `write_target_indices`는
   원래 Native Local에 대한 상대주소만 지정한다.

**Whole Anchor가 read support에 항상 전부 들어가는 구현은 아니다.** 의도 bbox의
Anchor는 selected reference patch이고, role resolution에서 expanded support와
whole Anchor의 교집합을 빨강으로 칠한다. support 밖의 알려진 Anchor는 실제로
주황색으로 남는다. 이를 전체 Anchor로 bbox를 인위 확장하지 않았다.

Empty case에는 `rectangle=None`이다. 표에서 intended/expanded/mod4가 `—`인 것은
실패가 아니라 tokenization 이전 early exit를 정확히 표현한 것이다. 그 case에
허구 bbox를 만들거나 `0×0 full cell`을 추가하지 않았다.

## 5. 41-case Adaptive geometry table

`A∩L`은 Anchor와 raw Local의 교집합, `A∩read`는 Anchor와 최종 read support의
교집합이다. resolved T가 L-shape이면 bbox만으로 모양을 대체하지 않고 그림의
pixel mask 및 JSON의 실제 support area를 따른다. token count는 **temporal frame당**이다.

| K | Local | Local bbox | A∩L | A∩read | raw T area | resolved T area | intended bbox | expanded bbox | H/W mod4 | H expansion | V expansion | C4 | C2 | Target Q | early exit | PNG |
|---|---:|---|---|---|---:|---:|---|---|---|---|---|---:|---:|---:|---|---|
| 4 | 00 | `[0:17, 0:30)` | — | `[17:36, 30:60)` | 510 | 510 | `[0:34, 0:60)` | `[0:36, 0:60)` | [0, 0] | none | down:2 | 56 | 316 | 135 | no | [grid4_local_00.png](adaptive_grid_visualizations/grid4_local_00.png) |
| 4 | 01 | `[0:17, 30:60)` | — | `[17:36, 30:62)` | 510 | 510 | `[0:34, 30:60)` | `[0:36, 30:62)` | [0, 0] | right:2 | down:2 | 0 | 288 | 135 | no | [grid4_local_01.png](adaptive_grid_visualizations/grid4_local_01.png) |
| 4 | 02 | `[0:17, 60:90)` | — | `[17:36, 58:90)` | 510 | 510 | `[0:34, 60:90)` | `[0:36, 58:90)` | [0, 0] | left:2 | down:2 | 0 | 288 | 135 | no | [grid4_local_02.png](adaptive_grid_visualizations/grid4_local_02.png) |
| 4 | 03 | `[0:17, 90:120)` | — | `[17:36, 60:90)` | 510 | 510 | `[0:34, 60:120)` | `[0:36, 60:120)` | [0, 0] | none | down:2 | 56 | 316 | 135 | no | [grid4_local_03.png](adaptive_grid_visualizations/grid4_local_03.png) |
| 4 | 04 | `[17:34, 0:30)` | — | `[17:37, 30:60)` | 510 | 510 | `[17:34, 0:60)` | `[17:37, 0:60)` | [0, 0] | none | down:3 | 0 | 300 | 135 | no | [grid4_local_04.png](adaptive_grid_visualizations/grid4_local_04.png) |
| 4 | 05 | `[17:34, 30:60)` | `[17:34, 30:60)` | — | 510 | 0 | — | — | — | N/A | N/A | 0 | 0 | 0 | YES | [grid4_local_05.png](adaptive_grid_visualizations/grid4_local_05.png) |
| 4 | 06 | `[17:34, 60:90)` | `[17:34, 60:90)` | — | 510 | 0 | — | — | — | N/A | N/A | 0 | 0 | 0 | YES | [grid4_local_06.png](adaptive_grid_visualizations/grid4_local_06.png) |
| 4 | 07 | `[17:34, 90:120)` | — | `[17:37, 60:90)` | 510 | 510 | `[17:34, 60:120)` | `[17:37, 60:120)` | [0, 0] | none | down:3 | 0 | 300 | 135 | no | [grid4_local_07.png](adaptive_grid_visualizations/grid4_local_07.png) |
| 4 | 08 | `[34:51, 0:30)` | — | `[31:51, 30:60)` | 510 | 510 | `[34:51, 0:60)` | `[31:51, 0:60)` | [0, 0] | none | up:3 | 0 | 300 | 135 | no | [grid4_local_08.png](adaptive_grid_visualizations/grid4_local_08.png) |
| 4 | 09 | `[34:51, 30:60)` | `[34:51, 30:60)` | — | 510 | 0 | — | — | — | N/A | N/A | 0 | 0 | 0 | YES | [grid4_local_09.png](adaptive_grid_visualizations/grid4_local_09.png) |
| 4 | 10 | `[34:51, 60:90)` | `[34:51, 60:90)` | — | 510 | 0 | — | — | — | N/A | N/A | 0 | 0 | 0 | YES | [grid4_local_10.png](adaptive_grid_visualizations/grid4_local_10.png) |
| 4 | 11 | `[34:51, 90:120)` | — | `[31:51, 60:90)` | 510 | 510 | `[34:51, 60:120)` | `[31:51, 60:120)` | [0, 0] | none | up:3 | 0 | 300 | 135 | no | [grid4_local_11.png](adaptive_grid_visualizations/grid4_local_11.png) |
| 4 | 12 | `[51:68, 0:30)` | — | `[32:51, 30:60)` | 510 | 510 | `[34:68, 0:60)` | `[32:68, 0:60)` | [0, 0] | none | up:2 | 56 | 316 | 135 | no | [grid4_local_12.png](adaptive_grid_visualizations/grid4_local_12.png) |
| 4 | 13 | `[51:68, 30:60)` | — | `[32:51, 30:62)` | 510 | 510 | `[34:68, 30:60)` | `[32:68, 30:62)` | [0, 0] | right:2 | up:2 | 0 | 288 | 135 | no | [grid4_local_13.png](adaptive_grid_visualizations/grid4_local_13.png) |
| 4 | 14 | `[51:68, 60:90)` | — | `[32:51, 58:90)` | 510 | 510 | `[34:68, 60:90)` | `[32:68, 58:90)` | [0, 0] | left:2 | up:2 | 0 | 288 | 135 | no | [grid4_local_14.png](adaptive_grid_visualizations/grid4_local_14.png) |
| 4 | 15 | `[51:68, 90:120)` | — | `[32:51, 60:90)` | 510 | 510 | `[34:68, 60:120)` | `[32:68, 60:120)` | [0, 0] | none | up:2 | 56 | 316 | 135 | no | [grid4_local_15.png](adaptive_grid_visualizations/grid4_local_15.png) |
| 5 | 00 | `[0:14, 0:24)` | — | `[17:32, 30:56)` | 336 | 336 | `[0:31, 0:54)` | `[0:32, 0:56)` | [0, 0] | right:2 | down:1 | 60 | 208 | 84 | no | [grid5_local_00.png](adaptive_grid_visualizations/grid5_local_00.png) |
| 5 | 01 | `[0:14, 24:48)` | — | `[17:32, 30:56)` | 336 | 336 | `[0:31, 24:54)` | `[0:32, 24:56)` | [0, 0] | right:2 | down:1 | 12 | 208 | 84 | no | [grid5_local_01.png](adaptive_grid_visualizations/grid5_local_01.png) |
| 5 | 02 | `[0:14, 48:72)` | — | `[17:32, 48:72)` | 336 | 336 | `[0:31, 48:72)` | `[0:32, 48:72)` | [0, 0] | none | down:1 | 0 | 192 | 84 | no | [grid5_local_02.png](adaptive_grid_visualizations/grid5_local_02.png) |
| 5 | 03 | `[0:14, 72:96)` | — | `[17:32, 64:90)` | 336 | 336 | `[0:31, 66:96)` | `[0:32, 64:96)` | [0, 0] | left:2 | down:1 | 12 | 208 | 84 | no | [grid5_local_03.png](adaptive_grid_visualizations/grid5_local_03.png) |
| 5 | 04 | `[0:14, 96:120)` | — | `[17:32, 64:90)` | 336 | 336 | `[0:31, 66:120)` | `[0:32, 64:120)` | [0, 0] | left:2 | down:1 | 60 | 208 | 84 | no | [grid5_local_04.png](adaptive_grid_visualizations/grid5_local_04.png) |
| 5 | 05 | `[14:28, 0:24)` | — | `[17:34, 30:56)` | 336 | 336 | `[14:31, 0:54)` | `[14:34, 0:56)` | [0, 0] | right:2 | down:3 | 11 | 236 | 84 | no | [grid5_local_05.png](adaptive_grid_visualizations/grid5_local_05.png) |
| 5 | 06 | `[14:28, 24:48)` | `[17:28, 30:48)` | `[17:34, 30:56)` | 336 | 138 | `[14:31, 24:54)` | `[14:34, 24:56)` | [0, 0] | right:2 | down:3 | 1 | 156 | 39 | no | [grid5_local_06.png](adaptive_grid_visualizations/grid5_local_06.png) |
| 5 | 07 | `[14:28, 48:72)` | `[17:28, 48:72)` | `[17:34, 48:72)` | 336 | 72 | `[14:31, 48:72)` | `[14:34, 48:72)` | [0, 0] | none | down:3 | 0 | 120 | 24 | no | [grid5_local_07.png](adaptive_grid_visualizations/grid5_local_07.png) |
| 5 | 08 | `[14:28, 72:96)` | `[17:28, 72:90)` | `[17:34, 64:90)` | 336 | 138 | `[14:31, 66:96)` | `[14:34, 64:96)` | [0, 0] | left:2 | down:3 | 1 | 156 | 39 | no | [grid5_local_08.png](adaptive_grid_visualizations/grid5_local_08.png) |
| 5 | 09 | `[14:28, 96:120)` | — | `[17:34, 64:90)` | 336 | 336 | `[14:31, 66:120)` | `[14:34, 64:120)` | [0, 0] | left:2 | down:3 | 11 | 236 | 84 | no | [grid5_local_09.png](adaptive_grid_visualizations/grid5_local_09.png) |
| 5 | 10 | `[27:41, 0:24)` | — | `[27:43, 30:56)` | 336 | 336 | `[27:41, 0:54)` | `[27:43, 0:56)` | [0, 0] | right:2 | down:2 | 4 | 208 | 84 | no | [grid5_local_10.png](adaptive_grid_visualizations/grid5_local_10.png) |
| 5 | 11 | `[27:41, 24:48)` | `[27:41, 30:48)` | `[27:43, 30:56)` | 336 | 84 | `[27:41, 24:54)` | `[27:43, 24:56)` | [0, 0] | right:2 | down:2 | 0 | 128 | 21 | no | [grid5_local_11.png](adaptive_grid_visualizations/grid5_local_11.png) |
| 5 | 12 | `[27:41, 48:72)` | `[27:41, 48:72)` | — | 336 | 0 | — | — | — | N/A | N/A | 0 | 0 | 0 | YES | [grid5_local_12.png](adaptive_grid_visualizations/grid5_local_12.png) |
| 5 | 13 | `[27:41, 72:96)` | `[27:41, 72:90)` | `[27:43, 64:90)` | 336 | 84 | `[27:41, 66:96)` | `[27:43, 64:96)` | [0, 0] | left:2 | down:2 | 0 | 128 | 21 | no | [grid5_local_13.png](adaptive_grid_visualizations/grid5_local_13.png) |
| 5 | 14 | `[27:41, 96:120)` | — | `[27:43, 64:90)` | 336 | 336 | `[27:41, 66:120)` | `[27:43, 64:120)` | [0, 0] | left:2 | down:2 | 4 | 208 | 84 | no | [grid5_local_14.png](adaptive_grid_visualizations/grid5_local_14.png) |
| 5 | 15 | `[41:55, 0:24)` | — | `[35:51, 30:56)` | 336 | 336 | `[37:55, 0:54)` | `[35:55, 0:56)` | [0, 0] | right:2 | up:2 | 18 | 208 | 84 | no | [grid5_local_15.png](adaptive_grid_visualizations/grid5_local_15.png) |
| 5 | 16 | `[41:55, 24:48)` | `[41:51, 30:48)` | `[35:51, 30:56)` | 336 | 156 | `[37:55, 24:54)` | `[35:55, 24:56)` | [0, 0] | right:2 | up:2 | 3 | 148 | 39 | no | [grid5_local_16.png](adaptive_grid_visualizations/grid5_local_16.png) |
| 5 | 17 | `[41:55, 48:72)` | `[41:51, 48:72)` | `[35:51, 48:72)` | 336 | 96 | `[37:55, 48:72)` | `[35:55, 48:72)` | [0, 0] | none | up:2 | 0 | 120 | 24 | no | [grid5_local_17.png](adaptive_grid_visualizations/grid5_local_17.png) |
| 5 | 18 | `[41:55, 72:96)` | `[41:51, 72:90)` | `[35:51, 64:90)` | 336 | 156 | `[37:55, 66:96)` | `[35:55, 64:96)` | [0, 0] | left:2 | up:2 | 3 | 148 | 39 | no | [grid5_local_18.png](adaptive_grid_visualizations/grid5_local_18.png) |
| 5 | 19 | `[41:55, 96:120)` | — | `[35:51, 64:90)` | 336 | 336 | `[37:55, 66:120)` | `[35:55, 64:120)` | [0, 0] | left:2 | up:2 | 18 | 208 | 84 | no | [grid5_local_19.png](adaptive_grid_visualizations/grid5_local_19.png) |
| 5 | 20 | `[54:68, 0:24)` | — | `[36:51, 30:56)` | 336 | 336 | `[37:68, 0:54)` | `[36:68, 0:56)` | [0, 0] | right:2 | up:1 | 60 | 208 | 84 | no | [grid5_local_20.png](adaptive_grid_visualizations/grid5_local_20.png) |
| 5 | 21 | `[54:68, 24:48)` | — | `[36:51, 30:56)` | 336 | 336 | `[37:68, 24:54)` | `[36:68, 24:56)` | [0, 0] | right:2 | up:1 | 12 | 208 | 84 | no | [grid5_local_21.png](adaptive_grid_visualizations/grid5_local_21.png) |
| 5 | 22 | `[54:68, 48:72)` | — | `[36:51, 48:72)` | 336 | 336 | `[37:68, 48:72)` | `[36:68, 48:72)` | [0, 0] | none | up:1 | 0 | 192 | 84 | no | [grid5_local_22.png](adaptive_grid_visualizations/grid5_local_22.png) |
| 5 | 23 | `[54:68, 72:96)` | — | `[36:51, 64:90)` | 336 | 336 | `[37:68, 66:96)` | `[36:68, 64:96)` | [0, 0] | left:2 | up:1 | 12 | 208 | 84 | no | [grid5_local_23.png](adaptive_grid_visualizations/grid5_local_23.png) |
| 5 | 24 | `[54:68, 96:120)` | — | `[36:51, 64:90)` | 336 | 336 | `[37:68, 66:120)` | `[36:68, 64:120)` | [0, 0] | left:2 | up:1 | 60 | 208 | 84 | no | [grid5_local_24.png](adaptive_grid_visualizations/grid5_local_24.png) |

## 6. 자동 invariant 검사 결과

| invariant | 결과 | 검사 방법 |
|---|---|---|
| H%4=0, W%4=0 | PASS, nonempty 36/36 | actual expanded `Rect` extent |
| inside full canvas | PASS | 모든 cell index와 bbox boundary |
| minimal extent expansion | PASS | added H/W가 정확히 `(-intended_extent)%4` |
| center direction / tie / remainder | PASS | helper source와 별도 deterministic probes |
| A>T>B ownership | PASS | 모든 support pixel의 독립 expected role과 비교 |
| mutually exclusive / no holes | PASS, overlap=0/hole=0 | role은 1/2/4 중 정확히 하나 |
| token read coverage | PASS, drop=0/duplication=0 | 한 support의 canonical index Counter |
| pure B only C4 | PASS | parent별 role과 emitted cell 교차검사 |
| A/T parent -> four C2 | PASS | parent별 child 수=4, 각 scale2 |
| promoted B child remains C2 | PASS | one-A/one-T 추가 probe와41개 parent 검사 |
| full C2 / full C4 | PASS | count=s², offsets=range(s²), bounds=s×s |
| partial C2 / partial C4 / C1 | 0 / 0 / 0 | emitted metadata 전수 검사 |
| Target Q eligibility | PASS | query prefix iff T in pixel_roles, scale2 |
| every emitted cell is K/V | PASS (source + layout) | `mixed_attention` K/V 전체 입력 |
| Target write only | PASS, wrong write=0 | query tap role 및 Local-relative destination 검사 |
| Target write drop/duplication | 0 / 0 | destination set==resolved T, unique |
| original Native Local scatter coordinates | PASS, exact integer equality | canonical coordinate label tensor의 주소 보존 |
| repeated helper layout | PASS 41/41 | 같은 immutable inputs의 dataclass equality |
| empty Target early exit | PASS 5/5 | support None; cells/Q/write 모두 없음 |

위 검사에서 context cell이 A/T/B hidden을 함께 읽는 것은 **허용된 full-cell
read**다. 과거의 semantic field-strict compression rule로 판단하지 않았다.
Target-only write invariant와 full-cell context read를 분리했다.

### 6.1 추가 pure geometry probes

| case | 결과 | 근거/actual |
|---|---|---|
| already_aligned | PASS | `{"input": [4, 4, 12, 12], "canvas": [20, 20], "actual": [4, 4, 12, 12]}` |
| left_and_above_expand_right_down | PASS | `{"input": [2, 2, 7, 7], "canvas": [20, 20], "actual": [2, 2, 10, 10]}` |
| right_and_below_expand_left_up | PASS | `{"input": [13, 13, 18, 18], "canvas": [20, 20], "actual": [10, 10, 18, 18]}` |
| tie_expand_right_down | PASS | `{"input": [7, 7, 13, 13], "canvas": [20, 20], "actual": [7, 7, 15, 15]}` |
| preferred_boundary_remainder | PASS | `{"input": [1, 1, 18, 18], "canvas": [20, 20], "actual": [0, 0, 20, 20]}` |
| pure_B_parent | PASS | `{"C2_C4_Q": [0, 1, 0]}` |
| one_T_promotes_entire_parent | PASS | `{"C2_C4_Q": [4, 0, 1]}` |
| one_A_promotes_B_only_subcells | PASS | `{"C2_C4_Q": [4, 0, 0]}` |
| mixed_ATTT_query | PASS | `{"C2_C4_Q": [4, 0, 1]}` |
| anchor_precedes_target | PASS | `{"4": 12, "1": 4}` |
| impossible_alignment_rejects | PASS | `"5x5 canvas fully occupied; no padding/crop fallback"` |

추가 probes는 41개 production PNG 수에 포함하지 않는다. impossible alignment는
expected ValueError를 확인하는 fail-closed 검사이며, production 41개 failure가 아니다.

### 6.2 PMA pure coordinate 검사

`position.target_query_positions()`를 C2/C4가 동시에 있는 3개의 translated small
geometry에 직접 적용했다. model/attention/rotary forward는 호출하지 않았다.

- `P_k=o/r_k+1/2`, `P_k^PMA=(P_k-1/2)(r_k/2)+1/2=o/2+1/2`.
- 실제 native stride를 이용한 spatial phase 최대 오차 **0.0**.
- spatial endpoint 두 개가 동일 nominal phase point임을 확인.
- original native temporal bounds는 exact equality; spatial scale로 나누지 않음.
- packed token order와 canonical position을 혼동하지 않음.

## 7. Token count distribution

모든 수는 spatial frame당 token 수이며 **early-exit의0도 통계에 포함**한다.
실제 T=9 attention sequence에서는 K/V length=9×spatial K/V, Q length=9×spatial Q다.
A-LoRA 입력 feature 차원D는 고정이며 sequence length만 변한다. dummy padding
sequence를 추가하지 않는다.

| group | metric | min | max | mean | median | sum |
|---|---|---:|---:|---:|---:|---:|
| K4 | C2 | 0 | 316 | 226.000000 | 294.0 | 3616 |
| K4 | C4 | 0 | 56 | 14.000000 | 0.0 | 224 |
| K4 | K/V | 0 | 372 | 240.000000 | 294.0 | 3840 |
| K4 | Target Q | 0 | 135 | 101.250000 | 135.0 | 1620 |
| K5 | C2 | 0 | 236 | 178.240000 | 208 | 4456 |
| K5 | C4 | 0 | 60 | 14.480000 | 4 | 362 |
| K5 | K/V | 0 | 268 | 192.720000 | 212 | 4818 |
| K5 | Target Q | 0 | 84 | 63.600000 | 84 | 1590 |
| all41 | C2 | 0 | 316 | 196.878049 | 208 | 8072 |
| all41 | C4 | 0 | 60 | 14.292683 | 1 | 586 |
| all41 | K/V | 0 | 372 | 211.170732 | 220 | 8658 |
| all41 | Target Q | 0 | 135 | 78.292683 | 84 | 3210 |

4×4 partition는 early-exit 4개, 5×5 partition는 early-exit 1개다. 전체 최대 한
Adaptive 호출의 spatial KV372, Q135이므로 9 latent positions에서 각각3348/1215다.
모든 Local의 count 합은 서로 다른 attention 호출 합계이며 한 attention sequence가 아니다.

## 8. Visualization 읽는 법과 파일

각 개별 PNG의 왼쪽 **3840×2160 영역**이 visible canvas이며 오른쪽1180px은
annotation panel이다(파일 전체5020×2160). contact sheets만 display thumbnail로
축소했고 model geometry와 무관하다.

- 검정: read support 바깥 canvas(whole known Anchor 제외).
- 주황: whole Anchor 중 현재 read support 밖.
- 적갈색/red fill: Anchor ∩ read support.
- 초록: Anchor 우선순위 적용 후 resolved T.
- 회색: read support 안 Bridge.
- 굵은 빨간 테두리: expanded Adaptive bbox.
- 노란 점선: intended bbox.
- 흰 점선: raw Native Local/Target.
- 밝은 parent grid: full4×4 parent.
- 가는 cyan subdivision: promoted parent의 full2×2 children.

노란/흰 점선 일부가 bbox 경계와 겹치는 것은 실제 boundary가 같기 때문이다.
Early-exit 이미지에는 orange Anchor와 raw Local outline만 있고 read bbox/token은
표시하지 않는다. 이것이 actual early return과 일치한다.

### Contact sheets

- [K4 전체16](adaptive_grid_visualizations/grid4_all_16.png)
- [K5 전체25](adaptive_grid_visualizations/grid5_all_25.png)
- [41개 전체](adaptive_grid_visualizations/all_41_cases.png)

![K4 전체16](adaptive_grid_visualizations/grid4_all_16.png)

![K5 전체25](adaptive_grid_visualizations/grid5_all_25.png)

## 9. Global C4 통일 이후 경계 검사

현재 `global_cells → cells_for_rect`는 H/W가4배수가 아니면 cell 생성 전에
`ValueError`를 발생시킨다. `SpatialCompressor`도 incomplete/masked/noncanonical
descriptor를 Conv 호출 전에 거부한다. `CanvasGeometry`를 모든 독립 Adaptive
geometry에 대해 축소 제한한 것이 아니라 **Global cell producer에서** 거부하는 계약이다.

| canonical H×W | 현재 결과 | full C4 | partial C4 | compressed H×W |
|---|---|---:|---:|---|
| [68,120] | PASS | 510 | 0 | [17,30] |
| [17,20] | ValueError: H/W not divisible by4 | 생성 안 됨 | 0 | 없음 |
| [5,5], [67,120], [68,119], [1,4] | 동일 fail-closed | 생성 안 됨 | 0 | 없음 |

새 검증 `tests/test_shared_full_cell_compression.py`는 small CPU C4 Conv 연산으로
canonical 평균 초기화/shape, learned weights, caller별 동일 footprint 결과,
parameter identity, invalid geometry를 검사했다. 41 Adaptive layouts도 다시 계산하여
C2/C4/Q 합계, support coverage와 Target write가 기존 결과와 같음을 확인했다.
선택한 CPU 검사는 총 **62 PASS**다. 증거:
[cpu_unit_tests.log](evidence/c4_unification/cpu_unit_tests.log).

`evidence/geometry_results.json::global_general_path_evidence`의17×20 partial5개
반례와 원래 one-off `.txt` 스크립트의 Global probe는 **수정 전 역사적 증거**로
남긴다. 이를 현재 지원 geometry로 해석하거나 이전 스크립트 전체를 현재 성공
기대값으로 실행하면 안 된다. 그 파일의41 Adaptive layout/visualization 데이터는
최신 regression으로 재확인했다. Global Attention/R4/model forward는 실행하지 않았다.

R4는 기존 whole-grid bilinear interpolation + full-canvas depthwise refinement를
그대로 유지한다. 사용자 최종 정의에 따라 이것이 단일 Global-grid operation이며,
R2의 cell-as-batch primitive와 같아야 한다는 이전 판정 기준은 적용하지 않는다.

## 10. 검증 한계와 감사 도구 수정 내역

- model numerical output, trained checkpoint validity, attention equivalence, gradients, VRAM은 이번 실행 범위 밖이다.
- 입력 cache tensor 값 equality나 VAE re-encode 검사를 하지 않았다; 요구사항도 geometry/provenance 대응이지 GT crop과 Anchor latent equality가 아니다.
- 최초 one-off audit script의 raw-file contract hash 비교는 production canonical-JSON hash와 달라 실패했다. 실제 `_canonical_json` helper 사용으로 **감사 스크립트만** 바로잡았다.
- 추가 synthetic direction probe의 기대 rectangle 하나가 잘못되었고, preferred2/remainder1이 필요한 정확한 testcase로 **감사 스크립트만** 수정했다.
- 이 두 준비단계 오류 로그는 `evidence/geometry_audit_attempt01.log`, `geometry_audit_attempt02.log`에 남겼다. production cache/geometry bug나41-case failure가 아니다. 최종 완료 실행은 모두 PASS다.
- 원래 read-only 41-case 감사에서는 source/config/pinned/cache를 변경하지 않았다. 이후 이번 C4 통일 요청으로 geometry/compression/Global call site만 명시적으로 수정했으며, 현재 범위·diff·검증은 `evidence/c4_unification/`에 별도 보존한다. Adaptive helpers/config/pinned/cache는 변경하지 않았다.
