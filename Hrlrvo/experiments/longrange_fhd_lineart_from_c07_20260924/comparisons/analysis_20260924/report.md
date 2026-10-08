# Generated-region-only comparison: LineArt Global prior vs baseline

## 1. 결론 — NO CLEAR OVERALL IMPROVEMENT

**지금 두 결과 중 하나를 운영 기준으로 고른다면 LineArt 없는 `strength_1p0` baseline을 유지한다.**
이는 baseline이 모든 위치·시간에서 압도적으로 낫다는 뜻이 아니다. 이번 LineArt 결과에는 오른쪽 창 내부의 잡선 감소와 일부 위치의 안정화가 있지만, 왼쪽 구조의 일관된 개선은 없고 오른쪽 생성 인물/장비 형태의 배경 가림이 더 커진다. 바닥과 천장의 기존 오류도 해결되지 않았다. **전체 생성 영역 품질을 합치면 baseline 근소 선호, 건축 구조만으로는 혼재/명확한 승자 없음**이 가장 정직한 판단이다.

- 종합 시각 판정 신뢰도: **Medium** — 한 영상·한 seed의 결과, 정답이 없는 outpainting 영역, 상충하는 장단점.
- 프레임 정렬·출력 동일성·실행 설정 확인 신뢰도: **High** — 저장된 RGB 배열·metadata·현재 실험 코드 근거.
- “차이를 전부 LineArt가 유발했다”는 인과 주장: **확정 불가** — C06 bootstrap에 한 번의 RGB precision 차이가 존재한다.
- 이번 분석은 CPU 정적 분석 및 기존 결과 분석만 수행했다. 새 inference·학습·production 코드 변경은 하지 않았다.

## 2. 무엇을 어떻게 비교했나

### 비교 입력

- LineArt ON MP4: `../../outputs/strength_1p0_global_lineart_from_c07/progress/C00_C23_1169f.mp4`
- baseline OFF MP4: `/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923/outputs/strength_1p0/C0_C23_1169f_1920x1080.mp4`
- 두 영상 모두 1920×1080, 24 fps, 1,169 frames, duration 48.708333 s. 마지막 프레임의 timestamp는 48.666667 s.
- 확대 비교/metric의 주 입력은 MP4가 아니라 **각 run의 pre-codec uint8 `.npy` RGB segments**다. MP4 압축이나 절반 해상도 미리보기의 차이를 구조 차이로 오인하지 않도록 했다. 여기서 lossless란 uint8 저장 후 손실이 없다는 의미이며, 모델 FP32 RGB 원본과 같은 precision이라는 뜻은 아니다.
- LineArt source segments: `../../chunks/strength_1p0_global_lineart_from_c07/segment_*.npy`
- baseline source segments: `/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923/chunks/strength_1p0/segment_*.npy`

### 비교 범위

중앙 anchor `(x=480:1440, y=270:810)`는 품질 평가에서 제외했다. 사각형은 끝 좌표 제외 표기다.

| 구역 | 원본 1920×1080 기준 ROI |
|---|---|
| 왼쪽 건축 | x=0:480, y=270:810 |
| 오른쪽 건축 | x=1440:1920, y=270:810 |
| 바닥 | x=0:1920, y=810:1080 |
| 바닥 중앙 확대 | x=480:1440, y=810:1080 |
| 상부/천장 | x=0:1920, y=0:270 |
| 동일성 검사용 anchor 내부 | x=512:1408, y=302:778 |

- 72개 고유 시점의 crop/contact sheet를 생성했다. 주요 timestamp와 C07–C23의 working start, prefix last, new first, new middle, new last를 포함한다.
- 주요 timeline 및 선택한 boundary sheet를 시각적으로 확인했다. **1,169프레임 모두를 사람이 프레임별 수동 평가한 것은 아니다.**
- 시간 변화량 및 RGB 동일성은 **전체 1,169프레임**을 CPU에서 계산했다.
- 좌우 시각 검토는 A/B 중 어떤 것이 prior인지 가린 별도 검토도 참고했다. A=LineArt ON, B=baseline OFF. 이는 통계적 사용자 평가나 객관적인 정답 판정은 아니다.

### 정렬/통제 확인: Evidence

- 프레임 0–352, 즉 C0–C6의 최종 emitted RGB **353프레임은 완전히 동일**하다.
- 최초 출력 차이는 **frame 353 / 14.708333 s / C07 new first**다. 이는 두 run의 분기 시작점이지 geometry drift의 시작점이라는 뜻이 아니다.
- anchor 내부 ROI는 **전 1,169프레임에서 두 결과가 byte-identical**하다.
- 그러므로 중앙 원본 내용의 변화나 타임스탬프 오정렬을 generated-region 차이로 비교한 것은 아니다.
- 증거: [manifest](manifest.json), [전체 프레임 요약](temporal_metrics_summary.json), [설정 확인](config_verification.json).

## 3. 영역별 냉정한 판정

| 영역/기준 | 판정 | 직접 관찰 및 해석 | 신뢰도 |
|---|---|---|---|
| 왼쪽 창틀/기둥 | baseline 약간 선호, 예외 있음 | 20–30초 일부 창의 세로 분할과 가로 연결이 baseline에서 조금 더 읽힌다. 40초에는 반대로 LineArt의 분할이 더 선명하다. 두 결과 모두 기둥에 실타래/녹은 듯한 패턴이 남는다. | Medium |
| 오른쪽 큰 창틀/기둥 | 대체로 비슷 | 굵은 창살·외곽 프레임·주 기둥은 양쪽에 남아 있다. baseline 창 전체의 topology가 붕괴했다고 단정하면 과장이다. | Medium |
| 오른쪽 창 내부 표면 | LineArt 국소 우세 | 44초와 마지막 프레임 등에서 baseline의 철사/긁힌 선 같은 과잉 패턴이 더 뚜렷하다. 다만 42초에는 baseline 창이 파란 면과 규칙적인 큰 격자로 더 깔끔하게 보인다. | Medium |
| 오른쪽 인물/장비 같은 생성물 | baseline 선호 | 32초 이후 확인 시점들에서 LineArt의 인물·카메라 같은 형태가 창 앞을 더 크게 가린다. baseline에도 유사한 생성물은 있으나 대체로 하단/우측에 치우쳐 배경 가림이 덜하다. | Medium |
| 바닥 반사/격자 | 둘 다 문제, 명확한 승자 없음 | 양쪽 모두 길게 늘어난 수직 띠와 수평 패널 같은 구조가 유지된다. LineArt 일부 변형량 감소가 정상적인 반사 geometry의 회복을 의미하지 않는다. | Medium |
| 천장/상부 | 명확한 개선 없음 | 양쪽 모두 부자연스러운 선/색 구조가 남는다. LineArt 마지막 시점에는 상부의 글자·간판 같은 패치도 보인다. | Medium |
| chunk 경계 연속성 | 해결 근거 없음 | 양쪽 모두 new-first에서 평균 변화량이 올라간다. LineArt가 모든 영역의 boundary jump를 일관되게 줄이지 않는다. | Medium |

## 4. 확인할 대표 증거 — 좋아진 사례와 반례를 함께 제시

### 4.1 왼쪽: 24초는 baseline, 40초는 LineArt가 더 읽히는 예외

[왼쪽 24초·40초 원해상도 비교](evidence/left_windows_24_40.png)

- **frame 576 / 24.000 s / emitted C11**: 왼쪽 crop의 오른편 긴 창에서 LineArt는 사선/그물 같은 교차가 눈에 띈다. baseline은 상대적으로 세로 구분선과 가로 연결을 읽기 쉽다. 차이는 국소적이며 전체 벽이 한쪽만 정상이라는 뜻은 아니다.
- **frame 960 / 40.000 s / emitted C19**: 반례. 분홍색 창의 LineArt 쪽 세로 분할/가로 선이 더 명확하고 baseline에는 녹아 붙은 듯한 선이 보인다.
- **Inference:** 왼쪽에서 LineArt가 장기 구조를 일관되게 고정했다는 설명보다, 샘플 경로에 따라 국소 우열이 바뀌었다는 설명이 더 잘 맞는다.

### 4.2 오른쪽: “잡선 감소”와 “창틀 복구”는 다르다

[오른쪽 42초·44초 비교](evidence/right_windows_42_44.png)

- **frame 1008 / 42.000 s / emitted C20**: baseline은 파란 창 면과 큰 격자가 비교적 깔끔하다. 이 프레임만 보면 LineArt 우세라고 말하기 어렵다.
- **frame 1056 / 44.000 s / emitted C21**: baseline의 창 내부가 수많은 가는 철사 모양으로 채워진다. LineArt 쪽은 이 잡선이 덜하다.
- 그러나 굵은 창살과 외곽 프레임은 **둘 다 유지**된다. 이번 증거는 주로 창 내부 surface/texture 안정성 차이지, baseline의 전체 창 geometry가 완전히 무너지고 prior가 복구했다는 증거가 아니다.
- C20/C21의 추가 phase sheet에서 baseline은 파란 창 면 ↔ 잡선이 많은 창 면 상태가 번갈아 나타난다. f1000/f1008은 비교적 깔끔하고 f1024/f1025는 잡선이 늘며 f1048은 다시 파란 면, f1056은 잡선이 증가한다.
- 따라서 특정 44초 프레임을 “최초의 비가역적 구조 붕괴 시점”으로 선언하지 않는다.
- 증거: [C20 오른쪽](chunk_boundaries/right_C20.jpg), [C21 오른쪽](chunk_boundaries/right_C21.jpg).

### 4.3 오른쪽: 창이 덜 지저분해져도 새 생성물이 더 크게 가린다

[오른쪽 32초·48.667초 비교](evidence/right_objects_32_48.png)

- **frame 768 / 32.000 s / emitted C15**: LineArt에는 인물/카메라 같은 형태가 창 앞을 크게 차지한다. baseline의 유사한 형태는 우측 가장자리/하단에 더 치우쳐 있다.
- **frame 1168 / 48.666667 s / emitted C23**: 같은 경향이 남는다. LineArt의 창 내부는 상대적으로 단순하지만 앞의 인물 형태가 더 크다.
- generated ROI 안에서 생긴 내용이므로 중앙 anchor의 실제 공연자와 구분해야 한다. 실제 촬영 화면 바깥의 정답은 없으므로 그 인물이 현실에 없다고 증명할 수는 없다. 다만 **배경 구조를 유지하려는 현재 평가 목적에서는 더 큰 가림/주의 분산**이다.
- negative prompt에 사람 관련 단어가 있어도 양쪽 모두 해당 형태가 나온다. 이를 단순히 평가에서 제외하면 LineArt 쪽 가시적 부작용을 놓치게 된다.

### 4.4 바닥: 덜 움직이는 오류를 개선으로 세면 안 된다

[바닥 중앙 40초·48.667초 비교](evidence/floor_center_40_48.png)

- 두 결과 모두 중앙 하부가 자연스러운 공연장 반사보다 수직으로 늘어난 막대와 큰 수평 패널처럼 보이는 구간이 있다.
- LineArt는 일부 행/띠 구조를 오래 유지한다. baseline은 일부 구간에서 더 큰 색 띠/휘어진 경계 변화가 있다. 반면 LineArt 마지막 프레임에는 글자 같은 패턴도 나타난다.
- 이런 패턴의 일부는 공통 C0에도 존재한다. 모든 비정상 구조가 C07 prior 이후 새로 생긴 것으로 설명하면 틀린다.
- **Inference:** 전체 기간 평균 변화량 감소만으로 반사 정합성이나 geometry 회복을 주장할 수 없다. 30초 이후만 보면 바닥 평균 변화량 감소도 사라진다.

### 4.5 30초 전후와 최종 시점

| 시간/프레임 | emitted chunk | 판정 메모 |
|---|---|---|
| 10 s / f240 | C04 | 공통 복사 구간. 개선 효과 평가 대상이 아니다. |
| 20 s / f480 | C09 | 왼쪽 일부 분할은 baseline이 더 읽힘. 양쪽에 이미 거친 생성 패턴 존재. |
| 24 s / f576 | C11 | 왼쪽 국소 baseline 우세 사례. |
| 28 s / f672 | C13 | 왼쪽 개선 일관성 없음. 큰 무대 배치는 양쪽 대체로 유지. |
| 30 s / f720 | C14 | prior가 양쪽 구조를 일관되게 안정화했다는 명확한 전환은 보이지 않음. |
| 32 s / f768 | C15 | LineArt 오른쪽 인물/장비 같은 형태의 가림이 뚜렷한 대표 시점. 최초 등장 프레임이라고 주장하지 않음. |
| 34–38 s / f816,f864,f912 | C16–C18 | 같은 장단점이 지속. 바닥/기둥의 비정상 패턴을 양쪽 모두 해결하지 못함. |
| 40 s / f960 | C19 | 왼쪽 분홍 창은 LineArt 우세 반례. 바닥은 양쪽 띠/패널 오류. |
| 42 s / f1008 | C20 | baseline 오른쪽 창 면이 비교적 깔끔한 반례. |
| 44 s / f1056 | C21 | LineArt 오른쪽 창 내부 잡선 감소; 큰 창틀은 양쪽 남음. |
| 46–48 s / f1104,f1152 | C22–C23 | LineArt의 국소 표면 이득과 생성물/반사 오류가 함께 남음. |
| 48.667 s / f1168 | C23 | prior가 장기 geometry drift를 해결했다고 판정할 수준은 아님. |

프레임 번호와 chunk는 0-based다. `emitted chunk`는 최종 영상에 해당 프레임을 처음 내보낸 chunk다. 다음 working chunk의 중복 prefix에 다시 포함되는 것과 구분한다.

## 5. 보조 metric — 품질 점수가 아니다

Metric은 grayscale ROI를 area downsample한 뒤 프레임별 평균 밝기를 빼고, 인접 프레임의 absolute difference를 평균한 값이다.

`M(t) = mean(abs((Y_t - mean(Y_t)) - (Y_(t-1) - mean(Y_(t-1)))))`

표는 0–255 밝기 단위로 환산했다. 좌우 240×270, 바닥 480×68의 축소 ROI를 썼다. **광류/카메라 motion 보정은 하지 않았다.** 공간적으로 달라지는 조명·생성 인물 움직임·detail 밀도에 영향을 받으므로 낮을수록 더 정확한 geometry라는 뜻이 아니다. PSNR/LPIPS로 두 결과의 품질 순위를 매기지 않았다.

### C07–C23, frame 353–1168

| 영역 | LineArt ON | baseline OFF | ON 상대 변화 |
|---|---:|---:|---:|
| 왼쪽 건축 | 2.504 | 2.822 | -11.3% |
| 오른쪽 건축 | 2.749 | 2.428 | +13.2% |
| 바닥 | 6.364 | 6.894 | -7.7% |

### 실제 관심 구간인 30초 이후, frame 720–1168

| 영역 | LineArt ON | baseline OFF | ON 상대 변화 |
|---|---:|---:|---:|
| 왼쪽 건축 | 2.781 | 3.133 | -11.2% |
| 오른쪽 건축 | 3.054 | 2.471 | +23.6% |
| 바닥 | 7.510 | 7.387 | +1.7% |

- 전체 평균만 보면 LineArt 왼쪽/바닥 변화량이 낮지만, **오른쪽은 반대로 더 크다.**
- 30초 이후에는 오른쪽 +23.6%, 바닥 +1.7%로 바닥 평균 변화량 감소도 없다. 그러므로 “후반부 전체가 더 안정적”이라는 주장을 이 metric은 지지하지 않는다.
- C07–C23 바닥의 변화량 95분위수는 LineArt 13.866, baseline 12.576이다. 평균 감소만으로 큰 변화까지 줄었다고 설명할 수도 없다.
- anchor control은 두 결과 모두 전체 해당 구간 2.471, 후반 2.713으로 동일하다.

### chunk boundary 보조 확인

`new-first 변화량 평균 / 같은 분석 구간의 나머지 변화량 평균`:

| ROI | LineArt | baseline |
|---|---:|---:|
| 왼쪽 | 2.93× | 2.71× |
| 오른쪽 | 2.02× | 2.11× |
| 바닥 | 1.39× | 1.23× |

양쪽 모두 boundary에서 변화량이 커지는 경향이며 LineArt의 일관된 해소는 없다. 다만 boundary 17개와 나머지를 평균한 기술 통계로, 유의성 검정이나 geometry jump의 독립 측정은 아니다. 19.042초 등 큰 조명 전환은 양쪽 ROI 변화량도 동시에 높인다.

전체 원자료: [CSV](temporal_metrics.csv), [요약 JSON](temporal_metrics_summary.json).

## 6. 왜 detailed LineArt가 곧 상세 구조 제어를 뜻하지 않는가 — 코드 근거

### Evidence: 실제 주입

현재 실험 복사본:

- `../../source/VideoOutpainting_test/main/src/ltx_outpaint/lineart_global_prior.py:143–149` — 30-step schedule: 0–9=0.5, 10–19=0.3, 20–29=0.1.
- 같은 파일 `:185–204` — Global C4 **15×9 spatial grid**, line strength scalar를 시간 방향에 반복.
- 같은 파일 `:205–224` — eligible new/generated query에 `B(i,j) = lambda * g_i * g_j` additive bias.
- `../../source/VideoOutpainting_test/main/src/ltx_outpaint/global_attention.py:42–63` — 기존 Q/K/V로 biased Global attention 계산; fine prefix/anchor 위치에는 canonical Global 결과 복원.
- C07–C23 17개 chunk의 metadata/log에서 prior 활성 및 지정 schedule을 확인했다. “플래그만 켰고 실제로 주입되지 않았다”는 근거는 없다.

### Inference: 표현력의 한계

- raster에서 세밀한 선을 뽑아도, 추가 prior가 전달하는 것은 **15×9개의 선 반응 강도**다. 내부 padded canvas 기준 한 cell은 약 128×128 px다.
- 이 추가 bias 자체는 “같은 기둥의 위·아래”, “같은 창틀의 연결”, “직선 방향”, “교차점 대응”을 명시적으로 표현하지 않는다.
- 서로 멀리 떨어진 두 cell이라도 `g_i`, `g_j`가 크면 bias가 커진다. 같은 실제 구조선인지 판별하지 않는다.
- **모델의 기존 Q/K/RoPE가 가진 내용·위치 정보가 사라지는 것은 아니다.** 다만 추가한 prior가 상세 line topology를 직접 전달한다고 말할 수 없다는 뜻이다.
- 첫 프레임의 RGB/구조 feature를 별도 K/V memory로 제공하는 구현도 아니다. 따라서 “모델이 reference 창틀을 이해해서 매번 정확히 복원한다”는 설명은 이 코드보다 강한 주장이다.
- 이 제한은 국소 효과가 있어도 큰 구조를 보장하지 못하는 합리적 설명이다. **관찰된 특정 오류의 인과 원인으로 확정한 것은 아니다.**

## 7. 엄밀한 A/B의 한계: 꼭 공개해야 할 bootstrap 차이

### 확인된 동일 조건

checkpoint-007500, base seed 71634910 및 absolute start 기반 seed rule, 입력 anchor hash, prompt/negative prompt, strength 1.0, 30 steps, local time, latent carry, visible/internal geometry가 일치한다. C0–C6 RGB와 C06 BF16 carry는 공통 artifact다. 자세한 값은 [설정 검증](config_verification.json).

### 동일하지 않은 한 가지 상태

C06의 원본 FP32 RGB prefix resume slot이 이후 chunk로 덮여 남아 있지 않았다. 이번 C07 재시작은 C06의 **lossless uint8 RGB segment tail / 255**로 prefix를 복원했다. MP4 codec 재복호화는 아니다.

- 정확한 BF16 carry는 유지했다.
- causal-safe carry에서 첫 1-frame keyframe은 RGB로부터 다시 encode한다. 이 부분이 원래 FP32 RGB prefix로 재개한 baseline과 bit-exact하지 않다.
- 저장 전 `*255 -> uint8` truncation 때문에 bootstrap pixel 오차는 <1/255 + FP32 반올림 허용치다.
- **그 작은 입력 오차가 downstream autoregressive diffusion에서 작은 결과 차이만 만든다는 보장은 없다.**
- 증거: [C06 복구 audit](../../report/C06_STATE_AUDIT.md), [bootstrap provenance](../../bootstrap/c0_c06/provenance.json).

따라서 **현재 두 영상 중 무엇이 더 좋아 보이는지는 평가할 수 있지만, 차이 전부가 LineArt 때문에 생겼다고 엄밀히 분리할 수는 없다.**

## 8. 최종 판단과 다음에 필요한 최소 검증

1. 이번 결과는 **NO CLEAR OVERALL IMPROVEMENT**. 현재 운영 기준은 baseline 유지가 보수적 선택이다.
2. 오른쪽 창 내부 잡선 감소라는 국소 이득은 인정한다. 이를 큰 geometry 전체 해결이나 장기 drift 예방 성공으로 확대하지 않는다.
3. 왼쪽의 시간별 역전, 오른쪽 생성물 가림, 바닥/천장 오류가 남아 있어 λ=0.5 스케줄을 production 개선으로 채택할 충분한 근거는 없다.
4. 단일 seed·장면, C07부터 주입한 48.7초 결과다. C0부터 주입한 경우, 더 긴 영상, 다른 장면, 4K에 대한 결론이 아니다.
5. 인과를 분리할 다음 최소 실험은 **이번과 정확히 같은 복구 C06 bootstrap/RNG에서 prior OFF control**을 생성하여 이번 ON과 비교하는 것이다. 짧은 초기 구간만으로도 bootstrap 민감도 단서는 얻지만 장기 효과까지 판단하려면 동일한 후반 구간이 필요하다. **이번 분석에서는 실행하지 않았다.**

## 9. 증거 파일 안내

- [비교 갤러리](index.html): 재생 가능한 기존 위/아래 합본, 대표 확대 증거, 원자료 링크.
- [기존 위/아래 합본](../lineart_vs_baseline_vertical_half_960x1080.mp4): 위=LineArt ON, 아래=baseline OFF.
- `evidence/`: 한국어 ON/OFF 표기한 대표 원해상도 비교 PNG 4개.
- `left/`, `right/`, `floor/`, `floor_center/`, `top/`: timestamp별 생성 영역 crop.
- `full/`: 중앙 제외 영역을 표시한 전체 frame 미리보기.
- `timelines/`: 영역별 시간 흐름 contact sheet. A=LineArt ON, B=baseline OFF.
- `chunk_boundaries/`: C07–C23 prefix last/new first/new middle/new last. 위=A, 아래=B.
- `temporal_metrics.csv`, `temporal_metrics_summary.json`: 전 프레임 CPU 보조 metric 및 동일성 검사.
- `manifest.json`, `config_verification.json`: 샘플/ROI/설정 확인.

원본 pipeline/VideoOutpainting, 실험 inference 소스 및 기존 inference 결과는 이번 분석에서 수정하지 않았다. 입력 MP4 2개의 기존 checksum 검증을 통과했다. 새 파일은 이 `comparisons/analysis_20260924` 분석 산출물 디렉터리 안에만 저장했다.
