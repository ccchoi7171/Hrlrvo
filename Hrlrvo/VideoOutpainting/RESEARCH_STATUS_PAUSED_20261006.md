# Video Outpainting 연구 목표·결과·진행상황 — 일시정지 기록

작성일: 2026-10-06. 실행 상태 확인 시점: 2026-10-06 08:39 UTC / 17:39 KST.

**현재 연구 목표는 사용자의 요청으로 일시정지되어 있다.** 이 문서는 기존 연구를 보존하고 다음 목표를 정할 때 참고하기 위한 기록이다. 실험 재개나 기존 후보의 자동 승격을 의미하지 않는다.

## 1. 현재 결론과 상태

- checkpoint-013000의 정규 **2K → 4K baseline 5개**를 H200에서 생성하고 결과를 검토했다.
- Fast Motion은 서로 다른 가설 **8개**, 기존 LongRange 개선 가설은 **9개**의 실제 GPU 실험과 검토 기록이 있다. 이후 static structure H1~H6와 추가 STG 진단을 별도로 진행했다. 동일 가설의 장기 검증·파라미터 변경·CPU 분석은 독립 가설로 중복 집계하지 않았다.
- 축소 LongRange **1,025f continuous generation**을 baseline과 여러 후보에서 완료했다. 천장·벽의 반복 mesh가 약 500f 이후 뚜렷해지는 현상을 확인했다.
- **채택된 Fast Motion 개선안, LongRange 최종 개선안, 검증된 static structure 기반은 아직 없다.** 반복 실험 수와 생성 완료 조건은 상당 부분 충족했지만, 개선안 선정 및 최종 완료 검증은 미완료다.
- H5 structural residual은 **305f 생성·평가 완료 → 개선 근거 부족으로 후속 승격하지 않음**이다. H6 immutable mask는 A6000에서 161f 완료했으나 개선은 미확정이다. H6의 H200 실행은 사용자의 일시정지 지시에 따라 **첫 청크 저장 전에 종료**했다.
- **새 GPU 실험은 실행하지 않는다. A6000 GPU 사용 금지를 유지한다.** 이후 사용자가 새 목표나 재개 방향을 제시하면 H200에서만 GPU 작업을 진행한다.
- 기존 A6000 결과의 **CPU 자료 복사는 계속 진행 중**이다. 08:39 UTC 확인에서 복사 프로세스가 살아 있었고 최종 검증 보고서는 아직 없었다.

용어: `generation PASS`는 추론이 끝나 결과물이 저장되었다는 뜻이다. `NO_CLEAR_CHANGE`는 검토 범위에서 개선이 확인되지 않았다는 뜻이며, 모든 조건에서 효과가 없다는 증명은 아니다. `REGRESSION`은 해당 설정·입력·평가 구간의 과거 검토에서 악화로 기록했다는 뜻이다. **기존 161f 판정은 아래에 역사적 결과로 보존하며 새 연구의 자동 탈락 기준으로 적용하지 않는다.**

## 2. 기존 목표와 완료 범위

원래 목표는 코드 분석으로 끝내지 않고, 실제 추론을 반복하여 checkpoint-013000의 두 문제를 개선하는 것이었다.

| 목표 | 현재 상태 | 미완료 범위 |
|---|---|---|
| 실제 inference·학습 conditioning·recurrence 구조 이해 | 코드 및 tensor 경로 분석 기록 확보 | 원인별 인과관계의 완전한 분리는 미완료 |
| 지정된 2K anchor 5개를 정규 4K로 확장 | 5개 모두 생성·육안 검토 완료 | 정규 4K의 장거리 개선 검증은 없음 |
| 빠른 물체의 흐림·형상 손실 개선, 의미 있는 실험 5회 이상 | 8개 가설 검토 완료 | 효과가 입증된 개선안 없음 |
| 1,000f+ LongRange 재현 및 개선 실험 5회 이상 | 9개 기존 가설, 1,025f baseline·후보 검증 완료 | 장거리에서 채택할 개선안 없음 |
| Static Stage Geometry + Dynamic Lighting / Appearance | H1~H6 구현·실험 기록 확보 | 유망한 기반의 medium·problem-onset 검증 미완료 |
| 결과·설정·코드 fingerprint·runtime·VRAM·실패 보존 | 영상, 상태 JSON, 비교 이미지, metrics, source capsule 보존 | A6000 전체 자료 복사 및 최종 SHA 검증 진행 중 |
| 최종 baseline/best 1,000f+ 비교와 professor-critic 완료 판정 | baseline 및 비선택 후보의 full 검증 완료 | best 후보 미선정, 최종 완료 PASS 없음 |

우선순위는 이후 사용자 지시에 따라 Fast Motion보다 **LongRange의 정적 무대 구조 유지**로 이동했다. 추가 학습보다 inference 단계의 최소 변경을 우선했고, 현재까지 아래 개선 결과는 inference 실험에 해당한다.

현재 Codex goal 상태는 `paused`다. 별도 OMX mission 파일에는 H5 medium 결과 이후 기록된 `failed`가 남아 있다. 최종 목표 달성 상태는 아니며, 이 문서 작성으로 goal이나 mission 상태를 변경하지 않았다.

## 3. 기준 모델·설정·실제 추론 경로

| 항목 | 기준 |
|---|---|
| 체크포인트 | `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/hrlrvo_main_train_v1/checkpoints/checkpoint-013000` |
| 정규 config | `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/main/configs/inference_4k.yaml` |
| 정규 geometry | 1920×1080 anchor → 3840×2160 canvas |
| 축소 geometry | 960×540 anchor → 1920×1080 canvas |
| 공통 seed / steps | `33333333` / `30` |
| 청크·prefix·stride | 65 RGB frames / 17 RGB frames / 48 frames |
| 시간 압축 | 65f → 9 temporal latent slices, 17f → 3 slices |
| 출력 fps | 24 fps |
| 축소 실행 dtype | BF16 |
| H200 native 비교 | `backend=native`, `memory_fraction=0.44` |

기본 흐름은 다음과 같다.

```text
현재 source anchor + 직전 생성 결과의 마지막 17f FP32 RGB
  → VAE 격자에 맞게 padding
  → anchor 및 full-canvas prefix를 encode
  → known mask로 anchor와 prefix 조건화
  → Native Local / Adaptive Local / Global attention 및 learned gate fusion
  → denoising → VAE decode → canvas crop
  → 중앙 anchor RGB composite
  → 마지막 17f를 다음 recurrent chunk에 전달
```

첫 청크는 65f를 출력하고 이후에는 겹치는 17f를 제외한 48f씩 추가한다. 따라서 161f는 3청크, 305f는 6청크, 1,025f는 21청크다. 새 프레임이 이어지는 출력 경계는 F65, F113, F161…이며 다음 청크의 입력 시작점은 F48, F96…이다.

축소판은 입력만 resize한 실행이 아니다. 실험용 `recurrence.py`가 `reduced_geometry=True`일 때 VAE padding과 C4 격자 정렬을 적용한다. 1920×1080 canvas는 내부적으로 1920×1152로 padding된다. 원본 main·학습 코드는 보존하고 실험 복사본과 별도 hook을 사용했다.

중앙 anchor는 최종 RGB composite로 복원된다. 따라서 중앙 source 보존만으로 모델 내부 예측이나 generated exterior의 물체 복원이 정확하다고 판단할 수 없다. 기본 recurrence prefix는 FP32 RGB이며, 단순한 8-bit 재양자화가 누적 오류의 원인이라는 설명은 실제 경로와 맞지 않는다.

## 4. 정규 2K → 4K baseline 5개

아래 결과는 모두 H200, checkpoint-013000, native 설정, 3840×2160@24fps다. Runtime은 실제 기록값이며 동시 부하 등의 영향을 포함하므로 장치 성능 비교값으로 사용하지 않는다.

| 결과 | 입력 식별자 | 프레임 | Runtime | 관찰 |
|---|---|---:|---:|---|
| [baseline_1 영상](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/outputs/baseline_1/generated.mp4) | `0ad61bb2-a2c7-4e23-b5fa-266edf6b06fa` | 142 | 116.6분 | 생성영역으로 들어오는 차량의 차체·바퀴 형상이 GT와 다름. 도로·우측 전경 구조도 다름 |
| [baseline_2 영상](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/outputs/baseline_2/generated.mp4) | `0f2313f5-5a61-4ccc-811c-f3bbf1d41151` | 72 | 74.9분 | 관측 스키어는 유지되지만 외곽 스키·눈 윤곽과 주변 지형이 GT와 다름 |
| [baseline_3 영상](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/outputs/baseline_3/generated.mp4) | `b78e2011-b17e-4d55-a28e-862d58187799` | 83 | 74.3분 | rider가 anchor 밖으로 나갈 때 머리·차체·몸의 생성 형상이 GT와 다름 |
| [baseline_4 영상](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/outputs/baseline_4/generated.mp4) | `ef334007-aa0b-432d-a7e8-76a3eeeec252` | 94 | 65.1분 | 호수·산의 전체 장면은 유지되나 외곽 숲·절벽·반사 윤곽의 정확한 복원은 아님 |
| [baseline_5 영상](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/outputs/baseline_5/generated.mp4) | `fea48c69-5ce4-441f-9eaf-4b8bf195bb09` | 116 | 84.2분 | 공연 장면은 유지되나 생성된 주변 인물·관객 실루엣과 무대 구조가 GT와 다름 |

짧은 정규 baseline에서 전체 장면이 무너지는 현상은 뚜렷하지 않았지만, 이는 장거리 안정성이나 외곽 reconstruction 정확도의 증거가 아니다. 각 출력 디렉터리의 `job_status.json`에 input·config·source SHA·output SHA·runtime·VRAM이 보존되어 있다. 기록된 peak allocated는 약 43.5~52.9 GiB, reserved는 약 60.8 GiB다.

## 5. Fast Motion 실험

대표 testcase는 baseline_1의 차량 진입 장면이며, 주요 비교는 960×540 → 1920×1080, 142f로 진행했다. 아래 8개는 서로 다른 기존 가설이다.

| 실험 | 변경·가설 | 기록된 결과 |
|---|---|---|
| `motion_halo1` | Native K/V 공간 support를 1 cell 확장 | `REGRESSION`: 일부 차량 ROI 수치가 좋아져도 전체 외곽 숲의 늘어짐·반복과 새 구조물이 발생 |
| `motion_halo2` | K/V support를 2 cells 확장 | `REGRESSION`: 전체 외곽 오류 증가, 차량 내부·전경 및 숲 구조 왜곡 |
| `motion_native_bias75` | learned gate에 Native 방향 logit offset | `REGRESSION`: 생성영역의 그래픽·텍스트 형태 붕괴 |
| `motion_local_bias75` | Local/Global fusion을 Local 방향으로 bias | `REGRESSION`: 전체 외곽이 로고·텍스트 같은 패턴으로 변함 |
| `motion_early_global` | 초기 denoising에서 Global을 일시 강화 | `NO_CLEAR_CHANGE`: 큰 배경 붕괴는 없지만 차량 일부가 사라져 낮은 ROI 오차가 복원 성공을 뜻하지 않음 |
| `motion_halo1_motion_band` | 첫 청크의 특정 시간·좌하단 query에만 halo | `NO_CLEAR_CHANGE`: 전역 halo의 배경 악화는 줄었지만 차량 형상 복원은 미확인 |
| `motion_anchor_kv_bridge10` | 관측 anchor K/V를 작은 residual로 연결 | `NO_CLEAR_CHANGE`: 차량 형상 실패가 남고 전체 개선 근거 없음 |
| `motion_anchor_kv_t2_to_t1` | 미래 관측 T2 anchor K/V를 T1 query에 제한 연결 | `NO_CLEAR_CHANGE`: 작은 appearance 차이만 있으며 차체·바퀴 복원 실패가 남음 |

일부 scoped 방법은 H200 정규 4K에서도 추가 확인했으나 채택할 개선은 확인하지 못했다. late-STG Fast Motion screen도 두 장치에서 `NO_CLEAR_CHANGE`다. 빠른 물체 문제는 현재 해결되지 않았으며, 작은 ROI metric·edge energy·flow 수치만으로 성공을 선언하지 않았다.

## 6. 기존 LongRange 진단과 full 검증

### 6.1 문제 관찰

공연 source를 연속 생성하면 generated exterior의 천장·벽에 반복적 mesh / cellular texture가 커진다. 기존 full baseline 검토에서는 약 F500부터 뚜렷해지고 F1000 이후에도 남는다. 관측 anchor는 깨끗하지만 **이 source의 생성영역 전체에 대한 GT는 없다.** 물리적 정답 복원과 자기 일관성은 구분해야 한다.

### 6.2 기존 9개 가설의 단기 기록

| 실험 | 변경 | 과거 161f 판정·후속 상태 |
|---|---|---|
| `long_anchor_refresh_161f` | recurrent prefix의 중앙 anchor RGB를 실제 source로 재주입 | `NO_CLEAR_CHANGE`, 경계 개선이 일관되지 않음 |
| `long_selective_carry_161f` | latent carry weight 1.0 | `REGRESSION`, 색과 구조의 경계 변화 증가 |
| `long_prefix1slice_161f` | known prefix를 첫 latent slice로 제한 | `REGRESSION`, 생성 장비·구조의 경계 점프 |
| `long_periodic_reset_161f` | 주기적으로 history reset | `REGRESSION`, 외곽 천장·벽·장비가 다시 샘플링되며 크게 변경 |
| `long_partial_carry_161f` | 약한 carry 0.25 + 상대 cap | 단기 `NO_CLEAR_CHANGE`; 후속 1,025f는 `REGRESSION` |
| `long_anchor_latent_refresh_161f` | anchor latent를 재확인 | `NO_CLEAR_CHANGE`, 경계별 결과 혼재 |
| `long_prefix_static_filter_161f` | generated prefix의 작은 변화·고주파를 제한 | `NO_CLEAR_CHANGE`, 구조 개선 근거 없음 |
| `long_noise_correlation_161f` | noise correlation ρ=0.75 | 단기 `NO_CLEAR_CHANGE`; 후속 1,025f는 `REGRESSION` |
| `long_global_c4_memory_v2_161f` | 작은 canonical Global C4 memory 주입 | `NO_CLEAR_CHANGE`, 실제 activation은 확인했으나 개선 부족 |

Uniform temporal RoPE rebase는 공통 Q/K 회전이 상대 attention 위상에서 상쇄되는 구조여서 별도 수치 control로 다뤘다. noise ρ=0.5는 같은 가설의 parameter screen이며, Global memory v1의 activation 전 구현 실패도 별도로 보존했다.

### 6.3 실제 1,025f 결과

아래는 축소 1920×1080@24fps의 continuous 결과다. H200와 A6000의 baseline을 분리하여 같은 장치 계열의 후보와 비교했다.

| 실행 | 장치 | 길이 | 결과 |
|---|---|---:|---|
| Native baseline | A6000 | 1,025f / 21청크 | 생성 완료, F500 이후 천장·벽 mesh 관찰 |
| Partial carry | A6000 | 1,025f / 21청크 | `REGRESSION`: mesh가 남고 temporal 변화·flow residual 및 후기 색 불연속 악화 |
| Noise correlation ρ=0.75 | A6000 | 1,025f / 21청크 | `REGRESSION`: temporal 수치는 크게 낮아도 후기 반복 mesh·색 패턴 악화 |
| [Native baseline 영상](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/outputs/long_baseline_1025f/generated.mp4) | H200 | 1,025f / 21청크 | 생성 완료, runtime 19,936.72s, 후기 mesh 지속 |
| [Late-STG0 영상](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/outputs/long_late_stg0_1025f_h200/generated.mp4) | H200 | 1,025f / 21청크 | `NO_CLEAR_CHANGE`: runtime 18,611.93s, baseline과 함께 F500 이후 mesh가 남음 |

Noise correlation full 결과에서 temporal MAE 약 **−69.21%**, sampled flow residual 약 **−65.52%**였지만 edge·Laplacian variance와 반복 mesh는 증가했다. Late-STG0도 temporal MAE 약 **−13.18%**, near-join warp residual 약 **−21.80%**와 함께 edge density·Laplacian variance가 감소했다. 두 결과 모두 **temporal metric 감소만으로 구조 개선이나 motion 보존을 판정할 수 없음**을 보여준다. smoothing·조명·텍스처·장비 이동의 영향을 완전히 분리한 인과 증명은 없다.

### 6.4 VAE·시간 경계에서 얻은 정보

- 반복 encode/decode probe에서 RGB RMSE가 약 0.0205 → 0.0343으로 증가했다. 독립적으로 encode한 17f tail과 원래 65f encode의 대응 latent도 다르다. RGB 재인코딩 손실과 causal context 차이는 존재하지만 **LongRange mesh의 단독 원인으로 입증하지 못했다.**
- Encoder-only 검증에서 9f → 2 slices, 17f → 3 slices를 확인했다. 같은 17f 입력의 최근 8f를 바꿔도 첫 2 slices는 같았지만, 앞 9f 변경은 세 번째 slice에도 영향을 줬다. **9+8 분할이 서로 독립적인 latent block이라는 가정은 성립하지 않는다.**
- 별도 9f encode의 첫 2 slices와 17f encode의 첫 2 slices도 정확히 같지 않았다. shape가 같다는 이유만으로 직접 교체·carry의 의미가 같다고 볼 수 없다.

근거: [VAE origin 검토](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/evaluation/vae_origin_probe_20261005T2235/root_review.json), [Hybrid VAE 경계 검토](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/evaluation/hybrid_vae_temporal_probe/root_review.json).

## 7. Static Structure 연구 축: H1~H6

사용자가 추가한 prior는 **천장·벽·바닥·무대 layout은 유지하고 조명·색·밝기·그림자는 현재 영상에 맞춰 변화시키는 것**이다. 목표 상태는 `Canonical Static Structure + Dynamic Appearance`이며, 원래 generated history 전체를 계속 전달하던 recurrence의 구조 오류 feedback을 줄이는 방향이었다.

| 후보 | 핵심 변경 | 완료 범위·기록된 판정 |
|---|---|---|
| H1 Naive hybrid | canonical clean 9f + recent 8f의 RGB prefix | A6000 161f 완료, `NO_CLEAR_CHANGE`, medium 미실행 |
| H2 Generated-only hybrid | 현재 anchor를 유지하고 outer region에 hybrid 적용 | A6000 161f 완료, `REGRESSION`, smear 및 후기 temporal 악화 |
| H3 Structure-only hybrid | canonical 구조 성분과 recent appearance를 분리하려는 prefix 변형 | A6000 161f 완료, `REGRESSION`, 큰 foreground blur/smear 관찰 |
| H4 Immutable structural reference | 갱신하지 않는 canonical latent/reference와 dynamic prefix 결합 | A6000 161f 완료, `NO_CLEAR_CHANGE`, 작은 구조 지표 혼재·경계 residual 악화 |
| H5 Structural residual | Global C4 feature에 canonical static residual 직접 주입 | A6000 161f 후 305f 완료, `REJECT_NO_GAIN`; 545/1,025f 미실행 |
| H6 Immutable support mask | H5의 residual은 유지하고 첫 recurrent support mask를 이후 고정 | A6000 161f 완료, `NO_CLEAR_CHANGE`; H200 실행은 첫 commit 전 사용자 중단; 305f 미실행 |

H1~H4의 기존 공동 검토에서 승격 후보는 없었다. 이는 해당 구현·설정·161f 범위의 과거 판정이며 **static structure 연구 축 전체를 기각한 결과는 아니다.**

### H5에서 실제 확인한 것

H5는 block 28의 Global C4 T3 feature에 static residual을 주입하는 방식이다. canonical bank와 현재 feature의 차이를 support 영역에서 centered/RMS 제한한 뒤 작은 dose로 반영했다. P/N에는 같은 연산을 적용하고 STG에는 쓰지 않는 경로를 사용했다.

305f에서 support cell 수가 **34 → 13 → 14 → 6 → 4**로 줄어들었다. 그러나 mask 축소가 mesh의 원인이라는 인과 증명은 없다. 셀별 평균 RGB 변화만 제거한 CPU counterfactual도 33 → 15로 줄어들어, 단순 평균 밝기 보정만으로 해결된다는 가설의 근거는 약해졌다.

305f의 실제 비교에서는 천장·벽 fine mesh가 양쪽에 남고 명확한 static structure 개선은 없었다. 전체 304전환의 warp residual 약 **+7.93%**, generated flow magnitude 약 **+6.08%**였다. Runtime은 **7,399.89s**다. 그래서 기존 Stage2에서 `REJECT_NO_GAIN`으로 기록했고 더 긴 생성을 실행하지 않았다. Optical flow 증가만으로 semantic motion이 좋아졌거나 freeze가 없다고 확정하지 않았다.

근거: [H5 305f 공동 판정](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/evaluation/static_stage_smokes/H5_305f/root_review.json).

### H6에서 실제 확인한 것과 중단 지점

H6는 H5의 block·temporal placement·dose·RMS·P/N/STG 연산을 유지하면서 첫 recurrent 입력에서 선택된 **34개 cell mask를 고정**했다. H5의 support contraction을 바꿨을 때 결과가 달라지는지 보는 별도 가설이었다.

A6000 161f는 runtime **3,697.34s**, 3청크로 완료했다. 실제 mask capture/reuse, nonzero update, bank/reference를 검증했다. 첫 113f는 H5와 동일했고, 고정 mask로 갈라지는 효과는 **F113 이후의 한 청크**에 해당한다. 이 구간은 장거리 효과를 결론 내리기에 부족하다.

기존 검토는 `NO_CLEAR_CHANGE`였고 당시 한 번의 unchanged medium 검증을 고려했으나 **H6 305f는 어느 서버에서도 실행하지 않았다.** 이후 A6000 GPU 금지에 따라 H200 이관용 실행본을 준비했다.

H200 H6 실행은 PID 689135로 시작했지만 사용자의 “161f 실험은 의미 없어… 잠시 실험을 멈춰봐” 지시에 따라 SIGINT로 종료했다. **저장 완료 청크 0개, 품질 평가 불가**다. 종료 status의 `FAIL`은 사용자 중단 기록이며 과학적 실패 판정으로 집계하지 않는다. 종료 후 08:11 UTC에 H200 GPU memory 0 MiB와 compute process 없음이 확인되었다.

근거: [H6 A6000 검토](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/evaluation/immutable_static_mask_smoke_review/root_review.json), [H200 사용자 중단 receipt](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/evaluation/h200_immutable_static_mask_orchestration/user_pause_receipt_20261006T0808Z.json).

## 8. 실험 효율·판정 기준의 변경 이력

기존에는 사용자 지시로 신규 후보를 **113~161f smoke → 약 300f → 500~600f 문제 발생 구간 → 1,025f 최종 validation**으로 검증하는 stage gate를 사용했다. 단일 metric보다 visual·static structure·temporal appearance·motion·chunk continuity·artifact를 함께 보도록 했다. 과거 partial carry와 noise correlation은 단기 결과가 장기 성공을 보장하지 않는 사례였다.

**최신 사용자 지시는 161f 결과에서 문제가 있다고 해석한 현상이 실제로는 문제 아닐 수도 있다는 점을 제기하고 실험을 일시정지한 것이다.** 따라서 기존 smoke 판정과 승격 기준은 역사적으로 보존하되, 새 목표에서 길이·관측 현상·실패 기준이 정해지기 전까지 기계적으로 재사용하지 않는다. 짧은 테스트는 구현 확인에 유용할 수 있지만 LongRange 품질의 충분한 근거로 취급하지 않는다.

H200/A6000는 설정·seed가 같아도 실제 생성 tensor가 달랐다. 이후 품질 비교는 **H200 후보 대 H200 baseline**으로 진행해야 한다. 서로 다른 GPU 계열의 절대 metric·runtime 차이를 후보 개선이나 causal speedup으로 해석하지 않는다.

Exact CFG reuse는 별도 효율 control이다. 검증한 A6000 142f·1,025f 및 H200 4K 142f 조건에서는 output/raw segment의 동일성을 확인했지만 품질 개선은 아니며, 모든 guidance·prompt 조건이나 production integration에 대한 보장은 아니다.

## 9. 후속 Foreground Hallucination 축 — 요구사항만 보존

사용자가 제시한 다음 축은 **Canonical Background + Novel Foreground Suppression**이다.

- 실제 anchor에서 관측되고 생성영역 방향으로 이어지는 객체의 **Observed Continuation은 허용**한다.
- 생성영역 내부에서 객체별 관측 근거 없이 새로 생기는 사람·스태프·관객·foreground object의 **Unobserved Foreground Emergence는 억제**한다.
- 지미집이나 camera rig의 존재만으로 새 스태프·관객을 생성할 근거가 생기는 것은 아니다.
- 사람 전체를 지우거나 일반 negative prompt만 강화하는 방식으로 정의하지 않는다. 관측된 출연자·팔·장비의 continuation은 유지해야 한다.

이 축은 유망한 static foundation의 검증 이후 최소 변경으로 결합하도록 **보류**했다. 현재는 요구사항만 저장했고 구현·prompt 변경·GPU 실험·효과 검증은 없다. Canonical reference 자체에 이미 hallucinated foreground가 포함되는 문제, 반사·그림자·조명과 사람 emergence를 구별하는 문제도 미해결이다.

근거: [보류된 foreground 요구사항](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/deferred_foreground_axis.json). 이 파일의 초기 gate 값보다 최신 `scientific_reviews.json`의 완료·일시정지 기록을 우선한다.

## 10. 자료 보존·서버 사용 현황

### A6000 → H200 CPU 복사

원격 실험 tree `/home/oem/hrlrvo/experiments/vo013000_20261006`의 regular file **2,405개 / 96,101,030,673 bytes**를 대상으로 SHA inventory를 완료했다. 중복 content를 제외한 unique data는 **71,450,284,469 bytes**, 기존 H200 파일 중 SHA가 일치해 재사용한 data는 **2,335,930,267 bytes**다. 네 개의 CPU 압축 rsync stream으로 자료를 옮기고 있다.

H200 저장 위치:

```text
/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/a6000_full_archive_20261006T0724Z
```

08:39 UTC 확인에서 `transfer_all_v2.py` PID 445682는 살아 있었고 상태는 `COMPRESSED_UNIQUE_CONTENT_COPY`다. **최종 `verification_report_v2.json`은 아직 생성되지 않았다.** 따라서 모든 결과가 이전·검증되었다고 선언할 수 없다. 이후 최종 복사와 per-file SHA 검증 완료가 필요하다. Symlink 객체는 보존하지만 tree 밖의 symlink target을 자동으로 추가 복사하는 범위는 아니다.

CPU 복사는 GPU를 사용하지 않으며 원격 결과를 삭제하지 않는다. A6000 GPU는 이 연구에서 추가로 사용하지 않는다. 마지막 보존된 A6000 GPU 확인은 08:06 UTC에 compute process 없음·utilization 0%였다. 이는 해당 시점의 기록이며 서버 전체의 현재 사용자 부하를 보장하지 않는다.

복사 상태: [status_v2.json](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/evaluation/a6000_h200_handoff/status_v2.json).

### 구현·증거의 남은 제한

H5/H6의 bank·mask·residual activation 및 보존된 prefix는 실제 artifact로 검증했지만, 일부 역사적 실행의 official imported module identity와 저장되지 않은 decoded 17f FP32의 완전한 equality는 미검증이다. 검증 범위를 넘어 모든 내부 실행이 입증되었다고 표현하지 않는다.

중단된 H200 실행의 CPU prefix observer는 Python symlink를 resolve한 경로와 실제 argv의 경로가 달라 `WRONG_COMMAND`로 종료했다. 자료를 복사하지 않았고 producer 품질을 판정하지도 않았다. 실행된 frozen 파일과 실패 로그를 보존했으며 **일시정지 이후 수정·재실행하지 않았다.**

기존 original/source/config 보존 감사에서 H200 28개, A6000 45개 bound 파일의 hash 일치를 확인한 기록이 있다. 이번 정리는 새 Markdown 문서만 추가하며 기존 코드·체크포인트·실험 결과를 변경하지 않는다.

## 11. 다음 목표로 넘길 미완료 항목

다음 항목은 인계용 목록이며 현재 실행 계획이나 자동 재개 요청이 아니다.

1. **새 목표와 품질 판정 기준 정의:** 사용자가 문제라고 보는 현상, 허용할 변화, 평가할 frame 구간과 실험 길이를 다시 정한다. 기존 161f proxy 판정을 그대로 전제하지 않는다.
2. **기존 자료 복사 마무리:** CPU transfer의 종료 상태와 전체 SHA 검증 결과를 확인하고 H200 자료 위치를 확정한다.
3. **Static 구조 방법의 실효성 검증:** 기존 결과를 근거로 재선정할 수 있지만 H6를 승자로 가정하지 않는다. H6의 H200 품질 결과나 medium 결과는 현재 없다.
4. **문제 발생 구간에서 비교:** 유망한 방법이 생기면 static geometry·lighting·motion·artifact를 같은 frame에서 확인한다. 안정적으로 보이는 것이 blur나 움직임 감소 때문인지 구분한다.
5. **Foreground 축의 순서 결정:** static foundation 확인 후 구현하려던 기존 방향을 보존한다. 새 목표가 우선순위를 바꾸면 최신 지시를 따른다.
6. **최종 검증·보고:** 채택할 후보가 있을 때 baseline/best continuous 1,025f 비교와 최종 감사가 필요하다. 현재 best 선정과 최종 PASS는 없다.

T8 근처의 outgoing temporal state에 structural memory를 배치하는 방향도 소스 기반 CPU 검토 메모만 있다. 구현·GPU 실험은 없으며 효과가 확인된 후보로 취급하지 않는다.

## 12. 주요 결과·코드·기록 위치

| 자료 | 경로 |
|---|---|
| 전체 연구 root | `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006` |
| 저해상도 무대 anchor, 960×540 / 1,025f | [longrange_960x540_1025f.mkv](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/inputs/longrange_960x540_1025f.mkv) |
| 원본 LongRange source | `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data_backUP01/01_PreProc_2K_1920x1080/metadata/01/generated/01_PreProc_2K.mp4` |
| 실험용 inference 복사본 | [inference.py](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/source/ltx_outpaint/inference.py:1668) — `sample_chunk()` / `sample_long_range()` |
| 저해상도 실행 진입점 | [run_job.py](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/run_job.py:136) |
| 격자·recurrence 실험 hook | [recurrence.py](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/recurrence.py:18) |
| 저해상도 H200 config | [reduced_h200.yaml](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/configs/reduced_h200.yaml:50) |
| H200 native full 실행 설정 | [long_baseline_1025f.json](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/jobs/long_baseline_1025f.json) |
| A6000 H6 완료 영상의 검증된 H200 mirror | [generated.mp4](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/evaluation/immutable_static_mask_evaluation_watch_runtime_v1/mirror/long_static_residual_immutable_mask_161f/generated.mp4) |
| 기존 부분 결과 mirror | `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/a6000_results` |
| 구조·학습·tensor 경로 분석 | [architecture_audit.md](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/architecture_audit.md) |
| 종합 결과 초안 | [REPORT_DRAFT.md](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/REPORT_DRAFT.md) |
| 시간순 진행 기록 | [PROGRESS.md](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/PROGRESS.md) |
| 실험·판정 구조화 기록 | [scientific_reviews.json](/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/scientific_reviews.json) |
| 기존 mission·rubric·ledger | `/home/work/ai-graphics_33/.omx/goals/autoresearch/vo013000-overnight` |

저해상도 코드를 읽는 순서는 `run_job.py → recurrence.py → source/ltx_outpaint/inference.py`다. 실제 checkpoint는 job의 명시적 `checkpoint-013000` 인자가 결정한다. YAML에 남아 있는 다른 adapter 기본값만 보고 실행 checkpoint를 판단하면 안 된다.

이 문서는 위의 실제 status·review·사용자 중단 기록을 기준으로 작성한 별도 인계 기록이다. 기존 초안과 ledger에는 당시의 RUNNING·승격 검토 상태도 남아 있으므로, **현재 실행 상태와 161f 판정 기준은 이 문서의 일시정지 기록 및 최신 사용자 지시를 우선한다.**
