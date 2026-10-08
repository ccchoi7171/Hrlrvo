# LongRange inference-only experiment: Window B C21→C23

## A. VERDICT

**불확실(개선 입증 실패; 관찰 구간에서는 악화 신호가 우세).** 세 토글을 함께 켠 결과가 baseline보다 구조 왜곡을 분명히 완화하지 못했다. C21 중간(1048f)은 큰 차이가 없지만 C22에서 양쪽 생성영역이 흐리고 저대비가 되며, C23 중간/끝(1144/1168f)은 좌우에 파란색 패치/번짐과 여전한 휜 기둥·벽이 보인다. 특히 lossless RGB24 segment 기준 chunk 경계의 좌/우 프레임 MAE가 baseline보다 약 2–6배 커져 side continuity는 악화했다. 세 수정의 개별 책임과 전체 81.33초의 장기 결과는 이 실험만으로 확정할 수 없다.

## B. CODE CHANGES

- **프로덕션 `pipeline` 및 `VideoOutpainting` 파일 변경 없음.** 수정은 `source/ltx_outpaint/inference.py`라는 실험용 복사본 하나뿐이다.
- `_condition_latents` (복사본 760–887): optional clean latent prefix override, current independently encoded anchor box 우선(hybrid carry), 좌우 prefix latent 값에 0.8 배율. Center anchor 값은 배율 1.0. Upstream `VideoConditionByMask` strength는 1.0 그대로이므로 prefix known mask / denoise mask 0 / timestep 0 / no known-token noise 유지.
- `sample_chunk` (복사본 1056–1059, 1095–1106, 1126–1135, 1182–1191): 독립 토글, chunk-local model positions, 최종 sigma=0 post-Euler state의 last 3 BF16 temporal latent slices를 VAE decode 전에 CPU로 추출. RGB 저장 경로는 그대로. Flag OFF 기본값은 기존 경로로 분기되며 metric 필드도 추가하지 않음.
- 실행/비교 스크립트: `scripts/run_window.py`, `scripts/compare_window.py`, `scripts/continuity_metric.py`, `scripts/segment_boundary_metric.py`. 모두 이 실험 폴더에서만 출력.
- 원본 SHA256/백업/patch: `patches/original_sha256.txt`, `patches/inference.py.before`, `patches/sha256_before_after.txt`, `patches/inference_experiment.patch`. 원본 해시 `55525cd3fad501caf57fc1928104240521390bc0a9b7faa063f7f04625340286` 전후 동일, Git working tree clean.
- Rollback: 프로덕션 코드 롤백은 불필요. 이 실험을 비활성화하려면 실험용 명령을 사용하지 않고, 필요 시 이 폴더만 보관/삭제하면 된다. 이미 종료된 inference를 중단할 필요도 없다.
- CPU 검증: 원래 anchor/known-mask regression 8 passed, 새 hybrid carry/side scale/local temporal position 테스트 2 passed, py_compile 성공. `logs/cpu_existing_tests_nogpu.log`, `logs/cpu_experimental_tests_full.log`.

## C. EXPERIMENT SETUP

- Workspace: `/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fix_test_20260923`
- 수행: **Window B만**, C21(1008–1072f), C22(1056–1120f), C23(1104–1168f) 연속. 각각 17f prefix + 48f new, 출력 범위 `[1025,1073)`, `[1073,1121)`, `[1121,1169)`. Window A는 사용자의 추가 생성 중단 지시로 수행하지 않음.
- Baseline metadata: `pipeline/data/02_VideoOutPaint_4K_3840x2160/chunk/05/run.json`. Anchor `/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080/05_PreProc_2K.mp4`; checkpoint `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/hrlrvo_main_train_v1/checkpoints/checkpoint-007500`; prompt `""`; negative `people, person, human, staff, audience, crowd, backstage, seats`; steps 30; fps24; seed base71634910, per chunk global start added: 71635918/71635966/71636014. Baseline과 동일한 memory optimization(video-only GPU weights ON, streaming VAE decode ON, FFN chunk OFF), anchor composite feather16. `configs/effective_baseline_settings.json`에 전체 settings 보관.
- Baseline 대비 정확히 3개 토글: rebase temporal position ON, latent carry ON, generated-side prefix latent amplitude 0.8. 첫 C21에는 과거 실험 latent가 없으므로 baseline C20의 pre-MP4 uint8 segment last17을 FP32로 복원하여 시드했다. C22/C23의 carry input은 true이고 저장된 tensor `[1,128,3,68,120]` BF16이다.
- 실행: `commands/window_B.sh`; encode-only resume: `commands/window_B.sh --resume`; 비교: `commands/compare_B.sh`; 확인: `commands/verify_B.sh`.
- C23까지 GPU 추론/segment commit은 성공. 최초 CPU MP4 조립에서 실행 스크립트의 committed-record 참조 오류가 발생했으나, 스크립트 한 줄만 고치고 `--resume`으로 **GPU 재추론 없이** 저장된 3개 segment에서 MP4를 완성했다. `logs/window_B.log`, `logs/window_B_resume_encode.log`에 기록.

## D. RESULTS

- 정확한 전역 프레임 비교: 1008, 1024, 1025, 1048, 1056, 1072, 1073, 1096, 1104, 1120, 1121, 1144, 1168. C21–C23의 start/prefix-last/new-first/new-middle/last phase를 포함. Baseline은 원본 1952f MP4, modified는 글로벌 1008–1168f에 대응하는 161f MP4.
- 시각: 1048f는 baseline과 대체로 비슷하다. 1096/1120f에서는 생성영역 양쪽 대비·디테일이 줄고 회색으로 흐려진다. 1144/1168f에서 좌우 벽/창/기둥의 기울기 자체는 지속되며, modified에는 눈에 띄는 청색/자색 패치와 smear가 추가된다. 오른쪽은 F1168에서 반복 벽 텍스처가 옅어지고 검은 창 부근 청색 잡상이 커진다. 왼쪽도 아치/기둥이 여전히 기울고 내부 청색 패치가 나타난다. 중앙 anchor는 시각적으로 거의 동일하다.
- Continuity: pre-MP4 lossless segment에서 경계 one-frame MAE(RGB 0–255, left/right) baseline→modified: C20→21 `3.756/1.232 → 8.554/5.935`; C21→22 `1.684/1.459 → 7.262/6.964`; C22→23 `1.624/1.375 → 6.431/6.279`. Center는 각각 대략 2.5–3.6으로 비슷하다. 경계 직전/직후 일반 one-frame 변화는 좌우 대개 0.5–2.8로 이보다 작다. 경계에서 side step이 증가한 것으로 해석하되, 이는 geometry 전용 metric이 아니라 밝기·질감도 포함한다. 원자료: `report/window_B_lossless_segment_continuity.tsv`; MP4 기반 교차검증: `report/window_B_temporal_mae.tsv`.
- 원인 귀속: 첫 C20→21에서도 step이 커지므로 latent carry만의 문제라고 단정할 수 없다. Rebase와 0.8 amplitude, 초기 uint8 seed를 분리하지 않았고, C22/C23에서는 추가로 latent carry가 작동한다. Hard prefix visible RGB는 그대로인데 다음 new frame의 side conditioning만 변한 점은 경계 jump와 일치하지만 확정 원인은 아님.

## E. OUTPUT FILES

- Modified video: `modified_outputs/window_B/window_B_global_1008_1168.mp4` (H.264 yuv420p, 3840×2160, 24fps, 161f), `modified_outputs/window_B/video_frame_map.json`.
- Lossless RGB segments, FP32 rolling prefix, BF16 carry, per-chunk metadata: `modified_outputs/window_B/segments/`.
- Extracted key frames: `frame_extracts/window_B/{baseline,modified}/`.
- Full-frame/left/right side-by-side: `comparisons/window_B/{full,left,right}/`.
- Contact sheets: `contact_sheets/window_B/{full,left,right}_phases.png`.
- Commands/logs/patch/report: `commands/`, `logs/`, `patches/`, `report/`.

## F. LIMITATIONS

- C21–C23의 6.7초짜리 독립 구간은 전체 81.33초 장기 안정성을 보증하지 않는다. C14–C16과 그 이후 chunk는 사용자의 추가 생성 중단 지시로 실행하지 않았다.
- 첫 C21 seed는 원본 FP32 recurrent prefix가 아닌 pre-MP4 RGB24 재구성본이다(rolling FP32 slot이 전체 baseline 종료 시 덮어써짐). 이 초기 precision 차이는 엄밀한 bitwise A/B를 제한한다.
- 0.8은 attention strength가 아니라 **known side latent amplitude scaling**이다. Prefix는 denoise되지 않지만 VAE latent 범위 자체를 바꾸므로 side washout/경계 jump 위험이 있다. 최종 latent의 last3을 새 17f prefix 좌표의 first3에 대응시키는 causal VAE temporal mapping mismatch도 가능하다.
- 세 요소를 동시에 켜서 기여도를 분리할 수 없다. 다음 추가 실험이 허용된다면 최소 1) **rebase ON, carry OFF, strength1.0** vs baseline, 2) **rebase ON, carry ON, strength1.0** vs 1)의 같은 3-chunk window만 실행하여 0.8 amplitude의 해와 latent carry의 해를 분리하는 것이 우선이다. 이 보고서 작성 단계에는 어떤 추가 inference도 실행하지 않았다.
