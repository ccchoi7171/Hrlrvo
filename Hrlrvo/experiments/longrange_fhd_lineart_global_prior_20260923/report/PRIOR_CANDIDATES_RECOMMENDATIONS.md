# C0 첫 프레임 구조 prior 후보 — 시각 비교 추천

## 상태
- LongRange diffusion: 중지. LineArt ON C1 생성은 아직 한 번도 시작하지 않았음.
- 기존 strength_1p0 결과/코드 및 원본 VideoOutpainting/pipeline은 변경하지 않았음.
- 동일 C0 first RGB1920x1080 한 장에서 LineArt4종 + 실제 depth5종(4개checkpoint, V2Small만2개입력해상도)을 생성. Depth도 CPU only.
- 후보를 active reference로 교체하지 않았음.

## 추천
### LineArt: L2 detailed 추천 / L1 balanced 보수적 대안
L2는 얇은 창틀·벽·아치선이 기존 coarse보다 많이 유지되면서 L3처럼 텍스처가 화면을 채우지 않음.
- L1 balanced: Canny50/140, Gaussianσ1.2; 큰 구조 중심, 가장 보수적.
- L2 detailed: Canny35/100, Gaussianσ0.8; 상세도/잡선 균형 추천.
- L3 very_detailed: Canny20/70, σ0.5; 식물/배우/반사/표면무늬가 과도해 첫 주입에는 비추천.
- L4 clahe_lowlight: CLAHE1.5+ Canny55/145, σ1.0; 어두운 부분 보완이나 텍스처도 증가.
모두 native1920x1080, 사전 축소/재확대·dilation 없음. 실제 파라미터는 lineart/parameters_and_stats.json.

### Depth: D5 V2Base518 큰 구조 추천 / D4 V2Small1036 세부 대안
이는 이 한 프레임의 시각적 적합성 판단이지, depth정확도 검증은 아님. Ground truth가 없음.
- D1 V2Small518: 큰 깊이 흐름은 안정적이나 벽/기둥/배우 구분이 많이 완만함.
- D2 V1Small518: 벽/천장 큰 면은 구분되지만 바닥에 급격한 밴드가 보임.
- D3 V2MetricIndoorSmall518: 중앙무대/배우 주변 깊이가 조각나고 주변 면과 대비가 강함. 이 영상의 첫 구조 prior로는 후순위. 숫자는 검증된 실제 거리가 아니라 모델 추정치.
- D4 V2Small1036: 기둥·무대 주변 물체 구분이 D1보다 세밀함. 세부 분리가 필요하면 추천하지만 장식/사람까지 구조로 기억할 위험.
- D5 V2Base518: 벽/천장의 큰 면과 바닥-후면무대 구분이 상대적으로 정돈되어 coarse layout 용도로 우선 추천. 중앙 배우는 합쳐져 보이며 fine window geometry 보장은 없음.

## 표시와 데이터
- depth_raw.npy는 보간된 원래 모델 출력float32. 모델간scale가 다르므로 수치 직접비교 금지.
- depth_gray.png: white=near, black=far. Metric만 방향을 뒤집어 표시했으며 raw는 보존.
- depth_color.png: 같은예측의 Turbo표시. Warm=near.
- depth_structure_gamma05.png: 배경을 보기 위한 sqrt(gamma0.5) 표시변환뿐; raw예측을 개선하거나 바꾼 것이 아님.
- 서로 다른표시색/반전map은 별도모델 후보로 세지 않았음.

## 가장 중요한 한계
현재 GlobalC4는9x15=135 spatial cells, 각cell이128x128pixel에 해당함. gi*gj는 선의 개수/밀도 salience만 반영하고 방향·직선성·연결성은 보존하지 않음. 따라서 LineArt가 더 상세해도 그 자체로 세밀한 구조제약을 전달한다고 주장할 수 없음. Depth를 이 방식에 넣는 경로도 아직 구현/실행하지 않았음.

## 모델 근거 / 사용범위
- https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- https://huggingface.co/LiheYoung/depth-anything-small-hf
- https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
- https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf — Base weights CC-BY-NC-4.0; 이번은 로컬 연구비교이며 상용production적용이 아님.

[index.html](index.html)에서 전체후보/원해상도/overlay/흑백/컬러/좌우crop비교를 열 수 있음.
