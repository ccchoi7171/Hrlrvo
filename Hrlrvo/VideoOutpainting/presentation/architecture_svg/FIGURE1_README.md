# Video Outpainting — Figure 1

기존 발표용 SVG를 논문 개요 그림에 맞게 다시 구성한 버전입니다. 그림 안에 큰
제목을 넣는 대신 `(a) Video outpainting`, `(b) Hierarchical attention adaptation`
두 패널로 구성했습니다. 색은 역할을 구분하는 데만 사용하고, 도형과 선은 얇고
단순하게 유지했습니다.

## 파일

- `video_outpainting_figure1.svg`: 두 패널 전체 그림, 2220×960, 투명 배경, 편집 가능한 텍스트.
- `video_outpainting_figure1_outlined.svg`: 전체 그림의 글자 path 버전.
- `video_outpainting_figure1_architecture.svg`: 아키텍처 패널만 분리한 편집용 SVG.
- `video_outpainting_figure1_architecture_outlined.svg`: 아키텍처 패널만 분리한 글자 path 버전.
- `video_outpainting_figure1_task.svg`, `video_outpainting_figure1_task_outlined.svg`: 공간적 과업 정의만 분리한 SVG.
- `video_outpainting_figure1_preview.png`, `video_outpainting_figure1_architecture_preview.png`: 흰 배경 미리보기.
- `video_outpainting_figure1_package.zip`: 위 파일과 이 설명을 묶은 패키지.

모든 SVG는 bitmap, external image/font, filter, SVG marker 없이 만들어졌습니다.
편집용 텍스트의 글꼴은 DejaVu Sans입니다. 글꼴이 없거나 PowerPoint에서 모양을
그대로 유지해야 할 때는 outlined SVG를 사용하세요. Figma 자체에서의 import는
직접 실행하지 않았습니다.

## 그림을 읽는 법

`(a)`는 실제 생성 결과가 아닌 **가시 RGB 영역의 개념도**입니다. 중심의 known 2K
Anchor는 전체 4K canvas의 가로·세로 각각 절반이며, 나머지가 생성할 주변 영역입니다.
겹쳐진 프레임은 동일한 공간 확장이 영상 시간축에 적용됨을 나타냅니다. 이 패널은
최종 품질, 실제 inference 결과 또는 temporal prefix 구현을 주장하지 않습니다.

`(b)`의 **실선은 feature flow**, **점선은 projection에 더해지는 adapter parameter
경로**입니다. 하나의 OP-QKV LoRA rail이 세 경로에 같은 block 안의 shared weights를
공급합니다. Base Q/K/V도 같은 frozen module 객체를 재사용합니다. A-LoRA와 G-LoRA는
각 branch에 추가되는 delta이며, 별도 self-attention 모델을 직렬로 붙인 것이 아닙니다.
Native 전용 LoRA는 없습니다.

Local은 창별 Native와 Adaptive를 Target-masked learned fusion한 뒤 overlap-normalized
Window Assembly를 수행합니다. 이 LocalFull과 GlobalFull을 두 번째 learned fusion으로
합치고, shared O + OP-O r32를 한 번 실행합니다. 그 뒤 native Cross-Attention과
FFN + F-LoRA r32가 한 번씩 실행됩니다. 그림의 LoRA rank/ownership과 순서는 이전에
검증한 현재 코드/PDF의 계약을 유지합니다.

회색은 frozen LTX base, 실선은 feature flow, 점선은 adapter parameter 경로입니다.
adapter와 fusion은 학습 가능하며, 이 그림에서는 생략했지만 compressor/decompressor에도
trainable parameter가 있습니다. Output O와 FFN은 frozen base와 trainable LoRA의
결합이므로 회색 모듈 안에 각각 amber/purple LoRA 표시를 넣었습니다.

## 생략한 세부 구현

단일 canonical hidden state X를 세 attention view가 읽습니다. 별도 transformer 세
개가 아닙니다. 가독성을 위해 X input wire, C2/C4·R2/R4, PMA/RoPE, Native
normalization/modulation, head gate, residual, STG bypass, early Global boost,
VAE·sampler, cache·optimizer·loss·checkpoint, long-range temporal prefix는 생략했습니다.
일반 attention path를 표시하며 STG perturbation의 V-bypass까지 모두 그린 그림은 아닙니다.

참조한 코드와 frozen/shared ownership의 상세 근거는 같은 폴더의 `README.md` 및
`main/src/ltx_outpaint/model.py`, `lora.py`, `adaptive_local_attention.py`,
`global_attention.py`에 있습니다.
