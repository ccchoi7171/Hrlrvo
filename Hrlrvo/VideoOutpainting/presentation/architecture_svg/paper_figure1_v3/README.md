# Publication Figure 1 — 실제 동일 프레임 + 아키텍처

## 파일

- `figure1_video_outpainting.svg`: 실제 Input/4K reference 프레임과 아키텍처 전체. 텍스트·도형은 편집 가능하고 두 PNG 프레임은 SVG 안에 원본 해상도로 포함됩니다.
- `figure1_video_outpainting_outlined.svg`: 글자를 vector path로 바꾼 전체 도식.
- `figure1_architecture.svg`, `figure1_architecture_outlined.svg`: 아키텍처만 분리한 **순수 벡터**.
- `figure1_matched_frames.svg`, `figure1_matched_frames_outlined.svg`: Input과 4K reference 프레임 패널만 분리한 SVG.
- `figure1_print.pdf`: 폭 180 mm의 인쇄용 PDF. 사진은 raster, 아키텍처/글자는 vector입니다.
- `figure1_publication_package.zip`: 위 SVG, 미리보기, PDF, 선택된 원본 PNG 두 개, 프레임 정렬·geometry 근거를 묶은 파일.

SVG는 투명 배경입니다. 도식과 프레임 모두 외부 파일에 의존하지 않으며, PNG는
base64로 내장되어 있습니다. 사진을 포함한 전체 SVG를 순수 벡터라고 부르지는 않습니다.
편집본 폰트는 DejaVu Sans이고, 폰트가 없는 환경에서는 outlined SVG를 쓰세요.
전체 Figure를 폭 180 mm로 인쇄할 때 최소 글자 크기는 약 **7 pt**입니다.
Figma에서 import해 수정할 수 있도록 표준 SVG rect/path/text/image와 semantic group
ID를 사용했습니다. Figma 자체에서 직접 import하는 검증은 수행하지 않았습니다.

## 실제 프레임 검증

두 영상은 모두 24 fps, 94프레임, 3.916667초입니다. Figure에는 **decoded frame index
24, t=1.000초**를 사용했습니다. 서로 다른 time_base의 PTS도 동일한 시각을 가리킵니다.

- Input: `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/input/anchor/ef334007-aa0b-432d-a7e8-76a3eeeec252.mp4`, 1920×1080.
- 4K canvas reference: `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/input/test_set/ef334007-aa0b-432d-a7e8-76a3eeeec252.mp4`, 3840×2160.
- Reference 중앙 `[x=960:2880, y=540:1620]`은 **94개 모든 decoded frame에서 Input과 pixel-identical**입니다.
- Frame 24의 Input PNG와 reference의 중앙 crop PNG는 SHA-256까지 동일하며, SVG에도 두 source 프레임의 원본 PNG를 그대로 넣었습니다.
- Input 표시 크기와 4K 프레임 안 blue known footprint는 모두 **336×189**로 같아, 확대/축소 비율에 따른 시각적 착각을 줄였습니다.

이 일치는 두 source 영상의 정렬에 대한 증거이며 생성 품질이나 모델 성능 수치가 아닙니다.
4K `test_set` 파일은 저장소의 inference script에서 입력으로 사용됩니다. 실제 모델
생성 결과라는 근거가 없으므로 그림에서는 사용자가 요청한 Output 역할을 **“4K
output canvas”**, 그 아래에 **“reference frame”**으로 명시했습니다. 이 패널은 Input/목표 canvas의 과업 예시이며
model prediction을 보여주는 결과 figure가 아닙니다.

## 아키텍처의 핵심

1. 하나의 shared OP-QKV LoRA r32가 Native/Adaptive/Global Q/K/V에 delta를 추가합니다.
2. A-LoRA r64는 Adaptive에만, G-LoRA r64는 Global에만 추가됩니다. Native 전용 LoRA는 없습니다.
3. **① Native+Adaptive**를 Target-masked learned fusion합니다. 그 후 Window Assembly가 overlap-normalized LocalFull을 만듭니다.
4. **② LocalFull+GlobalFull**을 다시 learned fusion합니다. 단순한 세 경로 flat fusion이 아닙니다.
5. shared O + OP-O r32 → native Text Cross-Attention → native FFN + F-LoRA r32는 각 block에서 한 번 실행합니다.
6. 회색은 frozen LTX base, 실선 화살표는 feature flow, `+`로 끝나는 점선은 projection에 가산되는 LoRA parameter 경로입니다.

## 작은 공간 범위 도식

세 가지 miniature는 같은 canvas에서 각 attention view가 **어디를 읽는지** 보여줍니다.
임의 geometry를 그린 것이 아니라 현재 `geometry.py`의 `local_windows`, `anchor_patch`,
`adaptive_layout`을 직접 호출해 **K4의 대표 Local 하나**를 골랐습니다.

- canonical grid: H68×W120; Anchor `[17:51,30:90]`.
- 대표 Target: `[0:17,90:120]`.
- Native: 해당 Target Local만 읽음.
- Adaptive: 선택 Anchor patch와 Target을 포함하고, C4 support로 최소 확장된 bbox를 읽음.
- Global: 전체 canvas를 읽음.

blue는 known Anchor, purple은 예시 Target, teal은 Global 전체 read support입니다.
이 그림은 공간적 범위만 보여줍니다. 실제 T축, C2/C4 token packing, query/write
ownership, block별 K4/K5 교대와 PMA/RoPE는 이 작은 그림에서 생략했습니다.
Global miniature의 성긴 grid는 장식적 support 표기이며 C4 token count를 뜻하지 않습니다.
`figure_geometry.json`에 실제 helper가 반환한 좌표와 source hash가 들어 있습니다.

Native normalization/modulation, head gate, residual, C/R 복원, STG bypass, early
Global boost, VAE/sampler, temporal prefix, cache/loss/optimizer/checkpoint도 주 도식에서
생략했습니다. C/R와 fusion 역시 학습 가능하므로 “LoRA만 학습”이라는 주장은 아닙니다.

## 캡션 초안

**Figure 1. Frame-aligned video outpainting and hierarchical attention adaptation.**
(a) A 1920×1080 input anchor and its matched 3840×2160 reference frame illustrate the
expanded output canvas at t=1.000 s. The blue rectangle marks the observed input
footprint; the surrounding region is the outpainting target. The reference is used
to illustrate the task, rather than to report a model prediction. (b) Within an
LTX-2.3 block, Native and Adaptive Local views are fused before normalized window
assembly, and the resulting Local representation is fused with Global context.
Shared Outpainting QKV LoRA is supplemented by Adaptive/Global specializers. A single
shared output projection, Text Cross-Attention and FFN follow the two-stage fusion.
The spatial-support thumbnails illustrate one representative Local window; solid
arrows indicate feature flow and dashed lines with `+` terminals indicate additive LoRA parameter paths.
