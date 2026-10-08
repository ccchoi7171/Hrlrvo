# Video Outpainting — 발표용 SVG

첨부된 `.architecture 2.pdf` 첫 장의 모듈/LoRA 계층을 기준으로, 현재
`main/src/ltx_outpaint/` 코드를 대조해 만든 발표용 도식입니다.
생성 품질이나 학습 완료를 주장하는 그림이 아니라 **각 LTX transformer block의
attention/LoRA 구조를 요약한 그림**입니다.

## 사용할 파일

- `video_outpainting_core.svg`: 제목 없는 핵심 도식, 1840×940, 투명 배경, 편집 가능한 텍스트. PPT의 기존 제목 아래에 넣기에 적합합니다.
- `video_outpainting_core_outlined.svg`: 핵심 도식의 글자 path 버전.
- `video_outpainting_architecture.svg`: 전체 도식, 1920×1080, 투명 배경, 편집 가능한 텍스트.
- `video_outpainting_architecture_outlined.svg`: 동일 도식, 모든 글자를 vector path로 변환. 글꼴 설치 없이 모양이 유지되지만 글자를 텍스트로 수정할 수는 없습니다.
- `architecture_preview.png`, `architecture_core_preview.png`: 흰 배경 미리보기. SVG에는 bitmap이 들어 있지 않습니다.
- `modules/editable/`: Native, Adaptive, Global, 두 Fusion, Shared post-attention의 개별 SVG 6개.
- `modules/outlined/`: 같은 개별 모듈의 글자 path 버전.
- `video_outpainting_svg_package.zip`: 전체 도식, 모듈 두 버전, PNG 미리보기, 이 설명을 묶은 파일.

Figma 캔버스에 SVG를 드래그해 넣은 뒤 그룹을 해제하거나 내부 레이어를 선택하면
도형·선·글자를 수정할 수 있습니다. 편집본의 글꼴은 DejaVu Sans입니다. 글꼴이
없거나 다른 환경에서 모양을 유지해야 할 때는 outlined 파일을 사용하세요.
PowerPoint에서는 outlined SVG가 글꼴 차이 없이 동일하게 표시됩니다.
흰색/밝은 슬라이드를 기준으로 색상을 설계했습니다.

## 코드와 일치시키기 위해 유지한 핵심

1. Local은 Native와 Adaptive를 **창별로** 먼저 fusion하고, Window Assembly를 거쳐
   full-canvas 결과를 만듭니다. 그 후 Global과 두 번째 fusion을 합니다.
2. 두 fusion은 단순 합/곱/concat이 아니라 학습 가능한 gate입니다. Local fusion은
   resolved Target에서만 적용되고 나머지 Local 위치는 Native를 그대로 사용합니다.
3. 세 카드의 `Shared OP LoRA · r32`는 서로 다른 세 어댑터가 아니라 같은 block 안의
   **동일 Q/K/V 어댑터 가중치**입니다. block 사이까지 공유한다는 뜻은 아닙니다.
4. Adaptive는 A-LoRA r64, Global은 G-LoRA r64를 추가합니다. Native 전용 LoRA는 없습니다.
5. LoRA는 Q/K/V 투영에 델타를 더합니다. 카드 안의 `+`는 projection의 가산 델타를
   뜻하며 별도 attention 네트워크를 직렬로 연결한다는 뜻이 아닙니다.
6. 두 fusion 뒤에 shared O + OP O-LoRA r32를 **한 번** 실행합니다.
   이어지는 native Cross-Attention과 FFN도 block당 한 번 실행됩니다.
7. LTX base는 frozen이고 FFN에 F-LoRA r32가 추가됩니다. Cross-Attention에는
   별도 LoRA를 표시하지 않았습니다.

## 발표용으로 생략한 내용

Native normalization/modulation, residual 연결, head gate, C2/C4 압축·R2/R4 복원,
Adaptive Target-only Q 및 Target-only write의 내부 구현, PMA/RoPE, STG bypass,
early Global boost, diffusion sampler, VAE, temporal prefix, cache, training loss,
optimizer, checkpoint, GPU 메모리 옵션입니다. C/R와 fusion에도 trainable parameter가
있으므로 이 그림을 “학습되는 것은 LoRA뿐”으로 해석하면 안 됩니다.

일반 attention 경로의 구조를 표현했습니다. STG의 perturbation pass는 native
V-bypass를 사용할 수 있으므로 모든 model call에서 세 branch가 항상 실행된다는
주장은 하지 않습니다. Native/Adaptive/Global은 하나의 canonical hidden state를
읽는 self-attention view이며 서로 독립된 세 transformer 모델이 아닙니다.

## 확인한 소스

- `main/src/ltx_outpaint/model.py`: block별 shared adapter 객체, 두 fusion, assembly, single O.
- `main/src/ltx_outpaint/native_local_attention.py`: Native projection과 native attention.
- `main/src/ltx_outpaint/adaptive_local_attention.py`: shared + Adaptive Q/K/V delta, Target-mask.
- `main/src/ltx_outpaint/global_attention.py`: Global view.
- `main/src/ltx_outpaint/lora.py`: projection에 가산되는 LoRA delta.
- `main/configs/train_4k.yaml`, `main/configs/inference_4k.yaml`: r32/r64 구성, LTX-2.3 base.
- pinned LTX `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`: SA → CA → FFN의 native 순서.

## 검증

XML parse, SVG의 raster/external-reference 부재, outlined 파일의 text 부재,
모듈별 export, 전체 도식의 PNG rendering 및 시각 검토를 수행합니다.
Figma 자체에서의 import는 이 환경에서 직접 실행하지 않았습니다.
plain SVG 도형/path/text와 semantic group ID를 사용하고, filter, foreignObject,
external image/font, SVG marker, CSS variable을 사용하지 않았습니다.

`build_svg.py`는 source 생성 도구입니다. 모델 코드/학습 환경을 변경하지 않고,
임시 설치된 렌더러로 도식을 재생성합니다.
