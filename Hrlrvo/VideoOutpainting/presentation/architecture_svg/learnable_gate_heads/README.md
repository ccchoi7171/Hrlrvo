# Learnable gate heads: paper figure inset

`learnable_gate_heads.svg` contains editable text and vector shapes on a transparent
background. `learnable_gate_heads_outlined.svg` converts lettering into vector paths.
The PNG provides a white-background preview. The PDF is a 140 mm wide vector figure.

Caption: **Timestep-conditioned fusion gating.** The pretrained MSA gate conditions
two independent learnable heads for local fusion and local–global fusion.

The outer “Learnable Gate Heads” box denotes a group of two heads, not an extra
shared trainable trunk. The timestep arrow abstracts the pretrained timestep
embedding and AdaLN path. Each head has its own learned channel-wise scale and
bias followed by a sigmoid. No additional timestep encoder is introduced.
Alpha_1 weights the native branch against the adaptive branch in local fusion;
alpha_2 weights the local result against the global result.

Implementation reference:
`/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/src/video_outpainting/main/src/ltx_outpaint/model.py`
(`NativeGateFusion`, `local_fusion`, `global_fusion`).

Rebuild using the existing temporary SVG tool environment:

```sh
PYTHONPATH=/tmp/videooutpainting-svg-tools python /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/presentation/architecture_svg/learnable_gate_heads/build_gate_figure.py
```
