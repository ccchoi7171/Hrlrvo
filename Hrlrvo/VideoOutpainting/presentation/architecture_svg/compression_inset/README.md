# Spatial compression / decompression inset

The Global C4 example shows full 4x4 canonical cells compressed into single
tokens by a trainable depthwise Conv2D, with kernel=stride=4. Decompression takes
post-attention features, applies fixed bilinear spatial upsampling, then a
trainable depthwise Conv3D refinement (kernel 3, stride 1, padding 1).
Temporal length T and channel count D remain unchanged.

Fine and coarse colors are schematic. Decompression restores spatial resolution,
not the exact original features. Adaptive uses the same primitives at scales 2
and 4; this Global example illustrates the scale-4 route. Canonical cell geometry
is used separately to form RoPE; the main figure already contains attention.

Source:
`/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/src/video_outpainting/main/src/ltx_outpaint/compression.py`
and `global_attention.py` in the same directory.

Editable SVG and outlined SVG use transparent backgrounds. PNG preview is white.
The vector PDF is 120 mm wide. To fit a smaller inset, remove secondary labels
in Figma rather than shrinking text excessively.

Rebuild:
```sh
PYTHONPATH=/tmp/videooutpainting-svg-tools python /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/presentation/architecture_svg/compression_inset/build_compression_inset.py
```
