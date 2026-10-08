"""Vector-only operator inset, faithful to compression.py (Global C4 example)."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cairosvg

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from build_svg import root, group, rect, text, line, label_width, outlined, write_svg

W, H = 1320, 780
INK, MUTED, LINE = "#35475E", "#687A90", "#6B7D92"
COLORS = ("#92B6E5", "#B8A2D6", "#8CCBC2", "#EDC190")


def blend(colors, weights):
    rgb = [tuple(int(c[j:j + 2], 16) for j in (1, 3, 5)) for c in colors]
    return "#" + "".join(f"{round(sum(w * c[k] for w, c in zip(weights, rgb))):02x}" for k in range(3))


def grid(parent, name, x, y, size, n, smooth=False):
    node = group(parent, name)
    step = size / n
    for row in range(n):
        for col in range(n):
            if smooth:
                u = min(1, max(0, (col + .5) * 2 / n - .5))
                v = min(1, max(0, (row + .5) * 2 / n - .5))
                fill = blend(COLORS, ((1-u)*(1-v), u*(1-v), (1-u)*v, u*v))
            else:
                fill = COLORS[(2 if row >= n / 2 else 0) + (1 if col >= n / 2 else 0)]
            rect(node, x + col * step, y + row * step, step, step,
                 fill=fill, stroke="#FFFFFF", radius=0, sw=1.3)
    rect(node, x, y, size, size, fill="none", stroke="#A7B5C7", radius=0, sw=2)
    if n == 8 and not smooth:
        for points in ([(x+size/2, y), (x+size/2, y+size)],
                       [(x, y+size/2), (x+size, y+size/2)]):
            line(node, points, color="#60738D", width=2.5, arrow=False)
        # A 4x4 cell, not an arbitrary crop: its sixteen fine tokens become one coarse token.
        rect(node, x, y, size/2, size/2, fill="none", stroke="#315FAD", radius=0, sw=3.5)
    return node


def operator(parent, name, x, y, width, label, subtitle, learned):
    node = group(parent, name)
    rect(node, x, y, width, 130,
         fill="#F0F5FE" if learned else "#F4F6F9",
         stroke="#92AED3" if learned else "#BEC8D5", radius=13, sw=2.4)
    title_lines = label.split("\n")
    title_size = min(35, int((width-38) / max(label_width(s, 1, bold=True) for s in title_lines)))
    assert title_size >= 28
    for i, title_line in enumerate(title_lines):
        baseline = y + 56 if len(title_lines) == 1 else y + 42 + i * 38
        text(node, x+width/2, baseline, title_line, size=title_size, weight=600, anchor="middle", color=INK)
    text(node, x+width/2, y+(100 if len(title_lines) == 1 else 115), subtitle, size=29, anchor="middle", color=MUTED)
    assert all(label_width(s, title_size, bold=True) < width-28 for s in title_lines)
    assert label_width(subtitle, 29) < width-28
    return node


svg = root(W, H, "Spatial compression and decompression",
    "A schematic Global C4 example. On each frame and channel, a trainable depthwise "
    "Conv2D reduces each full 4x4 canonical cell to one token. After attention, "
    "fixed bilinear spatial upsampling followed by trainable depthwise Conv3D "
    "refinement restores the fine spatial grid. Temporal length and channel count "
    "are preserved. Color grids are schematic and decompression is not an exact "
    "inverse. Adaptive reuses the same primitives with C2/C4 cells. Attention and "
    "RoPE are shown in the main architecture and intentionally outside this inset.")

text(svg, 50, 65, "Spatial Compression / Decompression", size=39, weight=600, color=INK)
text(svg, 50, 122, "Compression · Global C₄ example", size=32, color=MUTED)

comp = group(svg, "spatial-compression-c4")
grid(comp, "fine-grid-canonical-4x4-cells", 60, 165, 200, 8)
text(comp, 160, 409, "H × W", size=34, anchor="middle", color=INK)
text(comp, 160, 448, "Fine features", size=29, anchor="middle", color=MUTED)
operator(comp, "learnable-depthwise-conv2d", 385, 200, 515,
         "Depthwise Conv2D", "Learnable · kernel = stride = 4", True)
line(comp, [(260, 265), (385, 265)], color=LINE, width=3.2)
line(comp, [(900, 265), (1080, 265)], color=LINE, width=3.2)
grid(comp, "compressed-token-grid", 1080, 200, 130, 2)
text(comp, 1145, 409, "H/4 × W/4", size=32, anchor="middle", color=INK)
text(comp, 1145, 448, "Coarse tokens", size=29, anchor="middle", color=MUTED)
text(comp, 642.5, 382, "4 × 4 tokens → 1 token", size=31, anchor="middle", color="#315FAD")

line(svg, [(50, 480), (1270, 480)], color="#E2E7EF", width=2, arrow=False)
text(svg, 50, 526, "Decompression", size=32, color=MUTED)
decomp = group(svg, "spatial-decompression-r4")
grid(decomp, "coarse-features-after-attention", 105, 555, 110, 2)
operator(decomp, "fixed-bilinear-spatial-upsampling", 310, 545, 350,
         "Bilinear\nUpsampling", "Fixed · spatial ×4", False)
operator(decomp, "learnable-depthwise-conv3d-refinement", 720, 545, 350,
         "Depthwise Conv3D", "Learnable refinement", True)
grid(decomp, "restored-fine-feature-grid", 1130, 545, 140, 8, smooth=True)
line(decomp, [(215, 610), (310, 610)], color=LINE, width=3.2)
line(decomp, [(660, 610), (720, 610)], color=LINE, width=3.2)
line(decomp, [(1070, 610), (1130, 610)], color=LINE, width=3.2)
text(decomp, 160, 720, "After attention", size=29, anchor="middle", color=MUTED)
text(decomp, 1200, 720, "H × W", size=32, anchor="middle", color=INK)
text(svg, 660, 767, "Spatial resolution changes; temporal length T and channels D are preserved.",
     size=28, anchor="middle", color=MUTED)

ids = [n.get("id") for n in svg.iter() if n.get("id")]
assert len(ids) == len(set(ids))
assert not svg.findall(".//{http://www.w3.org/2000/svg}image")
write_svg(OUT / "spatial_compression_decompression.svg", svg)
fixed = outlined(svg)
assert not fixed.findall(".//{http://www.w3.org/2000/svg}text")
fixed_path = OUT / "spatial_compression_decompression_outlined.svg"
write_svg(fixed_path, fixed)
cairosvg.svg2png(url=str(fixed_path), write_to=str(OUT / "spatial_compression_decompression_preview.png"),
                output_width=W, output_height=H, background_color="white")
pdf_width = 120 / 25.4 * 96
cairosvg.svg2pdf(url=str(fixed_path), write_to=str(OUT / "spatial_compression_decompression.pdf"),
                output_width=pdf_width, output_height=pdf_width * H / W)
print("Verified unique module IDs, vector-only SVGs, text-to-path conversion, and operator label widths.")
