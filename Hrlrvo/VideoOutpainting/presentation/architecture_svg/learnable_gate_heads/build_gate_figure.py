"""Publication inset: shared MSA conditioning, two independent learned heads."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cairosvg

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from build_svg import root, group, rect, text, line, el, outlined, write_svg, label_width

W, H = 1740, 510
INK, LINE = "#243448", "#66778C"
BLUE, TEAL = "#315FAD", "#197B76"

svg = root(W, H, "Timestep-conditioned learnable fusion gates",
    "The existing pretrained LTX MSA gate supplies conditioning to two independent "
    "learnable heads, which produce the local fusion weight alpha_1 and the "
    "local-global fusion weight alpha_2. The enclosing box is a grouping, not an "
    "additional shared trainable layer. The timestep-to-MSA arrow abstracts the "
    "existing timestep embedding and AdaLN path. Each head uses its own learned "
    "channel-wise affine parameters and a sigmoid; no extra timestep encoder is added.")

inp = group(svg, "timestep-input")
el(inp, "circle", cx=145, cy=250, r=57, fill="#FFFFFF", stroke="#CAD3DF", stroke_width=2.4)
text(inp, 145, 267, "t", size=55, anchor="middle", color=INK)
text(inp, 145, 350, "Timestep", size=36, anchor="middle", color=INK)

base = group(svg, "pretrained-msa-gate")
rect(base, 325, 180, 300, 140, fill="#F1F4F8", stroke="#BBC7D5", radius=13, sw=2.4)
text(base, 475, 240, "MSA Gate", size=40, weight=600, anchor="middle", color=INK)
text(base, 475, 285, "vgate_msa", size=32, anchor="middle", color="#66778C")
text(base, 475, 363, "Pretrained · Frozen", size=32, anchor="middle", color="#66778C")

heads = group(svg, "learnable-gate-heads-group")
rect(heads, 765, 45, 680, 420, fill="#FFFFFF", stroke="#C7D2E0", radius=17, sw=2.4)
text(heads, 1105, 102, "Learnable Gate Heads", size=40, weight=600, anchor="middle", color=INK)

local = group(heads, "independent-local-fusion-head")
rect(local, 825, 140, 565, 105, fill="#EFF4FF", stroke="#8CA8D7", radius=12, sw=2.4)
text(local, 1107.5, 206, "Local Fusion Gate", size=38, weight=600, anchor="middle", color=BLUE)

global_local = group(heads, "independent-local-global-fusion-head")
rect(global_local, 825, 315, 565, 105, fill="#EDF8F5", stroke="#80B9B1", radius=12, sw=2.4)
text(global_local, 1107.5, 381, "Local–Global Fusion Gate", size=36, weight=600, anchor="middle", color=TEAL)

wires = group(svg, "conditioning-fanout")
line(wires, [(202, 250), (325, 250)], color=LINE, width=3.4)
line(wires, [(625, 250), (700, 250)], color=LINE, width=3.4, arrow=False)
line(wires, [(700, 192.5), (700, 367.5)], color=LINE, width=3.4, arrow=False)
el(wires, "circle", cx=700, cy=250, r=5.5, fill=LINE)
line(wires, [(700, 192.5), (825, 192.5)], color=LINE, width=3.4)
line(wires, [(700, 367.5), (825, 367.5)], color=LINE, width=3.4)

outputs = group(svg, "fusion-weight-outputs")
text(outputs, 1595, 102, "Fusion weights", size=32, anchor="middle", color="#66778C")
line(outputs, [(1390, 192.5), (1540, 192.5)], color=BLUE, width=3.4)
line(outputs, [(1390, 367.5), (1540, 367.5)], color=TEAL, width=3.4)
text(outputs, 1595, 211, "α₁", size=55, anchor="middle", color=BLUE)
text(outputs, 1595, 386, "α₂", size=55, anchor="middle", color=TEAL)

# Prove that labels fit their cards and that there are two distinct trainable heads.
assert label_width("Local Fusion Gate", 38, bold=True) < 565 - 40
assert label_width("Local–Global Fusion Gate", 36, bold=True) < 565 - 40
ids = [node.get("id") for node in svg.iter() if node.get("id")]
assert len(ids) == len(set(ids))
assert "independent-local-fusion-head" in ids
assert "independent-local-global-fusion-head" in ids
assert not svg.findall(".//{http://www.w3.org/2000/svg}image")

editable = OUT / "learnable_gate_heads.svg"
fixed = OUT / "learnable_gate_heads_outlined.svg"
write_svg(editable, svg)
outlined_svg = outlined(svg)
assert not outlined_svg.findall(".//{http://www.w3.org/2000/svg}text")
write_svg(fixed, outlined_svg)
cairosvg.svg2png(url=str(fixed), write_to=str(OUT / "learnable_gate_heads_preview.png"),
                output_width=W, output_height=H, background_color="white")
# A 140 mm vector PDF. The smallest ordinary label is ~7.3 pt at this width.
pdf_width_px = 140 / 25.4 * 96
cairosvg.svg2pdf(url=str(fixed), write_to=str(OUT / "learnable_gate_heads.pdf"),
                output_width=pdf_width_px, output_height=pdf_width_px * H / W)
print("Verified: editable SVG, outlined SVG, white PNG preview, and 140 mm vector PDF.")

