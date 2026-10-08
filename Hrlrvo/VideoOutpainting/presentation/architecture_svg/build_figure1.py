"""Publication-style SVG overview; uses the existing vector/outline helpers.

Run with the temporary renderer, without touching the model environment:
PYTHONPATH=/tmp/videooutpainting-svg-tools python /absolute/path/to/build_figure1.py
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import cairosvg

import build_svg as v


OUT = Path(__file__).resolve().parent
W, H = 2220, 960
v.C.update({
    "ink": "#223043",
    "muted": "#667387",
    "line": "#465468",
    "border": "#CCD5DF",
    "base": "#F0F2F5",
    "native": "#3D689D",
    "native_soft": "#EDF3FA",
    "adaptive": "#8564AC",
    "adaptive_soft": "#F4EFF9",
    "global": "#377F78",
    "global_soft": "#EDF6F3",
    "shared": "#AF7925",
    "shared_soft": "#FCF4E5",
    "fusion": "#3F5A7B",
    "fusion_soft": "#F0F4F8",
})
C = v.C

DESCRIPTION = (
    "Panel (a) is a schematic spatial support map, not a generated example: "
    "a centered 1920 by 1080 known RGB anchor inside a 3840 by 2160 outpainting canvas. "
    "Panel (b) abstracts the normal attention path in one LTX-2.3 transformer block. "
    "Solid arrows carry hidden features; dashed lines provide additive projection adapters. "
    "One shared OP QKV LoRA r32 and frozen base projections serve all three views. "
    "Adaptive adds A LoRA r64; Global adds G LoRA r64. Native and Adaptive are fused "
    "per local window with a Target mask, assembled, then fused with Global. "
    "A single shared O projection with OP O LoRA r32 is followed by native cross-attention "
    "and FFN with F LoRA r32. Compression/restoration, modulation, residuals, head gates, "
    "position encoding and guidance bypasses are intentionally omitted."
)


def dashed(parent, points, color=None, arrow=False):
    start = len(parent)
    v.line(parent, points, color=color or C["shared"], width=2.2, arrow=arrow)
    parent[start].set("stroke-dasharray", "7 6")


def panel_header(parent, x, y, letter, title):
    v.text(parent, x, y, f"({letter})", size=30, weight=700)
    v.text(parent, x + 58, y, title, size=30, weight=700)


def task_panel(parent):
    p = v.group(parent, "a-spatial-outpainting")
    panel_header(p, 40, 57, "a", "Video outpainting")
    v.text(p, 98, 96, "2K input → 4K canvas", size=23, color=C["muted"])
    panel = p
    p = v.group(panel, "a-spatial-support-map")
    p.set("transform", "translate(0 -70)")
    x, y, w, h = 62, 340, 370, 208.125
    v.text(p, x, 260, "4K spatial support", size=28, weight=700)
    v.text(p, x, 296, "3840 × 2160", size=25, color=C["muted"])
    for offset in (28, 14):
        v.rect(p, x + offset, y - offset, w, h, fill="#FBFCFE",
               stroke="#D6DEE8", radius=3, sw=1.6)
        v.rect(p, x + offset + w / 4, y - offset + h / 4, w / 2, h / 2,
               fill="#EEF3FA", stroke="#C7D6E9", radius=1, sw=1.2)
    v.rect(p, x, y, w, h, fill="#F3F1F7", stroke=C["ink"], radius=3, sw=2)
    grid = v.group(p, "target-support-grid")
    for i in range(1, 10):
        v.line(grid, [(x + i * w / 10, y), (x + i * w / 10, y + h)],
               color="#E3DEE9", width=0.9, arrow=False)
    for i in range(1, 8):
        v.line(grid, [(x, y + i * h / 8), (x + w, y + i * h / 8)],
               color="#E3DEE9", width=0.9, arrow=False)
    # Both spatial axes are exactly half the full canvas. This is a schematic
    # of visible RGB support; latent reflection padding is intentionally absent.
    v.rect(p, x + w / 4, y + h / 4, w / 2, h / 2,
           fill="#DDE8F7", stroke=C["native"], radius=1, sw=2.4)
    v.text(p, x + w / 2, y + h / 2 - 5, "Known 2K", size=26,
           color=C["native"], weight=700, anchor="middle")
    v.text(p, x + w / 2, y + h / 2 + 28, "1920 × 1080", size=21,
           color=C["native"], anchor="middle")
    v.text(p, x + w / 2, y + h - 16, "Target region", size=23,
           color=C["muted"], anchor="middle")
    v.text(p, 62, 603, "Known center across video frames", size=22, color=C["muted"])
    return panel


def attention_node(parent, name, x, w, title, subtitle, kind):
    g = v.group(parent, name)
    center = x + w / 2
    v.text(g, center, 311, title, size=32, color=C[kind], weight=700, anchor="middle")
    v.text(g, center, 342, subtitle, size=22, color=C["muted"], anchor="middle")
    v.rect(g, x, 424, w, 70, fill=C["base"], stroke="#AEBBCB", radius=4, sw=1.5)
    v.text(g, center, 453, "LTX Self-Attention", size=27, weight=700, anchor="middle")
    v.text(g, center, 483, "Q / K / V projections", size=21, color=C["muted"], anchor="middle")
    return g


def adapter(parent, name, x, y, w, label, kind):
    g = v.group(parent, name)
    v.rect(g, x, y, w, 38, fill=C[kind + "_soft"], stroke=C[kind], radius=3, sw=1.2)
    v.text(g, x + w / 2, y + 27, label, size=24, color=C[kind], weight=700, anchor="middle")
    return g


def gate(parent, name, x, y, w, title, subtitle):
    g = v.group(parent, name)
    v.rect(g, x, y, w, 56, fill=C["fusion_soft"], stroke=C["fusion"], radius=5, sw=1.6)
    v.text(g, x + w / 2, y + 25, title, size=27, color=C["fusion"], weight=700, anchor="middle")
    v.text(g, x + w / 2, y + 47, subtitle, size=20, color=C["muted"], anchor="middle")
    return g


def main_panel(parent):
    p = v.group(parent, "b-hierarchical-attention-adaptation")
    panel_header(p, 520, 57, "b", "Hierarchical attention adaptation")
    v.text(p, 2165, 97, "Within each LTX-2.3 block", size=23, color=C["muted"], anchor="end")

    regions = v.group(p, "local-global-hierarchy")
    v.rect(regions, 520, 260, 800, 432, fill="#FAFBFD",
           stroke="#D7DFE9", radius=6, sw=1.2)
    v.text(regions, 540, 248, "LOCAL PATHWAY", size=20, color=C["muted"], weight=700)
    v.rect(regions, 1470, 260, 390, 260, fill="#FAFCFC",
           stroke="#D4E1DE", radius=6, sw=1.2)
    v.text(regions, 1490, 248, "GLOBAL PATHWAY", size=20, color=C["muted"], weight=700)

    # The shared adapter is a parameter bus, not a tensor-processing stage.
    parameter_paths = v.group(p, "shared-parameter-paths")
    dashed(parameter_paths, [(1245, 185), (1245, 218)])
    dashed(parameter_paths, [(533, 218), (1485, 218)])
    for sx, entry in [(533, 550), (935, 950), (1485, 1500)]:
        dashed(parameter_paths, [(sx, 218), (sx, 459), (entry, 459)], arrow=True)
    shared_qkv = v.group(p, "one-shared-op-qkv-adapter")
    v.rect(shared_qkv, 880, 129, 730, 56, fill=C["shared_soft"], stroke=C["shared"], radius=4, sw=1.4)
    v.text(shared_qkv, 1245, 165, "Shared OP-QKV LoRA · r = 32", size=29,
           color=C["shared"], weight=700, anchor="middle")

    tensor_paths = v.group(p, "solid-hidden-feature-flow")
    v.line(tensor_paths, [(715, 494), (715, 520), (915, 520), (915, 546)])
    v.line(tensor_paths, [(1115, 494), (1115, 520), (915, 520)], arrow=False)
    v.line(tensor_paths, [(915, 602), (915, 622)])
    v.line(tensor_paths, [(915, 666), (915, 750), (1095, 750)])
    v.line(tensor_paths, [(1665, 494), (1665, 750), (1485, 750)])
    v.line(tensor_paths, [(1290, 778), (1290, 824)])
    v.line(tensor_paths, [(1520, 864), (1570, 864)])
    v.line(tensor_paths, [(1870, 864), (1920, 864)])

    native = attention_node(p, "native-local-attention", 550, 330, "Native", "Local windows", "native")
    adaptive = attention_node(p, "adaptive-local-attention", 950, 330, "Adaptive", "Anchor–target context", "adaptive")
    global_node = attention_node(p, "global-attention", 1500, 330, "Global", "Full-canvas context", "global")
    adapter(p, "adaptive-qkv-delta", 980, 361, 270, "A-LoRA · r = 64", "adaptive")
    adapter(p, "global-qkv-delta", 1530, 361, 270, "G-LoRA · r = 64", "global")
    dashed(p, [(1115, 399), (1115, 424)], color=C["adaptive"], arrow=True)
    dashed(p, [(1665, 399), (1665, 424)], color=C["global"], arrow=True)

    gate(p, "local-target-masked-gated-fusion", 760, 546, 310, "Local Fusion", "learned · Target-masked")
    assembly = v.group(p, "normalized-window-assembly")
    v.rect(assembly, 760, 622, 310, 44, fill="#FFFFFF", stroke=C["border"], radius=4, sw=1.3)
    v.text(assembly, 915, 652, "Window Assembly", size=26, anchor="middle")
    gate(p, "local-global-gated-fusion", 1095, 722, 390, "Local–Global Fusion", "learned gate")
    v.text(p, 1000, 724, "LocalFull", size=20, color=C["muted"], anchor="middle")
    v.text(p, 1608, 724, "GlobalFull", size=20, color=C["muted"], anchor="middle")

    trunk = v.group(p, "single-shared-output-and-native-backbone")
    v.rect(trunk, 1060, 824, 460, 80, fill=C["base"], stroke="#AEBBCB", radius=4, sw=1.5)
    v.text(trunk, 1290, 856, "Shared Output O", size=29, weight=700, anchor="middle")
    v.text(trunk, 1290, 889, "Base O + OP-O LoRA · r = 32", size=24, color=C["shared"], anchor="middle")
    v.rect(trunk, 1570, 824, 300, 80, fill=C["base"], stroke="#AEBBCB", radius=4, sw=1.5)
    v.text(trunk, 1720, 857, "Text Cross-Attn.", size=26, weight=700, anchor="middle")
    v.text(trunk, 1720, 890, "Frozen", size=22, color=C["muted"], anchor="middle")
    v.rect(trunk, 1920, 824, 270, 80, fill=C["base"], stroke="#AEBBCB", radius=4, sw=1.5)
    v.text(trunk, 2055, 856, "LTX FFN", size=29, weight=700, anchor="middle")
    v.text(trunk, 2055, 889, "+ F-LoRA · r = 32", size=24, color=C["adaptive"], weight=700, anchor="middle")

    legend = v.group(p, "frozen-trainable-and-sharing-legend")
    v.rect(legend, 535, 829, 21, 21, fill=C["base"], stroke="#AEBBCB", radius=2, sw=1.2)
    v.text(legend, 571, 849, "Frozen LTX base", size=24)
    v.line(legend, [(535, 879), (565, 879)])
    v.text(legend, 584, 888, "Feature flow", size=24)
    dashed(legend, [(535, 923), (570, 923)])
    v.text(legend, 584, 931, "Adapter parameters", size=22, color=C["muted"])
    return p


def subset(module, bounds, title):
    x, y, w, h = bounds
    svg = v.root(w, h, title, DESCRIPTION)
    g = v.el(svg, "g", transform=f"translate({-x} {-y})")
    g.append(copy.deepcopy(module))
    return svg


def check(svg, filename):
    report = v.validate_svg(svg, filename)
    ids = [node.get("id") for node in svg.iter() if node.get("id")]
    assert len(ids) == len(set(ids)), (filename, "duplicate IDs")
    assert all("font-style" not in node.attrib for node in svg.iter()), filename
    if "outlined" in filename:
        assert report["text_elements"] == 0, filename
    return report


def main():
    svg = v.root(W, H, "Video Outpainting — Figure 1", DESCRIPTION)
    task = task_panel(svg)
    architecture = main_panel(svg)
    # Two compact panel exports allow independent reuse in a slide or Figma.
    architecture_only = subset(architecture, (500, 10, 1710, 940), "Hierarchical Attention & LoRA")
    for element in list(architecture_only.iter()):
        if element.tag == f"{{{v.NS}}}text" and element.text == "(b)":
            element.text = ""
        elif element.tag == f"{{{v.NS}}}text" and element.text == "Hierarchical attention adaptation":
            element.set("x", "520")
    task_only = subset(task, (28, 10, 448, 702), "2K to 4K Outpainting Task")
    assets = {
        "video_outpainting_figure1.svg": svg,
        "video_outpainting_figure1_outlined.svg": v.outlined(svg),
        "video_outpainting_figure1_architecture.svg": architecture_only,
        "video_outpainting_figure1_architecture_outlined.svg": v.outlined(architecture_only),
        "video_outpainting_figure1_task.svg": task_only,
        "video_outpainting_figure1_task_outlined.svg": v.outlined(task_only),
    }
    reports = {}
    for name, content in assets.items():
        reports[name] = check(content, name)
        v.write_svg(OUT / name, content)
        data = (OUT / name).read_bytes()
        reports[name]["sha256"] = hashlib.sha256(data).hexdigest()
        reports[name]["bytes"] = len(data)
    for name in ["video_outpainting_figure1_outlined.svg", "video_outpainting_figure1_architecture_outlined.svg"]:
        preview = name.replace("_outlined.svg", "_preview.png")
        cairosvg.svg2png(url=str(OUT / name), write_to=str(OUT / preview), background_color="#FFFFFF")
    (OUT / "figure1_validation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    with zipfile.ZipFile(OUT / "video_outpainting_figure1_package.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [*assets, "video_outpainting_figure1_preview.png", "video_outpainting_figure1_architecture_preview.png", "FIGURE1_README.md"]:
            path = OUT / name
            if path.exists():
                archive.write(path, name)
    print(json.dumps({"SVGs": len(assets), "canvas": [W, H], "folder": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
