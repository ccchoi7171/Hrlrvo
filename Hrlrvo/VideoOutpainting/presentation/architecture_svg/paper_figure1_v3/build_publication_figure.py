"""Build the verified-frame publication overview without running model code.

PYTHONPATH=/tmp/videooutpainting-svg-tools python /absolute/path/to/this_file.py
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

import cairosvg
from PIL import Image

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
import build_svg as v

W, H = 2400, 1230
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("xlink", XLINK)
SOURCE = OUT.parents[2] / "main" / "src" / "ltx_outpaint" / "geometry.py"

v.C.update({
    "ink": "#233149", "muted": "#68768B", "line": "#49586D",
    "border": "#CDD6E2", "base": "#F1F4F8",
    "native": "#326AA5", "native_soft": "#EAF2FC",
    "adaptive": "#8662AA", "adaptive_soft": "#F4EEF9",
    "global": "#307F78", "global_soft": "#EDF7F5",
    "shared": "#A87420", "shared_soft": "#FFF6E6",
    "fusion": "#3D5577", "fusion_soft": "#F1F5FA",
})
C = v.C

DESCRIPTION = (
    "Frame-aligned video outpainting overview. Panel (a) embeds actual decoded frame 24 "
    "at 1.000 seconds from the user-provided 1920x1080 anchor and 3840x2160 test-set "
    "reference videos. The reference center crop is pixel-identical to the anchor. "
    "The reference illustrates the 4K output canvas and is not presented as model-generated "
    "performance. Input and its footprint in the reference have identical displayed width "
    "and height. Panel (b) depicts the normal LTX-2.3 attention path: one shared OP QKV "
    "LoRA r32, Adaptive A r64 and Global G r64 deltas, Native+Adaptive Target-masked "
    "fusion, normalized window assembly, Local+Global fusion, one shared O with OP O r32, "
    "native Text Cross-Attention, and FFN with F r32. Spatial-support icons depict a code-derived "
    "example of one K4 Local window, not the complete temporal token layout. Solid arrows "
    "carry features, dashed lines terminate in plus signs for additive LoRA parameters. Norms, residuals, "
    "C/R tokenization, RoPE, guidance and VAE/sampler details are intentionally omitted."
)


def geometry_example():
    spec = importlib.util.spec_from_file_location("figure_source_geometry", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    anchor = module.Rect(17, 30, 51, 90)
    target = module.local_windows(68, 120, 4)[3]
    patch = module.anchor_patch(anchor, target)
    layout = module.adaptive_layout(anchor, target, 68, 120, 4)
    def coords(rect):
        return [rect.y0, rect.x0, rect.y1, rect.x1]
    result = {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "coordinate_convention": "half-open [y0,x0,y1,x1] in the native spatial latent grid",
        "grid": [68, 120], "example_partition_count": 4,
        "anchor": coords(anchor), "target": coords(target),
        "selected_anchor_patch": coords(patch), "adaptive_read_support": coords(layout.rectangle),
        "spatial_icon_scope": "one representative Local window; time/token resolution omitted",
    }
    (OUT / "figure_geometry.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def dashed(parent, points, kind="shared", arrow=False):
    n = len(parent)
    v.line(parent, points, color=C[kind], width=2.5, arrow=arrow)
    parent[n].set("stroke-dasharray", "8 6")


def injection(parent, x, y, kind="shared"):
    v.el(parent, "circle", cx=x, cy=y, r=7, fill="#FFFFFF", stroke=C[kind], stroke_width=1.8)
    v.line(parent, [(x - 3.4, y), (x + 3.4, y)], color=C[kind], width=1.5, arrow=False)
    v.line(parent, [(x, y - 3.4), (x, y + 3.4)], color=C[kind], width=1.5, arrow=False)


def photo(parent, path, x, y, width, height):
    with Image.open(path) as image:
        assert image.width * height == image.height * width, (path, "aspect ratio changed")
    node = v.el(parent, "image", x=x, y=y, width=width, height=height,
                preserveAspectRatio="xMidYMid meet")
    node.set(f"{{{XLINK}}}href", "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii"))
    return node


def border(parent, x, y, w, h, color, sw=3):
    return v.rect(parent, x, y, w, h, fill="none", stroke=color, radius=0, sw=sw)


def header(parent, x, title, letter):
    v.text(parent, x, 55, f"({letter})", size=34, weight=700)
    v.text(parent, x + 63, 55, title, size=34, weight=700)


def task_panel(svg):
    p = v.group(svg, "a-matched-real-video-frames")
    header(p, 48, "2K → 4K video outpainting", "a")
    v.text(p, 48, 98, "Matched frame · t = 1.000 s", size=33, color=C["muted"])

    # Identical display scale: the 2K input is the exact size of the known
    # central footprint in the 4K reference (336x189 in both places).
    v.text(p, 216, 142, "2K input", size=34, weight=700)
    v.text(p, 216, 179, "1920 × 1080", size=33, color=C["muted"])
    photo(p, OUT / "media" / "input_frame_024.png", 216, 200, 336, 189)
    border(p, 216, 200, 336, 189, C["native"], sw=3.5)

    v.line(p, [(384, 389), (384, 448)], width=3)
    model = v.group(p, "task-spatial-expansion")
    v.rect(model, 116, 448, 536, 82, fill=C["fusion_soft"], stroke="#B9C6D9", radius=5, sw=1.5)
    v.text(model, 384, 482, "Spatial outpainting", size=33, weight=700, anchor="middle")
    v.text(model, 384, 513, "Expand beyond the input", size=33,
           color=C["muted"], anchor="middle")
    v.line(p, [(384, 530), (384, 585)], width=3)
    v.text(p, 48, 604, "4K output canvas", size=33, weight=700)
    v.text(p, 48, 645, "3840 × 2160 · reference frame", size=33, color=C["muted"])
    photo(p, OUT / "media" / "output_frame_024.png", 48, 656, 672, 378)
    border(p, 48, 656, 672, 378, C["adaptive"], sw=2.4)
    # Pixel-verified crop x=960:2880, y=540:1620 in the 3840x2160 frame.
    border(p, 216, 750.5, 336, 189, C["native"], sw=4)

    legend = v.group(p, "known-anchor-and-target-surround")
    border(legend, 64, 1066, 24, 24, C["native"], sw=3)
    v.text(legend, 104, 1087, "Known anchor", size=33)
    border(legend, 375, 1066, 24, 24, C["adaptive"], sw=2.4)
    v.text(legend, 415, 1087, "Target surround", size=33)
    v.text(p, 48, 1135, "Input matches the reference center.", size=33, color=C["muted"])
    return p


def support_icon(parent, name, x, kind, geo):
    g = v.group(parent, name)
    y, width, height = 348, 240, 136
    scale = 2.0
    bg = "#F9FAFD" if kind != "global" else C["global_soft"]
    v.rect(g, x, y, width, height, fill=bg, stroke="#8D9FB7", radius=1, sw=1.7)
    ay0, ax0, ay1, ax1 = geo["anchor"]
    ty0, tx0, ty1, tx1 = geo["target"]
    anchor_x, anchor_y = x + ax0 * scale, y + ay0 * scale
    anchor_w, anchor_h = (ax1 - ax0) * scale, (ay1 - ay0) * scale
    v.rect(g, anchor_x, anchor_y, anchor_w, anchor_h, fill="#EDF3FD", stroke="#7595C0", radius=0, sw=1.6)
    if kind == "native":
        v.rect(g, x + tx0 * scale, y + ty0 * scale, (tx1 - tx0) * scale, (ty1 - ty0) * scale,
               fill="#D8C5E7", stroke=C["adaptive"], radius=0, sw=1.8)
    elif kind == "adaptive":
        sy0, sx0, sy1, sx1 = geo["adaptive_read_support"]
        v.rect(g, x + sx0 * scale, y + sy0 * scale, (sx1 - sx0) * scale, (sy1 - sy0) * scale,
               fill="#EAE4F0", radius=0)
        iy0, ix0, iy1, ix1 = max(ay0, sy0), max(ax0, sx0), min(ay1, sy1), min(ax1, sx1)
        v.rect(g, x + ix0 * scale, y + iy0 * scale, (ix1 - ix0) * scale, (iy1 - iy0) * scale,
               fill="#C7DCF7", radius=0)
        v.rect(g, x + tx0 * scale, y + ty0 * scale, (tx1 - tx0) * scale, (ty1 - ty0) * scale,
               fill="#D8C5E7", stroke=C["adaptive"], radius=0, sw=1.8)
        py0, px0, py1, px1 = geo["selected_anchor_patch"]
        border(g, x + px0 * scale, y + py0 * scale, (px1 - px0) * scale, (py1 - py0) * scale,
               C["native"], sw=1.8)
        border(g, x + sx0 * scale, y + sy0 * scale, (sx1 - sx0) * scale, (sy1 - sy0) * scale,
               C["adaptive"], sw=1.5)
    else:
        # Uniform global read support. The sparse decorative grid deliberately
        # does not claim a C4 token count or a 4K pixel-patch partition.
        for i in range(1, 6):
            v.line(g, [(x + i * width / 6, y), (x + i * width / 6, y + height)],
                   color="#B1CEC6", width=1.1, arrow=False)
        for i in range(1, 4):
            v.line(g, [(x, y + i * height / 4), (x + width, y + i * height / 4)],
                   color="#B1CEC6", width=1.1, arrow=False)
        border(g, x, y, width, height, C["global"], sw=2)
    return g


def branch(parent, x, width, title, subtitle, kind, geo):
    g = v.group(parent, f"{kind}-attention-view")
    center = x + width / 2
    v.text(g, center, 314, title, size=36, color=C[kind], weight=700, anchor="middle")
    support_icon(g, f"{kind}-illustrative-read-support", center - 120, kind, geo)
    v.text(g, center, 523, subtitle, size=33, color=C["muted"], anchor="middle")
    if kind in ("adaptive", "global"):
        delta = "A" if kind == "adaptive" else "G"
        v.rect(g, center - 150, 549, 300, 42, fill=C[kind + "_soft"], stroke=C[kind], radius=3, sw=1.3)
        v.text(g, center, 579, f"{delta}-LoRA · r = 64", size=33, color=C[kind], weight=700, anchor="middle")
        dashed(g, [(center, 591), (center, 622)], kind=kind)
    v.rect(g, x, 622, width, 82, fill=C["base"], stroke="#B6C3D5", radius=4, sw=1.5)
    v.text(g, center, 657, "Self-Attention", size=33, weight=700, anchor="middle")
    v.text(g, center, 689, "Q / K / V", size=33, color=C["muted"], anchor="middle")
    return g


def fusion(parent, name, x, y, width, label, subtitle, number):
    g = v.group(parent, name)
    v.rect(g, x, y, width, 74, fill=C["fusion_soft"], stroke=C["fusion"], radius=4, sw=1.7)
    v.el(g, "circle", cx=x + 32, cy=y + 37, r=20, fill=C["fusion"])
    v.text(g, x + 32, y + 49, str(number), size=33, color="#FFFFFF", weight=700, anchor="middle")
    v.text(g, x + width / 2 + 19, y + 30, label, size=33, color=C["fusion"], weight=700, anchor="middle")
    v.text(g, x + width / 2 + 19, y + 64, subtitle, size=33, color=C["muted"], anchor="middle")
    return g


def architecture_panel(svg, geo):
    p = v.group(svg, "b-two-stage-hierarchical-attention")
    header(p, 790, "Hierarchical attention & LoRA", "b")
    v.text(p, 790, 99, "LTX-2.3 transformer block", size=33, color=C["muted"])

    regions = v.group(p, "local-and-global-pathways")
    v.rect(regions, 790, 272, 790, 663, fill="#FAFBFD", stroke="#D5DFEB", radius=5, sw=1.3)
    v.text(regions, 838, 255, "LOCAL PATHWAY", size=33, color=C["muted"], weight=700)
    v.rect(regions, 1650, 272, 390, 468, fill="#F8FCFB", stroke="#D1E3DE", radius=5, sw=1.3)
    v.text(regions, 1685, 255, "GLOBAL PATHWAY", size=33, color=C["muted"], weight=700)

    params = v.group(p, "shared-and-specialized-projection-parameters")
    dashed(params, [(1495, 201), (1495, 226)], arrow=False)
    dashed(params, [(814, 226), (1664, 226)], arrow=False)
    for bus_x, left_x in [(814, 830), (1204, 1220), (1664, 1680)]:
        dashed(params, [(bus_x, 226), (bus_x, 663), (left_x, 663)])
    shared = v.group(p, "one-shared-op-qkv-lora")
    v.rect(shared, 1090, 143, 810, 58, fill=C["shared_soft"], stroke=C["shared"], radius=4, sw=1.4)
    v.text(shared, 1495, 182, "Shared OP-QKV LoRA · r = 32", size=33,
           color=C["shared"], weight=700, anchor="middle")

    flow = v.group(p, "solid-feature-flow")
    v.line(flow, [(975, 704), (975, 733), (1175, 733), (1175, 758)], width=2.8)
    v.line(flow, [(1375, 704), (1375, 733), (1175, 733)], width=2.8, arrow=False)
    v.line(flow, [(1175, 832), (1175, 858)], width=2.8)
    v.line(flow, [(1175, 910), (1175, 1010), (1280, 1010)], width=2.8)
    v.line(flow, [(1845, 704), (1845, 1010), (1740, 1010)], width=2.8)
    v.line(flow, [(1510, 1047), (1510, 1080)], width=2.8)
    v.line(flow, [(1740, 1121), (1790, 1121)], width=2.8)
    v.line(flow, [(2055, 1121), (2110, 1121)], width=2.8)

    branch(p, 830, 290, "Native", "Local detail", "native", geo)
    branch(p, 1220, 310, "Adaptive", "Anchor–target link", "adaptive", geo)
    branch(p, 1680, 330, "Global", "Full-canvas context", "global", geo)
    points = v.group(p, "additive-lora-injection-markers")
    for edge in (830, 1220, 1680):
        injection(points, edge, 663)
    injection(points, 1375, 622, "adaptive")
    injection(points, 1845, 622, "global")
    fusion(p, "1-native-adaptive-local-fusion", 1000, 758, 350, "Local Fusion", "Target-masked", 1)
    assembly = v.group(p, "overlap-normalized-window-assembly")
    v.rect(assembly, 1000, 858, 350, 52, fill="#FFFFFF", stroke=C["border"], radius=4, sw=1.4)
    v.text(assembly, 1175, 893, "Window Assembly", size=33, anchor="middle")
    fusion(p, "2-local-global-fusion", 1280, 973, 460, "Local–Global Fusion", "learned gate", 2)
    v.text(p, 1260, 964, "LocalFull", size=33, color=C["muted"], anchor="middle")
    v.text(p, 1755, 964, "GlobalFull", size=33, color=C["muted"], anchor="middle")

    tail = v.group(p, "single-shared-post-attention-tail")
    v.rect(tail, 1280, 1080, 460, 82, fill=C["base"], stroke="#B6C3D5", radius=4, sw=1.5)
    v.text(tail, 1510, 1114, "Shared Output O", size=33, weight=700, anchor="middle")
    v.text(tail, 1510, 1148, "+ OP-O LoRA · r = 32", size=33, color=C["shared"], anchor="middle")
    v.rect(tail, 1790, 1080, 265, 82, fill=C["base"], stroke="#B6C3D5", radius=4, sw=1.5)
    v.text(tail, 1922.5, 1114, "Text", size=33, weight=700, anchor="middle")
    v.text(tail, 1922.5, 1148, "Cross-Attn.", size=33, weight=700, anchor="middle")
    v.rect(tail, 2110, 1080, 250, 82, fill=C["base"], stroke="#B6C3D5", radius=4, sw=1.5)
    v.text(tail, 2235, 1114, "LTX FFN", size=33, weight=700, anchor="middle")
    v.text(tail, 2235, 1148, "+ F-LoRA · r32", size=33, color=C["adaptive"], anchor="middle")

    legend = v.group(p, "feature-and-parameter-legend")
    v.rect(legend, 814, 1070, 24, 24, fill=C["base"], stroke="#B6C3D5", radius=2, sw=1.2)
    v.text(legend, 858, 1092, "Frozen base", size=33)
    v.line(legend, [(814, 1121), (847, 1121)], width=2.8)
    v.text(legend, 865, 1130, "Features", size=33)
    dashed(legend, [(814, 1161), (847, 1161)], arrow=False)
    injection(legend, 847, 1161)
    v.text(legend, 865, 1170, "LoRA parameters", size=33, color=C["muted"])
    return p


def cropped(module, bounds, title, panel_letter):
    x, y, w, h = bounds
    svg = v.root(w, h, title, DESCRIPTION)
    shifted = v.el(svg, "g", transform=f"translate({-x} {-y})")
    module = copy.deepcopy(module)
    for node in module.iter(f"{{{v.NS}}}text"):
        if node.text == f"({panel_letter})":
            node.text = ""
        elif panel_letter == "b" and node.text == "Hierarchical attention & LoRA":
            node.set("x", "790")
        elif panel_letter == "a" and node.text == "2K → 4K video outpainting":
            node.set("x", "48")
    shifted.append(module)
    return svg


def validate(svg, name):
    ids = [n.get("id") for n in svg.iter() if n.get("id")]
    assert len(ids) == len(set(ids)), name
    images = []
    for n in svg.iter():
        assert n.tag.split("}")[-1] not in {"filter", "foreignObject", "script", "use"}, name
        if n.tag == f"{{{v.NS}}}image":
            uri = n.get(f"{{{XLINK}}}href")
            assert uri.startswith("data:image/png;base64,"), name
            data = base64.b64decode(uri.split(",", 1)[1], validate=True)
            images.append(hashlib.sha256(data).hexdigest())
        else:
            assert not any("href" in key for key in n.attrib), name
    texts = list(svg.iter(f"{{{v.NS}}}text"))
    minimum_font_size = min((float(n.get("font-size")) for n in texts), default=None)
    if texts:
        assert minimum_font_size >= 33, (name, "manuscript text is too small")
    if "outlined" in name:
        assert not texts, name
    if "architecture" in name:
        assert not images, name
    return {"text_count": len(texts), "embedded_photo_count": len(images),
            "photo_sha256": images, "external_dependencies": 0, "unique_layer_ids": True,
            "minimum_font_size_svg_units": minimum_font_size}


def main():
    geo = geometry_example()
    alignment = json.loads((OUT / "media" / "media_alignment.json").read_text())
    assert alignment["analysis_summary"]["representative_frame_index"] == 24
    assert alignment["all_frame_center_crop_check"]["pixel_exact_frames"] == 94
    svg = v.root(W, H, "Video Outpainting — Frame-aligned Figure 1", DESCRIPTION)
    task = task_panel(svg)
    architecture = architecture_panel(svg, geo)
    task_only = cropped(task, (24, 12, 728, 1175), "Frame-aligned 2K Input and 4K Reference", "a")
    architecture_only = cropped(architecture, (772, 12, 1604, 1190), "Hierarchical Attention and LoRA", "b")
    roots = {
        "figure1_video_outpainting.svg": svg,
        "figure1_video_outpainting_outlined.svg": v.outlined(svg),
        "figure1_architecture.svg": architecture_only,
        "figure1_architecture_outlined.svg": v.outlined(architecture_only),
        "figure1_matched_frames.svg": task_only,
        "figure1_matched_frames_outlined.svg": v.outlined(task_only),
    }
    report = {}
    for name, root in roots.items():
        report[name] = validate(root, name)
        v.write_svg(OUT / name, root)
        report[name]["bytes"] = (OUT / name).stat().st_size
        report[name]["sha256"] = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    for svg_name, preview in [
        ("figure1_video_outpainting_outlined.svg", "figure1_preview.png"),
        ("figure1_architecture_outlined.svg", "figure1_architecture_preview.png"),
        ("figure1_matched_frames_outlined.svg", "figure1_matched_frames_preview.png"),
    ]:
        cairosvg.svg2png(url=str(OUT / svg_name), write_to=str(OUT / preview), background_color="#FFFFFF")
    # Actual manuscript-width vector PDF (180 mm), alongside the editable SVG.
    cairosvg.svg2pdf(url=str(OUT / "figure1_video_outpainting_outlined.svg"),
                    write_to=str(OUT / "figure1_print.pdf"), output_width=180 * 96 / 25.4)
    (OUT / "figure_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    members = [*roots, "figure1_preview.png", "figure1_architecture_preview.png", "figure1_matched_frames_preview.png",
               "figure1_print.pdf", "README.md", "figure_geometry.json", "figure_validation.json",
               "media/input_frame_024.png", "media/output_frame_024.png", "media/media_alignment.json"]
    with zipfile.ZipFile(OUT / "figure1_publication_package.zip", "w", zipfile.ZIP_DEFLATED) as package:
        for member in members:
            path = OUT / member
            if path.exists():
                package.write(path, member)
    print(json.dumps({"directory": str(OUT), "canvas": [W, H], "SVGs": len(roots),
                      "frame_index": 24, "timestamp_seconds": 1.0, "source_center_pixel_exact": True}, indent=2))


if __name__ == "__main__":
    main()
