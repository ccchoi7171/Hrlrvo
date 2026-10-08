"""Build editable, plain-vector presentation assets from the reviewed architecture.

Rendering/outline helpers live in a temporary tools directory, not the model env.
PYTHONPATH=/tmp/videooutpainting-svg-tools python /absolute/path/to/build_svg.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import cairosvg
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont


OUT = Path(__file__).resolve().parent
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)

C = {
    "ink": "#18263D",
    "muted": "#69768B",
    "line": "#77859B",
    "border": "#D9E1ED",
    "base": "#F1F4F8",
    "native": "#326BD1",
    "native_soft": "#F0F5FF",
    "adaptive": "#8760C9",
    "adaptive_soft": "#F7F2FF",
    "global": "#188982",
    "global_soft": "#EFFAF8",
    "shared": "#AD6916",
    "shared_soft": "#FFF5E4",
    "fusion": "#304F75",
    "fusion_soft": "#F2F6FC",
}

FONT_PATHS = {
    "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}
FONTS = {k: TTFont(v) for k, v in FONT_PATHS.items()}


def el(parent, name, **attrs):
    return ET.SubElement(parent, f"{{{NS}}}{name}", {
        k.replace("_", "-"): str(v) for k, v in attrs.items() if v is not None
    })


def group(parent, name):
    return el(parent, "g", id=name)


def rect(parent, x, y, w, h, fill="#FFFFFF", stroke=None, radius=12, sw=1.6):
    return el(parent, "rect", x=x, y=y, width=w, height=h, rx=radius,
              fill=fill, stroke=stroke, stroke_width=sw if stroke else None)


def text(parent, x, y, label, size=22, color=None, weight=400, anchor="start"):
    node = el(parent, "text", x=x, y=y, font_family="DejaVu Sans",
              font_size=size, font_weight=weight, fill=color or C["ink"],
              text_anchor=anchor)
    node.text = label
    return node


def line(parent, points, color=None, width=2.5, arrow=True):
    color = color or C["line"]
    el(parent, "polyline", points=" ".join(f"{x},{y}" for x, y in points),
       fill="none", stroke=color, stroke_width=width,
       stroke_linecap="round", stroke_linejoin="round")
    if arrow:
        a, b = points[-2:]
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dx, dy)
        dx, dy = dx / norm, dy / norm
        back = (b[0] - dx * 10, b[1] - dy * 10)
        pts = [b, (back[0] - dy * 5, back[1] + dx * 5),
               (back[0] + dy * 5, back[1] - dx * 5)]
        el(parent, "polygon", points=" ".join(f"{x:g},{y:g}" for x, y in pts), fill=color)


def root(w, h, title, description):
    svg = ET.Element(f"{{{NS}}}svg", {
        "width": str(w), "height": str(h), "viewBox": f"0 0 {w} {h}",
        "version": "1.1", "role": "img",
    })
    el(svg, "title").text = title
    el(svg, "desc").text = description
    return svg


def label_width(label, size, bold=False):
    font = FONTS["bold" if bold else "regular"]
    cmap, metrics = font.getBestCmap(), font["hmtx"].metrics
    return sum(metrics[cmap[ord(c)]][0] for c in label) * size / font["head"].unitsPerEm


def card(parent, name, x, w, label, subtitle, kind, specializer=None):
    node = group(parent, name)
    y, h = 232, 300
    rect(node, x, y, w, h, stroke=C["border"], radius=18)
    rect(node, x + 24, y + 24, 5, 35, fill=C[kind], radius=2)
    text(node, x + 44, y + 51, label, size=29, weight=700)
    text(node, x + 44, y + 79, subtitle, size=18, color=C["muted"])
    rect(node, x + 26, y + 99, w - 52, 69, fill=C["base"], radius=10)
    text(node, x + w / 2, y + 128, "LTX Self-Attention", size=24, weight=700, anchor="middle")
    text(node, x + w / 2, y + 153, "Base Q / K / V", size=18, color=C["muted"], anchor="middle")
    text(node, x + w / 2, y + 189, "+", size=23, color=C["muted"], anchor="middle")
    rect(node, x + 26, y + 201, w - 52, 40, fill=C["shared_soft"], radius=9)
    text(node, x + w / 2, y + 227, "Shared OP LoRA · r32", size=21,
         color=C["shared"], weight=700, anchor="middle")
    if specializer:
        text(node, x + w / 2, y + 260, "+", size=21, color=C["muted"], anchor="middle")
        rect(node, x + 26, y + 268, w - 52, 28, fill=C[kind + "_soft"], radius=8)
        text(node, x + w / 2, y + 289, specializer, size=20,
             color=C[kind], weight=700, anchor="middle")
    # Both branch specialization rows remain inside the card. Native has no
    # Native-specific LoRA; its deliberately empty lower area is not a module.
    return node, (x - 2, y - 2, w + 4, 312)


def fusion(parent, name, x, y, w, title, subtitle, number):
    node = group(parent, name)
    rect(node, x, y, w, 58, fill=C["fusion_soft"], stroke="#CFDAE9", radius=14)
    el(node, "circle", cx=x + 29, cy=y + 29, r=15, fill=C["fusion"])
    text(node, x + 29, y + 35, str(number), size=17, color="#FFFFFF", weight=700, anchor="middle")
    text(node, x + w / 2 + 15, y + 25, title, size=22,
         color=C["fusion"], weight=700, anchor="middle")
    text(node, x + w / 2 + 15, y + 46, subtitle, size=16,
         color=C["muted"], anchor="middle")
    return node, (x - 2, y - 2, w + 4, 62)


def write_svg(path, svg):
    ET.indent(svg, space="  ")
    ET.ElementTree(svg).write(path, encoding="utf-8", xml_declaration=True)


def outlined(svg):
    result = copy.deepcopy(svg)
    count = 0
    for parent in list(result.iter()):
        for index, node in enumerate(list(parent)):
            if node.tag != f"{{{NS}}}text":
                continue
            label = node.text or ""
            font = FONTS["bold" if int(node.get("font-weight", "400")) >= 600 else "regular"]
            glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
            metrics = font["hmtx"].metrics
            scale = float(node.get("font-size")) / font["head"].unitsPerEm
            advance = sum(metrics[cmap[ord(c)]][0] for c in label) * scale
            x, y = float(node.get("x")), float(node.get("y"))
            if node.get("text-anchor") == "middle":
                x -= advance / 2
            elif node.get("text-anchor") == "end":
                x -= advance
            new = ET.Element(f"{{{NS}}}g", {
                "id": f"label-{count:03d}", "aria-label": label,
                "fill": node.get("fill", C["ink"]),
                "transform": f"translate({x:g} {y:g}) scale({scale:.9f} {-scale:.9f})",
            })
            cursor = 0
            for char in label:
                name = cmap[ord(char)]
                pen = SVGPathPen(glyphs)
                glyphs[name].draw(pen)
                if pen.getCommands():
                    el(new, "path", d=pen.getCommands(), transform=f"translate({cursor} 0)")
                cursor += metrics[name][0]
            parent.remove(node)
            parent.insert(index, new)
            count += 1
    return result


def crop(module, bounds, title):
    x, y, w, h = bounds
    margin = 12
    cropped = root(w + 2 * margin, h + 2 * margin, title, "Editable standalone architecture module; transparent background.")
    shift = el(cropped, "g", transform=f"translate({margin - x} {margin - y})")
    shift.append(copy.deepcopy(module))
    return cropped


def validate_svg(svg, label):
    forbidden = {"image", "foreignObject", "use", "filter", "script"}
    for node in svg.iter():
        tag = node.tag.split("}")[-1]
        assert tag not in forbidden, (label, "unsupported/raster", tag)
        assert not any("href" in attr for attr in node.attrib), (label, "external reference")
    for node in svg.iter(f"{{{NS}}}text"):
        content = node.text or ""
        size = float(node.get("font-size"))
        width = label_width(content, size, int(node.get("font-weight", "400")) >= 600)
        assert width < 1800, (label, content, width)
    return {
        "elements": sum(1 for _ in svg.iter()),
        "text_elements": sum(1 for _ in svg.iter(f"{{{NS}}}text")),
        "raster_elements": 0,
        "external_references": 0,
    }


def main():
    (OUT / "modules" / "editable").mkdir(parents=True, exist_ok=True)
    (OUT / "modules" / "outlined").mkdir(parents=True, exist_ok=True)
    svg = root(1920, 1080, "2K to 4K Video Outpainting — Hierarchical Attention & LoRA",
               "Presentation abstraction of each LTX-2.3 transformer block. Native and Adaptive are fused per local window, then assembled; the Local result is fused with Global. Shared OP QKV LoRA has the same weights across all three views. A/G LoRA specialize the two branches. A single shared O projection is followed by native cross-attention and FFN. Native normalization, residuals, head gates, compression/restoration and guidance are omitted for clarity.")
    heading = group(svg, "heading")
    text(heading, 72, 69, "2K → 4K Video Outpainting", size=39, weight=700)
    text(heading, 74, 106, "Hierarchical Attention & LoRA", size=22, color=C["muted"])
    text(heading, 1848, 68, "LTX-2.3 · Transformer Block ×48", size=21, color=C["muted"], anchor="end")

    zones = group(svg, "local-global-hierarchy")
    rect(zones, 64, 162, 1148, 542, fill="#FCFDFF", stroke=C["border"], radius=24)
    text(zones, 96, 205, "Local System", size=26, weight=700)
    text(zones, 1180, 203, "per local window", size=18, color=C["muted"], anchor="end")
    rect(zones, 1250, 162, 606, 542, fill="#FCFEFE", stroke=C["border"], radius=24)
    text(zones, 1282, 205, "Global", size=26, weight=700)
    text(zones, 1824, 203, "full canvas", size=18, color=C["muted"], anchor="end")

    connectors = group(svg, "flow-connectors")
    # Two-stage fusion. Window assembly belongs between the two fusion nodes.
    line(connectors, [(346, 532), (346, 550), (638, 550), (638, 566)], color=C["native"])
    line(connectors, [(930, 532), (930, 550), (638, 550)], color=C["adaptive"], arrow=False)
    line(connectors, [(638, 624), (638, 646)])
    line(connectors, [(638, 688), (638, 753), (770, 753)])
    line(connectors, [(1553, 532), (1553, 753), (1170, 753)], color=C["global"])
    line(connectors, [(970, 782), (970, 810)])

    modules = {}
    modules["01_native_local"] = card(svg, "native-local", 96, 500, "Native", "Local detail", "native")
    modules["02_adaptive_local"] = card(svg, "adaptive-local", 680, 500, "Adaptive", "Anchor–Target context", "adaptive", "A-LoRA · r64")
    modules["03_global"] = card(svg, "global-attention", 1282, 542, "Global", "Canvas-wide context", "global", "G-LoRA · r64")
    modules["04_local_fusion"] = fusion(svg, "local-fusion", 458, 566, 360, "Local Fusion", "learned gate · Target-masked", 1)
    assemble = group(svg, "local-window-assembly")
    rect(assemble, 458, 646, 360, 42, fill="#FFFFFF", stroke=C["border"], radius=10)
    text(assemble, 638, 673, "Window Assembly", size=20, color=C["muted"], anchor="middle")
    modules["05_local_global_fusion"] = fusion(svg, "local-global-fusion", 770, 724, 400, "Local–Global Fusion", "learned gate", 2)

    shared = group(svg, "shared-post-attention")
    line(shared, [(970, 872), (970, 896)])
    line(shared, [(970, 952), (970, 976)])
    projection = group(shared, "shared-output-projection")
    rect(projection, 710, 810, 520, 62, fill=C["shared_soft"], stroke="#EFDCBE", radius=13)
    text(projection, 970, 835, "Shared Output Projection", size=23, weight=700, anchor="middle")
    text(projection, 970, 859, "Base O + OP LoRA · r32", size=19, color=C["shared"], anchor="middle")
    ca = group(shared, "native-cross-attention")
    rect(ca, 740, 896, 460, 56, fill=C["base"], stroke=C["border"], radius=13)
    text(ca, 970, 931, "LTX Cross-Attention", size=24, weight=700, anchor="middle")
    ffn = group(shared, "shared-ffn")
    rect(ffn, 710, 976, 520, 62, fill=C["base"], stroke=C["border"], radius=13)
    text(ffn, 970, 1001, "LTX FFN", size=23, weight=700, anchor="middle")
    text(ffn, 970, 1025, "+ F-LoRA · r32", size=19, color=C["adaptive"], weight=700, anchor="middle")
    modules["06_shared_post_attention"] = (shared, (708, 808, 524, 232))

    notes = group(svg, "weight-sharing-notes")
    rect(notes, 96, 817, 4, 115, fill=C["shared"], radius=2)
    text(notes, 116, 841, "SHARED OP WEIGHTS", size=17, color=C["shared"], weight=700)
    text(notes, 116, 878, "Same Q/K/V LoRA in", size=21)
    text(notes, 116, 911, "Native · Adaptive · Global", size=21)
    text(notes, 1430, 842, "LTX BASE FROZEN", size=17, color=C["muted"], weight=700)
    text(notes, 1430, 879, "Shared O, Cross-Attention", size=20)
    text(notes, 1430, 912, "and FFN run once per block", size=20)

    line(svg, [(970, 1038), (970, 1063)], arrow=True)

    records = {}
    editable_path = OUT / "video_outpainting_architecture.svg"
    outline_path = OUT / "video_outpainting_architecture_outlined.svg"
    records[editable_path.name] = validate_svg(svg, editable_path.name)
    write_svg(editable_path, svg)
    outline = outlined(svg)
    records[outline_path.name] = validate_svg(outline, outline_path.name)
    assert records[outline_path.name]["text_elements"] == 0
    write_svg(outline_path, outline)
    # Slide content only: the presenter can use the deck's own title above it.
    core = root(1840, 940, "Video Outpainting — Core Architecture",
                svg.find(f"{{{NS}}}desc").text)
    core_group = el(core, "g", transform="translate(-40 -145)")
    for child in svg:
        if child.tag in (f"{{{NS}}}title", f"{{{NS}}}desc") or child.get("id") == "heading":
            continue
        core_group.append(copy.deepcopy(child))
    core_path = OUT / "video_outpainting_core.svg"
    core_outline_path = OUT / "video_outpainting_core_outlined.svg"
    for destination, content in [(core_path, core), (core_outline_path, outlined(core))]:
        records[destination.name] = validate_svg(content, destination.name)
        write_svg(destination, content)
    for name, (module, bounds) in modules.items():
        asset = crop(module, bounds, name.replace("_", " "))
        for variant, content in [("editable", asset), ("outlined", outlined(asset))]:
            destination = OUT / "modules" / variant / f"{name}.svg"
            records[str(destination.relative_to(OUT))] = validate_svg(content, name)
            write_svg(destination, content)

    cairosvg.svg2png(url=str(editable_path), write_to=str(OUT / "architecture_preview.png"),
                    output_width=1920, output_height=1080, background_color="#FFFFFF")
    cairosvg.svg2png(url=str(outline_path), write_to=str(OUT / "architecture_outlined_preview.png"),
                    output_width=1920, output_height=1080, background_color="#FFFFFF")
    cairosvg.svg2png(url=str(core_outline_path), write_to=str(OUT / "architecture_core_preview.png"),
                    output_width=1840, output_height=940, background_color="#FFFFFF")

    for name in records:
        data = (OUT / name).read_bytes()
        records[name]["sha256"] = hashlib.sha256(data).hexdigest()
        records[name]["bytes"] = len(data)
    (OUT / "validation.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with zipfile.ZipFile(OUT / "video_outpainting_svg_package.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        paths = [editable_path, outline_path, core_path, core_outline_path,
                 OUT / "architecture_preview.png", OUT / "architecture_core_preview.png", OUT / "README.md"]
        paths.extend(sorted((OUT / "modules").rglob("*.svg")))
        for path in paths:
            if path.exists():
                archive.write(path, path.relative_to(OUT))
    print(json.dumps({"output": str(OUT), "svg_files": len(records), "size": "1920 × 1080", "raster_in_svg": 0}, indent=2))


if __name__ == "__main__":
    main()
