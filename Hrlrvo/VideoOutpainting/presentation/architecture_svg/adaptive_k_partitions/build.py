"""Draw the actual 4K K=4 and K=5 Adaptive Local geometry for slides."""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree as ET


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "main" / "src"))

from ltx_outpaint.geometry import Rect, adaptive_layout, anchor_patch, local_windows  # noqa: E402


CANVAS_H, CANVAS_W = 68, 120
ANCHOR = Rect(17, 30, 51, 90)
PLOT_X, PLOT_Y, PLOT_W = 52, 119, 616
PLOT_H = PLOT_W * CANVAS_H / CANVAS_W
SX = PLOT_W / CANVAS_W
SY = PLOT_H / CANVAS_H

COLORS = {
    "ink": "#28252D",
    "muted": "#625D6A",
    "window": "#BC9ACA",
    "canvas": "#8655A8",
    "bridge": "#B888CC",
    "bridge_fill": "#F1E8F7",
    "anchor": "#5D80DF",
    "anchor_fill": "#CCDAFF",
    "target": "#ED559B",
    "target_fill": "#FFBAD8",
}


def add(parent: ET.Element, tag: str, **attributes: object) -> ET.Element:
    return ET.SubElement(parent, tag, {key.replace("_", "-"): str(value) for key, value in attributes.items()})


def text(parent: ET.Element, x: float, y: float, label: str, *, size: int = 22,
         fill: str | None = None, weight: int = 600, anchor: str = "start") -> None:
    node = add(parent, "text", x=f"{x:.2f}", y=f"{y:.2f}", fill=fill or COLORS["ink"],
               font_family="DejaVu Sans, Arial, sans-serif", font_size=size,
               font_weight=weight, text_anchor=anchor)
    node.text = label


def region(parent: ET.Element, rect: Rect, **styles: object) -> ET.Element:
    return add(parent, "rect", x=f"{PLOT_X + rect.x0 * SX:.2f}",
               y=f"{PLOT_Y + rect.y0 * SY:.2f}",
               width=f"{rect.width * SX:.2f}", height=f"{rect.height * SY:.2f}", **styles)


def center(rect: Rect) -> tuple[float, float]:
    return PLOT_X + (rect.x0 + rect.x1) * SX / 2, PLOT_Y + (rect.y0 + rect.y1) * SY / 2


def build(k: int) -> Path:
    windows = local_windows(CANVAS_H, CANVAS_W, k)
    target = windows[k - 1]  # Upper-right Local window for both partitions.
    patch = anchor_patch(ANCHOR, target)
    layout = adaptive_layout(ANCHOR, target, CANVAS_H, CANVAS_W, 4)
    assert layout.rectangle is not None
    assert len(windows) == k * k
    assert patch.height == target.height and patch.width == target.width

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": "720", "height": "700", "viewBox": "0 0 720 700",
        "role": "img", "aria-label": f"4K Adaptive Local geometry with {k} by {k} windows",
    })
    add(svg, "title").text = f"4K Adaptive Local: {k} × {k} windows"
    add(svg, "desc").text = (
        "Actual 68 × 120 latent grid with the central 34 × 60 Anchor, the upper-right "
        "Target Local window, and its Adaptive read support. The dashed blue box is "
        "the same-sized Anchor reference patch."
    )
    add(svg, "rect", x="0", y="0", width="720", height="700", fill="#FFFFFF")
    text(svg, 52, 49, f"K = {k}  ·  {k} × {k} Local windows", size=29, weight=700)
    text(svg, 52, 82, "4K canvas  ·  120 × 68 latent", size=19, fill=COLORS["muted"], weight=400)

    region(svg, layout.rectangle, fill=COLORS["bridge_fill"])
    region(svg, ANCHOR, fill=COLORS["anchor_fill"])
    region(svg, target, fill=COLORS["target_fill"])

    # Local windows may overlap by one latent cell. Draw one line at the middle
    # of each shared boundary so their two outlines do not appear as a stripe.
    rows = sorted({(window.y0, window.y1) for window in windows})
    columns = sorted({(window.x0, window.x1) for window in windows})
    assert len(rows) == len(columns) == k
    for previous, following in zip(rows, rows[1:]):
        boundary = (previous[1] + following[0]) / 2
        y = PLOT_Y + boundary * SY
        add(svg, "line", x1=f"{PLOT_X:.2f}", y1=f"{y:.2f}",
            x2=f"{PLOT_X + PLOT_W:.2f}", y2=f"{y:.2f}",
            stroke=COLORS["window"], stroke_width="1.45")
    for previous, following in zip(columns, columns[1:]):
        boundary = (previous[1] + following[0]) / 2
        x = PLOT_X + boundary * SX
        add(svg, "line", x1=f"{x:.2f}", y1=f"{PLOT_Y:.2f}",
            x2=f"{x:.2f}", y2=f"{PLOT_Y + PLOT_H:.2f}",
            stroke=COLORS["window"], stroke_width="1.45")

    region(svg, Rect(0, 0, CANVAS_H, CANVAS_W), fill="none",
           stroke=COLORS["canvas"], stroke_width="2.2")
    region(svg, layout.rectangle, fill="none", stroke=COLORS["bridge"],
           stroke_width="3", stroke_dasharray="8 5")
    region(svg, ANCHOR, fill="none", stroke=COLORS["anchor"], stroke_width="2.8")
    region(svg, patch, fill="none", stroke=COLORS["anchor"],
           stroke_width="2.3", stroke_dasharray="6 4")
    region(svg, target, fill="none", stroke=COLORS["target"], stroke_width="4")

    ax, ay = center(patch)
    tx, ty = center(target)
    # The purple read support outside A/T is Bridge. This point stays within a
    # pure-Bridge area in both the K=4 and K=5 examples.
    bx = PLOT_X + ((layout.rectangle.x0 + target.x0) / 2) * SX
    by = PLOT_Y + (target.y0 + target.height / 2) * SY
    text(svg, ax, ay + 8, "A", size=29, fill=COLORS["anchor"], weight=700, anchor="middle")
    text(svg, bx, by + 8, "B", size=29, fill=COLORS["canvas"], weight=700, anchor="middle")
    text(svg, tx, ty + 8, "T", size=29, fill=COLORS["target"], weight=700, anchor="middle")

    add(svg, "rect", x="53", y="511", width="15", height="15", rx="2",
        fill=COLORS["anchor_fill"], stroke=COLORS["anchor"], stroke_width="1.8")
    text(svg, 77, 525, "Anchor", size=17, fill=COLORS["muted"], weight=500)
    add(svg, "rect", x="237", y="511", width="15", height="15", rx="2",
        fill=COLORS["bridge_fill"], stroke=COLORS["bridge"], stroke_width="1.8")
    text(svg, 261, 525, "Bridge", size=17, fill=COLORS["muted"], weight=500)
    add(svg, "rect", x="432", y="511", width="15", height="15", rx="2",
        fill=COLORS["target_fill"], stroke=COLORS["target"], stroke_width="1.8")
    text(svg, 456, 525, "Target", size=17, fill=COLORS["muted"], weight=500)

    add(svg, "line", x1="53", y1="558", x2="98", y2="558",
        stroke=COLORS["bridge"], stroke_width="3", stroke_dasharray="8 5")
    text(svg, 109, 565, "Adaptive read support", size=16, fill=COLORS["muted"], weight=500)
    add(svg, "line", x1="407", y1="558", x2="452", y2="558",
        stroke=COLORS["anchor"], stroke_width="2.3", stroke_dasharray="6 4")
    text(svg, 463, 565, "Selected Anchor patch", size=16, fill=COLORS["muted"], weight=500)

    add(svg, "rect", x="52", y="590", width="616", height="82", rx="10",
        fill="#F8F5FA", stroke="#E1D4E8", stroke_width="1.4")
    text(svg, 69, 613, "Anchor reference selection", size=18, weight=700)
    text(svg, 69, 638,
         f"T: x[{target.x0},{target.x1}) y[{target.y0},{target.y1})"
         f"   →   A patch: x[{patch.x0},{patch.x1}) y[{patch.y0},{patch.y1})",
         size=16, weight=600)
    text(svg, 69, 660,
         "T above/right → Anchor top/right edge · same size as T · latent x→, y↓",
         size=14, fill=COLORS["muted"], weight=500)

    path = HERE / f"adaptive_local_4k_k{k}.svg"
    ET.indent(svg, space="  ")
    ET.ElementTree(svg).write(path, encoding="utf-8", xml_declaration=True)
    return path


if __name__ == "__main__":
    for partition in (4, 5):
        print(build(partition))
