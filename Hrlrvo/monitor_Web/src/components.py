"""Reusable Dash component helpers."""
from __future__ import annotations

from typing import Any
from dash import html
from .metrics import GRADIENT_GROUPS, display_value


def status_card(label: str, component_id: str, *, card_id: str | None = None):
    kwargs = {"className": "status-card"}
    if card_id is not None:
        kwargs["id"] = card_id
    return html.Div(
        [html.Div(label, className="status-label"), html.Div("N/A", id=component_id, className="status-value")],
        **kwargs,
    )


def section(title: str, children: Any, *, class_name: str = "panel"):
    return html.Section([html.H2(title), children], className=class_name)


def gradient_table(latest: dict[str, Any]):
    header = html.Tr([html.Th("Module"), html.Th("Grad Norm")])
    rows = [html.Tr([html.Td(label), html.Td(display_value(latest.get(field)))]) for label, field in GRADIENT_GROUPS.items()]
    return html.Table([html.Thead(header), html.Tbody(rows)], className="metric-table")


def ratio_table(latest: dict[str, Any], groups: list[tuple[str, str]]):
    header=html.Tr([html.Th("Contribution"),html.Th("Mean"),html.Th("Min"),html.Th("Max")])
    rows=[]
    for label,prefix in groups:
        rows.append(html.Tr([html.Td(label),html.Td(display_value(latest.get(prefix+"_mean"))),html.Td(display_value(latest.get(prefix+"_min"))),html.Td(display_value(latest.get(prefix+"_max")))]))
    return html.Table([html.Thead(header),html.Tbody(rows)],className="metric-table compact")


def health_panel(summary: dict[str, Any], latest: dict[str, Any]):
    items=[]
    if summary["critical"]:
        items.extend(html.Div(text,className="health-critical") for text in summary["critical"])
    if summary["warnings"]:
        items.extend(html.Div(text,className="health-warning") for text in summary["warnings"])
    if not items:
        items.append(html.Div("No gradient-health alert in the latest record",className="health-ok"))
    clipping=latest.get("gradient_clipped")
    items.append(html.Div([
        html.Span(f"Gradient clipped: {display_value(clipping)}"),
        html.Span(f"Before: {display_value(latest.get('grad_norm_before_clip'))}"),
        html.Span(f"After: {display_value(latest.get('grad_norm_after_clip'))}"),
        html.Span(f"Global: {display_value(latest.get('grad_norm_global'))}"),
    ],className="health-stats"))
    return items
