"""Flexible metric normalization, health evaluation, and figure builders."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd
import plotly.graph_objects as go

ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "logged_at", "created_at"),
    "run_id": ("run_id", "run", "run_name"),
    "optimizer_step": ("optimizer_step", "global_step", "step"),
    "batches_seen": ("batches_seen", "batch_count", "batches"),
    "samples_seen": ("samples_seen", "sample_count", "samples"),
    "loss": ("loss", "train_loss", "training_loss"),
    "learning_rate": ("learning_rate", "lr", "optimizer_lr"),
    "grad_norm_global": ("grad_norm_global", "global_grad_norm", "grad_global_norm"),
    "grad_norm_op_lora": ("grad_norm_op_lora", "op_lora_grad_norm"),
    "grad_norm_a_lora": ("grad_norm_a_lora", "a_lora_grad_norm"),
    "grad_norm_g_lora": ("grad_norm_g_lora", "g_lora_grad_norm"),
    "grad_norm_f_lora": ("grad_norm_f_lora", "f_lora_grad_norm"),
    "grad_norm_c2": ("grad_norm_c2", "c2_grad_norm"),
    "grad_norm_c4_shared": ("grad_norm_c4_shared", "shared_c4_grad_norm", "grad_norm_shared_c4"),
    "grad_norm_r2": ("grad_norm_r2", "r2_grad_norm"),
    "grad_norm_r4": ("grad_norm_r4", "r4_grad_norm"),
    "grad_norm_local_fusion": ("grad_norm_local_fusion", "local_fusion_grad_norm"),
    "grad_norm_global_fusion": ("grad_norm_global_fusion", "global_fusion_grad_norm"),
    "local_native_ratio_mean": ("local_native_ratio_mean", "native_ratio_mean"),
    "local_native_ratio_min": ("local_native_ratio_min", "native_ratio_min"),
    "local_native_ratio_max": ("local_native_ratio_max", "native_ratio_max"),
    "local_adaptive_ratio_mean": ("local_adaptive_ratio_mean", "adaptive_ratio_mean"),
    "local_adaptive_ratio_min": ("local_adaptive_ratio_min", "adaptive_ratio_min"),
    "local_adaptive_ratio_max": ("local_adaptive_ratio_max", "adaptive_ratio_max"),
    "global_local_ratio_mean": ("global_local_ratio_mean", "local_ratio_mean"),
    "global_local_ratio_min": ("global_local_ratio_min", "local_ratio_min"),
    "global_local_ratio_max": ("global_local_ratio_max", "local_ratio_max"),
    "global_global_ratio_mean": ("global_global_ratio_mean", "global_ratio_mean"),
    "global_global_ratio_min": ("global_global_ratio_min", "global_ratio_min"),
    "global_global_ratio_max": ("global_global_ratio_max", "global_ratio_max"),
    "local_native_ratio_by_block": ("local_native_ratio_by_block", "native_ratio_by_block"),
    "local_adaptive_ratio_by_block": ("local_adaptive_ratio_by_block", "adaptive_ratio_by_block"),
    "global_local_ratio_by_block": ("global_local_ratio_by_block", "local_ratio_by_block"),
    "global_global_ratio_by_block": ("global_global_ratio_by_block", "global_ratio_by_block"),
    "nan_gradient_count": ("nan_gradient_count", "nan_grad_count"),
    "inf_gradient_count": ("inf_gradient_count", "inf_grad_count"),
    "zero_gradient_trainable_groups": ("zero_gradient_trainable_groups", "zero_grad_groups"),
    "frozen_parameter_grad_violation_count": ("frozen_parameter_grad_violation_count", "frozen_grad_violation_count"),
    "gradient_clipped": ("gradient_clipped", "grad_clipped"),
    "grad_norm_before_clip": ("grad_norm_before_clip", "gradient_norm_before_clip"),
    "grad_norm_after_clip": ("grad_norm_after_clip", "gradient_norm_after_clip"),
}

GRADIENT_GROUPS = {
    "OP-LoRA": "grad_norm_op_lora",
    "A-LoRA": "grad_norm_a_lora",
    "G-LoRA": "grad_norm_g_lora",
    "F-LoRA": "grad_norm_f_lora",
    "C2": "grad_norm_c2",
    "Shared C4": "grad_norm_c4_shared",
    "R2": "grad_norm_r2",
    "R4": "grad_norm_r4",
    "Local Fusion": "grad_norm_local_fusion",
    "Global Fusion": "grad_norm_global_fusion",
}

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#dbe4f0", size=12),
    margin=dict(l=48, r=20, t=48, b=42),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat = dict(record)
    for parent, value in record.items():
        if isinstance(value, dict):
            for key, child in value.items():
                flat.setdefault(f"{parent}_{key}", child)
                flat.setdefault(key, child)
    return flat


def canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_record(record)
    output = {}
    for canonical, aliases in ALIASES.items():
        output[canonical] = next((flat[name] for name in aliases if name in flat), None)
    return output


def canonical_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [canonicalize_record(record) for record in records]


def scalar_frame(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = canonical_records(records)
    if not rows:
        return pd.DataFrame(columns=list(ALIASES))
    scalar_rows = []
    for row in rows:
        scalar_rows.append({key: (None if isinstance(value, (list, dict)) else value) for key, value in row.items()})
    frame = pd.DataFrame(scalar_rows)
    numeric = [key for key in frame.columns if key not in {"timestamp", "run_id", "gradient_clipped", "zero_gradient_trainable_groups"}]
    for key in numeric:
        frame[key] = pd.to_numeric(frame[key], errors="coerce")
    return frame


def display_value(value: Any, *, precision: int = 5) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value).upper()
        if value == 0:
            return "0"
        if abs(value) < 1e-4 or abs(value) >= 1e5:
            return f"{value:.3e}"
        return f"{value:.{precision}g}"
    return str(value)


def empty_figure(title: str, message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, **DARK_LAYOUT)
    fig.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper", showarrow=False, font=dict(size=18, color="#94a3b8"))
    return fig


def line_figure(frame: pd.DataFrame, fields: dict[str, str], title: str, *, y_range=None, log_y=False) -> go.Figure:
    if frame.empty or "optimizer_step" not in frame:
        return empty_figure(title)
    fig = go.Figure()
    for label, field in fields.items():
        if field in frame and frame[field].notna().any():
            fig.add_trace(go.Scatter(x=frame["optimizer_step"], y=frame[field], mode="lines", name=label, connectgaps=False))
    if not fig.data:
        return empty_figure(title, "Metric unavailable")
    fig.update_layout(title=title, xaxis_title="Optimizer step", yaxis_title="Value", **DARK_LAYOUT)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def loss_figure(frame: pd.DataFrame, ema_alpha: float = 0.1) -> go.Figure:
    if frame.empty or "loss" not in frame or not frame["loss"].notna().any():
        return empty_figure("Loss", "Loss unavailable")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["optimizer_step"], y=frame["loss"], mode="lines", name="Raw loss", line=dict(color="#38bdf8", width=1)))
    ema = frame["loss"].ewm(alpha=ema_alpha, adjust=False, ignore_na=True).mean()
    fig.add_trace(go.Scatter(x=frame["optimizer_step"], y=ema, mode="lines", name=f"EMA α={ema_alpha}", line=dict(color="#f59e0b", width=2.5)))
    fig.update_layout(title="Loss", xaxis_title="Optimizer step", yaxis_title="Loss", **DARK_LAYOUT)
    return fig


def gradient_bar(latest: dict[str, Any]) -> go.Figure:
    labels, values = [], []
    for label, field in GRADIENT_GROUPS.items():
        value = latest.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            labels.append(label); values.append(float(value))
    if not values:
        return empty_figure("Gradient Magnitude", "Gradient metrics unavailable")
    maximum = max(values) or 1.0
    fig = go.Figure(go.Bar(x=[v / maximum for v in values], y=labels, orientation="h", marker_color="#22d3ee", customdata=values, hovertemplate="%{y}<br>normalized=%{x:.3f}<br>raw=%{customdata:.6g}<extra></extra>"))
    fig.update_layout(title="Gradient Magnitude (latest, normalized to max)", xaxis_title="Relative magnitude", yaxis=dict(autorange="reversed"), **DARK_LAYOUT)
    return fig


def block_figure(record: dict[str, Any] | None, first: tuple[str, str], second: tuple[str, str], title: str) -> go.Figure:
    if not record:
        return empty_figure(title)
    canonical = canonicalize_record(record)
    a, b = canonical.get(first[1]), canonical.get(second[1])
    if not isinstance(a, list) or not isinstance(b, list):
        return empty_figure(title, "48-block metric unavailable")
    if len(a) != 48 or len(b) != 48:
        return empty_figure(title, f"Expected 48 blocks, got {len(a)} / {len(b)}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(48)), y=a, mode="lines+markers", name=first[0], line=dict(color="#38bdf8")))
    fig.add_trace(go.Scatter(x=list(range(48)), y=b, mode="lines+markers", name=second[0], line=dict(color="#f59e0b")))
    fig.update_layout(title=title, xaxis_title="DiT block", yaxis_title="Fusion ratio", **DARK_LAYOUT)
    fig.update_yaxes(range=[0, 1])
    return fig


def block_records(records: Iterable[dict[str, Any]]) -> list[tuple[Any, dict[str, Any]]]:
    result=[]
    for raw in records:
        c=canonicalize_record(raw)
        arrays=(c.get("local_native_ratio_by_block"), c.get("local_adaptive_ratio_by_block"), c.get("global_local_ratio_by_block"), c.get("global_global_ratio_by_block"))
        if any(isinstance(value,list) and len(value)==48 for value in arrays):
            result.append((c.get("optimizer_step"),raw))
    return result


def latest_record(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    raw=list(records)
    return canonicalize_record(raw[-1]) if raw else {}


def status_from(latest: dict[str, Any], file_mtime: float | None, now: float | None = None) -> tuple[str, str, float | None]:
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    candidates=[]
    if file_mtime is not None: candidates.append(file_mtime)
    timestamp=latest.get("timestamp")
    if isinstance(timestamp,str):
        try:
            parsed=datetime.fromisoformat(timestamp.replace("Z","+00:00"))
            if parsed.tzinfo is None: parsed=parsed.replace(tzinfo=timezone.utc)
            candidates.append(parsed.timestamp())
        except ValueError: pass
    if not candidates:
        return "ERROR", "status-critical", None
    age=max(0.0,now-max(candidates))
    if age <= 10: return "RUNNING", "status-ok", age
    if age <= 60: return "STALE", "status-warning", age
    return "STOPPED / NO UPDATE", "status-muted", age


def health_summary(latest: dict[str, Any], missing_fields: list[str]) -> dict[str, Any]:
    nan_count=latest.get("nan_gradient_count")
    inf_count=latest.get("inf_gradient_count")
    frozen=latest.get("frozen_parameter_grad_violation_count")
    critical=[]
    for label,value in (("NaN gradients",nan_count),("Inf gradients",inf_count),("Frozen gradient violations",frozen)):
        try:
            if value is not None and float(value)>0: critical.append(f"{label}: {value}")
        except (TypeError,ValueError):
            critical.append(f"{label}: invalid value {value!r}")
    zero=latest.get("zero_gradient_trainable_groups")
    warnings=[]
    if isinstance(zero,list) and zero: warnings.append("Zero-gradient groups: "+", ".join(map(str,zero)))
    if missing_fields: warnings.append(f"Missing latest metrics ({len(missing_fields)}): "+", ".join(missing_fields[:8]))
    level="critical" if critical else "warning" if warnings else "healthy"
    return {"level":level,"critical":critical,"warnings":warnings,"nan":nan_count,"inf":inf_count,"frozen":frozen,"zero_groups":zero if isinstance(zero,list) else None}


def sustained_zero_gradient_groups(frame: pd.DataFrame, window: int = 10) -> list[str]:
    """Identify logged trainable groups whose last ``window`` values are all zero."""
    if frame.empty or window < 1:
        return []
    result=[]
    for label,field in GRADIENT_GROUPS.items():
        if field not in frame:
            continue
        values=frame[field].dropna().tail(window)
        if len(values)==window and bool((values.abs() <= 1e-15).all()):
            result.append(label)
    return result
