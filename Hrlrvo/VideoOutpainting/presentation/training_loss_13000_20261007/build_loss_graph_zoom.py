"""Zoom the original loss chart without changing data or smoothing."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parent
csv_path = ROOT / "training_loss_13000.csv"
provenance = json.loads((ROOT / "provenance.json").read_text())
assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == provenance["csv_sha256"]
with csv_path.open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
steps = np.array([int(row["step"]) for row in rows])
loss = np.array([float(row["loss"]) for row in rows])
assert np.array_equal(steps, np.arange(20, 13_001, 20))
assert np.isfinite(loss).all()
window = 50
trend = np.convolve(loss, np.ones(window) / window, mode="valid")
trend_steps = steps[window - 1:]
y_limits = (0.14, 0.32)
assert y_limits[0] < trend.min() < trend.max() < y_limits[1]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 22, "axes.labelsize": 15,
    "axes.edgecolor": "#8b95a5", "axes.linewidth": 0.8,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(11.5, 6.3), layout="constrained")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.plot(steps, loss, color="#9aafca", alpha=0.22, linewidth=0.8,
        label="Recorded loss (every 20 steps)", zorder=1)
ax.plot(trend_steps, trend, color="#1464b4", linewidth=2.8,
        label="1,000-step moving average", zorder=3)
ax.scatter([trend_steps[0], trend_steps[-1]], [trend[0], trend[-1]],
           color="#1464b4", s=32, zorder=4, clip_on=False)
ax.annotate(f"First 1,000 steps\n{trend[0]:.4f}",
            xy=(trend_steps[0], trend[0]), xytext=(700, 0.292),
            fontsize=11, color="#1464b4", ha="left",
            arrowprops={"arrowstyle": "-", "color": "#1464b4", "linewidth": 1},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92})
ax.annotate(f"Last 1,000 steps\n{trend[-1]:.4f}",
            xy=(trend_steps[-1], trend[-1]), xytext=(10_600, 0.150),
            fontsize=11, color="#1464b4", ha="left",
            arrowprops={"arrowstyle": "-", "color": "#1464b4", "linewidth": 1},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92})
ax.set_title("Hrlrvo Training Loss", loc="left", fontweight="bold", pad=18)
ax.text(1, 1.04, "Trend view | 13,000 optimizer steps", transform=ax.transAxes,
        ha="right", color="#526076", fontsize=12)
ax.set_xlabel("Optimizer step", labelpad=10)
ax.set_ylabel("Masked flow-matching MSE", labelpad=10)
ax.set_xlim(0, 13_000)
ax.set_ylim(*y_limits)
ax.set_xticks([0, 2_000, 4_000, 6_000, 8_000, 10_000, 12_000, 13_000])
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
ax.grid(axis="y", color="#e7ecf2", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="both", length=0, pad=8)
ax.legend(loc="upper right", frameon=False, fontsize=11)
ax.text(0, -0.19,
        "Y-axis zoom: 0.14-0.32. Same data and moving average; raw spikes outside this range are clipped.",
        transform=ax.transAxes, fontsize=9, color="#657389")
for extension in ("png", "svg", "pdf"):
    fig.savefig(ROOT / f"training_loss_13000_zoom.{extension}", dpi=300, facecolor="white")
plt.close(fig)
summary = {
    "csv_sha256": provenance["csv_sha256"], "recorded_observations": len(rows),
    "log_interval_steps": 20, "moving_average_observations": window,
    "moving_average_step_window": 1_000, "y_limits": list(y_limits),
    "first_1000_steps_mean": float(trend[0]),
    "last_1000_steps_mean": float(trend[-1]),
    "first_vs_last_relative_decrease_percent": float((1 - trend[-1] / trend[0]) * 100),
    "data_or_smoothing_modified": False,
}
(ROOT / "zoom_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
