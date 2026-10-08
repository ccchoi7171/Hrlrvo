"""Extract the real training log and export one presentation-ready loss graph."""

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent.parent
LIMIT = 13_000
INTERVAL = 20
WINDOW = 50  # 50 recorded observations, covering 1,000 optimizer steps.
CSV = OUT / "training_loss_13000.csv"

if not CSV.exists():
    capture = subprocess.check_output(
        ["tmux", "capture-pane", "-p", "-J", "-S", "-", "-t", "hrlrvo_train:1.1"],
        text=True,
    )
    assert "hrlrvo_main_train_v1" in capture, "Unexpected training run"
    # Rich may wrap Time/Step onto a separate line; loss and LR remain together.
    pattern = re.compile(
        r"Step\s+(\d+)\s*/\s*(\d+)\s*-\s*Loss:\s*([-+\d.eE]+),\s*LR:\s*([-+\d.eE]+)"
    )
    records = []
    for match in pattern.finditer(capture):
        step, total = int(match[1]), int(match[2])
        if total == 30_000 and 0 < step <= LIMIT:
            records.append((step, float(match[3]), float(match[4]), match.group(0)))
    # The later occurrence follows the last resume used to reach checkpoint 013000.
    selected = {row[0]: row for row in records}
    assert set(selected) == set(range(INTERVAL, LIMIT + 1, INTERVAL))
    with CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "loss", "learning_rate"])
        writer.writerows((s, f"{loss:.4f}", f"{lr:.2e}") for s, loss, lr, _ in sorted(selected.values()))
    (OUT / "source_loss_messages.txt").write_text(
        "\n".join(row[3] for row in records) + "\n", encoding="utf-8"
    )
    manifest_path = PROJECT / "runs/hrlrvo_main_train_v1/checkpoints/checkpoint-013000/outpainting_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["global_step"] == LIMIT
    summary = {
        "run": "hrlrvo_main_train_v1",
        "source": "hrlrvo_train:1.1 tmux scrollback; native trainer INFO messages",
        "checkpoint_manifest": str(manifest_path),
        "optimizer_step_limit": LIMIT,
        "first_recorded_step": INTERVAL,
        "log_interval_steps": INTERVAL,
        "number_of_observations": len(selected),
        "number_of_original_records": len(records),
        "duplicate_policy": "Keep last occurrence at each optimizer step after resume",
        "duplicated_steps": sorted(s for s in selected if sum(row[0] == s for row in records) > 1),
        "missing_expected_observations": [],
        "loss_precision": "4 decimal places, as printed by the native trainer",
        "smoothing": "Trailing arithmetic mean of 50 observations (1,000-step window); full windows only",
        "csv_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
    }
    (OUT / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")

with CSV.open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
steps = np.array([int(row["step"]) for row in rows])
loss = np.array([float(row["loss"]) for row in rows])
assert np.array_equal(steps, np.arange(INTERVAL, LIMIT + 1, INTERVAL))
assert np.isfinite(loss).all() and (loss >= 0).all()
smoothed = np.convolve(loss, np.ones(WINDOW) / WINDOW, mode="valid")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 22, "axes.labelsize": 15,
    "axes.edgecolor": "#8b95a5", "axes.linewidth": 0.8,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(11.5, 6.3), layout="constrained")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.plot(steps, loss, color="#9aafca", alpha=0.65, linewidth=1.0,
        label="Recorded loss (every 20 steps)", zorder=2)
ax.plot(steps[WINDOW - 1:], smoothed, color="#1464b4", linewidth=2.8,
        label="1,000-step moving average", zorder=3)
ax.set_title("Hrlrvo Training Loss", loc="left", fontweight="bold", pad=18)
ax.text(1, 1.04, "Up to 13,000 optimizer steps", transform=ax.transAxes,
        ha="right", color="#526076", fontsize=12)
ax.set_xlabel("Optimizer step", labelpad=10)
ax.set_ylabel("Masked flow-matching MSE", labelpad=10)
ax.set_xlim(0, LIMIT)
ax.set_ylim(0, float(loss.max()) * 1.08)
ax.set_xticks([0, 2_000, 4_000, 6_000, 8_000, 10_000, 12_000, 13_000])
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
ax.grid(axis="y", color="#e7ecf2", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="both", length=0, pad=8)
ax.legend(loc="upper right", frameon=False, fontsize=12)
for extension in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"training_loss_13000.{extension}", dpi=300, facecolor="white")
plt.close(fig)
print(json.dumps({
    "observations": len(rows), "first_step": int(steps[0]), "last_step": int(steps[-1]),
    "loss_at_13000": float(loss[-1]), "loss_range": [float(loss.min()), float(loss.max())],
    "first_1000_step_mean": float(smoothed[0]), "last_1000_step_mean": float(smoothed[-1]),
    "png": str(OUT / "training_loss_13000.png"),
}, indent=2))
