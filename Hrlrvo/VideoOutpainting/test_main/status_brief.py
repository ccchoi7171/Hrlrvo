"""Read-only compact status for the two long-running experiments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
state = json.loads((ROOT / "scheduler_status.json").read_text())
summary = {
    "utc": state["checked_utc"],
    "gpu_mib": state["gpu_used_mib"],
    "running": [{"name": name, "pid": task["pid"], "mib": task.get("measured_mib")}
                for name, task in state["tasks"].items() if task["status"] == "RUNNING"],
    "failed": [name for name, task in state["tasks"].items() if task["status"] == "FAILED"],
    "bridges": {},
    "bodies": {},
    "experiment_2": {"frames": 0, "chunks": 0},
}
for kind, folder in (("bridges", "bridge_bank"), ("bodies", "bodies")):
    for status in sorted((ROOT / "experiment_1" / folder).glob("*/status.json")):
        item = json.loads(status.read_text())
        summary[kind][status.parent.name] = item["status"]
lineage = ROOT / "experiment_2/output/phase_emission_lineage.jsonl"
if lineage.exists():
    rows = [json.loads(row) for row in lineage.read_text().splitlines()]
    if rows:
        summary["experiment_2"].update(frames=rows[-1]["new_output_frame_range"][1], chunks=len(rows))
for task in summary["running"]:
    log = Path(state["tasks"][task["name"]]["log"])
    # Avoid reading the large model report on every status check.
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 8192))
        tail = stream.read().decode(errors="replace").splitlines()
    for line in reversed(tail):
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if event.get("event") in ("denoising_progress", "body_denoising_progress", "body_noise_captured", "bridge_noise_captured", "chunk_completed"):
            task["latest_event"] = event
            break
final = ROOT / "validation/finalization_status.json"
if final.exists():
    summary["finalization"] = json.loads(final.read_text())["status"]
print(json.dumps(summary))
