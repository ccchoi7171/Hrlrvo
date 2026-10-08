"""CPU-only assembly and decoded-frame verification for the two requested runs."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def all_bodies_ready() -> bool:
    for i in range(14):
        status = ROOT / "experiment_1/bodies" / f"I{i:02d}" / "status.json"
        if not status.is_file() or json.loads(status.read_text()).get("status") != "PASS":
            return False
    return True


def verify_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    video = json.loads(result.stdout)["streams"][0]
    if (video["width"], video["height"], video["avg_frame_rate"], int(video["nb_read_frames"])) != (1920, 1080, "24/1", 1025):
        raise RuntimeError(f"Video contract failed for {path}: {video}")
    # Decode the entire file with errors promoted to failures.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        check=True,
    )
    return {"path": str(path), "sha256": digest(path), **video}


def main() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    state_path = ROOT / "validation/finalization_status.json"
    state = {"status": "WAITING_FOR_FULL_RUNS", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(state_path, state)
    assembled = ROOT / "experiment_1/generated.mp4"
    manifest = ROOT / "experiment_1/assembly_manifest.json"
    exp2_status = ROOT / "experiment_2/output/job_status.json"
    while True:
        if not manifest.is_file() and all_bodies_ready():
            state["status"] = "ASSEMBLING_EXPERIMENT_1"
            atomic_json(state_path, state)
            subprocess.run([PYTHON, str(ROOT / "experiment_1/assemble.py")], check=True, env=env)
        if manifest.is_file() and exp2_status.is_file():
            second = json.loads(exp2_status.read_text())
            if second.get("experiment_2_contract", {}).get("status") == "PASS_FULL1025_PHASE_ALIGNED_TEMPORAL_REBASE":
                break
        time.sleep(20)

    state["status"] = "VERIFYING_FULL_OUTPUTS"
    atomic_json(state_path, state)
    videos = [verify_video(assembled), verify_video(ROOT / "experiment_2/output/generated.mp4")]
    protected = json.loads((ROOT / "validation/protected_sources_start.json").read_text())
    before = json.loads(Path(protected["before_manifest"]).read_text())
    changed = [path for path, expected in before.items() if digest(Path(path)) != expected]
    if changed:
        raise RuntimeError(f"Protected source files changed: {changed}")
    destination = ROOT / "deliverables"
    destination.mkdir(exist_ok=True)
    for index, video in enumerate(videos, start=1):
        output = destination / f"experiment_{index}_1025f_1080p_24fps.mp4"
        if not output.exists():
            os.link(video["path"], output)
        if digest(output) != video["sha256"]:
            raise RuntimeError(f"Deliverable link differs: {output}")
        video["deliverable"] = str(output)
    state.update(status="PASS_BOTH_FULL_1025F", ended_utc=datetime.now(timezone.utc).isoformat(), videos=videos,
                 protected_source_files_checked=len(before), protected_source_files_changed=[])
    atomic_json(state_path, state)
    atomic_json(destination / "manifest.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
