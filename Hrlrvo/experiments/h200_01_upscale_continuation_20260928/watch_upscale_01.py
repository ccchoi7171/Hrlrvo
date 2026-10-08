#!/usr/bin/env python3
"""One-shot, fail-closed handoff from the already-running 01 outpaint to upscale.

This watcher does not change the production package or the producer process.
It exits after starting an independently sessioned upscale child; it never
retries an ambiguous or partially started upscale automatically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_start_ticks(pid: int) -> tuple[str, str] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return fields[19], fields[0]


def process_matches(pid: int, ticks: str) -> bool:
    current = process_start_ticks(pid)
    return current is not None and current[0] == ticks and current[1] != "Z"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Journal:
    def __init__(self, work: Path):
        self.work = work
        self.state = work / "watcher_state.json"
        self.log = work / "watcher.log"

    def emit(self, status: str, **fields: Any) -> None:
        record = {"time_utc": utc_now(), "status": status, "watcher_pid": os.getpid(), **fields}
        atomic_json(self.state, record)
        line = json.dumps(record, ensure_ascii=False)
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(line, flush=True)


def locks_free(root: Path) -> bool:
    for name in ("outpaint", "upscale", "video_01"):
        path = root / "state" / "locks" / f"{name}.lock"
        if not path.exists():
            continue
        with path.open("rb") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(stream, fcntl.LOCK_UN)
    return True


def validate_plan(plan: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(plan["production_root"]).resolve(strict=True)
    config_path = root / "configs/certification/01.json"
    predecessor = Path(plan["predecessor_state"]).resolve(strict=True)
    require(root == Path("/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K"), "Wrong production root")
    require(sha256(config_path) == plan["config_sha256"], "01 certification config changed")
    config = read_json(config_path)
    require(config["input"]["video_id"] == "01", "Wrong video ID")
    require(config["seed"] == 50374151, "Wrong 01 seed")
    require(config["frames"] == {"outpaint": "all", "upscale": "all"}, "Not a full-frame config")
    require(config["artifacts"]["upscale"] == "data/03_Upscale_16K_15360x8640/01", "Wrong 01 upscale target")
    require(config["input"]["path"] == "data/00_Input_8K_7680x4320/01_Input_8K.mp4", "Wrong 01 anchor")
    require(plan["expected_outpaint_frames"] == 1952, "Wrong planned frame count")
    require(
        plan["expected_final"] == f"{config['artifacts']['upscale']}/final/{config['final']['filename']}",
        "Wrong planned final path",
    )
    for key in ("outpaint_python", "flash_python"):
        path = Path(plan[key])
        require(path.is_file() and os.access(path, os.X_OK), f"Python unavailable: {path}")
    require((root / "run_cert.sh").is_file(), "Production entrypoint missing")
    return root, config_path, predecessor


def stage_sentinel(root: Path, config: dict[str, Any]) -> str:
    """Return EMPTY, COMPLETE or PARTIAL; never treat an MP4 alone as done."""
    stage = root / config["artifacts"]["upscale"]
    final = stage / "final" / config["final"]["filename"]
    manifest = stage / "metadata/final_manifest.json"
    meaningful = [
        path
        for sub in ("final", "metadata", "logs", "raw_flashvsr", "segments")
        for path in (stage / sub).rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    if not meaningful:
        return "EMPTY"
    if final.is_file() and manifest.is_file():
        record = read_json(manifest)
        if record.get("status") == "PASS" and record.get("video_id") == "01":
            info = record.get("output_file", {})
            video = record.get("output", {})
            if (Path(info.get("path", "")).resolve() == final.resolve()
                    and info.get("size") == final.stat().st_size
                    and info.get("mtime_ns") == final.stat().st_mtime_ns
                    and (video.get("width"), video.get("height"), video.get("frame_count"))
                    == (15360, 8640, 1952)
                    and video.get("avg_frame_rate") == "16/1"
                    and video.get("r_frame_rate") == "16/1"):
                return "COMPLETE"
    return "PARTIAL"


def validate_outpaint(root: Path, config_path: Path, expected_hash: str) -> Path:
    require(sha256(config_path) == expected_hash, "01 config changed while waiting")
    # Import is CPU-only: the model is loaded only inside run_outpaint_stage.
    sys.path.insert(0, str(root))
    production = importlib.import_module("run_cert")
    config = read_json(config_path)
    result, frames = production.validate_outpaint_artifact(config)
    require(frames == 1952, f"01 Outpainting frames={frames}, expected 1952")
    receipt = read_json(root / config["artifacts"]["outpaint"] / "metadata/stage_complete.json")
    require(receipt.get("status") == "PASS", "01 Outpainting receipt not PASS")
    require(receipt.get("output_fps") == 16, "01 Outpainting deliverable not 16fps")
    require(Path(receipt["output"]).resolve() == result.resolve(), "01 Outpainting receipt path mismatch")
    return result


def wait_for_predecessor(plan: dict[str, Any], predecessor: Path, journal: Journal) -> None:
    pid = int(plan["outpaint_pid"])
    ticks = str(plan["outpaint_start_ticks"])
    first_exited: float | None = None
    last_state: str | None = None
    while True:
        status = read_json(predecessor).get("status")
        if status != last_state:
            journal.emit("WAITING_FOR_01_OUTPAINT", producer_status=status, outpaint_pid=pid)
            last_state = status
        require(status != "BLOCKED_NO_DESTRUCTIVE_RECOVERY", "Original 01 producer failed")
        active = process_matches(pid, ticks)
        if not active and status == "01_OUTPAINT_COMPLETE":
            return
        if not active:
            if first_exited is None:
                first_exited = time.monotonic()
            require(time.monotonic() - first_exited < 3600, "Producer exited without PASS within 1 hour")
        time.sleep(20)


def run(plan_path: Path, *, preflight: bool) -> int:
    plan = read_json(plan_path)
    work = plan_path.parent
    journal = Journal(work)
    try:
        root, config_path, predecessor = validate_plan(plan)
    except Exception as error:
        if not preflight:
            journal.emit("FAILED", reason=f"Preflight: {type(error).__name__}: {error}")
        raise
    producer = read_json(predecessor)
    require(producer.get("child_pid") == plan["outpaint_pid"] or producer.get("status") == "01_OUTPAINT_COMPLETE", "Predecessor child PID mismatch")
    require(
        process_matches(int(plan["outpaint_pid"]), str(plan["outpaint_start_ticks"]))
        or producer.get("status") == "01_OUTPAINT_COMPLETE",
        "Original producer identity cannot be confirmed",
    )
    config = read_json(config_path)
    stage_status = stage_sentinel(root, config)
    if preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "producer": producer.get("status"), "stage": stage_status, "no_gpu_load": True}))
        return 0
    with (work / "watcher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GateError("Another 01 continuation watcher already owns the lock") from error
        if journal.state.exists():
            previous = read_json(journal.state).get("status")
            require(previous not in {"LAUNCH_INTENT", "UPSCALE_STARTED", "UPSCALE_RUNNING", "UPSCALE_COMPLETE", "ALREADY_COMPLETE"}, f"One-shot watcher already committed: {previous}")
            require(previous not in {"FAILED", "UPSCALE_FAILED", "REFUSED_PARTIAL"}, f"Prior watcher failure requires manual audit: {previous}")
        if stage_status == "COMPLETE":
            journal.emit("ALREADY_COMPLETE", final=str(root / config["artifacts"]["upscale"] / "final" / config["final"]["filename"]))
            return 0
        if stage_status != "EMPTY":
            journal.emit("REFUSED_PARTIAL", reason="01 upscale stage already has artifacts; no automatic duplicate/resume")
            return 1
        launched = False
        try:
            wait_for_predecessor(plan, predecessor, journal)
            artifact = validate_outpaint(root, config_path, plan["config_sha256"])
            journal.emit("OUTPAINT_VALIDATED", outpaint=str(artifact), frames=1952)
            while not locks_free(root):
                journal.emit("WAITING_FOR_STAGE_LOCKS")
                time.sleep(20)
            require(stage_sentinel(root, config) == "EMPTY", "01 upscale artifacts appeared before launch")
            command = [str(root / "run_cert.sh"), "--config", str(config_path), "--stage", "upscale"]
            journal.emit("LAUNCH_INTENT", command=command)
            env = dict(os.environ, CERT_OUTPAINT_PYTHON=plan["outpaint_python"],
                       CERT_FLASH_PYTHON=plan["flash_python"], PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUNBUFFERED="1")
            output_log = work / "01_upscale_process.log"
            with output_log.open("ab", buffering=0) as stream:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                launched = True
                ticks = process_start_ticks(child.pid)
                journal.emit("UPSCALE_STARTED", child_pid=child.pid,
                             child_start_ticks=ticks[0] if ticks else None,
                             command=command, child_log=str(output_log),
                             expected_final=str(root / config["artifacts"]["upscale"] / "final" / config["final"]["filename"]))
                time.sleep(3)
                code = child.poll()
                require(code is None, f"01 upscale exited immediately with code={code}; see {output_log}")
                journal.emit("UPSCALE_RUNNING", child_pid=child.pid,
                             child_start_ticks=ticks[0] if ticks else None,
                             child_log=str(output_log))
                # The detached watcher, not Codex, waits and records the outcome.
                code = child.wait()
                require(code == 0, f"01 upscale exited with code={code}; see {output_log}")
                require(stage_sentinel(root, config) == "COMPLETE",
                        "01 upscale exited 0 without a verified final")
                journal.emit("UPSCALE_COMPLETE", child_pid=child.pid,
                             final=str(root / config["artifacts"]["upscale"]
                                       / "final" / config["final"]["filename"]))
            return 0
        except Exception as error:
            journal.emit("UPSCALE_FAILED" if launched else "FAILED",
                         reason=f"{type(error).__name__}: {error}")
            return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    return run(args.plan.resolve(strict=True), preflight=args.preflight)


if __name__ == "__main__":
    raise SystemExit(main())
