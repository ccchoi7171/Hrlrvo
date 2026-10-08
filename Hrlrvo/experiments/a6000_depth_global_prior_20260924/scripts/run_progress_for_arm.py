"""Run one CPU-only cumulative progress watcher for a pinned A6000 arm."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from watch_progress_videos import load_committed_records, run_watcher


OLD_ROOT = Path("/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924")
DEPTH_ROOT = Path("/home/oem/hrlrvo/experiments/a6000_depth_global_prior_20260924")
DEPTH_MODE = "strength_1p0_global_depth_prior"


@dataclass(frozen=True)
class ArmSpec:
    name: str
    root: Path
    mode: str
    gpu: str
    runner: Path
    required_environment: tuple[tuple[str, str], ...]
    require_c0_before_watcher: bool = False


def arm_spec(name: str) -> ArmSpec:
    if name == "depth":
        return ArmSpec(
            name=name,
            root=DEPTH_ROOT,
            mode=DEPTH_MODE,
            gpu="0",
            runner=DEPTH_ROOT / "run_depth_chain.py",
            required_environment=(("HRLRVO_EXPERIMENT", "a6000_depth_global_prior_20260924"),),
            require_c0_before_watcher=True,
        )
    gpu_by_arm = {"02_no_carry": "1", "03_global_boost": "2", "04_native_halo": "3"}
    if name not in gpu_by_arm:
        raise ValueError(f"Unknown arm: {name}")
    return ArmSpec(
        name=name,
        root=OLD_ROOT / "arms" / name,
        mode="",
        gpu=gpu_by_arm[name],
        runner=OLD_ROOT / "run_arm.py",
        required_environment=(("HRLRVO_ARM", name),),
    )


def _proc_environment(path: Path) -> dict[str, str]:
    entries = path.read_bytes().split(b"\0")
    environment: dict[str, str] = {}
    for entry in entries:
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        environment[key.decode("utf-8", "strict")] = value.decode("utf-8", "strict")
    return environment


def validate_producer(pid: int, spec: ArmSpec, *, proc_root: Path = Path("/proc")) -> None:
    """Fail closed unless PID is the exact expected producer command and environment."""
    if pid <= 0:
        raise RuntimeError("Producer PID must be positive")
    process = proc_root / str(pid)
    stat = (process / "stat").read_text(encoding="utf-8").split()
    if len(stat) < 3 or stat[2] == "Z":
        raise RuntimeError(f"Producer {pid} is not alive")
    command = [
        value.decode("utf-8", "strict")
        for value in (process / "cmdline").read_bytes().split(b"\0")
        if value
    ]
    cwd = (process / "cwd").resolve(strict=True)
    candidates = []
    for argument in command:
        if Path(argument).name != spec.runner.name:
            continue
        path = Path(argument)
        candidates.append((path if path.is_absolute() else cwd / path).resolve(strict=False))
    if spec.runner.resolve(strict=False) not in candidates:
        raise RuntimeError(
            f"PID {pid} command is not the expected producer {spec.runner}: {command!r}"
        )
    if spec.name == "depth":
        try:
            mode_index = command.index("--mode")
        except ValueError as error:
            raise RuntimeError("Depth producer command has no --mode") from error
        if mode_index + 1 >= len(command) or command[mode_index + 1] != spec.mode:
            raise RuntimeError("Depth producer command has the wrong mode")
    environment = _proc_environment(process / "environ")
    expected_environment = (("CUDA_VISIBLE_DEVICES", spec.gpu), *spec.required_environment)
    for key, expected in expected_environment:
        if environment.get(key) != expected:
            raise RuntimeError(
                f"PID {pid} environment {key}={environment.get(key)!r}, expected {expected!r}"
            )


def _read_status(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        raise RuntimeError(f"Status is not a JSON object: {path}")
    return value


def wait_for_ready(spec: ArmSpec, *, poll_seconds: float = 10.0) -> int:
    """Wait for an owned live producer, or accept a fully committed completed run."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    status_path = spec.root / "status.json"
    while True:
        status = _read_status(status_path)
        if status is None:
            time.sleep(poll_seconds)
            continue
        state = status.get("state")
        if state == "complete":
            records, missing, geometry = load_committed_records(
                spec.root / "chunks" / spec.mode, 23
            )
            if records is None or geometry is None:
                raise RuntimeError(
                    f"Completed {spec.name} status is missing committed chunk C{missing}"
                )
            return os.getpid()
        if state != "running":
            raise RuntimeError(f"Producer for {spec.name} has terminal/invalid state {state!r}")
        pid = status.get("pid")
        if not isinstance(pid, int):
            raise RuntimeError(f"Running {spec.name} status has no integer PID")
        try:
            validate_producer(pid, spec)
        except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
            raise RuntimeError(f"Cannot authenticate running producer {pid}") from error
        if spec.require_c0_before_watcher:
            segment_root = spec.root / "chunks" / spec.mode
            if not (segment_root / "chunk_000000.json").is_file():
                time.sleep(poll_seconds)
                continue
            records, _missing, geometry = load_committed_records(segment_root, 0)
            if records is None or geometry is None:
                time.sleep(poll_seconds)
                continue
        return pid


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("02_no_carry", "03_global_boost", "04_native_halo", "depth"),
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' exactly; progress encoding is CPU-only")
    spec = arm_spec(args.arm)
    logs = spec.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "progress.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Progress watcher already holds the {spec.name} lock") from error
        producer_pid = wait_for_ready(spec, poll_seconds=args.poll_seconds)
        run_watcher(
            root=spec.root,
            mode=spec.mode,
            end_chunk=23,
            producer_pid=producer_pid,
            poll_seconds=args.poll_seconds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
