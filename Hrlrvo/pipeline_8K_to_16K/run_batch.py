#!/usr/bin/env python3
"""FIFO outpaint/upscale queue for the certification pipeline.

The queue is orchestration only. Every model stage is executed through the
single ``run_cert.sh --stage ...`` entrypoint in a separate OS process.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping


ALLOWED_STATES = {
    "PENDING",
    "PREPROCESS_READY",
    "OUTPAINT_RUNNING",
    "OUTPAINT_DONE",
    "UPSCALE_WAITING",
    "UPSCALE_RUNNING",
    "DONE",
    "FAILED_OUTPAINT",
    "FAILED_UPSCALE",
}


@dataclass(frozen=True)
class Job:
    video_id: str
    job_name: str
    config_path: Path


@dataclass(frozen=True)
class BatchResult:
    states: dict[str, str]
    interrupted: bool

    @property
    def succeeded(self) -> bool:
        return not self.interrupted and all(value == "DONE" for value in self.states.values())


ChildRunner = Callable[[str, Job, bool, Callable[[str], None]], int]
SAFE_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


class BatchJournal:
    def __init__(self, state_path: Path, log_path: Path, initial_states: Mapping[str, str]) -> None:
        self.state_path = state_path
        self.log_path = log_path
        self._states = dict(initial_states)
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.state_path, self._states)

    def log(self, message: str) -> None:
        line = f"{_utc_timestamp()} {message}"
        with self._lock:
            with self.log_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    def transition(self, video_id: str, state: str, message: str) -> None:
        if state not in ALLOWED_STATES:
            raise ValueError(f"unsupported queue state: {state}")
        with self._lock:
            self._states[video_id] = state
            _atomic_write_json(self.state_path, self._states)
            line = f"{_utc_timestamp()} {message}"
            with self.log_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._states)


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_within_root(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside pipeline root: {path}") from exc


def _resolve_config_path(root: Path, batch_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    root_candidate = (root / candidate).resolve()
    if root_candidate.exists():
        return root_candidate
    return (batch_path.parent / candidate).resolve()


def load_jobs(root: Path, batch_path: Path) -> list[Job]:
    _require_within_root(batch_path.resolve(), root.resolve(), "batch config")
    batch = _load_json(batch_path)
    if batch.get("outpaint_workers") != 1:
        raise ValueError("outpaint_workers must be exactly 1")
    if batch.get("upscale_workers") != 1:
        raise ValueError("upscale_workers must be exactly 1")
    if batch.get("queue_policy") != "fifo":
        raise ValueError("queue_policy must be 'fifo'")
    raw_jobs = batch.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("jobs must be a non-empty list of config paths")

    jobs: list[Job] = []
    seen: set[str] = set()
    for raw_path in raw_jobs:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("each jobs entry must be a non-empty string")
        config_path = _resolve_config_path(root, batch_path, raw_path)
        _require_within_root(config_path, root, "job config")
        config = _load_json(config_path)
        input_config = config.get("input")
        video_id = input_config.get("video_id") if isinstance(input_config, dict) else None
        job_name = config.get("job_name")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"input.video_id is required: {config_path}")
        if not SAFE_VIDEO_ID.fullmatch(video_id):
            raise ValueError(f"input.video_id is not a safe state label: {video_id!r}")
        if not isinstance(job_name, str) or not job_name:
            raise ValueError(f"job_name is required: {config_path}")
        if video_id in seen:
            raise ValueError(f"duplicate video_id in batch: {video_id}")
        seen.add(video_id)
        jobs.append(Job(video_id=video_id, job_name=job_name, config_path=config_path))
    return jobs


def _batch_request(root: Path, batch_config_path: Path, jobs: Iterable[Job]) -> dict:
    return {
        "schema_version": 1,
        "batch_config": {
            "path": str(batch_config_path.relative_to(root)),
            "sha256": _sha256(batch_config_path),
        },
        "jobs": [
            {
                "video_id": job.video_id,
                "job_name": job.job_name,
                "config": str(job.config_path.relative_to(root)),
                "sha256": _sha256(job.config_path),
            }
            for job in jobs
        ],
        "outpaint_workers": 1,
        "upscale_workers": 1,
        "queue_policy": "fifo",
    }


def _bind_batch_request(
    request_path: Path,
    state_path: Path,
    request: dict,
    *,
    resume: bool,
) -> None:
    if resume:
        if not request_path.exists():
            raise ValueError(f"cannot resume without batch request binding: {request_path}")
        persisted = _load_json(request_path)
        if persisted != request:
            raise ValueError("batch or job config bytes changed since the recorded batch request")
        return
    if request_path.exists() or state_path.exists():
        raise ValueError("existing batch state requires --resume; refusing to overwrite queue history")
    _atomic_write_json(request_path, request)


def _production_child_runner(root: Path) -> ChildRunner:
    entrypoint = root / "run_cert.sh"

    def run(stage: str, job: Job, resume: bool, emit: Callable[[str], None]) -> int:
        command = [str(entrypoint), "--config", str(job.config_path), "--stage", stage]
        if resume:
            command.append("--resume")
        emit(f"[QUEUE_COMMAND] video={job.video_id} stage={stage} command={' '.join(command)}")
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            emit(f"[QUEUE_CHILD_ERROR] video={job.video_id} stage={stage} error={exc}")
            return 127
        assert process.stdout is not None
        for line in process.stdout:
            emit(f"[CHILD video={job.video_id} stage={stage}] {line.rstrip()}")
        return process.wait()

    return run


def _read_resume_states(state_path: Path, jobs: Iterable[Job]) -> dict[str, str]:
    expected = {job.video_id for job in jobs}
    if not state_path.exists():
        return {video_id: "PENDING" for video_id in expected}
    raw = _load_json(state_path)
    states: dict[str, str] = {}
    for video_id in expected:
        value = raw.get(video_id, "PENDING")
        if value not in ALLOWED_STATES:
            raise ValueError(f"invalid persisted state for {video_id}: {value}")
        if value in {"OUTPAINT_RUNNING", "FAILED_OUTPAINT", "PREPROCESS_READY"}:
            value = "PENDING"
        elif value in {"UPSCALE_RUNNING", "FAILED_UPSCALE", "OUTPAINT_DONE", "DONE"}:
            value = "UPSCALE_WAITING"
        states[video_id] = value
    return states


def run_batch(
    batch_config_path: Path,
    *,
    resume: bool = False,
    child_runner: ChildRunner | None = None,
    root: Path | None = None,
) -> BatchResult:
    root = (root or Path(__file__).resolve().parent).resolve()
    batch_config_path = batch_config_path.resolve()
    jobs = load_jobs(root, batch_config_path)
    state_path = root / "state" / "batch_state.json"
    request_path = root / "state" / "batch_request.json"
    log_path = root / "logs" / "batch_pipeline.log"
    request = _batch_request(root, batch_config_path, jobs)
    _bind_batch_request(request_path, state_path, request, resume=resume)
    initial = _read_resume_states(state_path, jobs) if resume else {
        job.video_id: "PENDING" for job in jobs
    }
    producer_skip = {
        video_id for video_id, state in initial.items() if state == "UPSCALE_WAITING"
    }
    journal = BatchJournal(state_path, log_path, initial)
    runner = child_runner or _production_child_runner(root)
    ready: queue.Queue[Job | None] = queue.Queue()
    stop_requested = threading.Event()
    interrupted = False

    # Preserve FIFO order for completed outpaint artifacts recovered by --resume.
    for job in jobs:
        if initial[job.video_id] == "UPSCALE_WAITING":
            ready.put(job)

    def invoke(stage: str, job: Job) -> int:
        try:
            return runner(stage, job, resume, journal.log)
        except Exception as exc:  # Child-launch boundary: keep the other jobs runnable.
            journal.log(
                f"[QUEUE_CHILD_ERROR] video={job.video_id} stage={stage} "
                f"error={type(exc).__name__}:{exc}"
            )
            return 125

    def producer() -> None:
        try:
            for job in jobs:
                if stop_requested.is_set():
                    break
                if job.video_id in producer_skip:
                    continue
                journal.transition(
                    job.video_id,
                    "OUTPAINT_RUNNING",
                    f"[QUEUE] start_outpaint video={job.video_id}",
                )
                return_code = invoke("outpaint", job)
                if return_code != 0:
                    journal.transition(
                        job.video_id,
                        "FAILED_OUTPAINT",
                        f"[QUEUE] outpaint_failed video={job.video_id} returncode={return_code}",
                    )
                    continue
                journal.transition(
                    job.video_id,
                    "OUTPAINT_DONE",
                    f"[QUEUE] outpaint_done video={job.video_id}",
                )
                journal.transition(
                    job.video_id,
                    "UPSCALE_WAITING",
                    f"[QUEUE] enqueue_upscale video={job.video_id}",
                )
                ready.put(job)
        finally:
            ready.put(None)

    def consumer() -> None:
        while True:
            job = ready.get()
            try:
                if job is None:
                    return
                if stop_requested.is_set():
                    continue
                if journal.snapshot()[job.video_id] == "DONE":
                    continue
                journal.transition(
                    job.video_id,
                    "UPSCALE_RUNNING",
                    f"[QUEUE] start_upscale video={job.video_id}",
                )
                return_code = invoke("upscale", job)
                if return_code != 0:
                    journal.transition(
                        job.video_id,
                        "FAILED_UPSCALE",
                        f"[QUEUE] upscale_failed video={job.video_id} returncode={return_code}",
                    )
                    continue
                journal.transition(
                    job.video_id,
                    "DONE",
                    f"[QUEUE] upscale_done video={job.video_id}",
                )
            finally:
                ready.task_done()

    journal.log(
        f"[QUEUE] batch_start jobs={len(jobs)} outpaint_workers=1 upscale_workers=1 resume={str(resume).lower()}"
    )
    producer_thread = threading.Thread(target=producer, name="outpaint-worker")
    consumer_thread = threading.Thread(target=consumer, name="upscale-worker")
    producer_thread.start()
    consumer_thread.start()
    try:
        producer_thread.join()
        consumer_thread.join()
    except KeyboardInterrupt:
        interrupted = True
        stop_requested.set()
        journal.log("[QUEUE] interrupt_received action=finish_running_children_no_new_jobs")
        producer_thread.join()
        consumer_thread.join()

    states = journal.snapshot()
    done = sum(value == "DONE" for value in states.values())
    failed_outpaint = sum(value == "FAILED_OUTPAINT" for value in states.values())
    failed_upscale = sum(value == "FAILED_UPSCALE" for value in states.values())
    journal.log(
        "[QUEUE] batch_summary "
        f"done={done}/{len(jobs)} failed_outpaint={failed_outpaint} "
        f"failed_upscale={failed_upscale} interrupted={str(interrupted).lower()}"
    )
    return BatchResult(states=states, interrupted=interrupted)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FIFO outpaint/upscale certification queue")
    parser.add_argument("--config", required=True, type=Path, help="batch JSON configuration")
    parser.add_argument("--resume", action="store_true", help="resume recorded queue/stage state")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parent
    lock_path = root / "state" / "batch_runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[QUEUE] already_running lock={lock_path}", file=sys.stderr)
            return 2
        try:
            config_path = args.config if args.config.is_absolute() else root / args.config
            result = run_batch(config_path, resume=args.resume, root=root)
        except (ValueError, OSError) as exc:
            print(f"[QUEUE] configuration_error error={exc}", file=sys.stderr)
            return 2
        return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
