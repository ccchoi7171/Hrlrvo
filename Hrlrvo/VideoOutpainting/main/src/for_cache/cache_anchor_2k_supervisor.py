"""Bounded, hold-aware supervisor for the delegated Anchor-2K cache job.

This process never imports torch or loads a model.  It starts the existing
``for_cache.precache_anchor_2k`` runner only after an NVML-backed ``nvidia-smi``
idle check, monitors it, and requests a clean batch-boundary process exit when
the leader hold returns, another GPU process appears, or GPU visibility cannot
be queried.  Process exit is used deliberately so a paused VAE cannot retain
VRAM through a Python CUDA context.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python")
TMUX_SESSION = "Hrlrvo_cache_2K"
PAUSED_EXIT_CODE = 75
IDLE_GPU_MEMORY_MIB = 64
MIN_CGROUP_HEADROOM_BYTES = 20 * 2**30
STATUS_INTERVAL_SECONDS = 15.0
GPU_POLL_SECONDS = 2.0
PREPARED_VIDEO_BYTES = 1 * 3 * 65 * 1088 * 1920 * 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] supervisor: {message}", flush=True)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def atomic_json_replace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (FileNotFoundError, OSError, ValueError):
        return None


def count_cache(root: Path) -> tuple[int, int]:
    count = 0
    size = 0
    if not root.exists():
        return 0, 0
    for directory, _, files in os.walk(root):
        base = Path(directory)
        for name in files:
            if not name.startswith(".") and name.endswith(".pt"):
                path = base / name
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                if path.is_file():
                    count += 1
                    size += stat.st_size
    return count, size


def process_tree(root_pid: int) -> set[int]:
    found = {root_pid}
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        try:
            text = Path(f"/proc/{pid}/task/{pid}/children").read_text()
        except (FileNotFoundError, OSError):
            continue
        for value in text.split():
            child = int(value)
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def process_metrics(root_pid: int) -> dict[str, int | None]:
    rss = 0
    read_bytes = 0
    write_bytes = 0
    cpu_ticks = 0
    observed = False
    for pid in process_tree(root_pid):
        try:
            status = Path(f"/proc/{pid}/status").read_text().splitlines()
            rss += int(next(line.split()[1] for line in status if line.startswith("VmRSS:"))) * 1024
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            cpu_ticks += int(fields[13]) + int(fields[14])
            io_lines = Path(f"/proc/{pid}/io").read_text().splitlines()
            io_values = {line.split(":", 1)[0]: int(line.split(":", 1)[1]) for line in io_lines}
            read_bytes += io_values.get("read_bytes", 0)
            write_bytes += io_values.get("write_bytes", 0)
            observed = True
        except (FileNotFoundError, OSError, StopIteration, ValueError, IndexError):
            continue
    return {
        "rss_bytes": rss if observed else None,
        "read_bytes": read_bytes if observed else None,
        "write_bytes": write_bytes if observed else None,
        "cpu_ticks": cpu_ticks if observed else None,
    }


def _nvidia_query(arguments: list[str]) -> str:
    result = subprocess.run(
        ["nvidia-smi", *arguments], text=True, capture_output=True, timeout=10, check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(message or f"nvidia-smi exited {result.returncode}")
    return result.stdout


def gpu_snapshot() -> dict[str, Any]:
    try:
        gpu_lines = _nvidia_query([
            "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]).splitlines()
        gpus = []
        for line in gpu_lines:
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 5:
                raise RuntimeError(f"unexpected GPU query row: {line}")
            utilization = None if fields[4] in {"[Not Found]", "N/A"} else int(fields[4])
            gpus.append({
                "uuid": fields[0],
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "utilization_percent": utilization,
            })
        if not gpus:
            raise RuntimeError("nvidia-smi returned no visible GPUs")
        process_lines = _nvidia_query([
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]).splitlines()
        processes = []
        for line in process_lines:
            if not line.strip():
                continue
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 4:
                raise RuntimeError(f"unexpected compute-process query row: {line}")
            processes.append({
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": None if fields[3] in {"[Not Found]", "N/A"} else int(fields[3]),
            })
        return {"query": "PASS", "gpus": gpus, "compute_processes": processes, "error": None}
    except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
        return {"query": "UNAVAILABLE", "gpus": [], "compute_processes": [], "error": str(exc)}


def gpu_is_idle(snapshot: dict[str, Any]) -> bool:
    return (
        snapshot["query"] == "PASS"
        and not snapshot["compute_processes"]
        and all(gpu["memory_used_mib"] <= IDLE_GPU_MEMORY_MIB for gpu in snapshot["gpus"])
    )


def external_gpu_processes(snapshot: dict[str, Any], child_pid: int) -> list[dict[str, Any]]:
    allowed = process_tree(child_pid)
    return [item for item in snapshot["compute_processes"] if item["pid"] not in allowed]


def resource_snapshot(anchor_root: Path, total: int) -> dict[str, Any]:
    memory_max = read_int(Path("/sys/fs/cgroup/memory.max"))
    memory_current = read_int(Path("/sys/fs/cgroup/memory.current"))
    disk = shutil.disk_usage(anchor_root if anchor_root.exists() else anchor_root.parent)
    cached, cache_bytes = count_cache(anchor_root)
    mean_entry = cache_bytes / cached if cached else 4_704_425
    estimated_remaining = int(max(0, total - cached) * mean_entry * 1.20)
    try:
        filesystem = subprocess.run(
            ["stat", "-f", "-c", "%T", str(anchor_root if anchor_root.exists() else anchor_root.parent)],
            text=True, capture_output=True, timeout=5, check=False,
        ).stdout.strip() or "unavailable"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        filesystem = "unavailable"
    return {
        "updated_utc": utc_now(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_memory_max_bytes": memory_max,
        "cgroup_memory_current_bytes": memory_current,
        "cgroup_memory_headroom_bytes": (
            memory_max - memory_current
            if memory_max is not None and memory_current is not None else None
        ),
        "filesystem_type": filesystem,
        "filesystem_free_bytes": disk.free,
        "filesystem_total_bytes": disk.total,
        "estimated_remaining_cache_bytes_with_20pct_margin": estimated_remaining,
        "prepared_video_bytes_each": PREPARED_VIDEO_BYTES,
        "anchor_file_count": cached,
        "anchor_bytes": cache_bytes,
        "gpu": gpu_snapshot(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }


class Supervisor:
    def __init__(self, *, config: Path, output_dir: Path) -> None:
        self.config = config.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output_dir / "delegated_status.json"
        self.status_lock_path = self.output_dir / "delegated_status.lock"
        self.hold_path = self.output_dir / "gpu_test_priority.hold"
        self.yield_path = self.output_dir / "gpu_external_yield.request"
        self.progress_path = self.output_dir / "runner_progress.json"
        self.benchmark_path = self.output_dir / "anchor_cache_benchmark.json"
        self.log_path = self.output_dir / "cache_anchor_2k.log"
        self.anchor_root = Path("/home/work/ai-graphics_33/Hrlrvo/cache_Data_Anchor_2K/train/latents")
        previous = read_json(self.status_path) or {}
        self.initial_count = int(
            previous.get(
                "delegation_initial_cached_count",
                previous.get("cached_count", count_cache(self.anchor_root)[0]),
            )
        )
        current_count = int(previous.get("cached_count", count_cache(self.anchor_root)[0]))
        self.total = int(
            previous.get(
                "total_compatible",
                current_count + int(previous.get("missing_count", 30_024 - current_count)),
            )
        )
        self.child: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self.command = [
            str(PYTHON), "-B", "-u", "-m", "for_cache.precache_anchor_2k",
            "--config", str(self.config), "--all", "--auto-tune",
            "--benchmark-report", str(self.benchmark_path),
            "--gpu-hold-file", str(self.hold_path),
            "--gpu-yield-file", str(self.yield_path),
            "--progress-report", str(self.progress_path),
        ]
        self.command_text = shlex.join(self.command)

    def update_status(self, **changes: Any) -> dict[str, Any]:
        with self.status_lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            status = read_json(self.status_path) or {}
            status.update({
                "accepted": True,
                "updated_utc": utc_now(),
                "tmux_session": TMUX_SESSION,
                "delegation_initial_cached_count": self.initial_count,
                "total_compatible": self.total,
                "actual_command": self.command_text,
                "supervisor_command": str(self.output_dir / "run_cache_anchor_2k_supervisor.sh"),
                "log_path": str(self.log_path),
            })
            status.update(changes)
            atomic_json_replace(self.status_path, status)
            fcntl.flock(lock, fcntl.LOCK_UN)
        return status

    def refresh_counts(self) -> tuple[int, int]:
        count, _ = count_cache(self.anchor_root)
        return count, max(0, self.total - count)

    def benchmark_status(self) -> dict[str, Any]:
        complete = read_json(self.benchmark_path)
        if complete is not None:
            return {
                "phase": "COMPLETE",
                "report_path": str(self.benchmark_path),
                "selected_config": complete.get("selected_config"),
                "selection_reason": complete.get("selection_reason"),
                "bounded_new_records": complete.get("bounded_new_records"),
                "max_process_rss_gib": complete.get("max_process_rss_gib"),
                "max_torch_cuda_allocated_gib": complete.get("max_torch_cuda_allocated_gib"),
            }
        partial = read_json(self.benchmark_path.with_name(
            f"{self.benchmark_path.stem}.partial{self.benchmark_path.suffix}"
        ))
        return {
            "phase": partial.get("phase", "NOT_STARTED") if partial else "NOT_STARTED",
            "report_path": str(self.benchmark_path),
            "results": partial.get("results") if partial else None,
        }

    def request_external_yield(self, reason: str) -> None:
        if self.yield_path.exists():
            return
        temporary = self.yield_path.with_name(f".{self.yield_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(f"created_utc={utc_now()}\nreason={reason}\n")
            os.replace(temporary, self.yield_path)
        finally:
            temporary.unlink(missing_ok=True)

    def clear_external_yield(self) -> None:
        self.yield_path.unlink(missing_ok=True)

    def waiting_gate(self) -> dict[str, Any]:
        last_update = 0.0
        while not self.stop_requested:
            resources = resource_snapshot(self.anchor_root, self.total)
            gpu = resources["gpu"]
            hold = self.hold_path.exists()
            phase = "READY_TO_START"
            error = None
            ready_for_gpu_test = False
            if hold:
                phase = "WAITING_GPU_TEST_PRIORITY"
                ready_for_gpu_test = gpu["query"] == "PASS"
            elif gpu["query"] != "PASS":
                phase = "WAITING_GPU_QUERY"
                error = f"GPU availability unavailable (not treated as idle): {gpu['error']}"
            elif not gpu_is_idle(gpu):
                phase = "WAITING_GPU_AVAILABLE"
            elif (
                resources["cgroup_memory_headroom_bytes"] is not None
                and resources["cgroup_memory_headroom_bytes"] < MIN_CGROUP_HEADROOM_BYTES
            ):
                phase = "WAITING_MEMORY_HEADROOM"
            elif resources["filesystem_free_bytes"] <= resources[
                "estimated_remaining_cache_bytes_with_20pct_margin"
            ]:
                phase = "WAITING_DISK_CAPACITY"
                error = "NFS free space is below the estimated remaining cache size plus 20% margin"
            else:
                self.clear_external_yield()
                cached, missing = self.refresh_counts()
                self.update_status(
                    phase="STARTING_BENCHMARK_OR_CACHE",
                    gpu_test_hold=False,
                    cache_gpu_memory_released=True,
                    ready_for_gpu_test=False,
                    resources=resources,
                    cached_count=cached,
                    missing_count=missing,
                    newly_created_since_delegation=max(0, cached - self.initial_count),
                    benchmark=self.benchmark_status(),
                    error=None,
                )
                return resources

            now = time.monotonic()
            if now - last_update >= STATUS_INTERVAL_SECONDS:
                cached, missing = self.refresh_counts()
                self.update_status(
                    phase=phase,
                    gpu_test_hold=hold,
                    cache_gpu_memory_released=gpu["query"] == "PASS",
                    ready_for_gpu_test=ready_for_gpu_test,
                    runner_pid=None,
                    resources=resources,
                    cached_count=cached,
                    missing_count=missing,
                    newly_created_since_delegation=max(0, cached - self.initial_count),
                    benchmark=self.benchmark_status(),
                    error=error,
                )
                log(f"{phase}; hold={hold} gpu_query={gpu['query']} processes={gpu['compute_processes']}")
                last_update = now
            time.sleep(GPU_POLL_SECONDS)
        raise KeyboardInterrupt

    def launch(self) -> subprocess.Popen[str]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        old_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + (
            f":{old_pythonpath}" if old_pythonpath else ""
        )
        log(f"launching: {self.command_text}")
        child = subprocess.Popen(
            self.command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
        )
        self.child = child
        return child

    def monitor(self, child: subprocess.Popen[str], baseline: dict[str, Any]) -> tuple[int, dict[str, Any], str | None]:
        started = time.monotonic()
        baseline_process = process_metrics(child.pid)
        peak_gpu_used = 0
        peak_gpu_util: int | None = None
        peak_cgroup_memory = baseline.get("cgroup_memory_current_bytes") or 0
        peak_process_rss = 0
        last_process = baseline_process
        last_gpu = baseline["gpu"]
        pause_reason: str | None = None
        last_update = 0.0
        while child.poll() is None:
            gpu = gpu_snapshot()
            last_gpu = gpu
            current_process = process_metrics(child.pid)
            last_process = current_process
            if current_process["rss_bytes"] is not None:
                peak_process_rss = max(peak_process_rss, int(current_process["rss_bytes"]))
            current_cgroup = read_int(Path("/sys/fs/cgroup/memory.current"))
            if current_cgroup is not None:
                peak_cgroup_memory = max(peak_cgroup_memory, current_cgroup)
            if gpu["query"] == "PASS":
                peak_gpu_used = max(peak_gpu_used, *(item["memory_used_mib"] for item in gpu["gpus"]))
                utils = [item["utilization_percent"] for item in gpu["gpus"] if item["utilization_percent"] is not None]
                if utils:
                    peak_gpu_util = max([peak_gpu_util or 0, *utils])

            hold = self.hold_path.exists()
            external = external_gpu_processes(gpu, child.pid) if gpu["query"] == "PASS" else []
            if hold:
                pause_reason = "leader GPU test priority hold"
            elif gpu["query"] != "PASS":
                pause_reason = f"GPU query unavailable: {gpu['error']}"
                self.request_external_yield(pause_reason)
            elif external:
                pause_reason = f"external GPU processes: {external}"
                self.request_external_yield(pause_reason)

            now = time.monotonic()
            if now - last_update >= STATUS_INTERVAL_SECONDS or pause_reason is not None:
                progress = read_json(self.progress_path) or {}
                cached = int(progress.get("cached_count", self.refresh_counts()[0]))
                missing = max(0, self.total - cached)
                if pause_reason:
                    phase = "YIELDING_GPU_TEST_PRIORITY" if hold else "YIELDING_GPU_EXTERNAL_ACTIVITY"
                elif str(progress.get("stage", "")).startswith("benchmark"):
                    phase = "BENCHMARKING"
                else:
                    phase = "RUNNING_FULL_CACHE"
                self.update_status(
                    phase=phase,
                    gpu_test_hold=hold,
                    cache_gpu_memory_released=False,
                    ready_for_gpu_test=False,
                    runner_pid=child.pid,
                    cached_count=cached,
                    missing_count=missing,
                    newly_created_since_delegation=max(0, cached - self.initial_count),
                    runner_progress=progress or None,
                    benchmark=self.benchmark_status(),
                    current_gpu=gpu,
                    error=pause_reason,
                )
                last_update = now
            time.sleep(GPU_POLL_SECONDS)

        return_code = child.wait()
        elapsed = time.monotonic() - started
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        start_ticks = baseline_process.get("cpu_ticks")
        end_ticks = last_process.get("cpu_ticks")
        cpu_core_equivalent = None
        if start_ticks is not None and end_ticks is not None and elapsed > 0:
            cpu_core_equivalent = max(0.0, (int(end_ticks) - int(start_ticks)) / ticks_per_second / elapsed)
        metrics = {
            "started_utc": datetime.fromtimestamp(time.time() - elapsed, timezone.utc).isoformat(),
            "ended_utc": utc_now(),
            "wall_seconds": elapsed,
            "return_code": return_code,
            "peak_process_tree_rss_bytes": peak_process_rss or None,
            "peak_cgroup_memory_bytes_shared": peak_cgroup_memory or None,
            "peak_nvidia_memory_used_mib": peak_gpu_used,
            "peak_nvidia_utilization_percent": peak_gpu_util,
            "process_read_bytes_last_observed": last_process.get("read_bytes"),
            "process_write_bytes_last_observed": last_process.get("write_bytes"),
            "cpu_core_equivalent_last_observed": cpu_core_equivalent,
            "last_gpu_snapshot": last_gpu,
            "unavailable_note": (
                "GPU utilization is unavailable on this device/query"
                if peak_gpu_util is None else None
            ),
        }
        return return_code, metrics, pause_reason

    def confirm_release(self, child_pid: int) -> dict[str, Any]:
        while not self.stop_requested:
            gpu = gpu_snapshot()
            if gpu["query"] == "PASS" and all(
                item["pid"] != child_pid for item in gpu["compute_processes"]
            ):
                return gpu
            self.update_status(
                phase="CONFIRMING_CACHE_GPU_MEMORY_RELEASE",
                gpu_test_hold=self.hold_path.exists(),
                cache_gpu_memory_released=False,
                ready_for_gpu_test=False,
                runner_pid=None,
                current_gpu=gpu,
                error=(
                    "GPU query unavailable; not treating cache memory as released"
                    if gpu["query"] != "PASS" else "cache process still visible to nvidia-smi"
                ),
            )
            time.sleep(GPU_POLL_SECONDS)
        raise KeyboardInterrupt

    def run(self) -> int:
        log("controller active; no model is loaded in the supervisor")
        while not self.stop_requested:
            baseline = self.waiting_gate()
            child = self.launch()
            self.update_status(
                phase="STARTING_RUNNER",
                gpu_test_hold=False,
                cache_gpu_memory_released=False,
                ready_for_gpu_test=False,
                runner_pid=child.pid,
                actual_command=self.command_text,
                error=None,
            )
            return_code, metrics, pause_reason = self.monitor(child, baseline)
            released_gpu = self.confirm_release(child.pid)
            self.child = None
            cached, missing = self.refresh_counts()
            hold = self.hold_path.exists()
            if return_code == 0:
                self.update_status(
                    phase="COMPLETE",
                    gpu_test_hold=hold,
                    cache_gpu_memory_released=True,
                    ready_for_gpu_test=hold,
                    runner_pid=None,
                    cached_count=cached,
                    missing_count=missing,
                    newly_created_since_delegation=max(0, cached - self.initial_count),
                    benchmark=self.benchmark_status(),
                    last_attempt_resources=metrics,
                    current_gpu=released_gpu,
                    error=None,
                )
                log(f"runner complete; cached={cached} missing={missing}")
                return 0
            if return_code == PAUSED_EXIT_CODE or hold or self.yield_path.exists():
                phase = "WAITING_GPU_TEST_PRIORITY" if hold else "WAITING_GPU_AVAILABLE"
                self.update_status(
                    phase=phase,
                    gpu_test_hold=hold,
                    cache_gpu_memory_released=True,
                    ready_for_gpu_test=hold,
                    runner_pid=None,
                    cached_count=cached,
                    missing_count=missing,
                    newly_created_since_delegation=max(0, cached - self.initial_count),
                    benchmark=self.benchmark_status(),
                    last_attempt_resources=metrics,
                    current_gpu=released_gpu,
                    error=pause_reason,
                )
                log(f"runner yielded with VRAM released; reason={pause_reason}; cached={cached}")
                continue
            self.update_status(
                phase="ERROR",
                gpu_test_hold=hold,
                cache_gpu_memory_released=True,
                ready_for_gpu_test=hold,
                runner_pid=None,
                cached_count=cached,
                missing_count=missing,
                newly_created_since_delegation=max(0, cached - self.initial_count),
                benchmark=self.benchmark_status(),
                last_attempt_resources=metrics,
                current_gpu=released_gpu,
                error=f"runner exited {return_code}; see append-only log",
            )
            log(f"runner failed closed with exit={return_code}")
            return return_code
        return 130


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    supervisor = Supervisor(config=args.config, output_dir=args.output_dir)

    lock_path = supervisor.output_dir / "supervisor.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another Anchor-2K supervisor already holds the singleton lock") from None

        def stop(signum: int, frame: Any) -> None:
            del frame
            supervisor.stop_requested = True
            supervisor.request_external_yield(f"supervisor received signal {signum}")

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop)
        try:
            raise SystemExit(supervisor.run())
        except KeyboardInterrupt:
            supervisor.stop_requested = True
            if supervisor.child is not None and supervisor.child.poll() is None:
                supervisor.request_external_yield("supervisor interrupted")
                supervisor.child.wait()
            raise SystemExit(130) from None


if __name__ == "__main__":
    main()
