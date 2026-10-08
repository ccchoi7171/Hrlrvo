"""Multi-lane, hold-aware supervisor for the delegated Anchor-2K cache job.

Each lane is an independent batch-size-one invocation of the validated
``for_cache.precache_anchor_2k`` runner.  GT-order modulo sharding keeps record
ownership disjoint while every decode, VAE encode, lock, validation, and atomic
save remains in the shared ``LazyAnchorCache`` implementation.

This controller deliberately does not import torch.  A hold, an external GPU
process, an unavailable GPU query, or critical cgroup pressure requests a clean
batch-boundary exit from every lane so their CUDA contexts return VRAM.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shlex
import signal
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

from for_cache import cache_anchor_2k_supervisor as base

REQUESTED_LANES = 5
PARALLEL_CPU_WORKERS = 3
PARALLEL_PREFETCH = 3
FALLBACK_CPU_WORKERS = 4
FALLBACK_PREFETCH = 4
MIN_RUNNING_WORKING_HEADROOM_BYTES = 12 * 2**30
ESTIMATED_PEAK_PRIVATE_BYTES_PER_LANE = 18 * 2**30
MIN_LAUNCH_RESERVE_BYTES = 12 * 2**30
LANE_START_STAGGER_SECONDS = 2.0
RATE_WINDOW_SECONDS = 120.0


def memory_events() -> dict[str, int]:
    try:
        return {
            key: int(value)
            for key, value in (
                line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
            )
        }
    except (FileNotFoundError, OSError, ValueError):
        return {}


def cgroup_memory_state() -> dict[str, int | None]:
    current = base.read_int(Path("/sys/fs/cgroup/memory.current"))
    maximum = base.read_int(Path("/sys/fs/cgroup/memory.max"))
    try:
        stats = {
            key: int(value)
            for key, value in (
                line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
            )
        }
    except (FileNotFoundError, OSError, ValueError):
        stats = {}
    inactive_file = stats.get("inactive_file")
    working_set = (
        max(0, current - inactive_file)
        if current is not None and inactive_file is not None else None
    )
    return {
        "current_bytes": current,
        "max_bytes": maximum,
        "inactive_file_bytes": inactive_file,
        "anon_bytes": stats.get("anon"),
        "file_bytes": stats.get("file"),
        "kernel_bytes": stats.get("kernel"),
        "working_set_bytes": working_set,
        "working_headroom_bytes": (
            maximum - working_set
            if maximum is not None and working_set is not None else None
        ),
    }


def aggregate_process_metrics(root_pids: list[int]) -> dict[str, int | None]:
    metrics = [base.process_metrics(pid) for pid in root_pids]

    def total(key: str) -> int | None:
        values = [item[key] for item in metrics if item[key] is not None]
        return sum(int(value) for value in values) if values else None

    return {
        "rss_bytes": total("rss_bytes"),
        "read_bytes": total("read_bytes"),
        "write_bytes": total("write_bytes"),
        "cpu_ticks": total("cpu_ticks"),
    }


def external_gpu_processes(snapshot: dict[str, Any], root_pids: list[int]) -> list[dict[str, Any]]:
    allowed: set[int] = set()
    for pid in root_pids:
        allowed.update(base.process_tree(pid))
    return [item for item in snapshot["compute_processes"] if item["pid"] not in allowed]


class ParallelSupervisor:
    def __init__(self, *, config: Path, output_dir: Path) -> None:
        self.config = config.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output_dir / "delegated_status.json"
        self.status_lock_path = self.output_dir / "delegated_status.lock"
        self.hold_path = self.output_dir / "gpu_test_priority.hold"
        self.yield_path = self.output_dir / "gpu_external_yield.request"
        self.log_path = self.output_dir / "cache_anchor_2k.log"
        self.anchor_root = Path("/home/work/ai-graphics_33/Hrlrvo/cache_Data_Anchor_2K/train/latents")
        previous = base.read_json(self.status_path) or {}
        current_count = base.count_cache(self.anchor_root)[0]
        self.initial_count = int(previous.get("delegation_initial_cached_count", current_count))
        self.total = int(previous.get("total_compatible", 30_024))
        self.lane_count = REQUESTED_LANES
        self.children: dict[int, subprocess.Popen[str]] = {}
        self.commands: dict[int, list[str]] = {}
        self.stop_requested = False
        self.memory_fallbacks = 0

    @property
    def workers(self) -> int:
        return PARALLEL_CPU_WORKERS if self.lane_count > 1 else FALLBACK_CPU_WORKERS

    @property
    def prefetch(self) -> int:
        return PARALLEL_PREFETCH if self.lane_count > 1 else FALLBACK_PREFETCH

    def progress_path(self, lane: int) -> Path:
        return self.output_dir / f"runner_progress_lane{lane}.json"

    def command(self, lane: int) -> list[str]:
        return [
            str(base.PYTHON), "-B", "-u", "-m", "for_cache.precache_anchor_2k",
            "--config", str(self.config),
            "--all",
            "--gpu-batch-size", "1",
            "--cpu-workers", str(self.workers),
            "--prefetch", str(self.prefetch),
            "--shard-count", str(self.lane_count),
            "--shard-index", str(lane),
            "--gpu-hold-file", str(self.hold_path),
            "--gpu-yield-file", str(self.yield_path),
            "--progress-report", str(self.progress_path(lane)),
        ]

    def command_texts(self) -> list[str]:
        return [shlex.join(self.command(lane)) for lane in range(self.lane_count)]

    def update_status(self, **changes: Any) -> dict[str, Any]:
        command_texts = self.command_texts()
        with self.status_lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            status = base.read_json(self.status_path) or {}
            status.update({
                "accepted": True,
                "updated_utc": base.utc_now(),
                "tmux_session": base.TMUX_SESSION,
                "delegation_initial_cached_count": self.initial_count,
                "total_compatible": self.total,
                "actual_command": " ; ".join(command_texts),
                "actual_commands": command_texts,
                "supervisor_command": str(
                    self.output_dir / "run_cache_anchor_2k_parallel_supervisor.sh"
                ),
                "log_path": str(self.log_path),
                "parallelization": {
                    "requested_lanes": REQUESTED_LANES,
                    "active_configuration_lanes": self.lane_count,
                    "gpu_batch_size_per_lane": 1,
                    "cpu_workers_per_lane": self.workers,
                    "prefetch_per_lane": self.prefetch,
                    "total_cpu_decode_workers": self.lane_count * self.workers,
                    "record_partition": "GT-order index modulo lane count",
                    "batch2_selected": False,
                    "batch2_rejection": "prior same-input numeric equivalence and throughput gate failed",
                    "memory_fallbacks": self.memory_fallbacks,
                    "memory_policy": (
                        "cgroup current-inactive_file working set; reduce lanes one at a time "
                        "on <12 GiB working headroom or new OOM event"
                    ),
                },
            })
            status.update(changes)
            base.atomic_json_replace(self.status_path, status)
            fcntl.flock(lock, fcntl.LOCK_UN)
        return status

    def refresh_counts(self) -> tuple[int, int]:
        cached = base.count_cache(self.anchor_root)[0]
        return cached, max(0, self.total - cached)

    def request_yield(self, reason: str) -> None:
        if self.yield_path.exists():
            return
        temporary = self.yield_path.with_name(f".{self.yield_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(f"created_utc={base.utc_now()}\nreason={reason}\n")
            os.replace(temporary, self.yield_path)
        finally:
            temporary.unlink(missing_ok=True)

    def waiting_gate(self) -> dict[str, Any]:
        last_update = 0.0
        while not self.stop_requested:
            resources = base.resource_snapshot(self.anchor_root, self.total)
            memory = cgroup_memory_state()
            gpu = resources["gpu"]
            hold = self.hold_path.exists()
            phase = "READY_TO_START_PARALLEL_CACHE"
            error = None
            if hold:
                phase = "WAITING_GPU_TEST_PRIORITY"
            elif gpu["query"] != "PASS":
                phase = "WAITING_GPU_QUERY"
                error = f"GPU availability unavailable (not treated as idle): {gpu['error']}"
            elif not base.gpu_is_idle(gpu):
                phase = "WAITING_GPU_AVAILABLE"
            elif (
                memory["working_headroom_bytes"] is not None
                and memory["working_headroom_bytes"]
                < self.lane_count * ESTIMATED_PEAK_PRIVATE_BYTES_PER_LANE + MIN_LAUNCH_RESERVE_BYTES
            ):
                affordable = max(
                    1,
                    int(
                        max(0, memory["working_headroom_bytes"] - MIN_LAUNCH_RESERVE_BYTES)
                        // ESTIMATED_PEAK_PRIVATE_BYTES_PER_LANE
                    ),
                )
                if affordable < self.lane_count:
                    previous = self.lane_count
                    self.lane_count = affordable
                    self.memory_fallbacks += 1
                    base.log(
                        f"launch-time memory bound reduced lanes {previous}->{affordable}; "
                        f"working_headroom={memory['working_headroom_bytes'] / 2**30:.2f} GiB"
                    )
                    continue
                phase = "WAITING_MEMORY_HEADROOM"
            elif resources["filesystem_free_bytes"] <= resources[
                "estimated_remaining_cache_bytes_with_20pct_margin"
            ]:
                phase = "WAITING_DISK_CAPACITY"
                error = "NFS free space is below remaining cache estimate plus 20% margin"
            else:
                self.yield_path.unlink(missing_ok=True)
                return resources

            now = time.monotonic()
            if now - last_update >= base.STATUS_INTERVAL_SECONDS:
                cached, missing = self.refresh_counts()
                self.update_status(
                    phase=phase,
                    gpu_test_hold=hold,
                    cache_gpu_memory_released=gpu["query"] == "PASS",
                    ready_for_gpu_test=hold and gpu["query"] == "PASS",
                    runner_pid=None,
                    runner_pids=[],
                    resources=resources,
                    cgroup_memory=memory,
                    cached_count=cached,
                    missing_count=missing,
                    newly_created_since_delegation=max(0, cached - self.initial_count),
                    current_gpu=gpu,
                    error=error,
                )
                base.log(f"parallel {phase}; hold={hold} gpu_query={gpu['query']}")
                last_update = now
            time.sleep(base.GPU_POLL_SECONDS)
        raise KeyboardInterrupt

    def launch(self) -> dict[int, subprocess.Popen[str]]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        old_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(base.PROJECT_ROOT / "src") + (
            f":{old_pythonpath}" if old_pythonpath else ""
        )
        self.commands = {}
        self.children = {}
        for lane in range(self.lane_count):
            command = self.command(lane)
            self.commands[lane] = command
            base.log(f"launching parallel lane {lane}/{self.lane_count}: {shlex.join(command)}")
            self.children[lane] = subprocess.Popen(
                command,
                cwd=base.PROJECT_ROOT,
                env=environment,
                text=True,
            )
            if lane + 1 < self.lane_count:
                time.sleep(LANE_START_STAGGER_SECONDS)
        return self.children

    def lane_progress(self) -> dict[str, Any]:
        return {
            str(lane): base.read_json(self.progress_path(lane))
            for lane in range(self.lane_count)
        }

    def monitor(
        self,
        children: dict[int, subprocess.Popen[str]],
        baseline: dict[str, Any],
    ) -> tuple[dict[int, int], dict[str, Any], str | None]:
        started = time.monotonic()
        started_utc = base.utc_now()
        starting_count = self.refresh_counts()[0]
        roots = [child.pid for child in children.values()]
        baseline_process = aggregate_process_metrics(roots)
        last_process = baseline_process
        initial_events = memory_events()
        last_gpu = baseline["gpu"]
        peak_gpu_used = 0
        peak_gpu_util: int | None = None
        peak_cgroup_memory = baseline.get("cgroup_memory_current_bytes") or 0
        peak_process_rss = 0
        pause_reason: str | None = None
        memory_pressure = False
        last_update = 0.0
        rate_history: deque[tuple[float, int]] = deque([(started, starting_count)])
        outcomes: dict[int, int] = {}

        while len(outcomes) < len(children):
            for lane, child in children.items():
                if lane not in outcomes and child.poll() is not None:
                    outcomes[lane] = child.wait()

            live_roots = [child.pid for lane, child in children.items() if lane not in outcomes]
            gpu = base.gpu_snapshot()
            last_gpu = gpu
            current_process = aggregate_process_metrics(live_roots or roots)
            last_process = current_process
            if current_process["rss_bytes"] is not None:
                peak_process_rss = max(peak_process_rss, int(current_process["rss_bytes"]))
            memory = cgroup_memory_state()
            memory_current = memory["current_bytes"]
            memory_max = memory["max_bytes"]
            headroom = memory["working_headroom_bytes"]
            if memory_current is not None:
                peak_cgroup_memory = max(peak_cgroup_memory, memory_current)
            events = memory_events()
            oom_increased = any(
                events.get(key, 0) > initial_events.get(key, 0) for key in ("oom", "oom_kill")
            )
            if gpu["query"] == "PASS":
                peak_gpu_used = max(peak_gpu_used, *(item["memory_used_mib"] for item in gpu["gpus"]))
                utilizations = [
                    item["utilization_percent"] for item in gpu["gpus"]
                    if item["utilization_percent"] is not None
                ]
                if utilizations:
                    peak_gpu_util = max(peak_gpu_util or 0, *utilizations)

            hold = self.hold_path.exists()
            external = external_gpu_processes(gpu, roots) if gpu["query"] == "PASS" else []
            nonpause_failures = {
                lane: code for lane, code in outcomes.items() if code not in (0, base.PAUSED_EXIT_CODE)
            }
            if self.stop_requested:
                pause_reason = "parallel supervisor received stop signal"
            elif hold:
                pause_reason = "leader GPU test priority hold"
            elif gpu["query"] != "PASS":
                pause_reason = f"GPU query unavailable: {gpu['error']}"
            elif external:
                pause_reason = f"external GPU processes: {external}"
            elif oom_increased:
                pause_reason = "cgroup OOM event increased during parallel attempt"
                memory_pressure = True
            elif headroom is not None and headroom < MIN_RUNNING_WORKING_HEADROOM_BYTES:
                pause_reason = (
                    f"cgroup working-set headroom {headroom / 2**30:.2f} GiB below safety floor "
                    f"{MIN_RUNNING_WORKING_HEADROOM_BYTES / 2**30:.0f} GiB"
                )
                memory_pressure = True
            elif nonpause_failures:
                pause_reason = f"parallel lane failure: {nonpause_failures}"

            if pause_reason is not None:
                self.request_yield(pause_reason)

            now = time.monotonic()
            cached, missing = self.refresh_counts()
            rate_history.append((now, cached))
            while len(rate_history) > 2 and now - rate_history[0][0] > RATE_WINDOW_SECONDS:
                rate_history.popleft()
            window_seconds = now - rate_history[0][0]
            recent_rate = (
                (cached - rate_history[0][1]) / window_seconds if window_seconds >= 1 else 0.0
            )
            elapsed = now - started
            overall_rate = (cached - starting_count) / elapsed if elapsed > 0 else 0.0
            eta_hours = missing / recent_rate / 3600 if recent_rate > 0 else None
            if now - last_update >= base.STATUS_INTERVAL_SECONDS or pause_reason is not None:
                if pause_reason:
                    phase = "YIELDING_PARALLEL_CACHE"
                elif all(
                    (progress or {}).get("stage") == "full"
                    for progress in self.lane_progress().values()
                ):
                    phase = "RUNNING_PARALLEL_FULL_CACHE"
                else:
                    phase = "STARTING_PARALLEL_RUNNERS"
                self.update_status(
                    phase=phase,
                    gpu_test_hold=hold,
                    cache_gpu_memory_released=False,
                    ready_for_gpu_test=False,
                    runner_pid=None,
                    runner_pids=roots,
                    lane_outcomes=outcomes,
                    lane_progress=self.lane_progress(),
                    cached_count=cached,
                    missing_count=missing,
                    newly_created_since_delegation=max(0, cached - self.initial_count),
                    current_gpu=gpu,
                    parallel_observation={
                        "attempt_started_utc": started_utc,
                        "attempt_start_cached_count": starting_count,
                        "created_this_attempt": cached - starting_count,
                        "wall_seconds": elapsed,
                        "overall_records_per_second_including_startup": overall_rate,
                        "recent_window_seconds": window_seconds,
                        "recent_records_per_second": recent_rate,
                        "projected_remaining_hours_at_recent_rate": eta_hours,
                        "cgroup_memory_current_bytes": memory_current,
                        "cgroup_memory_max_bytes": memory_max,
                        "cgroup_raw_headroom_bytes": (
                            memory_max - memory_current
                            if memory_max is not None and memory_current is not None else None
                        ),
                        "cgroup_inactive_file_bytes": memory["inactive_file_bytes"],
                        "cgroup_anon_bytes": memory["anon_bytes"],
                        "cgroup_file_bytes": memory["file_bytes"],
                        "cgroup_kernel_bytes": memory["kernel_bytes"],
                        "cgroup_working_set_bytes": memory["working_set_bytes"],
                        "cgroup_working_headroom_bytes": headroom,
                        "aggregate_process_tree_rss_bytes": current_process["rss_bytes"],
                        "nvidia_memory_used_mib": sum(
                            item["memory_used_mib"] for item in gpu.get("gpus", [])
                        ) if gpu["query"] == "PASS" else None,
                        "nvidia_utilization_percent": [
                            item["utilization_percent"] for item in gpu.get("gpus", [])
                        ] if gpu["query"] == "PASS" else None,
                    },
                    error=pause_reason,
                )
                last_update = now
            if len(outcomes) < len(children):
                time.sleep(base.GPU_POLL_SECONDS)

        elapsed = time.monotonic() - started
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        start_ticks = baseline_process.get("cpu_ticks")
        end_ticks = last_process.get("cpu_ticks")
        cpu_core_equivalent = None
        if start_ticks is not None and end_ticks is not None and elapsed > 0:
            cpu_core_equivalent = max(
                0.0,
                (int(end_ticks) - int(start_ticks)) / ticks_per_second / elapsed,
            )
        metrics = {
            "started_utc": started_utc,
            "ended_utc": base.utc_now(),
            "wall_seconds": elapsed,
            "lane_return_codes": outcomes,
            "lane_count": self.lane_count,
            "cpu_workers_per_lane": self.workers,
            "prefetch_per_lane": self.prefetch,
            "peak_process_tree_rss_bytes": peak_process_rss or None,
            "peak_cgroup_memory_bytes_shared": peak_cgroup_memory or None,
            "peak_nvidia_memory_used_mib": peak_gpu_used,
            "peak_nvidia_utilization_percent": peak_gpu_util,
            "process_read_bytes_last_observed": last_process.get("read_bytes"),
            "process_write_bytes_last_observed": last_process.get("write_bytes"),
            "cpu_core_equivalent_last_observed": cpu_core_equivalent,
            "memory_events_before": initial_events,
            "memory_events_after": memory_events(),
            "memory_pressure_fallback_requested": memory_pressure,
            "last_gpu_snapshot": last_gpu,
        }
        return outcomes, metrics, pause_reason

    def confirm_release(self, root_pids: list[int]) -> dict[str, Any]:
        roots = set(root_pids)
        while True:
            gpu = base.gpu_snapshot()
            own_visible = [item for item in gpu["compute_processes"] if item["pid"] in roots]
            if gpu["query"] == "PASS" and not own_visible:
                return gpu
            self.update_status(
                phase="CONFIRMING_PARALLEL_GPU_MEMORY_RELEASE",
                gpu_test_hold=self.hold_path.exists(),
                cache_gpu_memory_released=False,
                ready_for_gpu_test=False,
                runner_pids=[],
                current_gpu=gpu,
                error=(
                    "GPU query unavailable; not treating cache VRAM as released"
                    if gpu["query"] != "PASS" else f"cache processes still visible: {own_visible}"
                ),
            )
            time.sleep(base.GPU_POLL_SECONDS)

    def run(self) -> int:
        base.log("parallel controller active; controller itself loads no model")
        while not self.stop_requested:
            baseline = self.waiting_gate()
            children = self.launch()
            roots = [child.pid for child in children.values()]
            self.update_status(
                phase="STARTING_PARALLEL_RUNNERS",
                gpu_test_hold=False,
                cache_gpu_memory_released=False,
                ready_for_gpu_test=False,
                runner_pid=None,
                runner_pids=roots,
                error=None,
            )
            outcomes, metrics, pause_reason = self.monitor(children, baseline)
            released_gpu = self.confirm_release(roots)
            self.children = {}
            cached, missing = self.refresh_counts()
            hold = self.hold_path.exists()
            all_complete = all(code == 0 for code in outcomes.values())
            paused = all(code in (0, base.PAUSED_EXIT_CODE) for code in outcomes.values())

            if all_complete and missing == 0:
                self.update_status(
                    phase="COMPLETE",
                    gpu_test_hold=hold,
                    cache_gpu_memory_released=True,
                    ready_for_gpu_test=hold,
                    runner_pids=[],
                    cached_count=cached,
                    missing_count=missing,
                    newly_created_since_delegation=max(0, cached - self.initial_count),
                    last_parallel_attempt_resources=metrics,
                    current_gpu=released_gpu,
                    error=None,
                )
                base.log(f"parallel cache complete; cached={cached} missing={missing}")
                return 0

            if metrics["memory_pressure_fallback_requested"] and self.lane_count > 1:
                self.memory_fallbacks += 1
                self.lane_count -= 1
                self.update_status(
                    phase="REDUCING_LANES_FOR_MEMORY_SAFETY",
                    gpu_test_hold=hold,
                    cache_gpu_memory_released=True,
                    ready_for_gpu_test=hold,
                    runner_pids=[],
                    cached_count=cached,
                    missing_count=missing,
                    newly_created_since_delegation=max(0, cached - self.initial_count),
                    last_parallel_attempt_resources=metrics,
                    current_gpu=released_gpu,
                    error=pause_reason,
                )
                base.log(
                    f"parallel memory safety fallback to {self.lane_count} lanes; cached={cached}"
                )
                continue

            if self.stop_requested:
                self.update_status(
                    phase="STOPPED_WITH_GPU_RELEASED",
                    gpu_test_hold=hold,
                    cache_gpu_memory_released=True,
                    ready_for_gpu_test=hold,
                    runner_pids=[],
                    cached_count=cached,
                    missing_count=missing,
                    last_parallel_attempt_resources=metrics,
                    current_gpu=released_gpu,
                    error=pause_reason,
                )
                return 130

            if paused and (pause_reason is not None or hold or self.yield_path.exists()):
                phase = "WAITING_GPU_TEST_PRIORITY" if hold else "WAITING_GPU_AVAILABLE"
                self.update_status(
                    phase=phase,
                    gpu_test_hold=hold,
                    cache_gpu_memory_released=True,
                    ready_for_gpu_test=hold,
                    runner_pids=[],
                    cached_count=cached,
                    missing_count=missing,
                    newly_created_since_delegation=max(0, cached - self.initial_count),
                    last_parallel_attempt_resources=metrics,
                    current_gpu=released_gpu,
                    error=pause_reason,
                )
                base.log(f"parallel lanes yielded with VRAM released; reason={pause_reason}")
                continue

            if all_complete and missing > 0:
                base.log(f"parallel snapshot completed with {missing} new/unseen misses; refreshing")
                continue

            self.update_status(
                phase="ERROR",
                gpu_test_hold=hold,
                cache_gpu_memory_released=True,
                ready_for_gpu_test=hold,
                runner_pids=[],
                cached_count=cached,
                missing_count=missing,
                newly_created_since_delegation=max(0, cached - self.initial_count),
                last_parallel_attempt_resources=metrics,
                current_gpu=released_gpu,
                error=f"parallel runners failed closed: {outcomes}; {pause_reason}",
            )
            base.log(f"parallel runners failed closed: {outcomes}")
            return next((code for code in outcomes.values() if code != 0), 1)
        return 130


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    supervisor = ParallelSupervisor(config=args.config, output_dir=args.output_dir)
    lock_path = supervisor.output_dir / "supervisor.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another Anchor-2K supervisor already holds the singleton lock") from None

        def stop(signum: int, frame: Any) -> None:
            del frame
            supervisor.stop_requested = True
            supervisor.request_yield(f"parallel supervisor received signal {signum}")

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop)
        try:
            raise SystemExit(supervisor.run())
        except KeyboardInterrupt:
            supervisor.stop_requested = True
            supervisor.request_yield("parallel supervisor interrupted")
            for child in supervisor.children.values():
                child.wait()
            raise SystemExit(130) from None


if __name__ == "__main__":
    main()
