#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

from common.gpu_monitor import query_gpu
from common.media_utils import (
    atomic_write_json,
    atomic_write_text,
    metadata_lines,
    probe_media,
    select_encoder,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT = ROOT / "input" / "0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4"
RESULT_ROOT = ROOT / "result" / "test"
OUTPUT_ROOT = ROOT / "output" / "test"
FLASH_PYTHON = Path("/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-FlashVSR/bin/python")
SEED_PYTHON = Path("/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-SeedVR2/bin/python")
FLASH_SCRIPT = HERE / "FlashVSR_v1.1" / "upscale.py"
SEED_SCRIPT = HERE / "SeedVR2" / "upscale.py"
FLASH_COMMIT = "cf910c61a60733e610e9c6e8b607f80c3a6c202b"
SEED_COMMIT = "e4de8c24441a67e1b7df56abea10645059bb1185"


class BenchmarkError(RuntimeError):
    pass


class _SignalCancellation(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


class _OneShotSignalHandler:
    """Raise once, then ignore repeated terminal signals during child cleanup."""

    def __init__(self, managed_signals: Sequence[int]) -> None:
        self.managed_signals = tuple(managed_signals)
        self.triggered = False

    def __call__(self, signum: int, _frame: Any) -> None:
        if self.triggered:
            return
        self.triggered = True
        # The shell and orchestrator can share a terminal process group, so the
        # shell trap may forward a signal Python already received. Ignore that
        # duplicate while the first cancellation unwinds and reaps children.
        for managed in self.managed_signals:
            signal.signal(managed, signal.SIG_IGN)
        raise _SignalCancellation(signum)


class _ActiveChildRegistry:
    """Track subprocess groups so cancellation cannot orphan GPU workers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen[str]] = {}
        self._cancelling = False

    def begin_run(self) -> None:
        with self._lock:
            if self._processes:
                raise RuntimeError("cannot reset child cancellation while processes are active")
            self._cancelling = False

    def register(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes[process.pid] = process
            cancelling = self._cancelling
        # Close the race where a sibling fails after Popen but just before this
        # worker registers. A late entrant is terminated before it can continue.
        if cancelling:
            _terminate_process_groups([process], grace_seconds=0.0)

    def unregister(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes.pop(process.pid, None)

    def terminate_all(self, grace_seconds: float = 5.0) -> None:
        with self._lock:
            self._cancelling = True
            processes = list(self._processes.values())
        _terminate_process_groups(processes, grace_seconds=grace_seconds)


_ACTIVE_CHILDREN = _ActiveChildRegistry()


def _process_group_has_live_members(pgid: int) -> bool:
    """Return whether a Linux process group still has a non-zombie member."""
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for stat_path in proc_root.glob("[0-9]*/stat"):
            try:
                stat = stat_path.read_text(encoding="utf-8")
                # comm is parenthesized and may contain whitespace; fields after
                # the final ')' begin with state, ppid, then process-group ID.
                fields = stat.rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != "Z":
                    return True
            except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
                continue
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_groups(
    processes: Sequence[subprocess.Popen[str]], *, grace_seconds: float = 5.0
) -> None:
    groups = {process.pid for process in processes}
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + max(0.0, grace_seconds)
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass

    # Kill the entire original group, not just a still-running direct child.
    # A child may exit on TERM while one of its descendants ignores it.
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    # kill(2) only queues SIGKILL; a descendant can remain runnable for a brief
    # scheduling window after its direct parent has already been reaped. Do not
    # return cancellation control until every group has no live (non-zombie)
    # member, while keeping the cleanup wait bounded.
    extinction_deadline = time.monotonic() + 1.0
    remaining_groups = set(groups)
    while remaining_groups and time.monotonic() < extinction_deadline:
        remaining_groups = {
            pgid for pgid in remaining_groups if _process_group_has_live_members(pgid)
        }
        if remaining_groups:
            for pgid in remaining_groups:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.01)


def _run_concurrently(
    jobs: Sequence[tuple[str, Callable[[], None]]], *, logger: "Logger | None" = None
) -> None:
    """Run jobs together and promptly cancel every child if one job fails."""
    _ACTIVE_CHILDREN.begin_run()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs))
    futures: dict[concurrent.futures.Future[None], str] = {}
    try:
        futures = {executor.submit(job): name for name, job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                if logger is not None:
                    logger.write(f"Concurrent worker {futures[future]} failed: {error}")
                raise
    except BaseException:
        # This outer guard deliberately covers as_completed() itself. SIGINT or
        # SIGTERM can interrupt the main thread while it is blocked there; kill
        # child groups before executor shutdown waits for worker threads.
        _ACTIVE_CHILDREN.terminate_all()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


class Logger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def write(self, message: str, *, console: bool = True) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{stamp}] {message.rstrip()}"
        with self.lock:
            self.handle.write(line + "\n")
            if console:
                print(line, flush=True)

    def close(self) -> None:
        self.handle.close()


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise BenchmarkError(f"cannot read git revision for {path}: {result.stderr.strip()}")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets() -> None:
    required: dict[Path, int] = {
        INPUT: 1,
        ROOT / "models" / "FlashVSR_v1.1" / "checkpoint" / "diffusion_pytorch_model_streaming_dmd.safetensors": 5_000_000_000,
        ROOT / "models" / "FlashVSR_v1.1" / "checkpoint" / "LQ_proj_in.ckpt": 500_000_000,
        ROOT / "models" / "FlashVSR_v1.1" / "checkpoint" / "TCDecoder.ckpt": 150_000_000,
        ROOT / "models" / "FlashVSR_v1.1" / "checkpoint" / "Wan2.1_VAE.pth": 450_000_000,
        ROOT / "models" / "FlashVSR_v1.1" / "checkpoint" / "config.json": 1,
        ROOT / "models" / "FlashVSR_v1.1" / "checkpoint" / "model_index.json": 1,
        ROOT / "models" / "SeedVR2" / "checkpoint" / "seedvr2_ema_3b.pth": 13_000_000_000,
        ROOT / "models" / "SeedVR2" / "checkpoint" / "ema_vae.pth": 900_000_000,
        ROOT / "models" / "SeedVR2" / "checkpoint" / "pos_emb.pt": 1,
        ROOT / "models" / "SeedVR2" / "checkpoint" / "neg_emb.pt": 1,
        FLASH_PYTHON: 1,
        SEED_PYTHON: 1,
        FLASH_SCRIPT: 1,
        SEED_SCRIPT: 1,
    }
    invalid = [
        f"{path} (missing or smaller than {minimum} bytes)"
        for path, minimum in required.items()
        if not path.is_file() or path.stat().st_size < minimum
    ]
    if invalid:
        raise BenchmarkError("required input/model/environment assets are invalid:\n" + "\n".join(invalid))
    flash_repo = ROOT / "models" / "FlashVSR_v1.1" / "repository"
    seed_repo = ROOT / "models" / "SeedVR2" / "repository"
    revisions = {"FlashVSR": _git_head(flash_repo), "SeedVR2": _git_head(seed_repo)}
    expected = {"FlashVSR": FLASH_COMMIT, "SeedVR2": SEED_COMMIT}
    mismatches = [f"{name}: {revisions[name]} != {commit}" for name, commit in expected.items() if revisions[name] != commit]
    if mismatches:
        raise BenchmarkError("official repository revision mismatch: " + "; ".join(mismatches))

    manifests = (
        ROOT / "models" / "FlashVSR_v1.1" / "MODEL_INFO.json",
        ROOT / "models" / "SeedVR2" / "MODEL_INFO.json",
    )
    checksum_errors: list[str] = []
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkError(f"invalid model manifest {manifest_path}: {error}") from error
        checkpoint_root = manifest_path.parent / "checkpoint"
        for filename, expected_digest in manifest.get("checkpoint_sha256", {}).items():
            checkpoint_path = checkpoint_root / filename
            if not checkpoint_path.is_file():
                checksum_errors.append(f"{checkpoint_path}: missing")
                continue
            actual_digest = _sha256(checkpoint_path)
            if actual_digest != expected_digest:
                checksum_errors.append(
                    f"{checkpoint_path}: SHA-256 {actual_digest} != {expected_digest}"
                )
    if checksum_errors:
        raise BenchmarkError("official checkpoint integrity failure:\n" + "\n".join(checksum_errors))


def _input_identity(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _available_cpu_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _staged_cpu_plan(cpu_cores: int) -> dict[str, int]:
    """Scale the tuned 23-core overlap plan without assuming this host size."""
    cores = max(1, int(cpu_cores))
    reserve = 1 if cores > 1 else 0
    usable = max(1, cores - reserve)

    def scaled(numerator: int) -> int:
        # The measured plan uses 22 usable CPUs: encode=12 and Seed=18.
        return max(1, min(usable, int(round(usable * numerator / 22.0))))

    analysis_workers = min(2, usable)
    return {
        "available_cores": cores,
        "reserved_cores": reserve,
        "flash_gpu_threads": usable,
        "flash_encode_threads": scaled(12),
        "seed_threads": scaled(18),
        "analysis_workers": analysis_workers,
        "analysis_threads_per_worker": max(1, usable // analysis_workers),
    }


def _gpu_release_threshold_mb(seed_peak_mb: float, guard_mb: float = 512.0) -> float:
    return max(0.0, float(seed_peak_mb)) + max(0.0, float(guard_mb))


def _thread_environment(base: dict[str, str], threads: int, *, cuda: bool = True) -> dict[str, str]:
    environment = base.copy()
    value = str(max(1, int(threads)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = value
    if not cuda:
        # The staged Flash encoder must not retain or create a CUDA context
        # while SeedVR2 owns the single GPU inference slot.
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment.pop("CUDA_VISIBLE_DEVICE_INDEX", None)
    return environment


def _wait_for_gpu_release(
    required_free_mb: float,
    *,
    logger: Logger,
    timeout_seconds: float = 900.0,
    poll_seconds: float = 5.0,
) -> dict[str, Any]:
    started = time.monotonic()
    last: dict[str, Any] | None = None
    while time.monotonic() - started <= timeout_seconds:
        last = query_gpu(0)
        if float(last["free_vram_mb"]) >= required_free_mb:
            logger.write(
                f"GPU release gate passed: free={last['free_vram_mb']:.0f} MiB, "
                f"required={required_free_mb:.0f} MiB"
            )
            return last
        logger.write(
            f"Waiting for GPU release: free={last['free_vram_mb']:.0f} MiB, "
            f"required={required_free_mb:.0f} MiB"
        )
        time.sleep(max(0.1, poll_seconds))
    available = float(last["free_vram_mb"]) if last else 0.0
    raise BenchmarkError(
        f"GPU did not release enough VRAM within {timeout_seconds:.0f}s: "
        f"free={available:.0f} MiB, required={required_free_mb:.0f} MiB"
    )


def _validate_staged_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BenchmarkError(f"Flash staged inference did not create manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"invalid Flash stage manifest {path}: {error}") from error
    staged_status = None
    if isinstance(manifest, dict):
        staged_status = manifest.get("status")
        if staged_status is None and isinstance(manifest.get("partial_report"), dict):
            staged_status = manifest["partial_report"].get("status")
    if (
        not isinstance(manifest, dict)
        or staged_status != "STAGED"
        or manifest.get("state") != "COMPLETE"
    ):
        raise BenchmarkError(
            "Flash stage manifest is not ready for encoding: "
            f"status={staged_status!r}, state={manifest.get('state') if isinstance(manifest, dict) else None!r}"
        )
    return manifest


def _extract_peak(report: dict[str, Any]) -> float:
    candidates = [report.get("peak_vram_mb")]
    gpu = report.get("gpu", {})
    for key in (
        "peak_process_gpu_memory_mb",
        "peak_process_gpu_mib",
        "peak_torch_reserved_mb",
        "peak_torch_reserved_mib",
    ):
        candidates.append(gpu.get(key))
    numeric = [float(value) for value in candidates if isinstance(value, (int, float))]
    if not numeric:
        raise BenchmarkError(f"probe did not report a VRAM peak: {report}")
    return max(numeric)


def _run_logged(
    name: str,
    command: Sequence[str],
    *,
    environment: dict[str, str],
    model_log: Path,
    logger: Logger,
) -> int:
    model_log.parent.mkdir(parents=True, exist_ok=True)
    logger.write(f"{name} command: {shlex.join(command)}")
    with model_log.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"\n$ {shlex.join(command)}\n")
        process = subprocess.Popen(
            list(command),
            cwd=HERE,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        _ACTIVE_CHILDREN.register(process)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                logger.write(f"[{name}] {line.rstrip()}")
            return_code = process.wait()
        except BaseException:
            _terminate_process_groups([process])
            raise
        finally:
            _ACTIVE_CHILDREN.unregister(process)
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None:
                process.wait()
    logger.write(f"{name} exit code: {return_code}")
    return return_code


def _failure_report(model: str, path: Path, reason: str) -> None:
    if path.is_file():
        return
    atomic_write_json(path, {"model": model, "status": "FAIL", "error": reason})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete pretrained 4K-to-16K comparison")
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument(
        "--parallel-mode",
        choices=("auto", "parallel", "sequential"),
        default=os.environ.get("PARALLEL_MODE", "auto"),
    )
    args = parser.parse_args()
    scratch = Path(args.scratch_root).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    logger = Logger(RESULT_ROOT / "run_all.log")
    _ACTIVE_CHILDREN.begin_run()
    previous_handlers: dict[int, Any] = {}

    if threading.current_thread() is threading.main_thread():
        handled_signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            handled_signals.append(signal.SIGHUP)
        handle_signal = _OneShotSignalHandler(handled_signals)
        for signum in handled_signals:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)
    input_identity: tuple[int, int, int] | None = None
    try:
        logger.write(f"Benchmark root: {ROOT}")
        logger.write(f"Requested PARALLEL_MODE={args.parallel_mode}")

        for stale_comparison in (RESULT_ROOT / "comparison.md", RESULT_ROOT / "comparison.json"):
            stale_comparison.unlink(missing_ok=True)
        for directory in (RESULT_ROOT / "FlashVSR_v1.1", RESULT_ROOT / "SeedVR2"):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "run.log").write_text("", encoding="utf-8")
            for report_name in ("report.json", "report.md"):
                (directory / report_name).unlink(missing_ok=True)
        for model, suffix in (("FlashVSR_v1.1", "FlashVSR_v1.1"), ("SeedVR2", "SeedVR2")):
            output_dir = OUTPUT_ROOT / model
            output_dir.mkdir(parents=True, exist_ok=True)
            for old in output_dir.glob(f"{INPUT.stem}_{suffix}_x4.*"):
                old.unlink(missing_ok=True)

        verify_assets()
        input_identity = _input_identity(INPUT)
        metadata = probe_media(INPUT, count_frames=True)
        for line in metadata_lines(metadata):
            logger.write(line)
        gpu_before = query_gpu(0)
        cpu_cores = _available_cpu_cores()
        staged_cpu = _staged_cpu_plan(cpu_cores)
        cpu_reserve = staged_cpu["reserved_cores"]
        usable_cpu = max(1, cpu_cores - cpu_reserve)
        logger.write(
            f"Hardware: GPU={gpu_before['name']}, total={gpu_before['total_vram_mb']:.0f} MiB, "
            f"free={gpu_before['free_vram_mb']:.0f} MiB, CPU cores={cpu_cores}, reserve={cpu_reserve}"
        )

        selected_encoder, encoder_attempts = select_encoder(
            metadata.expected_x4_width,
            metadata.expected_x4_height,
            threads=max(1, min(16, usable_cpu)),
            scratch_dir=scratch / "encoder_preflight",
            preferred="auto",
        )
        logger.write(
            f"Exact {metadata.expected_x4_width}x{metadata.expected_x4_height} encoder: "
            f"{selected_encoder['name']} ({selected_encoder['extension']}); attempts={encoder_attempts}"
        )

        base_environment = os.environ.copy()
        base_environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": "0",
                "CUDA_VISIBLE_DEVICE_INDEX": "0",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        import_commands = {
            "FlashVSR import": [
                str(FLASH_PYTHON),
                "-c",
                (
                    "import sys; "
                    f"sys.path[:0]=[{str(ROOT / 'models' / 'FlashVSR_v1.1' / 'repository')!r},"
                    f"{str(ROOT / 'models' / 'FlashVSR_v1.1' / 'repository' / 'examples' / 'WanVSR')!r}]; "
                    "import torch, block_sparse_attn, diffsynth; print(torch.__version__, torch.version.cuda)"
                ),
            ],
            "SeedVR2 import": [
                str(SEED_PYTHON),
                "-c",
                (
                    "import sys; "
                    f"sys.path.insert(0,{str(ROOT / 'models' / 'SeedVR2' / 'repository')!r}); "
                    "import torch, apex; from projects.video_diffusion_sr.infer import VideoDiffusionInfer; "
                    "print(torch.__version__, torch.version.cuda)"
                ),
            ],
        }
        for name, command in import_commands.items():
            code = _run_logged(
                name,
                command,
                environment=base_environment,
                model_log=RESULT_ROOT / ("FlashVSR_v1.1" if name.startswith("Flash") else "SeedVR2") / "run.log",
                logger=logger,
            )
            if code:
                raise BenchmarkError(f"{name} failed; see model run.log")

        probe_specs = {
            "FlashVSR_v1.1": {
                "python": FLASH_PYTHON,
                "script": FLASH_SCRIPT,
                "json": scratch / "flash_probe.json",
                "result": RESULT_ROOT / "FlashVSR_v1.1",
                "output": OUTPUT_ROOT / "FlashVSR_v1.1",
            },
            "SeedVR2": {
                "python": SEED_PYTHON,
                "script": SEED_SCRIPT,
                "json": scratch / "seed_probe.json",
                "result": RESULT_ROOT / "SeedVR2",
                "output": OUTPUT_ROOT / "SeedVR2",
            },
        }
        probe_reports: dict[str, dict[str, Any]] = {}
        probe_threads = max(1, min(16, usable_cpu))
        for model, spec in probe_specs.items():
            command = [
                str(spec["python"]),
                str(spec["script"]),
                "--input",
                str(INPUT),
                "--output-dir",
                str(spec["output"]),
                "--result-dir",
                str(spec["result"]),
                "--scratch-dir",
                str(scratch / f"probe_{model}"),
                "--probe-only",
                "--probe-json",
                str(spec["json"]),
                "--ffmpeg-threads",
                str(probe_threads),
            ]
            code = _run_logged(
                f"{model} VRAM probe",
                command,
                environment=base_environment,
                model_log=spec["result"] / "run.log",
                logger=logger,
            )
            if code or not spec["json"].is_file():
                raise BenchmarkError(f"{model} VRAM probe failed")
            probe_reports[model] = json.loads(spec["json"].read_text(encoding="utf-8"))
            spec["json"].unlink(missing_ok=True)
            spec["json"].with_suffix(".md").unlink(missing_ok=True)

        flash_peak = _extract_peak(probe_reports["FlashVSR_v1.1"])
        seed_peak = _extract_peak(probe_reports["SeedVR2"])
        gpu_decision = query_gpu(0)
        safety_margin = max(gpu_decision["total_vram_mb"] * 0.18, 8192.0)
        combined = flash_peak + seed_peak
        safe_parallel = combined + safety_margin <= gpu_decision["free_vram_mb"]
        if args.parallel_mode == "auto":
            parallel_count = 2 if safe_parallel else 1
            selected_mode = "parallel" if parallel_count == 2 else "staged_overlap"
            comparator = "<=" if safe_parallel else ">"
            reason = (
                f"combined estimated peak {combined:.0f} MiB + safety margin {safety_margin:.0f} MiB "
                f"{comparator} currently available {gpu_decision['free_vram_mb']:.0f} MiB; "
                + ("full model concurrency is safe" if safe_parallel else "serial GPU stages with CPU/GPU overlap selected")
            )
        elif args.parallel_mode == "parallel":
            # "parallel" requests concurrency but never bypasses the measured
            # safety gate.  The absolute no-OOM requirement takes precedence
            # over forcing two unsafe processes onto the same device.
            parallel_count = 2 if safe_parallel else 1
            selected_mode = "parallel" if safe_parallel else "sequential"
            comparator = "<=" if safe_parallel else ">"
            reason = (
                "explicit PARALLEL_MODE=parallel request; "
                f"combined estimated peak {combined:.0f} MiB + safety margin "
                f"{safety_margin:.0f} MiB {comparator} currently available "
                f"{gpu_decision['free_vram_mb']:.0f} MiB; "
                + ("request accepted" if safe_parallel else "unsafe request downgraded to sequential")
            )
        else:
            parallel_count = 1
            selected_mode = "sequential"
            reason = (
                f"explicit PARALLEL_MODE=sequential override; estimated combined peak {combined:.0f} MiB"
            )
        logger.write(
            f"Selected {selected_mode}: FlashVSR={flash_peak:.0f} MiB, SeedVR2={seed_peak:.0f} MiB, "
            f"margin={safety_margin:.0f} MiB, available={gpu_decision['free_vram_mb']:.0f} MiB. {reason}"
        )
        logger.write(
            "Staged CPU plan: "
            f"reserve={staged_cpu['reserved_cores']}, "
            f"Flash GPU={staged_cpu['flash_gpu_threads']} threads, "
            f"Flash encode={staged_cpu['flash_encode_threads']} threads, "
            f"Seed={staged_cpu['seed_threads']} threads, "
            f"analysis={staged_cpu['analysis_workers']}x"
            f"{staged_cpu['analysis_threads_per_worker']} threads"
        )

        threads_per_process = max(1, usable_cpu // parallel_count)
        process_environment = _thread_environment(base_environment, threads_per_process)
        encode_preflight_threads = (
            staged_cpu["flash_encode_threads"]
            if selected_mode == "staged_overlap"
            else threads_per_process
        )

        # Re-run the exact-resolution encoder probe with the thread count that
        # the model processes will actually receive.  The early capability
        # probe intentionally caps its cost at 16 threads, while a sequential
        # run may use more CPU workers.  Keeping this second gate ahead of the
        # expensive neural inference prevents a late encoder-only failure.
        selected_encoder, final_encoder_attempts = select_encoder(
            metadata.expected_x4_width,
            metadata.expected_x4_height,
            threads=encode_preflight_threads,
            scratch_dir=scratch / "encoder_final_preflight",
            preferred=selected_encoder["name"],
        )
        logger.write(
            f"Final exact-resolution encoder check at {encode_preflight_threads} requested threads: "
            f"{selected_encoder['name']} ({selected_encoder['extension']}); "
            f"attempts={final_encoder_attempts}"
        )

        flash_probe = probe_reports["FlashVSR_v1.1"]
        seed_probe = probe_reports["SeedVR2"]
        flash_stage_manifest = scratch / "flash_full" / "stage_manifest.json"
        flash_stage_manifest.unlink(missing_ok=True)
        commands = {
            "FlashVSR_v1.1": [
                str(FLASH_PYTHON), str(FLASH_SCRIPT), "--input", str(INPUT),
                "--output-dir", str(OUTPUT_ROOT / "FlashVSR_v1.1"),
                "--result-dir", str(RESULT_ROOT / "FlashVSR_v1.1"),
                "--scratch-dir", str(scratch / "flash_full"),
                "--tile-size", str(flash_probe["recommended_tile_size"]),
                "--tile-overlap", str(flash_probe["recommended_tile_overlap"]),
                "--ffmpeg-threads", str(threads_per_process), "--encoder", selected_encoder["name"],
            ],
            "SeedVR2": [
                str(SEED_PYTHON), str(SEED_SCRIPT), "--input", str(INPUT),
                "--output-dir", str(OUTPUT_ROOT / "SeedVR2"),
                "--result-dir", str(RESULT_ROOT / "SeedVR2"),
                "--scratch-dir", str(scratch / "seed_full"),
                "--tile-size", str(seed_probe["recommended_tile_size"]),
                "--tile-overlap", str(seed_probe["recommended_tile_overlap"]),
                "--temporal-window", str(seed_probe["recommended_temporal_window"]),
                "--temporal-overlap", str(seed_probe["recommended_temporal_overlap"]),
                "--ffmpeg-threads", str(threads_per_process), "--encoder", selected_encoder["name"],
            ],
        }
        benchmark_started = time.perf_counter()
        return_codes: dict[str, int] = {}
        stage_durations: dict[str, float] = {}

        def run_model(
            model: str,
            *,
            command: Sequence[str] | None = None,
            environment: dict[str, str] | None = None,
            stage_name: str | None = None,
            log_name: str | None = None,
        ) -> None:
            started = time.perf_counter()
            code = _run_logged(
                log_name or f"{model} full inference",
                command or commands[model],
                environment=environment or process_environment,
                model_log=RESULT_ROOT / model / "run.log",
                logger=logger,
            )
            return_codes[model] = code
            if stage_name:
                stage_durations[stage_name] = time.perf_counter() - started
            if code:
                raise BenchmarkError(f"{log_name or model} exited with code {code}")

        if selected_mode == "staged_overlap":
            flash_gpu_command = [
                *commands["FlashVSR_v1.1"][:-4],
                "--ffmpeg-threads", str(staged_cpu["flash_gpu_threads"]),
                "--encoder", selected_encoder["name"],
                "--stage-mode", "inference",
                "--stage-manifest", str(flash_stage_manifest),
            ]
            flash_gpu_environment = _thread_environment(
                base_environment, staged_cpu["flash_gpu_threads"]
            )
            run_model(
                "FlashVSR_v1.1",
                command=flash_gpu_command,
                environment=flash_gpu_environment,
                stage_name="flash_gpu_inference_seconds",
                log_name="FlashVSR_v1.1 staged GPU inference",
            )
            if return_codes["FlashVSR_v1.1"]:
                raise BenchmarkError(
                    f"FlashVSR staged inference exited with code {return_codes['FlashVSR_v1.1']}"
                )
            _validate_staged_manifest(flash_stage_manifest)

            release_guard_mb = 512.0
            release_required_mb = _gpu_release_threshold_mb(seed_peak, release_guard_mb)
            release_timeout = float(os.environ.get("GPU_RELEASE_TIMEOUT_SECONDS", "900"))
            release_started = time.perf_counter()
            _wait_for_gpu_release(
                release_required_mb,
                logger=logger,
                timeout_seconds=release_timeout,
            )
            stage_durations["gpu_release_wait_seconds"] = time.perf_counter() - release_started

            flash_encode_command = [
                *commands["FlashVSR_v1.1"][:-4],
                "--ffmpeg-threads", str(staged_cpu["flash_encode_threads"]),
                "--encoder", selected_encoder["name"],
                "--stage-mode", "encode",
                "--stage-manifest", str(flash_stage_manifest),
            ]
            seed_command = commands["SeedVR2"].copy()
            seed_command[seed_command.index("--ffmpeg-threads") + 1] = str(staged_cpu["seed_threads"])
            flash_encode_environment = _thread_environment(
                base_environment, staged_cpu["flash_encode_threads"], cuda=False
            )
            seed_environment = _thread_environment(base_environment, staged_cpu["seed_threads"])

            # Only the Seed worker owns CUDA during this overlap. Flash consumes
            # the staged frame manifest in a CPU-only encoder process.
            overlap_started = time.perf_counter()
            jobs = [
                (
                    "FlashVSR-encode",
                    lambda: run_model(**{
                        "model": "FlashVSR_v1.1",
                        "command": flash_encode_command,
                        "environment": flash_encode_environment,
                        "stage_name": "flash_cpu_encode_seconds",
                        "log_name": "FlashVSR_v1.1 staged CPU encode",
                    }),
                ),
                (
                    "SeedVR2-inference",
                    lambda: run_model(**{
                        "model": "SeedVR2",
                        "command": seed_command,
                        "environment": seed_environment,
                        "stage_name": "seed_full_inference_seconds",
                        "log_name": "SeedVR2 full inference",
                    }),
                ),
            ]
            _run_concurrently(jobs, logger=logger)
            stage_durations["flash_encode_seed_overlap_wall_seconds"] = (
                time.perf_counter() - overlap_started
            )
        elif parallel_count == 2:
            _run_concurrently(
                [(model, lambda model=model: run_model(model)) for model in commands],
                logger=logger,
            )
        else:
            for model in commands:
                run_model(model)
        whole_benchmark_wall = time.perf_counter() - benchmark_started
        logger.write(f"Whole model benchmark wall time: {whole_benchmark_wall:.3f} sec")

        for model, code in return_codes.items():
            if code:
                _failure_report(model, RESULT_ROOT / model / "report.json", f"full inference exited with code {code}")

        execution = {
            "hardware": {
                "gpu_name": gpu_before["name"],
                "gpu_total_vram_mb": gpu_before["total_vram_mb"],
                "gpu_available_before_mb": gpu_before["free_vram_mb"],
                "gpu_available_at_decision_mb": gpu_decision["free_vram_mb"],
                "cpu_cores": cpu_cores,
                "cpu_reserved_cores": cpu_reserve,
                "cpu_threads_per_process": threads_per_process,
                "cpu_plan": staged_cpu,
            },
            "requested_parallel_mode": args.parallel_mode,
            "selected_parallel_count": parallel_count,
            "selected_mode": selected_mode,
            "selection_reason": reason,
            "safety_margin_mb": safety_margin,
            "probe": {
                "FlashVSR_v1.1_peak_mb": flash_peak,
                "SeedVR2_peak_mb": seed_peak,
                "combined_peak_mb": combined,
                "FlashVSR_v1.1": flash_probe,
                "SeedVR2": seed_probe,
            },
            "whole_benchmark_wall_seconds": whole_benchmark_wall,
            "stage_durations_seconds": stage_durations,
            "overlap_mode": selected_mode,
            "gpu_release_gate": {
                "seed_probe_peak_mb": seed_peak,
                "guard_mb": 512.0,
                "required_free_mb": _gpu_release_threshold_mb(seed_peak),
            },
            "encoder_preflight": {
                "selected": selected_encoder,
                "attempts": encoder_attempts,
                "final_thread_count": encode_preflight_threads,
                "final_attempts": final_encoder_attempts,
                "exact_resolution": (
                    f"{metadata.expected_x4_width}x{metadata.expected_x4_height}"
                ),
            },
        }
        execution_path = scratch / "execution_summary.json"
        atomic_write_json(execution_path, execution)
        analysis_command = [
            sys.executable,
            str(HERE / "analyze_results.py"),
            "--input",
            str(INPUT),
            "--result-root",
            str(RESULT_ROOT),
            "--execution-json",
            str(execution_path),
            "--ffmpeg-threads",
            str(usable_cpu),
        ]
        analysis_code = _run_logged(
            "result analysis",
            analysis_command,
            environment=base_environment,
            model_log=RESULT_ROOT / "run_all.log.analysis.tmp",
            logger=logger,
        )
        (RESULT_ROOT / "run_all.log.analysis.tmp").unlink(missing_ok=True)
        if input_identity is None or _input_identity(INPUT) != input_identity:
            raise BenchmarkError("source input file identity changed during benchmark")
        if any(return_codes.values()) or analysis_code:
            raise BenchmarkError(
                f"benchmark finished with failures: model_codes={return_codes}, analysis_code={analysis_code}"
            )
        logger.write("Benchmark and automatic analysis completed successfully")
        return 0
    except _SignalCancellation as error:
        logger.write(f"CANCELLED: signal {error.signum}; terminating active child groups")
        return 128 + error.signum
    except Exception as error:
        logger.write(f"FATAL: {type(error).__name__}: {error}")
        logger.write(traceback.format_exc())
        # Preserve an explicit machine- and human-readable report for failures
        # that occur before normal result analysis (for example, when no
        # installed codec can encode the exact 4x dimensions).  Never overwrite
        # a richer comparison already produced by the analyzer.
        failure = {
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
            "requested_parallel_mode": args.parallel_mode,
        }
        comparison_json = RESULT_ROOT / "comparison.json"
        comparison_markdown = RESULT_ROOT / "comparison.md"
        if not comparison_json.exists():
            atomic_write_json(comparison_json, failure)
        if not comparison_markdown.exists():
            atomic_write_text(
                comparison_markdown,
                "# 4K → 16K Upscaler Test\n\n"
                f"- Result: **FAIL**\n- Reason: `{failure['error']}`\n",
            )
        return 1
    finally:
        _ACTIVE_CHILDREN.terminate_all()
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
