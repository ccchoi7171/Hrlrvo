from __future__ import annotations

import os
import re
import subprocess
import threading
from dataclasses import dataclass, asdict
from typing import Any


class GPUError(RuntimeError):
    pass


def _number(value: str) -> float:
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", value)
    if not match:
        raise ValueError(value)
    return float(match.group(0))


def _optional_number(value: str) -> float | None:
    try:
        return _number(value)
    except ValueError:
        return None


def query_gpu(index: int = 0) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise GPUError(f"nvidia-smi failed: {result.stderr.strip()}")
    fields = [field.strip() for field in result.stdout.strip().splitlines()[0].split(",")]
    if len(fields) != 6:
        raise GPUError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return {
        "index": index,
        "name": fields[0],
        "total_vram_mb": _number(fields[1]),
        "used_vram_mb": _number(fields[2]),
        "free_vram_mb": _number(fields[3]),
        "utilization_percent": _optional_number(fields[4]),
        "utilization_available": _optional_number(fields[4]) is not None,
        "driver_version": fields[5],
    }


def query_process_gpu_memory(pid: int, index: int = 0) -> float:
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    if result.returncode != 0:
        return 0.0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            process_id = int(fields[0])
            memory = _number(fields[1])
        except (ValueError, IndexError):
            continue
        if process_id == pid:
            return memory
    return 0.0


@dataclass
class GPUSamples:
    gpu_name: str
    gpu_total_vram_mb: float
    gpu_available_before_mb: float
    peak_process_gpu_memory_mb: float = 0.0
    peak_total_gpu_used_memory_mb: float = 0.0
    average_gpu_utilization_percent: float | None = None
    peak_gpu_utilization_percent: float | None = None
    samples: int = 0
    utilization_samples: int = 0
    utilization_unavailable_reason: str | None = None
    sampling_interval_seconds: float = 0.75
    peak_torch_allocated_mb: float = 0.0
    peak_torch_reserved_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPUMonitor:
    def __init__(self, *, pid: int | None = None, index: int = 0, interval: float = 0.75) -> None:
        self.pid = pid or os.getpid()
        self.index = index
        self.interval = max(0.1, interval)
        initial = query_gpu(index)
        self.samples = GPUSamples(
            gpu_name=initial["name"],
            gpu_total_vram_mb=initial["total_vram_mb"],
            gpu_available_before_mb=initial["free_vram_mb"],
            sampling_interval_seconds=self.interval,
        )
        self._utilization_sum = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        try:
            gpu = query_gpu(self.index)
            process_memory = query_process_gpu_memory(self.pid, self.index)
        except Exception:
            return
        self.samples.peak_process_gpu_memory_mb = max(
            self.samples.peak_process_gpu_memory_mb, process_memory
        )
        self.samples.peak_total_gpu_used_memory_mb = max(
            self.samples.peak_total_gpu_used_memory_mb, float(gpu["used_vram_mb"])
        )
        utilization = gpu["utilization_percent"]
        if utilization is not None:
            current_peak = self.samples.peak_gpu_utilization_percent
            self.samples.peak_gpu_utilization_percent = max(
                current_peak if current_peak is not None else utilization,
                utilization,
            )
            self._utilization_sum += utilization
            self.samples.utilization_samples += 1
        else:
            self.samples.utilization_unavailable_reason = (
                "nvidia-smi utilization.gpu returned an unavailable value"
            )
        self.samples.samples += 1

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval)
        self._sample_once()

    def start(self) -> "GPUMonitor":
        if self._thread is not None:
            return self
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 3))
        if self.samples.utilization_samples:
            self.samples.average_gpu_utilization_percent = (
                self._utilization_sum / self.samples.utilization_samples
            )
        try:
            import torch

            if torch.cuda.is_available():
                self.samples.peak_torch_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
                self.samples.peak_torch_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
        except Exception:
            pass
        return self.samples.to_dict()

    def __enter__(self) -> "GPUMonitor":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
