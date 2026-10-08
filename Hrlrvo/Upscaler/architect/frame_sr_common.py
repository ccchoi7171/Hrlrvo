"""Shared streaming benchmark runner for frame-independent x4 SR models."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from common.gpu_monitor import GPUMonitor
from common.media_utils import (
    RawVideoEncoder,
    atomic_publish_file,
    atomic_write_json,
    decode_selected_frames,
    probe_media,
    select_encoder,
)

SCALE = 4
MICRO_FRAMES = 21
MINI_FRAMES = 29


class FrameSRBackend(Protocol):
    """One loaded, deterministic x4 image-SR model."""

    model_details: dict[str, Any]

    def upscale_rgb(self, frame: Any) -> tuple[Any, dict[str, float]]:
        """Return uint8 RGB x4 output and neural/merge timing components."""


def benchmark_frame_range(
    mode: str, total_frames: int, start_frame: int, max_frames: int | None
) -> tuple[int, int]:
    if mode not in {"micro", "mini", "full"}:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    if not 0 <= start_frame < total_frames:
        raise ValueError(f"--start-frame must be in [0, {total_frames - 1}]")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    default = {"micro": MICRO_FRAMES, "mini": MINI_FRAMES, "full": total_frames - start_frame}[mode]
    count = min(default, total_frames - start_frame)
    if max_frames is not None:
        count = min(count, max_frames)
    return start_frame, count


def spatial_tile_count(width: int, height: int, tile_size: int) -> int:
    if min(width, height, tile_size) <= 0:
        raise ValueError("dimensions and tile size must be positive")
    return math.ceil(width / tile_size) * math.ceil(height / tile_size)


def verify_pinned_assets(
    repository: Path, expected_commit: str, weight: Path, expected_sha256: str
) -> dict[str, str]:
    """Fail closed before unpickling a checkpoint or importing repo code."""
    if not repository.is_dir() or not weight.is_file():
        raise FileNotFoundError(f"official repository/checkpoint missing: {repository}, {weight}")
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip() if result.returncode == 0 else ""
    if commit != expected_commit:
        raise RuntimeError(f"repository commit mismatch: got {commit!r}, expected {expected_commit}")
    digest = hashlib.sha256()
    with weight.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch: got {actual_sha256}, expected {expected_sha256}"
        )
    return {"repository_commit": commit, "checkpoint_sha256": actual_sha256}


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_input: Path,
    default_output_dir: Path,
    default_result_dir: Path,
    default_overlap: int,
) -> None:
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--result-dir", default=str(default_result_dir))
    parser.add_argument("--report-json")
    parser.add_argument("--benchmark-mode", choices=["micro", "mini", "full"], default="full")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=default_overlap)
    parser.add_argument("--scale", type=int, default=4, choices=[4])
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--ffmpeg-threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument(
        "--encoder",
        default="auto",
        choices=["auto", "ffv1", "libx265", "libsvtav1", "hevc_nvenc", "av1_nvenc"],
    )
    parser.add_argument(
        "--smoke-frame",
        action="store_true",
        help="Process exactly one frame (implementation smoke only; not a benchmark unit)",
    )


def _output_base(args: argparse.Namespace, model_name: str, input_path: Path, start: int, count: int) -> Path:
    mode = "smoke" if args.smoke_frame else args.benchmark_mode
    stem = f"{input_path.stem}_{model_name}_x4_{mode}_f{start:06d}_n{count}"
    return Path(args.output_dir).resolve() / stem


def _report_path(args: argparse.Namespace) -> Path:
    if args.report_json:
        return Path(args.report_json).resolve()
    mode = "smoke" if args.smoke_frame else args.benchmark_mode
    return Path(args.result_dir).resolve() / f"benchmark_{mode}.json"


def run_frame_sr(
    args: argparse.Namespace,
    *,
    model_name: str,
    load_backend: Callable[[argparse.Namespace], FrameSRBackend],
    preflight_assets: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Decode one frame at a time, run tiled SR, and stream directly into ffmpeg."""
    started = time.perf_counter()
    source = probe_media(args.input, count_frames=True)
    if args.scale != SCALE:
        raise ValueError("official pretrained wrappers support exact native x4 only")
    if args.tile_size <= 0 or args.tile_overlap < 0:
        raise ValueError("tile size must be positive and overlap cannot be negative")
    start, count = benchmark_frame_range(
        args.benchmark_mode, source.frame_count, args.start_frame, args.max_frames
    )
    if args.smoke_frame:
        count = 1
    output_width, output_height = source.width * SCALE, source.height * SCALE
    output_base = _output_base(args, model_name, Path(source.path), start, count)
    scratch = Path(tempfile.mkdtemp(prefix=f"{model_name.lower()}_", dir="/tmp"))
    monitor = GPUMonitor(index=args.gpu_index, interval=args.gpu_sample_interval).start()
    encoder_process: RawVideoEncoder | None = None
    video_only: Path | None = None
    succeeded = False
    timings = {
        "preflight_assets_seconds": 0.0,
        "model_load_seconds": 0.0,
        "neural_inference_seconds": 0.0,
        # Image SR has no learned video/VAE decoder. Input video decode is
        # separate so cross-model `neural + decode` remains comparable.
        "decode_seconds": 0.0,
        "input_decode_seconds": 0.0,
        "merge_seconds": 0.0,
        "encode_seconds": 0.0,
    }
    try:
        preflight_started = time.perf_counter()
        assets = preflight_assets() if preflight_assets else {}
        timings["preflight_assets_seconds"] = time.perf_counter() - preflight_started
        load_started = time.perf_counter()
        backend = load_backend(args)
        timings["model_load_seconds"] = time.perf_counter() - load_started

        selected, attempts = select_encoder(
            output_width,
            output_height,
            threads=args.ffmpeg_threads,
            scratch_dir=scratch,
            preferred=args.encoder,
        )
        video_only = scratch / f"video_only{selected['extension']}"
        sample_source = dataclasses.replace(
            source, frame_count=count, duration=count / source.fps_float, audio_present=False, audio_codec=None
        )
        encoder_process = RawVideoEncoder(
            video_only,
            width=output_width,
            height=output_height,
            fps=source.fps,
            frame_count=count,
            encoder=selected,
            threads=args.ffmpeg_threads,
            source_metadata=sample_source,
        )

        indices = list(range(start, start + count))
        decode_started = time.perf_counter()
        iterator = decode_selected_frames(source.path, indices, threads=args.ffmpeg_threads)
        for frame_number, frame in enumerate(iterator, start=1):
            # The iterator performs lazy ffmpeg reads, so charge time since the last
            # iteration boundary to decode rather than neural inference.
            timings["input_decode_seconds"] += time.perf_counter() - decode_started
            output, components = backend.upscale_rgb(frame)
            if output.shape != (output_height, output_width, 3) or output.dtype.name != "uint8":
                raise RuntimeError(
                    f"backend returned {output.shape}/{output.dtype}; expected "
                    f"{(output_height, output_width, 3)}/uint8"
                )
            timings["neural_inference_seconds"] += float(components["neural_inference_seconds"])
            timings["merge_seconds"] += float(components.get("merge_seconds", 0.0))
            encode_started = time.perf_counter()
            encoder_process.write(output)
            encoder_process.frame_complete()
            timings["encode_seconds"] += time.perf_counter() - encode_started
            print(f"[{model_name}] frame {frame_number}/{count}", flush=True)
            del output, frame
            decode_started = time.perf_counter()
        close_started = time.perf_counter()
        encoder_process.close()
        encoder_process = None
        timings["encode_seconds"] += time.perf_counter() - close_started
        destination = output_base.with_name(output_base.name + selected["extension"])
        atomic_publish_file(video_only, destination)
        video_only = None
        total = time.perf_counter() - started
        timings.update(
            {
                "total_wall_seconds": total,
                "pure_inference_seconds_per_input_frame": timings["neural_inference_seconds"] / count,
                "end_to_end_seconds_per_input_frame": total / count,
                "pure_inference_seconds_per_output_frame": timings["neural_inference_seconds"] / count,
                "end_to_end_seconds_per_output_frame": total / count,
            }
        )
        gpu = monitor.stop()
        report: dict[str, Any] = {
            "model": model_name,
            "status": "PASS",
            "pipeline_mode": "frame_independent_image_sr",
            "frame_independent": True,
            "chunk_boundaries": [],
            "scale": SCALE,
            "input": source.to_dict(),
            "output_path": str(destination),
            "frame_selection": {"mode": args.benchmark_mode, "start_frame": start, "frame_count": count},
            "configuration": {
                "tile_size": args.tile_size,
                "tile_overlap": args.tile_overlap,
                "tile_count_per_frame": spatial_tile_count(source.width, source.height, args.tile_size),
                "total_tile_inferences": spatial_tile_count(source.width, source.height, args.tile_size) * count,
                "native_scale": SCALE,
                "face_enhancer": False,
                "outscale_resize": False,
                "model": backend.model_details,
            },
            "asset_verification": assets,
            "effective_input_frames_processed": count,
            "effective_output_frames_written": count,
            "tile_count": spatial_tile_count(source.width, source.height, args.tile_size),
            "timings": timings,
            "timing": timings,
            "timings_detail": {"input_video_decode_seconds": timings["input_decode_seconds"]},
            "gpu": gpu,
            "encoder": selected["name"],
            "encoder_arguments": selected["arguments"],
            "encoder_probe_attempts": attempts,
        }
        for key, value in timings.items():
            report[key] = value
        report.update(
            {
                "peak_torch_allocated_mib": gpu["peak_torch_allocated_mb"],
                "peak_torch_reserved_mib": gpu["peak_torch_reserved_mb"],
                "peak_process_gpu_mib": gpu["peak_process_gpu_memory_mb"],
                "peak_total_gpu_used_mib": gpu["peak_total_gpu_used_memory_mb"],
                "peak_vram_mb": max(gpu["peak_torch_reserved_mb"], gpu["peak_process_gpu_memory_mb"]),
            }
        )
        succeeded = True
        return report
    finally:
        if not succeeded:
            with contextlib.suppress(Exception):
                monitor.stop()
        if encoder_process is not None:
            encoder_process.abort()
        if video_only is not None:
            video_only.unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)


def main_for_model(
    args: argparse.Namespace,
    *,
    model_name: str,
    load_backend: Callable[[argparse.Namespace], FrameSRBackend],
    preflight_assets: Callable[[], dict[str, str]] | None = None,
) -> int:
    path = _report_path(args)
    try:
        report = run_frame_sr(
            args,
            model_name=model_name,
            load_backend=load_backend,
            preflight_assets=preflight_assets,
        )
        atomic_write_json(path, report)
        print(json.dumps({"status": "PASS", "output_path": report["output_path"], "report_json": str(path)}))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary must emit a durable failure report
        failure = {"model": model_name, "status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        with contextlib.suppress(Exception):
            atomic_write_json(path, failure)
        print(f"ERROR: {failure['error']}", file=sys.stderr, flush=True)
        return 1
