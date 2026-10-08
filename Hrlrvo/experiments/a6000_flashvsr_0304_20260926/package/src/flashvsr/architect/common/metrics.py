from __future__ import annotations

import math
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .media_utils import (
    MediaError,
    VideoMetadata,
    decode_selected_frames,
    probe_media,
    uniform_sample_indices,
)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _finite_psnr(first: np.ndarray, second: np.ndarray) -> float:
    difference = first.astype(np.float32) - second.astype(np.float32)
    mse = float(np.mean(difference * difference))
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _sharpness(image: np.ndarray) -> tuple[float, float]:
    gray = _gray(image).astype(np.float32)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = float(np.mean(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)))
    return laplacian_variance, edge_energy


def validate_technical(
    input_metadata: VideoMetadata,
    output_path: str,
    *,
    scale: int = 4,
) -> dict[str, Any]:
    output = probe_media(output_path, count_frames=True)
    expected_width = input_metadata.width * scale
    expected_height = input_metadata.height * scale
    fps_delta = abs(output.fps_float - input_metadata.fps_float)
    fps_tolerance = max(1e-6, abs(input_metadata.fps_float) * 1e-6)
    duration_delta = abs(output.duration - input_metadata.duration)
    duration_tolerance = max(0.1, 1.0 / input_metadata.fps_float + 0.05)
    checks = {
        "width_exact": output.width == expected_width,
        "height_exact": output.height == expected_height,
        "frame_count_exact": output.frame_count == input_metadata.frame_count,
        "fps_preserved": fps_delta <= fps_tolerance,
        "duration_preserved": duration_delta <= duration_tolerance,
        "audio_preserved": (not input_metadata.audio_present) or output.audio_present,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_resolution": f"{expected_width}x{expected_height}",
        "output_resolution": f"{output.width}x{output.height}",
        "actual_scale_x": output.width / input_metadata.width,
        "actual_scale_y": output.height / input_metadata.height,
        "input_frame_count": input_metadata.frame_count,
        "output_frame_count": output.frame_count,
        "input_fps": input_metadata.fps,
        "output_fps": output.fps,
        "fps_delta": fps_delta,
        "input_duration_seconds": input_metadata.duration,
        "output_duration_seconds": output.duration,
        "duration_delta_seconds": duration_delta,
        "input_audio_present": input_metadata.audio_present,
        "output_audio_present": output.audio_present,
        "output_pixel_format": output.pixel_format,
        "output_codec": output.video_codec,
    }


def analyze_structure(
    input_path: str,
    output_path: str,
    input_metadata: VideoMetadata,
    *,
    ffmpeg_threads: int,
    source_start_frame: int = 0,
) -> dict[str, Any]:
    indices = uniform_sample_indices(input_metadata.frame_count)
    source_indices = [source_start_frame + index for index in indices]
    input_frames = decode_selected_frames(input_path, source_indices, threads=ffmpeg_threads)
    output_frames = decode_selected_frames(
        output_path,
        indices,
        output_width=input_metadata.width,
        output_height=input_metadata.height,
        threads=ffmpeg_threads,
    )
    per_frame: list[dict[str, float | int]] = []
    for index, original, restored in zip(indices, input_frames, output_frames):
        original_gray = _gray(original)
        restored_gray = _gray(restored)
        original_laplacian, original_edge = _sharpness(original)
        restored_laplacian, restored_edge = _sharpness(restored)
        channel_drift = np.mean(restored.astype(np.float32), axis=(0, 1)) - np.mean(
            original.astype(np.float32), axis=(0, 1)
        )
        per_frame.append(
            {
                "frame": index,
                "ssim": float(
                    structural_similarity(original_gray, restored_gray, data_range=255)
                ),
                "psnr_db": _finite_psnr(original, restored),
                "mean_absolute_difference": float(
                    np.mean(np.abs(original.astype(np.float32) - restored.astype(np.float32)))
                ),
                "luminance_drift": float(np.mean(restored_gray) - np.mean(original_gray)),
                "red_drift": float(channel_drift[0]),
                "green_drift": float(channel_drift[1]),
                "blue_drift": float(channel_drift[2]),
                "input_laplacian_variance": original_laplacian,
                "downsampled_output_laplacian_variance": restored_laplacian,
                "input_edge_energy": original_edge,
                "downsampled_output_edge_energy": restored_edge,
                "laplacian_ratio": restored_laplacian / max(original_laplacian, 1e-6),
                "edge_energy_ratio": restored_edge / max(original_edge, 1e-6),
            }
        )
    if len(per_frame) != len(indices):
        raise MediaError(f"structure analysis decoded {len(per_frame)} of {len(indices)} sample pairs")
    laplacian_ratio = _mean([float(item["laplacian_ratio"]) for item in per_frame])
    edge_ratio = _mean([float(item["edge_energy_ratio"]) for item in per_frame])
    oversharpen_warning = bool(
        laplacian_ratio is not None
        and edge_ratio is not None
        and laplacian_ratio > 1.8
        and edge_ratio > 1.35
    )
    return {
        "sample_indices": indices,
        "source_sample_indices": source_indices,
        "sample_count": len(indices),
        "downsample_filter": "ffmpeg lanczos",
        "mean_ssim": _mean([float(item["ssim"]) for item in per_frame]),
        "mean_psnr_db": _mean([float(item["psnr_db"]) for item in per_frame]),
        "mean_absolute_difference": _mean(
            [float(item["mean_absolute_difference"]) for item in per_frame]
        ),
        "mean_absolute_luminance_drift": _mean(
            [abs(float(item["luminance_drift"])) for item in per_frame]
        ),
        "mean_laplacian_ratio": laplacian_ratio,
        "mean_edge_energy_ratio": edge_ratio,
        "oversharpen_or_halo_warning": oversharpen_warning,
        "per_frame": per_frame,
    }


def _iter_scaled_frames(
    path: str,
    width: int,
    height: int,
    *,
    start_frame: int = 0,
    frame_count: int | None = None,
) -> Iterator[np.ndarray]:
    filters: list[str] = []
    if start_frame or frame_count is not None:
        end = "" if frame_count is None else f":end_frame={start_frame + frame_count}"
        filters.extend([f"trim=start_frame={start_frame}{end}", "setpts=PTS-STARTPTS"])
    filters.append(f"scale={width}:{height}:flags=area")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(Path(path).resolve()),
        "-map",
        "0:v:0",
        "-vf",
        ",".join(filters),
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        raise MediaError("failed to start temporal decoder")
    frame_bytes = width * height * 3
    stderr = ""
    code: int | None = None
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if not raw:
                break
            while len(raw) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(raw))
                if not chunk:
                    break
                raw += chunk
            if len(raw) != frame_bytes:
                raise MediaError("temporal decoder returned a partial frame")
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if code != 0:
        raise MediaError(f"temporal decode failed: {stderr.strip()}")


def analyze_temporal(
    input_path: str,
    output_path: str,
    input_metadata: VideoMetadata,
    *,
    chunk_boundaries: Sequence[int] | None = None,
    analysis_width: int = 640,
    source_start_frame: int = 0,
) -> dict[str, Any]:
    width = min(analysis_width, input_metadata.width)
    height = max(2, round(input_metadata.height * width / input_metadata.width))
    if height % 2:
        height += 1
    input_iterator = _iter_scaled_frames(
        input_path,
        width,
        height,
        start_frame=source_start_frame,
        frame_count=input_metadata.frame_count,
    )
    output_iterator = _iter_scaled_frames(
        output_path,
        width,
        height,
        frame_count=input_metadata.frame_count,
    )
    input_previous: np.ndarray | None = None
    output_previous: np.ndarray | None = None
    input_differences: list[float] = []
    output_differences: list[float] = []
    residual_differences: list[float] = []
    output_to_input_motion_ratios: list[float] = []
    brightness: list[float] = []
    colors: list[np.ndarray] = []
    residual_previous: np.ndarray | None = None
    decoded = 0
    for input_frame, output_frame in zip(input_iterator, output_iterator):
        input_gray = _gray(input_frame).astype(np.float32)
        output_gray = _gray(output_frame).astype(np.float32)
        residual = output_gray - input_gray
        brightness.append(float(np.mean(output_gray)))
        colors.append(np.mean(output_frame.astype(np.float32), axis=(0, 1)))
        if input_previous is not None and output_previous is not None:
            input_difference = float(np.mean(np.abs(input_gray - input_previous)))
            output_difference = float(np.mean(np.abs(output_gray - output_previous)))
            input_differences.append(input_difference)
            output_differences.append(output_difference)
            output_to_input_motion_ratios.append(
                output_difference / max(input_difference, 1e-6)
            )
            if residual_previous is not None:
                residual_differences.append(
                    float(np.mean(np.abs(residual - residual_previous)))
                )
        input_previous = input_gray
        output_previous = output_gray
        residual_previous = residual
        decoded += 1
    if decoded != input_metadata.frame_count:
        raise MediaError(f"temporal analysis decoded {decoded}, expected {input_metadata.frame_count}")
    suspected_duplicates = [
        index + 1
        for index, (input_difference, output_difference) in enumerate(
            zip(input_differences, output_differences)
        )
        if output_difference < 0.35 and input_difference >= 0.7
    ]
    longest_frozen_run = 0
    current_run = 0
    for difference in output_differences:
        if difference < 0.35:
            current_run += 1
            longest_frozen_run = max(longest_frozen_run, current_run)
        else:
            current_run = 0
    brightness_jumps = np.abs(np.diff(np.asarray(brightness, dtype=np.float32)))
    color_jumps = (
        np.linalg.norm(np.diff(np.stack(colors, axis=0), axis=0), axis=1)
        if len(colors) > 1
        else np.empty((0,), dtype=np.float32)
    )

    def outliers(values: np.ndarray, floor: float) -> list[int]:
        if not values.size:
            return []
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        threshold = max(floor, median + 6.0 * max(mad, 1e-6))
        return [int(index + 1) for index, value in enumerate(values) if value > threshold]

    brightness_discontinuities = outliers(brightness_jumps, 8.0)
    color_discontinuities = outliers(color_jumps, 12.0)
    boundaries = sorted(
        {
            int(boundary)
            for boundary in (chunk_boundaries or [])
            if 0 < int(boundary) < input_metadata.frame_count
        }
    )
    median_difference = float(np.median(output_differences)) if output_differences else 0.0
    boundary_risks: list[dict[str, float | int]] = []
    for boundary in boundaries:
        difference = output_differences[boundary - 1]
        local_start = max(0, boundary - 4)
        local_end = min(len(output_differences), boundary + 3)
        neighbors = [
            value
            for idx, value in enumerate(output_differences[local_start:local_end], start=local_start)
            if idx != boundary - 1
        ]
        local_median = float(np.median(neighbors)) if neighbors else median_difference
        if difference > max(5.0, local_median * 3.0):
            boundary_risks.append(
                {
                    "frame": boundary,
                    "difference": difference,
                    "local_median_difference": local_median,
                }
            )
    warning = bool(
        suspected_duplicates
        or longest_frozen_run >= 3
        or brightness_discontinuities
        or color_discontinuities
        or boundary_risks
    )
    return {
        "analysis_resolution": f"{width}x{height}",
        "decoded_frames": decoded,
        "suspected_duplicate_frames": suspected_duplicates,
        "suspected_duplicate_ratio": len(suspected_duplicates) / max(1, decoded - 1),
        "longest_near_frozen_transition_run": longest_frozen_run,
        "frozen_frame_suspected": longest_frozen_run >= 3,
        "brightness_discontinuity_frames": brightness_discontinuities,
        "color_discontinuity_frames": color_discontinuities,
        "chunk_boundaries_checked": boundaries,
        "chunk_boundary_risks": boundary_risks,
        "mean_input_interframe_difference": _mean(input_differences),
        "mean_interframe_difference": _mean(output_differences),
        "mean_output_to_input_motion_ratio": _mean(output_to_input_motion_ratios),
        "mean_temporal_restoration_residual_difference": _mean(residual_differences),
        "status": "WARN" if warning else "PASS",
    }
