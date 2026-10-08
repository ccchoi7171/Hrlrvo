#!/usr/bin/env python3
"""Pipeline stage 02 -> 04: FlashVSR x4 plus original-8K center composite.

The neural portion is delegated to the repository's validated FlashVSR v1.1
implementation in ``Upscaler/architect/FlashVSR_v1.1/upscale.py`` using its
``--stage-mode inference`` path.  That path keeps one continuous causal/KV
timeline per spatial tile and stages lossless tile PNGs on disk.

This module consumes the 24fps outpaint result directly. It performs the
CPU-only final stage: spatial tile feather merge, frame-index-synchronous
original-8K insertion, the same inward smoothstep alpha feather used by
VideoOutpainting, and one exact-resolution 16fps final encode. No full video
and no raw 16K intermediate video are held or written.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence


PIPELINE_SCHEMA_VERSION = 1
EXPECTED_4K = (3840, 2160)
EXPECTED_8K = (7680, 4320)
EXPECTED_16K = (15360, 8640)
EXPECTED_INPUT_4K_FPS = Fraction(24, 1)
EXPECTED_ORIGINAL_8K_FPS = Fraction(16, 1)
EXPECTED_OUTPUT_FPS = Fraction(16, 1)
EXPECTED_FRAMES = 1952
SCALE = 4
CENTER_BBOX = (2160, 3840, 6480, 11520)  # y0, x0, y1, x1
FLASHVSR_MODEL_NAME = "FlashVSR_v1.1"
FLASHVSR_TEMPORAL_MODE = "official_continuous_causal_kv_no_independent_chunks"


class PipelineError(RuntimeError):
    """A deterministic pipeline validation or execution failure."""


def run(
    command: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=check,
        capture_output=capture_output,
        text=text,
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, with_sha256: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if with_sha256:
        result["sha256"] = sha256_file(path)
    return result


def parse_positive_fraction(value: str | None, *, field: str) -> Fraction:
    if not value or value in {"0/0", "N/A"}:
        raise PipelineError(f"ffprobe did not report {field}")
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise PipelineError(f"invalid {field}: {value!r}") from error
    if fraction <= 0:
        raise PipelineError(f"non-positive {field}: {value!r}")
    return fraction


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,"
            "nb_read_frames,duration,pix_fmt,codec_name,color_space,"
            "color_transfer,color_primaries,time_base,start_time:format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = run(command, check=False, capture_output=True)
    if result.returncode:
        raise PipelineError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError) as error:
        raise PipelineError(f"ffprobe returned incomplete metadata for {path}") from error

    avg_rate = parse_positive_fraction(stream.get("avg_frame_rate"), field="avg_frame_rate")
    r_rate = parse_positive_fraction(stream.get("r_frame_rate"), field="r_frame_rate")
    frame_text = stream.get("nb_read_frames") or stream.get("nb_frames")
    try:
        frame_count = int(frame_text)
    except (TypeError, ValueError) as error:
        raise PipelineError(f"ffprobe could not count frames for {path}: {frame_text!r}") from error
    duration_text = stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = float(duration_text)
    except (TypeError, ValueError) as error:
        raise PipelineError(f"ffprobe could not determine duration for {path}") from error
    return {
        "path": str(path.resolve()),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": f"{avg_rate.numerator}/{avg_rate.denominator}",
        "r_frame_rate": f"{r_rate.numerator}/{r_rate.denominator}",
        "fps": float(avg_rate),
        "frame_count": frame_count,
        "duration": duration,
        "pix_fmt": stream.get("pix_fmt"),
        "codec_name": stream.get("codec_name"),
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
        "time_base": stream.get("time_base"),
        "start_time": stream.get("start_time"),
    }


def validate_video_spec(
    metadata: dict[str, Any],
    *,
    label: str,
    resolution: tuple[int, int],
    fps: Fraction,
    frame_count: int,
) -> None:
    actual_resolution = (metadata["width"], metadata["height"])
    if actual_resolution != resolution:
        raise PipelineError(
            f"{label} resolution is {actual_resolution[0]}x{actual_resolution[1]}, "
            f"expected {resolution[0]}x{resolution[1]}"
        )
    expected_rate = f"{fps.numerator}/{fps.denominator}"
    if metadata["avg_frame_rate"] != expected_rate:
        raise PipelineError(
            f"{label} avg_frame_rate is {metadata['avg_frame_rate']}, expected {expected_rate}"
        )
    if metadata["r_frame_rate"] != expected_rate:
        raise PipelineError(
            f"{label} r_frame_rate is {metadata['r_frame_rate']}, expected {expected_rate}"
        )
    if metadata["frame_count"] != frame_count:
        raise PipelineError(
            f"{label} has {metadata['frame_count']} frames, expected {frame_count}"
        )
    expected_duration = frame_count / float(fps)
    tolerance = max(0.05, 1.0 / float(fps))
    if abs(metadata["duration"] - expected_duration) > tolerance:
        raise PipelineError(
            f"{label} duration is {metadata['duration']:.6f}s, "
            f"expected approximately {expected_duration:.6f}s"
        )


def frame_timestamps(path: Path) -> list[float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    result = run(command, check=False, capture_output=True)
    if result.returncode:
        raise PipelineError(f"frame timestamp probe failed for {path}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        timestamps = [float(frame["best_effort_timestamp_time"]) for frame in payload["frames"]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise PipelineError(f"incomplete frame timestamps for {path}") from error
    return timestamps


def validate_frame_index_alignment(
    input_4k: Path,
    original_8k: Path,
    *,
    frame_count: int,
    input_fps: Fraction,
    original_fps: Fraction,
) -> None:
    input_timestamps = frame_timestamps(input_4k)
    original_timestamps = frame_timestamps(original_8k)
    if len(input_timestamps) != frame_count or len(original_timestamps) != frame_count:
        raise PipelineError(
            "timestamp frame counts differ from the required frame count: "
            f"4K={len(input_timestamps)}, 8K={len(original_timestamps)}, expected={frame_count}"
        )
    tolerance = 1e-3
    input_step = 1.0 / float(input_fps)
    original_step = 1.0 / float(original_fps)
    for index, (input_pts, original_pts) in enumerate(zip(input_timestamps, original_timestamps)):
        expected_input_relative = index * input_step
        if abs((input_pts - input_timestamps[0]) - expected_input_relative) > tolerance:
            raise PipelineError(
                f"4K frame {index} is not on the expected {input_fps} CFR timeline"
            )
        expected_original_relative = index * original_step
        if abs((original_pts - original_timestamps[0]) - expected_original_relative) > tolerance:
            raise PipelineError(
                f"8K frame {index} is not on the expected {original_fps} CFR timeline"
            )


def read_outpainting_feather_width(config_path: Path) -> int:
    pattern = re.compile(r"^\s*boundary_blend_width_pixels\s*:\s*(\d+)\s*(?:#.*)?$")
    matches = [
        int(match.group(1))
        for line in config_path.read_text(encoding="utf-8").splitlines()
        if (match := pattern.match(line))
    ]
    if len(matches) != 1:
        raise PipelineError(
            f"expected exactly one video.boundary_blend_width_pixels in {config_path}"
        )
    return matches[0]


def load_reference_module(path: Path) -> Any:
    module_name = "hrlrvo_flashvsr_v1_1_reference"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PipelineError(f"could not load FlashVSR reference implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_media_utils(reference_path: Path) -> Any:
    architect_dir = reference_path.parent.parent
    if str(architect_dir) not in sys.path:
        sys.path.insert(0, str(architect_dir))
    from common import media_utils

    return media_utils


def required_assets(model_root: Path, reference_path: Path) -> list[Path]:
    return [
        reference_path,
        model_root / "repository" / "diffsynth" / "pipelines" / "flashvsr_tiny_long.py",
        model_root / "repository" / "examples" / "WanVSR" / "utils" / "utils.py",
        model_root / "repository" / "examples" / "WanVSR" / "utils" / "TCDecoder.py",
        model_root / "repository" / "examples" / "WanVSR" / "prompt_tensor" / "posi_prompt.pth",
        model_root / "checkpoint" / "diffusion_pytorch_model_streaming_dmd.safetensors",
        model_root / "checkpoint" / "LQ_proj_in.ckpt",
        model_root / "checkpoint" / "TCDecoder.ckpt",
    ]


def stage_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "manifest": work_dir / "flashvsr_stage_manifest.json",
        "request": work_dir / "flashvsr_stage_request.json",
        "report": work_dir / "flashvsr_stage_report.json",
        "result_dir": work_dir / "flashvsr_result",
        "scratch_parent": work_dir / "scratch",
    }


def build_stage_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        str(args.flashvsr_reference),
        "--input",
        str(args.input_4k),
        "--output",
        str(args.output),
        "--result-dir",
        str(paths["result_dir"]),
        "--report-json",
        str(paths["report"]),
        "--scale",
        str(SCALE),
        "--tile-size",
        str(args.tile_size),
        "--tile-overlap",
        str(args.tile_overlap),
        "--min-tile-size",
        str(args.min_tile_size),
        "--sparse-ratio",
        str(args.sparse_ratio),
        "--local-range",
        str(args.local_range),
        "--seed",
        str(args.seed),
        "--gpu-index",
        str(args.gpu_index),
        "--ffmpeg-threads",
        str(args.ffmpeg_threads),
        "--encoder",
        str(args.encoder),
        "--scratch-dir",
        str(paths["scratch_parent"]),
        "--benchmark-mode",
        "full",
        "--stage-mode",
        "inference",
        "--stage-manifest",
        str(paths["manifest"]),
    ]


def stage_request_payload(
    args: argparse.Namespace,
    input_metadata: dict[str, Any],
    original_metadata: dict[str, Any],
    feather_width: int,
) -> dict[str, Any]:
    model_files = required_assets(args.model_root, args.flashvsr_reference)[1:]
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "video_id": args.video_id,
        "input_4k": file_identity(args.input_4k),
        "original_8k": file_identity(args.original_8k),
        "input_4k_metadata": input_metadata,
        "original_8k_metadata": original_metadata,
        "flashvsr_reference": file_identity(args.flashvsr_reference, with_sha256=True),
        "outpainting_config": file_identity(args.outpainting_config, with_sha256=True),
        "model_root": str(args.model_root),
        "model_assets": [file_identity(path) for path in model_files],
        "configuration": {
            "scale": SCALE,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "min_tile_size": args.min_tile_size,
            "sparse_ratio": args.sparse_ratio,
            "local_range": args.local_range,
            "seed": args.seed,
            "encoder": args.encoder,
            "ffmpeg_threads": args.ffmpeg_threads,
            "expected_frames": args.expected_frames,
            "input_4k_fps": (
                f"{EXPECTED_INPUT_4K_FPS.numerator}/{EXPECTED_INPUT_4K_FPS.denominator}"
            ),
            "original_8k_fps": (
                f"{EXPECTED_ORIGINAL_8K_FPS.numerator}/{EXPECTED_ORIGINAL_8K_FPS.denominator}"
            ),
            "output_fps": (
                f"{EXPECTED_OUTPUT_FPS.numerator}/{EXPECTED_OUTPUT_FPS.denominator}"
            ),
            "outpainting_config": str(args.outpainting_config),
            "center_bbox_yx": list(CENTER_BBOX),
            "feather_method": "inward_nearest_edge_smoothstep_alpha",
            "feather_width_pixels": feather_width,
        },
    }


def validate_stage_manifest(
    manifest: dict[str, Any],
    *,
    args: argparse.Namespace,
    input_metadata: dict[str, Any],
    reference: Any,
) -> tuple[list[Any], dict[int, Path], Path]:
    if manifest.get("schema_version") != reference.STAGE_MANIFEST_SCHEMA:
        raise PipelineError("unsupported FlashVSR stage manifest schema")
    if manifest.get("model") != FLASHVSR_MODEL_NAME or manifest.get("state") != "COMPLETE":
        raise PipelineError("FlashVSR stage manifest is not complete")
    if Path(manifest["input"]["path"]).resolve() != args.input_4k:
        raise PipelineError("FlashVSR stage input does not match the requested 4K input")
    if int(manifest.get("processed_input_frames", -1)) != args.expected_frames:
        raise PipelineError("FlashVSR stage frame count does not match the requested full video")
    if Path(manifest.get("output_target", "")).resolve() != args.output:
        raise PipelineError("FlashVSR stage output target does not match the requested final output")

    report = manifest.get("partial_report", {})
    configuration = report.get("configuration", {})
    if configuration.get("temporal_mode") != FLASHVSR_TEMPORAL_MODE:
        raise PipelineError("FlashVSR stage did not use the continuous causal/KV temporal mode")
    if float(configuration.get("sparse_ratio", -1)) != float(args.sparse_ratio):
        raise PipelineError("FlashVSR stage sparse_ratio mismatch")
    if int(configuration.get("local_range", -1)) != args.local_range:
        raise PipelineError("FlashVSR stage local_range mismatch")
    expected_output = report.get("expected_output", {})
    if (expected_output.get("width"), expected_output.get("height")) != EXPECTED_16K:
        raise PipelineError("FlashVSR staged output geometry is not exact 15360x8640")

    scratch = Path(manifest["scratch_path"]).resolve()
    if not scratch.is_dir():
        raise PipelineError(f"FlashVSR stage scratch directory is missing: {scratch}")
    try:
        scratch.relative_to(args.work_dir)
    except ValueError as error:
        raise PipelineError(f"FlashVSR scratch escapes the configured work directory: {scratch}") from error

    tile_size = int(configuration["tile_size"])
    tile_overlap = int(configuration["tile_overlap"])
    expected_tiles = reference.make_tiles(
        input_metadata["width"], input_metadata["height"], tile_size, tile_overlap
    )
    artifacts = manifest.get("tile_artifacts", [])
    if len(artifacts) != len(expected_tiles) or len(artifacts) != int(manifest.get("tile_count", -1)):
        raise PipelineError("FlashVSR stage has an incomplete spatial tile set")

    tiles: list[Any] = []
    tile_dirs: dict[int, Path] = {}
    for expected, artifact in zip(expected_tiles, artifacts):
        tile = reference.Tile(**artifact["tile"])
        if asdict(tile) != asdict(expected):
            raise PipelineError(f"FlashVSR tile geometry mismatch at tile {expected.index}")
        directory = Path(artifact["directory"]).resolve()
        try:
            directory.relative_to(scratch)
        except ValueError as error:
            raise PipelineError(f"tile directory escapes FlashVSR scratch: {directory}") from error
        if not directory.is_dir():
            raise PipelineError(f"staged tile directory is missing: {directory}")
        png_count = sum(1 for _ in directory.glob("*.png"))
        if png_count != args.expected_frames:
            raise PipelineError(
                f"staged tile {tile.index} has {png_count} frames, expected {args.expected_frames}"
            )
        first = directory / "000000000.png"
        last = directory / f"{args.expected_frames - 1:09d}.png"
        if not first.is_file() or not last.is_file():
            raise PipelineError(f"staged tile {tile.index} is missing endpoint frames")
        tiles.append(tile)
        tile_dirs[tile.index] = directory
    return tiles, tile_dirs, scratch


class OriginalFrameReader:
    """Sequential, no-resize RGB decoder for the original 8K video."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        frame_count: int,
        threads: int,
    ) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.frame_bytes = width * height * 3
        self.frames_read = 0
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-threads",
            str(max(1, threads)),
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self.command = command
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if self.process.stdout is None or self.process.stderr is None:
            raise PipelineError("failed to open the original-8K decoder pipes")

    @staticmethod
    def _read_exact(stream: Any, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read(self, expected_index: int) -> Any:
        import numpy as np

        if expected_index != self.frames_read:
            raise PipelineError(
                f"original decoder index mismatch: requested {expected_index}, next is {self.frames_read}"
            )
        raw = self._read_exact(self.process.stdout, self.frame_bytes)
        if len(raw) != self.frame_bytes:
            error = self.process.stderr.read().decode("utf-8", errors="replace")
            raise PipelineError(
                f"original 8K decode ended at frame {expected_index}/{self.frame_count}: {error.strip()}"
            )
        self.frames_read += 1
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)

    def close(self) -> None:
        extra = self.process.stdout.read(1)
        error = self.process.stderr.read().decode("utf-8", errors="replace")
        return_code = self.process.wait()
        if return_code:
            raise PipelineError(f"original 8K decoder failed: {error.strip()}")
        if extra:
            raise PipelineError("original 8K decoder produced more frames than expected")
        if self.frames_read != self.frame_count:
            raise PipelineError(
                f"decoded {self.frames_read} original frames, expected {self.frame_count}"
            )

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


def tile_weight_axes(tile: Any, tiles: list[Any], reference: Any) -> tuple[Any, Any]:
    """Use the reference adapter's exact linear spatial feather profile."""
    same_row = [candidate for candidate in tiles if candidate.y == tile.y]
    same_column = [candidate for candidate in tiles if candidate.x == tile.x]
    left = max(
        (candidate.x + candidate.width - tile.x for candidate in same_row if candidate.x < tile.x),
        default=0,
    ) * SCALE
    right = max(
        (tile.x + tile.width - candidate.x for candidate in same_row if candidate.x > tile.x),
        default=0,
    ) * SCALE
    top = max(
        (candidate.y + candidate.height - tile.y for candidate in same_column if candidate.y < tile.y),
        default=0,
    ) * SCALE
    bottom = max(
        (tile.y + tile.height - candidate.y for candidate in same_column if candidate.y > tile.y),
        default=0,
    ) * SCALE
    y_axis = reference.feather_axis(tile.height * SCALE, top, bottom)
    x_axis = reference.feather_axis(tile.width * SCALE, left, right)
    return y_axis, x_axis


def inward_smoothstep_alpha(
    *,
    anchor_row_start: int,
    anchor_row_end: int,
    anchor_height: int,
    anchor_width: int,
    feather_width: int,
) -> Any:
    """NumPy equivalent of VideoOutpainting's inward nearest-edge smoothstep mask."""
    import numpy as np

    if feather_width <= 0:
        raise ValueError("feather_width must be positive")
    rows = np.arange(anchor_row_start, anchor_row_end, dtype=np.float32)
    columns = np.arange(anchor_width, dtype=np.float32)
    row_distance = np.minimum(rows, np.float32(anchor_height - 1) - rows)
    column_distance = np.minimum(columns, np.float32(anchor_width - 1) - columns)
    distance = np.minimum(row_distance[:, None], column_distance[None, :])
    normalized = np.clip(distance / np.float32(feather_width), 0.0, 1.0)
    return normalized * normalized * (np.float32(3.0) - np.float32(2.0) * normalized)


def composite_original_center_rows(
    generated_rows: Any,
    original_frame: Any,
    *,
    canvas_row_start: int,
    feather_width: int,
) -> None:
    """Composite exact original RGB except in the narrow inward feather band."""
    import numpy as np

    center_y0, center_x0, center_y1, center_x1 = CENTER_BBOX
    canvas_row_end = canvas_row_start + generated_rows.shape[0]
    overlap_y0 = max(canvas_row_start, center_y0)
    overlap_y1 = min(canvas_row_end, center_y1)
    if overlap_y0 >= overlap_y1:
        return

    output_row_start = overlap_y0 - canvas_row_start
    output_row_end = overlap_y1 - canvas_row_start
    anchor_row_start = overlap_y0 - center_y0
    anchor_row_end = overlap_y1 - center_y0
    source = original_frame[anchor_row_start:anchor_row_end]
    region = generated_rows[output_row_start:output_row_end, center_x0:center_x1]
    if source.shape != region.shape:
        raise PipelineError(
            f"original/center row shape mismatch: original={source.shape}, output={region.shape}"
        )

    if feather_width == 0:
        region[...] = source
        return
    alpha = inward_smoothstep_alpha(
        anchor_row_start=anchor_row_start,
        anchor_row_end=anchor_row_end,
        anchor_height=center_y1 - center_y0,
        anchor_width=center_x1 - center_x0,
        feather_width=feather_width,
    )
    blend_pixels = alpha < np.float32(1.0)
    if np.any(blend_pixels):
        weights = alpha[blend_pixels][:, None]
        blended = (
            region[blend_pixels].astype(np.float32) * (np.float32(1.0) - weights)
            + source[blend_pixels].astype(np.float32) * weights
        )
        blended = np.rint(blended).clip(0, 255).astype(np.uint8)
    else:
        blended = None
    region[...] = source
    if blended is not None:
        region[blend_pixels] = blended


def make_vertical_bands(tiles: list[Any], output_height: int) -> list[tuple[int, int, list[Any]]]:
    boundaries = {0, output_height}
    bounds: dict[int, tuple[int, int]] = {}
    for tile in tiles:
        top = tile.y * SCALE
        bottom = (tile.y + tile.height) * SCALE
        boundaries.update((top, bottom))
        bounds[tile.index] = (top, bottom)
    ordered = sorted(boundaries)
    bands: list[tuple[int, int, list[Any]]] = []
    for start, end in zip(ordered, ordered[1:]):
        if start == end:
            continue
        active = [
            tile
            for tile in tiles
            if bounds[tile.index][0] <= start and bounds[tile.index][1] >= end
        ]
        if not active:
            raise PipelineError(f"FlashVSR tiles do not cover output rows [{start}, {end})")
        bands.append((start, end, active))
    if not bands or bands[0][0] != 0 or bands[-1][1] != output_height:
        raise PipelineError("FlashVSR tile bands do not cover the complete output height")
    return bands


def source_video_metadata(media_utils: Any, metadata: dict[str, Any]) -> Any:
    return media_utils.VideoMetadata(
        path=metadata["path"],
        width=metadata["width"],
        height=metadata["height"],
        fps=metadata["avg_frame_rate"],
        fps_float=metadata["fps"],
        frame_count=metadata["frame_count"],
        duration=metadata["duration"],
        pixel_format=metadata.get("pix_fmt"),
        video_codec=metadata.get("codec_name"),
        color_space=metadata.get("color_space"),
        color_transfer=metadata.get("color_transfer"),
        color_primaries=metadata.get("color_primaries"),
        time_base=metadata.get("time_base"),
        audio_present=False,
        audio_codec=None,
    )


def encode_staged_tiles(
    *,
    args: argparse.Namespace,
    reference: Any,
    media_utils: Any,
    input_metadata: dict[str, Any],
    tiles: list[Any],
    tile_dirs: dict[int, Path],
    selected_encoder: dict[str, Any],
    partial_output: Path,
    feather_width: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    cv2.setNumThreads(max(1, args.ffmpeg_threads))
    output_width, output_height = EXPECTED_16K
    bands = make_vertical_bands(tiles, output_height)
    axes = {tile.index: tile_weight_axes(tile, tiles, reference) for tile in tiles}
    tile_bounds = {
        tile.index: (
            tile.y * SCALE,
            tile.x * SCALE,
            (tile.y + tile.height) * SCALE,
            (tile.x + tile.width) * SCALE,
        )
        for tile in tiles
    }
    reader = OriginalFrameReader(
        args.original_8k,
        width=EXPECTED_8K[0],
        height=EXPECTED_8K[1],
        frame_count=args.expected_frames,
        threads=args.ffmpeg_threads,
    )
    encoder = media_utils.RawVideoEncoder(
        partial_output,
        width=output_width,
        height=output_height,
        fps=f"{EXPECTED_OUTPUT_FPS.numerator}/{EXPECTED_OUTPUT_FPS.denominator}",
        frame_count=args.expected_frames,
        encoder=selected_encoder,
        threads=args.ffmpeg_threads,
        source_metadata=source_video_metadata(media_utils, input_metadata),
    )

    started = time.perf_counter()
    try:
        for frame_index in range(args.expected_frames):
            original = reader.read(frame_index)
            image_cache: dict[int, Any] = {}
            rows_written = 0
            for band_start, band_end, active_tiles in bands:
                active_ids = {tile.index for tile in active_tiles}
                for stale_index in tuple(image_cache):
                    if stale_index not in active_ids:
                        del image_cache[stale_index]
                for tile in active_tiles:
                    if tile.index in image_cache:
                        continue
                    image_path = tile_dirs[tile.index] / f"{frame_index:09d}.png"
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise PipelineError(
                            f"missing/corrupt staged tile {tile.index}, frame {frame_index}: {image_path}"
                        )
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    expected_shape = (tile.height * SCALE, tile.width * SCALE, 3)
                    if image.shape != expected_shape:
                        raise PipelineError(
                            f"staged tile {tile.index}, frame {frame_index} has shape "
                            f"{image.shape}, expected {expected_shape}"
                        )
                    image_cache[tile.index] = image

                band_height = band_end - band_start
                accumulator = np.zeros((band_height, output_width, 3), dtype=np.float32)
                normalization = np.zeros((band_height, output_width), dtype=np.float32)
                for tile in active_tiles:
                    tile_y0, tile_x0, _, tile_x1 = tile_bounds[tile.index]
                    local_y0 = band_start - tile_y0
                    local_y1 = band_end - tile_y0
                    y_axis, x_axis = axes[tile.index]
                    weight = y_axis[local_y0:local_y1, None] * x_axis[None, :]
                    pixels = image_cache[tile.index][local_y0:local_y1].astype(np.float32)
                    pixels *= weight[..., None]
                    accumulator[:, tile_x0:tile_x1] += pixels
                    normalization[:, tile_x0:tile_x1] += weight
                if np.any(normalization <= 0):
                    raise PipelineError(
                        f"zero tile normalization in output rows [{band_start}, {band_end})"
                    )
                output_rows = np.rint(
                    accumulator / normalization[..., None]
                ).clip(0, 255).astype(np.uint8)
                composite_original_center_rows(
                    output_rows,
                    original,
                    canvas_row_start=band_start,
                    feather_width=feather_width,
                )
                encoder.write(output_rows)
                rows_written += output_rows.shape[0]
            if rows_written != output_height:
                raise PipelineError(
                    f"frame {frame_index} wrote {rows_written} rows, expected {output_height}"
                )
            encoder.frame_complete()
            if frame_index == 0 or (frame_index + 1) % 10 == 0 or frame_index + 1 == args.expected_frames:
                print(
                    f"[ENCODE] video_id={args.video_id} frame={frame_index + 1}/{args.expected_frames}",
                    flush=True,
                )
        reader.close()
        encoder.close()
    except BaseException:
        reader.abort()
        encoder.abort()
        raise
    return {
        "seconds": time.perf_counter() - started,
        "frames_written": encoder.frames_written,
        "vertical_band_count": len(bands),
        "memory_mode": "one original 8K frame plus bounded 16K row bands",
    }


def validate_final_output(path: Path, *, expected_frames: int) -> dict[str, Any]:
    metadata = probe_video(path)
    validate_video_spec(
        metadata,
        label="final 16K output",
        resolution=EXPECTED_16K,
        fps=EXPECTED_OUTPUT_FPS,
        frame_count=expected_frames,
    )
    return metadata


def validate_existing_result(
    args: argparse.Namespace,
    *,
    request: dict[str, Any],
) -> dict[str, Any]:
    if not args.final_manifest.is_file():
        raise PipelineError(
            f"final output exists without its provenance manifest: {args.output}"
        )
    try:
        manifest = json.loads(args.final_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PipelineError(f"invalid final manifest: {args.final_manifest}") from error
    if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise PipelineError("existing final manifest schema does not match this pipeline")
    if manifest.get("status") != "PASS" or manifest.get("video_id") != args.video_id:
        raise PipelineError("existing final manifest is not a PASS result for this video ID")
    if manifest.get("request") != request:
        raise PipelineError(
            "existing final output provenance does not match the current inputs/model/options"
        )
    current_identity = file_identity(args.output)
    if manifest.get("output_file") != current_identity:
        raise PipelineError("existing final output identity does not match its manifest")
    return validate_final_output(args.output, expected_frames=args.expected_frames)


def print_dry_run(args: argparse.Namespace, feather_width: int) -> None:
    paths = stage_paths(args.work_dir)
    command = build_stage_command(args, paths)
    payload = {
        "mode": "dry-run",
        "video_id": args.video_id,
        "input_4k": str(args.input_4k),
        "input_4k_status": "present" if args.input_4k.is_file() else "missing",
        "original_8k": str(args.original_8k),
        "original_8k_status": "present" if args.original_8k.is_file() else "missing",
        "flashvsr_model_root": str(args.model_root),
        "flashvsr_reference": str(args.flashvsr_reference),
        "video_outpainting_config": str(args.outpainting_config),
        "final_output": str(args.output),
        "final_manifest": str(args.final_manifest),
        "work_dir": str(args.work_dir),
        "stage_manifest": str(paths["manifest"]),
        "expected_input_4k": {
            "resolution": list(EXPECTED_4K),
            "fps": "24/1",
            "frames": args.expected_frames,
        },
        "expected_original_8k": {
            "resolution": list(EXPECTED_8K),
            "fps": "16/1",
            "frames": args.expected_frames,
        },
        "expected_output": {
            "resolution": list(EXPECTED_16K),
            "fps": "16/1",
            "frames": args.expected_frames,
        },
        "center_bbox": {
            "x": [CENTER_BBOX[1], CENTER_BBOX[3]],
            "y": [CENTER_BBOX[0], CENTER_BBOX[2]],
        },
        "feather": {
            "method": "VideoOutpainting inward nearest-edge smoothstep alpha",
            "source_width_4k_pixels": feather_width // SCALE,
            "applied_width_16k_pixels": feather_width,
        },
        "temporal_mode": FLASHVSR_TEMPORAL_MODE,
        "frame_policy": (
            "same decoded-frame index; 4K 24fps input is emitted at 16fps with no "
            "drop, duplication, or interpolation"
        ),
        "intermediate_policy": "lossless staged tile PNGs; no raw 16K intermediate video",
        "final_encode_policy": "single exact-resolution encode after tile merge and center blend",
        "encoder": args.encoder,
        "flashvsr_stage_command": shlex.join(command),
        "cuda_initialized": False,
        "model_loaded": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_self_test() -> None:
    import numpy as np

    alpha = inward_smoothstep_alpha(
        anchor_row_start=0,
        anchor_row_end=6,
        anchor_height=12,
        anchor_width=12,
        feather_width=2,
    )
    assert alpha.shape == (6, 12)
    assert alpha[0, 6] == 0.0
    assert alpha[2, 6] == 1.0
    assert alpha[5, 5] == 1.0

    generated = np.zeros((6, 24, 3), dtype=np.uint8)
    original = np.full((12, 12, 3), 200, dtype=np.uint8)
    old_bbox = globals()["CENTER_BBOX"]
    try:
        globals()["CENTER_BBOX"] = (2, 6, 14, 18)
        composite_original_center_rows(
            generated,
            original,
            canvas_row_start=2,
            feather_width=2,
        )
        assert np.all(generated[0, 6:18] == 0)
        assert np.all(generated[2, 8:16] == 200)
        assert np.all(generated[:, :6] == 0)
        assert np.all(generated[:, 18:] == 0)
    finally:
        globals()["CENTER_BBOX"] = old_bbox
    print("SELF_TEST_PASS alpha=smoothstep center_composite=exact_interior")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise PipelineError(f"required executable is not available: {executable}")
    for path in (args.input_4k, args.original_8k, args.outpainting_config):
        if not path.is_file():
            raise PipelineError(f"required input/config file is missing: {path}")
    missing_assets = [path for path in required_assets(args.model_root, args.flashvsr_reference) if not path.is_file()]
    if missing_assets:
        raise PipelineError("missing FlashVSR asset(s):\n" + "\n".join(map(str, missing_assets)))

    source_feather_width = read_outpainting_feather_width(args.outpainting_config)
    feather_width = source_feather_width * SCALE
    if feather_width != 64:
        raise PipelineError(
            f"the current outpainting config resolves to {feather_width}px at 16K; expected 64px"
        )

    input_metadata = probe_video(args.input_4k)
    original_metadata = probe_video(args.original_8k)
    validate_video_spec(
        input_metadata,
        label="4K outpaint input",
        resolution=EXPECTED_4K,
        fps=EXPECTED_INPUT_4K_FPS,
        frame_count=args.expected_frames,
    )
    validate_video_spec(
        original_metadata,
        label="original 8K input",
        resolution=EXPECTED_8K,
        fps=EXPECTED_ORIGINAL_8K_FPS,
        frame_count=args.expected_frames,
    )
    if input_metadata["frame_count"] != original_metadata["frame_count"]:
        raise PipelineError("4K and original 8K frame counts differ")
    validate_frame_index_alignment(
        args.input_4k,
        args.original_8k,
        frame_count=args.expected_frames,
        input_fps=EXPECTED_INPUT_4K_FPS,
        original_fps=EXPECTED_ORIGINAL_8K_FPS,
    )

    request = stage_request_payload(
        args, input_metadata, original_metadata, feather_width
    )
    if args.output.is_file():
        existing = validate_existing_result(args, request=request)
        print(f"[SKIP] valid final output already exists: {args.output}", flush=True)
        return {"status": "SKIPPED_VALID", "output": existing}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    paths = stage_paths(args.work_dir)
    for directory_key in ("result_dir", "scratch_parent"):
        paths[directory_key].mkdir(parents=True, exist_ok=True)

    reference = load_reference_module(args.flashvsr_reference)
    if Path(reference.MODEL_ROOT).resolve() != args.model_root:
        raise PipelineError(
            f"reference MODEL_ROOT is {Path(reference.MODEL_ROOT).resolve()}, requested {args.model_root}"
        )
    if int(reference.SCALE) != SCALE:
        raise PipelineError(f"reference FlashVSR scale is {reference.SCALE}, expected {SCALE}")

    reused_stage = False
    if paths["manifest"].is_file():
        if not paths["request"].is_file():
            raise PipelineError("stage manifest exists without its request fingerprint")
        saved_request = json.loads(paths["request"].read_text(encoding="utf-8"))
        if saved_request != request:
            raise PipelineError(
                "existing FlashVSR stage request does not match current inputs/options; "
                f"preserved work directory: {args.work_dir}"
            )
        reused_stage = True
        print(f"[RESUME] reusing complete FlashVSR stage: {paths['manifest']}", flush=True)
    else:
        atomic_write_json(paths["request"], request)
        command = build_stage_command(args, paths)
        print("[FLASHVSR] " + shlex.join(command), flush=True)
        result = run(command, check=False, capture_output=False)
        if result.returncode:
            raise PipelineError(f"FlashVSR stage failed with exit code {result.returncode}")
        if not paths["manifest"].is_file():
            raise PipelineError("FlashVSR completed without publishing the stage manifest")

    manifest = reference.read_stage_manifest(paths["manifest"])
    tiles, tile_dirs, stage_scratch = validate_stage_manifest(
        manifest,
        args=args,
        input_metadata=input_metadata,
        reference=reference,
    )

    media_utils = load_media_utils(args.flashvsr_reference)
    encoder_probe_dir = args.work_dir / "encoder_probe"
    selected_encoder, encoder_attempts = media_utils.select_encoder(
        EXPECTED_16K[0],
        EXPECTED_16K[1],
        threads=args.ffmpeg_threads,
        scratch_dir=encoder_probe_dir,
        preferred=args.encoder,
    )
    if selected_encoder["extension"] != ".mp4":
        raise PipelineError(
            f"selected encoder {selected_encoder['name']} requires {selected_encoder['extension']}; "
            "this pipeline contract requires exact-resolution MP4"
        )

    partial_output = args.output.with_name(
        f".{args.output.stem}.partial.{os.getpid()}{selected_encoder['extension']}"
    )
    partial_output.unlink(missing_ok=True)
    published = False
    try:
        encode_metrics = encode_staged_tiles(
            args=args,
            reference=reference,
            media_utils=media_utils,
            input_metadata=input_metadata,
            tiles=tiles,
            tile_dirs=tile_dirs,
            selected_encoder=selected_encoder,
            partial_output=partial_output,
            feather_width=feather_width,
        )
        output_metadata = validate_final_output(
            partial_output, expected_frames=args.expected_frames
        )
        os.replace(partial_output, args.output)
        published = True
        final_manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "status": "PASS",
            "video_id": args.video_id,
            "request": request,
            "input_4k": input_metadata,
            "original_8k": original_metadata,
            "output": {**output_metadata, "path": str(args.output)},
            "output_file": file_identity(args.output),
            "flashvsr": {
                "model": FLASHVSR_MODEL_NAME,
                "model_root": str(args.model_root),
                "reference_implementation": str(args.flashvsr_reference),
                "reference_sha256": sha256_file(args.flashvsr_reference),
                "scale": SCALE,
                "configuration": manifest["partial_report"]["configuration"],
                "stage_reused": reused_stage,
                "temporal_overlap": None,
                "temporal_mode": FLASHVSR_TEMPORAL_MODE,
                "tile_count": len(tiles),
            },
            "frame_mapping": {
                "policy": "same zero-based decoded frame index; final output retimed to 16/1",
                "input_4k_fps": "24/1",
                "original_8k_fps": "16/1",
                "output_fps": "16/1",
                "drop": False,
                "duplication": False,
                "interpolation": False,
                "input_and_original_cfr_timelines_verified": True,
                "equal_frame_counts_verified": True,
            },
            "center_composite": {
                "bbox_yx": list(CENTER_BBOX),
                "x_range": [CENTER_BBOX[1], CENTER_BBOX[3]],
                "y_range": [CENTER_BBOX[0], CENTER_BBOX[2]],
                "original_resized": False,
                "method": "Alpha Feather Blending",
                "alpha_profile": "inward nearest-edge smoothstep t^2*(3-2t)",
                "source_4k_feather_width_pixels": source_feather_width,
                "output_16k_feather_width_pixels": feather_width,
            },
            "encoding": {
                "single_final_video_encode": True,
                "raw_16k_intermediate_video": False,
                "encoder": selected_encoder,
                "encoder_probe_attempts": encoder_attempts,
                **encode_metrics,
            },
        }
        atomic_write_json(args.final_manifest, final_manifest)
    except BaseException:
        partial_output.unlink(missing_ok=True)
        if published:
            # A final MP4 without its matching provenance must never become a
            # resumable result. Preserve the completed FlashVSR stage for retry.
            args.output.unlink(missing_ok=True)
        raise

    # Successful final validation makes the very large staged tile cache disposable.
    try:
        shutil.rmtree(stage_scratch)
        paths["manifest"].unlink(missing_ok=True)
    except OSError as error:
        print(f"[WARN] final output is valid but stage cleanup failed: {error}", file=sys.stderr)
    print(f"[DONE] video_id={args.video_id} output={args.output}", flush=True)
    return final_manifest


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="FlashVSR v1.1 x4 + original 8K center Alpha Feather pipeline"
    )
    argument_parser.add_argument("--video-id", required=True, choices=["01", "02", "03", "04", "05"])
    argument_parser.add_argument("--input-4k", type=Path, required=True)
    argument_parser.add_argument("--original-8k", type=Path, required=True)
    argument_parser.add_argument("--model-root", type=Path, required=True)
    argument_parser.add_argument("--flashvsr-reference", type=Path, required=True)
    argument_parser.add_argument("--outpainting-config", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument("--final-manifest", type=Path, required=True)
    argument_parser.add_argument("--work-dir", type=Path, required=True)
    argument_parser.add_argument("--expected-frames", type=int, default=EXPECTED_FRAMES)
    argument_parser.add_argument("--tile-size", default="auto")
    argument_parser.add_argument("--tile-overlap", default="auto")
    argument_parser.add_argument("--min-tile-size", type=int, default=128)
    argument_parser.add_argument("--sparse-ratio", type=float, default=2.0, choices=[1.5, 2.0])
    argument_parser.add_argument("--local-range", type=int, default=11, choices=[9, 11])
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--gpu-index", type=int, default=0)
    argument_parser.add_argument("--ffmpeg-threads", type=int, default=32)
    argument_parser.add_argument(
        "--encoder",
        default="libx265",
        choices=["libx265", "hevc_nvenc"],
        help="MP4 encoder; each run performs the existing exact-15360x8640 capability probe",
    )
    argument_parser.add_argument("--dry-run", action="store_true")
    argument_parser.add_argument("--self-test", action="store_true")
    return argument_parser


def resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "input_4k",
        "original_8k",
        "model_root",
        "flashvsr_reference",
        "outpainting_config",
        "output",
        "final_manifest",
        "work_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        run_self_test()
        return 0
    args = parser().parse_args()
    if args.self_test:
        run_self_test()
        return 0
    resolve_paths(args)
    try:
        if args.expected_frames != EXPECTED_FRAMES:
            raise PipelineError(
                f"this pipeline stage requires {EXPECTED_FRAMES} frames, got {args.expected_frames}"
            )
        if args.output.suffix.lower() != ".mp4":
            raise PipelineError("final output path must use the .mp4 container")
        if args.output == args.input_4k or args.output == args.original_8k:
            raise PipelineError("final output must not overwrite either input")
        if not args.outpainting_config.is_file():
            if args.dry_run:
                feather_width = 16 * SCALE
            else:
                raise PipelineError(f"outpainting config is missing: {args.outpainting_config}")
        else:
            feather_width = read_outpainting_feather_width(args.outpainting_config) * SCALE
        if args.dry_run:
            print_dry_run(args, feather_width)
            return 0
        execute(args)
        return 0
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
