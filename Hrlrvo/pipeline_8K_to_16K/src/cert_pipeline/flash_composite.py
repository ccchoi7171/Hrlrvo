#!/usr/bin/env python3
"""Pipeline stage 02 -> 04: FlashVSR x4 plus original-8K center composite.

The neural portion is delegated to the repository's validated FlashVSR v1.1
implementation in ``Upscaler/architect/FlashVSR_v1.1/upscale.py`` using its
``--stage-mode inference`` path.  That path keeps one continuous causal/KV
timeline per spatial tile and stages lossless tile PNGs on disk.

This module consumes the frame-preserving 16fps outpaint result directly. It
retains the reference temporal tile PNGs and composites the final 16K output
directly from them. The historical raw 16K intermediate remains opt-in.
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
EXPECTED_INPUT_4K_FPS = Fraction(16, 1)
EXPECTED_ORIGINAL_8K_FPS = Fraction(16, 1)
EXPECTED_OUTPUT_FPS = Fraction(16, 1)
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


def jpeg_sequence_identity(
    path: Path, *, frame_count: int, start_frame: int = 0
) -> dict[str, Any]:
    """Stable identity for one numbered range in a JPEG frame directory."""
    return {
        "path": str(path.resolve()),
        "frame_count": frame_count,
        **({"start_frame": start_frame} if start_frame else {}),
        "files": [
            file_identity(path / f"{index}.jpg")
            for index in range(start_frame, start_frame + frame_count)
        ],
    }


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


def probe_video(path: Path, *, count_frames: bool = True) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
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
    if count_frames:
        command.insert(3, "-count_frames")
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
    frame_text = stream.get("nb_frames")
    if count_frames:
        frame_text = stream.get("nb_read_frames") or frame_text
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
        "decoded_frame_count_performed": count_frames,
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
    if len(input_timestamps) != frame_count or len(original_timestamps) < frame_count:
        raise PipelineError(
            "timestamp frame counts differ from the required frame count: "
            f"4K={len(input_timestamps)}, 8K={len(original_timestamps)}, expected={frame_count}"
        )
    tolerance = 1e-3
    input_step = 1.0 / float(input_fps)
    original_step = 1.0 / float(original_fps)
    for index, (input_pts, original_pts) in enumerate(zip(input_timestamps, original_timestamps[:frame_count])):
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
        "scratch_parent": work_dir / "tiles",
    }


def build_stage_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    command = [
        sys.executable,
        str(args.flashvsr_reference),
        "--input",
        str(args.input_4k),
        "--output",
        str(args.raw_output),
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
        "--scratch-path",
        str(paths["scratch_parent"] / "certification_stage"),
        "--stage-request",
        str(paths["request"]),
        "--benchmark-mode",
        "full",
        "--stage-mode",
        "inference",
        "--stage-manifest",
        str(paths["manifest"]),
        "--artifact-metadata-dir",
        str(args.work_dir / "metadata"),
    ]
    if getattr(args, "selected_range", False):
        command.extend([
            "--start-frame", str(args.start_frame),
            "--max-frames", str(args.expected_frames),
        ])
    return command


def stage_request_payload(
    args: argparse.Namespace,
    input_metadata: dict[str, Any],
    original_metadata: dict[str, Any],
    feather_width: int,
) -> dict[str, Any]:
    model_files = required_assets(args.model_root, args.flashvsr_reference)[1:]
    original_identity = file_identity(
        args.original_8k,
        with_sha256=not bool(getattr(args, "preverified_original_sha256", None)),
    )
    if getattr(args, "preverified_original_sha256", None):
        original_identity["sha256"] = args.preverified_original_sha256
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "video_id": args.video_id,
        "input_4k": file_identity(args.input_4k, with_sha256=True),
        "original_8k": original_identity,
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
            "selected_range": bool(getattr(args, "selected_range", False)),
            "input_start_frame": int(getattr(args, "start_frame", 0)),
            **({"direct_jpeg_output": True} if getattr(args, "direct_jpeg_output", False) else {}),
            "omit_original_audio": bool(getattr(args, "omit_original_audio", False)),
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
            **(
                {
                    "output_mode": "jpeg_sequence",
                    "jpeg_codec": "mjpeg",
                    "jpeg_quality": RawJpegSequenceEncoder.JPEG_QUALITY,
                    "jpeg_pixel_format": RawJpegSequenceEncoder.JPEG_PIXEL_FORMAT,
                    "jpeg_filename_pattern": "{frame_index}.jpg",
                }
                if args.output_mode == "jpeg_sequence" else {}
            ),
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
    if Path(manifest.get("output_target", "")).resolve() != args.raw_output:
        raise PipelineError("FlashVSR stage output target does not match the requested raw output")

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
        emissions = artifact.get("temporal_unit_metadata", [])
        if not emissions:
            raise PipelineError(f"staged tile {tile.index} has no temporal emission records")
        covered = 0
        for emission_path_string in emissions:
            emission_path = Path(emission_path_string).resolve()
            if not emission_path.is_file():
                raise PipelineError(f"temporal emission record is missing: {emission_path}")
            emission = json.loads(emission_path.read_text(encoding="utf-8"))
            if emission.get("state") != "COMPLETE" or emission.get("tile_index") != tile.index:
                raise PipelineError(f"invalid temporal emission record: {emission_path}")
            start = int(emission["frame_start"])
            end = int(emission["frame_end_exclusive"])
            if start != covered or end <= start:
                raise PipelineError(f"noncontiguous temporal emissions for tile {tile.index}")
            for frame_index in range(start, end):
                if not (directory / f"{frame_index:09d}.png").is_file():
                    raise PipelineError(f"missing emission PNG for tile {tile.index}, frame {frame_index}")
            covered = end
        if covered != args.expected_frames:
            raise PipelineError(f"tile {tile.index} temporal emissions cover {covered} frames")
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
        start_frame: int = 0,
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
        ]
        if start_frame:
            # Leave one second of preroll for HEVC open-GOP frames near a
            # keyframe. Seeking to the exact timestamp can skip that frame.
            preroll_frames = min(start_frame, int(EXPECTED_ORIGINAL_8K_FPS))
            command.extend([
                "-ss",
                f"{(start_frame - preroll_frames) / float(EXPECTED_ORIGINAL_8K_FPS):.9f}",
            ])
        command.extend([
            "-i",
            str(path),
        ])
        if start_frame:
            command.extend([
                "-ss", f"{preroll_frames / float(EXPECTED_ORIGINAL_8K_FPS):.9f}",
            ])
        command.extend([
            "-map",
            "0:v:0",
            "-frames:v",
            str(frame_count),
            "-an",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
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

        assert self.process.stdout is not None and self.process.stderr is not None
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
        assert self.process.stdout is not None and self.process.stderr is not None
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
    rows: np.ndarray = np.arange(anchor_row_start, anchor_row_end, dtype=np.float32)
    columns: np.ndarray = np.arange(anchor_width, dtype=np.float32)
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


class RawJpegSequenceEncoder:
    """Stream complete RGB frames to a fixed-quality JPEG image sequence."""

    JPEG_QUALITY = 2
    JPEG_PIXEL_FORMAT = "yuvj420p"

    def __init__(
        self,
        output_dir: Path,
        *,
        width: int,
        height: int,
        frame_count: int,
        threads: int,
        start_number: int = 0,
        existing_directory: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.frames_written = 0
        self.rows_in_frame = 0
        if existing_directory:
            if not output_dir.is_dir():
                raise PipelineError(f"direct JPEG output directory is missing: {output_dir}")
            for index in range(start_number, start_number + frame_count):
                if (output_dir / f"{index}.jpg").exists():
                    raise PipelineError(f"direct JPEG output frame already exists: {index}.jpg")
        else:
            output_dir.mkdir(parents=False, exist_ok=False)
        self.command = [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", f"{EXPECTED_OUTPUT_FPS.numerator}/{EXPECTED_OUTPUT_FPS.denominator}",
            "-i", "pipe:0", "-frames:v", str(frame_count), "-an",
            "-c:v", "mjpeg", "-q:v", str(self.JPEG_QUALITY),
            "-pix_fmt", self.JPEG_PIXEL_FORMAT,
            "-threads", str(max(1, threads)),
            "-start_number", str(start_number),
            "-n" if existing_directory else "-y", str(output_dir / "%d.jpg"),
        ]
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self.process.stdin is None or self.process.stderr is None:
            raise PipelineError("failed to open JPEG encoder pipes")

    def write(self, rows: Any) -> None:
        import numpy as np

        if rows.dtype != np.uint8 or rows.ndim != 3 or rows.shape[1:] != (self.width, 3):
            raise PipelineError(
                f"invalid JPEG encoder row band shape/type: {rows.shape} {rows.dtype}"
            )
        if self.rows_in_frame + rows.shape[0] > self.height:
            raise PipelineError("JPEG encoder received more than one frame of rows")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(np.ascontiguousarray(rows).tobytes())
        except BrokenPipeError as error:
            detail = self.process.stderr.read().decode("utf-8", errors="replace")
            raise PipelineError(f"JPEG encoder terminated early: {detail.strip()}") from error
        self.rows_in_frame += rows.shape[0]

    def frame_complete(self) -> None:
        if self.rows_in_frame != self.height:
            raise PipelineError(
                f"JPEG frame has {self.rows_in_frame} rows, expected {self.height}"
            )
        self.frames_written += 1
        self.rows_in_frame = 0

    def close(self) -> None:
        assert self.process.stdin is not None and self.process.stderr is not None
        self.process.stdin.close()
        error = self.process.stderr.read().decode("utf-8", errors="replace")
        return_code = self.process.wait()
        if return_code:
            raise PipelineError(f"JPEG encoder failed: {error.strip()}")
        if self.rows_in_frame:
            raise PipelineError("JPEG encoder closed with an incomplete frame")
        if self.frames_written != self.frame_count:
            raise PipelineError(
                f"JPEG encoder received {self.frames_written} frames, expected {self.frame_count}"
            )

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


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
    import torch

    cv2.setNumThreads(max(1, args.ffmpeg_threads))
    composite_device = (
        torch.device(f"cuda:{args.gpu_index}")
        if torch.cuda.is_available() and args.gpu_index < torch.cuda.device_count()
        else None
    )
    device_label = str(composite_device) if composite_device is not None else "cpu"
    print(
        f"[COMPOSITE_DEVICE] device={device_label} "
        f"verify_first_frame={composite_device is not None}",
        flush=True,
    )
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
        start_frame=getattr(args, "start_frame", 0),
    )
    if args.output_mode == "jpeg_sequence":
        encoder = RawJpegSequenceEncoder(
            partial_output,
            width=output_width,
            height=output_height,
            frame_count=args.expected_frames,
            threads=args.ffmpeg_threads,
            start_number=args.start_frame if getattr(args, "direct_jpeg_output", False) else 0,
            existing_directory=bool(getattr(args, "direct_jpeg_output", False)),
        )
    else:
        encoder = media_utils.RawVideoEncoder(
            partial_output,
            width=output_width,
            height=output_height,
            fps=f"{EXPECTED_OUTPUT_FPS.numerator}/{EXPECTED_OUTPUT_FPS.denominator}",
            frame_count=args.expected_frames,
            encoder=selected_encoder,
            threads=args.ffmpeg_threads,
            source_metadata=source_video_metadata(
                media_utils,
                {**input_metadata, "frame_count": args.expected_frames,
                 "duration": args.expected_frames / float(EXPECTED_INPUT_4K_FPS)},
            ),
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
                accumulator: np.ndarray = np.zeros((band_height, output_width, 3), dtype=np.float32)
                normalization: np.ndarray = np.zeros((band_height, output_width), dtype=np.float32)
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
                if composite_device is not None:
                    with torch.no_grad():
                        gpu_accumulator = torch.from_numpy(accumulator).to(composite_device)
                        gpu_normalization = torch.from_numpy(normalization).to(composite_device)
                        output_rows = (
                            (gpu_accumulator / gpu_normalization.unsqueeze(-1))
                            .round()
                            .clamp_(0, 255)
                            .to(torch.uint8)
                            .cpu()
                            .numpy()
                        )
                    if frame_index == 0:
                        reference_rows = np.rint(
                            accumulator / normalization[..., None]
                        ).clip(0, 255).astype(np.uint8)
                        if not np.array_equal(output_rows, reference_rows):
                            raise PipelineError(
                                f"GPU composite differs from CPU at rows [{band_start}, {band_end})"
                            )
                else:
                    # Reuse the float32 accumulator to avoid full-size temporaries.
                    np.divide(accumulator, normalization[..., None], out=accumulator)
                    np.rint(accumulator, out=accumulator)
                    np.clip(accumulator, 0, 255, out=accumulator)
                    output_rows = accumulator.astype(np.uint8)
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
            if frame_index == 0 and composite_device is not None:
                print(f"[COMPOSITE_GPU_CHECK] frame=1 bands={len(bands)} status=PASS", flush=True)
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
        "output_kind": (
            "jpeg_sequence" if args.output_mode == "jpeg_sequence" else "video"
        ),
    }


def validate_final_output(
    path: Path, *, expected_frames: int, count_frames: bool = True
) -> dict[str, Any]:
    metadata = probe_video(path) if count_frames else probe_video_headers(path)
    validate_video_spec(
        metadata,
        label="final 16K output",
        resolution=EXPECTED_16K,
        fps=EXPECTED_OUTPUT_FPS,
        frame_count=expected_frames,
    )
    return metadata


def probe_video_headers(path: Path) -> dict[str, Any]:
    """Fail closed on missing container frame count, without decoding 16K video."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise PipelineError(f"video output is missing or empty: {path}")
    return probe_video(path, count_frames=False)


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG SOF dimensions without decoding the 16K pixel payload."""
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    if not path.is_file() or path.stat().st_size < 4:
        raise PipelineError(f"JPEG output is missing or empty: {path}")
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise PipelineError(f"JPEG output has no SOI marker: {path}")
        while True:
            marker_prefix = stream.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b"\xff":
                continue
            marker_byte = stream.read(1)
            while marker_byte == b"\xff":
                marker_byte = stream.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]
            if marker in {0x01, *range(0xD0, 0xD9)}:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                raise PipelineError(f"invalid JPEG segment length in {path}")
            if marker in sof_markers:
                payload = stream.read(5)
                if len(payload) != 5:
                    break
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                if width <= 0 or height <= 0:
                    break
                return width, height
            stream.seek(segment_length - 2, os.SEEK_CUR)
    raise PipelineError(f"JPEG output has no valid SOF dimensions: {path}")


def validate_jpeg_sequence(
    path: Path, *, expected_frames: int, start_frame: int = 0,
    allow_other_frames: bool = False,
) -> dict[str, Any]:
    if not path.is_dir():
        raise PipelineError(f"JPEG sequence output directory is missing: {path}")
    expected_names = [
        f"{index}.jpg" for index in range(start_frame, start_frame + expected_frames)
    ]
    actual_names = [entry.name for entry in path.iterdir()]
    names_valid = (
        set(expected_names).issubset(actual_names)
        if allow_other_frames else set(actual_names) == set(expected_names)
    )
    if not names_valid:
        raise PipelineError(
            f"JPEG sequence names/count mismatch in {path}: expected "
            f"{start_frame}.jpg..{start_frame + expected_frames - 1}.jpg"
        )
    for name in expected_names:
        frame_path = path / name
        width, height = jpeg_dimensions(frame_path)
        if (width, height) != EXPECTED_16K:
            raise PipelineError(
                f"JPEG {name} is {width}x{height}, expected {EXPECTED_16K[0]}x{EXPECTED_16K[1]}"
            )
        with frame_path.open("rb") as stream:
            stream.seek(-2, os.SEEK_END)
            if stream.read(2) != b"\xff\xd9":
                raise PipelineError(f"JPEG output has no EOI marker: {frame_path}")
    return {
        "path": str(path.resolve()),
        "format": "jpeg_sequence",
        "codec_name": "mjpeg",
        "width": EXPECTED_16K[0],
        "height": EXPECTED_16K[1],
        "frame_count": expected_frames,
        "filename_pattern": "{frame_index}.jpg",
        "first_frame": f"{start_frame}.jpg",
        "last_frame": f"{start_frame + expected_frames - 1}.jpg",
    }


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
    existing_mode = manifest.get("encoding", {}).get("output_mode", "legacy_raw")
    allow_legacy_resume = (
        getattr(args, "allow_existing_legacy", False)
        and args.output_mode == "final_only"
        and existing_mode == "legacy_raw"
    )
    if existing_mode != args.output_mode and not allow_legacy_resume:
        raise PipelineError(
            f"existing final output mode is {existing_mode}, requested {args.output_mode}; "
            "use a separate output root or an explicit --force run"
        )
    if existing_mode == "jpeg_sequence":
        validate_jpeg_sequence(
            args.output, expected_frames=args.expected_frames,
            start_frame=args.start_frame if getattr(args, "direct_jpeg_output", False) else 0,
            allow_other_frames=bool(getattr(args, "direct_jpeg_output", False)),
        )
        current_identity = jpeg_sequence_identity(
            args.output, frame_count=args.expected_frames,
            start_frame=args.start_frame if getattr(args, "direct_jpeg_output", False) else 0,
        )
        if manifest.get("output_sequence") != current_identity:
            raise PipelineError("existing JPEG sequence identity does not match its manifest")
    else:
        current_identity = file_identity(args.output)
        if manifest.get("output_file") != current_identity:
            raise PipelineError("existing final output identity does not match its manifest")
    if existing_mode == "legacy_raw" and not args.raw_output.is_file():
        raise PipelineError("verified final output is missing its retained raw FlashVSR video")
    paths = stage_paths(args.work_dir)
    if not paths["manifest"].is_file() or not paths["request"].is_file():
        raise PipelineError("verified final output is missing its native FlashVSR stage records")
    if json.loads(paths["request"].read_text(encoding="utf-8")) != request:
        raise PipelineError("native FlashVSR stage request no longer matches the final result")
    reference = load_reference_module(args.flashvsr_reference)
    validate_stage_manifest(
        reference.read_stage_manifest(paths["manifest"]),
        args=args, input_metadata=probe_video(args.input_4k), reference=reference,
    )
    if existing_mode == "legacy_raw":
        validate_final_output(args.raw_output, expected_frames=args.expected_frames)
    elif existing_mode == "final_only":
        if manifest.get("encoding", {}).get("frames_written") != args.expected_frames:
            raise PipelineError("final-only manifest encoder frame count mismatch")
    if existing_mode == "jpeg_sequence":
        return validate_jpeg_sequence(
            args.output, expected_frames=args.expected_frames,
            start_frame=args.start_frame if getattr(args, "direct_jpeg_output", False) else 0,
            allow_other_frames=bool(getattr(args, "direct_jpeg_output", False)),
        )
    return validate_final_output(
        args.output,
        expected_frames=args.expected_frames,
        count_frames=existing_mode == "legacy_raw",
    )


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
        "raw_output": str(args.raw_output),
        "final_output": str(args.output),
        "final_manifest": str(args.final_manifest),
        "work_dir": str(args.work_dir),
        "stage_manifest": str(paths["manifest"]),
        "expected_input_4k": {
            "resolution": list(EXPECTED_4K),
            "fps": "16/1",
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
            "same decoded-frame index; 4K 16fps input is emitted at 16fps with no "
            "drop, duplication, or interpolation"
        ),
        "output_mode": args.output_mode,
        "intermediate_policy": (
            "lossless staged tile PNGs and raw 16K MP4 retained"
            if args.output_mode == "legacy_raw"
            else "lossless staged tile PNGs retained; raw 16K MP4 skipped"
        ),
        "final_encode_policy": (
            "tile PNG reconstruction and original-center composite streamed to numbered JPEGs"
            if args.output_mode == "jpeg_sequence"
            else "tile PNG reconstruction and original-center composite in one final encode"
        ),
        "encoder": (
            {
                "codec": "mjpeg",
                "quality": RawJpegSequenceEncoder.JPEG_QUALITY,
                "pixel_format": RawJpegSequenceEncoder.JPEG_PIXEL_FORMAT,
            }
            if args.output_mode == "jpeg_sequence" else args.encoder
        ),
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

    generated: np.ndarray = np.zeros((6, 24, 3), dtype=np.uint8)
    original: np.ndarray = np.full((12, 12, 3), 200, dtype=np.uint8)
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

    selected_range = bool(getattr(args, "selected_range", False))
    start_frame = int(getattr(args, "start_frame", 0))
    input_metadata = probe_video(args.input_4k, count_frames=not selected_range)
    preverified_original = bool(getattr(args, "preverified_original_sha256", None))
    cached_path = getattr(args, "preverified_original_metadata_json", None)
    cached_original = selected_range and cached_path is not None
    if cached_original:
        identity = json.loads(cached_path.read_text(encoding="utf-8"))
        original_metadata = identity["original_metadata"]
    else:
        original_metadata = probe_video(args.original_8k, count_frames=not preverified_original)
    validate_video_spec(
        input_metadata,
        label="4K outpaint input",
        resolution=EXPECTED_4K,
        fps=EXPECTED_INPUT_4K_FPS,
        frame_count=input_metadata["frame_count"] if selected_range else args.expected_frames,
    )
    if selected_range and input_metadata["frame_count"] < start_frame + args.expected_frames:
        raise PipelineError("4K outpaint input ends before the selected Upscale frame range")
    if not cached_original:
        if (original_metadata["width"], original_metadata["height"]) != EXPECTED_8K:
            raise PipelineError("original 8K input resolution mismatch")
        if original_metadata["avg_frame_rate"] != "16/1" or original_metadata["r_frame_rate"] != "16/1":
            raise PipelineError("original 8K input FPS mismatch")
        if original_metadata["frame_count"] < start_frame + args.expected_frames:
            raise PipelineError("original 8K has fewer frames than requested")
    if not preverified_original:
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
    direct_jpeg_output = (
        args.output_mode == "jpeg_sequence"
        and bool(getattr(args, "direct_jpeg_output", False))
    )
    if direct_jpeg_output:
        if not args.output.is_dir():
            raise PipelineError(f"direct JPEG output directory is missing: {args.output}")
        if args.final_manifest.is_file():
            existing = validate_existing_result(args, request=request)
            print(f"[SKIP] valid JPEG range already exists: {args.output}", flush=True)
            return {"status": "SKIPPED_VALID", "output": existing}
        for index in range(args.start_frame, args.start_frame + args.expected_frames):
            if (args.output / f"{index}.jpg").exists():
                raise PipelineError(
                    f"uncommitted JPEG frame {index}.jpg exists; "
                    "the KOCCA stage must clear this range before retry"
                )
    elif args.output.exists():
        if args.final_manifest.is_file():
            existing = validate_existing_result(args, request=request)
            print(f"[SKIP] valid final output already exists: {args.output}", flush=True)
            return {"status": "SKIPPED_VALID", "output": existing}
        # Power loss between atomic output publication and the PASS manifest
        # leaves an untrusted orphan. Preserve it for audit and rebuild from
        # the committed FlashVSR stage rather than treating it as complete.
        prior_paths = stage_paths(args.work_dir)
        if (not prior_paths["manifest"].is_file()
                or not prior_paths["request"].is_file()
                or json.loads(prior_paths["request"].read_text(encoding="utf-8")) != request):
            raise PipelineError("final output has no PASS manifest or matching FlashVSR stage request")
        orphan_dir = args.work_dir / "orphaned_outputs"
        orphan_dir.mkdir(parents=True, exist_ok=True)
        orphan = orphan_dir / f"{args.output.name}.{time.time_ns()}"
        os.replace(args.output, orphan)
        print(f"[RESUME] preserved uncommitted final output: {orphan}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output_mode == "legacy_raw":
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
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

    raw_created = False
    if args.output_mode == "legacy_raw" and args.raw_output.is_file():
        try:
            validate_final_output(args.raw_output, expected_frames=args.expected_frames)
        except Exception:
            # A crashed encoder may leave a named but incomplete raw file.
            # Keep it for inspection and regenerate from committed tile PNGs.
            orphan_dir = args.work_dir / "orphaned_outputs"
            orphan_dir.mkdir(parents=True, exist_ok=True)
            orphan = orphan_dir / (
                f"{args.raw_output.stem}.{time.time_ns()}{args.raw_output.suffix}"
            )
            os.replace(args.raw_output, orphan)
            print(f"[RESUME] preserved incomplete raw FlashVSR file: {orphan}", flush=True)
        else:
            print(f"[RESUME] reusing verified raw FlashVSR video: {args.raw_output}", flush=True)
    if args.output_mode == "legacy_raw" and not args.raw_output.is_file():
        print(f"[FLASHVSR_RAW_ENCODE] output={args.raw_output}", flush=True)
        raw_report = reference.execute_encode_stage(argparse.Namespace(
            stage_manifest=str(paths["manifest"]),
            ffmpeg_threads=args.ffmpeg_threads,
            encoder=args.encoder,
            keep_scratch=True,
        ))
        if Path(raw_report["output_path"]).resolve() != args.raw_output:
            raise PipelineError("FlashVSR raw encoder published an unexpected output path")
        atomic_write_json(paths["report"], raw_report)
        validate_final_output(args.raw_output, expected_frames=args.expected_frames)
        raw_created = True
    elif args.output_mode in {"final_only", "jpeg_sequence"}:
        print(f"[FLASHVSR_RAW_ENCODE] status=SKIPPED mode={args.output_mode}", flush=True)

    media_utils = load_media_utils(args.flashvsr_reference)
    if args.output_mode == "jpeg_sequence":
        selected_encoder = {
            "name": "mjpeg",
            "codec": "mjpeg",
            "quality": RawJpegSequenceEncoder.JPEG_QUALITY,
            "pixel_format": RawJpegSequenceEncoder.JPEG_PIXEL_FORMAT,
            "extension": ".jpg",
        }
        encoder_attempts: list[dict[str, Any]] = []
        partial_output = (
            args.output if direct_jpeg_output else args.output.with_name(
                f".{args.output.name}.partial.{os.getpid()}.{time.time_ns()}"
            )
        )
        if not direct_jpeg_output and partial_output.exists():
            raise PipelineError(f"JPEG temporary directory already exists: {partial_output}")
    else:
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
        if encode_metrics.get("frames_written") != args.expected_frames:
            raise PipelineError("final encoder frame counter mismatch")
        if args.output_mode == "jpeg_sequence":
            output_metadata = validate_jpeg_sequence(
                partial_output, expected_frames=args.expected_frames,
                start_frame=args.start_frame if direct_jpeg_output else 0,
                allow_other_frames=direct_jpeg_output,
            )
            original_media = None
            if not direct_jpeg_output:
                os.replace(partial_output, args.output)
        else:
            output_metadata = (
                validate_final_output(partial_output, expected_frames=args.expected_frames)
                if args.output_mode == "legacy_raw" else None
            )
            original_media = (
                None if getattr(args, "omit_original_audio", False)
                else media_utils.probe_media(args.original_8k, count_frames=False)
            )
        if original_media is not None and original_media.audio_present:
            mux_partial = args.output.with_name(
                f".{args.output.stem}.audio.partial.{os.getpid()}.mp4"
            )
            try:
                mux_command = [
                    "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
                    "-i", str(partial_output), "-i", str(args.original_8k),
                    "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
                    "-t", f"{args.expected_frames / float(EXPECTED_OUTPUT_FPS):.9f}",
                    "-movflags", "+faststart", "-y", str(mux_partial),
                ]
                mux_result = run(mux_command, check=False, capture_output=True)
                if mux_result.returncode:
                    raise PipelineError(f"original-audio mux failed: {mux_result.stderr}")
                if args.output_mode == "legacy_raw":
                    validate_final_output(mux_partial, expected_frames=args.expected_frames)
                os.replace(mux_partial, args.output)
            finally:
                mux_partial.unlink(missing_ok=True)
                partial_output.unlink(missing_ok=True)
        elif args.output_mode != "jpeg_sequence":
            os.replace(partial_output, args.output)
        published = True
        if args.output_mode == "final_only":
            output_metadata = validate_final_output(
                args.output, expected_frames=args.expected_frames, count_frames=False
            )
        elif args.output_mode == "jpeg_sequence":
            output_metadata = validate_jpeg_sequence(
                args.output, expected_frames=args.expected_frames,
                start_frame=args.start_frame if direct_jpeg_output else 0,
                allow_other_frames=direct_jpeg_output,
            )
        if output_metadata is None:
            raise PipelineError("final output metadata was not verified")
        final_manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "status": "PASS",
            "video_id": args.video_id,
            "request": request,
            "input_4k": input_metadata,
            "original_8k": original_metadata,
            "output": {**output_metadata, "path": str(args.output)},
            "output_file": (
                None if args.output_mode == "jpeg_sequence" else file_identity(args.output)
            ),
            "output_sequence": (
                jpeg_sequence_identity(
                    args.output, frame_count=args.expected_frames,
                    start_frame=args.start_frame if direct_jpeg_output else 0,
                )
                if args.output_mode == "jpeg_sequence" else None
            ),
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
                "policy": "same zero-based decoded frame index; 16/1 input and output",
                "input_4k_fps": "16/1",
                "original_8k_fps": "16/1",
                "output_fps": "16/1",
                "drop": False,
                "duplication": False,
                "interpolation": False,
                "input_and_original_cfr_timelines_verified": not preverified_original,
                "source_has_at_least_requested_frames_verified": True,
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
                "single_final_video_encode": args.output_mode != "jpeg_sequence",
                "output_mode": args.output_mode,
                "image_sequence": args.output_mode == "jpeg_sequence",
                "filename_pattern": (
                    "{frame_index}.jpg" if args.output_mode == "jpeg_sequence" else None
                ),
                "raw_16k_intermediate_video": args.output_mode == "legacy_raw",
                "raw_16k_output": str(args.raw_output) if args.output_mode == "legacy_raw" else None,
                "raw_status": "PRESENT" if args.output_mode == "legacy_raw" else "SKIPPED",
                "raw_created_by_this_run": raw_created,
                "preexisting_raw_file_ignored": (
                    args.output_mode in {"final_only", "jpeg_sequence"}
                    and args.raw_output.is_file()
                ),
                "original_audio_source": (
                    str(args.original_8k)
                    if original_media is not None and original_media.audio_present else None
                ),
                "encoder": selected_encoder,
                "encoder_probe_attempts": encoder_attempts,
                **encode_metrics,
            },
            "verification_policy": {
                "full_16k_pixel_decode_performed": False,
                "ffprobe_count_frames_performed": args.output_mode == "legacy_raw",
                "final_frame_count_evidence": (
                    "ffprobe_count_frames_and_encoder_counter"
                    if args.output_mode == "legacy_raw"
                    else (
                        "exact_numbered_file_set_jpeg_headers_and_encoder_counter"
                        if args.output_mode == "jpeg_sequence"
                        else "ffprobe_container_nb_frames_and_encoder_counter"
                    )
                ),
                "limitations": (
                    "final-only PASS does not prove a full pixel decode"
                    if args.output_mode == "final_only"
                    else (
                        "JPEG dimensions and terminal markers verified without full pixel decode"
                        if args.output_mode == "jpeg_sequence" else None
                    )
                ),
            },
        }
        atomic_write_json(args.final_manifest, final_manifest)
    except BaseException:
        if direct_jpeg_output:
            for index in range(args.start_frame, args.start_frame + args.expected_frames):
                (args.output / f"{index}.jpg").unlink(missing_ok=True)
        elif partial_output.is_dir():
            shutil.rmtree(partial_output)
        else:
            partial_output.unlink(missing_ok=True)
        if published and not direct_jpeg_output:
            # A final output without provenance is not resumable as PASS. Retain
            # it for the certification audit while allowing a safe retry.
            orphan_dir = args.work_dir / "orphaned_outputs"
            orphan_dir.mkdir(parents=True, exist_ok=True)
            os.replace(
                args.output,
                orphan_dir / f"{args.output.name}.{time.time_ns()}",
            )
        raise

    # Staged PNGs and temporal manifests remain; raw MP4 is opt-in only.
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
    argument_parser.add_argument("--raw-output", type=Path, required=True)
    argument_parser.add_argument(
        "--output-mode",
        choices=("final_only", "legacy_raw", "jpeg_sequence"),
        default="final_only",
    )
    argument_parser.add_argument("--allow-existing-legacy", action="store_true")
    argument_parser.add_argument("--final-manifest", type=Path, required=True)
    argument_parser.add_argument("--work-dir", type=Path, required=True)
    argument_parser.add_argument("--expected-frames", type=int, required=True)
    argument_parser.add_argument("--selected-range", action="store_true")
    argument_parser.add_argument("--direct-jpeg-output", action="store_true")
    argument_parser.add_argument("--start-frame", type=int, default=0)
    argument_parser.add_argument("--omit-original-audio", action="store_true")
    argument_parser.add_argument("--preverified-original-metadata-json", type=Path)
    argument_parser.add_argument(
        "--preverified-original-sha256",
        help="Reuse the KOCCA run's original-input SHA256 without rescanning its frames or bytes",
    )
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
        "raw_output",
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
        if args.expected_frames <= 0:
            raise PipelineError("expected frame count must be positive")
        if args.start_frame < 0 or (args.start_frame and not args.selected_range):
            raise PipelineError("selected Upscale frame start must be nonnegative")
        if args.direct_jpeg_output and (args.output_mode != "jpeg_sequence" or not args.selected_range):
            raise PipelineError("direct JPEG output requires a selected JPEG frame range")
        if args.selected_range and not args.preverified_original_sha256:
            raise PipelineError("selected Upscale range requires the KOCCA original input SHA256")
        if args.preverified_original_sha256 and not re.fullmatch(
            r"[0-9a-f]{64}", args.preverified_original_sha256
        ):
            raise PipelineError("preverified original SHA256 must be 64 lowercase hex characters")
        if args.output_mode == "jpeg_sequence":
            if args.output.suffix:
                raise PipelineError("JPEG sequence output path must be a directory without a suffix")
        elif args.output.suffix.lower() != ".mp4":
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
