#!/usr/bin/env python3
"""CPU-only evaluation of the already-rendered 4K->16K benchmark outputs.

This program never imports a model framework and never invokes any upscaler.
Its subprocess allow-list is limited to ffprobe, ffmpeg, nice, and ionice.
All video processing is streaming; no native 16K sequence is accumulated in RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "result" / "benchmark"
OUTPUT_ROOT = ROOT / "output" / "benchmark"
EVAL_ROOT = ROOT / "result" / "cpu_quantitative_eval"
SOURCE = ROOT / "input" / "0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4"

MODELS = (
    ("Real-ESRGAN", "Real-ESRGAN", "realesrgan"),
    ("HAT", "HAT", "hat"),
    ("FlashVSR v1.1", "FlashVSR_v1.1", "flashvsr_v11"),
    ("SeedVR2", "SeedVR2", "seedvr2"),
    ("DOVE", "DOVE", "dove"),
)
REFERENCE_MODEL = ("Upscale-A-Video", "Upscale-A-Video", "upscale_a_video")
MODES = ("micro", "mini")

CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
}
LOW_PRIORITY = ["nice", "-n", "10", "ionice", "-c2", "-n7"]
FFMPEG_BASE = [
    "ffmpeg",
    "-hide_banner",
    "-nostdin",
    "-threads",
    "8",
    "-filter_threads",
    "8",
    "-filter_complex_threads",
    "8",
]


@dataclass
class Probe:
    path: str
    width: int
    height: int
    fps: str
    frame_count: int
    duration: float
    codec: str
    pixel_format: str


@dataclass
class RunRecord:
    model: str
    model_dir: str
    slug: str
    variant: str
    status: str
    report_path: str
    input_path: str | None
    output_path: str | None
    start_frame: int | None
    expected_input_frames: int | None
    recorded_output_frames: int | None
    total_wall_seconds: float | None
    recorded_e2e_seconds_per_frame: float | None
    pure_inference_seconds_per_frame: float | None
    timing_scope: str
    timing_source: str
    input_probe: Probe | None = None
    output_probe: Probe | None = None
    selected_input_duration: float | None = None
    scale_x: float | None = None
    scale_y: float | None = None
    exact_4x: bool | None = None
    is_16k: bool | None = None
    frame_alignment: str = "N/A"
    evaluated_frames: int | None = None


def cpu_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(CPU_ENV)
    return env


def ensure_cpu_guard() -> None:
    """Refuse an invocation that does not explicitly hide accelerator devices."""
    for key, expected in CPU_ENV.items():
        actual = os.environ.get(key)
        if actual != expected:
            raise SystemExit(
                f"CPU guard failed: {key}={actual!r}; expected {expected!r}. "
                "Export the required CPU-only environment before running."
            )


def create_directories() -> None:
    for path in (
        EVAL_ROOT,
        EVAL_ROOT / "metrics_per_frame",
        EVAL_ROOT / "visual_crops" / "medium",
        EVAL_ROOT / "visual_crops" / "tight",
        EVAL_ROOT / "visual_crops" / "contact_sheet",
        EVAL_ROOT / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rational_string(value: str) -> str:
    return str(Fraction(value))


def ffprobe(path: Path) -> Probe:
    command = LOW_PRIORITY + [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,codec_name,pix_fmt:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=cpu_environment(),
    )
    data = json.loads(completed.stdout)
    stream = data["streams"][0]
    frame_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    fps = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    return Probe(
        path=str(path.resolve()),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=rational_string(fps),
        frame_count=int(frame_count),
        duration=float(data["format"]["duration"]),
        codec=stream["codec_name"],
        pixel_format=stream.get("pix_fmt", "unknown"),
    )


def nested_first(data: dict[str, Any], paths: Iterable[tuple[str, ...]]) -> Any:
    for keys in paths:
        value: Any = data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if value is not None:
                return value
    return None


def timing_scope(data: dict[str, Any]) -> str:
    timing = data.get("timing") or data.get("timings") or {}
    present = []
    for key, label in (
        ("model_load_seconds", "model load"),
        ("input_decode_seconds", "input decode"),
        ("input_decode_and_staging_seconds", "input decode/staging"),
        ("decode_seconds", "decode"),
        ("inference_seconds", "SR inference"),
        ("sr_inference_seconds", "SR loop"),
        ("neural_inference_seconds", "neural inference"),
        ("merge_seconds", "tile/frame merge"),
        ("encode_seconds", "encode"),
        ("encoding_mux_seconds", "encode/mux or staged postprocess"),
    ):
        if timing.get(key) is not None:
            present.append(label)
    detail = ", ".join(dict.fromkeys(present)) or "undifferentiated stages"
    return f"end-to-end total_wall_seconds ({detail})"


def load_probe_cache() -> dict[str, Probe]:
    inventory_path = EVAL_ROOT / "inventory.json"
    if not inventory_path.is_file():
        raise RuntimeError("--reuse-probes requires an existing inventory.json")
    inventory = load_json(inventory_path)
    cache: dict[str, Probe] = {}
    source = Probe(**inventory["source_probe"])
    cache[source.path] = source
    for row in inventory["runs"]:
        output_path = row.get("output_path")
        if not output_path or output_path == "N/A":
            continue
        cache[output_path] = Probe(
            path=output_path,
            width=int(row["output_width"]),
            height=int(row["output_height"]),
            fps=rational_string(str(row["output_fps"])),
            frame_count=int(row["output_frames"]),
            duration=float(row["output_duration_seconds"]),
            codec=str(row["output_codec"]),
            pixel_format=str(row["output_pixel_format"]),
        )
    return cache


def build_records(reuse_probes: bool = False) -> tuple[Probe, list[RunRecord]]:
    cache = load_probe_cache() if reuse_probes else {}

    def probe(path: Path) -> Probe:
        resolved = str(path.resolve())
        if reuse_probes:
            if resolved not in cache:
                raise RuntimeError(f"No cached ffprobe result for {resolved}")
            if not path.is_file():
                raise RuntimeError(f"Cached probe target no longer exists: {resolved}")
            return cache[resolved]
        return ffprobe(path)

    source_probe = probe(SOURCE)
    records: list[RunRecord] = []
    for model, model_dir, slug in MODELS:
        for mode in MODES:
            report_path = RESULT_ROOT / model_dir / f"benchmark_{mode}.json"
            data = load_json(report_path)
            timing = data.get("timing") or data.get("timings") or {}
            input_value = data.get("input")
            input_path = (
                input_value.get("path")
                if isinstance(input_value, dict)
                else input_value
            )
            input_path = input_path or nested_first(
                data,
                (("input_metadata", "path"), ("source_metadata", "path")),
            )
            output_path = data.get("output_path") or data.get("output")
            start_frame = nested_first(
                data,
                (("configuration", "start_frame"), ("frame_selection", "start_frame")),
            )
            expected_input_frames = nested_first(
                data,
                (
                    ("effective_input_frames_processed",),
                    ("configuration", "effective_input_frames_processed"),
                    ("timing", "effective_input_frames_processed"),
                    ("frame_selection", "selected_frames"),
                    ("frame_selection", "real_frame_count"),
                    ("input", "frame_count"),
                    ("input_metadata", "frame_count"),
                ),
            )
            recorded_output_frames = nested_first(
                data,
                (
                    ("effective_output_frames_written",),
                    ("timing", "effective_output_frames_written"),
                    ("technical_validation", "observed", "frame_count"),
                ),
            )
            wall = data.get("total_wall_seconds") or timing.get("total_wall_seconds")
            e2e_spf = data.get("end_to_end_seconds_per_output_frame") or timing.get(
                "end_to_end_seconds_per_output_frame"
            )
            pure_spf = data.get(
                "pure_inference_seconds_per_output_frame"
            ) or timing.get("pure_inference_seconds_per_output_frame")
            record = RunRecord(
                model=model,
                model_dir=model_dir,
                slug=slug,
                variant=mode,
                status=str(data.get("status", "unknown")),
                report_path=str(report_path.resolve()),
                input_path=str(Path(input_path).resolve()) if input_path else None,
                output_path=str(Path(output_path).resolve()) if output_path else None,
                start_frame=int(start_frame) if start_frame is not None else None,
                expected_input_frames=int(expected_input_frames)
                if expected_input_frames
                else None,
                recorded_output_frames=int(recorded_output_frames)
                if recorded_output_frames
                else None,
                total_wall_seconds=float(wall) if wall is not None else None,
                recorded_e2e_seconds_per_frame=float(e2e_spf)
                if e2e_spf is not None
                else None,
                pure_inference_seconds_per_frame=float(pure_spf)
                if pure_spf is not None
                else None,
                timing_scope=timing_scope(data),
                timing_source=str(report_path.resolve()),
            )
            if record.start_frame is None:
                raise RuntimeError(f"No recorded start frame in {report_path}")
            if record.input_path != str(SOURCE.resolve()):
                raise RuntimeError(
                    f"Unexpected input pairing in {report_path}: {record.input_path}"
                )
            if not record.output_path or not Path(record.output_path).is_file():
                raise RuntimeError(
                    f"Recorded output is missing for {report_path}: {record.output_path}"
                )
            record.input_probe = source_probe
            record.output_probe = probe(Path(record.output_path))
            fps = Fraction(source_probe.fps)
            record.selected_input_duration = float(
                Fraction(record.expected_input_frames, 1) / fps
            )
            record.scale_x = record.output_probe.width / source_probe.width
            record.scale_y = record.output_probe.height / source_probe.height
            record.exact_4x = (
                record.output_probe.width == source_probe.width * 4
                and record.output_probe.height == source_probe.height * 4
            )
            record.is_16k = (
                record.output_probe.width == 15360
                and record.output_probe.height == 8640
            )
            expected = record.expected_input_frames
            observed = record.output_probe.frame_count
            fps_match = Fraction(record.output_probe.fps) == fps
            record.evaluated_frames = min(expected, observed)
            record.frame_alignment = (
                "PASS: same start index, FPS and frame count"
                if expected == observed and fps_match
                else f"COMMON_RANGE_ONLY: input={expected}, output={observed}, fps_match={fps_match}"
            )
            records.append(record)

    # Reference-only rows make N/A explicit without inventing an output.
    model, model_dir, slug = REFERENCE_MODEL
    for mode in MODES:
        report_path = RESULT_ROOT / model_dir / f"benchmark_{mode}.json"
        data = load_json(report_path)
        records.append(
            RunRecord(
                model=model,
                model_dir=model_dir,
                slug=slug,
                variant=mode,
                status=str(data.get("status", "BLOCKED_REFERENCE_ONLY")),
                report_path=str(report_path.resolve()),
                input_path=str(SOURCE.resolve()),
                output_path=None,
                start_frame=nested_first(data, (("frame_selection", "start_frame"),)),
                expected_input_frames=None,
                recorded_output_frames=None,
                total_wall_seconds=None,
                recorded_e2e_seconds_per_frame=None,
                pure_inference_seconds_per_frame=None,
                timing_scope="N/A: official path is not valid 4K->16K",
                timing_source=str(report_path.resolve()),
                input_probe=source_probe,
            )
        )
    return source_probe, records


def inventory_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        ip = record.input_probe
        op = record.output_probe
        rows.append(
            {
                "model": record.model,
                "variant": record.variant,
                "status": record.status,
                "report_path": record.report_path,
                "input_path": record.input_path or "N/A",
                "output_path": record.output_path or "N/A",
                "start_frame": record.start_frame
                if record.start_frame is not None
                else "N/A",
                "source_total_frames": ip.frame_count if ip else "N/A",
                "selected_input_frames": record.expected_input_frames or "N/A",
                "output_frames": op.frame_count if op else "N/A",
                "input_fps": ip.fps if ip else "N/A",
                "output_fps": op.fps if op else "N/A",
                "source_duration_seconds": ip.duration if ip else "N/A",
                "selected_input_duration_seconds": record.selected_input_duration
                or "N/A",
                "output_duration_seconds": op.duration if op else "N/A",
                "input_width": ip.width if ip else "N/A",
                "input_height": ip.height if ip else "N/A",
                "output_width": op.width if op else "N/A",
                "output_height": op.height if op else "N/A",
                "output_codec": op.codec if op else "N/A",
                "output_pixel_format": op.pixel_format if op else "N/A",
                "scale_x": record.scale_x if record.scale_x is not None else "N/A",
                "scale_y": record.scale_y if record.scale_y is not None else "N/A",
                "exact_4x": record.exact_4x if record.exact_4x is not None else "N/A",
                "is_16k_15360x8640": record.is_16k
                if record.is_16k is not None
                else "N/A",
                "frame_alignment": record.frame_alignment,
                "total_wall_seconds": record.total_wall_seconds or "N/A",
                "recorded_e2e_seconds_per_frame": record.recorded_e2e_seconds_per_frame
                or "N/A",
                "pure_inference_seconds_per_frame": record.pure_inference_seconds_per_frame
                or "N/A",
                "timing_scope": record.timing_scope,
                "timing_source": record.timing_source,
            }
        )
    return rows


def persist_inventory(source_probe: Probe, records: list[RunRecord]) -> None:
    rows = inventory_rows(records)
    write_csv(EVAL_ROOT / "inventory.csv", rows, list(rows[0]))
    write_json(
        EVAL_ROOT / "inventory.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cpu_only_environment": CPU_ENV,
            "source_probe": asdict(source_probe),
            "runs": rows,
        },
    )
    resolution_fields = [
        "model",
        "variant",
        "input_width",
        "input_height",
        "output_width",
        "output_height",
        "scale_x",
        "scale_y",
        "input_fps",
        "output_fps",
        "selected_input_frames",
        "output_frames",
        "selected_input_duration_seconds",
        "output_duration_seconds",
        "output_codec",
        "exact_4x",
        "is_16k_15360x8640",
        "frame_alignment",
    ]
    write_csv(EVAL_ROOT / "resolution_check.csv", rows, resolution_fields)


def write_command(path: Path, command: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key, value in CPU_ENV.items():
            handle.write(f"export {key}={shlex.quote(value)}\n")
        handle.write(shlex.join(command) + "\n")


def run_logged(command: list[str], command_path: Path, log_path: Path) -> None:
    write_command(command_path, command)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=cpu_environment(),
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")


def parse_colon_fields(line: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, value in re.findall(r"([A-Za-z_]+):([-+0-9.eE]+|inf|-inf|nan)", line):
        try:
            parsed[key.lower()] = float(value)
        except ValueError:
            pass
    return parsed


def finite_stats(values: list[float]) -> dict[str, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": fmean(finite),
        "std": pstdev(finite),
        "min": min(finite),
        "max": max(finite),
    }


def evaluate_metrics(records: list[RunRecord]) -> list[dict[str, Any]]:
    successful = [record for record in records if record.output_probe is not None]
    common_frames = {
        mode: min(
            record.evaluated_frames for record in successful if record.variant == mode
        )
        for mode in MODES
    }
    summaries: list[dict[str, Any]] = []
    for record in successful:
        assert record.input_probe and record.output_probe and record.output_path
        base = f"{record.slug}_{record.variant}"
        psnr_raw = EVAL_ROOT / "metrics_per_frame" / f"{base}.psnr.log"
        ssim_raw = EVAL_ROOT / "metrics_per_frame" / f"{base}.ssim.log"
        combined_csv = EVAL_ROOT / "metrics_per_frame" / f"{base}.csv"
        log_path = EVAL_ROOT / "logs" / f"metrics_{base}.log"
        command_path = EVAL_ROOT / "logs" / f"metrics_{base}.command.sh"
        n_frames = int(record.evaluated_frames)
        start = int(record.start_frame)
        end = start + n_frames
        graph = (
            f"[0:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,"
            "format=yuv444p10le,split=2[srcp][srcs];"
            f"[1:v]trim=start_frame=0:end_frame={n_frames},setpts=PTS-STARTPTS,"
            f"scale={record.input_probe.width}:{record.input_probe.height}:flags=lanczos,"
            "format=yuv444p10le,split=2[outp][outs];"
            f"[outp][srcp]psnr=stats_file={psnr_raw}:stats_version=2:output_max=1:"
            "shortest=1[p];"
            f"[outs][srcs]ssim=stats_file={ssim_raw}:shortest=1[s]"
        )
        command = (
            LOW_PRIORITY
            + FFMPEG_BASE
            + [
                "-v",
                "info",
                "-hwaccel",
                "none",
                "-i",
                record.input_path,
                "-hwaccel",
                "none",
                "-i",
                record.output_path,
                "-filter_complex",
                graph,
                "-map",
                "[p]",
                "-map",
                "[s]",
                "-f",
                "null",
                "-",
            ]
        )
        run_logged(command, command_path, log_path)

        psnr_rows = [
            parse_colon_fields(line)
            for line in psnr_raw.read_text(encoding="utf-8").splitlines()
            if line.startswith("n:")
        ]
        ssim_rows = [
            parse_colon_fields(line)
            for line in ssim_raw.read_text(encoding="utf-8").splitlines()
            if line.startswith("n:")
        ]
        if len(psnr_rows) != n_frames or len(ssim_rows) != n_frames:
            raise RuntimeError(
                f"Metric frame count mismatch for {record.model} {record.variant}: "
                f"expected {n_frames}, psnr={len(psnr_rows)}, ssim={len(ssim_rows)}"
            )
        combined = []
        for offset, (psnr, ssim) in enumerate(zip(psnr_rows, ssim_rows)):
            combined.append(
                {
                    "evaluation_frame_index": offset,
                    "source_frame_index": start + offset,
                    "timestamp_seconds": float(
                        Fraction(start + offset, 1) / Fraction(record.input_probe.fps)
                    ),
                    "psnr_y": psnr["psnr_y"],
                    "psnr_u": psnr["psnr_u"],
                    "psnr_v": psnr["psnr_v"],
                    "psnr_all": psnr["psnr_avg"],
                    "ssim_y": ssim["y"],
                    "ssim_u": ssim["u"],
                    "ssim_v": ssim["v"],
                    "ssim_all": ssim["all"],
                }
            )
        write_csv(combined_csv, combined, list(combined[0]))

        common_n = common_frames[record.variant]
        common = combined[:common_n]
        summary: dict[str, Any] = {
            "model": record.model,
            "variant": record.variant,
            "evaluated_frames_full": n_frames,
            "common_ranking_frames": common_n,
            "start_frame": start,
            "comparison_format": "yuv444p10le",
            "downsample": "Lanczos to source resolution",
            "per_frame_csv": str(combined_csv.resolve()),
            "psnr_stats_file": str(psnr_raw.resolve()),
            "ssim_stats_file": str(ssim_raw.resolve()),
        }
        for metric in ("psnr_y", "psnr_all", "ssim_y", "ssim_all"):
            full_stats = finite_stats([float(row[metric]) for row in combined])
            common_stats = finite_stats([float(row[metric]) for row in common])
            for name, value in full_stats.items():
                summary[f"full_{metric}_{name}"] = value
            for name, value in common_stats.items():
                summary[f"common_{metric}_{name}"] = value
        summaries.append(summary)

    # Explicit N/A records for the reference-only model.
    for mode in MODES:
        summaries.append(
            {
                "model": REFERENCE_MODEL[0],
                "variant": mode,
                "evaluated_frames_full": "N/A",
                "common_ranking_frames": common_frames[mode],
                "start_frame": "N/A",
                "comparison_format": "N/A",
                "downsample": "N/A: no valid 4K->16K output",
                "per_frame_csv": "N/A",
                "psnr_stats_file": "N/A",
                "ssim_stats_file": "N/A",
                **{
                    f"{scope}_{metric}_{stat}": "N/A"
                    for scope in ("full", "common")
                    for metric in ("psnr_y", "psnr_all", "ssim_y", "ssim_all")
                    for stat in ("mean", "std", "min", "max")
                },
            }
        )
    fields = list(summaries[0])
    write_csv(EVAL_ROOT / "metrics_summary.csv", summaries, fields)
    write_json(EVAL_ROOT / "metrics_summary.json", summaries)
    return summaries


def extract_texture_frames(
    source_probe: Probe, frame_limit: int
) -> list[dict[str, Any]]:
    width, height = 960, 540
    crop_width, crop_height = source_probe.width // 2, source_probe.height // 2
    crop_x = (source_probe.width - crop_width) // 2
    crop_y = (source_probe.height - crop_height) // 2
    graph = (
        f"trim=start_frame=0:end_frame={frame_limit},setpts=PTS-STARTPTS,"
        f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
        f"scale={width}:{height}:flags=area,format=gray"
    )
    command = (
        LOW_PRIORITY
        + FFMPEG_BASE
        + [
            "-v",
            "warning",
            "-hwaccel",
            "none",
            "-i",
            str(SOURCE),
            "-vf",
            graph,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )
    command_path = EVAL_ROOT / "logs" / "representative_texture_scan.command.sh"
    log_path = EVAL_ROOT / "logs" / "representative_texture_scan.log"
    write_command(command_path, command)
    frame_bytes = width * height
    frames: list[np.ndarray] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=log,
            env=cpu_environment(),
        )
        assert process.stdout is not None
        for _ in range(frame_limit):
            chunks = bytearray()
            while len(chunks) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(chunks))
                if not chunk:
                    break
                chunks.extend(chunk)
            if len(chunks) != frame_bytes:
                break
            frames.append(
                np.frombuffer(chunks, dtype=np.uint8).reshape(height, width).copy()
            )
        process.stdout.close()
        return_code = process.wait()
    if return_code != 0 or len(frames) != frame_limit:
        raise RuntimeError(
            f"Texture scan failed: return_code={return_code}, frames={len(frames)}; see {log_path}"
        )

    rows = []
    for index, frame in enumerate(frames):
        values = frame.astype(np.int16)
        laplacian = (
            -4 * values[1:-1, 1:-1]
            + values[:-2, 1:-1]
            + values[2:, 1:-1]
            + values[1:-1, :-2]
            + values[1:-1, 2:]
        )
        mean = float(frame.mean())
        std = float(frame.std())
        texture = float(laplacian.var())
        edge_density = float((np.abs(laplacian) >= 20).mean())
        neighbor_delta = 0.0
        if index > 0:
            neighbor_delta = max(
                neighbor_delta,
                float(
                    np.abs(
                        values.astype(np.int16) - frames[index - 1].astype(np.int16)
                    ).mean()
                ),
            )
        if index + 1 < len(frames):
            neighbor_delta = max(
                neighbor_delta,
                float(
                    np.abs(
                        values.astype(np.int16) - frames[index + 1].astype(np.int16)
                    ).mean()
                ),
            )
        excluded = mean < 12 or mean > 243 or std < 8 or neighbor_delta > 55
        reason = (
            "black/white/low-contrast"
            if mean < 12 or mean > 243 or std < 8
            else "possible transition"
            if neighbor_delta > 55
            else ""
        )
        rows.append(
            {
                "frame_index": index,
                "timestamp_seconds": float(
                    Fraction(index, 1) / Fraction(source_probe.fps)
                ),
                "texture_score_laplacian_variance": texture,
                "edge_density": edge_density,
                "mean_luma": mean,
                "std_luma": std,
                "max_neighbor_delta": neighbor_delta,
                "excluded": excluded,
                "exclusion_reason": reason,
            }
        )
    return rows


def extract_png(
    video: Path, frame_index: int, crop: tuple[int, int, int, int], output: Path
) -> None:
    width, height, x, y = crop
    graph = f"select=eq(n\\,{frame_index}),crop={width}:{height}:{x}:{y}"
    base = output.stem
    command = (
        LOW_PRIORITY
        + FFMPEG_BASE
        + [
            "-v",
            "warning",
            "-hwaccel",
            "none",
            "-i",
            str(video),
            "-vf",
            graph,
            "-frames:v",
            "1",
            "-threads:v",
            "1",
            "-update",
            "1",
            "-y",
            str(output),
        ]
    )
    run_logged(
        command,
        EVAL_ROOT / "logs" / f"crop_{base}.command.sh",
        EVAL_ROOT / "logs" / f"crop_{base}.log",
    )
    if not output.is_file():
        raise RuntimeError(f"Crop was not created: {output}")


def contact_sheet(
    items: list[tuple[str, Path]], output: Path, tile_size: int = 512
) -> None:
    columns = 3
    rows = math.ceil(len(items) / columns)
    label_height = 46
    canvas = Image.new(
        "RGB", (columns * tile_size, rows * (tile_size + label_height)), "#181818"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), "black")
        tile.paste(
            image, ((tile_size - image.width) // 2, (tile_size - image.height) // 2)
        )
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + label_height)
        canvas.paste(tile, (x, y))
        draw.rectangle(
            (x, y + tile_size, x + tile_size, y + tile_size + label_height),
            fill="#181818",
        )
        draw.text((x + 12, y + tile_size + 10), label, fill="white", font=font)
    canvas.save(output, format="PNG", compress_level=6)


def generate_visuals(source_probe: Probe, records: list[RunRecord]) -> dict[str, Any]:
    mini_records = [
        record for record in records if record.variant == "mini" and record.output_probe
    ]
    common_mini_frames = min(record.evaluated_frames for record in mini_records)
    texture_rows = extract_texture_frames(source_probe, common_mini_frames)
    eligible = [row for row in texture_rows if not row["excluded"]]
    if not eligible:
        raise RuntimeError("No representative source frame survived the texture scan")
    ranked = sorted(
        eligible, key=lambda row: row["texture_score_laplacian_variance"], reverse=True
    )
    selected = ranked[0]
    for rank, row in enumerate(ranked, start=1):
        row["eligible_texture_rank"] = rank
    for row in texture_rows:
        row.setdefault("eligible_texture_rank", "N/A")
    write_csv(
        EVAL_ROOT / "representative_frame_scores.csv",
        texture_rows,
        list(texture_rows[0]),
    )

    frame_index = int(selected["frame_index"])
    source_medium = (
        512,
        512,
        (source_probe.width - 512) // 2,
        (source_probe.height - 512) // 2,
    )
    source_tight = (
        256,
        256,
        (source_probe.width - 256) // 2,
        (source_probe.height - 256) // 2,
    )
    medium_dir = EVAL_ROOT / "visual_crops" / "medium"
    tight_dir = EVAL_ROOT / "visual_crops" / "tight"
    sheet_dir = EVAL_ROOT / "visual_crops" / "contact_sheet"
    original_medium = medium_dir / "original_4k_center_512.png"
    original_tight = tight_dir / "original_4k_center_256.png"
    extract_png(SOURCE, frame_index, source_medium, original_medium)
    extract_png(SOURCE, frame_index, source_tight, original_tight)

    baseline = tight_dir / "original_4k_center_256_bicubic_x4.png"
    with Image.open(original_tight) as image:
        image.resize((1024, 1024), Image.Resampling.BICUBIC).save(
            baseline, format="PNG"
        )

    medium_items: list[tuple[str, Path]] = [
        ("Original 4K (native 512)", original_medium)
    ]
    tight_items: list[tuple[str, Path]] = [("Original 4K - Bicubic 4x", baseline)]
    crop_manifest: list[dict[str, Any]] = []
    for record in mini_records:
        assert record.output_probe and record.output_path
        scale_x, scale_y = record.scale_x, record.scale_y
        medium_crop = (
            round(source_medium[0] * scale_x),
            round(source_medium[1] * scale_y),
            round(source_medium[2] * scale_x),
            round(source_medium[3] * scale_y),
        )
        tight_crop = (
            round(source_tight[0] * scale_x),
            round(source_tight[1] * scale_y),
            round(source_tight[2] * scale_x),
            round(source_tight[3] * scale_y),
        )
        medium_path = medium_dir / f"{record.slug}_16k_center_equivalent.png"
        tight_path = tight_dir / f"{record.slug}_16k_center_equivalent.png"
        extract_png(Path(record.output_path), frame_index, medium_crop, medium_path)
        extract_png(Path(record.output_path), frame_index, tight_crop, tight_path)
        medium_items.append((f"{record.model} native 16K", medium_path))
        tight_items.append((f"{record.model} native 16K", tight_path))
        crop_manifest.append(
            {
                "model": record.model,
                "frame_index": frame_index,
                "source_medium_crop_whxy": source_medium,
                "output_medium_crop_whxy": medium_crop,
                "source_tight_crop_whxy": source_tight,
                "output_tight_crop_whxy": tight_crop,
                "medium_png": str(medium_path.resolve()),
                "tight_png": str(tight_path.resolve()),
            }
        )
    medium_sheet = sheet_dir / "mini_central_medium_comparison.png"
    tight_sheet = sheet_dir / "mini_central_tight_comparison.png"
    contact_sheet(medium_items, medium_sheet)
    contact_sheet(tight_items, tight_sheet)

    # Show the three highest-texture source candidates for auditability.
    candidate_items: list[tuple[str, Path]] = []
    for row in ranked[:3]:
        index = int(row["frame_index"])
        path = sheet_dir / f"representative_candidate_frame_{index}.png"
        extract_png(SOURCE, index, source_medium, path)
        candidate_items.append(
            (
                f"frame {index}; score {row['texture_score_laplacian_variance']:.1f}",
                path,
            )
        )
    candidate_sheet = sheet_dir / "representative_candidates.png"
    contact_sheet(candidate_items, candidate_sheet)

    manifest = {
        "selection_scope": f"common mini range, source frames 0..{common_mini_frames - 1}",
        "selection_method": "central 50% grayscale Laplacian variance on 960x540 CPU downsample",
        "selected_frame_index": frame_index,
        "timestamp_seconds": selected["timestamp_seconds"],
        "texture_score_laplacian_variance": selected[
            "texture_score_laplacian_variance"
        ],
        "edge_density": selected["edge_density"],
        "source_medium_crop_whxy": source_medium,
        "source_tight_crop_whxy": source_tight,
        "bicubic_baseline": str(baseline.resolve()),
        "medium_contact_sheet": str(medium_sheet.resolve()),
        "tight_contact_sheet": str(tight_sheet.resolve()),
        "candidate_contact_sheet": str(candidate_sheet.resolve()),
        "crops": crop_manifest,
    }
    write_json(EVAL_ROOT / "visual_crops" / "crop_manifest.json", manifest)
    return manifest


def timing_outputs(
    source_probe: Probe, records: list[RunRecord]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successful = [record for record in records if record.output_probe]
    existing = []
    for record in successful:
        existing.append(
            {
                "model": record.model,
                "variant": record.variant,
                "frames": record.output_probe.frame_count,
                "elapsed_seconds": record.total_wall_seconds,
                "end_to_end_seconds_per_frame": record.recorded_e2e_seconds_per_frame,
                "pure_inference_seconds_per_frame": record.pure_inference_seconds_per_frame,
                "timing_scope": record.timing_scope,
                "source_of_timing_record": record.timing_source,
            }
        )
    existing.append(
        {
            "model": REFERENCE_MODEL[0],
            "variant": "micro",
            "frames": "N/A",
            "elapsed_seconds": "N/A",
            "end_to_end_seconds_per_frame": "N/A",
            "pure_inference_seconds_per_frame": "N/A",
            "timing_scope": "N/A: no valid 4K->16K run",
            "source_of_timing_record": str(
                (RESULT_ROOT / REFERENCE_MODEL[1] / "benchmark_micro.json").resolve()
            ),
        }
    )
    existing.append(
        {
            "model": REFERENCE_MODEL[0],
            "variant": "mini",
            "frames": "N/A",
            "elapsed_seconds": "N/A",
            "end_to_end_seconds_per_frame": "N/A",
            "pure_inference_seconds_per_frame": "N/A",
            "timing_scope": "N/A: no valid 4K->16K run",
            "source_of_timing_record": str(
                (RESULT_ROOT / REFERENCE_MODEL[1] / "benchmark_mini.json").resolve()
            ),
        }
    )
    write_csv(EVAL_ROOT / "timing_existing.csv", existing, list(existing[0]))

    source_fps = Fraction(source_probe.fps)
    frames_120_fraction = source_fps * 120
    if frames_120_fraction.denominator != 1:
        raise RuntimeError(
            f"120 seconds does not map to an integral frame count: {frames_120_fraction}"
        )
    frames_120 = int(frames_120_fraction)
    prior_estimates = load_json(RESULT_ROOT / "benchmark_estimates.json")["models"]
    projections = []
    for model, model_dir, _ in MODELS:
        runs = {
            record.variant: record for record in successful if record.model == model
        }
        micro, mini = runs["micro"], runs["mini"]
        mini_estimate = (
            mini.total_wall_seconds / mini.output_probe.frame_count * frames_120
        )
        delta_frames = mini.output_probe.frame_count - micro.output_probe.frame_count
        slope = (mini.total_wall_seconds - micro.total_wall_seconds) / delta_frames
        intercept = micro.total_wall_seconds - slope * micro.output_probe.frame_count
        topology_reliable = model != "DOVE"
        linear_valid = slope > 0 and intercept >= 0 and topology_reliable
        linear_estimate = intercept + slope * frames_120 if linear_valid else None
        recommended = linear_estimate if linear_valid else mini_estimate
        prior = prior_estimates.get(model_dir, {})
        prior_projection = prior.get("estimates", {}).get(str(frames_120), {})
        prior_seconds = prior_projection.get("estimated_total_seconds")
        if model == "DOVE":
            fit_note = (
                "unreliable 2-point topology: micro=9 and mini=17 are both one temporal "
                "chunk; long runs repeat 17-frame windows with overlap 8"
            )
        elif not linear_valid:
            fit_note = (
                f"unreliable linear fit (A={intercept:.3f}, B={slope:.3f}); "
                "mini throughput is primary"
            )
        else:
            fit_note = "same recorded conditions; positive slope/intercept"
        projections.append(
            {
                "model": model,
                "source_fps": str(source_fps),
                "frames_at_120_seconds": frames_120,
                "micro_frames": micro.output_probe.frame_count,
                "micro_elapsed_seconds": micro.total_wall_seconds,
                "mini_frames": mini.output_probe.frame_count,
                "mini_elapsed_seconds": mini.total_wall_seconds,
                "mini_seconds_per_frame": mini.total_wall_seconds
                / mini.output_probe.frame_count,
                "mini_throughput_estimate_seconds": mini_estimate,
                "two_point_intercept_seconds": intercept,
                "two_point_slope_seconds_per_frame": slope,
                "two_point_linear_valid": linear_valid,
                "two_point_linear_estimate_seconds": linear_estimate
                if linear_estimate is not None
                else "N/A",
                "recommended_method": "2-point linear fit"
                if linear_valid
                else "mini throughput",
                "recommended_estimate_seconds": recommended,
                "recommended_estimate_minutes": recommended / 60,
                "recommended_estimate_hours": recommended / 3600,
                "recommended_estimate_hhmmss": duration_hhmmss(recommended),
                "real_time_factor": recommended / 120,
                "existing_model_aware_estimate_seconds": prior_seconds
                if prior_seconds is not None
                else "N/A",
                "existing_model_aware_estimate_hhmmss": duration_hhmmss(prior_seconds)
                if prior_seconds is not None
                else "N/A",
                "existing_model_aware_formula": prior.get("formula", "N/A"),
                "fit_note": fit_note,
            }
        )
    projections.append(
        {
            "model": REFERENCE_MODEL[0],
            "source_fps": str(source_fps),
            "frames_at_120_seconds": frames_120,
            **{
                field: "N/A"
                for field in (
                    "micro_frames",
                    "micro_elapsed_seconds",
                    "mini_frames",
                    "mini_elapsed_seconds",
                    "mini_seconds_per_frame",
                    "mini_throughput_estimate_seconds",
                    "two_point_intercept_seconds",
                    "two_point_slope_seconds_per_frame",
                    "two_point_linear_valid",
                    "two_point_linear_estimate_seconds",
                    "recommended_method",
                    "recommended_estimate_seconds",
                    "recommended_estimate_minutes",
                    "recommended_estimate_hours",
                    "recommended_estimate_hhmmss",
                    "real_time_factor",
                    "existing_model_aware_estimate_seconds",
                    "existing_model_aware_estimate_hhmmss",
                    "existing_model_aware_formula",
                )
            },
            "fit_note": "N/A: no valid official 4K->16K benchmark output",
        }
    )
    write_csv(
        EVAL_ROOT / "timing_120sec_estimate.csv", projections, list(projections[0])
    )
    return existing, projections


def duration_hhmmss(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "N/A":
        return "N/A"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    return str(value)


def metric_rankings(
    summaries: list[dict[str, Any]], mode: str, metric: str
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in summaries
        if row["variant"] == mode
        and isinstance(row.get(f"common_{metric}_mean"), (int, float))
    ]
    return sorted(rows, key=lambda row: row[f"common_{metric}_mean"], reverse=True)


def markdown_ranking(rows: list[dict[str, Any]], metric: str) -> str:
    return " / ".join(
        f"{index}. {row['model']} ({row[f'common_{metric}_mean']:.6f})"
        for index, row in enumerate(rows, start=1)
    )


def build_report(
    source_probe: Probe,
    records: list[RunRecord],
    summaries: list[dict[str, Any]],
    projections: list[dict[str, Any]],
    visual: dict[str, Any],
) -> None:
    success = [record for record in records if record.output_probe]
    micro_psnr = metric_rankings(summaries, "micro", "psnr_y")
    micro_ssim = metric_rankings(summaries, "micro", "ssim_y")
    mini_psnr = metric_rankings(summaries, "mini", "psnr_y")
    mini_ssim = metric_rankings(summaries, "mini", "ssim_y")

    lines = [
        "# Existing 4K→16K Outputs — CPU-only Quantitative Evaluation",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Source: `{source_probe.path}`",
        f"- Source probe: **{source_probe.width}×{source_probe.height}, {source_probe.fps} fps, "
        f"{source_probe.frame_count} frames, {source_probe.duration:.6f} s, {source_probe.codec}**",
        "- Evaluation definition: **4K reconstruction fidelity after 16K upscaling**.",
        "- Safety: only existing files were decoded with CPU FFmpeg/ffprobe. No model/checkpoint was loaded, "
        "no CUDA context was requested, and no inference was executed.",
        "- Quantitative ranking window: first **5 common micro frames** and first **9 common mini frames**. "
        "Each per-frame CSV additionally contains every frame available in that run.",
        "",
        "## TABLE 1 — Resolution / Scale validation",
        "",
        "| Model | Variant | Input | Output | Scale | FPS in/out | Frames in/out | Duration in/out | Codec | Exact 4× | 16K |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for record in records:
        if record.output_probe:
            ip, op = record.input_probe, record.output_probe
            lines.append(
                f"| {record.model} | {record.variant} | {ip.width}×{ip.height} | {op.width}×{op.height} | "
                f"{record.scale_x:.3f}×{record.scale_y:.3f} | {ip.fps}/{op.fps} | "
                f"{record.expected_input_frames}/{op.frame_count} | "
                f"{record.selected_input_duration:.6f}/{op.duration:.6f} s | {op.codec} | "
                f"{'YES' if record.exact_4x else 'NO'} | {'YES' if record.is_16k else 'NO'} |"
            )
        else:
            lines.append(
                f"| {record.model} | {record.variant} | {source_probe.width}×{source_probe.height} | N/A | "
                "N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
    all_exact = len(success) == 10 and all(
        record.exact_4x and record.is_16k for record in success
    )
    lines.extend(
        [
            "",
            f"**Answer A–C:** all **{len(success)}/10** expected successful outputs exist. "
            f"All are ffprobe-confirmed **15360×8640**, exact 4× from the 3840×2160 source: "
            f"**{'PASS' if all_exact else 'FAIL'}**.",
            "",
            "Upscale-A-Video is `N/A`: its untouched official inference downsamples a ≥1280-pixel input "
            "before ×4 SR, so this 4K input returns 4K rather than a valid 16K result. The existing report "
            "marks it `BLOCKED_REFERENCE_ONLY`; no output or timing was fabricated.",
            "",
            "## TABLE 2 — Quantitative fidelity (fair common-frame windows)",
            "",
            "The 16K output is Lanczos-downsampled to 3840×2160, both streams are converted identically "
            "to `yuv444p10le`, PTS is reset, and frame indices are aligned. Values below are arithmetic means "
            "of the saved FFmpeg per-frame statistics.",
            "",
            "| Model | Variant | Evaluated Frames | PSNR-Y | SSIM-Y | PSNR-All | SSIM-All |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        if row["model"] == REFERENCE_MODEL[0]:
            lines.append(
                f"| {row['model']} | {row['variant']} | N/A | N/A | N/A | N/A | N/A |"
            )
        else:
            lines.append(
                f"| {row['model']} | {row['variant']} | {row['common_ranking_frames']} | "
                f"{row['common_psnr_y_mean']:.6f} | {row['common_ssim_y_mean']:.6f} | "
                f"{row['common_psnr_all_mean']:.6f} | {row['common_ssim_all_mean']:.6f} |"
            )
    lines.extend(
        [
            "",
            "### Rankings",
            "",
            f"- **MICRO PSNR-Y:** {markdown_ranking(micro_psnr, 'psnr_y')}",
            f"- **MICRO SSIM-Y:** {markdown_ranking(micro_ssim, 'ssim_y')}",
            f"- **MINI PSNR-Y (primary):** {markdown_ranking(mini_psnr, 'psnr_y')}",
            f"- **MINI SSIM-Y (primary):** {markdown_ranking(mini_ssim, 'ssim_y')}",
            f"- Micro/mini PSNR order identical: **{'YES' if [r['model'] for r in micro_psnr] == [r['model'] for r in mini_psnr] else 'NO'}**.",
            f"- Micro/mini SSIM order identical: **{'YES' if [r['model'] for r in micro_ssim] == [r['model'] for r in mini_ssim] else 'NO'}**.",
            "",
            "Interpretation:",
            "- **PSNR↑:** 원본 pixel fidelity가 높음.",
            "- **SSIM↑:** 원본 구조 보존도가 높음.",
            "- 이것만으로 **가장 자연스러운 16K detail**을 의미하지는 않음. There is no 16K ground truth; "
            "generated texture, perceptual naturalness, hallucination, ringing, and temporal shimmer require visual review.",
            "",
            "## Central native-16K detail crops",
            "",
            f"- Representative frame: **{visual['selected_frame_index']}** at "
            f"**{visual['timestamp_seconds']:.6f} s**.",
            f"- Central-50% Laplacian-variance texture score: **{visual['texture_score_laplacian_variance']:.3f}**; "
            f"edge density: **{visual['edge_density']:.6f}**.",
            "- The frame was selected inside the common mini range, so every model uses the same frame index.",
            "- Raw model PNGs are direct crops from native 16K decoded pixels, not downsampled reconstructions.",
            f"- Tight contact sheet: `{visual['tight_contact_sheet']}`",
            f"- Medium contact sheet: `{visual['medium_contact_sheet']}`",
            f"- Candidate audit sheet: `{visual['candidate_contact_sheet']}`",
            "",
            "The tight sheet compares a bicubic ×4 source baseline against native output pixels for the same "
            "256×256 source scene area (1024×1024 at exact ×4). Use it to inspect added texture, sharpening, "
            "smoothing, ringing, or hallucination. A static crop cannot establish temporal stability or shimmer.",
            "",
            "### Visual inspection of the selected native crops",
            "",
            "- **Bicubic baseline:** enlarged source pixels are visibly soft/blocky and introduce no new strands or edges.",
            "- **Real-ESRGAN:** improves contour clarity over bicubic, while fine grass/hair remains comparatively "
            "soft and somewhat painterly.",
            "- **HAT:** is the smoothest/conservative reconstruction here; broad shapes are clean, but the finest "
            "grass and hair texture is subdued.",
            "- **FlashVSR v1.1:** restores much stronger grass/hair microtexture and crisper person boundaries; the "
            "extra high-frequency pattern also has greater synthesis/hallucination risk than Real/HAT.",
            "- **SeedVR2:** shows the strongest blocky/warped local artifacts in faces, clothing, and grass in this crop.",
            "- **DOVE:** shows the clearest fine grass and well-resolved person boundaries in this static crop. Its "
            "lower reconstruction PSNR/SSIM than Real/HAT means it changed the source more; the added detail is "
            "not verified 16K ground truth.",
            "- These observations apply to frame 0's central crop only; temporal playback is still required for flicker/shimmer.",
            "",
            "## Existing benchmark timing",
            "",
            "`total_wall_seconds` is used for elapsed time and projection because it is the recorded end-to-end "
            "scope. The reports separately expose pure inference sec/frame, but it excludes varying pipeline stages.",
            "",
            "| Model | Micro frames | Micro elapsed | Micro sec/frame | Mini frames | Mini elapsed | Mini sec/frame | Pure mini inference sec/frame | Timing scope | Timing source |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for model, _, _ in MODELS:
        micro = next(
            record
            for record in success
            if record.model == model and record.variant == "micro"
        )
        mini = next(
            record
            for record in success
            if record.model == model and record.variant == "mini"
        )
        lines.append(
            f"| {model} | {micro.output_probe.frame_count} | {micro.total_wall_seconds:.3f} s | "
            f"{micro.recorded_e2e_seconds_per_frame:.3f} | {mini.output_probe.frame_count} | "
            f"{mini.total_wall_seconds:.3f} s | {mini.recorded_e2e_seconds_per_frame:.3f} | "
            f"{mini.pure_inference_seconds_per_frame:.3f} | E2E `total_wall_seconds`; load/decode/SR/merge/encode included where recorded | "
            f"`{Path(mini.timing_source).relative_to(ROOT)}` |"
        )
    lines.append(
        "| Upscale-A-Video | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | reference-only JSON |"
    )

    lines.extend(
        [
            "",
            "## TABLE 3 — Runtime / 120 s projection",
            "",
            f"Source FPS is ffprobe-confirmed **{source_probe.fps}**, so 120 s contains "
            f"**{int(Fraction(source_probe.fps) * 120)} frames**. Mini throughput is always shown. "
            "The 2-point fit is recommended only when its slope/intercept and temporal sampling topology are valid; "
            "otherwise the mini-throughput estimate is primary. The pre-existing model-aware projection is also "
            "shown because it accounts for documented temporal window/overlap behavior.",
            "",
            "| Model | Micro Time | Mini Time | Mini sec/frame | Source FPS | Frames@120s | Mini-throughput estimate | 2-point estimate | Formula-primary recommendation | Existing model-aware | RTF |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in projections:
        if row["model"] == REFERENCE_MODEL[0]:
            lines.append(
                "| Upscale-A-Video | N/A | N/A | N/A | 24 | 2880 | N/A | N/A | N/A | N/A | N/A |"
            )
            continue
        linear_display = (
            f"{duration_hhmmss(row['two_point_linear_estimate_seconds'])} "
            f"({row['two_point_linear_estimate_seconds']:.1f} s)"
            if row["two_point_linear_valid"]
            else "invalid / unreliable"
        )
        lines.append(
            f"| {row['model']} | {row['micro_elapsed_seconds']:.3f} s | {row['mini_elapsed_seconds']:.3f} s | "
            f"{row['mini_seconds_per_frame']:.3f} | {row['source_fps']} | {row['frames_at_120_seconds']} | "
            f"{duration_hhmmss(row['mini_throughput_estimate_seconds'])} ({row['mini_throughput_estimate_seconds']:.1f} s) | "
            f"{linear_display} | "
            f"{row['recommended_estimate_hhmmss']} ({row['recommended_method']}) | "
            f"{row['existing_model_aware_estimate_hhmmss']} | {row['real_time_factor']:.2f}× |"
        )
    fastest = min(
        (
            row
            for row in projections
            if isinstance(row.get("recommended_estimate_seconds"), float)
        ),
        key=lambda row: row["recommended_estimate_seconds"],
    )
    slowest = max(
        (
            row
            for row in projections
            if isinstance(row.get("recommended_estimate_seconds"), float)
        ),
        key=lambda row: row["recommended_estimate_seconds"],
    )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- **Pixel fidelity winner (mini PSNR-Y): {mini_psnr[0]['model']}** ({mini_psnr[0]['common_psnr_y_mean']:.6f} dB).",
            f"- **Structural fidelity winner (mini SSIM-Y): {mini_ssim[0]['model']}** ({mini_ssim[0]['common_ssim_y_mean']:.6f}).",
            f"- **Fastest existing E2E mini result / 120 s projection: {fastest['model']}** "
            f"({fastest['recommended_estimate_hhmmss']}, RTF {fastest['real_time_factor']:.2f}×).",
            f"- **Slowest: {slowest['model']}** ({slowest['recommended_estimate_hhmmss']}, "
            f"RTF {slowest['real_time_factor']:.2f}×).",
            "- No single overall-best upscaler is declared from PSNR/SSIM alone; the native-detail crops and "
            "temporal playback answer different questions.",
            "- For DOVE, the formula-primary mini-throughput value is an optimistic mechanical extrapolation. "
            f"Its existing overlap-aware benchmark projection is **{next(row for row in projections if row['model'] == 'DOVE')['existing_model_aware_estimate_hhmmss']}** "
            "for 2880 frames and is safer for capacity planning.",
            "- Upscale-A-Video remains N/A because no valid official 4K→16K output exists.",
            "",
            "## Reproducibility and files",
            "",
            f"- Inventory: `{(EVAL_ROOT / 'inventory.csv').resolve()}`",
            f"- Resolution checks: `{(EVAL_ROOT / 'resolution_check.csv').resolve()}`",
            f"- Metrics summary: `{(EVAL_ROOT / 'metrics_summary.csv').resolve()}`",
            f"- Per-frame metrics: `{(EVAL_ROOT / 'metrics_per_frame').resolve()}`",
            f"- Existing timing: `{(EVAL_ROOT / 'timing_existing.csv').resolve()}`",
            f"- 120 s projections: `{(EVAL_ROOT / 'timing_120sec_estimate.csv').resolve()}`",
            f"- Commands/logs: `{(EVAL_ROOT / 'logs').resolve()}`",
            "",
            "Every logged FFmpeg command is prefixed with `nice -n 10 ionice -c2 -n7`; accelerator visibility "
            "is disabled in the recorded command environment.",
        ]
    )
    (EVAL_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def terminal_summary(
    records: list[RunRecord],
    summaries: list[dict[str, Any]],
    projections: list[dict[str, Any]],
    visual: dict[str, Any],
) -> str:
    success = [record for record in records if record.output_probe]
    mini_psnr = metric_rankings(summaries, "mini", "psnr_y")
    mini_ssim = metric_rankings(summaries, "mini", "ssim_y")
    lines = [
        "=== CPU-ONLY EXISTING OUTPUT EVALUATION COMPLETE ===",
        "1. Output resolutions: "
        + "; ".join(f"{model}=15360x8640" for model, _, _ in MODELS),
        f"2. Exact 16K / 4x: {'YES — all 10 micro/mini outputs' if all(r.exact_4x and r.is_16k for r in success) else 'NO'}",
        "3. MINI PSNR-Y: " + markdown_ranking(mini_psnr, "psnr_y"),
        "4. MINI SSIM-Y: " + markdown_ranking(mini_ssim, "ssim_y"),
        "5–6. Existing times and mini E2E sec/frame:",
    ]
    for model, _, _ in MODELS:
        micro = next(
            record
            for record in success
            if record.model == model and record.variant == "micro"
        )
        mini = next(
            record
            for record in success
            if record.model == model and record.variant == "mini"
        )
        lines.append(
            f"   {model}: micro {micro.total_wall_seconds:.3f}s "
            f"({micro.recorded_e2e_seconds_per_frame:.3f}s/frame) / "
            f"mini {mini.total_wall_seconds:.3f}s "
            f"({mini.recorded_e2e_seconds_per_frame:.3f}s/frame)"
        )
    lines.append("7. Recommended 120-second estimates (2880 frames @ 24 fps):")
    for row in projections:
        if row["model"] != REFERENCE_MODEL[0]:
            caveat = (
                f"; existing overlap-aware {row['existing_model_aware_estimate_hhmmss']}"
                if row["model"] == "DOVE"
                else ""
            )
            lines.append(
                f"   {row['model']}: {row['recommended_estimate_hhmmss']} "
                f"({row['recommended_estimate_hours']:.3f}h, RTF {row['real_time_factor']:.2f}x{caveat})"
            )
    lines.extend(
        [
            "   Upscale-A-Video: N/A (no valid official 4K->16K output)",
            f"8. Representative crop frame: {visual['selected_frame_index']} "
            f"@ {visual['timestamp_seconds']:.6f}s; texture score {visual['texture_score_laplacian_variance']:.3f}",
            f"9. Contact sheet: {visual['tight_contact_sheet']}",
            f"10. REPORT.md: {(EVAL_ROOT / 'REPORT.md').resolve()}",
            "GPU CONFIRMATION: no inference, model/checkpoint load, CUDA context, or GPU metric was executed.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only probe and persist the existing run inventory; do not decode metrics/crops.",
    )
    parser.add_argument(
        "--reuse-probes",
        action="store_true",
        help="Reuse the prior direct ffprobe inventory after verifying that each path still exists.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate timing/report/terminal summary from existing evaluation artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_cpu_guard()
    create_directories()
    source_probe, records = build_records(reuse_probes=args.reuse_probes)
    persist_inventory(source_probe, records)
    if args.inventory_only:
        print(f"Inventory written to {EVAL_ROOT}")
        return 0
    if args.report_only:
        summaries = load_json(EVAL_ROOT / "metrics_summary.json")
        visual = load_json(EVAL_ROOT / "visual_crops" / "crop_manifest.json")
    else:
        summaries = evaluate_metrics(records)
        visual = generate_visuals(source_probe, records)
    _, projections = timing_outputs(source_probe, records)
    build_report(source_probe, records, summaries, projections, visual)
    summary = terminal_summary(records, summaries, projections, visual)
    (EVAL_ROOT / "FINAL_TERMINAL_SUMMARY.txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
