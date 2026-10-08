#!/usr/bin/env python3
"""Sequential micro/mini benchmark launcher for official pretrained models."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INPUT = ROOT / "input" / "0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "benchmark"
DEFAULT_RESULT_ROOT = ROOT / "result" / "benchmark"
MODELS = (
    "FlashVSR_v1.1",
    "SeedVR2",
    "DOVE",
    "Upscale-A-Video",
    "Real-ESRGAN",
    "HAT",
)
MODES = ("micro", "mini")
PYTHONS = {
    "FlashVSR_v1.1": Path("/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-FlashVSR/bin/python"),
    "SeedVR2": Path("/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-SeedVR2/bin/python"),
    "DOVE": Path("/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-DOVE/bin/python"),
    "Upscale-A-Video": Path("/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python"),
    "Real-ESRGAN": Path("/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-ImageSR/bin/python"),
    "HAT": Path("/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-ImageSR/bin/python"),
}
WRAPPERS = {
    "FlashVSR_v1.1": HERE / "FlashVSR_v1.1" / "upscale.py",
    "SeedVR2": HERE / "SeedVR2" / "upscale.py",
    "DOVE": HERE / "DOVE" / "upscale.py",
    "Upscale-A-Video": HERE / "Upscale-A-Video" / "upscale.py",
    "Real-ESRGAN": HERE / "Real-ESRGAN" / "upscale.py",
    "HAT": HERE / "HAT" / "upscale.py",
}
SEED_REQUIRED_FREE_MIB = 142_830.0
DOVE_REQUIRED_FREE_MIB = 100_000.0


class BenchmarkCancelled(RuntimeError):
    pass


def _cancel_from_signal(signum: int, _frame: Any) -> None:
    raise BenchmarkCancelled(f"received signal {signum}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    from common.media_utils import atomic_write_json

    atomic_write_json(path, payload)


def build_command(model: str, mode: str, args: argparse.Namespace) -> list[str]:
    result_dir = Path(args.result_root) / model
    report = result_dir / f"benchmark_{mode}.json"
    common = [
        str(PYTHONS[model]),
        str(WRAPPERS[model]),
        "--input", str(Path(args.input).resolve()),
        "--benchmark-mode", mode,
        "--start-frame", str(args.start_frame),
        "--result-dir", str(result_dir),
        "--report-json", str(report),
    ]
    if model == "FlashVSR_v1.1":
        return common + [
            "--output-dir", str(Path(args.output_root) / model),
            "--tile-size", "320",
            "--tile-overlap", "32",
            "--ffmpeg-threads", str(args.ffmpeg_threads),
            "--encoder", args.encoder,
        ]
    if model == "SeedVR2":
        return common + [
            "--output-dir", str(Path(args.output_root) / model),
            "--temporal-window", "5",
            "--temporal-overlap", "1",
            "--tile-size", "488",
            "--tile-overlap", "60",
            "--max-oom-retries", "3",
            "--ffmpeg-threads", str(args.ffmpeg_threads),
            "--encoder", args.encoder,
        ]
    if model == "DOVE":
        return common + [
            "--output-dir", str(Path(args.output_root) / model),
            "--tile-height", "768",
            "--tile-width", "1360",
            "--tile-overlap", "32",
            "--temporal-window", "17",
            "--temporal-overlap", "8",
            "--max-oom-retries", "2",
            "--ffmpeg-threads", str(args.ffmpeg_threads),
            "--encoder", args.encoder,
        ]
    if model == "Upscale-A-Video":
        return common + ["--availability-only"]
    if model in {"Real-ESRGAN", "HAT"}:
        overlap = "10" if model == "Real-ESRGAN" else "32"
        return common + [
            "--output-dir", str(Path(args.output_root) / model),
            "--tile-size", "512",
            "--tile-overlap", overlap,
            "--ffmpeg-threads", str(args.ffmpeg_threads),
            "--encoder", args.encoder,
        ]
    raise ValueError(f"unknown model: {model}")


def free_gpu_mib() -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return float(result.stdout.strip().splitlines()[0])


def blocked_by_prior_mode_failure(
    model: str, mode: str, failed_modes: set[tuple[str, str]]
) -> bool:
    return model == "DOVE" and mode == "mini" and (model, "micro") in failed_modes


def normalize_temporal_boundaries(report: dict[str, Any]) -> list[int]:
    """Return analyzer transition indices and migrate legacy DOVE chunk ranges."""
    raw = report.get("chunk_boundaries", [])
    if raw and all(isinstance(item, list) and len(item) == 2 for item in raw):
        report.setdefault("temporal_chunk_ranges", raw)
        boundaries = [int(item[0]) for item in raw[1:]]
        report["chunk_boundaries"] = boundaries
        return boundaries
    return [int(boundary) for boundary in raw]


def validate_output_report(report_path: Path, *, ffmpeg_threads: int = 12) -> dict[str, Any]:
    from common.media_utils import atomic_write_text, probe_media
    from common.metrics import analyze_structure, analyze_temporal

    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_text = report.get("output_path") or report.get("output")
    if not output_text:
        return {"status": "FAIL", "reason": "report has no output path"}
    output = probe_media(output_text, count_frames=True)
    input_value = report.get("input")
    input_path = (
        input_value.get("path") if isinstance(input_value, dict) else input_value
    )
    if not input_path:
        return {"status": "FAIL", "reason": "report has no source input path"}
    full_source = probe_media(input_path, count_frames=True)
    source = report.get("input") if isinstance(report.get("input"), dict) else report.get("input_metadata", {})
    expected_width = int(source.get("width", full_source.width)) * 4
    expected_height = int(source.get("height", full_source.height)) * 4
    expected_frames = int(report["effective_output_frames_written"])
    source_fps = source.get("fps_fraction", source.get("fps", full_source.fps))
    expected_fps = float(Fraction(str(source_fps)))
    checks = {
        "width": output.width == expected_width,
        "height": output.height == expected_height,
        "frame_count": output.frame_count == expected_frames,
        "fps": abs(output.fps_float - expected_fps) < 1e-6,
    }
    validation = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed": output.to_dict(),
        "expected": {
            "width": expected_width,
            "height": expected_height,
            "frame_count": expected_frames,
            "fps": str(source_fps),
        },
    }
    report["technical_validation"] = validation
    start_frame = int(
        report.get("frame_selection", {}).get(
            "start_frame", report.get("configuration", {}).get("start_frame", 0)
        )
    )
    sample_metadata = dataclasses.replace(
        full_source,
        frame_count=expected_frames,
        duration=expected_frames / full_source.fps_float,
        audio_present=False,
        audio_codec=None,
    )
    analysis_status = "PASS"
    analysis_error = None
    try:
        report["structure_analysis"] = analyze_structure(
            str(input_path),
            str(output_text),
            sample_metadata,
            ffmpeg_threads=ffmpeg_threads,
            source_start_frame=start_frame,
        )
        report["temporal_analysis"] = analyze_temporal(
            str(input_path),
            str(output_text),
            sample_metadata,
            chunk_boundaries=normalize_temporal_boundaries(report),
            source_start_frame=start_frame,
        )
    except Exception as error:  # noqa: BLE001 - persist analysis failure in the report
        analysis_status = "FAIL"
        analysis_error = f"{type(error).__name__}: {error}"
        report["quality_analysis_error"] = analysis_error
    atomic_json(report_path, report)
    structure = report.get("structure_analysis", {})
    temporal = report.get("temporal_analysis", {})
    marker = "<!-- MINIMAL_QUALITY_ANALYSIS -->"
    quality_markdown = f"""{marker}
## Automatic sample quality analysis

- Source frame range: `{start_frame}..{start_frame + expected_frames - 1}`
- Downsampled structure SSIM: `{structure.get('mean_ssim')}`
- Downsampled structure PSNR: `{structure.get('mean_psnr_db')}` dB
- Mean absolute difference: `{structure.get('mean_absolute_difference')}`
- Laplacian variance ratio: `{structure.get('mean_laplacian_ratio')}`
- Edge-energy ratio: `{structure.get('mean_edge_energy_ratio')}`
- Temporal sanity: **{temporal.get('status', analysis_status)}**
- Suspected duplicate ratio: `{temporal.get('suspected_duplicate_ratio')}`
- Analysis error: `{analysis_error}`

These are source-preservation and temporal-sanity measurements after the 16K
output is downsampled to source resolution. They are not ground-truth 16K or
subjective perceptual-quality scores.
"""
    markdown_path = report_path.with_suffix(".md")
    prior = markdown_path.read_text(encoding="utf-8") if markdown_path.is_file() else ""
    prior = prior.split(marker, 1)[0].rstrip()
    atomic_write_text(markdown_path, prior + "\n\n" + quality_markdown)
    return {
        "status": "PASS" if validation["status"] == "PASS" and analysis_status == "PASS" else "FAIL",
        "technical_validation": validation,
        "quality_analysis_status": analysis_status,
        "quality_analysis_error": analysis_error,
    }


def run_logged(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        list(command),
        cwd=HERE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
        return process.wait()
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded official-pretrained 4K-to-16K micro/mini benchmarks"
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument(
        "--status-json",
        help="Optional dedicated launcher status path; useful for partial model runs",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--ffmpeg-threads", type=int, default=12)
    parser.add_argument(
        "--encoder",
        choices=["auto", "libx265", "libsvtav1", "ffv1"],
        default="libx265",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        managed_signal = getattr(signal, signal_name, None)
        if managed_signal is not None:
            signal.signal(managed_signal, _cancel_from_signal)
    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    # A command preview must never replace the durable status of a completed GPU run.
    if args.status_json:
        status_path = Path(args.status_json).resolve()
    else:
        status_path = result_root / (
            "run_status_dry_run.json" if args.dry_run else "run_status.json"
        )
    status: dict[str, Any] = {
        "status": "RUNNING",
        "started_at": utc_now(),
        "input": str(Path(args.input).resolve()),
        "models": args.models,
        "modes": args.modes,
        "runs": [],
    }
    atomic_json(status_path, status)
    overall_code = 0
    failed_modes: set[tuple[str, str]] = set()
    try:
        for mode in args.modes:
            for model in args.models:
                command = build_command(model, mode, args)
                run_record: dict[str, Any] = {
                    "model": model,
                    "mode": mode,
                    "command": command,
                    "started_at": utc_now(),
                }
                status["runs"].append(run_record)
                atomic_json(status_path, status)
                print(f"\n=== {model} {mode} ===", flush=True)
                print(" ".join(command), flush=True)
                if blocked_by_prior_mode_failure(model, mode, failed_modes):
                    run_record.update({
                        "status": "SKIPPED_PREREQUISITE_FAILED",
                        "reason": "DOVE micro failed; larger mini run was not attempted",
                        "exit_code": 6,
                        "completed_at": utc_now(),
                    })
                    overall_code = 1
                    atomic_json(status_path, status)
                    continue
                if args.dry_run:
                    run_record.update({"status": "DRY_RUN", "exit_code": 0, "completed_at": utc_now()})
                    continue
                if model in {"SeedVR2", "DOVE"}:
                    free_mib = free_gpu_mib()
                    run_record["free_gpu_mib_before"] = free_mib
                    required_free_mib = (
                        SEED_REQUIRED_FREE_MIB
                        if model == "SeedVR2"
                        else DOVE_REQUIRED_FREE_MIB
                    )
                    if free_mib < required_free_mib:
                        run_record.update({
                            "status": "BLOCKED_INSUFFICIENT_FREE_VRAM",
                            "required_free_gpu_mib": required_free_mib,
                            "exit_code": 3,
                            "completed_at": utc_now(),
                        })
                        overall_code = 1
                        failed_modes.add((model, mode))
                        atomic_json(status_path, status)
                        continue
                started = time.perf_counter()
                code = run_logged(command, result_root / model / f"run_{mode}.log")
                validation = None
                if code == 0 and model != "Upscale-A-Video":
                    validation = validate_output_report(
                        result_root / model / f"benchmark_{mode}.json",
                        ffmpeg_threads=args.ffmpeg_threads,
                    )
                    if validation["status"] != "PASS":
                        code = 4
                run_record.update({
                    "status": "PASS" if code == 0 else "FAIL",
                    "exit_code": code,
                    "wall_seconds": time.perf_counter() - started,
                    "completed_at": utc_now(),
                })
                if validation is not None:
                    run_record["validation"] = validation
                if code:
                    overall_code = 1
                    failed_modes.add((model, mode))
                atomic_json(status_path, status)
        if not args.dry_run:
            try:
                from summarize_benchmarks import (
                    analyze_common_quality,
                    generate_contact_sheets,
                    render,
                    render_quality_comparison,
                    synthesize,
                )

                synthesis = synthesize(result_root)
                atomic_json(result_root / "benchmark_estimates.json", synthesis)
                from common.media_utils import atomic_write_text

                atomic_write_text(
                    result_root / "performance_comparison.md",
                    render(result_root, synthesis),
                )
                quality = analyze_common_quality(
                    result_root, ffmpeg_threads=args.ffmpeg_threads
                )
                atomic_json(result_root / "quality_comparison.json", quality)
                atomic_write_text(
                    result_root / "quality_comparison.md",
                    render_quality_comparison(quality),
                )
                contact_sheets = generate_contact_sheets(result_root, quality)
                status["synthesis"] = {
                    "status": "PASS",
                    "json": str(result_root / "benchmark_estimates.json"),
                    "markdown": str(result_root / "performance_comparison.md"),
                    "quality_json": str(result_root / "quality_comparison.json"),
                    "quality_markdown": str(result_root / "quality_comparison.md"),
                    "contact_sheets": contact_sheets,
                }
            except FileNotFoundError as error:
                status["synthesis"] = {
                    "status": "PENDING_REQUIRED_REPORTS",
                    "reason": str(error),
                }
            except Exception as error:  # noqa: BLE001 - synthesis is an isolated report stage
                status["synthesis"] = {
                    "status": "FAIL",
                    "reason": f"{type(error).__name__}: {error}",
                }
                overall_code = 1
        status["status"] = "PASS" if overall_code == 0 else "FAIL"
        status["completed_at"] = utc_now()
        atomic_json(status_path, status)
        return overall_code
    except BaseException as error:
        status["status"] = (
            "CANCELLED"
            if isinstance(error, (KeyboardInterrupt, BenchmarkCancelled))
            else "FAIL"
        )
        status["error"] = f"{type(error).__name__}: {error}"
        status["completed_at"] = utc_now()
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
