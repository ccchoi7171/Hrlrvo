#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from typing import Any

from common.media_utils import atomic_write_json, atomic_write_text, probe_media
from common.metrics import analyze_structure, analyze_temporal, validate_technical


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INPUT = ROOT / "input" / "0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4"
DEFAULT_RESULT_ROOT = ROOT / "result" / "test"
MODELS = ("FlashVSR_v1.1", "SeedVR2")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _number(
    mapping: dict[str, Any], *keys: str, default: float | None = None
) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return default


def _timing(report: dict[str, Any]) -> dict[str, float | None]:
    source = report.get("timings") or report.get("timing") or {}
    return {
        "model_load_seconds": _number(source, "model_load_seconds"),
        "inference_seconds": _number(source, "inference_seconds", "sr_inference_seconds"),
        "encoding_mux_seconds": _number(source, "encoding_mux_seconds", "encoding_and_muxing_seconds"),
        "total_wall_seconds": _number(source, "total_wall_seconds"),
        "inference_seconds_per_frame": _number(source, "inference_seconds_per_frame"),
        "total_seconds_per_frame": _number(source, "total_seconds_per_frame"),
        "effective_inference_fps": _number(source, "effective_inference_fps"),
        "effective_end_to_end_fps": _number(source, "effective_end_to_end_fps"),
    }


def _gpu(report: dict[str, Any]) -> dict[str, Any]:
    source = report.get("gpu", {})
    return {
        "gpu_name": source.get("gpu_name") or source.get("name") or "unknown",
        "gpu_total_vram_mb": _number(source, "gpu_total_vram_mb", "total_mib"),
        "peak_torch_allocated_mb": _number(
            source, "peak_torch_allocated_mb", "peak_torch_allocated_mib"
        ),
        "peak_torch_reserved_mb": _number(
            source, "peak_torch_reserved_mb", "peak_torch_reserved_mib"
        ),
        "peak_process_gpu_memory_mb": _number(
            source, "peak_process_gpu_memory_mb", "peak_process_gpu_mib"
        ),
        "peak_total_gpu_used_memory_mb": _number(
            source, "peak_total_gpu_used_memory_mb", "peak_total_gpu_used_mib"
        ),
        "average_gpu_utilization_percent": _number(source, "average_gpu_utilization_percent"),
        "peak_gpu_utilization_percent": _number(source, "peak_gpu_utilization_percent"),
        "utilization_unavailable_reason": source.get("utilization_unavailable_reason"),
    }


def _output_path(report: dict[str, Any]) -> str | None:
    value = report.get("output_path") or report.get("output")
    return str(value) if value else None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_model_report(model: str, report: dict[str, Any]) -> str:
    timing = report.get("normalized_timing", {})
    gpu = report.get("normalized_gpu", {})
    technical = report.get("technical_validation", {})
    structure = report.get("structure_analysis", {})
    temporal = report.get("temporal_analysis", {})
    input_meta = report.get("benchmark_input", {})
    execution_plan = report.get("execution_plan") or report.get("configuration") or {}
    return f"""# {model} — 4K → 16K Pretrained VSR Report

## Result

- Result: **{report.get('benchmark_result', 'FAIL')}**
- Input: `{input_meta.get('path', 'unknown')}`
- Output: `{_output_path(report) or 'not produced'}`
- Input resolution: {input_meta.get('width')}x{input_meta.get('height')}
- Expected/output resolution: {technical.get('expected_resolution', 'N/A')} / {technical.get('output_resolution', 'N/A')}
- Scale: {_fmt(technical.get('actual_scale_x'))}× × {_fmt(technical.get('actual_scale_y'))}×
- Frames: {technical.get('input_frame_count', input_meta.get('frame_count'))} → {technical.get('output_frame_count', 'N/A')}
- FPS: {technical.get('input_fps', input_meta.get('fps'))} → {technical.get('output_fps', 'N/A')}
- Audio preserved: {_fmt(technical.get('checks', {}).get('audio_preserved'))}

## Timing

| Metric | Value |
|---|---:|
| Input frames | {input_meta.get('frame_count', 'N/A')} |
| Model initialization / checkpoint loading | {_fmt(timing.get('model_load_seconds'))} sec |
| Actual SR inference | {_fmt(timing.get('inference_seconds'))} sec |
| Video encoding / muxing | {_fmt(timing.get('encoding_mux_seconds'))} sec |
| Total wall-clock | {_fmt(timing.get('total_wall_seconds'))} sec |
| Inference sec/frame | {_fmt(timing.get('inference_seconds_per_frame'), 6)} |
| Total sec/frame | {_fmt(timing.get('total_seconds_per_frame'), 6)} |
| Effective inference FPS | {_fmt(timing.get('effective_inference_fps'), 6)} |
| Effective end-to-end FPS | {_fmt(timing.get('effective_end_to_end_fps'), 6)} |

## GPU

| Metric | Value |
|---|---:|
| GPU | {gpu.get('gpu_name', 'unknown')} |
| GPU total VRAM | {_fmt(gpu.get('gpu_total_vram_mb'), 0)} MiB |
| Peak torch allocated VRAM | {_fmt(gpu.get('peak_torch_allocated_mb'), 0)} MiB |
| Peak torch reserved VRAM | {_fmt(gpu.get('peak_torch_reserved_mb'), 0)} MiB |
| Peak process GPU memory | {_fmt(gpu.get('peak_process_gpu_memory_mb'), 0)} MiB |
| Peak total GPU used memory | {_fmt(gpu.get('peak_total_gpu_used_memory_mb'), 0)} MiB |
| Average GPU utilization | {_fmt(gpu.get('average_gpu_utilization_percent'), 1)}% |
| Peak GPU utilization | {_fmt(gpu.get('peak_gpu_utilization_percent'), 1)}% |

Utilization availability: {gpu.get('utilization_unavailable_reason') or 'nvidia-smi counters sampled successfully'}

## Automatic structure/detail analysis

- Uniform sample frames: {structure.get('sample_count', 0)}
- Structure SSIM: {_fmt(structure.get('mean_ssim'), 6)}
- Structure PSNR: {_fmt(structure.get('mean_psnr_db'))} dB
- Mean absolute difference: {_fmt(structure.get('mean_absolute_difference'))}
- Mean absolute luminance drift: {_fmt(structure.get('mean_absolute_luminance_drift'))}
- Laplacian variance ratio: {_fmt(structure.get('mean_laplacian_ratio'))}
- Edge-energy ratio: {_fmt(structure.get('mean_edge_energy_ratio'))}
- Oversharpen/halo heuristic: {'WARN' if structure.get('oversharpen_or_halo_warning') else 'PASS'}

These metrics measure preservation after Lanczos downsampling to the source resolution; they are not 16K ground-truth quality scores.

## Temporal sanity

- Status: **{temporal.get('status', 'N/A')}**
- Suspected duplicate ratio: {_fmt(temporal.get('suspected_duplicate_ratio'), 6)}
- Frozen-frame suspected: {_fmt(temporal.get('frozen_frame_suspected'))}
- Brightness discontinuities: {temporal.get('brightness_discontinuity_frames', [])}
- Color discontinuities: {temporal.get('color_discontinuity_frames', [])}
- Chunk boundaries checked: {temporal.get('chunk_boundaries_checked', [])}
- Chunk-boundary risks: {temporal.get('chunk_boundary_risks', [])}

## Execution/environment

```json
{json.dumps({'execution_plan': execution_plan, 'environment': report.get('environment', {})}, indent=2, ensure_ascii=False)}
```

## Method boundary

The neural inference uses the official pretrained model without training or fine-tuning. Memory-bounded spatial/temporal aggregation is an inference adapter and is identified in the execution plan; no per-frame image-SR model is substituted.
"""


def analyze_one(
    model: str,
    report_path: Path,
    *,
    input_path: Path,
    ffmpeg_threads: int,
) -> dict[str, Any]:
    report = _load_json(report_path)
    input_metadata = probe_media(input_path, count_frames=True)
    report["benchmark_input"] = input_metadata.to_dict()
    report["normalized_timing"] = _timing(report)
    report["normalized_gpu"] = _gpu(report)
    output_path = _output_path(report)
    if not output_path or not Path(output_path).is_file():
        report["benchmark_result"] = "FAIL"
        report["analysis_error"] = report.get("error", "output video was not produced")
    else:
        try:
            report["technical_validation"] = validate_technical(input_metadata, output_path, scale=4)
            report["structure_analysis"] = analyze_structure(
                str(input_path), output_path, input_metadata, ffmpeg_threads=ffmpeg_threads
            )
            report["temporal_analysis"] = analyze_temporal(
                str(input_path),
                output_path,
                input_metadata,
                chunk_boundaries=report.get("chunk_boundaries", []),
            )
            report["benchmark_result"] = (
                "PASS" if report["technical_validation"]["passed"] else "FAIL"
            )
        except Exception as error:
            report["benchmark_result"] = "FAIL"
            report["analysis_error"] = f"{type(error).__name__}: {error}"
    atomic_write_json(report_path, report)
    atomic_write_text(report_path.with_suffix(".md"), render_model_report(model, report))
    return report


def analyze_reports_concurrently(
    *,
    result_root: Path,
    input_path: Path,
    total_ffmpeg_threads: int,
    analysis_workers: int,
) -> dict[str, dict[str, Any]]:
    """Analyze independent model outputs concurrently within one CPU budget."""
    worker_count = max(1, min(analysis_workers, len(MODELS)))
    threads_per_worker = max(1, total_ffmpeg_threads // worker_count)

    def analyze_model(model: str) -> dict[str, Any]:
        report_path = result_root / model / "report.json"
        if not report_path.is_file():
            report_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                report_path,
                {"model": model, "status": "FAIL", "error": "report.json missing"},
            )
        try:
            return analyze_one(
                model,
                report_path,
                input_path=input_path,
                ffmpeg_threads=threads_per_worker,
            )
        except Exception as error:
            # Keep one broken model report from suppressing analysis of the
            # other completed output. This worker only writes its own files.
            try:
                report = _load_json(report_path)
            except Exception:
                report = {"model": model}
            report["benchmark_result"] = "FAIL"
            report["analysis_error"] = f"{type(error).__name__}: {error}"
            atomic_write_json(report_path, report)
            atomic_write_text(
                report_path.with_suffix(".md"), render_model_report(model, report)
            )
            return report

    completed: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="result-analysis"
    ) as executor:
        future_models = {
            executor.submit(analyze_model, model): model for model in MODELS
        }
        for future in as_completed(future_models):
            model = future_models[future]
            completed[model] = future.result()

    # Completion order is nondeterministic; aggregation and serialized output
    # retain the canonical model order.
    return {model: completed[model] for model in MODELS}


def _cell(report: dict[str, Any], key: str, default: str = "N/A") -> str:
    value = report
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return _fmt(value)


def _winner(
    reports: dict[str, dict[str, Any]], key: str, *, higher_is_better: bool
) -> str:
    values: list[tuple[str, float]] = []
    for model, report in reports.items():
        if report.get("benchmark_result") != "PASS":
            continue
        value: Any = report
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append((model, float(value)))
    if len(values) != 2:
        return "Insufficient successful measurements."
    values.sort(key=lambda item: item[1], reverse=higher_is_better)
    return f"{values[0][0]} has the {'higher' if higher_is_better else 'lower'} measured value ({values[0][1]:.6g} vs {values[1][1]:.6g})."


def render_comparison(
    reports: dict[str, dict[str, Any]],
    execution: dict[str, Any],
) -> str:
    flash = reports["FlashVSR_v1.1"]
    seed = reports["SeedVR2"]
    input_meta = flash.get("benchmark_input") or seed.get("benchmark_input") or {}
    hardware = execution.get("hardware", {})
    rows = [
        ("Output resolution", "technical_validation.output_resolution"),
        ("Scale", "technical_validation.actual_scale_x"),
        ("Frames", "technical_validation.output_frame_count"),
        ("FPS", "technical_validation.output_fps"),
        ("Peak VRAM", "normalized_gpu.peak_process_gpu_memory_mb"),
        ("Inference sec/frame", "normalized_timing.inference_seconds_per_frame"),
        ("Total sec/frame", "normalized_timing.total_seconds_per_frame"),
        ("Effective inference FPS", "normalized_timing.effective_inference_fps"),
        ("Total runtime", "normalized_timing.total_wall_seconds"),
        ("Structure SSIM", "structure_analysis.mean_ssim"),
        ("Structure PSNR", "structure_analysis.mean_psnr_db"),
        ("Sharpness change", "structure_analysis.mean_laplacian_ratio"),
        ("Temporal sanity", "temporal_analysis.status"),
        ("Result", "benchmark_result"),
    ]
    table = ["| Metric | FlashVSR v1.1 | SeedVR2 |", "|---|---:|---:|"]
    table.extend(f"| {label} | {_cell(flash, key)} | {_cell(seed, key)} |" for label, key in rows)
    return f"""# 4K → 16K Upscaler Test

## Input

- Resolution: {input_meta.get('width')}x{input_meta.get('height')}
- Frames: {input_meta.get('frame_count')}
- FPS: {input_meta.get('fps')}
- Duration: {_fmt(input_meta.get('duration'))} sec

## Hardware

- GPU: {hardware.get('gpu_name', 'unknown')}
- VRAM: {_fmt(hardware.get('gpu_total_vram_mb'), 0)} MiB
- VRAM available before run: {_fmt(hardware.get('gpu_available_before_mb'), 0)} MiB
- CPU cores: {hardware.get('cpu_cores', 'unknown')}

## Execution

- Parallel count: {execution.get('selected_parallel_count')}
- Selected mode: {execution.get('selected_mode')}
- Reason: {execution.get('selection_reason')}
- FlashVSR estimated peak: {_fmt(execution.get('probe', {}).get('FlashVSR_v1.1_peak_mb'), 0)} MiB
- SeedVR2 estimated peak: {_fmt(execution.get('probe', {}).get('SeedVR2_peak_mb'), 0)} MiB
- Whole benchmark wall time: {_fmt(execution.get('whole_benchmark_wall_seconds'))} sec

{os.linesep.join(table)}

## Automatic interpretation

- **Quality / structure preservation:** {_winner(reports, 'structure_analysis.mean_ssim', higher_is_better=True)} This comparison is limited to automatic downsampled structure metrics and does not assert unobserved subjective 16K quality.
- **Speed:** {_winner(reports, 'normalized_timing.total_wall_seconds', higher_is_better=False)}
- **VRAM efficiency:** {_winner(reports, 'normalized_gpu.peak_process_gpu_memory_mb', higher_is_better=False)}
- **Long-video scalability:** FlashVSR uses its official continuous causal/KV temporal path within each spatial tile. SeedVR2 upstream has no released streaming/tile path, so the benchmark uses explicitly reported overlapping temporal windows and spatial tiles; inspect the boundary-sanity result before drawing conclusions.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and compare FlashVSR v1.1 and SeedVR2 outputs")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--execution-json", required=True)
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="total FFmpeg thread budget shared evenly by model analyses",
    )
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=len(MODELS),
        help="number of concurrent per-model analyses (default: 2)",
    )
    args = parser.parse_args()
    if args.ffmpeg_threads < 1:
        parser.error("--ffmpeg-threads must be at least 1")
    if args.analysis_workers < 1:
        parser.error("--analysis-workers must be at least 1")
    result_root = Path(args.result_root).resolve()
    reports = analyze_reports_concurrently(
        result_root=result_root,
        input_path=Path(args.input).resolve(),
        total_ffmpeg_threads=args.ffmpeg_threads,
        analysis_workers=args.analysis_workers,
    )
    execution = _load_json(Path(args.execution_json))
    comparison = {
        "input": (reports["FlashVSR_v1.1"].get("benchmark_input") or {}),
        "execution": execution,
        "models": {
            model: {
                "result": report.get("benchmark_result"),
                "output": _output_path(report),
                "technical_validation": report.get("technical_validation"),
                "structure_analysis": report.get("structure_analysis"),
                "temporal_analysis": report.get("temporal_analysis"),
                "timing": report.get("normalized_timing"),
                "gpu": report.get("normalized_gpu"),
            }
            for model, report in reports.items()
        },
    }
    atomic_write_json(result_root / "comparison.json", comparison)
    atomic_write_text(result_root / "comparison.md", render_comparison(reports, execution))
    return 0 if all(report.get("benchmark_result") == "PASS" for report in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
