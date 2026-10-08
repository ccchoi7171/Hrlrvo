#!/usr/bin/env python3
"""Fit long-video timing models from completed micro and mini reports."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from common.media_utils import atomic_write_json, atomic_write_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RESULT_ROOT = ROOT / "result" / "benchmark"
TARGETS = (10, 92, 100, 1000, 1920, 2880)
ORIGINAL_MODELS = ("FlashVSR_v1.1", "SeedVR2", "Real-ESRGAN", "HAT")
ACTUAL_MODELS = (*ORIGINAL_MODELS[:2], "DOVE", *ORIGINAL_MODELS[2:])
MODEL_TYPES = {
    "FlashVSR_v1.1": "causal video SR",
    "SeedVR2": "windowed video diffusion SR",
    "DOVE": "one-step tiled video diffusion SR",
    "Real-ESRGAN": "frame-independent image SR",
    "HAT": "frame-independent image SR",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _timing(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("timings") or report.get("timing") or {}


def _pure(report: dict[str, Any]) -> float:
    timing = _timing(report)
    return float(timing["neural_inference_seconds"]) + float(timing["decode_seconds"])


def _inference_work(report: dict[str, Any]) -> float:
    timing = _timing(report)
    value = timing.get("inference_seconds", timing.get("sr_inference_seconds"))
    return float(value) if value is not None else _pure(report)


def _cpu_io(report: dict[str, Any]) -> float:
    timing = _timing(report)
    staging = float(report.get("timings_detail", {}).get("intermediate_tile_staging_seconds", 0.0))
    return staging + float(timing.get("merge_seconds", 0.0)) + float(timing.get("encode_seconds", 0.0))


def flash_steps(frames: int) -> int:
    return 1 if frames <= 21 else 1 + math.ceil((frames - 21) / 8)


def seed_chunks(frames: int) -> int:
    return 1 if frames <= 5 else 1 + math.ceil((frames - 5) / 4)


def dove_work_plan(frames: int) -> tuple[int, int]:
    """Return (chunks, native 8-frame units) for the locked 17/8 plan."""
    if frames <= 0:
        raise ValueError("frames must be positive")
    if frames <= 17:
        return 1, 1 if frames <= 9 else 2
    chunks = 1 + math.ceil((frames - 17) / 9)
    return chunks, chunks * 2


def _validated_pair(result_root: Path, model: str) -> dict[str, dict[str, Any]] | None:
    pair: dict[str, dict[str, Any]] = {}
    for mode in ("micro", "mini"):
        path = result_root / model / f"benchmark_{mode}.json"
        if not path.is_file():
            return None
        report = _load(path)
        if report.get("technical_validation", {}).get("status") != "PASS":
            return None
        pair[mode] = report
    return pair


def fit_frame_independent_pair(
    micro: dict[str, Any],
    mini: dict[str, Any],
    *,
    targets: Sequence[int],
) -> dict[str, Any]:
    """Fit T(N)=load+fixed+steady*N from the 21/29-frame measurements."""
    micro_frames = int(micro["effective_output_frames_written"])
    mini_frames = int(mini["effective_output_frames_written"])
    if mini_frames <= micro_frames:
        raise ValueError("frame-independent mini run must contain more frames than micro")
    micro_timing, mini_timing = _timing(micro), _timing(mini)
    micro_total = float(micro_timing["total_wall_seconds"])
    mini_total = float(mini_timing["total_wall_seconds"])
    micro_load = float(micro_timing["model_load_seconds"])
    load_once = float(mini_timing["model_load_seconds"])
    # Remove the independently measured load from each run before taking the
    # slope; otherwise normal load jitter is falsely charged to every frame.
    steady = ((mini_total - load_once) - (micro_total - micro_load)) / (
        mini_frames - micro_frames
    )
    if steady <= 0:
        raise ValueError(f"frame-independent steady per-frame delta must be positive, got {steady}")
    fixed = (micro_total - micro_load) - steady * micro_frames
    pure_steady = (_pure(mini) - _pure(micro)) / (mini_frames - micro_frames)
    estimates = {
        str(frames): {
            "frames": frames,
            "frame_inference_units": frames,
            "estimated_total_seconds": load_once + fixed + steady * frames,
            "estimated_total_hours": (load_once + fixed + steady * frames) / 3600.0,
        }
        for frames in targets
    }
    return {
        "coefficients": {
            "model_load_once_seconds": load_once,
            "fixed_orchestration_seconds": fixed,
            "steady_end_to_end_seconds_per_frame": steady,
            "steady_pure_neural_decode_seconds_per_frame": pure_steady,
        },
        "estimates": estimates,
        "formula": "load + fixed + steady_end_to_end_per_frame*N",
    }


def fit_pair(
    micro: dict[str, Any],
    mini: dict[str, Any],
    *,
    unit_name: str,
    unit_count: Any,
    targets: Sequence[int],
) -> dict[str, Any]:
    micro_pure = _pure(micro)
    mini_pure = _pure(mini)
    warmup_pure = micro_pure
    steady_pure = mini_pure - micro_pure
    warmup = _inference_work(micro)
    steady = _inference_work(mini) - warmup
    if steady <= 0:
        raise ValueError(f"{unit_name} steady-state delta must be positive, got {steady}")
    mini_frames = int(mini["effective_output_frames_written"])
    cpu_io_per_frame = _cpu_io(mini) / mini_frames
    load_once = float(_timing(mini)["model_load_seconds"])
    mini_total = float(
        _timing(mini).get(
            "total_wall_seconds", load_once + _inference_work(mini) + _cpu_io(mini)
        )
    )
    fixed_orchestration = max(
        0.0, mini_total - load_once - _inference_work(mini) - _cpu_io(mini)
    )
    estimates: dict[str, dict[str, float | int]] = {}
    for frames in targets:
        units = int(unit_count(frames))
        total = (
            load_once
            + fixed_orchestration
            + warmup
            + max(0, units - 1) * steady
            + cpu_io_per_frame * frames
        )
        estimates[str(frames)] = {
            "frames": frames,
            unit_name: units,
            "estimated_total_seconds": total,
            "estimated_total_hours": total / 3600.0,
        }
    return {
        "coefficients": {
            "model_load_once_seconds": load_once,
            "fixed_orchestration_seconds": fixed_orchestration,
            "warmup_full_sr_inference_seconds": warmup,
            "steady_additional_unit_full_sr_inference_seconds": steady,
            "warmup_pure_neural_decode_seconds": warmup_pure,
            "steady_additional_unit_pure_neural_decode_seconds": steady_pure,
            "cpu_io_seconds_per_output_frame": cpu_io_per_frame,
        },
        "estimates": estimates,
    }


def fit_dove_pair(
    micro: dict[str, Any],
    mini: dict[str, Any],
    *,
    targets: Sequence[int],
) -> dict[str, Any]:
    """Fit DOVE's 9/17-frame native units and repeat chunk warm-up."""
    if int(micro["effective_output_frames_written"]) != 9:
        raise ValueError("DOVE micro report must contain 9 output frames")
    if int(mini["effective_output_frames_written"]) != 17:
        raise ValueError("DOVE mini report must contain 17 output frames")
    warmup = _inference_work(micro)
    additional_unit = _inference_work(mini) - warmup
    warmup_pure = _pure(micro)
    additional_unit_pure = _pure(mini) - warmup_pure
    if additional_unit <= 0:
        raise ValueError(
            f"DOVE additional native temporal unit must be positive, got {additional_unit}"
        )
    mini_frames = int(mini["effective_output_frames_written"])
    cpu_io_per_frame = _cpu_io(mini) / mini_frames
    load_once = float(_timing(mini)["model_load_seconds"])
    retry_overhead = float(
        mini.get("timings_detail", {}).get("oom_retry_overhead_seconds", 0.0)
    )
    mini_total = float(_timing(mini)["total_wall_seconds"]) - retry_overhead
    fixed_orchestration = max(
        0.0,
        mini_total - load_once - _inference_work(mini) - _cpu_io(mini),
    )
    estimates: dict[str, dict[str, float | int]] = {}
    for frames in targets:
        chunks, native_units = dove_work_plan(frames)
        full_sr = chunks * warmup + (native_units - chunks) * additional_unit
        total = (
            load_once
            + fixed_orchestration
            + full_sr
            + cpu_io_per_frame * frames
        )
        estimates[str(frames)] = {
            "frames": frames,
            "temporal_chunks": chunks,
            "native_8_frame_units_processed": native_units,
            "estimated_total_seconds": total,
            "estimated_total_hours": total / 3600.0,
        }
    return {
        "coefficients": {
            "model_load_once_seconds": load_once,
            "fixed_orchestration_seconds": fixed_orchestration,
            "chunk_warmup_full_sr_inference_seconds": warmup,
            "additional_native_8_frame_unit_full_sr_inference_seconds": additional_unit,
            "chunk_warmup_pure_neural_decode_seconds": warmup_pure,
            "additional_native_8_frame_unit_pure_neural_decode_seconds": additional_unit_pure,
            "cpu_io_seconds_per_output_frame": cpu_io_per_frame,
            "excluded_one_time_oom_retry_overhead_seconds": retry_overhead,
        },
        "estimates": estimates,
        "formula": (
            "load + fixed + chunks(N)*chunk_warmup + "
            "(native_units(N)-chunks(N))*additional_native_unit + cpu_io_per_frame*N; "
            "window=17, overlap=8, stride=9"
        ),
    }


def synthesize(result_root: Path, targets: Sequence[int] = TARGETS) -> dict[str, Any]:
    reports = {
        model: pair
        for model in ACTUAL_MODELS
        if (pair := _validated_pair(result_root, model)) is not None
    }
    if not reports:
        raise FileNotFoundError("no model has both micro and mini PASS benchmark reports")
    fitted: dict[str, Any] = {}
    if "FlashVSR_v1.1" in reports:
        pair = reports["FlashVSR_v1.1"]
        fitted["FlashVSR_v1.1"] = fit_pair(
            pair["micro"], pair["mini"], unit_name="streaming_steps",
            unit_count=flash_steps, targets=targets,
        )
        fitted["FlashVSR_v1.1"]["formula"] = (
            "load + fixed + warmup_full_sr + max(0, steps(N)-1)*steady_full_sr + "
            "cpu_io_per_frame*N; steps(N)=1 if N<=21 else 1+ceil((N-21)/8)"
        )
    if "SeedVR2" in reports:
        pair = reports["SeedVR2"]
        fitted["SeedVR2"] = fit_pair(
            pair["micro"], pair["mini"], unit_name="temporal_chunks",
            unit_count=seed_chunks, targets=targets,
        )
        fitted["SeedVR2"]["formula"] = (
            "load + fixed + warmup_full_sr + max(0, chunks(N)-1)*steady_full_sr + "
            "cpu_io_per_frame*N; chunks(N)=1 if N<=5 else 1+ceil((N-5)/4)"
        )
    if "DOVE" in reports:
        pair = reports["DOVE"]
        fitted["DOVE"] = fit_dove_pair(
            pair["micro"], pair["mini"], targets=targets
        )
    for model in ("Real-ESRGAN", "HAT"):
        if model in reports:
            fitted[model] = fit_frame_independent_pair(
                reports[model]["micro"], reports[model]["mini"], targets=targets
            )
    return {
        "targets": list(targets),
        "models": fitted,
        "complete_four_model_comparison": all(
            model in fitted for model in ORIGINAL_MODELS
        ),
        "complete_five_model_comparison": all(
            model in fitted for model in ACTUAL_MODELS
        ),
    }


def analyze_common_quality(result_root: Path, *, ffmpeg_threads: int) -> dict[str, Any]:
    """Compare identical source-frame ranges despite model-specific temporal minima."""
    from common.media_utils import probe_media
    from common.metrics import analyze_structure, analyze_temporal

    payload: dict[str, Any] = {"method": "identical leading frame range per mode", "modes": {}}
    for mode in ("micro", "mini"):
        reports: dict[str, dict[str, Any]] = {}
        for model in ACTUAL_MODELS:
            path = result_root / model / f"benchmark_{mode}.json"
            if not path.is_file():
                continue
            report = _load(path)
            if report.get("technical_validation", {}).get("status") == "PASS":
                reports[model] = report
        if not reports:
            continue
        starts = {
            int(report.get("frame_selection", {}).get(
                "start_frame", report.get("configuration", {}).get("start_frame", 0)
            ))
            for report in reports.values()
        }
        if len(starts) != 1:
            raise ValueError(f"{mode} reports use different source starts: {sorted(starts)}")
        start_frame = starts.pop()
        common_frames = min(
            int(report["effective_output_frames_written"]) for report in reports.values()
        )
        input_value = next(iter(reports.values())).get("input")
        input_path = str(input_value.get("path") if isinstance(input_value, dict) else input_value)
        full_source = probe_media(input_path, count_frames=True)
        sample_metadata = dataclasses.replace(
            full_source,
            frame_count=common_frames,
            duration=common_frames / full_source.fps_float,
            audio_present=False,
            audio_codec=None,
        )
        mode_result: dict[str, Any] = {
            "source_start_frame": start_frame,
            "common_frame_count": common_frames,
            "models": {},
        }
        for model, report in reports.items():
            output = report.get("output_path") or report.get("output")
            if not output:
                raise RuntimeError(f"{model} {mode} has no output")
            mode_result["models"][model] = {
                "structure": analyze_structure(
                    input_path,
                    str(output),
                    sample_metadata,
                    ffmpeg_threads=ffmpeg_threads,
                    source_start_frame=start_frame,
                ),
                "temporal": analyze_temporal(
                    input_path,
                    str(output),
                    sample_metadata,
                    chunk_boundaries=report.get("chunk_boundaries", []),
                    source_start_frame=start_frame,
                ),
            }
        payload["modes"][mode] = mode_result
    return payload


def render_quality_comparison(payload: dict[str, Any]) -> str:
    rows: list[str] = []
    for mode, item in payload["modes"].items():
        for model, result in item["models"].items():
            structure, temporal = result["structure"], result["temporal"]
            rows.append(
                f"| {mode} | {item['common_frame_count']} | {model} | "
                f"{structure['mean_ssim']:.6f} | {structure['mean_psnr_db']:.3f} | "
                f"{structure['mean_absolute_difference']:.3f} | "
                f"{structure['mean_laplacian_ratio']:.3f} | "
                f"{structure['mean_edge_energy_ratio']:.3f} | "
                f"{temporal['status']} | {temporal['suspected_duplicate_ratio']:.6f} | "
                f"{float(temporal.get('mean_output_to_input_motion_ratio', 0.0)):.3f} | "
                f"{float(temporal.get('mean_temporal_restoration_residual_difference', 0.0)):.3f} |"
            )
    return """# Common-frame automatic quality comparison

| Mode | Common frames | Model | SSIM | PSNR dB | Mean abs diff | Laplacian ratio | Edge ratio | Temporal | Duplicate ratio | Motion ratio | Temporal residual delta |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
{rows}

Every available PASS model is evaluated on exactly the same leading source
frames within each mode. With all five runnable models this is micro: 5 and
mini: 9, because SeedVR2 has the smallest valid temporal samples. Outputs are
Lanczos-downsampled to source resolution.
These are preservation/sanity diagnostics, not ground-truth 16K perceptual
quality scores. Motion ratio is output/input mean inter-frame change after both
are reduced to the same analysis resolution. Temporal residual delta measures
how much the model's source-relative correction changes between frames; lower
is more stable, but smoothing can also lower both metrics.
""".format(rows="\n".join(rows))


def _value(report: dict[str, Any], *path: str) -> Any:
    value: Any = report
    for item in path:
        if not isinstance(value, dict):
            return None
        value = value.get(item)
    return value


def render(result_root: Path, payload: dict[str, Any]) -> str:
    rows = []
    for model in payload["models"]:
        for mode in ("micro", "mini"):
            report = _load(result_root / model / f"benchmark_{mode}.json")
            timing = _timing(report)
            gpu = report.get("gpu", {})
            structure = report.get("structure_analysis", {})
            units = report.get(
                "native_temporal_unit_count",
                report.get(
                    "streaming_step_count",
                    report.get(
                        "temporal_chunk_count",
                        report.get("effective_input_frames_processed"),
                    ),
                ),
            )
            rows.append(
                "| {model} | {kind} | {mode} | {frames} | {units} | {pure:.3f} | {e2e:.3f} | {vram:.0f} | {ssim} | {psnr} |".format(
                    model=model,
                    kind=MODEL_TYPES[model],
                    mode=mode,
                    frames=report.get("effective_output_frames_written"),
                    units=units,
                    pure=float(timing["pure_inference_seconds_per_output_frame"]),
                    e2e=float(timing["end_to_end_seconds_per_output_frame"]),
                    vram=float(gpu.get("peak_process_gpu_memory_mb", gpu.get("peak_process_gpu_mib", 0.0))),
                    ssim=structure.get("mean_ssim", "N/A"),
                    psnr=structure.get("mean_psnr_db", "N/A"),
                )
            )
    estimates = []
    for frames in payload["targets"]:
        values = [
            f"{payload['models'][model]['estimates'][str(frames)]['estimated_total_hours']:.3f}"
            for model in payload["models"]
        ]
        estimates.append(f"| {frames} | " + " | ".join(values) + " |")
    estimate_headers = " | ".join(payload["models"])
    estimate_rules = "|".join("---:" for _ in payload["models"])
    return """# Micro/mini measured comparison

| Model | Model type | Mode | Frames | Processing units | Pure sec/output frame | E2E sec/output frame | Peak process VRAM MiB | Downsample SSIM | Downsample PSNR dB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Long-video projection fitted from micro + mini

| Frames | {estimate_headers} |
|---:|{estimate_rules}|
{estimates}

The fit separates one-time model load, causal/window warm-up, the incremental
steady temporal unit, and mini-run CPU/I/O seconds per output frame. Quality
numbers are source-preservation after downsampling, not ground-truth 16K scores.
""".format(
        rows="\n".join(rows), estimates="\n".join(estimates),
        estimate_headers=estimate_headers, estimate_rules=estimate_rules,
    )


def generate_contact_sheets(result_root: Path, quality: dict[str, Any]) -> dict[str, Any]:
    """Create matched source/model crop sheets without loading 16K frames in Python."""
    output_dir = result_root / "qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "method": (
            "first/middle/last matched center region; source 512x288 and each "
            "native-x4 output 2048x1152 are scaled to 640x360"
        ),
        "layout": "columns=source_then_model_order_left_to_right; rows=sample_frames",
        "modes": {},
        "same_length_modes": {},
    }

    def make_sheet(
        *,
        mode: str,
        models: list[str],
        common_frames: int,
        source_start: int,
        destination: Path,
    ) -> dict[str, Any]:
        if not models:
            return {"status": "SKIPPED", "reason": "no models"}
        outputs: list[str] = []
        reports: list[dict[str, Any]] = []
        for model in models:
            report = _load(result_root / model / f"benchmark_{mode}.json")
            reports.append(report)
            outputs.append(str(report.get("output_path") or report.get("output")))
        input_value = reports[0].get("input")
        source = str(
            input_value.get("path") if isinstance(input_value, dict) else input_value
        )
        sample_frames = sorted({0, common_frames // 2, common_frames - 1})
        filters = []
        labels: list[str] = []
        layout: list[str] = []
        source_labels = "".join(f"[source_{row}]" for row in range(len(sample_frames)))
        filters.append(f"[0:v]split={len(sample_frames)}{source_labels}")
        for row, frame in enumerate(sample_frames):
            label = f"c0_{row}"
            filters.append(
                f"[source_{row}]select='eq(n,{source_start + frame})',"
                "crop=512:288:(iw-512)/2:(ih-288)/2,"
                "scale=640:360:flags=lanczos,"
                "drawbox=x=0:y=0:w=iw:h=38:color=black@0.55:t=fill,"
                f"drawtext=text='source f{source_start + frame}':x=10:y=7:fontsize=22:fontcolor=white,"
                f"setsar=1,setpts=PTS-STARTPTS[{label}]"
            )
            labels.append(f"[{label}]")
            layout.append(f"0_{row * 360}")
        for model_index in range(len(outputs)):
            input_index = model_index + 1
            column = model_index + 1
            split_labels = "".join(
                f"[s{model_index}_{row}]" for row in range(len(sample_frames))
            )
            filters.append(f"[{input_index}:v]split={len(sample_frames)}{split_labels}")
            for row, frame in enumerate(sample_frames):
                label = f"c{column}_{row}"
                filters.append(
                    f"[s{model_index}_{row}]select='eq(n,{frame})',"
                    "crop=2048:1152:(iw-2048)/2:(ih-1152)/2,"
                    "scale=640:360:flags=lanczos,"
                    "drawbox=x=0:y=0:w=iw:h=38:color=black@0.55:t=fill,"
                    f"drawtext=text='{models[model_index]} f{source_start + frame}':"
                    "x=10:y=7:fontsize=22:fontcolor=white,"
                    f"setsar=1,setpts=PTS-STARTPTS[{label}]"
                )
                labels.append(f"[{label}]")
                layout.append(f"{column * 640}_{row * 360}")
        filters.append(
            "".join(labels)
            + f"xstack=inputs={len(labels)}:layout={'|'.join(layout)}[sheet]"
        )
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", source]
        for output in outputs:
            command += ["-i", output]
        command += ["-filter_complex", ";".join(filters), "-map", "[sheet]", "-frames:v", "1", "-y", str(destination)]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        return {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "column_order_left_to_right": ["source_lanczos_preview", *models],
            "sample_frames_top_to_bottom": sample_frames,
            "path": str(destination),
            "error": result.stderr.strip() or None,
        }

    for mode, item in quality.get("modes", {}).items():
        models = list(item["models"])
        source_start = int(item.get("source_start_frame", 0))
        manifest["modes"][mode] = make_sheet(
            mode=mode,
            models=models,
            common_frames=int(item["common_frame_count"]),
            source_start=source_start,
            destination=output_dir / f"{mode}_center_crop_contact_sheet.png",
        )
        same_length_models = [
            model
            for model in ("FlashVSR_v1.1", "Real-ESRGAN", "HAT")
            if model in item["models"]
        ]
        if same_length_models:
            same_length_frames = min(
                int(
                    _load(result_root / model / f"benchmark_{mode}.json").get(
                        "effective_output_frames_written", item["common_frame_count"]
                    )
                )
                for model in same_length_models
            )
            manifest["same_length_modes"][mode] = make_sheet(
                mode=mode,
                models=same_length_models,
                common_frames=same_length_frames,
                source_start=source_start,
                destination=(
                    output_dir / f"{mode}_matched_{same_length_frames}f_contact_sheet.png"
                ),
            )
        else:
            manifest["same_length_modes"][mode] = {
                "status": "SKIPPED",
                "reason": "no FlashVSR/frame-image reports",
            }
    atomic_write_json(output_dir / "contact_sheet_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--frames", type=int, nargs="+", default=list(TARGETS))
    parser.add_argument("--ffmpeg-threads", type=int, default=12)
    args = parser.parse_args(argv)
    if any(frame <= 0 for frame in args.frames):
        parser.error("--frames values must be positive")
    result_root = Path(args.result_root).resolve()
    payload = synthesize(result_root, args.frames)
    atomic_write_json(result_root / "benchmark_estimates.json", payload)
    atomic_write_text(result_root / "performance_comparison.md", render(result_root, payload))
    quality = analyze_common_quality(result_root, ffmpeg_threads=args.ffmpeg_threads)
    atomic_write_json(result_root / "quality_comparison.json", quality)
    atomic_write_text(
        result_root / "quality_comparison.md", render_quality_comparison(quality)
    )
    generate_contact_sheets(result_root, quality)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
