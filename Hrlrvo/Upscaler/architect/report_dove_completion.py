#!/usr/bin/env python3
"""Write a compact post-run DOVE and cross-model comparison report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.media_utils import atomic_write_json, atomic_write_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT_ROOT = ROOT / "result" / "benchmark"
MODELS = ("FlashVSR_v1.1", "SeedVR2", "DOVE", "Real-ESRGAN", "HAT")
DOVE_MARKER = "<!-- DOVE_BENCHMARK_ADDENDUM -->"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def timing(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("timings") or report.get("timing") or {}


def build_payload(result_root: Path, launcher_exit_code: int) -> dict[str, Any]:
    status = load_json(result_root / "run_status_dove.json") or {}
    estimates = load_json(result_root / "benchmark_estimates.json") or {}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for mode in ("micro", "mini"):
            report = load_json(result_root / model / f"benchmark_{mode}.json")
            if not report:
                continue
            values = timing(report)
            gpu = report.get("gpu", {})
            structure = report.get("structure_analysis", {})
            rows.append({
                "model": model,
                "mode": mode,
                "status": report.get("status"),
                "technical_validation": report.get("technical_validation", {}).get("status"),
                "frames": report.get("effective_output_frames_written"),
                "total_wall_seconds": values.get("total_wall_seconds"),
                "pure_seconds_per_output_frame": values.get("pure_inference_seconds_per_output_frame"),
                "end_to_end_seconds_per_output_frame": values.get("end_to_end_seconds_per_output_frame"),
                "peak_process_gpu_mib": gpu.get(
                    "peak_process_gpu_memory_mb", gpu.get("peak_process_gpu_mib")
                ),
                "ssim": structure.get("mean_ssim"),
                "psnr_db": structure.get("mean_psnr_db"),
                "temporal_status": report.get("temporal_analysis", {}).get("status"),
                "output_path": report.get("output_path") or report.get("output"),
            })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "launcher_exit_code": launcher_exit_code,
        "launcher_status": status.get("status", "UNKNOWN"),
        "dove_runs": status.get("runs", []),
        "comparison_rows": rows,
        "long_video_estimates": estimates.get("models", {}).get("DOVE"),
        "artifacts": {
            "performance_comparison": str(result_root / "performance_comparison.md"),
            "quality_comparison": str(result_root / "quality_comparison.md"),
            "contact_sheet_manifest": str(
                result_root / "qualitative" / "contact_sheet_manifest.json"
            ),
        },
    }


def render(payload: dict[str, Any]) -> str:
    def number(value: Any, digits: int = 3) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "N/A"

    rows = []
    for item in payload["comparison_rows"]:
        rows.append(
            "| {model} | {mode} | {frames} | {wall} | {pure} | {e2e} | {vram} | {ssim} | {psnr} | {temporal} |".format(
                model=item["model"],
                mode=item["mode"],
                frames=item["frames"],
                wall=number(item["total_wall_seconds"]),
                pure=number(item["pure_seconds_per_output_frame"]),
                e2e=number(item["end_to_end_seconds_per_output_frame"]),
                vram=number(item["peak_process_gpu_mib"], 0),
                ssim=number(item["ssim"], 6),
                psnr=number(item["psnr_db"]),
                temporal=item["temporal_status"] or "N/A",
            )
        )
    dove_outputs = [
        f"- `{item['mode']}`: `{item['output_path']}`"
        for item in payload["comparison_rows"]
        if item["model"] == "DOVE" and item.get("output_path")
    ]
    estimates = payload.get("long_video_estimates") or {}
    estimate_rows = []
    for frames, value in estimates.get("estimates", {}).items():
        estimate_rows.append(
            f"| {frames} | {number(value.get('estimated_total_seconds'))} | "
            f"{number(value.get('estimated_total_hours'))} | "
            f"{value.get('temporal_chunks', 'N/A')} |"
        )
    return f"""# DOVE benchmark completion report

- Launcher status: **{payload['launcher_status']}**
- Launcher exit code: `{payload['launcher_exit_code']}`
- Generated: `{payload['generated_at']}`

## Measured comparison

| Model | Mode | Frames | Wall s | Pure s/frame | E2E s/frame | Peak process VRAM MiB | SSIM | PSNR dB | Temporal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## DOVE long-video projection

| Frames | Estimated seconds | Estimated hours | Temporal chunks |
|---:|---:|---:|---:|
{chr(10).join(estimate_rows) if estimate_rows else '| N/A | N/A | N/A | N/A |'}

## DOVE outputs

{chr(10).join(dove_outputs) if dove_outputs else '- No validated DOVE output was produced.'}

## Review artifacts

- Quantitative/time/VRAM table: `{payload['artifacts']['performance_comparison']}`
- Common-frame quality metrics: `{payload['artifacts']['quality_comparison']}`
- Matched visual crops: `{payload['artifacts']['contact_sheet_manifest']}`

SSIM/PSNR here measure source preservation after 16K output is downsampled to
4K. They are not ground-truth 16K perceptual scores. Review the matched contact
sheet and the native DOVE mini video for hallucination, seams, and flicker.
"""


def render_summary_addendum(payload: dict[str, Any]) -> str:
    dove_rows = [
        item for item in payload["comparison_rows"] if item["model"] == "DOVE"
    ]
    lines = [
        DOVE_MARKER,
        "## DOVE Stage-2 추가 benchmark",
        "",
        f"- Launcher status: **{payload['launcher_status']}**",
        "- Model type: one-step tiled video diffusion SR",
        "- Temporal samples: micro 9 frames / mini 17 frames",
        "- Spatial plan: 768×1360 output-domain tiles, overlap 32; bounded OOM fallback",
        "- Checkpoint provenance: official-author DOVE components + official CogVideoX1.5 VAE",
        "",
        "| Mode | Frames | Wall s | Pure s/frame | E2E s/frame | Peak process VRAM MiB | SSIM | PSNR dB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def value(row: dict[str, Any], key: str, digits: int = 3) -> str:
        raw = row.get(key)
        try:
            return f"{float(raw):.{digits}f}"
        except (TypeError, ValueError):
            return "N/A"

    for item in dove_rows:
        lines.append(
            f"| {item['mode']} | {item['frames']} | {value(item, 'total_wall_seconds')} | "
            f"{value(item, 'pure_seconds_per_output_frame')} | "
            f"{value(item, 'end_to_end_seconds_per_output_frame')} | "
            f"{value(item, 'peak_process_gpu_mib', 0)} | {value(item, 'ssim', 6)} | "
            f"{value(item, 'psnr_db')} |"
        )
    if not dove_rows:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    estimates = (payload.get("long_video_estimates") or {}).get("estimates", {})
    lines.extend(["", "### DOVE projections"])
    for frames in ("92", "1920", "2880"):
        if frames in estimates:
            item = estimates[frames]
            lines.append(
                f"- {frames} frames: {float(item['estimated_total_seconds']):.1f} s "
                f"({float(item['estimated_total_hours']):.3f} h), "
                f"temporal chunks {item.get('temporal_chunks')}"
            )
    lines.extend([
        "",
        "The DOVE rank must be decided from both its native 16K mini playback and",
        "the regenerated common-frame metrics/contact sheet; source-preservation",
        "metrics alone do not measure perceptual detail quality.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    parser.add_argument("--launcher-exit-code", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.result_root).resolve()
    destination = root / "DOVE"
    destination.mkdir(parents=True, exist_ok=True)
    payload = build_payload(root, args.launcher_exit_code)
    atomic_write_json(destination / "final_completion.json", payload)
    atomic_write_text(destination / "final_comparison.md", render(payload))
    summary_path = root / "summary.md"
    prior = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else "# Benchmark summary\n"
    prior = prior.split(DOVE_MARKER, 1)[0].rstrip()
    atomic_write_text(summary_path, prior + "\n\n" + render_summary_addendum(payload))
    print(json.dumps({
        "status": payload["launcher_status"],
        "report": str(destination / "final_comparison.md"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
