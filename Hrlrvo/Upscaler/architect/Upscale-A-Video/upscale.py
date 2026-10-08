#!/usr/bin/env python3
"""Official-path availability adapter for Upscale-A-Video.

The released upstream inference downsamples inputs whose height and width are
both at least 1280 before applying its 4x model.  A 3840x2160 source therefore
returns 3840x2160 rather than the required 15360x8640.  This adapter records
that reproducibly and refuses to label the official path as a 4K-to-16K run.
It never trains, fine-tunes, or silently patches upstream inference behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ARCHITECT = HERE.parent
ROOT = ARCHITECT.parent
if str(ARCHITECT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT))

from common.media_utils import atomic_write_json, atomic_write_text, probe_media


MODEL_NAME = "Upscale-A-Video"
MODEL_ROOT = ROOT / "models" / MODEL_NAME
REPOSITORY = MODEL_ROOT / "repository"
PRETRAINED = REPOSITORY / "pretrained_models" / "upscale_a_video"
DEFAULT_INPUT = ROOT / "input" / "0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4"
DEFAULT_RESULT_DIR = ROOT / "result" / "benchmark" / MODEL_NAME
PINNED_COMMIT = "10ca1d751cad25b8ac0b9b0cab415a82d9df1ac6"
REQUIRED_WEIGHT_DIRS = (
    "low_res_scheduler",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "unet",
    "vae",
)


def official_output_dimensions(width: int, height: int) -> tuple[int, int]:
    if height >= 1280 and width >= 1280:
        width, height = width // 4, height // 4
    return width * 4, height * 4


def benchmark_frame_count(
    mode: str,
    total_frames: int,
    start_frame: int,
    max_frames: int | None,
) -> int:
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start frame {start_frame} is outside [0, {total_frames})")
    defaults = {"micro": 8, "mini": 14, "full": total_frames - start_frame}
    count = min(max_frames if max_frames is not None else defaults[mode], total_frames - start_frame)
    if count <= 0:
        raise ValueError("selected frame count must be positive")
    if mode in {"micro", "mini"} and count < 8:
        raise ValueError("Upscale-A-Video temporal benchmark requires at least 8 frames")
    return count


def theoretical_tile_count(width: int, height: int, tile_core: int = 256) -> int:
    return math.ceil(width / tile_core) * math.ceil(height / tile_core)


def git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    metadata = probe_media(args.input, count_frames=True)
    selected_frames = benchmark_frame_count(
        args.benchmark_mode,
        metadata.frame_count,
        args.start_frame,
        args.max_frames,
    )
    official_width, official_height = official_output_dimensions(metadata.width, metadata.height)
    required_width, required_height = metadata.width * 4, metadata.height * 4
    missing_weights = [name for name in REQUIRED_WEIGHT_DIRS if not (PRETRAINED / name).is_dir()]
    repository_commit = git_output("rev-parse", "HEAD")
    repository_origin = git_output("remote", "get-url", "origin")
    valid_16k = (official_width, official_height) == (required_width, required_height)
    return {
        "model": MODEL_NAME,
        "status": "AVAILABLE" if valid_16k and not missing_weights else "BLOCKED_REFERENCE_ONLY",
        "benchmark_mode": args.benchmark_mode,
        "input": str(Path(args.input).resolve()),
        "input_metadata": metadata.to_dict(),
        "frame_selection": {
            "start_frame": args.start_frame,
            "selected_frames": selected_frames,
            "minimum_meaningful_temporal_unit": 8,
            "mini_temporal_unit": 14,
        },
        "required_output": {"width": required_width, "height": required_height},
        "untouched_official_output": {"width": official_width, "height": official_height},
        "untouched_official_path_is_valid_4k_to_16k": valid_16k,
        "blocking_reason": (
            "Official inference downsamples 4K to 960x540 before 4x SR, producing 4K rather than 16K. "
            "Bypassing that branch would be a custom inference modification."
            if not valid_16k
            else None
        ),
        "official_temporal_structure": {
            "denoise_window": 8,
            "denoise_overlap": 2,
            "stride": 6,
            "vae_decode_chunk": 3,
            "diffusion_steps_default": 30,
            "direction": "local bidirectional window; optional RAFT bidirectional recurrent propagation",
        },
        "official_spatial_structure": {
            "tile_core": 256,
            "context_halo": 64,
            "merge": "crop halo and assign core; no weighted blend",
            "theoretical_direct_4k_tile_count_if_official_downsample_were_bypassed": theoretical_tile_count(
                metadata.width, metadata.height
            ),
        },
        "repository": str(REPOSITORY),
        "repository_origin": repository_origin,
        "repository_commit": repository_commit,
        "expected_pinned_commit": PINNED_COMMIT,
        "pretrained_root": str(PRETRAINED),
        "missing_weight_directories": missing_weights,
        "weights_download_source": (
            "https://drive.google.com/drive/folders/1O8pbeR1hsRlFUU8O4EULe-lOKNGEWZl1"
        ),
        "training_or_finetuning_performed": False,
    }


def markdown(report: dict[str, Any]) -> str:
    source = report["input_metadata"]
    required = report["required_output"]
    official = report["untouched_official_output"]
    return f"""# Upscale-A-Video availability

- Status: **{report['status']}**
- Official repository commit: `{report['repository_commit']}`
- Input: {source['width']}×{source['height']}
- Required output: {required['width']}×{required['height']}
- Untouched official output: {official['width']}×{official['height']}
- Valid untouched official 4K→16K path: **{report['untouched_official_path_is_valid_4k_to_16k']}**
- Missing official weight directories: `{report['missing_weight_directories']}`
- Meaningful temporal unit: 8 frames; mini overlap test: 14 frames (8/2, stride 6)
- Direct-4K theoretical tiles if the official downsample were bypassed: {report['official_spatial_structure']['theoretical_direct_4k_tile_count_if_official_downsample_were_bypassed']}

## Decision

{report['blocking_reason']}

The official repository and download location are recorded, but no modified or
mislabelled 16K inference is run. No training or fine-tuning is performed.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the official Upscale-A-Video 4K-to-16K path")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--report-json")
    parser.add_argument("--benchmark-mode", choices=["micro", "mini", "full"], default="micro")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--availability-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    result_dir = Path(args.result_dir)
    report_path = Path(args.report_json) if args.report_json else result_dir / f"benchmark_{args.benchmark_mode}.json"
    atomic_write_json(report_path, report)
    atomic_write_text(report_path.with_suffix(".md"), markdown(report))
    atomic_write_text(result_dir / "availability.md", markdown(report))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.availability_only:
        return 0
    return 0 if report["status"] == "AVAILABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
