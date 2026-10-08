#!/usr/bin/env python3
"""Encode/composite one validated 170-tile aggregate without starting inference."""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = EXPERIMENT_ROOT / "package"
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cert_pipeline import flash_composite  # noqa: E402


def main() -> int:
    parser = flash_composite.parser()
    parser.description = (
        "Postprocess a COMPLETE global 170-tile A6000 aggregate; inference fallback is disabled"
    )
    parser.add_argument("--aggregate-manifest", type=Path, required=True)
    parser.set_defaults(
        tile_size=256,
        tile_overlap=32,
        seed=71634910,
        sparse_ratio=2.0,
        local_range=11,
    )
    args = parser.parse_args()
    flash_composite.resolve_paths(args)
    args.aggregate_manifest = args.aggregate_manifest.expanduser().resolve()
    try:
        if args.dry_run or args.self_test:
            raise flash_composite.PipelineError(
                "postprocess_video does not provide inference dry-run/self-test modes"
            )
        if args.expected_frames != 1952:
            raise flash_composite.PipelineError("A6000 03/04 postprocess requires 1952 frames")
        if str(args.tile_size) != "256" or str(args.tile_overlap) != "32":
            raise flash_composite.PipelineError("tile_size=256 and tile_overlap=32 are mandatory")
        if args.output.suffix.lower() != ".mp4" or args.raw_output.suffix.lower() != ".mp4":
            raise flash_composite.PipelineError("raw and final outputs must use MP4")
        if args.output in {args.input_4k, args.original_8k}:
            raise flash_composite.PipelineError("output must not overwrite either source")
        flash_composite.execute_postprocess(args)
        return 0
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
