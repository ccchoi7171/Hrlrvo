#!/usr/bin/env python3
"""Final-only A6000 postprocess for an already validated 170-tile aggregate.

This versioned overlay deliberately does not create, probe, or validate the
redundant raw 16K intermediate.  It reuses the frozen production compositor,
encoder selection, center-anchor blend, audio mux, provenance validation, and
bounded final-output evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable


OVERLAY_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = OVERLAY_ROOT.parent
FROZEN_COMPOSITE_PATH = (
    EXPERIMENT_ROOT / "package" / "src" / "cert_pipeline" / "flash_composite.py"
)
RAW_SKIP_REASON = "user_authorized_final_only_no_dependency"


def _load_frozen_composite() -> Any:
    """Load the pinned local implementation by absolute path, not PYTHONPATH order."""
    module_name = "_a6000_frozen_flash_composite_v1"
    existing = sys.modules.get(module_name)
    if existing is not None:
        try:
            existing_path = Path(existing.__file__).resolve()
        except (AttributeError, TypeError) as error:
            raise RuntimeError("cached frozen compositor has no trustworthy file path") from error
        if existing_path != FROZEN_COMPOSITE_PATH:
            raise RuntimeError("cached frozen compositor resolved to an unexpected file")
        return existing
    spec = importlib.util.spec_from_file_location(module_name, FROZEN_COMPOSITE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen compositor: {FROZEN_COMPOSITE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != FROZEN_COMPOSITE_PATH:
        raise RuntimeError("frozen compositor resolved to an unexpected file")
    return module


flash_composite = _load_frozen_composite()
EncodeFunction = Callable[..., dict[str, Any]]


def require_fresh_final_targets(output: Path, final_manifest: Path) -> None:
    """Reject only artifacts this final-only path owns.

    ``raw_output`` remains a legacy identity target in the aggregate manifest.
    It is neither read nor written here, so a historical file at that path is
    intentionally irrelevant and remains untouched.
    """
    existing = [path for path in (output, final_manifest) if path.exists() or path.is_symlink()]
    if existing:
        raise flash_composite.PipelineError(
            "final-only postprocess is fresh-only; refusing existing output/manifest: "
            + ", ".join(map(str, existing))
        )


def validate_args_contract(args: argparse.Namespace) -> None:
    """Pin the production inference/composite contract accepted by this overlay."""
    expected = {
        "expected_frames": 1952,
        "tile_size": 256,
        "tile_overlap": 32,
        "min_tile_size": 256,
        "seed": 25621683,
        "sparse_ratio": 2.0,
        "local_range": 11,
        "ffmpeg_threads": 32,
    }
    if args.video_id != "02":
        raise flash_composite.PipelineError("final-only A6000 path is restricted to video 02")
    for field, required in expected.items():
        actual = getattr(args, field)
        try:
            matches = float(actual) == float(required)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise flash_composite.PipelineError(
                f"final-only contract requires {field}={required}, got {actual!r}"
            )
    if not 1 <= args.compose_workers <= 16:
        raise flash_composite.PipelineError("compose_workers must be in the bounded range 1..16")
    if args.encoder != "libx265":
        raise flash_composite.PipelineError(
            "final-only production contract requires encoder=libx265"
        )
    if args.renderer not in {"original", "threaded"}:
        raise flash_composite.PipelineError(f"unsupported renderer: {args.renderer!r}")
    if args.output.suffix.lower() != ".mp4" or args.raw_output.suffix.lower() != ".mp4":
        raise flash_composite.PipelineError("raw legacy target and final output must use MP4")
    if args.output in {args.input_4k, args.original_8k, args.raw_output}:
        raise flash_composite.PipelineError("final output must not overwrite a source or raw target")


def load_sibling_parallel_renderer() -> EncodeFunction:
    path = OVERLAY_ROOT / "parallel_renderer.py"
    spec = importlib.util.spec_from_file_location("_a6000_parallel_renderer_v2", path)
    if spec is None or spec.loader is None:
        raise flash_composite.PipelineError(f"threaded renderer is unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "encode_staged_tiles", None)
    if not callable(function):
        raise flash_composite.PipelineError(
            "threaded renderer does not export callable encode_staged_tiles"
        )
    return function


def select_encode_function(args: argparse.Namespace) -> EncodeFunction:
    if args.renderer == "original":
        return flash_composite.encode_staged_tiles
    return load_sibling_parallel_renderer()


def execute_final_only(
    args: argparse.Namespace,
    *,
    encode_function: EncodeFunction | None = None,
) -> dict[str, Any]:
    """Create only the final center-composited MP4 from a trusted aggregate."""
    # Preserve the production invariant that an input/anchor mix-up fails
    # before probes, imports, directory creation, or encoding.
    flash_composite.assert_video_identity_pair(
        args.video_id, args.input_4k, args.original_8k
    )
    validate_args_contract(args)
    if not args.aggregate_manifest.is_file():
        raise flash_composite.PipelineError(
            "validated aggregate manifest is required; inference fallback is disabled: "
            f"{args.aggregate_manifest}"
        )
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise flash_composite.PipelineError(
                f"required executable is not available: {executable}"
            )
    for path in (args.input_4k, args.original_8k, args.outpainting_config):
        if not path.is_file():
            raise flash_composite.PipelineError(f"required input/config file is missing: {path}")
    missing_assets = [
        path
        for path in flash_composite.required_assets(
            args.model_root, args.flashvsr_reference
        )
        if not path.is_file()
    ]
    if missing_assets:
        raise flash_composite.PipelineError(
            "missing FlashVSR asset(s):\n" + "\n".join(map(str, missing_assets))
        )
    require_fresh_final_targets(args.output, args.final_manifest)

    approved_reference_sha256 = flash_composite.require_approved_reference(
        args.flashvsr_reference
    )
    reference = flash_composite.load_reference_module(args.flashvsr_reference)
    if Path(reference.MODEL_ROOT).resolve() != args.model_root:
        raise flash_composite.PipelineError("FlashVSR reference MODEL_ROOT mismatch")
    if int(reference.SCALE) != flash_composite.SCALE:
        raise flash_composite.PipelineError("FlashVSR reference scale mismatch")
    try:
        aggregate = reference.read_stage_manifest(args.aggregate_manifest)
    except Exception as error:
        raise flash_composite.PipelineError(
            f"aggregate manifest is not a complete FlashVSR stage: {error}"
        ) from error

    input_metadata = flash_composite.probe_video_headers(args.input_4k)
    original_metadata = flash_composite.probe_video_headers(args.original_8k)
    flash_composite.validate_video_spec(
        input_metadata,
        label="4K outpaint input",
        resolution=flash_composite.EXPECTED_4K,
        fps=flash_composite.EXPECTED_INPUT_4K_FPS,
        frame_count=args.expected_frames,
    )
    flash_composite.validate_video_spec(
        original_metadata,
        label="8K anchor input",
        resolution=flash_composite.EXPECTED_8K,
        fps=flash_composite.EXPECTED_ORIGINAL_8K_FPS,
        frame_count=args.expected_frames,
    )
    input_identity = flash_composite.validate_trusted_source_identity(
        args.input_4k,
        aggregate.get("input_identity") or aggregate.get("input"),
        label="4K input",
    )
    anchor_identity = flash_composite.validate_trusted_source_identity(
        args.original_8k,
        aggregate.get("anchor_identity"),
        label="8K anchor",
    )
    source_feather_width = flash_composite.read_outpainting_feather_width(
        args.outpainting_config
    )
    feather_width = source_feather_width * flash_composite.SCALE
    if feather_width != 64:
        raise flash_composite.PipelineError(
            f"center feather width is {feather_width}, expected 64"
        )
    tiles, tile_dirs, _ = flash_composite.validate_aggregate_contract(
        aggregate,
        args=args,
        input_metadata=input_metadata,
        reference=reference,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.final_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    media_utils = flash_composite.load_media_utils(args.flashvsr_reference)
    selected_encoder, encoder_attempts = media_utils.select_encoder(
        flash_composite.EXPECTED_16K[0],
        flash_composite.EXPECTED_16K[1],
        threads=args.ffmpeg_threads,
        scratch_dir=args.work_dir / "encoder_probe",
        preferred=args.encoder,
    )
    if selected_encoder["extension"] != ".mp4":
        raise flash_composite.PipelineError(
            "final composite contract requires an MP4-capable encoder"
        )

    encoder_callback = encode_function or select_encode_function(args)
    partial = args.output.with_name(f".{args.output.stem}.partial.{os.getpid()}.mp4")
    mux_partial = args.output.with_name(
        f".{args.output.stem}.audio.partial.{os.getpid()}.mp4"
    )
    manifest_partial = args.final_manifest.with_name(
        f".{args.final_manifest.name}.partial.{os.getpid()}.json"
    )
    if any(path.exists() or path.is_symlink() for path in (partial, mux_partial, manifest_partial)):
        raise flash_composite.PipelineError("final-only partial output path already exists")
    published = False
    committed = False
    published_inode = None
    mux_exit_code: int | None = None
    try:
        encode_metrics = encoder_callback(
            args=args,
            reference=reference,
            media_utils=media_utils,
            input_metadata=input_metadata,
            tiles=tiles,
            tile_dirs=tile_dirs,
            selected_encoder=selected_encoder,
            partial_output=partial,
            feather_width=feather_width,
        )
        if int(encode_metrics.get("frames_written", -1)) != args.expected_frames:
            raise flash_composite.PipelineError(
                "final composite encoder frame counter mismatch"
            )

        require_fresh_final_targets(args.output, args.final_manifest)

        original_media = media_utils.probe_media(args.original_8k, count_frames=False)
        if original_media.audio_present:
            mux_command = [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(partial),
                "-i",
                str(args.original_8k),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                "-t",
                f"{args.expected_frames / float(flash_composite.EXPECTED_OUTPUT_FPS):.9f}",
                "-movflags",
                "+faststart",
                "-y",
                str(mux_partial),
            ]
            mux_result = flash_composite.run(
                mux_command, check=False, capture_output=True
            )
            mux_exit_code = mux_result.returncode
            if mux_exit_code:
                raise flash_composite.PipelineError(
                    f"original-audio mux failed: {mux_result.stderr}"
                )
            publish_source = mux_partial
        else:
            mux_exit_code = 0
            publish_source = partial
        source_stat = publish_source.stat()
        os.link(publish_source, args.output)
        published = True
        published_inode = (source_stat.st_dev, source_stat.st_ino)
        for temporary in (partial, mux_partial):
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)

        final_metadata = flash_composite.probe_video_headers(args.output)
        final_evidence = flash_composite.bounded_output_evidence(
            args.output,
            final_metadata,
            expected_frames=args.expected_frames,
            encoder_frames_written=int(encode_metrics["frames_written"]),
            encoder_exit_code=0,
            role="final 16K center composite",
        )
        final_evidence["audio_mux_exit_code"] = mux_exit_code
        request = flash_composite.stage_request_payload(
            args,
            input_metadata,
            original_metadata,
            feather_width,
            input_identity=input_identity,
            original_identity=anchor_identity,
        )
        raw_skip = {
            "status": "SKIPPED",
            "raw_16k_intermediate_video": False,
            "reason": RAW_SKIP_REASON,
            "legacy_target": str(args.raw_output),
            "file_created_by_final_only": False,
            "file_probed_by_final_only": False,
        }
        manifest = {
            "schema_version": flash_composite.PIPELINE_SCHEMA_VERSION,
            "status": "PASS",
            "validation_level": "BOUNDED_PRODUCER_AND_HEADER_EVIDENCE",
            "video_id": args.video_id,
            "request": request,
            "aggregate_manifest": flash_composite.file_identity(
                args.aggregate_manifest, with_sha256=True
            ),
            "trusted_sources": {
                "input_4k": input_identity,
                "original_8k": anchor_identity,
            },
            "input_4k": input_metadata,
            "original_8k": original_metadata,
            "output": final_metadata,
            "output_file": flash_composite.file_identity(args.output),
            "flashvsr": {
                "model": flash_composite.FLASHVSR_MODEL_NAME,
                "reference_implementation": str(args.flashvsr_reference),
                "reference_sha256": approved_reference_sha256,
                "configuration": aggregate["partial_report"]["configuration"],
                "tile_count": len(tiles),
                "global_tile_indices": list(range(170)),
                "inference_started_by_postprocessor": False,
            },
            "center_composite": {
                "bbox_yx": list(flash_composite.CENTER_BBOX),
                "original_resized": False,
                "method": "Alpha Feather Blending",
                "alpha_profile": "inward nearest-edge smoothstep t^2*(3-2t)",
                "source_4k_feather_width_pixels": source_feather_width,
                "output_16k_feather_width_pixels": feather_width,
            },
            "encoding": {
                "raw_status": "SKIPPED",
                "raw_16k_intermediate_video": False,
                "raw_skip_reason": RAW_SKIP_REASON,
                "raw_output_legacy_target": str(args.raw_output),
                "renderer": args.renderer,
                "compose_workers": args.compose_workers,
                "encoder": selected_encoder,
                "encoder_probe_attempts": encoder_attempts,
                **encode_metrics,
            },
            "verification_policy": {
                "schema_version": flash_composite.BOUNDED_VERIFICATION_SCHEMA_VERSION,
                "name": "producer_receipts_encoder_counters_and_header_probe",
                "full_16k_decode_performed": False,
                "frame_timestamp_decode_performed": False,
                "tile_completion": (
                    "170 global tile artifacts and committed worker emissions"
                ),
                "raw_output": raw_skip,
                "final_output": final_evidence,
                "limitations": (
                    "PASS does not claim a decoded full-frame count or visual-quality review"
                ),
            },
        }
        flash_composite.atomic_write_json(manifest_partial, manifest)
        os.link(manifest_partial, args.final_manifest)
        committed = True
        with contextlib.suppress(OSError):
            manifest_partial.unlink()
        print(f"[DONE] video_id={args.video_id} output={args.output}", flush=True)
        return manifest
    except BaseException:
        for temporary in (partial, mux_partial, manifest_partial):
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
        output_stat = args.output.stat() if args.output.exists() else None
        own_output = output_stat is not None and (
            output_stat.st_dev, output_stat.st_ino
        ) == published_inode
        if published and own_output and not committed:
            orphan_dir = args.work_dir / "orphaned_outputs"
            orphan_dir.mkdir(parents=True, exist_ok=True)
            os.replace(
                args.output,
                orphan_dir / f"{args.output.stem}.{time.time_ns()}.mp4",
            )
        raise


def parser() -> argparse.ArgumentParser:
    argument_parser = flash_composite.parser()
    argument_parser.description = (
        "Final-only postprocess of a COMPLETE global 170-tile A6000 aggregate"
    )
    argument_parser.add_argument("--aggregate-manifest", type=Path, required=True)
    argument_parser.add_argument("--compose-workers", type=int, default=1)
    argument_parser.add_argument(
        "--renderer", choices=["original", "threaded"], default="original"
    )
    argument_parser.set_defaults(
        tile_size=256,
        tile_overlap=32,
        min_tile_size=256,
        seed=25621683,
        sparse_ratio=2.0,
        local_range=11,
    )
    return argument_parser


def resolve_paths(args: argparse.Namespace) -> None:
    for name in ("output", "final_manifest"):
        if getattr(args, name).expanduser().is_symlink():
            raise flash_composite.PipelineError(f"final-only {name} must not be a symlink")
    flash_composite.resolve_paths(args)
    args.aggregate_manifest = args.aggregate_manifest.expanduser().resolve()


def main() -> int:
    args = parser().parse_args()
    try:
        resolve_paths(args)
        if args.dry_run or args.self_test:
            raise flash_composite.PipelineError(
                "final-only postprocess does not provide inference dry-run/self-test modes"
            )
        validate_args_contract(args)
        execute_final_only(args)
        return 0
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
