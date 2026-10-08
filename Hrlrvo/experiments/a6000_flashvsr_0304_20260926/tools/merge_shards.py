#!/usr/bin/env python3
"""Validate two complementary FlashVSR shards and publish one full stage manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSCALE = ROOT / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py"
BASE_SEED = 71634910
APPROVED_UPSCALE_SHA256 = "3db9c0524418070a87b26b948612f8988cd47f007c6e7739a89d3911ed0d703d"
EXPECTED_INDICES = list(range(170))
ASSIGNMENTS = {
    ("04", 0): list(range(0, 85)), ("04", 1): list(range(85, 170)),
    ("03", 2): list(range(0, 85)), ("03", 3): list(range(85, 170)),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_upscale(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("a6000_merge_flashvsr_upscale", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import FlashVSR snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes aggregate work directory: {resolved}") from error
    return resolved


def _expected_layout(upscale_path: Path) -> list[dict[str, int]]:
    upscale = load_upscale(upscale_path)
    return [
        {"index": tile.index, "x": tile.x, "y": tile.y, "width": tile.width, "height": tile.height}
        for tile in upscale.make_tiles(3840, 2160, 256, 32)
    ]


def require_approved_upscale(path: Path) -> str:
    actual = sha256_file(path)
    if actual != APPROVED_UPSCALE_SHA256:
        raise RuntimeError(
            f"Aggregate reference is not the approved FlashVSR upscale snapshot: {path} "
            f"(expected {APPROVED_UPSCALE_SHA256}, got {actual})"
        )
    return actual


def _check_identity(identity: Any, label: str) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError(f"Missing {label} identity")
    if set(("path", "size", "sha256")) - set(identity):
        raise ValueError(f"Incomplete {label} identity")
    if not isinstance(identity["size"], int) or identity["size"] <= 0:
        raise ValueError(f"Invalid {label} size")
    digest = identity["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Invalid {label} sha256")
    return identity


def _require_exact_names(actual: set[str], expected: set[str], label: str) -> None:
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f"{label} differs; missing={missing}, unexpected={unexpected}")


def validate_shard_receipt(
    receipt_path: Path,
    *,
    video_id: str,
    aggregate_work_dir: Path,
    expected_layout: list[dict[str, int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = contained(receipt_path, aggregate_work_dir, "Shard receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "a6000_flashvsr_tile_shard/v1":
        raise ValueError("Unsupported tile shard receipt schema")
    # A shard deliberately cannot masquerade as the full-stage sentinel.
    if receipt.get("state") != "SHARD_COMPLETE" or receipt.get("completion_scope") != "PARTIAL_SPATIAL_TILES":
        raise ValueError("Receipt is not a bounded partial spatial shard")
    if receipt.get("video_id") != video_id or receipt.get("mode") != "full":
        raise ValueError("Shard receipt video/mode does not match the requested full aggregation")
    if receipt.get("subprocess", {}).get("exit_code") != 0:
        raise ValueError("Shard child process did not exit successfully")
    if receipt.get("upscale_sha256") != APPROVED_UPSCALE_SHA256:
        raise ValueError("Shard receipt does not prove the approved FlashVSR upscale snapshot")
    params = receipt.get("parameters", {})
    required_params = {
        "tile_size": 256,
        "tile_overlap": 32,
        "base_seed": BASE_SEED,
        "sparse_ratio": 2.0,
        "local_range": 11,
        "precision": "torch.bfloat16 (original implementation)",
        "ffmpeg_threads": 32,
        "encoder": "libx265",
        "benchmark_mode": "full",
        "max_frames": None,
    }
    if params != required_params:
        raise ValueError(f"Shard parameters differ from the approved production values: {params}")
    _check_identity(receipt.get("input_identity"), "input")
    _check_identity(receipt.get("anchor_identity"), "anchor")
    shard_work = contained(Path(receipt["work_dir"]), aggregate_work_dir, "Shard work directory")
    request_path = contained(Path(receipt["request_path"]), shard_work, "Shard request")
    manifest_path = contained(Path(receipt["private_partial_manifest"]), shard_work, "Partial manifest")
    log_path = contained(Path(receipt["subprocess"]["log_path"]), shard_work, "Shard log")
    for path, digest, label in (
        (request_path, receipt["request_sha256"], "request"),
        (manifest_path, receipt["private_partial_manifest_sha256"], "partial manifest"),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Shard {label} hash mismatch: {path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Shard log is missing: {log_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE" or manifest.get("model") != "FlashVSR_v1.1":
        raise ValueError("Private partial manifest is not an original successful inference result")
    if int(manifest.get("processed_input_frames", -1)) != 1952:
        raise ValueError("Full shard must contain all 1952 temporal frames")
    if manifest.get("input", {}).get("path") != receipt["input_identity"]["path"]:
        raise ValueError("Partial manifest input differs from its trusted transfer identity")
    input_meta = manifest["input"]
    if (int(input_meta.get("width", -1)), int(input_meta.get("height", -1)),
            input_meta.get("fps_fraction"), int(input_meta.get("frame_count", -1))) != (3840, 2160, "16", 1952):
        raise ValueError("Shard input metadata is not the approved 3840x2160, 16fps, 1952-frame source")
    artifacts = manifest.get("tile_artifacts", [])
    indices = [int(item["tile"]["index"]) for item in artifacts]
    if indices != receipt.get("global_tile_indices"):
        raise ValueError("Receipt and private manifest tile indices differ")
    if len(indices) != len(set(indices)) or not indices:
        raise ValueError("Shard tile indices are empty or duplicated")
    if indices != ASSIGNMENTS.get((video_id, int(receipt.get("gpu_index", -1)))):
        raise ValueError("Shard global tile indices do not match the approved video/GPU assignment")
    scratch = contained(Path(manifest["scratch_path"]), shard_work, "Shard scratch")
    expected_top = {"certification_identity.json", "tiles", "tile_commits"}
    actual_top = {item.name for item in scratch.iterdir()}
    _require_exact_names(actual_top, expected_top, "Shard scratch entries")
    expected_tile_dirs = {f"tile_{index:04d}" for index in indices}
    tile_root = scratch / "tiles"
    commit_root = scratch / "tile_commits"
    _require_exact_names({item.name for item in tile_root.iterdir()}, expected_tile_dirs,
                         "Shard tile directory set")
    _require_exact_names({item.name for item in commit_root.iterdir()},
                         {f"tile_{index:04d}.json" for index in indices}, "Shard tile commit set")
    referenced_metadata: set[Path] = set()
    for artifact in artifacts:
        tile = artifact["tile"]
        index = int(tile["index"])
        if tile != expected_layout[index]:
            raise ValueError(f"Global tile {index} geometry changed")
        if int(artifact["tile_seed"]) != BASE_SEED + index:
            raise ValueError(f"Global tile {index} seed changed")
        directory = contained(Path(artifact["directory"]), shard_work, f"Tile {index} directory")
        if directory != (tile_root / f"tile_{index:04d}").resolve():
            raise ValueError(f"Tile {index} directory does not match its isolated scratch")
        expected_pngs = {f"{frame:09d}.png" for frame in range(1952)}
        actual_pngs = {item.name for item in directory.iterdir()}
        _require_exact_names(actual_pngs, expected_pngs, f"Tile {index} frame file set")
        commit_path = commit_root / f"tile_{index:04d}.json"
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        if (commit.get("state") != "COMPLETE" or commit.get("tile") != tile
                or commit.get("frame_count") != 1952 or commit.get("tile_seed") != BASE_SEED + index
                or commit.get("request_sha256") != receipt["request_sha256"]):
            raise RuntimeError(f"Tile {index} commit identity is inconsistent")
        png_entries = commit.get("png_files", [])
        if len(png_entries) != 1952:
            raise RuntimeError(f"Tile {index} commit frame counter is incomplete")
        for frame, entry in enumerate(png_entries):
            expected_path = (directory / f"{frame:09d}.png").resolve()
            if Path(entry.get("path", "")).resolve() != expected_path:
                raise RuntimeError(f"Tile {index} commit PNG ordering/path changed")
            if expected_path.stat().st_size != entry.get("size_bytes"):
                raise RuntimeError(f"Tile {index} committed PNG size changed")
        metadata_paths = [contained(Path(path), shard_work, f"Tile {index} temporal metadata")
                          for path in artifact.get("temporal_unit_metadata", [])]
        if not metadata_paths or [str(path) for path in metadata_paths] != commit.get("temporal_unit_metadata"):
            raise RuntimeError(f"Tile {index} temporal metadata identity differs from commit")
        if any(not path.is_file() for path in metadata_paths):
            raise FileNotFoundError(f"Tile {index} temporal metadata is missing")
        metadata_hashes = commit.get("temporal_unit_hashes", [])
        if len(metadata_hashes) != len(metadata_paths):
            raise RuntimeError(f"Tile {index} temporal metadata hash count differs from commit")
        for path, expected_hash in zip(metadata_paths, metadata_hashes):
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"Tile {index} temporal metadata hash changed: {path}")
        referenced_metadata.update(metadata_paths)
    metadata_root = contained(
        Path(manifest["partial_report"]["configuration"]["artifact_metadata_root"]),
        shard_work,
        "Temporal metadata root",
    )
    actual_metadata = {path.resolve() for path in metadata_root.rglob("*") if path.is_file()}
    if actual_metadata != referenced_metadata:
        raise RuntimeError("Temporal metadata tree contains missing or unexpected files")
    return receipt, manifest


def aggregate_shards(
    *, video_id: str, shard_paths: list[Path], work_dir: Path,
    output_manifest: Path, output_target: Path, upscale_path: Path = DEFAULT_UPSCALE,
) -> Path:
    if len(shard_paths) != 2:
        raise ValueError("Exactly two complementary shard receipts are required")
    work_dir = work_dir.resolve()
    output_manifest = contained(output_manifest, work_dir, "Aggregate manifest")
    output_target = contained(output_target, work_dir, "Aggregate output target")
    upscale_path = upscale_path.resolve()
    require_approved_upscale(upscale_path)
    expected_layout = _expected_layout(upscale_path)
    if len(expected_layout) != 170:
        raise RuntimeError("Packaged production geometry is not the expected 170 tiles")
    validated = [
        validate_shard_receipt(path, video_id=video_id, aggregate_work_dir=work_dir,
                               expected_layout=expected_layout)
        for path in shard_paths
    ]
    receipts = [pair[0] for pair in validated]
    manifests = [pair[1] for pair in validated]
    comparable = (
        "input_identity", "anchor_identity", "parameters", "upscale_sha256",
        "model", "repository_commit", "processed_input_frames",
    )
    for key in comparable:
        if receipts[0].get(key) != receipts[1].get(key):
            raise ValueError(f"Shard identity mismatch for {key}")
    all_artifacts = [artifact for manifest in manifests for artifact in manifest["tile_artifacts"]]
    indices = [int(item["tile"]["index"]) for item in all_artifacts]
    if sorted(indices) != EXPECTED_INDICES or len(indices) != len(set(indices)):
        raise ValueError("Shards are not an exact, gap-free, duplicate-free 170-tile complement")
    artifacts = sorted(all_artifacts, key=lambda item: int(item["tile"]["index"]))
    base_report = dict(manifests[0]["partial_report"])
    base_report["status"] = "STAGED"
    base_report["stage_mode"] = "inference"
    base_report["tile_count"] = 170
    base_report["effective_input_frames_processed"] = 1952
    base_report["effective_output_frames_written"] = 1952
    base_report["configuration"] = {
        **base_report.get("configuration", {}),
        "tile_size": 256,
        "tile_overlap": 32,
        "sparse_ratio": 2.0,
        "local_range": 11,
        "base_seed": BASE_SEED,
        "precision": "torch.bfloat16 (original implementation)",
        "temporal_mode": "official_continuous_causal_kv_no_independent_chunks",
    }
    temporal = [path for artifact in artifacts for path in artifact["temporal_unit_metadata"]]
    base_report["temporal_unit_metadata"] = temporal
    aggregate = {
        "schema_version": 1,
        "model": "FlashVSR_v1.1",
        "state": "COMPLETE",
        "video_id": video_id,
        "input": manifests[0]["input"],
        "input_identity": receipts[0]["input_identity"],
        "anchor_identity": receipts[0]["anchor_identity"],
        "upscale_sha256": receipts[0]["upscale_sha256"],
        "output_target": str(output_target),
        "scratch_path": str(work_dir),
        "processed_input_frames": 1952,
        "padded_model_frames": manifests[0]["padded_model_frames"],
        "tile_count": 170,
        "tile_artifacts": artifacts,
        "temporal_unit_metadata": temporal,
        "encode_defaults": manifests[0]["encode_defaults"],
        "partial_report": base_report,
        "aggregation": {
            "schema": "a6000_flashvsr_full_layout/v1",
            "completion_scope": "FULL_SPATIAL_LAYOUT",
            "source_shard_receipts": [str(path.resolve()) for path in shard_paths],
            "global_tile_indices": EXPECTED_INDICES,
        },
    }
    write_json_atomic(output_manifest, aggregate)
    return output_manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-id", required=True, choices=["03", "04"])
    p.add_argument("--shard", action="append", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--output-manifest", required=True)
    p.add_argument("--output-target", required=True)
    p.add_argument("--upscale-path", default=str(DEFAULT_UPSCALE))
    return p


def main() -> int:
    args = parser().parse_args()
    result = aggregate_shards(
        video_id=args.video_id,
        shard_paths=[Path(path) for path in args.shard],
        work_dir=Path(args.work_dir),
        output_manifest=Path(args.output_manifest),
        output_target=Path(args.output_target),
        upscale_path=Path(args.upscale_path),
    )
    print(json.dumps({"status": "COMPLETE", "completion_scope": "FULL_SPATIAL_LAYOUT",
                      "manifest": str(result)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
