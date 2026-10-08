"""Lossless decoded-RGB prefix artifacts for optional inference modes.

The MP4 is only a human preview. Continuation always loads the FP32 tensor,
whose file digest is committed in the metadata after both outputs are written.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ltx_outpaint.streaming_video import (
    _atomic_json,
    _fsync_directory,
    _load_json,
    _write_uint8_segment,
    encode_segments,
    file_sha256,
)


def _validate_prefix(prefix: torch.Tensor, *, expected_shape: tuple[int, ...] | None = None) -> None:
    if prefix.device.type != "cpu" or prefix.dtype != torch.float32:
        raise TypeError("Prefix must be a CPU FP32 decoded RGB tensor")
    if prefix.ndim != 4 or prefix.shape[0] < 1 or prefix.shape[1] != 3:
        raise ValueError("Prefix must have shape [frames, 3, height, width]")
    if prefix.shape[2] < 1 or prefix.shape[3] < 1:
        raise ValueError("Prefix spatial dimensions must be positive")
    if expected_shape is not None and tuple(prefix.shape) != expected_shape:
        raise ValueError(f"Prefix shape {tuple(prefix.shape)} != expected {expected_shape}")
    if not bool(torch.isfinite(prefix).all().item()):
        raise ValueError("Prefix contains NaN or Inf")
    if prefix.min().item() < 0.0 or prefix.max().item() > 1.0:
        raise ValueError("Prefix values must be in decoded RGB [0,1] range")


def write_prefix_candidate(
    prefix: torch.Tensor,
    output_dir: Path,
    *,
    seed: int,
    fps: float,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Atomically commit an exact prefix, encoder-equivalent preview and metadata.

    ``metadata`` carries generation lineage (input/checkpoint/config paths,
    source range, timestamp and memory flags). The three returned paths name
    the independently readable artifacts. Metadata is the completion marker.
    """
    _validate_prefix(prefix)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < fps < float("inf"):
        raise ValueError("fps must be finite and positive")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = f"seed_{seed}"
    prefix_path = output_dir / f"{basename}_prefix.pt"
    preview_path = output_dir / f"{basename}_preview.mp4"
    meta_path = output_dir / f"{basename}_meta.json"
    segment_path = output_dir / f".{basename}_preview_segment.npy"
    temporary_prefix = output_dir / f".{prefix_path.name}.partial"
    lock_path = output_dir / f".{basename}.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"Prefix candidate is already being written: {lock_path}") from error
    os.close(lock_fd)
    try:
        if meta_path.exists():
            raise FileExistsError(f"Prefix candidate is already complete: {meta_path}")
        # Orphan data from an interrupted attempt has no committed metadata and
        # can safely be replaced by this same candidate. A contiguous clone also
        # avoids serializing a larger backing storage held by a tensor view.
        saved_prefix = prefix.contiguous().clone()
        try:
            with temporary_prefix.open("wb") as handle:
                torch.save(saved_prefix, handle)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            del saved_prefix
        os.replace(temporary_prefix, prefix_path)
        _fsync_directory(output_dir)

        _write_uint8_segment(segment_path, prefix)
        try:
            encode_segments(
                [{"segment_file": segment_path.name}],
                output_dir,
                preview_path,
                fps=fps,
                width=int(prefix.shape[3]),
                height=int(prefix.shape[2]),
                expected_frames=int(prefix.shape[0]),
            )
        finally:
            segment_path.unlink(missing_ok=True)

        generated = dict(metadata)
        generated.update(
            {
                "schema_version": 1,
                "seed": seed,
                "fps": fps,
                "frame_count": int(prefix.shape[0]),
                "resolution": [int(prefix.shape[3]), int(prefix.shape[2])],
                "tensor_file": prefix_path.name,
                "tensor_shape": list(prefix.shape),
                "tensor_dtype": "float32",
                "tensor_sha256": file_sha256(prefix_path),
                "tensor_byte_size": prefix_path.stat().st_size,
                "preview_file": preview_path.name,
            }
        )
        _atomic_json(meta_path, generated)
        return prefix_path, preview_path, meta_path, generated
    finally:
        temporary_prefix.unlink(missing_ok=True)
        segment_path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def load_prefix_artifact(
    path: Path,
    *,
    expected_frames: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Verify a committed FP32 prefix and return it unchanged for conditioning."""
    path = Path(path)
    suffix = "_prefix.pt"
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected a seed_<seed>{suffix} artifact: {path}")
    meta_path = path.with_name(path.name[: -len(suffix)] + "_meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Prefix completion metadata does not exist: {meta_path}")
    metadata = _load_json(meta_path)
    expected_shape = (expected_frames, 3, height, width)
    if (
        metadata.get("schema_version") != 1
        or metadata.get("tensor_file") != path.name
        or metadata.get("tensor_shape") != list(expected_shape)
        or metadata.get("tensor_dtype") != "float32"
        or metadata.get("frame_count") != expected_frames
        or metadata.get("resolution") != [width, height]
    ):
        raise ValueError("Prefix metadata does not match the requested tensor geometry")
    if path.stat().st_size != metadata.get("tensor_byte_size"):
        raise ValueError("Prefix tensor file size does not match metadata")
    if file_sha256(path) != metadata.get("tensor_sha256"):
        raise ValueError("Prefix tensor SHA-256 does not match metadata")
    prefix = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(prefix, torch.Tensor):
        raise TypeError("Prefix artifact does not contain a tensor")
    _validate_prefix(prefix, expected_shape=expected_shape)
    return prefix, metadata
