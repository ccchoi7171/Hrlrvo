"""Atomic CPU causal-latent carry, distinct from native RGB-prefix/RNG state."""

import hashlib
import os
from pathlib import Path

import torch


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_carry(root, index, tensor):
    root = Path(root)
    if tensor.device.type != "cpu" or tensor.ndim != 5 or tensor.shape[2] != 3:
        raise ValueError("Hybrid carry must be CPU[B,C,3,H,W]")
    if not torch.isfinite(tensor).all():
        raise ValueError("Nonfinite hybrid carry")
    path = root / f"causal_latent_carry_{index:06d}.pt"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(tensor, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _hash(temporary)
        os.replace(temporary, path)
        fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {"file": path.name, "sha256": digest, "shape": list(tensor.shape),
            "dtype": str(tensor.dtype), "chunk_index": index,
            "semantics": "last3 final-clean unpatchified slices; next first slice reencoded from RGB; current anchor overwrites"}


def load_carry(root, record, expected_shape=None, expected_dtype=None):
    item = record.get("causal_latent_carry")
    index = int(record["chunk_index"])
    if not isinstance(item, dict) or item.get("file") != f"causal_latent_carry_{index:06d}.pt":
        raise RuntimeError(f"Missing/invalid causal carry for chunk {index}")
    path = Path(root) / item["file"]
    if not path.is_file() or _hash(path) != item.get("sha256"):
        raise RuntimeError(f"Causal carry digest mismatch: {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 5
            or tensor.shape[2] != 3 or list(tensor.shape) != item.get("shape")
            or str(tensor.dtype) != item.get("dtype") or item.get("chunk_index") != index
            or not torch.isfinite(tensor).all()):
        raise RuntimeError("Invalid causal carry tensor metadata")
    if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
        raise RuntimeError("Causal carry geometry mismatch")
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise RuntimeError("Causal carry dtype mismatch")
    return tensor
