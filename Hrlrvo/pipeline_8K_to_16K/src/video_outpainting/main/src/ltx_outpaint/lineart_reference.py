"""CPU C0 reference extraction; no learned model and no random operations."""

import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from cert_pipeline.recurrence import DEFAULT_RECURRENCE


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def detailed_lines(frame):
    if frame.dtype != np.uint8 or frame.shape != (2160, 3840, 3):
        raise ValueError("Line-art extraction expects final uint8 RGB2160x3840")
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0.8, borderType=cv2.BORDER_REFLECT_101)
    edges = cv2.Canny(blur, 35, 100, apertureSize=3, L2gradient=True)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    for index in range(1, count):
        _, _, width, height, pixels = stats[index]
        keep[index] = pixels >= 12 and max(width, height) >= 12
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def direct_global_maps(lines):
    """Mean line density on the existing 4K C4 spatial footprints, no resize.

    Model boundary: 2160 RGB rows +8/+8 padding =2176 encoded rows.
    VAE32 times GlobalC4 gives128x128px per token, exactly17x30 cells.
    Padding outside the actual final RGB has zero observed line evidence.
    """
    if lines.shape != (2160, 3840) or lines.dtype != np.uint8:
        raise ValueError("Expected native 4K uint8 line image")
    binary = (lines > 0).astype(np.float32)
    padded = np.pad(binary, ((8, 8), (0, 0)), constant_values=0)
    density = padded.reshape(17, 128, 30, 128).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
    return density, np.clip(density / 0.05, 0, 1).astype(np.float32)


def reference_from_segment(directory, segment_path, source_identity):
    """Create after C0 commit, or verify/reload without changing canonical C0.

    Missing uncommitted derived artifacts can be reconstructed from the SHA-
    validated C0 native segment on resume. A committed corrupt reference fails.
    """
    directory, segment_path = Path(directory), Path(segment_path)
    metadata_path = directory / "lineart_metadata.json"
    contract = {k: v for k, v in DEFAULT_RECURRENCE.items() if k.startswith("lineart_")}
    names = {"source_frame_4k.png", "lineart_4k.png",
             "global_line_density.npy", "global_line_strength.npy"}
    # Re-derive from the independently validated native C0 even on resume.
    # File hashes in a mutable sidecar alone cannot establish source identity.
    segment = np.load(segment_path, mmap_mode="r", allow_pickle=False)
    if segment.dtype != np.uint8 or segment.shape[1:] != (2160, 3840, 3) or len(segment) < 1:
        raise ValueError("Expected committed C0 uint8 4K segment")
    frame4k = np.array(segment[0], copy=True)
    del segment
    lines = detailed_lines(frame4k)
    density, expected_strength = direct_global_maps(lines)
    png_arrays = {"source_frame_4k.png": frame4k, "lineart_4k.png": lines}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("source_identity") != source_identity or metadata.get("contract") != contract
                or set(metadata.get("files", {})) != names):
            raise RuntimeError("Line-art source/contract/file-set mismatch on resume")
        for name, record in metadata["files"].items():
            path = directory / name
            if not path.is_file() or _sha(path) != record["sha256"]:
                raise RuntimeError(f"Corrupt line-art artifact: {path}")
        if metadata.get("source_rgb_sha256") != hashlib.sha256(frame4k.tobytes()).hexdigest():
            raise RuntimeError("Line-art original C0 RGB digest mismatch")
        for name, expected in png_arrays.items():
            readback = cv2.imread(str(directory / name), cv2.IMREAD_UNCHANGED)
            if expected.ndim == 3 and readback is not None:
                readback = cv2.cvtColor(readback, cv2.COLOR_BGR2RGB)
            if not np.array_equal(readback, expected):
                raise RuntimeError(f"Line-art PNG is not derived from C0: {name}")
        strength = np.load(directory / "global_line_strength.npy", allow_pickle=False)
        if strength.shape != (17, 30) or strength.dtype != np.float32 or not np.isfinite(strength).all():
            raise RuntimeError("Invalid direct 4K line-art strength")
        if (not np.array_equal(strength, expected_strength)
                or not np.array_equal(np.load(directory / "global_line_density.npy", allow_pickle=False), density)):
            raise RuntimeError("Line-art direct 4K density mismatch")
        return torch.from_numpy(strength), metadata, _sha(metadata_path)

    strength = expected_strength
    for name, array in png_arrays.items():
        encoded = cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if array.ndim == 3 else array
        ok, content = cv2.imencode(".png", encoded, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError(f"PNG encode failed: {name}")
        roundtrip = cv2.imdecode(content, cv2.IMREAD_UNCHANGED)
        if not np.array_equal(roundtrip, encoded):
            raise RuntimeError(f"PNG roundtrip failed: {name}")
        _atomic(directory / name, content.tobytes())
    for name, array in (("global_line_density.npy", density), ("global_line_strength.npy", strength)):
        content = io.BytesIO()
        np.save(content, array, allow_pickle=False)
        _atomic(directory / name, content.getvalue())
    metadata = {
        "source_identity": source_identity,
        "source_segment": str(segment_path),
        "source_frame_index": 0,
        "source": "chunk00 final RGB frame 0",
        "source_resolution": [3840, 2160],
        "lineart_resolution": [3840, 2160],
        "source_rgb_sha256": hashlib.sha256(frame4k.tobytes()).hexdigest(),
        "source_shape": list(frame4k.shape),
        "reference_shape": list(frame4k.shape),
        "contract": contract,
        "opencv_version": cv2.__version__,
        "source_global_hw": [17, 30],
        "runtime_global_hw": [17, 30],
        "global_grid": [30, 17],
        "density_normalizer": 0.05,
        "lambda_schedule": contract["lineart_bias_schedule"],
        "density_method": "direct mean over 128x128px footprints; zero line evidence in 8px top/bottom VAE padding; no resize",
        "nonzero_source_cells": int(np.count_nonzero(strength)),
        "files": {name: {"sha256": _sha(directory / name), "size_bytes": (directory / name).stat().st_size}
                  for name in sorted(names)},
    }
    _atomic(metadata_path, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    return torch.from_numpy(strength), metadata, _sha(metadata_path)
