from __future__ import annotations

import hashlib

import numpy as np
import pytest
from assemble import raw_array_digest, verify_raw_artifact


def raw_sha(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def test_raw_artifact_accepts_exact_tensor_bytes(tmp_path):
    expected = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "artifact.npy"
    np.save(path, expected, allow_pickle=False)

    verify_raw_artifact(path, raw_sha(expected), expected.shape, "fixture")
    mapped = np.load(path, mmap_mode="r")
    assert raw_array_digest(mapped) == raw_sha(expected)


def test_raw_artifact_rejects_corruption_against_status_hash(tmp_path):
    original = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected_status_sha = raw_sha(original)
    corrupted = original.copy()
    corrupted[1, 2, 3] += 1.0
    path = tmp_path / "artifact.npy"
    np.save(path, corrupted, allow_pickle=False)

    with pytest.raises(RuntimeError, match="raw tensor SHA-256 mismatch"):
        verify_raw_artifact(
            path,
            expected_status_sha,
            original.shape,
            "fixture",
        )
