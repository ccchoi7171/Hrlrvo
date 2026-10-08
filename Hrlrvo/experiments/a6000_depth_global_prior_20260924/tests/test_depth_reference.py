import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_depth_reference.py"
SPEC = importlib.util.spec_from_file_location("prepare_depth_reference", SCRIPT)
depth_reference = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(depth_reference)


def test_normalize_inverse_depth_matches_percentile_clip():
    raw = np.linspace(-3.0, 7.0, 1080 * 1920, dtype=np.float32).reshape(1080, 1920)
    normalized, low, high = depth_reference.normalize_inverse_depth(raw)
    expected_low, expected_high = np.percentile(raw, [0.5, 99.5]).tolist()
    expected = np.clip((raw - expected_low) / (expected_high - expected_low), 0, 1).astype(np.float32)
    assert low == expected_low
    assert high == expected_high
    np.testing.assert_array_equal(normalized, expected)
    assert normalized.dtype == np.float32
    assert normalized[0, 0] == 0.0
    assert normalized[-1, -1] == 1.0


@pytest.mark.parametrize(
    "value, error",
    [
        (np.zeros((12, 12), dtype=np.float32), ValueError),
        (np.zeros((1080, 1920), dtype=np.uint8), TypeError),
        (np.zeros((1080, 1920), dtype=np.float32), ValueError),
    ],
)
def test_normalize_inverse_depth_rejects_invalid_inputs(value, error):
    with pytest.raises(error):
        depth_reference.normalize_inverse_depth(value)


def test_normalize_inverse_depth_rejects_nonfinite():
    value = np.linspace(0, 1, 1080 * 1920, dtype=np.float32).reshape(1080, 1920)
    value[20, 30] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        depth_reference.normalize_inverse_depth(value)


def test_pooling_is_exact_two_stage_area_mean_with_zero_padding():
    value = np.arange(1080 * 1920, dtype=np.float32).reshape(1080, 1920)
    value /= value.max()
    actual = depth_reference.pool_global_depth_strength(value)
    padded = np.pad(value, ((36, 36), (0, 0)))
    coarse = (
        padded.reshape(36, 32, 60, 32)
        .mean(axis=(1, 3), dtype=np.float64)
        .astype(np.float32)
    )
    expected = (
        coarse.reshape(9, 4, 15, 4)
        .mean(axis=(1, 3), dtype=np.float64)
        .astype(np.float32)
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (9, 15)
    assert actual.dtype == np.float32


def test_pooling_rejects_wrong_dtype_range_and_nonfinite():
    with pytest.raises(TypeError):
        depth_reference.pool_global_depth_strength(np.zeros((1080, 1920), dtype=np.float64))
    bad_range = np.zeros((1080, 1920), dtype=np.float32)
    bad_range[0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        depth_reference.pool_global_depth_strength(bad_range)
    bad_range[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        depth_reference.pool_global_depth_strength(bad_range)


def _make_c0(tmp_path: Path):
    arm = tmp_path / "arm"
    chunks = arm / "chunks"
    chunks.mkdir(parents=True)
    shape = (2, 3, 4, 3)
    pixels = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    segment = chunks / "segment_000000.npy"
    np.save(segment, pixels, allow_pickle=False)
    digest = hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
    record = {
        "chunk_index": 0,
        "absolute_start_frame": 0,
        "chunk_seed": depth_reference.CHUNK_SEED,
        "output_appended_count": 65,
        "new_output_frame_range": [0, 65],
        "segment_file": segment.name,
        "segment_dtype": "uint8",
        "segment_shape_fhwc": list(shape),
        "segment_byte_size": segment.stat().st_size,
        "segment_pixel_sha256": digest,
    }
    record_path = chunks / "chunk_000000.json"
    record_path.write_text(json.dumps(record))
    return arm, shape, pixels, record_path


def test_load_committed_c0_authenticates_record_file_and_pixels(tmp_path):
    arm, shape, pixels, _ = _make_c0(tmp_path)
    frame, source = depth_reference.load_committed_c0(arm, expected_shape=shape)
    np.testing.assert_array_equal(frame, pixels[0])
    assert source["chunk_seed"] == depth_reference.CHUNK_SEED
    assert source["segment_pixel_sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()
    assert source["frame_pixel_sha256"] == hashlib.sha256(pixels[0].tobytes()).hexdigest()


def test_load_committed_c0_fails_closed_on_digest_or_identity(tmp_path):
    arm, shape, _, record_path = _make_c0(tmp_path)
    record = json.loads(record_path.read_text())
    record["segment_pixel_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="pixel digest"):
        depth_reference.load_committed_c0(arm, expected_shape=shape)
    record["segment_pixel_sha256"] = "1" * 64
    record["segment_file"] = "../segment_000000.npy"
    record_path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="segment_file"):
        depth_reference.load_committed_c0(arm, expected_shape=shape)


def test_validate_model_manifest_checks_pinned_identity_and_hashes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    asset = model / "model.safetensors"
    asset.write_bytes(b"locally pinned weights")
    manifest = {
        "model_id": depth_reference.MODEL_ID,
        "revision": depth_reference.MODEL_REVISION,
        "sha256": {asset.name: hashlib.sha256(asset.read_bytes()).hexdigest()},
    }
    (model / "asset_manifest.json").write_text(json.dumps(manifest))
    actual = depth_reference.validate_model_manifest(model)
    assert actual["asset_sha256"][asset.name] == manifest["sha256"][asset.name]
    asset.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        depth_reference.validate_model_manifest(model)
