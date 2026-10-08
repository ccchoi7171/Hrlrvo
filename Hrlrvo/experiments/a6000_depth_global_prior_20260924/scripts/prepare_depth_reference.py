"""Prepare the pinned D5 inverse-depth reference from the A6000 baseline C0.

The model is loaded locally and runs on CPU only.  The source image is always
frame zero of the lossless, committed baseline chunk-zero segment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
DEFAULT_BASELINE_ARM = Path(
    "/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/arms/01_baseline"
)
DEFAULT_MODEL_PATH = ROOT / "models/D5_v2_base_518"
MODEL_REVISION = "b1958afc87fb45a9e3746cb387596094de553ed8"
MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
SEGMENT_SHAPE = (65, 1080, 1920, 3)
CHUNK_SEED = 71634910
PROCESSOR_SIZE = 518

ARTIFACT_PATHS = (
    Path("c0_frame000.png"),
    Path("depth_raw.npy"),
    Path("depth_inverse_normalized.npy"),
    Path("depth_gray.png"),
    Path("depth_gray_16bit.png"),
    Path("depth_color.png"),
    Path("global_depth_strength.npy"),
    Path("depth_reference.json"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(array: np.ndarray) -> str:
    """Hash logical C-order pixels, matching SegmentedOutputStore."""
    digest = hashlib.sha256()
    for frame in array:
        digest.update(np.asarray(frame).tobytes(order="C"))
    return digest.hexdigest()


def normalize_inverse_depth(
    depth: np.ndarray, *, percentiles: tuple[float, float] = (0.5, 99.5)
) -> tuple[np.ndarray, float, float]:
    """Percentile-clip relative inverse depth to float32 [0, 1]."""
    value = np.asarray(depth)
    if value.shape != (1080, 1920):
        raise ValueError(f"Depth must have shape (1080, 1920), got {value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("Depth must be floating point")
    if not np.isfinite(value).all():
        raise ValueError("Depth contains NaN or Inf")
    low_percentile, high_percentile = percentiles
    if not (0.0 <= low_percentile < high_percentile <= 100.0):
        raise ValueError("Invalid percentile interval")
    low, high = np.percentile(value, percentiles).tolist()
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Depth has no usable percentile range")
    normalized = np.clip((value.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    return normalized.astype(np.float32, copy=False), float(low), float(high)


def pool_global_depth_strength(normalized_inverse_depth: np.ndarray) -> np.ndarray:
    """Pad to 1152x1920, then exact area means 32x32 and 4x4 -> 9x15."""
    value = np.asarray(normalized_inverse_depth)
    if value.shape != (1080, 1920):
        raise ValueError(f"Normalized depth must be 1080x1920, got {value.shape}")
    if value.dtype != np.float32:
        raise TypeError("Normalized depth must be float32")
    if not np.isfinite(value).all():
        raise ValueError("Normalized depth contains NaN or Inf")
    if value.min() < 0.0 or value.max() > 1.0:
        raise ValueError("Normalized depth must be in [0, 1]")
    padded = np.pad(value, ((36, 36), (0, 0)), mode="constant", constant_values=0)
    coarse = (
        padded.reshape(36, 32, 60, 32)
        .mean(axis=(1, 3), dtype=np.float64)
        .astype(np.float32)
    )
    pooled = (
        coarse.reshape(9, 4, 15, 4)
        .mean(axis=(1, 3), dtype=np.float64)
        .astype(np.float32)
    )
    if pooled.shape != (9, 15) or pooled.dtype != np.float32 or not np.isfinite(pooled).all():
        raise RuntimeError("Invalid pooled depth reference")
    return pooled


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    return value.lower()


def validate_model_manifest(model_path: Path) -> dict[str, Any]:
    """Validate every file pinned by the local model asset manifest."""
    root = model_path.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest_path = root / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("revision") != MODEL_REVISION:
        raise RuntimeError("Model manifest revision is not the pinned D5 revision")
    if manifest.get("model_id") != MODEL_ID:
        raise RuntimeError("Model manifest model_id is not the pinned D5 model")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("Model manifest requires a non-empty sha256 map")
    validated: dict[str, str] = {}
    for relative_name, expected in sorted(hashes.items()):
        if not isinstance(relative_name, str) or not relative_name:
            raise ValueError("Model manifest contains an invalid path")
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or relative.name == "asset_manifest.json":
            raise ValueError(f"Unsafe model manifest path: {relative_name!r}")
        candidate = (root / relative).resolve(strict=True)
        if root not in candidate.parents or not candidate.is_file():
            raise ValueError(f"Model asset escapes model directory: {relative_name!r}")
        expected_digest = _require_sha256(expected, label=f"sha256[{relative_name!r}]")
        actual_digest = file_sha256(candidate)
        if actual_digest != expected_digest:
            raise RuntimeError(f"Model asset digest mismatch: {relative_name}")
        validated[relative.as_posix()] = actual_digest
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "asset_sha256": validated,
    }


def load_committed_c0(
    baseline_arm: Path, *, expected_shape: tuple[int, int, int, int] = SEGMENT_SHAPE
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and authenticate frame zero of the committed baseline segment."""
    arm = baseline_arm.resolve(strict=True)
    chunks = arm / "chunks"
    record_path = chunks / "chunk_000000.json"
    record = json.loads(record_path.read_text())
    required = {
        "chunk_index": 0,
        "absolute_start_frame": 0,
        "chunk_seed": CHUNK_SEED,
        "output_appended_count": 65,
        "new_output_frame_range": [0, 65],
        "segment_file": "segment_000000.npy",
        "segment_dtype": "uint8",
        "segment_shape_fhwc": list(expected_shape),
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise RuntimeError(f"C0 record {key!r} mismatch: {record.get(key)!r} != {expected!r}")
    expected_pixels = _require_sha256(
        record.get("segment_pixel_sha256"), label="segment_pixel_sha256"
    )
    segment_path = (chunks / record["segment_file"]).resolve(strict=True)
    if segment_path.parent != chunks.resolve() or segment_path.name != "segment_000000.npy":
        raise RuntimeError("C0 segment identity/path mismatch")
    if "segment_byte_size" in record and segment_path.stat().st_size != record["segment_byte_size"]:
        raise RuntimeError("C0 segment byte size differs from committed record")
    segment = np.load(segment_path, mmap_mode="r", allow_pickle=False)
    if segment.shape != expected_shape or segment.dtype != np.uint8:
        raise RuntimeError("C0 segment array shape/dtype mismatch")
    actual_pixels = pixel_sha256(segment)
    if actual_pixels != expected_pixels:
        raise RuntimeError("C0 segment pixel digest differs from committed record")
    frame = np.asarray(segment[0]).copy()
    source = {
        "baseline_arm": str(arm),
        "record_path": str(record_path),
        "record_sha256": file_sha256(record_path),
        "segment_path": str(segment_path),
        "segment_file_sha256": file_sha256(segment_path),
        "segment_pixel_sha256": actual_pixels,
        "segment_shape_fhwc": list(segment.shape),
        "segment_dtype": str(segment.dtype),
        "frame_index": 0,
        "frame_pixel_sha256": hashlib.sha256(frame.tobytes(order="C")).hexdigest(),
        "chunk_seed": record["chunk_seed"],
    }
    return frame, source


def _save_outputs(stage: Path, frame: np.ndarray, depth: np.ndarray, normalized: np.ndarray) -> None:
    stage.mkdir(parents=True)
    Image.fromarray(frame, mode="RGB").save(stage / "c0_frame000.png")
    np.save(stage / "depth_raw.npy", depth.astype(np.float32, copy=False), allow_pickle=False)
    np.save(stage / "depth_inverse_normalized.npy", normalized, allow_pickle=False)
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(stage / "depth_gray.png")
    Image.fromarray(np.rint(normalized * 65535.0).astype(np.uint16), mode="I;16").save(
        stage / "depth_gray_16bit.png"
    )
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - production environment dependency
        raise RuntimeError("OpenCV is required for the pinned color visualization") from error
    cv2.setNumThreads(2)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    if not cv2.imwrite(str(stage / "depth_color.png"), color):
        raise RuntimeError("Could not save depth_color.png")
    np.save(
        stage / "global_depth_strength.npy",
        pool_global_depth_strength(normalized),
        allow_pickle=False,
    )


def _assert_no_conflicts() -> None:
    conflicts = [str(REFERENCE / relative) for relative in ARTIFACT_PATHS if (REFERENCE / relative).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite depth reference artifacts: " + ", ".join(conflicts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-arm", type=Path, default=DEFAULT_BASELINE_ARM)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' exactly; depth extraction is CPU-only")
    _assert_no_conflicts()
    model_provenance = validate_model_manifest(args.model_path)
    frame, source = load_committed_c0(args.baseline_arm)

    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    image = Image.fromarray(frame, mode="RGB")
    load_started = time.monotonic()
    processor = AutoImageProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device="cpu", dtype=torch.float32).eval()
    load_seconds = time.monotonic() - load_started
    inputs = processor(
        images=image,
        size={"height": PROCESSOR_SIZE, "width": PROCESSOR_SIZE},
        return_tensors="pt",
    )
    if list(inputs["pixel_values"].shape) != [1, 3, 518, 924]:
        raise RuntimeError(f"Unexpected processor input shape: {list(inputs['pixel_values'].shape)}")
    if any(value.device.type != "cpu" for value in inputs.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("Depth inputs must stay on CPU")
    started = time.monotonic()
    with torch.inference_mode():
        outputs = model(**inputs)
        result = processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        depth = result["predicted_depth"].float().cpu().numpy().astype(np.float32, copy=False)
    inference_seconds = time.monotonic() - started
    normalized, low, high = normalize_inverse_depth(depth)

    REFERENCE.mkdir(parents=True, exist_ok=True)
    stage = REFERENCE / f".prepare_depth_reference.{os.getpid()}"
    if stage.exists():
        raise FileExistsError(stage)
    try:
        _save_outputs(stage, frame, depth, normalized)
        artifact_files = [
            stage / "c0_frame000.png",
            stage / "depth_raw.npy",
            stage / "depth_inverse_normalized.npy",
            stage / "depth_gray.png",
            stage / "depth_gray_16bit.png",
            stage / "depth_color.png",
            stage / "global_depth_strength.npy",
        ]
        artifacts = {
            path.relative_to(stage).as_posix(): {
                "sha256": file_sha256(path), "size_bytes": path.stat().st_size
            }
            for path in artifact_files
        }
        receipt = {
            "schema_version": 1,
            "source": source,
            "model": {
                **model_provenance,
                "model_class": type(model).__name__,
                "processor_class": type(processor).__name__,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "device": "cpu",
                "dtype": "float32",
                "local_files_only": True,
                "trust_remote_code": False,
            },
            "numerics": {
                "prediction_kind": "relative_inverse_depth_white_near",
                "requested_processor_size": PROCESSOR_SIZE,
                "actual_input_shape": list(inputs["pixel_values"].shape),
                "raw_network_output_shape": list(outputs.predicted_depth.shape),
                "postprocessed_shape": list(depth.shape),
                "percentile_clip": [0.5, 99.5],
                "percentile_values": [low, high],
                "normalization": "clip((inverse_depth-p0.5)/(p99.5-p0.5),0,1)",
                "pooling": "zero-pad top/bottom 36; mean 32x32 to 36x60; mean 4x4 to 9x15",
                "pooling_accumulation_dtype": "float64_then_cast_float32_at_each_reduction",
                "pooling_divisor": None,
                "gamma": None,
                "inversion": False,
                "raw_min_max": [float(depth.min()), float(depth.max())],
            },
            "timing_seconds": {"model_load": load_seconds, "inference_postprocess": inference_seconds},
            "artifacts": artifacts,
        }
        (stage / "depth_reference.json").write_text(json.dumps(receipt, indent=2) + "\n")
        _assert_no_conflicts()
        for relative in ARTIFACT_PATHS:
            destination = REFERENCE / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / relative, destination)
        print(json.dumps({"event": "complete", "receipt": str(REFERENCE / "depth_reference.json")}))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
