"""Select and freeze four sharp full-canvas references from bridge B00."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reference_memory import ReferenceArtifact, validate_reference_artifacts

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(
    "/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/"
    "longrange_bridge_013000_20261007/outputs/independent_start_0000/"
    "generated_65f_fp32.npy"
)
STRATA = ((0, 16), (16, 32), (32, 49), (49, 65))
ANCHOR = (270, 480, 810, 1440)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def exterior_sharpness(frame_chw: np.ndarray) -> float:
    rgb = np.moveaxis(np.clip(frame_chw, 0.0, 1.0), 0, -1)
    gray = cv2.cvtColor((rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
    y0, x0, y1, x1 = ANCHOR
    mask = np.ones(gray.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    values = laplacian[mask]
    return float(values.var(dtype=np.float64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reference_bank")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    frames = np.load(source, mmap_mode="r")
    if frames.shape != (65, 3, 1080, 1920) or frames.dtype != np.float32:
        raise ValueError(f"Expected B00 full-canvas FP32 [65,3,1080,1920], got {frames.shape} {frames.dtype}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    artifacts: list[ReferenceArtifact] = []
    for bank_index, (start, stop) in enumerate(STRATA):
        scores = [(frame, exterior_sharpness(frames[frame])) for frame in range(start, stop)]
        selected, score = max(scores, key=lambda item: (item[1], -item[0]))
        rgb = np.moveaxis(np.clip(frames[selected], 0.0, 1.0), 0, -1)
        rgb8 = (rgb * 255.0).round().astype(np.uint8)
        path = output / f"reference_{bank_index:02d}_frame_{selected:04d}.png"
        Image.fromarray(rgb8, mode="RGB").save(path, compress_level=1)
        artifact = ReferenceArtifact(str(path), digest(path), selected, True)
        artifacts.append(artifact)
        records.append({
            **artifact.__dict__, "stratum": [start, stop],
            "exterior_laplacian_variance": score,
        })
    validate_reference_artifacts(artifacts)
    manifest = {
        "schema": "hrlrvo_four_image_reference_memory_v1",
        "source": str(source),
        "source_sha256": digest(source),
        "source_shape": [65, 3, 1080, 1920],
        "selection": "maximum exterior Laplacian variance per fixed temporal stratum; earliest wins ties",
        "strata": [list(value) for value in STRATA],
        "anchor_yxyx_excluded_from_sharpness": list(ANCHOR),
        "image_conversion": "clip FP32 [0,1], round(value*255), RGB PNG",
        "reference_count": 4,
        "vae_policy": "body worker encodes each PNG independently as one frame",
        "stream": "frozen checkpoint clean-timestep sigma=0",
        "attention": "Global target Q reads immutable per-block reference K/V",
        "temporal_policy": "spatial coordinates retained; target time copied to reference K",
        "alpha": 0.1,
        "artifacts": records,
    }
    manifest_path = output / "reference_manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"Refusing to replace a different fixed manifest: {manifest_path}")
    manifest_path.write_text(encoded, encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "frames": [a.source_frame for a in artifacts]}))


if __name__ == "__main__":
    main()
