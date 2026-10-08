"""Create read-only original/clean Anchor comparison contacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = Path(
    "/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/"
    "autonomous_013000_20261006/inputs/longrange_960x540_1025f.mkv"
)
CLEAN = ROOT / "experiment_2/clean_anchor_1025f.mkv"
MASKS = ROOT / "experiment_2/combined_masks_uint8.npy"
FRAMES = (0, 73, 120, 250, 480, 600, 800, 960, 1024)
CAMERA_FRAME = 120


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def decode_selected(path: Path) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    selected: dict[int, np.ndarray] = {}
    wanted = set(FRAMES)
    index = 0
    while wanted:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            if bgr.shape != (540, 960, 3) or bgr.dtype != np.uint8:
                raise RuntimeError(f"Unexpected frame at {index}: {bgr.shape} {bgr.dtype}")
            selected[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            wanted.remove(index)
        index += 1
    capture.release()
    if wanted:
        raise RuntimeError(f"Missing frames in {path}: {sorted(wanted)}")
    return selected


def labeled(image: np.ndarray, text: str, width: int = 200) -> Image.Image:
    height = round(width * image.shape[0] / image.shape[1])
    resized = Image.fromarray(image, "RGB").resize((width, height), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (width, height + 25), "white")
    cell.paste(resized, (0, 25))
    ImageDraw.Draw(cell).text((5, 5), text, fill="black", font=ImageFont.load_default())
    return cell


def contact(rows: list[tuple[str, list[np.ndarray]]], frames: tuple[int, ...], path: Path, width: int) -> None:
    cells = [[labeled(image, f"{name} F{frame}", width) for image, frame in zip(images, frames, strict=True)] for name, images in rows]
    canvas = Image.new("RGB", (sum(cell.width for cell in cells[0]), sum(row[0].height for row in cells)), "white")
    y = 0
    for row in cells:
        x = 0
        for cell in row:
            canvas.paste(cell, (x, y))
            x += cell.width
        y += row[0].height
    canvas.save(path, quality=95)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original = decode_selected(SOURCE)
    clean = decode_selected(CLEAN)
    masks = np.load(MASKS, mmap_mode="r")
    if masks.shape != (1025, 540, 960) or masks.dtype != np.uint8:
        raise RuntimeError("Combined cleanup mask contract changed")

    differences = {
        frame: np.abs(original[frame].astype(np.int16) - clean[frame].astype(np.int16)).astype(np.uint8)
        for frame in FRAMES
    }
    heat = {frame: np.clip(differences[frame].astype(np.uint16) * 4, 0, 255).astype(np.uint8) for frame in FRAMES}
    rows = [
        ("ORIGINAL", [original[frame] for frame in FRAMES]),
        ("CLEAN", [clean[frame] for frame in FRAMES]),
        ("ABS_DIFF_X4", [heat[frame] for frame in FRAMES]),
    ]
    contact(rows, FRAMES, OUT / "original_vs_clean_all_frames.jpg", width=200)
    contact([(name, images[:5]) for name, images in rows], FRAMES[:5], OUT / "original_vs_clean_early_camera.jpg", width=320)
    contact([(name, images[5:]) for name, images in rows], FRAMES[5:], OUT / "original_vs_clean_late.jpg", width=380)

    camera_crops = []
    # Broad upper band exposes the observed camera/jib cleanup without assuming a semantic GT mask.
    for name, array in (("ORIGINAL", original[CAMERA_FRAME]), ("CLEAN", clean[CAMERA_FRAME]), ("ABS_DIFF_X4", heat[CAMERA_FRAME])):
        crop = array[:270]
        camera_crops.append((name, [crop]))
    contact(camera_crops, (CAMERA_FRAME,), OUT / "camera_frame120_upper_crop.jpg", width=960)

    records = []
    for frame in FRAMES:
        diff = differences[frame]
        support = masks[frame] > 0
        outside = ~support
        records.append(
            {
                "frame": frame,
                "cleanup_support_pixels": int(support.sum()),
                "cleanup_support_fraction": float(support.mean()),
                "mean_abs_rgb_difference_all": float(diff.mean()),
                "mean_abs_rgb_difference_inside_support": float(diff[support].mean()) if support.any() else None,
                "mean_abs_rgb_difference_outside_support": float(diff[outside].mean()) if outside.any() else None,
                "original_mean_rgb": [float(value) for value in original[frame].mean(axis=(0, 1))],
                "clean_mean_rgb": [float(value) for value in clean[frame].mean(axis=(0, 1))],
            }
        )
    payload = {
        "status": "COMPLETE_READ_ONLY_DIAGNOSTIC",
        "frames": list(FRAMES),
        "camera_inspection_frame": CAMERA_FRAME,
        "camera_frame_basis": "F120 has elevated combined cleanup support near the known F96-F128 upper-equipment interval; this is a diagnostic selection, not ground truth.",
        "inputs": {
            "original": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
            "clean": {"path": str(CLEAN), "sha256": sha256(CLEAN)},
            "combined_cleanup_mask": {"path": str(MASKS), "sha256": sha256(MASKS)},
        },
        "metrics": records,
        "artifacts": {},
        "limitations": [
            "The cleanup support combines detector and equipment masks; it is not semantic ground truth.",
            "Absolute RGB differences include donor fill, registration interpolation, and framewise color fitting.",
            "Sparse contacts cannot establish temporal stability or prove complete foreground removal.",
        ],
    }
    for artifact in sorted(OUT.glob("*.jpg")):
        payload["artifacts"][artifact.name] = sha256(artifact)
    (OUT / "report.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
