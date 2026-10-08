#!/usr/bin/env python3
"""Prepare a deterministic, CPU-only C0 frame line-art prior for Global attention.

The source is the lossless uint8 C0 segment.  Edge extraction is deliberately
low-resolution and sparse so that stage geometry is favored over fine texture.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_global_prior_20260923")
SOURCE = Path("/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923/chunks/strength_1p0/segment_000000.npy")
REF = ROOT / "reference"
REPORT = ROOT / "report" / "reference_extraction.json"

# Fixed extraction parameters.  All operations are deterministic CPU OpenCV calls.
WORK_WIDTH = 480
WORK_HEIGHT = 270
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 1.4
CANNY_LOW = 40
CANNY_HIGH = 110
MIN_COMPONENT_PIXELS = 10
MIN_COMPONENT_SPAN = 12
DENSITY_NORMALIZER = 0.05
PAD_TOP = 36
PAD_BOTTOM = 36


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"failed to write {path}")


def filter_short_components(edges: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    kept = 0
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        # Retain a component only when it has enough support and spatial extent.
        if area >= MIN_COMPONENT_PIXELS and max(w, h) >= MIN_COMPONENT_SPAN:
            keep[label] = True
            kept += 1
    result = np.where(keep[labels], 255, 0).astype(np.uint8)
    return result, {"components_total": count - 1, "components_kept": kept}


def main() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    segment = np.load(SOURCE, mmap_mode="r")
    if segment.dtype != np.uint8 or tuple(segment.shape[1:]) != (1080, 1920, 3):
        raise ValueError(f"unexpected segment: shape={segment.shape}, dtype={segment.dtype}")
    frame = np.array(segment[0], dtype=np.uint8, copy=True)
    frame_path = REF / "c0_frame000.png"
    write_rgb(frame_path, frame)

    # Confirm the saved PNG is an exact RGB round-trip of the uint8 source frame.
    decoded = cv2.cvtColor(cv2.imread(str(frame_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if not np.array_equal(decoded, frame):
        raise RuntimeError("c0_frame000.png did not round-trip exactly")

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    work = cv2.resize(gray, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA)
    work = cv2.GaussianBlur(work, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA, borderType=cv2.BORDER_REFLECT_101)
    raw_edges = cv2.Canny(work, CANNY_LOW, CANNY_HIGH, apertureSize=3, L2gradient=True)
    clean_edges, component_stats = filter_short_components(raw_edges)

    # Nearest-neighbor preserves the binary sparse map and avoids invented gray edges.
    full_edges_u8 = cv2.resize(clean_edges, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    full_edges = (full_edges_u8 > 0).astype(np.float32)
    line_png = REF / "c0_frame000_lineart.png"
    cv2.imwrite(str(line_png), full_edges_u8, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    np.save(REF / "c0_frame000_lineart.npy", full_edges, allow_pickle=False)

    overlay = frame.copy()
    mask = full_edges.astype(bool)
    # Green overlay keeps both dark and bright source regions legible.
    overlay[mask] = np.rint(0.25 * overlay[mask] + 0.75 * np.array([0, 255, 64])).astype(np.uint8)
    write_rgb(REF / "c0_frame000_lineart_overlay.png", overlay)

    # Visible 1920x1080 -> encoded 1920x1152, with literal zero padding only.
    padded = np.pad(full_edges, ((PAD_TOP, PAD_BOTTOM), (0, 0)), mode="constant", constant_values=0)
    if padded.shape != (1152, 1920) or padded[:PAD_TOP].any() or padded[-PAD_BOTTOM:].any():
        raise RuntimeError("zero-pad invariant failed")

    # Exact area means: pixels -> latent 36x60 (32x32) -> Global 9x15 (4x4 latent).
    latent_density = padded.reshape(36, 32, 60, 32).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
    global_density = latent_density.reshape(9, 4, 15, 4).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
    global_strength = np.clip(global_density / DENSITY_NORMALIZER, 0.0, 1.0).astype(np.float32)
    np.save(REF / "latent_line_density.npy", latent_density, allow_pickle=False)
    np.save(REF / "global_line_density.npy", global_density, allow_pickle=False)
    np.save(REF / "global_line_strength.npy", global_strength, allow_pickle=False)

    # Color heatmap with fixed [0,1] range, enlarged using nearest neighbor.
    heat_u8 = np.rint(global_strength * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_VIRIDIS)
    heat = cv2.resize(heat, (15 * 64, 9 * 64), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(REF / "global_line_strength_heatmap.png"), heat, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    report = {
        "source": {
            "path": str(SOURCE),
            "sha256": sha256(SOURCE),
            "array_shape": list(segment.shape),
            "array_dtype": str(segment.dtype),
            "selected_frame_index": 0,
            "saved_rgb_path": str(frame_path),
            "saved_rgb_sha256": sha256(frame_path),
            "png_exact_rgb_roundtrip": True,
        },
        "extractor": {
            "name": "OpenCV grayscale/downscale/Gaussian/Canny/component-filter",
            "cpu_only": True,
            "deterministic": True,
            "grayscale": "cv2.COLOR_RGB2GRAY",
            "work_size_wh": [WORK_WIDTH, WORK_HEIGHT],
            "resize_interpolation": "INTER_AREA",
            "gaussian_kernel": list(GAUSSIAN_KERNEL),
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "gaussian_border": "BORDER_REFLECT_101",
            "canny_low": CANNY_LOW,
            "canny_high": CANNY_HIGH,
            "canny_aperture": 3,
            "canny_l2gradient": True,
            "component_connectivity": 8,
            "component_keep_rule": "area >= 10 AND max(width,height) >= 12 at 480x270",
            **component_stats,
            "upsample_size_wh": [1920, 1080],
            "upsample_interpolation": "INTER_NEAREST",
            "output_values": [0.0, 1.0],
            "output_dtype": "float32",
        },
        "mapping": {
            "visible_hw": [1080, 1920],
            "zero_padding_top_bottom": [PAD_TOP, PAD_BOTTOM],
            "encoded_hw": [1152, 1920],
            "latent_hw": [36, 60],
            "pixel_per_latent_cell_hw": [32, 32],
            "global_hw": [9, 15],
            "latent_per_global_cell_hw": [4, 4],
            "pixel_per_global_cell_hw": [128, 128],
            "area_average": "exact reshape mean at both reductions",
            "strength_formula": "clip(global_edge_density / 0.05, 0, 1)",
            "density_normalizer": DENSITY_NORMALIZER,
        },
        "coverage_and_distribution": {
            "visible_edge_fraction": float(full_edges.mean()),
            "encoded_edge_fraction": float(padded.mean()),
            "raw_work_edge_fraction": float((raw_edges > 0).mean()),
            "filtered_work_edge_fraction": float((clean_edges > 0).mean()),
            "global_density_min": float(global_density.min()),
            "global_density_max": float(global_density.max()),
            "global_density_mean": float(global_density.mean()),
            "global_strength_min": float(global_strength.min()),
            "global_strength_max": float(global_strength.max()),
            "global_strength_mean": float(global_strength.mean()),
            "global_strength_std": float(global_strength.std()),
            "global_strength_zero_cells": int(np.count_nonzero(global_strength == 0)),
            "global_strength_one_cells": int(np.count_nonzero(global_strength == 1)),
            "global_strength_nonzero_cells": int(np.count_nonzero(global_strength > 0)),
            "global_strength_percentiles": {
                str(q): float(np.percentile(global_strength, q)) for q in (0, 10, 25, 50, 75, 90, 100)
            },
        },
        "outputs": {p.name: {"path": str(p), "sha256": sha256(p)} for p in sorted(REF.iterdir()) if p.is_file()},
        "runtime": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["coverage_and_distribution"], indent=2, sort_keys=True))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
