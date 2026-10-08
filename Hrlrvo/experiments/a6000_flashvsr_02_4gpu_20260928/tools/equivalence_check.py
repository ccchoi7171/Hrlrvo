#!/usr/bin/env python3
"""Bounded CPU-only serial-vs-final-only pixel equivalence on real 03 tiles."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = Path("/home/oem/hrlrvo/experiments/a6000_flashvsr_0304_20260926")
OUT = ROOT / "verification/final_only_equivalence.json"
FRAMES = (0, 1)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fc = load(ORIGIN / "package/src/cert_pipeline/flash_composite.py", "_eq_frozen_composite")
reference = fc.load_reference_module(ORIGIN / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py")
renderer = load(ROOT / "final_only/parallel_renderer.py", "_eq_parallel_renderer")


def tile_directories() -> tuple[list, dict[int, Path]]:
    manifest = json.loads((ORIGIN / "runs/03/aggregate_manifest.json").read_text())
    tiles = reference.make_tiles(3840, 2160, 256, 32)
    artifacts = {int(item["tile"]["index"]): Path(item["directory"])
                 for item in manifest["tile_artifacts"]}
    if set(artifacts) != set(range(170)):
        raise RuntimeError("video03 aggregate is not the complete real 170-tile layout")
    return tiles, artifacts


def decode_original(count: int) -> list[np.ndarray]:
    anchor = ORIGIN / "package/data/00_Input_8K_7680x4320/03_Input_8K.mp4"
    command = ["ffmpeg", "-v", "error", "-threads", "16", "-i", str(anchor),
               "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    frame_bytes = 7680 * 4320 * 3
    if len(completed.stdout) != frame_bytes * count:
        raise RuntimeError("bounded original decode returned the wrong byte count")
    return [np.frombuffer(completed.stdout[i*frame_bytes:(i+1)*frame_bytes], dtype=np.uint8)
            .reshape(4320, 7680, 3) for i in range(count)]


def serial_render(index: int, original: np.ndarray, tiles: list, directories: dict[int, Path]) -> np.ndarray:
    width, height = fc.EXPECTED_16K
    output = np.empty((height, width, 3), dtype=np.uint8)
    bands = fc.make_vertical_bands(tiles, height)
    axes = {tile.index: fc.tile_weight_axes(tile, tiles, reference) for tile in tiles}
    bounds = {tile.index: (tile.y*4, tile.x*4, (tile.y+tile.height)*4,
                           (tile.x+tile.width)*4) for tile in tiles}
    cache: dict[int, np.ndarray] = {}
    for start, end, active in bands:
        active_ids = {tile.index for tile in active}
        for stale in tuple(cache):
            if stale not in active_ids:
                del cache[stale]
        for tile in active:
            if tile.index not in cache:
                path = directories[tile.index] / f"{index:09d}.png"
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"missing real committed tile frame: {path}")
                cache[tile.index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        accumulator = np.zeros((end-start, width, 3), dtype=np.float32)
        normalization = np.zeros((end-start, width), dtype=np.float32)
        for tile in active:
            ty0, tx0, _, tx1 = bounds[tile.index]
            ly0, ly1 = start-ty0, end-ty0
            ya, xa = axes[tile.index]
            weight = ya[ly0:ly1, None] * xa[None, :]
            pixels = cache[tile.index][ly0:ly1].astype(np.float32)
            pixels *= weight[..., None]
            accumulator[:, tx0:tx1] += pixels
            normalization[:, tx0:tx1] += weight
        rows = np.rint(accumulator / normalization[..., None]).clip(0,255).astype(np.uint8)
        fc.composite_original_center_rows(rows, original, canvas_row_start=start, feather_width=64)
        output[start:end] = rows
    return output


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"fresh-only evidence path exists: {OUT}")
    tiles, directories = tile_directories()
    originals = decode_original(max(FRAMES)+1)
    records = []
    with renderer.FrameComposer(reference, tiles, directories, workers=8, feather_width=64) as composer:
        for index in FRAMES:
            serial = serial_render(index, originals[index], tiles, directories)
            direct = composer.render(index, originals[index])
            difference = np.abs(serial.astype(np.int16) - direct.astype(np.int16))
            max_abs = int(difference.max())
            mean_abs = float(difference.mean())
            mse = float(np.mean((serial.astype(np.float32)-direct.astype(np.float32))**2))
            psnr = "inf" if mse == 0.0 else 20*math.log10(255.0/math.sqrt(mse))
            y0, x0, y1, x1 = fc.CENTER_BBOX
            edge_mask = np.zeros((serial.shape[0], serial.shape[1]), dtype=bool)
            fw = 64
            edge_mask[y0:y0 + fw, x0:x1] = True
            edge_mask[y1 - fw:y1, x0:x1] = True
            edge_mask[y0:y1, x0:x0 + fw] = True
            edge_mask[y0:y1, x1 - fw:x1] = True
            edge_diff = difference[edge_mask]
            records.append({"frame": index, "max_abs_error": max_abs,
                            "mean_abs_error": mean_abs, "psnr_db": psnr,
                            "anchor_boundary_max_abs_error": int(edge_diff.max()),
                            "anchor_boundary_mean_abs_error": float(edge_diff.mean()),
                            "serial_rgb_sha256": hashlib.sha256(serial.tobytes()).hexdigest(),
                            "direct_rgb_sha256": hashlib.sha256(direct.tobytes()).hexdigest()})
            del serial, direct, difference, edge_diff
    passed = all(item["max_abs_error"] == 0 and item["anchor_boundary_max_abs_error"] == 0
                 for item in records)
    value = {"schema": "a6000_video02_final_only_equivalence/v1",
             "status": "PASS" if passed else "FAIL",
             "cpu_only": True, "source_video": "03", "frames": list(FRAMES),
             "scope": "real complete 170-tile committed PNG layout and correct 03 anchor",
             "path_a": "serial production float32 reconstruction then 64px anchor composite",
             "path_b": "final-only threaded direct reconstruction plus identical 64px anchor composite",
             "full_raw_16k_mp4_created": False, "full_video_decode_performed": False,
             "results": records}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_name(f".{OUT.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    os.replace(temporary, OUT)
    print(json.dumps(value))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
