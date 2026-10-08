#!/usr/bin/env python3
"""Official fidelity Real-HAT-GAN x4 frame-independent 4K -> 16K wrapper."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHITECT_ROOT = HERE.parent
PROJECT_ROOT = ARCHITECT_ROOT.parent
MODEL_ROOT = PROJECT_ROOT / "models" / "HAT"
REPOSITORY = MODEL_ROOT / "repository"
WEIGHT = MODEL_ROOT / "weights" / "Real_HAT_GAN_SRx4.pth"
DEFAULT_INPUT = PROJECT_ROOT / "input" / "0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "benchmark" / "HAT"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "result" / "benchmark" / "HAT"
MODEL_NAME = "HAT"
PINNED_COMMIT = "1638a9a822581657811867bf670717f8371fc3e5"
WEIGHT_SHA256 = "f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7"
WINDOW_SIZE = 16

if str(ARCHITECT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT_ROOT))

from frame_sr_common import add_common_arguments, main_for_model, verify_pinned_assets


def preflight_assets() -> dict[str, str]:
    return verify_pinned_assets(REPOSITORY, PINNED_COMMIT, WEIGHT, WEIGHT_SHA256)


def _install_torchvision_functional_tensor_shim() -> None:
    try:
        importlib.import_module("torchvision.transforms.functional_tensor")
    except ModuleNotFoundError:
        import torchvision.transforms._functional_tensor as implementation

        target = "torchvision.transforms.functional_tensor"
        shim = types.ModuleType(target)
        for name in dir(implementation):
            if not name.startswith("__"):
                setattr(shim, name, getattr(implementation, name))
        sys.modules[target] = shim


class HATBackend:
    def __init__(self, model: Any, torch_module: Any, args: argparse.Namespace) -> None:
        self.model = model
        self.torch = torch_module
        self.args = args
        self.model_details = {
            "official_repository": str(REPOSITORY),
            "pinned_commit": PINNED_COMMIT,
            "checkpoint": str(WEIGHT),
            "checkpoint_key": "params_ema",
            "checkpoint_sha256": WEIGHT_SHA256,
            "architecture": "HAT Real-GAN fidelity",
            "precision": "float32",
            "window_size": WINDOW_SIZE,
            "tile_strategy": "core plus halo, crop halo, direct placement",
        }

    def _infer_tile(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        torch = self.torch
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        tensor = tensor.div_(255.0).unsqueeze(0).cuda(self.args.gpu_index)
        height, width = tensor.shape[-2:]
        pad_h = (-height) % WINDOW_SIZE
        pad_w = (-width) % WINDOW_SIZE
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        torch.cuda.synchronize(self.args.gpu_index)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(tensor)
        torch.cuda.synchronize(self.args.gpu_index)
        seconds = time.perf_counter() - started
        output = output[:, :, : height * 4, : width * 4]
        array = (
            output.squeeze(0).float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        )
        return np.ascontiguousarray(array.transpose(1, 2, 0)), seconds

    def upscale_rgb(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        height, width = frame.shape[:2]
        output = np.empty((height * 4, width * 4, 3), dtype=np.uint8)
        neural_seconds = 0.0
        merge_seconds = 0.0
        core, halo = self.args.tile_size, self.args.tile_overlap
        for y0 in range(0, height, core):
            y1 = min(y0 + core, height)
            for x0 in range(0, width, core):
                x1 = min(x0 + core, width)
                hx0, hx1 = max(0, x0 - halo), min(width, x1 + halo)
                hy0, hy1 = max(0, y0 - halo), min(height, y1 + halo)
                tile, seconds = self._infer_tile(frame[hy0:hy1, hx0:hx1])
                neural_seconds += seconds
                tx0, ty0 = (x0 - hx0) * 4, (y0 - hy0) * 4
                tx1, ty1 = tx0 + (x1 - x0) * 4, ty0 + (y1 - y0) * 4
                merge_started = time.perf_counter()
                output[y0 * 4:y1 * 4, x0 * 4:x1 * 4] = tile[ty0:ty1, tx0:tx1]
                merge_seconds += time.perf_counter() - merge_started
                del tile
        return output, {
            "neural_inference_seconds": neural_seconds,
            "merge_seconds": merge_seconds,
        }


def load_backend(args: argparse.Namespace) -> HATBackend:
    if args.tile_size % WINDOW_SIZE or args.tile_overlap % WINDOW_SIZE:
        raise ValueError("HAT --tile-size and --tile-overlap must be multiples of window size 16")
    _install_torchvision_functional_tensor_shim()
    sys.path.insert(0, str(REPOSITORY))
    import torch
    architecture_spec = importlib.util.spec_from_file_location(
        "_official_hat_arch", REPOSITORY / "hat" / "archs" / "hat_arch.py"
    )
    if architecture_spec is None or architecture_spec.loader is None:
        raise ImportError("cannot load official hat/archs/hat_arch.py")
    architecture_module = importlib.util.module_from_spec(architecture_spec)
    architecture_spec.loader.exec_module(architecture_module)
    HAT = architecture_module.HAT

    model = HAT(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    checkpoint = torch.load(WEIGHT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params_ema"], strict=True)
    model.eval().cuda(args.gpu_index)
    return HATBackend(model, torch, args)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        value,
        default_input=DEFAULT_INPUT,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        default_result_dir=DEFAULT_RESULT_DIR,
        default_overlap=32,
    )
    return value


def main() -> int:
    return main_for_model(
        parser().parse_args(),
        model_name=MODEL_NAME,
        load_backend=load_backend,
        preflight_assets=preflight_assets,
    )


if __name__ == "__main__":
    raise SystemExit(main())
