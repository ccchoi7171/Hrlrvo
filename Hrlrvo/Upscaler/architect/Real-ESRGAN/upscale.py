#!/usr/bin/env python3
"""Official Real-ESRGAN x4plus frame-independent 4K -> 16K wrapper."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHITECT_ROOT = HERE.parent
PROJECT_ROOT = ARCHITECT_ROOT.parent
MODEL_ROOT = PROJECT_ROOT / "models" / "Real-ESRGAN"
REPOSITORY = MODEL_ROOT / "repository"
WEIGHT = MODEL_ROOT / "weights" / "RealESRGAN_x4plus.pth"
DEFAULT_INPUT = PROJECT_ROOT / "input" / "0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "benchmark" / "Real-ESRGAN"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "result" / "benchmark" / "Real-ESRGAN"
MODEL_NAME = "Real-ESRGAN"
PINNED_COMMIT = "a4abfb2979a7bbff3f69f58f58ae324608821e27"
WEIGHT_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"

if str(ARCHITECT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT_ROOT))

from frame_sr_common import add_common_arguments, main_for_model, verify_pinned_assets


def preflight_assets() -> dict[str, str]:
    return verify_pinned_assets(REPOSITORY, PINNED_COMMIT, WEIGHT, WEIGHT_SHA256)


def _install_torchvision_functional_tensor_shim() -> None:
    """BasicSR imports a module removed by recent torchvision releases."""
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


class RealESRGANBackend:
    def __init__(self, restorer: Any, torch_module: Any) -> None:
        self.restorer = restorer
        self.torch = torch_module
        self.model_details = {
            "official_repository": str(REPOSITORY),
            "pinned_commit": PINNED_COMMIT,
            "checkpoint": str(WEIGHT),
            "checkpoint_key": "params_ema",
            "checkpoint_sha256": WEIGHT_SHA256,
            "architecture": "RRDBNet(23 blocks)",
            "precision": "float16",
            "official_helper": "RealESRGANer",
        }

    def upscale_rgb(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        # The official helper accepts BGR and uses its own native tiled path.
        self.restorer._benchmark_neural_seconds = 0.0
        self.restorer._benchmark_merge_seconds = 0.0
        bgr = np.ascontiguousarray(frame[:, :, ::-1])
        output_bgr, _ = self.restorer.enhance(bgr, outscale=None)
        output_rgb = np.ascontiguousarray(output_bgr[:, :, ::-1])
        return output_rgb, {
            "neural_inference_seconds": self.restorer._benchmark_neural_seconds,
            "merge_seconds": self.restorer._benchmark_merge_seconds,
        }


def load_backend(args: argparse.Namespace) -> RealESRGANBackend:
    _install_torchvision_functional_tensor_shim()
    sys.path.insert(0, str(REPOSITORY))
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    helper_spec = importlib.util.spec_from_file_location(
        "_official_realesrgan_utils", REPOSITORY / "realesrgan" / "utils.py"
    )
    if helper_spec is None or helper_spec.loader is None:
        raise ImportError("cannot load official realesrgan/utils.py")
    helper_module = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper_module)
    RealESRGANer = helper_module.RealESRGANer

    class TimedRealESRGANer(RealESRGANer):
        """Unmodified official tiling math with CUDA-synchronized phase timers."""

        def tile_process(self) -> None:
            batch, channel, height, width = self.img.shape
            self.output = self.img.new_zeros((batch, channel, height * self.scale, width * self.scale))
            tiles_x = math.ceil(width / self.tile_size)
            tiles_y = math.ceil(height / self.tile_size)
            for y in range(tiles_y):
                for x in range(tiles_x):
                    input_start_x = x * self.tile_size
                    input_end_x = min(input_start_x + self.tile_size, width)
                    input_start_y = y * self.tile_size
                    input_end_y = min(input_start_y + self.tile_size, height)
                    x0 = max(input_start_x - self.tile_pad, 0)
                    x1 = min(input_end_x + self.tile_pad, width)
                    y0 = max(input_start_y - self.tile_pad, 0)
                    y1 = min(input_end_y + self.tile_pad, height)
                    tile = self.img[:, :, y0:y1, x0:x1]
                    torch.cuda.synchronize(args.gpu_index)
                    neural_started = time.perf_counter()
                    with torch.inference_mode():
                        output_tile = self.model(tile)
                    torch.cuda.synchronize(args.gpu_index)
                    self._benchmark_neural_seconds += time.perf_counter() - neural_started
                    ox0, ox1 = input_start_x * self.scale, input_end_x * self.scale
                    oy0, oy1 = input_start_y * self.scale, input_end_y * self.scale
                    tx0, ty0 = (input_start_x - x0) * self.scale, (input_start_y - y0) * self.scale
                    tx1 = tx0 + (input_end_x - input_start_x) * self.scale
                    ty1 = ty0 + (input_end_y - input_start_y) * self.scale
                    merge_started = time.perf_counter()
                    self.output[:, :, oy0:oy1, ox0:ox1] = output_tile[:, :, ty0:ty1, tx0:tx1]
                    torch.cuda.synchronize(args.gpu_index)
                    self._benchmark_merge_seconds += time.perf_counter() - merge_started

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    restorer = TimedRealESRGANer(
        scale=4,
        model_path=str(WEIGHT),
        model=model,
        tile=args.tile_size,
        tile_pad=args.tile_overlap,
        pre_pad=0,
        half=True,
        gpu_id=args.gpu_index,
    )
    return RealESRGANBackend(restorer, torch)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        value,
        default_input=DEFAULT_INPUT,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        default_result_dir=DEFAULT_RESULT_DIR,
        default_overlap=10,
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
