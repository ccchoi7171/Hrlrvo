from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ARCHITECT = Path(__file__).resolve().parents[1]
if str(ARCHITECT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT))

import frame_sr_common as common


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ARCHITECT / relative / "upscale.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


real = load("frame_sr_real", "Real-ESRGAN")
hat = load("frame_sr_hat", "HAT")


def test_frame_ranges_match_video_benchmarks_but_allow_one_frame_smoke() -> None:
    assert common.benchmark_frame_range("micro", 92, 0, None) == (0, 21)
    assert common.benchmark_frame_range("mini", 92, 3, None) == (3, 29)
    assert common.benchmark_frame_range("full", 92, 10, None) == (10, 82)
    assert common.benchmark_frame_range("micro", 92, 4, 1) == (4, 1)
    with pytest.raises(ValueError):
        common.benchmark_frame_range("micro", 92, 92, None)


def test_cli_contract_and_safe_model_defaults() -> None:
    real_args = real.parser().parse_args([])
    hat_args = hat.parser().parse_args([])
    assert real_args.tile_size == hat_args.tile_size == 512
    assert real_args.tile_overlap == 10
    assert hat_args.tile_overlap == 32
    assert real_args.scale == hat_args.scale == 4
    assert real_args.benchmark_mode == hat_args.benchmark_mode == "full"
    for args in (real_args, hat_args):
        assert hasattr(args, "output_dir")
        assert hasattr(args, "result_dir")
        assert hasattr(args, "report_json")
        assert hasattr(args, "encoder")


def test_native_4k_tile_count_is_40_per_frame() -> None:
    assert common.spatial_tile_count(3840, 2160, 512) == 40


def test_hat_halo_crop_produces_exact_x4_without_blending() -> None:
    backend = object.__new__(hat.HATBackend)
    backend.args = hat.argparse.Namespace(tile_size=32, tile_overlap=16)

    def nearest(tile: np.ndarray) -> tuple[np.ndarray, float]:
        return tile.repeat(4, axis=0).repeat(4, axis=1), 0.25

    backend._infer_tile = nearest
    image = np.arange(64 * 80 * 3, dtype=np.uint8).reshape(64, 80, 3)
    output, timings = backend.upscale_rgb(image)
    expected = image.repeat(4, axis=0).repeat(4, axis=1)
    np.testing.assert_array_equal(output, expected)
    assert output.shape == (256, 320, 3)
    assert timings["neural_inference_seconds"] == pytest.approx(1.5)
    assert timings["merge_seconds"] >= 0


def test_official_pins_and_fidelity_checkpoints_are_explicit() -> None:
    assert real.PINNED_COMMIT == "a4abfb2979a7bbff3f69f58f58ae324608821e27"
    assert hat.PINNED_COMMIT == "1638a9a822581657811867bf670717f8371fc3e5"
    assert real.WEIGHT.name == "RealESRGAN_x4plus.pth"
    assert hat.WEIGHT.name == "Real_HAT_GAN_SRx4.pth"
    assert real.preflight_assets()["checkpoint_sha256"] == real.WEIGHT_SHA256
    assert hat.preflight_assets()["checkpoint_sha256"] == hat.WEIGHT_SHA256
