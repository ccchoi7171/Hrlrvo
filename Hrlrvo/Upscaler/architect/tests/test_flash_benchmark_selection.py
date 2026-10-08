from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "FlashVSR_v1.1" / "upscale.py"
SPEC = importlib.util.spec_from_file_location("flashvsr_upscale_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
flash = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = flash
SPEC.loader.exec_module(flash)


def test_full_defaults_preserve_complete_video() -> None:
    assert flash.benchmark_frame_selection(92, "full", 0, None) == (0, 92)
    assert flash.benchmark_frame_selection(10, "full", 0, None) == (0, 10)


def test_cli_defaults_remain_legacy_full() -> None:
    args = flash.parser().parse_args([])
    assert args.benchmark_mode == "full"
    assert args.max_frames is None
    assert args.start_frame == 0
    assert args.stage_mode == "all"


def test_micro_and_mini_use_official_streaming_units() -> None:
    assert flash.benchmark_frame_selection(92, "micro", 3, None) == (3, 21)
    assert flash.benchmark_frame_selection(92, "mini", 3, None) == (3, 29)
    assert flash.padded_model_frames(21) == 25
    assert flash.padded_model_frames(29) == 33
    assert flash.streaming_step_count(21) == 1
    assert flash.streaming_step_count(29) == 2
    assert flash.streaming_step_count(100) == 11


def test_flash_estimator_accounts_for_streaming_steps_and_cpu_io() -> None:
    estimates = flash.estimate_flash_times(
        model_load_seconds=8.0,
        inference_work_seconds=100.0,
        merge_seconds=21.0,
        encode_seconds=21.0,
        staging_seconds=42.0,
        measured_frames=21,
        frame_targets=(21, 29),
    )
    assert estimates["21"]["streaming_steps"] == 1
    assert estimates["21"]["estimated_total_seconds"] == pytest.approx(192.0)
    assert estimates["29"]["streaming_steps"] == 2
    assert estimates["29"]["estimated_total_seconds"] == pytest.approx(324.0)


def test_explicit_range_cap_is_applied_after_start() -> None:
    assert flash.benchmark_frame_selection(92, "full", 10, 25) == (10, 25)
    assert flash.benchmark_frame_selection(92, "micro", 10, 50) == (10, 21)
    assert flash.benchmark_frame_selection(30, "mini", 5, None) == (5, 25)


def test_container_extension_preserves_dot_in_model_version() -> None:
    source = Path("clip_FlashVSR_v1.1_x4_micro_f000000_n21.mkv")
    assert flash.with_container_extension(source, ".mp4").name == (
        "clip_FlashVSR_v1.1_x4_micro_f000000_n21.mp4"
    )


@pytest.mark.parametrize(
    "total,mode,start,limit",
    [(92, "micro", -1, None), (92, "micro", 92, None), (92, "micro", 80, 10)],
)
def test_invalid_or_too_short_ranges_are_rejected(
    total: int, mode: str, start: int, limit: int | None,
) -> None:
    with pytest.raises(ValueError):
        flash.benchmark_frame_selection(total, mode, start, limit)
