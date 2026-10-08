from __future__ import annotations

import importlib.util
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SOURCE = REPO_ROOT / "pipeline/src/03to04/upscale_blend_16k.py"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("upscale_blend_16k", PIPELINE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PIPELINE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_pipeline_module()


class UpscaleInputContractTest(unittest.TestCase):
    def test_all_wrappers_read_directly_from_stage_02(self) -> None:
        for video_id in ("01", "02", "03", "04", "05"):
            wrapper = (
                REPO_ROOT
                / "pipeline/config"
                / video_id
                / f"{video_id}_x4Upscale_16K.sh"
            )
            text = wrapper.read_text(encoding="utf-8")
            expected = (
                f"data/02_VideoOutPaint_4K_3840x2160/"
                f"{video_id}_VideoOutPaint_4K.mp4"
            )
            self.assertIn(expected, text)
            self.assertNotIn("03_VideoOutPaint_4K_16fps_3840x2160", text)

    def test_pipeline_uses_24fps_input_and_16fps_output(self) -> None:
        self.assertEqual(pipeline.EXPECTED_INPUT_4K_FPS, Fraction(24, 1))
        self.assertEqual(pipeline.EXPECTED_ORIGINAL_8K_FPS, Fraction(16, 1))
        self.assertEqual(pipeline.EXPECTED_OUTPUT_FPS, Fraction(16, 1))

    def test_frame_index_alignment_accepts_distinct_cfr_timelines(self) -> None:
        input_timestamps = [10.0 + index / 24 for index in range(4)]
        original_timestamps = [20.0 + index / 16 for index in range(4)]
        with mock.patch.object(
            pipeline,
            "frame_timestamps",
            side_effect=[input_timestamps, original_timestamps],
        ):
            pipeline.validate_frame_index_alignment(
                Path("input.mp4"),
                Path("original.mp4"),
                frame_count=4,
                input_fps=Fraction(24, 1),
                original_fps=Fraction(16, 1),
            )

    def test_frame_index_alignment_rejects_non_cfr_input(self) -> None:
        input_timestamps = [0.0, 1 / 24, 0.2]
        original_timestamps = [0.0, 1 / 16, 2 / 16]
        with mock.patch.object(
            pipeline,
            "frame_timestamps",
            side_effect=[input_timestamps, original_timestamps],
        ), self.assertRaisesRegex(pipeline.PipelineError, "4K frame 2"):
            pipeline.validate_frame_index_alignment(
                Path("input.mp4"),
                Path("original.mp4"),
                frame_count=3,
                input_fps=Fraction(24, 1),
                original_fps=Fraction(16, 1),
            )


if __name__ == "__main__":
    unittest.main()
