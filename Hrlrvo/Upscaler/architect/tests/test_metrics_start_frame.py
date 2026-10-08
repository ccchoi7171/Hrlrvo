from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ARCHITECT = Path(__file__).resolve().parents[1]
if str(ARCHITECT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT))

from common import metrics


class MetricsStartFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = SimpleNamespace(frame_count=2, width=8, height=8)
        first = np.zeros((8, 8, 3), dtype=np.uint8)
        second = np.full((8, 8, 3), 10, dtype=np.uint8)
        self.frames = [first, second]

    def test_structure_offsets_only_source_indices(self) -> None:
        with patch.object(metrics, "uniform_sample_indices", return_value=[0, 1]), patch.object(
            metrics, "decode_selected_frames", side_effect=[iter(self.frames), iter(self.frames)]
        ) as decode:
            result = metrics.analyze_structure(
                "source.mp4", "output.mp4", self.metadata,
                ffmpeg_threads=2, source_start_frame=7,
            )
        self.assertEqual(decode.call_args_list[0].args[1], [7, 8])
        self.assertEqual(decode.call_args_list[1].args[1], [0, 1])
        self.assertEqual(result["source_sample_indices"], [7, 8])

    def test_temporal_trims_source_but_not_output(self) -> None:
        with patch.object(
            metrics, "_iter_scaled_frames", side_effect=[iter(self.frames), iter(self.frames)]
        ) as decoder:
            result = metrics.analyze_temporal(
                "source.mp4", "output.mp4", self.metadata, source_start_frame=7,
            )
        self.assertEqual(decoder.call_args_list[0].kwargs["start_frame"], 7)
        self.assertEqual(decoder.call_args_list[0].kwargs["frame_count"], 2)
        self.assertNotIn("start_frame", decoder.call_args_list[1].kwargs)
        self.assertEqual(result["decoded_frames"], 2)
        self.assertAlmostEqual(result["mean_input_interframe_difference"], 10.0)
        self.assertAlmostEqual(result["mean_interframe_difference"], 10.0)
        self.assertAlmostEqual(result["mean_output_to_input_motion_ratio"], 1.0)
        self.assertAlmostEqual(
            result["mean_temporal_restoration_residual_difference"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
