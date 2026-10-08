from __future__ import annotations

import sys
import unittest
from pathlib import Path


ARCHITECT = Path(__file__).resolve().parents[1]
if str(ARCHITECT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT))

from SeedVR2.upscale import benchmark_frame_range, estimate_seed_times, temporal_chunk_count


class SeedBenchmarkModeTest(unittest.TestCase):
    def test_default_ranges_follow_temporal_vae_units(self) -> None:
        self.assertEqual(benchmark_frame_range("micro", 92, 0, None), (0, 5))
        self.assertEqual(benchmark_frame_range("mini", 92, 0, None), (0, 9))
        self.assertEqual(benchmark_frame_range("full", 92, 0, None), (0, 92))

    def test_explicit_start_and_max_frames_select_source_range(self) -> None:
        self.assertEqual(benchmark_frame_range("mini", 92, 7, 13), (7, 13))
        self.assertEqual(benchmark_frame_range("full", 92, 90, None), (90, 2))

    def test_micro_benchmark_rejects_less_than_minimum_temporal_unit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 5"):
            benchmark_frame_range("micro", 92, 0, 4)

    def test_temporal_chunk_count_includes_overlap_stride(self) -> None:
        self.assertEqual(temporal_chunk_count(5, 5, 1), 1)
        self.assertEqual(temporal_chunk_count(9, 5, 1), 2)
        self.assertEqual(temporal_chunk_count(92, 5, 1), 23)

    def test_estimate_scales_tile_chunks_and_per_frame_cpu_work_separately(self) -> None:
        estimates = estimate_seed_times(
            model_load_seconds=10.0,
            inference_work_seconds=90.0,
            merge_seconds=5.0,
            encode_seconds=5.0,
            measured_frames=5,
            measured_tile_chunks=45,
            tile_count=45,
            window=5,
            overlap=1,
            frame_targets=(5, 9),
        )
        self.assertEqual(estimates["5"]["temporal_chunks"], 1)
        self.assertAlmostEqual(estimates["5"]["estimated_total_seconds"], 110.0)
        self.assertEqual(estimates["9"]["temporal_chunks"], 2)
        self.assertAlmostEqual(estimates["9"]["estimated_total_seconds"], 208.0)


if __name__ == "__main__":
    unittest.main()
