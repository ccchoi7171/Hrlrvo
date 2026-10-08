from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ARCHITECT = Path(__file__).resolve().parents[1]
MODULE_PATH = ARCHITECT / "Upscale-A-Video" / "upscale.py"
SPEC = importlib.util.spec_from_file_location("upscale_a_video_adapter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class UpscaleAVideoAvailabilityTest(unittest.TestCase):
    def test_official_4k_guard_returns_4k_not_16k(self) -> None:
        self.assertEqual(MODULE.official_output_dimensions(3840, 2160), (3840, 2160))

    def test_direct_4k_geometry_would_require_135_tile_cores(self) -> None:
        self.assertEqual(MODULE.theoretical_tile_count(3840, 2160), 135)

    def test_temporal_modes_match_native_window_and_overlap_path(self) -> None:
        self.assertEqual(MODULE.benchmark_frame_count("micro", 92, 0, None), 8)
        self.assertEqual(MODULE.benchmark_frame_count("mini", 92, 0, None), 14)


if __name__ == "__main__":
    unittest.main()
