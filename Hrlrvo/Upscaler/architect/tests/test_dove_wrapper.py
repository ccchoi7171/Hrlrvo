from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

WRAPPER = Path(__file__).resolve().parents[1] / "DOVE" / "upscale.py"
SPEC = importlib.util.spec_from_file_location("_dove_wrapper_test", WRAPPER)
assert SPEC is not None and SPEC.loader is not None
dove = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dove
SPEC.loader.exec_module(dove)


class DOVEWrapperUnitTest(unittest.TestCase):
    def test_benchmark_native_units(self) -> None:
        self.assertEqual(dove.benchmark_frame_range("micro", 92, 0, None), (0, 9))
        self.assertEqual(dove.benchmark_frame_range("mini", 92, 3, None), (3, 17))
        self.assertEqual(dove.padded_frame_count(1), 1)
        self.assertEqual(dove.padded_frame_count(9), 9)
        self.assertEqual(dove.padded_frame_count(10), 17)

    def test_temporal_chunks_are_native_lengths_and_cover_once(self) -> None:
        chunks = dove.make_temporal_chunks(33, 17, 8)
        self.assertTrue(all((chunk.size - 1) % 8 == 0 for chunk in chunks))
        counts = [0] * 33
        for chunk in chunks:
            for index in range(chunk.valid_start, chunk.valid_end):
                counts[index] += 1
        self.assertEqual(counts, [1] * 33)

    def test_report_chunk_boundaries_use_integer_transition_indices(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn(
            '"chunk_boundaries": [chunk.start for chunk in chunks[1:]]', source
        )
        self.assertIn(
            '"temporal_chunk_ranges": [[chunk.start, chunk.end] for chunk in chunks]',
            source,
        )

    def test_spatial_output_tiles_cover_every_pixel_once(self) -> None:
        tiles = dove.make_spatial_tiles(8640, 15360, 768, 1360, 32)
        self.assertEqual(len(tiles), 144)
        # Rectangle areas must exactly partition the target; this proves the
        # centre-valid merge has neither holes nor multiple writes.
        area = sum(
            (tile.y.valid_end - tile.y.valid_start) * (tile.x.valid_end - tile.x.valid_start)
            for tile in tiles
        )
        self.assertEqual(area, 8640 * 15360)
        self.assertTrue(all(tile.y.start % 16 == tile.x.start % 16 == 0 for tile in tiles))

    def test_invalid_unaligned_spatial_plan_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            dove.make_spatial_tiles(8640, 15360, 768, 1350, 32)

    def test_cli_defaults_encode_official_output_domain_plan(self) -> None:
        args = dove.build_parser().parse_args([])
        self.assertEqual(args.tile_height, 768)
        self.assertEqual(args.tile_width, 1360)
        self.assertEqual(args.tile_overlap, 32)
        self.assertEqual(args.temporal_window, 17)
        self.assertEqual(args.temporal_overlap, 8)
        self.assertEqual(args.max_oom_retries, 2)
        self.assertEqual(args.model_path, str(dove.DEFAULT_MODEL_PATH))
        self.assertIsNone(args.report_json)
        explicit = dove.build_parser().parse_args(["--report-json", "/tmp/dove.json"])
        self.assertEqual(explicit.report_json, "/tmp/dove.json")

    def test_bounded_spatial_oom_plan_only_shrinks(self) -> None:
        self.assertEqual(
            dove.spatial_retry_plans(768, 1360, 2),
            [(768, 1360), (640, 1024), (512, 768)],
        )
        self.assertEqual(dove.spatial_retry_plans(640, 1024, 2), [(640, 1024), (512, 768)])
        self.assertEqual(dove.spatial_retry_plans(768, 1360, 0), [(768, 1360)])

    def test_missing_gpu_utilization_does_not_break_memory_sampling(self) -> None:
        self.assertIsNone(dove.parse_optional_number("[Not Found]\n"))
        self.assertEqual(dove.parse_optional_number("87\n"), 87.0)

    def test_source_has_no_resize_fallback(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("fallback_output", source)
        self.assertIn("pipe.vae.encode(video).latent_dist.sample()", source)
        self.assertIn("pipe.scheduler.get_velocity", source)
        self.assertIn("pipe.decode_latents", source)
        self.assertNotIn(".mul(255).round()", source)
        self.assertIn("with torch.inference_mode()", source)

    def test_first_signal_becomes_cleanup_exception_and_ignores_duplicates(self) -> None:
        handler = dove._OneShotSignalHandler()
        with (
            patch.object(dove.signal, "signal") as install,
            self.assertRaises(dove.DOVECancelled),
        ):
            handler(dove.signal.SIGTERM, None)
        installed_signals = {call.args[0] for call in install.call_args_list}
        self.assertIn(dove.signal.SIGTERM, installed_signals)
        self.assertIn(dove.signal.SIGINT, installed_signals)


if __name__ == "__main__":
    unittest.main()
