from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

ARCHITECT = Path(__file__).resolve().parents[1]
if str(ARCHITECT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT))

import benchmark_minimal


class MinimalBenchmarkCommandTest(unittest.TestCase):
    def test_legacy_dove_chunk_ranges_migrate_to_transition_indices(self) -> None:
        report = {"chunk_boundaries": [[0, 17], [9, 26], [18, 33]]}
        self.assertEqual(
            benchmark_minimal.normalize_temporal_boundaries(report), [9, 18]
        )
        self.assertEqual(report["chunk_boundaries"], [9, 18])
        self.assertEqual(
            report["temporal_chunk_ranges"], [[0, 17], [9, 26], [18, 33]]
        )

    def setUp(self) -> None:
        self.args = Namespace(
            input=str(benchmark_minimal.DEFAULT_INPUT),
            output_root=str(benchmark_minimal.DEFAULT_OUTPUT_ROOT),
            result_root=str(benchmark_minimal.DEFAULT_RESULT_ROOT),
            start_frame=7,
            ffmpeg_threads=12,
            encoder="libx265",
        )

    def test_flash_micro_uses_native_causal_mode(self) -> None:
        command = benchmark_minimal.build_command("FlashVSR_v1.1", "micro", self.args)
        self.assertIn("--benchmark-mode", command)
        self.assertIn("micro", command)
        self.assertIn("--start-frame", command)
        self.assertIn("7", command)
        self.assertIn("320", command)

    def test_seed_uses_retained_safe_h200_plan(self) -> None:
        command = benchmark_minimal.build_command("SeedVR2", "mini", self.args)
        joined = " ".join(command)
        self.assertIn("--temporal-window 5", joined)
        self.assertIn("--temporal-overlap 1", joined)
        self.assertIn("--tile-size 488", joined)
        self.assertIn("--tile-overlap 60", joined)
        self.assertIn("--max-oom-retries 3", joined)

    def test_dove_uses_official_output_domain_tiling_and_temporal_plan(self) -> None:
        command = benchmark_minimal.build_command("DOVE", "mini", self.args)
        joined = " ".join(command)
        self.assertEqual(
            command[0],
            "/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-DOVE/bin/python",
        )
        self.assertIn("--tile-height 768", joined)
        self.assertIn("--tile-width 1360", joined)
        self.assertIn("--tile-overlap 32", joined)
        self.assertIn("--temporal-window 17", joined)
        self.assertIn("--temporal-overlap 8", joined)
        self.assertIn("--max-oom-retries 2", joined)

    def test_dove_mini_is_blocked_after_micro_failure(self) -> None:
        failed = {("DOVE", "micro")}
        self.assertTrue(
            benchmark_minimal.blocked_by_prior_mode_failure("DOVE", "mini", failed)
        )
        self.assertFalse(
            benchmark_minimal.blocked_by_prior_mode_failure("DOVE", "micro", failed)
        )
        self.assertFalse(
            benchmark_minimal.blocked_by_prior_mode_failure("SeedVR2", "mini", failed)
        )

    def test_upscale_a_video_is_availability_only(self) -> None:
        command = benchmark_minimal.build_command("Upscale-A-Video", "micro", self.args)
        self.assertIn("--availability-only", command)

    def test_real_esrgan_uses_image_sr_environment_and_official_halo(self) -> None:
        command = benchmark_minimal.build_command("Real-ESRGAN", "micro", self.args)
        joined = " ".join(command)
        self.assertEqual(command[0], "/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-ImageSR/bin/python")
        self.assertIn("--tile-size 512", joined)
        self.assertIn("--tile-overlap 10", joined)

    def test_hat_uses_image_sr_environment_and_larger_halo(self) -> None:
        command = benchmark_minimal.build_command("HAT", "mini", self.args)
        joined = " ".join(command)
        self.assertEqual(command[0], "/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-ImageSR/bin/python")
        self.assertIn("--tile-size 512", joined)
        self.assertIn("--tile-overlap 32", joined)

    def test_dry_run_does_not_overwrite_durable_run_status(self) -> None:
        source = Path(benchmark_minimal.__file__).read_text(encoding="utf-8")
        self.assertIn('"run_status_dry_run.json" if args.dry_run else "run_status.json"', source)
        args = benchmark_minimal.parse_args(["--status-json", "/tmp/dedicated.json"])
        self.assertEqual(args.status_json, "/tmp/dedicated.json")

    def test_successful_runner_refreshes_quality_and_contact_sheets(self) -> None:
        source = Path(benchmark_minimal.__file__).read_text(encoding="utf-8")
        self.assertIn("analyze_common_quality(", source)
        self.assertIn("generate_contact_sheets(result_root, quality)", source)
        self.assertIn('result_root / "quality_comparison.json"', source)


if __name__ == "__main__":
    unittest.main()
