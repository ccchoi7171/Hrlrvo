from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ARCHITECT = Path(__file__).resolve().parents[1]
if str(ARCHITECT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT))

from summarize_benchmarks import (
    dove_work_plan,
    fit_dove_pair,
    fit_frame_independent_pair,
    fit_pair,
    flash_steps,
    generate_contact_sheets,
    seed_chunks,
    synthesize,
)


class BenchmarkSummaryTest(unittest.TestCase):
    @staticmethod
    def report(frames: int, neural: float, decode: float, cpu: float) -> dict:
        inference = neural + decode + 20.0
        return {
            "effective_output_frames_written": frames,
            "timing": {
                "model_load_seconds": 10.0,
                "neural_inference_seconds": neural,
                "decode_seconds": decode,
                "sr_inference_seconds": inference,
                "merge_seconds": cpu / 2,
                "encode_seconds": cpu / 2,
                "total_wall_seconds": 10.0 + inference + cpu + 5.0,
            },
        }

    def test_temporal_unit_formulas(self) -> None:
        self.assertEqual([flash_steps(n) for n in (21, 22, 29, 30)], [1, 2, 2, 3])
        self.assertEqual([seed_chunks(n) for n in (5, 6, 9, 10)], [1, 2, 2, 3])
        self.assertEqual(
            [dove_work_plan(n) for n in (9, 10, 17, 18, 92)],
            [(1, 1), (1, 2), (1, 2), (2, 4), (10, 20)],
        )

    def test_dove_fit_repeats_chunk_warmup_for_locked_overlap_plan(self) -> None:
        fitted = fit_dove_pair(
            self.report(9, 80.0, 20.0, 45.0),
            self.report(17, 200.0, 50.0, 85.0),
            targets=(17, 18),
        )
        coefficients = fitted["coefficients"]
        self.assertEqual(coefficients["chunk_warmup_full_sr_inference_seconds"], 120.0)
        self.assertEqual(
            coefficients["additional_native_8_frame_unit_full_sr_inference_seconds"],
            150.0,
        )
        self.assertEqual(fitted["estimates"]["17"]["temporal_chunks"], 1)
        self.assertEqual(fitted["estimates"]["18"]["temporal_chunks"], 2)
        self.assertEqual(
            fitted["estimates"]["18"]["native_8_frame_units_processed"], 4
        )

    def test_dove_fit_excludes_one_time_oom_retry_overhead(self) -> None:
        micro = self.report(9, 80.0, 20.0, 45.0)
        mini = self.report(17, 200.0, 50.0, 85.0)
        baseline = fit_dove_pair(micro, mini, targets=(92,))
        mini["timing"]["total_wall_seconds"] += 37.0
        mini["timings_detail"] = {"oom_retry_overhead_seconds": 37.0}
        retried = fit_dove_pair(micro, mini, targets=(92,))
        self.assertEqual(
            baseline["estimates"]["92"]["estimated_total_seconds"],
            retried["estimates"]["92"]["estimated_total_seconds"],
        )
        self.assertEqual(
            retried["coefficients"]["excluded_one_time_oom_retry_overhead_seconds"],
            37.0,
        )

    def test_pair_fit_separates_warmup_steady_and_cpu_per_frame(self) -> None:
        fitted = fit_pair(
            self.report(5, 80.0, 20.0, 25.0),
            self.report(9, 200.0, 50.0, 45.0),
            unit_name="temporal_chunks",
            unit_count=seed_chunks,
            targets=(5, 9),
        )
        coefficients = fitted["coefficients"]
        self.assertEqual(coefficients["warmup_full_sr_inference_seconds"], 120.0)
        self.assertEqual(coefficients["steady_additional_unit_full_sr_inference_seconds"], 150.0)
        self.assertEqual(coefficients["warmup_pure_neural_decode_seconds"], 100.0)
        self.assertEqual(coefficients["steady_additional_unit_pure_neural_decode_seconds"], 150.0)
        self.assertEqual(coefficients["fixed_orchestration_seconds"], 5.0)
        self.assertEqual(coefficients["cpu_io_seconds_per_output_frame"], 5.0)
        self.assertEqual(fitted["estimates"]["9"]["estimated_total_seconds"], 330.0)

    def test_frame_independent_fit_uses_micro_mini_delta(self) -> None:
        micro = self.report(21, 80.0, 20.0, 25.0)
        mini = self.report(29, 120.0, 30.0, 45.0)
        # Totals are 160 and 230 seconds, so the measured steady delta is 8.75 s/frame.
        fitted = fit_frame_independent_pair(micro, mini, targets=(10, 92))
        coefficients = fitted["coefficients"]
        self.assertEqual(coefficients["model_load_once_seconds"], 10.0)
        self.assertAlmostEqual(coefficients["steady_end_to_end_seconds_per_frame"], 8.75)
        self.assertAlmostEqual(coefficients["fixed_orchestration_seconds"], -33.75)
        self.assertAlmostEqual(fitted["estimates"]["92"]["estimated_total_seconds"], 781.25)

    def test_frame_independent_fit_removes_load_jitter_from_slope(self) -> None:
        micro = self.report(21, 80.0, 20.0, 25.0)
        mini = self.report(29, 120.0, 30.0, 45.0)
        mini["timing"]["model_load_seconds"] = 18.0
        mini["timing"]["total_wall_seconds"] += 8.0
        fitted = fit_frame_independent_pair(micro, mini, targets=(29,))
        self.assertAlmostEqual(
            fitted["coefficients"]["steady_end_to_end_seconds_per_frame"], 8.75
        )

    def test_synthesize_accepts_an_image_model_only_partial_result(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "Real-ESRGAN"
            model_dir.mkdir()
            for mode, report in (("micro", self.report(21, 80, 20, 25)), ("mini", self.report(29, 120, 30, 45))):
                report["technical_validation"] = {"status": "PASS"}
                (model_dir / f"benchmark_{mode}.json").write_text(json.dumps(report))
            payload = synthesize(root, targets=(10,))
            self.assertEqual(list(payload["models"]), ["Real-ESRGAN"])
            self.assertFalse(payload["complete_four_model_comparison"])

    def test_contact_sheet_uses_three_identical_relative_frames(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for model in ("Real-ESRGAN", "HAT"):
                directory = root / model
                directory.mkdir()
                (directory / "benchmark_micro.json").write_text(
                    json.dumps({
                        "input": {"path": "/source.mp4"},
                        "output_path": f"/{model}.mp4",
                    })
                )
            quality = {
                "modes": {
                    "micro": {
                        "common_frame_count": 21,
                        "models": {"Real-ESRGAN": {}, "HAT": {}},
                    }
                }
            }
            with patch("summarize_benchmarks.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stderr = ""
                manifest = generate_contact_sheets(root, quality)
            command = run.call_args.args[0]
            filters = command[command.index("-filter_complex") + 1]
            self.assertIn("eq(n,0)", filters)
            self.assertIn("eq(n,10)", filters)
            self.assertIn("eq(n,20)", filters)
            self.assertIn("xstack=inputs=9", filters)
            self.assertIn("crop=512:288", filters)
            self.assertEqual(
                manifest["modes"]["micro"]["column_order_left_to_right"],
                ["source_lanczos_preview", "Real-ESRGAN", "HAT"],
            )
            self.assertEqual(
                manifest["modes"]["micro"]["sample_frames_top_to_bottom"], [0, 10, 20]
            )


if __name__ == "__main__":
    unittest.main()
