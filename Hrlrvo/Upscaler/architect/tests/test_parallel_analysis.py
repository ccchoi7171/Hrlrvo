from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


ARCHITECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARCHITECT))

import analyze_results


class ParallelAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="hrlrvo_parallel_analysis_test_"
        )
        self.result_root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_reports(self) -> None:
        for model in analyze_results.MODELS:
            report_path = self.result_root / model / "report.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(json.dumps({"model": model}), encoding="utf-8")

    def test_splits_total_ffmpeg_threads_evenly_between_model_workers(self) -> None:
        self._write_reports()
        assigned_threads: dict[str, int] = {}

        def fake_analyze(model, report_path, *, input_path, ffmpeg_threads):
            assigned_threads[model] = ffmpeg_threads
            return {"model": model, "benchmark_result": "PASS"}

        with mock.patch.object(analyze_results, "analyze_one", side_effect=fake_analyze):
            analyze_results.analyze_reports_concurrently(
                result_root=self.result_root,
                input_path=Path("input.mp4"),
                total_ffmpeg_threads=22,
                analysis_workers=2,
            )

        self.assertEqual(
            assigned_threads,
            {"FlashVSR_v1.1": 11, "SeedVR2": 11},
        )

    def test_runs_both_model_analyses_concurrently_and_returns_canonical_order(self) -> None:
        self._write_reports()
        rendezvous = threading.Barrier(2, timeout=2)

        def fake_analyze(model, report_path, *, input_path, ffmpeg_threads):
            rendezvous.wait()
            return {"model": model, "benchmark_result": "PASS"}

        with mock.patch.object(analyze_results, "analyze_one", side_effect=fake_analyze):
            reports = analyze_results.analyze_reports_concurrently(
                result_root=self.result_root,
                input_path=Path("input.mp4"),
                total_ffmpeg_threads=22,
                analysis_workers=2,
            )

        self.assertEqual(list(reports), list(analyze_results.MODELS))

    def test_worker_failure_is_written_without_suppressing_other_model_result(self) -> None:
        self._write_reports()

        def fake_analyze(model, report_path, *, input_path, ffmpeg_threads):
            if model == "FlashVSR_v1.1":
                raise RuntimeError("analysis failed")
            return {"model": model, "benchmark_result": "PASS"}

        with mock.patch.object(analyze_results, "analyze_one", side_effect=fake_analyze):
            reports = analyze_results.analyze_reports_concurrently(
                result_root=self.result_root,
                input_path=Path("input.mp4"),
                total_ffmpeg_threads=22,
                analysis_workers=2,
            )

        self.assertEqual(reports["FlashVSR_v1.1"]["benchmark_result"], "FAIL")
        self.assertEqual(reports["SeedVR2"]["benchmark_result"], "PASS")
        self.assertIn("RuntimeError: analysis failed", reports["FlashVSR_v1.1"]["analysis_error"])
        self.assertTrue((self.result_root / "FlashVSR_v1.1" / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
