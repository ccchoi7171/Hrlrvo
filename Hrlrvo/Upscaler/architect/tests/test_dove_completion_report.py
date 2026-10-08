from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ARCHITECT = Path(__file__).resolve().parents[1]
if str(ARCHITECT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT))

from report_dove_completion import build_payload, render, render_summary_addendum


class DOVECompletionReportTest(unittest.TestCase):
    def test_report_includes_dove_measurement_and_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "DOVE").mkdir()
            (root / "run_status_dove.json").write_text(
                json.dumps({"status": "PASS", "runs": []}), encoding="utf-8"
            )
            (root / "benchmark_estimates.json").write_text(
                json.dumps({
                    "models": {
                        "DOVE": {
                            "estimates": {
                                "92": {
                                    "estimated_total_seconds": 123.0,
                                    "estimated_total_hours": 123.0 / 3600,
                                    "temporal_chunks": 10,
                                }
                            }
                        }
                    }
                }),
                encoding="utf-8",
            )
            (root / "DOVE" / "benchmark_micro.json").write_text(
                json.dumps({
                    "status": "PASS",
                    "technical_validation": {"status": "PASS"},
                    "effective_output_frames_written": 9,
                    "timing": {
                        "total_wall_seconds": 90,
                        "pure_inference_seconds_per_output_frame": 8,
                        "end_to_end_seconds_per_output_frame": 10,
                    },
                    "gpu": {"peak_process_gpu_memory_mb": 12000},
                    "structure_analysis": {"mean_ssim": 0.9, "mean_psnr_db": 30},
                    "temporal_analysis": {"status": "PASS"},
                    "output_path": "/output/dove.mp4",
                }),
                encoding="utf-8",
            )
            payload = build_payload(root, 0)
            markdown = render(payload)
            addendum = render_summary_addendum(payload)
            self.assertEqual(payload["launcher_status"], "PASS")
            self.assertIn("DOVE | micro | 9", markdown)
            self.assertIn("| 92 | 123.000", markdown)
            self.assertIn("/output/dove.mp4", markdown)
            self.assertIn("DOVE Stage-2 추가 benchmark", addendum)
            self.assertIn("92 frames: 123.0 s", addendum)


if __name__ == "__main__":
    unittest.main()
