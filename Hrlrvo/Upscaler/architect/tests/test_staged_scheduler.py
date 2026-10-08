from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path


ARCHITECT = Path(__file__).resolve().parents[1]
if str(ARCHITECT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT))

from orchestrate import _gpu_release_threshold_mb, _staged_cpu_plan, _validate_staged_manifest


class StagedSchedulerTests(unittest.TestCase):
    def test_tuned_23_core_plan(self) -> None:
        self.assertEqual(
            _staged_cpu_plan(23),
            {
                "available_cores": 23,
                "reserved_cores": 1,
                "flash_gpu_threads": 22,
                "flash_encode_threads": 12,
                "seed_threads": 18,
                "analysis_workers": 2,
                "analysis_threads_per_worker": 11,
            },
        )

    def test_small_hosts_keep_every_stage_runnable(self) -> None:
        one = _staged_cpu_plan(1)
        self.assertEqual(one["reserved_cores"], 0)
        self.assertEqual(one["flash_gpu_threads"], 1)
        self.assertEqual(one["flash_encode_threads"], 1)
        self.assertEqual(one["seed_threads"], 1)
        self.assertEqual(one["analysis_workers"], 1)
        self.assertEqual(one["analysis_threads_per_worker"], 1)

    def test_gpu_release_gate_adds_fixed_guard(self) -> None:
        self.assertEqual(_gpu_release_threshold_mb(142_318.0), 142_830.0)
        self.assertEqual(_gpu_release_threshold_mb(100.0, guard_mb=-1.0), 100.0)

    def test_accepts_completed_wrapper_manifest_with_staged_partial_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage.json"
            payload = {"state": "COMPLETE", "partial_report": {"status": "STAGED"}}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_validate_staged_manifest(path), payload)


if __name__ == "__main__":
    unittest.main()
