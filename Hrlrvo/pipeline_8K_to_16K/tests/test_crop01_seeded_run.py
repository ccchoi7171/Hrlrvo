"""CPU-only checks for the isolated cropped-01 stage request."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "crop01_seeded_runner_test", ROOT / "tools/run_crop01_seed33333333.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Cropped01RunTests(unittest.TestCase):
    def test_only_preprocess_then_outpaint_with_approved_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            base = {
                "job_name": "01_certification",
                "output_root": "state/runs/01_certification",
                "seed": 50374151,
                "frames": {"outpaint": "all", "upscale": "all"},
            }
            expected = dict(base, job_name="01_crop_seed33333333",
                            output_root="state/runs/01_crop_seed33333333", seed=33333333)
            request = temp / "request.json"
            request.write_text(json.dumps(expected), encoding="utf-8")
            calls = []

            def preprocess(config, *, resume):
                calls.append(("preprocess", config, resume))
                return temp / "preprocessed.mp4"

            def outpaint(config, *, resume):
                calls.append(("outpaint", config, resume))
                return temp / "outpaint.mp4"

            with (
                patch.object(runner, "RUN", temp / "run"),
                patch.object(runner, "REQUEST", request),
                patch.object(runner, "SOURCE", temp / "input.mp4"),
                patch.object(runner, "SOURCE_SHA256", "expected-sha"),
                patch.object(runner, "validate_config", return_value=(base, None, None, None, None)),
                patch.object(runner, "sha256_file", return_value="expected-sha"),
                patch.object(runner, "artifact_path", return_value=temp / "outpaint-target"),
                patch.object(runner, "run_preprocess_stage", side_effect=preprocess),
                patch.object(runner, "run_outpaint_stage", side_effect=outpaint),
                patch.object(sys, "argv", ["run_crop01_seed33333333.py"]),
            ):
                self.assertEqual(runner.main(), 0)

            self.assertEqual([name for name, _, _ in calls], ["preprocess", "outpaint"])
            self.assertTrue(all(config == expected and not resume for _, config, resume in calls))
            self.assertEqual(
                json.loads((temp / "run/requested_config.json").read_text()), expected
            )

    def test_resume_refuses_changed_request(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            run = temp / "run"
            run.mkdir()
            (run / "requested_config.json").write_text("{}", encoding="utf-8")
            request = temp / "request.json"
            request.write_text(json.dumps({"seed": 33333333,
                                           "job_name": "01_crop_seed33333333",
                                           "output_root": "state/runs/01_crop_seed33333333"}),
                               encoding="utf-8")
            base = {"seed": 50374151, "job_name": "01_certification",
                    "output_root": "state/runs/01_certification"}
            with (
                patch.object(runner, "RUN", run),
                patch.object(runner, "REQUEST", request),
                patch.object(runner, "SOURCE", temp / "input.mp4"),
                patch.object(runner, "SOURCE_SHA256", "expected-sha"),
                patch.object(runner, "validate_config", return_value=(base, None, None, None, None)),
                patch.object(runner, "sha256_file", return_value="expected-sha"),
                patch.object(sys, "argv", ["run_crop01_seed33333333.py", "--resume"]),
                patch.object(runner, "run_preprocess_stage") as preprocess,
            ):
                with self.assertRaisesRegex(RuntimeError, "Resume request"):
                    runner.main()
                preprocess.assert_not_called()


if __name__ == "__main__":
    unittest.main()
