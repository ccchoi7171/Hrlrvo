from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_batch import run_batch  # noqa: E402


class RecordingRunner:
    def __init__(self, failures: set[tuple[str, str]] | None = None) -> None:
        self.failures = failures or set()
        self.events: list[tuple[str, str, str, float]] = []
        self.active = {"outpaint": 0, "upscale": 0}
        self.maximum = {"outpaint": 0, "upscale": 0}
        self.lock = threading.Lock()

    def __call__(self, stage, job, resume, emit):
        del resume, emit
        with self.lock:
            self.active[stage] += 1
            self.maximum[stage] = max(self.maximum[stage], self.active[stage])
            self.events.append(("start", stage, job.video_id, time.monotonic()))
        time.sleep(0.04 if stage == "outpaint" else 0.09)
        with self.lock:
            self.events.append(("end", stage, job.video_id, time.monotonic()))
            self.active[stage] -= 1
        return 1 if (stage, job.video_id) in self.failures else 0


class BatchQueueTests(unittest.TestCase):
    def make_workspace(self, directory: str, count: int = 3):
        root = Path(directory)
        (root / "configs" / "certification").mkdir(parents=True)
        jobs = []
        for index in range(1, count + 1):
            video_id = f"{index:02d}"
            relative = f"configs/certification/{video_id}.json"
            (root / relative).write_text(
                json.dumps({"job_name": f"{video_id}_certification", "input": {"video_id": video_id}}),
                encoding="utf-8",
            )
            jobs.append(relative)
        batch = root / "batch.json"
        batch.write_text(
            json.dumps(
                {
                    "jobs": jobs,
                    "outpaint_workers": 1,
                    "upscale_workers": 1,
                    "queue_policy": "fifo",
                }
            ),
            encoding="utf-8",
        )
        return root, batch

    def test_workers_overlap_but_each_stage_has_single_concurrency_and_fifo(self):
        with TemporaryDirectory() as directory:
            root, batch = self.make_workspace(directory)
            runner = RecordingRunner()
            result = run_batch(batch, root=root, child_runner=runner)

            self.assertTrue(result.succeeded)
            self.assertEqual(runner.maximum, {"outpaint": 1, "upscale": 1})
            upscale_order = [event[2] for event in runner.events if event[:2] == ("start", "upscale")]
            self.assertEqual(upscale_order, ["01", "02", "03"])
            starts = {(stage, video): at for action, stage, video, at in runner.events if action == "start"}
            ends = {(stage, video): at for action, stage, video, at in runner.events if action == "end"}
            self.assertLess(starts[("outpaint", "02")], ends[("upscale", "01")])

    def test_outpaint_failure_is_not_enqueued_and_other_jobs_continue(self):
        with TemporaryDirectory() as directory:
            root, batch = self.make_workspace(directory)
            runner = RecordingRunner({("outpaint", "02")})
            result = run_batch(batch, root=root, child_runner=runner)

            self.assertEqual(result.states["01"], "DONE")
            self.assertEqual(result.states["02"], "FAILED_OUTPAINT")
            self.assertEqual(result.states["03"], "DONE")
            upscale_videos = [event[2] for event in runner.events if event[:2] == ("start", "upscale")]
            self.assertEqual(upscale_videos, ["01", "03"])

    def test_upscale_failure_continues_fifo_and_atomic_state_is_valid(self):
        with TemporaryDirectory() as directory:
            root, batch = self.make_workspace(directory)
            runner = RecordingRunner({("upscale", "01")})
            result = run_batch(batch, root=root, child_runner=runner)

            self.assertEqual(result.states, {"01": "FAILED_UPSCALE", "02": "DONE", "03": "DONE"})
            persisted = json.loads((root / "state" / "batch_state.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted, result.states)
            self.assertFalse(list((root / "state").glob(".batch_state.json.*.tmp")))

    def test_child_exception_becomes_failure_and_does_not_deadlock_queue(self):
        with TemporaryDirectory() as directory:
            root, batch = self.make_workspace(directory, count=2)
            delegate = RecordingRunner()

            def raising_runner(stage, job, resume, emit):
                if stage == "outpaint" and job.video_id == "01":
                    raise RuntimeError("launcher failed")
                return delegate(stage, job, resume, emit)

            result = run_batch(batch, root=root, child_runner=raising_runner)

            self.assertEqual(result.states["01"], "FAILED_OUTPAINT")
            self.assertEqual(result.states["02"], "DONE")

    def write_request_binding(self, root, batch):
        import run_batch

        jobs = run_batch.load_jobs(root, batch)
        request = run_batch._batch_request(root, batch, jobs)
        (root / "state").mkdir(exist_ok=True)
        (root / "state" / "batch_request.json").write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def test_resume_revalidates_outpaint_done_and_done_through_upscale_only(self):
        with TemporaryDirectory() as directory:
            root, batch = self.make_workspace(directory, count=2)
            (root / "state").mkdir()
            (root / "state" / "batch_state.json").write_text(
                json.dumps({"01": "OUTPAINT_DONE", "02": "DONE"}), encoding="utf-8"
            )
            self.write_request_binding(root, batch)
            runner = RecordingRunner()
            result = run_batch(batch, root=root, child_runner=runner, resume=True)

            self.assertTrue(result.succeeded)
            stages = [(event[1], event[2]) for event in runner.events if event[0] == "start"]
            self.assertEqual(stages, [("upscale", "01"), ("upscale", "02")])

    def test_resume_skip_set_is_not_racy_with_consumer_state_transition(self):
        with TemporaryDirectory() as directory:
            root, batch = self.make_workspace(directory, count=1)
            (root / "state").mkdir()
            (root / "state" / "batch_state.json").write_text(
                json.dumps({"01": "UPSCALE_WAITING"}), encoding="utf-8"
            )
            self.write_request_binding(root, batch)
            runner = RecordingRunner()
            result = run_batch(batch, root=root, child_runner=runner, resume=True)

            self.assertTrue(result.succeeded)
            starts = [(event[1], event[2]) for event in runner.events if event[0] == "start"]
            self.assertEqual(starts, [("upscale", "01")])

    def test_fresh_run_refuses_existing_state_and_resume_binds_config_bytes(self):
        with TemporaryDirectory() as directory:
            root, batch = self.make_workspace(directory, count=1)
            runner = RecordingRunner()
            self.assertTrue(run_batch(batch, root=root, child_runner=runner).succeeded)
            with self.assertRaisesRegex(ValueError, "requires --resume"):
                run_batch(batch, root=root, child_runner=runner)
            job_config = root / "configs" / "certification" / "01.json"
            job_config.write_text(
                json.dumps({"job_name": "changed", "input": {"video_id": "01"}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "config bytes changed"):
                run_batch(batch, root=root, child_runner=runner, resume=True)

    def test_rejects_job_config_outside_root_and_unsafe_video_id(self):
        with TemporaryDirectory() as directory, TemporaryDirectory() as outside:
            root, batch = self.make_workspace(directory, count=1)
            external = Path(outside) / "job.json"
            external.write_text(
                json.dumps({"job_name": "external", "input": {"video_id": "01"}}), encoding="utf-8"
            )
            batch.write_text(
                json.dumps(
                    {
                        "jobs": [str(external)],
                        "outpaint_workers": 1,
                        "upscale_workers": 1,
                        "queue_policy": "fifo",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inside pipeline root"):
                run_batch(batch, root=root, child_runner=RecordingRunner())

            unsafe = root / "configs" / "certification" / "unsafe.json"
            unsafe.write_text(
                json.dumps({"job_name": "unsafe", "input": {"video_id": "../05"}}), encoding="utf-8"
            )
            batch.write_text(
                json.dumps(
                    {
                        "jobs": [str(unsafe.relative_to(root))],
                        "outpaint_workers": 1,
                        "upscale_workers": 1,
                        "queue_policy": "fifo",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "safe state label"):
                run_batch(batch, root=root, child_runner=RecordingRunner())


if __name__ == "__main__":
    unittest.main()
