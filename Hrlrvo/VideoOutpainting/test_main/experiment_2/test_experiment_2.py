from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load("experiment2_prep", HERE / "prepare_clean_anchor_gpu.py")
runner = load("experiment2_runner", HERE / "run_full_phase_rebase.py")


class Experiment2Tests(unittest.TestCase):
    def test_masked_normalized_donor_excludes_foreground_and_lighting_offset(self):
        samples = torch.full((5, 4, 4, 3), 40.0)
        for index in range(5):
            samples[index] += index * 5
        samples[2, 1, 1] = torch.tensor([255.0, 0.0, 0.0])
        invalid = torch.zeros((5, 4, 4), dtype=torch.bool)
        invalid[2, 1, 1] = True
        donor, count, mad = prep.masked_normalized_donor(samples, invalid)
        self.assertTrue(torch.allclose(donor[1, 1], torch.tensor([50.0, 50.0, 50.0])))
        self.assertEqual(int(count[1, 1]), 4)
        self.assertLess(float(mad.max()), 1e-5)

    def test_donor_fallback_records_zero_support_without_foreground_ghost(self):
        samples = torch.tensor(
            [[[[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]], [[[12.0, 12.0, 12.0], [200.0, 0.0, 0.0]]]]
        )
        invalid = torch.tensor([[[False, True]], [[False, True]]])
        fallback = torch.full((1, 2, 3), 33.0)
        donor, count, _ = prep.masked_normalized_donor(samples, invalid, fallback)
        self.assertEqual(int(count[0, 1]), 0)
        self.assertTrue(torch.isfinite(donor).all())
        self.assertTrue(torch.equal(donor[0, 1], torch.tensor([33.0, 33.0, 33.0])))

    def test_common_snapshot_is_current_and_exact(self):
        observed = runner.validate_common_snapshot()
        self.assertEqual(observed["latent_prefix_carry_phase_aligned.py"], "651411cb848193068b78b382d87318020b7563bf05d78f5f10ba0c78b3717261")
        self.assertEqual(len(observed), len(json.loads(runner.SNAPSHOT.read_text())["files"]))

    def test_resume_reconciliation_preserves_committed_and_archives_orphan(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows = [{"chunk_index": index} for index in range(3)]
            (folder / "phase_aligned_latent_carry_lineage.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (folder / "phase_emission_lineage.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            for index in range(3):
                (folder / f"phase_tail_chunk_{index:03d}_bf16.pt").write_bytes(str(index).encode())
            result = runner.reconcile_resume(folder, 2)
            kept = (folder / "phase_aligned_latent_carry_lineage.jsonl").read_text().splitlines()
            self.assertEqual(len(kept), 2)
            self.assertTrue((folder / "phase_tail_chunk_001_bf16.pt").exists())
            self.assertFalse((folder / "phase_tail_chunk_002_bf16.pt").exists())
            self.assertEqual(len(result["orphan_tail_files"]), 1)

    def test_initial_run_does_not_require_resume_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            segment_dir = Path(directory)
            self.assertFalse(runner.resume_enabled(segment_dir, 0))
            with self.assertRaisesRegex(FileNotFoundError, "resume metadata"):
                runner.resume_enabled(segment_dir, 1)
            (segment_dir / "run.json").write_text("{}")
            self.assertTrue(runner.resume_enabled(segment_dir, 0))
            self.assertTrue(runner.resume_enabled(segment_dir, 1))

    def test_validate_job_requires_clean_gpu_provenance_and_no_raw_reinsert(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "clean.mkv"; input_path.write_bytes(b"clean")
            manifest = root / "manifest.json"
            config = HERE / "native_65f_resume.yaml"
            preparation = {
                "status": "PASS_GPU_CLEAN_ANCHOR_READY",
                "clean_anchor": {"sha256": runner.digest(input_path)},
                "person_segmentation": {"dense_all_frames": True},
                "cleanup": {"gpu_donor_fill": True},
                "original_RAW_anchor_post_reinsertion": False,
            }
            manifest.write_text(json.dumps(preparation))
            job = {
                "name": "experiment_2_phase_rebase_full1025", "experiment": 2,
                "backend": "native", "seed": 33333333, "steps": 30, "frames": 1025,
                "recurrence": {"reduced_geometry": True, "temporal_rebase": True},
                "attention": {"collect_branch_stats": False}, "prompt": "",
                "restore_original_anchor_after_inference": False,
                "requires_inverse_registration": False,
                "latent_carry_mode": runner.MODE,
                "producer_script": "run_full_phase_rebase.py",
                "source_frame_start": 0, "source_frame_end_inclusive": 1024,
                "memory_fraction": 0.49, "input": str(input_path),
                "input_sha256": runner.digest(input_path), "config": str(config),
                "checkpoint": str(root), "clean_preparation_manifest": str(manifest),
                "clean_preparation_manifest_sha256": runner.digest(manifest),
            }
            job_path = root / "job.json"; job_path.write_text(json.dumps(job))
            self.assertEqual(runner.validate_job(job, job_path)["job_sha256"], runner.digest(job_path))
            job["restore_original_anchor_after_inference"] = True
            with self.assertRaisesRegex(ValueError, "restore_original"):
                runner.validate_job(job, job_path)

    def test_resume_aggregate_accepts_all_committed_chunks_and_rejects_seed_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            segment_dir = Path(directory)
            plan = []
            emissions = []
            for index in range(3):
                start = index * 48
                output_range = [0, 65] if index == 0 else [65 + (index - 1) * 48, 65 + index * 48]
                plan.append({"new_output_frame_range": output_range})
                event = {"chunk": index, "action": "temporal_rebase", "seconds_removed": start / 24}
                emissions.append({"chunk_index": index, "absolute_start_frame": start, "chunk_seed": 33333333 + start, "chronological_prefix_verified": True, "temporal_rebase_event": event})
                marker = {"chunk_seed": 33333333 + start, "absolute_start_frame": start, "new_output_frame_range": output_range, "prefix_output_frame_range": None if index == 0 else [start, start + 17]}
                (segment_dir / f"chunk_{index:06d}.json").write_text(json.dumps(marker))
            self.assertEqual(runner.validate_committed_evidence(plan, emissions, segment_dir), 3)
            emissions[1]["chunk_seed"] += 1
            with self.assertRaisesRegex(RuntimeError, "chunk 1"):
                runner.validate_committed_evidence(plan, emissions, segment_dir)

    def test_prepare_task_is_batched_and_run_task_waits_for_clean_input(self):
        prepare_task = json.loads((HERE / "prepare_task.json").read_text())
        run_task = json.loads((HERE / "run_task.json").read_text())
        self.assertEqual(prepare_task["argv"][-2:], ["--batch-size", "8"])
        self.assertLessEqual(prepare_task["reservation_mib"], 24000)
        self.assertEqual(run_task["reservation_mib"], 50000)
        self.assertIn(str(HERE / "clean_anchor_1025f.mkv"), run_task["requires"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
