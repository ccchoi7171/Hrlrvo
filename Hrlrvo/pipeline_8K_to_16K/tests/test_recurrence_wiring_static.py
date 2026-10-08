"""Final read-only wiring checks: AST/config inspection and mocked children only."""

import ast
import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "src/video_outpainting/main/src/ltx_outpaint"
TREE = ast.parse((NATIVE / "inference.py").read_text())
FUNCTIONS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}


def calls(node, name):
    return sorted([n for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, (ast.Name, ast.Attribute))
            and (n.func.id if isinstance(n.func, ast.Name) else n.func.attr) == name],
            key=lambda n: n.lineno)


def evaluate(expression, scope):
    return eval(compile(ast.Expression(expression), "<source-expression>", "eval"), scope)


class RecurrenceWiringStaticTests(unittest.TestCase):
    def test_every_c1_plus_iteration_passes_enhanced_flags_to_same_sampler(self):
        function = FUNCTIONS["sample_long_range"]
        sample_calls = calls(function, "sample_chunk")
        self.assertEqual(len(sample_calls), 1)
        sample = sample_calls[0]
        self.assertTrue(any(sample in list(ast.walk(n)) for n in ast.walk(function)
                            if isinstance(n, ast.For)))
        enhanced = next(n.value for n in ast.walk(function)
                        if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "enhanced" for t in n.targets))
        keywords = {k.arg: k.value for k in sample.keywords}
        recurrence = json.loads((ROOT / "configs/certification/05.json").read_text())["outpainting"]["recurrence"]
        sentinel = object()
        for index in (0, 1, 2, 3, 40, 1000):
            scope = {"record": {"chunk_index": index}, "recurrence": recurrence,
                     "previous_carry": sentinel}
            scope["enhanced"] = evaluate(enhanced, scope)
            self.assertEqual(scope["enhanced"], index > 0)
            self.assertIs(evaluate(keywords["rebase_temporal_position"], scope), index > 0)
            self.assertIs(evaluate(keywords["prefix_latent_override"], scope), sentinel if index else None)
            self.assertEqual(evaluate(keywords["generated_prefix_strength"], scope), 1.0)
        scope_expression = next(n.value for n in ast.walk(function)
                                if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == "scope" for t in n.targets))
        prior = SimpleNamespace(chunk_scope=lambda index, **kw: (index, kw))
        for index in (0, 1, 2, 40):
            value = evaluate(scope_expression, {"record": {"chunk_index": index},
                             "enhanced": index > 0, "lineart_prior": prior,
                             "nullcontext": contextlib.nullcontext})
            if index:
                self.assertEqual(value, (index, {"prefix_latent_frames": 3}))
            else:
                self.assertIsInstance(value, contextlib.nullcontext)

    def test_c0_legacy_time_position_operation_and_c1_rebase(self):
        sample = FUNCTIONS["sample_chunk"]
        branch = next(n for n in ast.walk(sample) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "absolute_start_frame == 0 or not rebase_temporal_position")
        for frame, enabled, apply_offset in ((0, False, True), (0, True, True),
                                             (48, True, False), (96, True, False), (48, False, True)):
            self.assertIs(evaluate(branch.test, {"absolute_start_frame": frame,
                                               "rebase_temporal_position": enabled}), apply_offset)
        self.assertIn("int(runner._config.seed) + absolute_start_frame", ast.unparse(sample))

    def test_carry_after_denoise_before_commit_and_resume_before_next_sample(self):
        sample, long = FUNCTIONS["sample_chunk"], FUNCTIONS["sample_long_range"]
        self.assertLess(calls(sample, "_run_denoising")[0].lineno,
                        calls(sample, "clear_conditioning")[0].lineno)
        self.assertLess(calls(long, "save_carry")[0].lineno,
                        calls(long, "commit_chunk")[-1].lineno)
        resume_loads = calls(long, "load_carry")
        self.assertEqual(len(resume_loads), 2)
        self.assertLess(resume_loads[0].lineno, calls(long, "_load_models")[0].lineno)
        self.assertLess(resume_loads[-1].lineno, calls(long, "sample_chunk")[0].lineno)
        text = ast.unparse(long)
        self.assertIn("previous_carry = load_carry", text)
        self.assertIn("previous_prefix = resume_state.previous_prefix", text)
        self.assertIn("_restore_post_chunk_rng", text)

    def test_lineart_created_after_c0_commit_and_bound_to_own_source(self):
        long = FUNCTIONS["sample_long_range"]
        prepare = next(n for n in long.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_lineart")
        source_call = calls(prepare, "reference_from_segment")[0]
        self.assertEqual(ast.unparse(source_call.args[1]), "store.root / c0_record['segment_file']")
        source = ast.unparse(prepare)
        for text in ("input_sha256", "adapter_sha256", "c0_segment_pixel_sha256", "validation_config.seed"):
            self.assertIn(text, source)
        after_commit = [n for n in calls(long, "prepare_lineart")
                        if n.lineno > calls(long, "commit_chunk")[-1].lineno]
        self.assertEqual(len(after_commit), 1)
        guard = next(n for n in ast.walk(long) if isinstance(n, ast.If)
                     and after_commit[0] in list(ast.walk(n))
                     and "chunk_index" in ast.unparse(n.test))
        self.assertEqual(ast.unparse(guard.test), "int(record['chunk_index']) == 0 and recurrence['lineart_global_prior']")
        stage_roots = []
        for index in range(1, 6):
            config = json.loads((ROOT / f"configs/certification/{index:02d}.json").read_text())
            stage_roots.append(config["artifacts"]["outpaint"])
            self.assertTrue(stage_roots[-1].endswith(f"/{index:02d}"))
        self.assertEqual(len(set(stage_roots)), 5)

    def test_only_global_attention_calls_bias_helpers(self):
        users = []
        for path in NATIVE.glob("*.py"):
            if path.name == "lineart_global_prior.py":
                continue
            tree = ast.parse(path.read_text())
            if calls(tree, "build_attention_bias") or calls(tree, "build_fine_write_mask"):
                users.append(path.name)
        self.assertEqual(users, ["global_attention.py"])
        source = (NATIVE / "global_attention.py").read_text()
        self.assertIn("return torch.where(fine_write, selected_full, original_full)", source)

    def test_native_merge_timing_and_single_final_postprocess(self):
        long = FUNCTIONS["sample_long_range"]
        encode = calls(long, "encode_segments")
        self.assertEqual(len(encode), 1)
        self.assertEqual(ast.unparse(next(k.value for k in encode[0].keywords if k.arg == "fps")), "expected_fps")
        self.assertFalse(calls(long, "_retime_to_16fps"))
        self.assertLess(calls(long, "commit_chunk")[-1].lineno, encode[0].lineno)
        wrapper = (ROOT / "src/cert_pipeline/outpaint.py").read_text()
        self.assertIn('settings["video"]["fps"] = 24.0', wrapper)

    def test_actual_full_batch_config_and_child_entrypoint_without_launch(self):
        spec = importlib.util.spec_from_file_location("final_batch_static", ROOT / "run_batch.py")
        batch = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = batch
        spec.loader.exec_module(batch)
        jobs = batch.load_jobs(ROOT, ROOT / "configs/batch/certification.json")
        self.assertEqual([j.video_id for j in jobs], ["05", "04", "03", "02", "01"])
        self.assertEqual([json.loads(j.config_path.read_text())["seed"] for j in jobs],
                         [71634910, 71634910, 33333333, 25621683, 50374151])
        for job in jobs:
            self.assertEqual(json.loads(job.config_path.read_text())["frames"],
                             {"outpaint": "all", "upscale": "all"})
        for stage in ("outpaint", "upscale"):
            with patch.object(batch.subprocess, "Popen") as popen:
                popen.return_value.stdout = iter(())
                popen.return_value.wait.return_value = 0
                self.assertEqual(batch._production_child_runner(ROOT)(stage, jobs[0], False, lambda _: None), 0)
                self.assertEqual(popen.call_args.args[0],
                                 [str(ROOT / "run_cert.sh"), "--config", str(jobs[0].config_path), "--stage", stage])
        # Exercise the real producer/consumer with fake children and copied configs.
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            for path in [ROOT / "configs/batch/certification.json", *(j.config_path for j in jobs)]:
                target = workspace / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            events = []
            def child(stage, job, resume, emit):
                events.append((stage, job.video_id))
                return 0
            with patch.object(batch.subprocess, "Popen", side_effect=AssertionError("No real child allowed")):
                result = batch.run_batch(workspace / "configs/batch/certification.json",
                                         root=workspace, child_runner=child)
            self.assertTrue(result.succeeded)
            for stage in ("outpaint", "upscale"):
                self.assertEqual([v for s, v in events if s == stage], ["05", "04", "03", "02", "01"])


if __name__ == "__main__":
    unittest.main()
