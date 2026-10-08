"""CPU-only orchestration regressions; all media/model operations mocked."""

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "migration_runner_test", ROOT / "run_cert.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DispatchTests(unittest.TestCase):
    def test_native_recurrence_validation_uses_same_stage_and_cannot_upscale(self):
        config = {"validation": {"native_outpaint_only": True}}
        with patch.object(runner, "run_outpaint_stage", return_value=Path("native24")) as call:
            self.assertEqual(runner.execute_stages(config, "outpaint"), Path("native24"))
            call.assert_called_once_with(config, resume=False, force=False)
        for selection in ("all", "preprocess", "upscale"):
            with self.subTest(selection=selection), self.assertRaises(runner.ConfigError):
                runner.execute_stages(config, selection)
        with self.assertRaisesRegex(runner.ConfigError, "not FlashVSR"):
            runner.validate_outpaint_artifact(config)

    def test_import_does_not_load_torch(self):
        self.assertNotIn("torch", sys.modules)

    def test_full_and_stage_only_use_identical_functions(self):
        for selected, expected in [
            ("all", ["preprocess", "outpaint", "upscale"]),
            ("preprocess", ["preprocess"]),
            ("outpaint", ["outpaint"]),
            ("upscale", ["upscale"]),
        ]:
            calls = []

            def fake(name):
                def call(config, **options):
                    calls.append((name, config, options))
                    return Path(name)

                return call

            with contextlib.ExitStack() as stack:
                for name in ("preprocess", "outpaint", "upscale"):
                    stack.enter_context(
                        patch.object(runner, f"run_{name}_stage", fake(name))
                    )
                config = {"sentinel": "same-object"}
                runner.execute_stages(config, selected, resume=True)
            self.assertEqual([x[0] for x in calls], expected)
            self.assertTrue(all(x[1] is config for x in calls))

    def test_fresh_preprocess_never_reuses_existing_canonical(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "ROOT", Path(tmp)),
        ):
            root = Path(tmp)
            src = root / "data/source.mp4"
            src.parent.mkdir()
            src.write_bytes(b"input")
            dest = root / "data/pre.mp4"
            dest.write_bytes(b"old-exact-input")
            dest.with_suffix(".json").write_text(
                json.dumps({"output_sha256": runner.sha256_file(dest), "crf": 18})
            )
            cfg = {
                "input": {"video_id": "05", "path": "data/source.mp4"},
                "preprocess": runner.preprocess.PREPROCESS_CONTRACT,
                "artifacts": {
                    "preprocess": "data/pre.mp4",
                    "preprocess_metadata": "data/meta",
                },
            }

            def generate(source, output, metadata, log, expected, **kwargs):
                output.mkdir(parents=True)
                p = output / "05_PreProc_2K.mp4"
                p.write_bytes(b"new-historical-output")
                return p

            with (
                patch.object(runner, "preflight_source", return_value=1952),
                patch.object(
                    runner.preprocess,
                    "_source_identity",
                    return_value={"sha256": "original"},
                ),
                patch.object(
                    runner.preprocess, "run_preprocess", side_effect=generate
                ) as call,
            ):
                runner.run_preprocess_stage(cfg)
            call.assert_called_once()
            self.assertEqual(dest.read_bytes(), b"new-historical-output")
            archives = list((root / "data/meta/preserved").rglob("pre.mp4"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_bytes(), b"old-exact-input")

    def test_upscale_rejects_uncommitted_video(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "ROOT", Path(tmp)),
        ):
            stage = Path(tmp) / "data/out/final"
            stage.mkdir(parents=True)
            (stage / "05_Outpaint_4K.mp4").write_bytes(b"not-certified")
            cfg = {"input": {"video_id": "05"}, "artifacts": {"outpaint": "data/out"}}
            with self.assertRaisesRegex(runner.ConfigError, "committed"):
                runner.validate_outpaint_artifact(cfg)

    def test_lock_rejects_second_worker_same_stage(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "ROOT", Path(tmp)),
        ):
            with runner.stage_lock("outpaint"):
                with self.assertRaises(runner.ConfigError):
                    with runner.stage_lock("outpaint"):
                        pass
                with runner.stage_lock("upscale"):
                    pass

    def test_path_cannot_escape_package(self):
        with self.assertRaises(runner.ConfigError):
            runner.absolute_package_path("../pipeline/run_cert.py", must_exist=False)

    def test_certification_force_keeps_nested_quick(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "ROOT", Path(tmp)),
        ):
            stage = Path(tmp) / "data/out/05"
            quick = stage / "quick/final/quick.mp4"
            quick.parent.mkdir(parents=True)
            quick.write_bytes(b"quick-result")
            config = {
                key: {}
                for key in ("input", "seed", "frames", "preprocess", "outpainting")
            }
            self.assertEqual(runner.stage_artifact_files(stage, "outpaint"), [])
            runner.prepare_stage(stage, config, "outpaint", resume=False, force=False)
            (stage / "final/full.mp4").write_bytes(b"full-result")
            runner.prepare_stage(stage, config, "outpaint", resume=False, force=True)
            self.assertEqual(quick.read_bytes(), b"quick-result")
            self.assertFalse((stage / "final/full.mp4").exists())

    def test_artifact_layout_cannot_archive_inputs(self):
        config = json.loads((ROOT / "configs/certification/05.json").read_text())
        config["artifacts"]["outpaint"] = "data/00_Input_8K_7680x4320"
        with self.assertRaisesRegex(runner.ConfigError, "layout"):
            runner.validate_artifact_layout(config)

    def test_layout_placeholder_does_not_trip_native_empty_store_guard(self):
        for resume in (False, True):
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.object(runner, "ROOT", Path(tmp)),
            ):
                stage = Path(tmp) / "data/out"
                native = stage / "chunks/native"
                native.mkdir(parents=True)
                (native / ".gitkeep").touch()
                config = {
                    key: {}
                    for key in ("input", "seed", "frames", "preprocess", "outpainting")
                }
                runner.prepare_stage(
                    stage, config, "outpaint", resume=resume, force=False
                )
                self.assertEqual(list(native.iterdir()), [])
                self.assertTrue((stage / "metadata/stage_request.json").is_file())

    def test_cached_preproc_wrong_original_sha_rejected(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "ROOT", Path(tmp)),
        ):
            root = Path(tmp)
            path = root / "data/pre.mp4"
            path.parent.mkdir()
            path.write_bytes(b"pre")
            source = root / "data/05_Input_8K.mp4"
            source.write_bytes(b"original")
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "output_sha256": runner.sha256_file(path),
                        "input_sha256": "wrong-source",
                        "producer": "run_preprocess_stage",
                        "preprocess_config": runner.preprocess.PREPROCESS_CONTRACT,
                        "producer_identity": {
                            "contract": runner.preprocess.PREPROCESS_CONTRACT
                        },
                    }
                )
            )
            config = {
                "input": {"video_id": "05", "path": "data/05_Input_8K.mp4"},
                "artifacts": {"preprocess": "data/pre.mp4"},
            }
            probe = {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "24/1",
                        "nb_read_frames": "65",
                        "pix_fmt": "yuv420p10le",
                    }
                ]
            }
            with (
                patch.object(runner.preprocess, "probe_media", return_value=probe),
                patch.object(
                    runner.preprocess,
                    "producer_identity",
                    return_value={"contract": runner.preprocess.PREPROCESS_CONTRACT},
                ),
            ):
                with self.assertRaisesRegex(runner.ConfigError, "source bytes"):
                    runner.validate_preprocessed(config)

    def test_preproc_audio_expectation_comes_from_original(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "ROOT", Path(tmp)),
        ):
            root = Path(tmp)
            source = root / "data/05_Input_8K.mp4"
            source.parent.mkdir()
            source.write_bytes(b"source")
            output = root / "data/pre.mp4"
            output.write_bytes(b"pre")
            output.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "input": str(source),
                        "input_sha256": runner.sha256_file(source),
                        "output_sha256": runner.sha256_file(output),
                        "frames": 65,
                        "producer": "run_preprocess_stage",
                        "preprocess_config": runner.preprocess.PREPROCESS_CONTRACT,
                        "producer_identity": {
                            "contract": runner.preprocess.PREPROCESS_CONTRACT
                        },
                    }
                )
            )
            config = {
                "input": {"video_id": "05", "path": "data/05_Input_8K.mp4"},
                "artifacts": {"preprocess": "data/pre.mp4"},
            }
            video = {
                "codec_type": "video",
                "nb_frames": "65",
                "pix_fmt": "yuv420p10le",
                "codec_name": "hevc",
                "color_range": "pc",
                "color_space": "bt709",
                "time_base": "1/24576",
            }
            pre = {
                "streams": [dict(video, width=1920, height=1080, avg_frame_rate="24/1")]
            }
            original = {
                "streams": [
                    dict(video, width=7680, height=4320, avg_frame_rate="16/1"),
                    {"codec_type": "audio"},
                ]
            }
            with (
                patch.object(runner.preprocess, "probe_media", return_value=pre),
                patch.object(
                    runner.preprocess,
                    "producer_identity",
                    return_value={"contract": runner.preprocess.PREPROCESS_CONTRACT},
                ),
                patch.object(
                    runner.subprocess, "check_output", return_value=json.dumps(original)
                ),
            ):
                with self.assertRaisesRegex(runner.preprocess.PreprocessError, "audio"):
                    runner.validate_preprocessed(config)

    def test_resolved_snapshot_is_available_before_model_call(self):
        config = json.loads((ROOT / "configs/quick/05.json").read_text())
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "ROOT", Path(tmp)),
        ):
            runner.snapshot_resolved_frames(config, 1952, "outpaint")
            record = json.loads(
                (Path(tmp) / config["output_root"] / "resolved_config.json").read_text()
            )
            self.assertEqual(
                record["frames"],
                {"source": 1952, "outpaint": 65, "upscale": 21, "final": 21},
            )
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
