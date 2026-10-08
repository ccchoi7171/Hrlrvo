"""CPU-only regressions for default final-only and optional legacy raw output."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("output_mode_runner_test", ROOT / "run_cert.py")
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from cert_pipeline import flash_composite as flash  # noqa: E402


class StopBeforeFinalEncode(Exception):
    pass


class OutputModeTests(unittest.TestCase):
    def test_legacy_profile_reuses_exact_outpaint_request_but_isolates_upscale(self):
        for video in ("01", "02", "03", "04", "05"):
            with self.subTest(video=video):
                cert = json.loads((ROOT / f"configs/certification/{video}.json").read_text())
                legacy = json.loads((ROOT / f"configs/legacy_raw/{video}.json").read_text())
                self.assertEqual(
                    runner.stage_identity(cert, "outpaint"),
                    runner.stage_identity(legacy, "outpaint"),
                )
                self.assertEqual(cert["artifacts"]["outpaint"], legacy["artifacts"]["outpaint"])
                self.assertNotEqual(cert["artifacts"]["upscale"], legacy["artifacts"]["upscale"])
                self.assertEqual(legacy["upscale"]["output_mode"], "legacy_raw")
                with self.assertRaisesRegex(runner.ConfigError, "select --stage upscale"):
                    runner.execute_stages(legacy, "all")

    def test_wrapper_default_and_explicit_legacy(self):
        base = [
            "--video-id", "05", "--input-4k", "input.mp4", "--original-8k", "anchor.mp4",
            "--model-root", "models", "--flashvsr-reference", "upscale.py",
            "--outpainting-config", "inference.yaml", "--output", "final.mp4",
            "--raw-output", "raw.mp4", "--final-manifest", "final.json",
            "--work-dir", "segments", "--expected-frames", "21",
        ]
        self.assertEqual(flash.parser().parse_args(base).output_mode, "final_only")
        self.assertEqual(
            flash.parser().parse_args(base + ["--output-mode", "legacy_raw"]).output_mode,
            "legacy_raw",
        )

    def test_header_probe_never_requests_count_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "final.mp4"
            video.write_bytes(b"nonempty")
            payload = {"streams": [{
                "width": 15360, "height": 8640, "avg_frame_rate": "16/1",
                "r_frame_rate": "16/1", "nb_frames": "21", "duration": "1.3125",
            }]}
            with patch.object(flash, "run", return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(payload), stderr=""
            )) as call:
                result = flash.probe_video_headers(video)
            self.assertNotIn("-count_frames", call.call_args.args[0])
            self.assertEqual(result["frame_count"], 21)
            self.assertIs(result["decoded_frame_count_performed"], False)

    def _arguments(self, root: Path, mode: str) -> argparse.Namespace:
        for name in ("input.mp4", "anchor.mp4", "inference.yaml", "upscale.py"):
            (root / name).write_bytes(b"input")
        model_root = root / "models"
        model_root.mkdir()
        return argparse.Namespace(
            video_id="05", input_4k=root / "input.mp4", original_8k=root / "anchor.mp4",
            outpainting_config=root / "inference.yaml", flashvsr_reference=root / "upscale.py",
            model_root=model_root, output=root / "final" / "final.mp4",
            raw_output=root / "raw" / "raw.mp4", final_manifest=root / "metadata" / "final.json",
            work_dir=root / "segments", expected_frames=21, ffmpeg_threads=2,
            encoder="libx265", output_mode=mode,
        )

    def test_final_only_skips_raw_encode_and_legacy_runs_it(self):
        for mode, expected_raw_calls in (("final_only", 0), ("legacy_raw", 1)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                args = self._arguments(Path(temporary), mode)
                paths = flash.stage_paths(args.work_dir)
                args.work_dir.mkdir()
                paths["request"].write_text(json.dumps({"request": 1}))
                paths["manifest"].write_text("{}")
                raw_calls = []

                def raw_encode(_options):
                    raw_calls.append(1)
                    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
                    args.raw_output.write_bytes(b"raw")
                    return {"output_path": str(args.raw_output)}

                reference = SimpleNamespace(
                    MODEL_ROOT=args.model_root, SCALE=4,
                    read_stage_manifest=lambda _path: {}, execute_encode_stage=raw_encode,
                )
                media = SimpleNamespace(select_encoder=lambda *_a, **_k: (
                    {"extension": ".mp4", "name": "libx265"}, []
                ))
                with (
                    patch.object(flash.shutil, "which", return_value="/usr/bin/tool"),
                    patch.object(flash, "required_assets", return_value=[]),
                    patch.object(flash, "read_outpainting_feather_width", return_value=16),
                    patch.object(flash, "probe_video", side_effect=[
                        {"width": 3840, "height": 2160, "frame_count": 21},
                        {"width": 7680, "height": 4320, "frame_count": 21,
                         "avg_frame_rate": "16/1", "r_frame_rate": "16/1"},
                    ]),
                    patch.object(flash, "validate_video_spec"),
                    patch.object(flash, "validate_frame_index_alignment"),
                    patch.object(flash, "stage_request_payload", return_value={"request": 1}),
                    patch.object(flash, "load_reference_module", return_value=reference),
                    patch.object(flash, "validate_stage_manifest", return_value=([], {}, args.work_dir)),
                    patch.object(flash, "validate_final_output"),
                    patch.object(flash, "load_media_utils", return_value=media),
                    patch.object(flash, "encode_staged_tiles", side_effect=StopBeforeFinalEncode),
                    self.assertRaises(StopBeforeFinalEncode),
                ):
                    flash.execute(args)
                self.assertEqual(len(raw_calls), expected_raw_calls)
                self.assertEqual(args.raw_output.exists(), mode == "legacy_raw")

    def test_final_only_mock_commit_has_one_final_encode_and_no_raw_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._arguments(Path(temporary), "final_only")
            paths = flash.stage_paths(args.work_dir)
            args.work_dir.mkdir()
            paths["request"].write_text(json.dumps({"request": 1}))
            paths["manifest"].write_text("{}")
            reference = SimpleNamespace(
                MODEL_ROOT=args.model_root, SCALE=4,
                read_stage_manifest=lambda _path: {"partial_report": {"configuration": {}}},
                execute_encode_stage=lambda _args: self.fail("raw encoder called"),
            )
            media = SimpleNamespace(
                select_encoder=lambda *_a, **_k: ({"extension": ".mp4", "name": "libx265"}, []),
                probe_media=lambda *_a, **_k: SimpleNamespace(audio_present=False),
            )

            def fake_final_encode(**kwargs):
                kwargs["partial_output"].write_bytes(b"mock-final")
                return {"frames_written": 21, "seconds": 0.01}

            header = {"width": 15360, "height": 8640, "frame_count": 21,
                      "avg_frame_rate": "16/1", "r_frame_rate": "16/1"}
            with (
                patch.object(flash.shutil, "which", return_value="/usr/bin/tool"),
                patch.object(flash, "required_assets", return_value=[]),
                patch.object(flash, "read_outpainting_feather_width", return_value=16),
                patch.object(flash, "probe_video", side_effect=[
                    {"width": 3840, "height": 2160, "frame_count": 21},
                    {"width": 7680, "height": 4320, "frame_count": 21,
                     "avg_frame_rate": "16/1", "r_frame_rate": "16/1"},
                ]),
                patch.object(flash, "validate_video_spec"),
                patch.object(flash, "validate_frame_index_alignment"),
                patch.object(flash, "stage_request_payload", return_value={"request": 1}),
                patch.object(flash, "load_reference_module", return_value=reference),
                patch.object(flash, "validate_stage_manifest", return_value=([], {}, args.work_dir)),
                patch.object(flash, "load_media_utils", return_value=media),
                patch.object(flash, "encode_staged_tiles", side_effect=fake_final_encode) as encode,
                patch.object(flash, "validate_final_output", return_value=header) as verify,
            ):
                manifest = flash.execute(args)
            encode.assert_called_once()
            verify.assert_called_once_with(args.output, expected_frames=21, count_frames=False)
            self.assertTrue(args.output.is_file())
            self.assertFalse(args.raw_output.exists())
            self.assertEqual(manifest["encoding"]["output_mode"], "final_only")
            self.assertIs(manifest["encoding"]["raw_16k_intermediate_video"], False)
            self.assertEqual(json.loads(args.final_manifest.read_text())["status"], "PASS")

    def test_final_only_resume_does_not_require_raw_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._arguments(Path(temporary), "final_only")
            args.output.parent.mkdir(parents=True)
            args.output.write_bytes(b"finished")
            args.final_manifest.parent.mkdir(parents=True)
            request = {"request": 1}
            args.final_manifest.write_text(json.dumps({
                "schema_version": flash.PIPELINE_SCHEMA_VERSION, "status": "PASS",
                "video_id": "05", "request": request,
                "output_file": flash.file_identity(args.output),
                "encoding": {"output_mode": "final_only", "frames_written": 21},
            }))
            paths = flash.stage_paths(args.work_dir)
            args.work_dir.mkdir()
            paths["request"].write_text(json.dumps(request))
            paths["manifest"].write_text("{}")
            reference = SimpleNamespace(read_stage_manifest=lambda _path: {})
            with (
                patch.object(flash, "load_reference_module", return_value=reference),
                patch.object(flash, "validate_stage_manifest"),
                patch.object(flash, "probe_video", return_value={}),
                patch.object(flash, "validate_final_output", return_value={"valid": True}) as check,
            ):
                self.assertEqual(
                    flash.validate_existing_result(args, request=request), {"valid": True}
                )
                self.assertEqual(check.call_count, 1)
                self.assertIs(check.call_args.kwargs["count_frames"], False)
            self.assertFalse(args.raw_output.exists())

    def test_legacy_resume_requires_explicit_compatibility_and_retained_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._arguments(Path(temporary), "final_only")
            args.output.parent.mkdir(parents=True)
            args.output.write_bytes(b"old-final")
            args.raw_output.parent.mkdir(parents=True)
            args.raw_output.write_bytes(b"old-raw")
            args.final_manifest.parent.mkdir(parents=True)
            request = {"request": 1}
            args.final_manifest.write_text(json.dumps({
                "schema_version": flash.PIPELINE_SCHEMA_VERSION, "status": "PASS",
                "video_id": "05", "request": request,
                "output_file": flash.file_identity(args.output), "encoding": {},
            }))
            paths = flash.stage_paths(args.work_dir)
            args.work_dir.mkdir()
            paths["request"].write_text(json.dumps(request))
            paths["manifest"].write_text("{}")
            with self.assertRaisesRegex(flash.PipelineError, "output mode"):
                flash.validate_existing_result(args, request=request)
            args.allow_existing_legacy = True
            reference = SimpleNamespace(read_stage_manifest=lambda _path: {})
            with (
                patch.object(flash, "load_reference_module", return_value=reference),
                patch.object(flash, "validate_stage_manifest"),
                patch.object(flash, "probe_video", return_value={}),
                patch.object(flash, "validate_final_output", return_value={"valid": True}) as check,
            ):
                flash.validate_existing_result(args, request=request)
                self.assertEqual(check.call_count, 2)  # raw and final
                self.assertTrue(check.call_args.kwargs["count_frames"])

    def test_parent_passes_mode_and_never_probes_raw_in_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            segments = stage / "segments"
            segments.mkdir()
            (segments / "flashvsr_stage_manifest.json").write_text(json.dumps({
                "tile_count": 170, "temporal_unit_metadata": []
            }))
            final_manifest = stage / "metadata" / "final_manifest.json"
            final_manifest.parent.mkdir()
            final_manifest.write_text(json.dumps({
                "status": "PASS", "encoding": {"output_mode": "final_only", "frames_written": 21}
            }))
            config = {
                "input": {"video_id": "05"}, "seed": 71634910,
                "upscale": {"runtime": "src/flashvsr/architect/FlashVSR_v1.1/upscale.py",
                            "model": "src/models/flashvsr_v1_1", "tile_size": "auto",
                            "tile_overlap": "auto", "min_tile_size": 128, "sparse_ratio": 2.0,
                            "local_range": 11, "gpu_index": 0, "ffmpeg_threads": 32,
                            "encoder": "libx265"},
                "final": {"filename": "final.mp4"},
            }
            commands = []
            with (
                patch.object(runner, "artifact_path", return_value=stage),
                patch.object(runner, "upscale_input", return_value=stage / "input.mp4"),
                patch.object(runner, "absolute_package_path", return_value=stage / "runtime.py"),
                patch.object(runner, "stage_log", return_value=contextlib.nullcontext()),
                patch.object(runner, "run_logged", side_effect=lambda cmd, **_k: commands.append(cmd)),
                patch.object(runner, "verify_video_headers", return_value={"frames": 21}) as header,
                patch.object(runner, "verify_video", side_effect=AssertionError("raw/full probe")),
                patch.object(runner, "atomic_json"),
                patch.object(runner, "log"),
            ):
                runner._run_flash_native(
                    config, stage / "anchor.mp4", stage / "input.mp4", stage,
                    21, 21, stage / "inference.yaml", stage / "models", resume=False,
                )
            self.assertEqual(commands[0][-2:], ["--output-mode", "final_only"])
            header.assert_called_once()


if __name__ == "__main__":
    unittest.main()
