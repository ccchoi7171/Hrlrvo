"""CPU-only historical producer regressions. No media or model execution."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "historical_runner_test", ROOT / "run_cert.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
pre = runner.preprocess


def source_probe():
    return {
        "streams": [
            {
                "codec_type": "video",
                "width": 7680,
                "height": 4320,
                "avg_frame_rate": "16/1",
                "nb_frames": "1952",
                "pix_fmt": "yuv420p10le",
                "color_range": "pc",
                "color_space": "bt709",
            },
            {"codec_type": "audio"},
        ]
    }


class HistoricalPreprocessTests(unittest.TestCase):
    def test_fresh_full_keeps_canonical_reference_and_selects_generated(self):
        config = json.loads((ROOT / "configs/certification/05.json").read_text())
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "ROOT", Path(tmp)),
        ):
            canonical = runner.artifact_path(config, "preprocess")
            canonical.parent.mkdir(parents=True)
            canonical.write_bytes(b"immutable-reference")
            source = Path(tmp) / config["input"]["path"]
            source.parent.mkdir(parents=True)
            source.write_bytes(b"source")
            self.assertEqual(runner.preprocessed_input_path(config), canonical)

            def generate(source, output, metadata, log, expected, **kwargs):
                output.mkdir(parents=True)
                result = output / "05_PreProc_2K.mp4"
                result.write_bytes(b"fresh-result")
                return result

            with (
                patch.object(runner, "preflight_source", return_value=1952),
                patch.object(pre, "run_preprocess", side_effect=generate),
                patch.object(
                    pre,
                    "producer_identity",
                    return_value={"contract": pre.PREPROCESS_CONTRACT},
                ),
            ):
                result = runner.run_preprocess_stage(config)
            self.assertEqual(canonical.read_bytes(), b"immutable-reference")
            self.assertNotEqual(result, canonical)
            self.assertEqual(result, runner.preprocessed_input_path(config))
            self.assertEqual(result.read_bytes(), b"fresh-result")

    def test_generated_artifact_with_changed_toolchain_is_rejected(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner, "ROOT", Path(tmp)),
        ):
            root = Path(tmp)
            output = root / "data/generated.mp4"
            output.parent.mkdir()
            output.write_bytes(b"generated")
            output.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "producer": "run_preprocess_stage",
                        "output_sha256": runner.sha256_file(output),
                        "preprocess_config": pre.PREPROCESS_CONTRACT,
                        "producer_identity": {
                            "contract": pre.PREPROCESS_CONTRACT,
                            "implementation_sha256": "old",
                        },
                    }
                )
            )
            config = {
                "input": {"video_id": "05"},
                "artifacts": {"preprocess": "data/generated.mp4"},
            }
            with (
                patch.object(pre, "probe_media", return_value=source_probe()),
                patch.object(
                    pre,
                    "producer_identity",
                    return_value={
                        "contract": pre.PREPROCESS_CONTRACT,
                        "implementation_sha256": "new",
                    },
                ),
            ):
                with self.assertRaisesRegex(runner.ConfigError, "toolchain"):
                    runner.validate_preprocessed(config)

    def test_output_time_color_audio_and_duration_contract(self):
        duration = 1952 / 24
        probe = {
            "streams": [
                dict(
                    source_probe()["streams"][0],
                    width=1920,
                    height=1080,
                    avg_frame_rate="24/1",
                    codec_name="hevc",
                    time_base="1/24576",
                    duration=str(duration),
                ),
                {"codec_type": "audio", "codec_name": "aac", "duration": str(duration)},
            ],
            "format": {"duration": str(duration)},
        }
        pre.validate_output_probe(probe, 1952, True)
        for field, value in [
            ("color_range", "tv"),
            ("time_base", "1/16384"),
            ("duration", "122"),
            ("codec_name", "h264"),
        ]:
            changed = json.loads(json.dumps(probe))
            changed["streams"][0][field] = value
            with self.subTest(field=field), self.assertRaises(pre.PreprocessError):
                pre.validate_output_probe(changed, 1952, True)
        changed = json.loads(json.dumps(probe))
        changed["streams"][1]["duration"] = "122"
        with self.assertRaisesRegex(pre.PreprocessError, "audio duration"):
            pre.validate_output_probe(changed, 1952, True)

    def test_producer_provenance_rejects_crf1_and_accepts_all_historical_records(self):
        for video in ("01", "02", "03", "04", "05"):
            record = json.loads(
                (
                    ROOT / f"data/01_PreProc_2K_1920x1080/{video}_PreProc_2K.json"
                ).read_text()
            )
            pre.validate_producer_provenance(record)
            record["ffmpeg_command"][record["ffmpeg_command"].index("-crf") + 1] = "1"
            with self.assertRaisesRegex(pre.PreprocessError, "contract"):
                pre.validate_producer_provenance(record)

    def test_validation_layout_is_isolated_for_all_five_videos(self):
        for path in (ROOT / "configs/validation/preprocess_reproduction").glob(
            "*.json"
        ):
            config, *_ = runner.validate_config(path)
            self.assertTrue(
                runner.artifact_path(config, "preprocess").is_relative_to(
                    ROOT / "data/validation/preprocess_reproduction"
                )
            )

    def test_entire_command_matches_all_five_historical_sidecars(self):
        for video in ("01", "02", "03", "04", "05"):
            record = json.loads(
                (
                    ROOT / f"data/01_PreProc_2K_1920x1080/{video}_PreProc_2K.json"
                ).read_text()
            )
            expected = record["ffmpeg_command"]
            actual = pre.build_ffmpeg_command(
                Path(expected[expected.index("-i") + 1]),
                Path(expected[-1]),
                expected[0],
                True,
            )
            with self.subTest(video=video):
                self.assertEqual(actual, expected)

    def test_historical_filename(self):
        self.assertEqual(
            pre.output_path_for(Path("05_Input_8K.mp4"), Path("output")),
            Path("output/05_PreProc_2K.mp4"),
        )

    def test_historical_color_and_audio_contract(self):
        pre.validate_input_probe(source_probe(), 1952)
        for field, value in [
            ("color_range", "tv"),
            ("color_space", "bt2020nc"),
            ("pix_fmt", "yuv420p"),
        ]:
            probe = source_probe()
            probe["streams"][0][field] = value
            with self.subTest(field=field), self.assertRaises(pre.PreprocessError):
                pre.validate_input_probe(probe, 1952)
        for audio_count in (0, 2):
            probe = source_probe()
            probe["streams"] = (
                probe["streams"][:1] + [{"codec_type": "audio"}] * audio_count
            )
            with (
                self.subTest(audio_count=audio_count),
                self.assertRaises(pre.PreprocessError),
            ):
                pre.validate_input_probe(probe, 1952)

    def test_builder_rejects_absent_audio(self):
        with self.assertRaises(pre.PreprocessError):
            pre.build_ffmpeg_command(
                Path("input.mp4"), Path("out.mp4"), "ffmpeg", False
            )

    def test_resume_refuses_legacy_producer_even_if_source_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "05_Input_8K.mp4"
            source.write_bytes(b"fake-input")
            output = root / "output"
            output.mkdir()
            pre.output_path_for(source, output).write_bytes(b"legacy-CRF1-output")
            metadata = root / "metadata"
            metadata.mkdir()
            identity = pre._source_identity(source)
            (metadata / "frame_alignment.json").write_text(
                json.dumps(
                    {"source_identity": identity, "preprocess_contract": {"crf": 1}}
                )
            )
            with (
                patch.object(pre, "probe_media", return_value=source_probe()),
                patch.object(pre, "_run_logged") as encode,
            ):
                with self.assertRaisesRegex(pre.PreprocessError, "contract"):
                    pre.run_preprocess(
                        source, output, metadata, root / "log.txt", 1952, resume=True
                    )
                encode.assert_not_called()

    def test_same_corrected_contract_in_all_active_configs_and_schema(self):
        schema = json.loads((ROOT / "configs/schemas/job.schema.json").read_text())
        self.assertEqual(schema["properties"]["preprocess"]["const"]["crf"], 18)
        for folder in ("certification", "quick", "validation"):
            for path in (ROOT / "configs" / folder).rglob("*.json"):
                value = json.loads(path.read_text())
                with self.subTest(path=path):
                    self.assertEqual(value["preprocess"], pre.PREPROCESS_CONTRACT)

    def test_stage_only_and_full_dispatch_same_preprocess(self):
        for selected in ("preprocess", "all"):
            config = json.loads((ROOT / "configs/quick/05.json").read_text())
            with (
                patch.object(
                    runner, "run_preprocess_stage", return_value=Path("pre.mp4")
                ) as shared,
                patch.object(runner, "run_outpaint_stage"),
                patch.object(runner, "run_upscale_stage"),
            ):
                runner.execute_stages(config, selected)
                shared.assert_called_once_with(config, resume=False, force=False)


if __name__ == "__main__":
    unittest.main()
