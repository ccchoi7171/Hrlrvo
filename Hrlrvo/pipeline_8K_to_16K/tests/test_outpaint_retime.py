"""CPU-only final-postprocess regressions; no model, CUDA or media execution."""

import ast
import inspect
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cert_pipeline import outpaint  # noqa: E402


def probe(fps, nominal=None):
    return {
        "streams": [
            {
                "codec_type": "video",
                "width": 3840,
                "height": 2160,
                "avg_frame_rate": f"{fps}/1",
                "r_frame_rate": nominal or f"{fps}/1",
                "nb_read_frames": "65",
                "pix_fmt": "yuv420p",
                "codec_name": "h264",
            }
        ]
    }


class FinalRetimeTests(unittest.TestCase):
    def test_chunk_audit_writer_and_reuse_are_native24(self):
        writer = inspect.getsource(outpaint._encode_segment_mp4)
        reuse = inspect.getsource(outpaint._persist_committed_chunk)
        self.assertIn("str(PROCESSING_FPS.numerator)", writer)
        self.assertNotIn("STORED_FPS", writer)
        self.assertNotIn("STORED_FPS", reuse)
        self.assertNotIn("_retime_to_16fps", writer + reuse)

    def test_historical_clock_and_lossless_encoder_after_full_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, final = root / "native24.mp4", root / "deliverable16.mp4"
            source.write_bytes(b"native")
            commands = []

            def encode(command, log):
                commands.append(command)
                Path(command[-1]).write_bytes(b"retimed")

            with (
                patch.object(
                    outpaint,
                    "_probe",
                    side_effect=lambda p: probe(24 if p == source else 16),
                ),
                patch.object(
                    outpaint, "_decoded_rgb_hashes", return_value=["same"] * 65
                ),
                patch.object(outpaint, "_run_ffmpeg", side_effect=encode),
            ):
                result = outpaint._retime_to_16fps(
                    source, final, frame_count=65, log_path=root / "log", resume=False
                )
            self.assertEqual(len(commands), 1)
            command = commands[0]
            self.assertEqual(command[command.index("-enc_time_base:v") + 1], "1:16")
            self.assertEqual(
                command[command.index("-vf") + 1], "settb=expr=1/16,setpts=N"
            )
            self.assertEqual(command[command.index("-c:v") + 1], "libx264")
            self.assertEqual(command[command.index("-crf") + 1], "0")
            self.assertNotIn("-r", command)
            self.assertNotIn("fps=16", " ".join(command))
            self.assertEqual(result["fps"], "16")
            self.assertEqual(result["r_frame_rate"], "16")
            self.assertTrue(
                json.loads(final.with_suffix(".retime.json").read_text())[
                    "decoded_rgb_sequence_exact"
                ]
            )

    def test_retime_rejects_pixel_change_and_does_not_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, final = root / "native24.mp4", root / "deliverable16.mp4"
            source.write_bytes(b"native")

            def encode(command, log):
                Path(command[-1]).write_bytes(b"wrong")

            with (
                patch.object(
                    outpaint,
                    "_probe",
                    side_effect=lambda p: probe(24 if p == source else 16),
                ),
                patch.object(
                    outpaint,
                    "_decoded_rgb_hashes",
                    side_effect=[["a"] * 65, ["b"] * 65],
                ),
                patch.object(outpaint, "_run_ffmpeg", side_effect=encode),
            ):
                with self.assertRaisesRegex(
                    outpaint.OutpaintError, "changed decoded RGB at frame 0"
                ):
                    outpaint._retime_to_16fps(
                        source,
                        final,
                        frame_count=65,
                        log_path=root / "log",
                        resume=False,
                    )
            self.assertFalse(final.exists())
            self.assertFalse(list(root.glob("*.partial.mp4")))

    def test_nominal_and_average_rates_are_both_checked(self):
        with self.assertRaisesRegex(outpaint.OutpaintError, "r_frame_rate"):
            outpaint._validate_video(
                probe(16, "780/49"),
                path=Path("x"),
                expected_frames=65,
                expected_fps=Fraction(16),
            )

    def test_retime_is_not_in_a_chunk_callback_or_loop(self):
        tree = ast.parse(inspect.getsource(outpaint.run_outpaint))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_retime_to_16fps"
        ]
        self.assertEqual(len(calls), 1)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)) or (
                isinstance(node, ast.FunctionDef) and node.name == "committed_observer"
            ):
                self.assertNotIn(calls[0], list(ast.walk(node)))
        native = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "sample_long_range"
        ]
        self.assertEqual(len(native), 1)
        self.assertLess(native[0].lineno, calls[0].lineno)


if __name__ == "__main__":
    unittest.main()
