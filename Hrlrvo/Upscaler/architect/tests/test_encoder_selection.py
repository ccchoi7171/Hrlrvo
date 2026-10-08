from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ARCHITECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARCHITECT))

from common.media_utils import select_encoder


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
class Libx265ThreadRegressionTest(unittest.TestCase):
    def test_requested_21_threads_uses_x265_safe_auto_frame_threads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hrlrvo_encoder_test_") as scratch:
            encoder, attempts = select_encoder(
                64,
                64,
                threads=21,
                scratch_dir=scratch,
                preferred="libx265",
                timeout=30,
            )

        self.assertEqual(encoder["name"], "libx265")
        x265_params = encoder["arguments"][
            encoder["arguments"].index("-x265-params") + 1
        ]
        self.assertIn("pools=21", x265_params)
        self.assertIn("frame-threads=0", x265_params)
        self.assertEqual(attempts, [{"encoder": "libx265", "result": "PASS", "reason": ""}])


if __name__ == "__main__":
    unittest.main()
