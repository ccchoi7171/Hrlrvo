from __future__ import annotations

import errno
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ARCHITECT = Path(__file__).resolve().parents[1]
if str(ARCHITECT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT))

from common.media_utils import VideoMetadata, finalize_video_with_audio


class CrossDevicePublicationTest(unittest.TestCase):
    @staticmethod
    def metadata() -> VideoMetadata:
        return VideoMetadata(
            path="input.mp4", width=1, height=1, fps="1/1", fps_float=1.0,
            frame_count=1, duration=1.0, pixel_format="yuv420p",
            video_codec="h264", color_space=None, color_transfer=None,
            color_primaries=None, time_base="1/1", audio_present=False,
            audio_codec=None,
        )

    def test_silent_video_is_published_when_direct_replace_crosses_filesystems(self) -> None:
        real_replace = os.replace
        with tempfile.TemporaryDirectory() as source_directory, tempfile.TemporaryDirectory() as output_directory:
            source = Path(source_directory) / "video_only.mp4"
            source.write_bytes(b"complete encoded video")
            output_base = Path(output_directory) / "final"
            metadata = self.metadata()
            direct_attempts = 0

            def replace_with_cross_device_failure(src, dst):
                nonlocal direct_attempts
                if Path(src) == source:
                    direct_attempts += 1
                    raise OSError(errno.EXDEV, "Invalid cross-device link")
                return real_replace(src, dst)

            with mock.patch("common.media_utils.os.replace", side_effect=replace_with_cross_device_failure):
                published = finalize_video_with_audio(
                    source,
                    output_base,
                    input_path="input.mp4",
                    input_metadata=metadata,
                    extension=".mp4",
                )

            self.assertEqual(direct_attempts, 1)
            self.assertEqual(published, output_base.with_suffix(".mp4"))
            self.assertEqual(published.read_bytes(), b"complete encoded video")
            self.assertFalse(source.exists())
            self.assertEqual(list(Path(output_directory).glob("*.partial")), [])

    def test_version_dot_in_extensionless_output_base_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "video_only.mp4"
            source.write_bytes(b"video")
            output_base = root / "clip_FlashVSR_v1.1_x4_mini"
            published = finalize_video_with_audio(
                source,
                output_base,
                input_path="input.mp4",
                input_metadata=self.metadata(),
                extension=".mp4",
            )
            self.assertEqual(published.name, "clip_FlashVSR_v1.1_x4_mini.mp4")
            self.assertEqual(published.read_bytes(), b"video")


if __name__ == "__main__":
    unittest.main()
