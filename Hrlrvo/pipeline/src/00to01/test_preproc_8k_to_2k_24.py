from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).with_name("preproc_8k_to_2k_24.py")


def load_module():
    spec = importlib.util.spec_from_file_location("preproc_8k_to_2k_24", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreprocessContractTests(unittest.TestCase):
    def test_all_video_configs_expose_one_combined_wrapper(self) -> None:
        for video_id in ("01", "02", "03", "04", "05"):
            with self.subTest(video_id=video_id):
                folder = PIPELINE / "config" / video_id
                wrappers = sorted(path.name for path in folder.glob("*.sh"))
                self.assertIn(f"{video_id}_preproc_2K.sh", wrappers)
                self.assertIn(f"{video_id}_seedsearch_4K.sh", wrappers)
                self.assertEqual(len(wrappers), 2)

                script = (folder / f"{video_id}_preproc_2K.sh").read_text(
                    encoding="utf-8"
                )
                self.assertIn(
                    f"data/00_Input_8K_7680x4320/{video_id}_Input_8K.mp4",
                    script,
                )
                self.assertIn("data/01_PreProc_2K_1920x1080", script)
                self.assertIn("preproc_8k_to_2k_24.py", script)

    def test_combined_command_has_resize_and_retime_filters(self) -> None:
        module = load_module()
        command = module.build_ffmpeg_command(
            Path("/input/01_Input_8K.mp4"),
            Path("/output/01_PreProc_2K.partial.mp4"),
            "ffmpeg",
        )
        video_filter = command[command.index("-vf") + 1]
        self.assertIn("scale=1920:1080:flags=lanczos", video_filter)
        self.assertIn("setpts=PTS*2/3", video_filter)
        self.assertIn("fps=24", video_filter)
        self.assertEqual(command[command.index("-af") + 1], "atempo=1.5")
        self.assertEqual(command[command.index("-video_track_timescale") + 1], "24576")

    def test_output_name_and_directory_are_single_stage(self) -> None:
        module = load_module()
        output = module.output_path_for(
            Path("/input/03_Input_8K.mp4"),
            Path("/pipeline/data/01_PreProc_2K_1920x1080"),
        )
        self.assertEqual(
            output,
            Path("/pipeline/data/01_PreProc_2K_1920x1080/03_PreProc_2K.mp4"),
        )


if __name__ == "__main__":
    unittest.main()
