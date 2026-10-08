"""Fast wrapper tests; real model/artifact tests are the mandatory GPU smokes."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import orchestrate_seedsearch as orchestrator
from common import INPUT_ROOT, VIDEO_IDS, input_for, output_for
from run_single_seedsearch import native_command
from seed_schedule import ensure_schedule


class SeedSearchTests(unittest.TestCase):
    def test_input_mapping_and_output_names(self) -> None:
        self.assertEqual(VIDEO_IDS, ("01", "02", "03", "04", "05"))
        for video_id in VIDEO_IDS:
            self.assertEqual(input_for(video_id), INPUT_ROOT / f"{video_id}_Input_2K.mp4")
            self.assertNotIn("01_Resize_2K_1920_1080", str(input_for(video_id)))
            mp4, tensor, meta = output_for(video_id, 33333333)
            self.assertEqual(mp4.name, f"{video_id}_33333333_4K.mp4")
            self.assertEqual(tensor.name, f"{video_id}_33333333_4K_prefix.pt")
            self.assertEqual(meta.name, f"{video_id}_33333333_4K_meta.json")

    def test_schedule_is_unique_and_immutable_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, path, digest = ensure_schedule(root)
            second, path2, digest2 = ensure_schedule(root)
            self.assertEqual(path, path2)
            self.assertEqual(digest, digest2)
            self.assertEqual(first, second)
            self.assertEqual(len(first["seeds_in_generation_order"]), 99)
            self.assertEqual(len(set(first["seeds_in_generation_order"])), 99)
            self.assertEqual(len(first["rounds"]), 33)
            self.assertEqual(first["rounds"][0]["seeds"], first["seeds_in_generation_order"][:3])
            self.assertEqual(first["rounds"][-1]["seeds"], first["seeds_in_generation_order"][-3:])
            edited = json.loads(path.read_text(encoding="utf-8"))
            edited["rounds"][0]["seeds"].reverse()
            path.write_text(json.dumps(edited), encoding="utf-8")
            with self.assertRaises(ValueError):
                ensure_schedule(root)

    def test_native_command_is_one_seed_b1_with_required_flags(self) -> None:
        command = native_command(input_video=input_for("03"), seed=123, stage=Path("/tmp/stage"),
                                 config=Path("/tmp/inference.yaml"), checkpoint=Path("/tmp/ckpt"), steps=30)
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[command.index("--prefix-seeds") + 1], "123")
        self.assertEqual(command[command.index("--mode") + 1], "prefix_seed_search")
        self.assertEqual(command[command.index("--anchor-video") + 1], str(input_for("03")))
        for flag in ("--video-only-gpu-weights", "--vae-decode-streaming", "--no-ffn-token-chunking"):
            self.assertIn(flag, command)
        self.assertNotIn("--num-prefix-samples", command)

    def test_group_spawns_three_before_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            launches: list[int] = []
            validations = 0

            class FakeProcess:
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    self.pid = 100 + len(launches)
                    launches.append(self.pid)

                def wait(self) -> int:
                    self_test.assertEqual(len(launches), 3)
                    return 0

            def fake_validate(*_args: object, **_kwargs: object) -> dict:
                nonlocal validations
                validations += 1
                if validations <= 3:
                    raise FileNotFoundError("not completed")
                return {}

            self_test = self
            args = argparse.Namespace(config=Path("/tmp/inference.yaml"),
                                      checkpoint=Path("/tmp/checkpoint"), steps=30)
            with (patch.object(orchestrator.subprocess, "Popen", FakeProcess),
                  patch.object(orchestrator, "validate_candidate", side_effect=fake_validate),
                  patch.object(orchestrator, "write_progress")):
                result = orchestrator.run_group(
                    args, root=Path(folder), schedule={}, schedule_sha="abc",
                    inputs={"01": {"sha256": "inputsha"}},
                    identity={"config_sha256": "configsha", "adapter_sha256": "adaptersha"},
                    video_id="01", round_index=1, seeds=[11, 22, 33],
                )
            self.assertEqual(len(result["launched"]), 3)
            self.assertEqual([entry["seed"] for entry in result["launched"]], [11, 22, 33])
            self.assertEqual(validations, 6)


if __name__ == "__main__":
    unittest.main()
