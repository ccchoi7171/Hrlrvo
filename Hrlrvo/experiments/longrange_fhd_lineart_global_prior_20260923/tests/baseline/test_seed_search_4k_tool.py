from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/seed_search_4k.py"
SPEC = importlib.util.spec_from_file_location("seed_search_4k", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
seed_search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = seed_search
SPEC.loader.exec_module(seed_search)


def test_native_command_is_exact_65f_and_forwards_negative_prompt(
    tmp_path: Path,
) -> None:
    prompt = seed_search.DEFAULT_NEGATIVE_PROMPT
    generation = {
        "config_path": "/repo/config.yaml",
        "checkpoint_path": "/repo/checkpoint-007500",
        "inputs": {"01": {"path": "/data/01_PreProc_2K.mp4"}},
        "negative_prompt": prompt,
    }
    args = seed_search.parse_args([])
    command = seed_search.native_command(
        args=args,
        generation=generation,
        video_id="01",
        seed=33333333,
        run_dir=tmp_path,
    )
    assert command[command.index("--mode") + 1] == "normal_long_range"
    assert command[command.index("--num-frames") + 1] == "65"
    assert command[command.index("--seed") + 1] == "33333333"
    assert command[command.index("--negative-prompt") + 1] == prompt
    assert command[command.index("--checkpoint") + 1].endswith("checkpoint-007500")
    assert command[command.index("--steps") + 1] == "30"
    assert command[command.index("--preview-step") + 1] == "10"
    frames_at = command.index("--preview-frames")
    assert command[frames_at + 1 : frames_at + 4] == ["0", "32", "64"]
    assert command[command.index("--seed-search-output-dir") + 1] == str(tmp_path)
    assert "--output" not in command
    assert "--segment-dir" not in command
    assert "prefix_seed_search" not in command


def test_seed_schedule_starts_fixed_then_uses_unique_eight_digit_seeds(
    monkeypatch,
) -> None:
    values = iter((1, 2, 2, 3))
    monkeypatch.setattr(seed_search.secrets, "randbelow", lambda _span: next(values))
    schedule = {
        "start_seed": 33333333,
        "entries": [],
    }
    first = seed_search.allocate_seeds(schedule, 2)
    assert first == [33333333, 10000001]
    schedule["entries"].append({"seeds": first})
    second = seed_search.allocate_seeds(schedule, 2)
    assert second == [10000002, 10000003]
    all_seeds = first + second
    assert len(all_seeds) == len(set(all_seeds))
    assert all(10_000_000 <= seed <= 99_999_999 for seed in all_seeds)


def test_scheduler_state_initializes_at_video_01_round_1(tmp_path: Path) -> None:
    contract = {
        "videos": ["01", "02", "03", "04", "05"],
        "max_concurrent": 2,
        "seeds_per_video_per_round": 2,
    }
    state = seed_search.load_or_create_scheduler_state(tmp_path, contract)
    assert state["next_video_index"] == 0
    assert state["next_round_index"] == 1
    assert state["current_batch"] is None
    stored = json.loads((tmp_path / "scheduler_contract.json").read_text())
    assert stored["contract"] == contract


def test_output_naming_contract(tmp_path: Path) -> None:
    assert seed_search.output_paths_for(tmp_path, "04", 48291357) == {
        0: tmp_path / "04/04_48291357_step10_f000.png",
        32: tmp_path / "04/04_48291357_step10_f032.png",
        64: tmp_path / "04/04_48291357_step10_f064.png",
    }
