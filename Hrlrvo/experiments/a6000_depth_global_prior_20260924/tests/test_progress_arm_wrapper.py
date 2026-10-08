from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_progress_for_arm.py"
SPEC = importlib.util.spec_from_file_location("run_progress_for_arm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
wrapper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wrapper
SPEC.loader.exec_module(wrapper)


def test_arm_mapping_is_pinned_to_expected_roots_modes_and_gpus():
    assert wrapper.arm_spec("02_no_carry").gpu == "1"
    assert wrapper.arm_spec("03_global_boost").gpu == "2"
    assert wrapper.arm_spec("04_native_halo").gpu == "3"
    depth = wrapper.arm_spec("depth")
    assert depth.gpu == "0"
    assert depth.mode == "strength_1p0_global_depth_prior"
    assert depth.require_c0_before_watcher is True
    with pytest.raises(ValueError):
        wrapper.arm_spec("01_baseline")


def _fake_process(tmp_path: Path, spec, pid: int = 321) -> Path:
    process = tmp_path / "proc" / str(pid)
    process.mkdir(parents=True)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    runner = cwd / spec.runner.name
    runner.touch()
    (process / "cwd").symlink_to(cwd, target_is_directory=True)
    (process / "stat").write_text(f"{pid} (python) S 1 2 3\n")
    command = ["python", str(runner)]
    if spec.name == "depth":
        command += ["--mode", spec.mode]
    (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in command) + b"\0")
    environment = {"CUDA_VISIBLE_DEVICES": spec.gpu, **dict(spec.required_environment)}
    (process / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in environment.items()) + b"\0"
    )
    return process.parent


def test_validate_producer_checks_command_mode_and_environment(tmp_path: Path):
    original = wrapper.arm_spec("depth")
    spec = wrapper.ArmSpec(
        **{**original.__dict__, "runner": tmp_path / "cwd/run_depth_chain.py"}
    )
    proc_root = _fake_process(tmp_path, spec)
    wrapper.validate_producer(321, spec, proc_root=proc_root)
    environ = proc_root / "321/environ"
    environ.write_bytes(b"CUDA_VISIBLE_DEVICES=7\0HRLRVO_EXPERIMENT=wrong\0")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        wrapper.validate_producer(321, spec, proc_root=proc_root)


def test_depth_waits_for_valid_c0_before_returning_pid(tmp_path: Path, monkeypatch):
    root = tmp_path / "depth"
    mode = wrapper.DEPTH_MODE
    chunks = root / "chunks" / mode
    chunks.mkdir(parents=True)
    (root / "status.json").write_text(json.dumps({"state": "running", "pid": 88}))
    original = wrapper.arm_spec("depth")
    spec = wrapper.ArmSpec(**{**original.__dict__, "root": root})
    monkeypatch.setattr(wrapper, "validate_producer", lambda pid, selected: None)
    sleeps = 0

    def sleep(_seconds):
        nonlocal sleeps
        sleeps += 1
        pixels = np.zeros((65, 2, 4, 3), dtype=np.uint8)
        np.save(chunks / "segment_000000.npy", pixels, allow_pickle=False)
        (chunks / "chunk_000000.json").write_text(
            json.dumps(
                {
                    "chunk_index": 0,
                    "segment_file": "segment_000000.npy",
                    "output_appended_count": 65,
                    "new_output_frame_range": [0, 65],
                }
            )
        )

    monkeypatch.setattr(wrapper.time, "sleep", sleep)
    assert wrapper.wait_for_ready(spec, poll_seconds=0.001) == 88
    assert sleeps == 1


def test_failed_status_is_terminal_and_never_calls_watcher(tmp_path: Path):
    root = tmp_path / "arm"
    root.mkdir()
    (root / "status.json").write_text(json.dumps({"state": "failed", "pid": 123}))
    original = wrapper.arm_spec("02_no_carry")
    spec = wrapper.ArmSpec(**{**original.__dict__, "root": root})
    with pytest.raises(RuntimeError, match="terminal/invalid"):
        wrapper.wait_for_ready(spec, poll_seconds=0.001)


def test_complete_status_needs_all_records_but_no_live_producer_pid(tmp_path: Path):
    root = tmp_path / "arm"
    chunks = root / "chunks"
    chunks.mkdir(parents=True)
    start = 0
    for chunk in range(24):
        count = 65 if chunk == 0 else 48
        name = f"segment_{chunk:06d}.npy"
        np.save(chunks / name, np.zeros((count, 2, 4, 3), dtype=np.uint8), allow_pickle=False)
        (chunks / f"chunk_{chunk:06d}.json").write_text(
            json.dumps(
                {
                    "chunk_index": chunk,
                    "segment_file": name,
                    "output_appended_count": count,
                    "new_output_frame_range": [start, start + count],
                }
            )
        )
        start += count
    (root / "status.json").write_text(json.dumps({"state": "complete", "exit_code": 0}))
    original = wrapper.arm_spec("02_no_carry")
    spec = wrapper.ArmSpec(**{**original.__dict__, "root": root})
    assert wrapper.wait_for_ready(spec, poll_seconds=0.001) == os.getpid()
