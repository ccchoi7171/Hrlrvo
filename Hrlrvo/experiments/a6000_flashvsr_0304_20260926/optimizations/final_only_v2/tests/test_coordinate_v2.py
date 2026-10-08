from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("test_coordinate_v2_module", HERE / "coordinate_v2.py")
coordinate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = coordinate
SPEC.loader.exec_module(coordinate)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeRC:
    def __init__(self, root: Path):
        self.ROOT = root
        self.PYTHON = root / "env/bin/python"
        self.gate_calls = 0

    def check_gate(self, smoke=False):
        assert smoke is True
        self.gate_calls += 1

    def runtime_env(self):
        return {"TEST": "1"}

    def postprocess_command(self, video):
        p = coordinate.paths(video)
        return [str(self.PYTHON), str(coordinate.ROOT / "tools/postprocess_video.py"),
                "--video-id", video, "--input-4k", str(p["input"]),
                "--original-8k", str(p["anchor"]), "--raw-output", str(p["raw"]),
                "--output", str(p["final"]), "--final-manifest", str(p["manifest"]),
                "--aggregate-manifest", str(p["aggregate"]), "--tile-size", "256",
                "--tile-overlap", "32", "--min-tile-size", "256", "--seed", "71634910",
                "--sparse-ratio", "2.0", "--local-range", "11", "--ffmpeg-threads", "32",
                "--encoder", "libx265"]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    root = tmp_path / "run"
    overlay = root / "optimizations/final_only_v2"
    overlay.mkdir(parents=True)
    (root / "verification").mkdir()
    (root / "tools").mkdir()
    for name in ("final_only.py", "parallel_renderer.py", "coordinate_v2.py"):
        (overlay / name).write_text(f"# {name}\n")
    code_manifest = root / "verification/code_manifest.json"
    code_manifest.write_text("{}\n")
    source = root / "verification/source_transfer_verification.json"
    source.write_text(json.dumps({"status": "PASS"}) + "\n")
    benchmark = root / "verification/final_only_v2_bench/benchmark_decision.json"
    benchmark.parent.mkdir()
    policy = {"renderer": "threaded", "compose_workers": 4, "max_parallel_videos": 2}
    benchmark.write_text(json.dumps({
        "status": "PASS", "pixel_equality_pass": True, "selected_policy": policy,
        "evidence": {"fixture": "short", "concurrent_video_comparison": {
            "serial_seconds": 100.0, "parallel_seconds": 80.0, "makespan_gain_percent": 20.0,
        }}, "limitations": ["not a full video benchmark"],
    }) + "\n")
    manifest = overlay / "overlay_manifest.json"
    value = {
        "schema": coordinate.OVERLAY_SCHEMA,
        "files": {name: sha(overlay / name) for name in
                  ("final_only.py", "parallel_renderer.py", "coordinate_v2.py")},
        "baseline_code_manifest_sha256": sha(code_manifest),
        "source_proof": {"path": "verification/source_transfer_verification.json",
                         "sha256": sha(source)},
        "benchmark": {"path": "verification/final_only_v2_bench/benchmark_decision.json",
                      "sha256": sha(benchmark)},
        "policy": policy,
    }
    manifest.write_text(json.dumps(value, sort_keys=True) + "\n")
    monkeypatch.setattr(coordinate, "ROOT", root)
    monkeypatch.setattr(coordinate, "OVERLAY_ROOT", overlay)
    monkeypatch.setattr(coordinate, "STATUS_ROOT", root / "status/final_only_v2")
    monkeypatch.setattr(coordinate, "BASELINE_CODE_MANIFEST_SHA256", sha(code_manifest))
    return SimpleNamespace(root=root, overlay=overlay, manifest=manifest,
                           manifest_sha=sha(manifest), policy=policy, rc=FakeRC(root),
                           benchmark=benchmark)


def complete_workers(case, video="04", missing_receipt=False):
    for gpu in coordinate.GPU_ASSIGNMENTS[video]:
        status = case.root / f"status/gpu{gpu}.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({"status": "COMPLETE", "worker_pid": 1000 + gpu}))
        if not missing_receipt:
            receipt = coordinate.paths(video)["work"] / f"gpu{gpu}/shard_receipt.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text("{}")


def test_overlay_rejects_modified_script_and_stale_benchmark(isolated):
    coordinate.verify_overlay(isolated.manifest, isolated.manifest_sha)
    (isolated.overlay / "final_only.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="Overlay script changed"):
        coordinate.verify_overlay(isolated.manifest, isolated.manifest_sha)
    (isolated.overlay / "final_only.py").write_text("# final_only.py\n")
    isolated.benchmark.write_text("{}")
    with pytest.raises(RuntimeError, match="Benchmark proof changed"):
        coordinate.verify_overlay(isolated.manifest, isolated.manifest_sha)


def test_remote_controller_is_hash_pinned_before_import(isolated, monkeypatch):
    changed = isolated.root / "tools/remote_control.py"
    changed.write_text("raise AssertionError('must not execute')\n")
    monkeypatch.setattr(coordinate, "REMOTE_CONTROL", changed)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        coordinate.load_remote_control()


def test_modified_source_proof_is_rejected(isolated):
    source = isolated.root / "verification/source_transfer_verification.json"
    source.write_text(json.dumps({"status": "PASS", "modified": True}))
    with pytest.raises(RuntimeError, match="Source proof changed"):
        coordinate.verify_overlay(isolated.manifest, isolated.manifest_sha)


def test_manifest_hash_and_selected_policy_are_pinned(isolated):
    with pytest.raises(RuntimeError, match="manifest SHA256 mismatch"):
        coordinate.verify_overlay(isolated.manifest, "0" * 64)
    value = json.loads(isolated.benchmark.read_text())
    value["selected_policy"]["compose_workers"] = 8
    isolated.benchmark.write_text(json.dumps(value))
    manifest = json.loads(isolated.manifest.read_text())
    manifest["benchmark"]["sha256"] = sha(isolated.benchmark)
    isolated.manifest.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="selected policy differs"):
        coordinate.verify_overlay(isolated.manifest, sha(isolated.manifest))


def test_policy_rejects_bool_parallelism_and_unmeasured_parallel_gain(isolated):
    with pytest.raises(RuntimeError, match="integer 1 or 2"):
        coordinate.validate_policy({"renderer": "threaded", "compose_workers": 4,
                                    "max_parallel_videos": True})
    decision = json.loads(isolated.benchmark.read_text())
    decision["evidence"]["concurrent_video_comparison"] = {
        "serial_seconds": 100.0, "parallel_seconds": 95.0, "makespan_gain_percent": 5.0,
    }
    isolated.benchmark.write_text(json.dumps(decision))
    manifest = json.loads(isolated.manifest.read_text())
    manifest["benchmark"]["sha256"] = sha(isolated.benchmark)
    isolated.manifest.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="at least 10%"):
        coordinate.verify_overlay(isolated.manifest, sha(isolated.manifest))


def test_duplicate_same_video_coordinator_lock_rejected(isolated):
    lock = coordinate.STATUS_ROOT / "video04.lock"
    with coordinate.exclusive_lock(lock):
        with pytest.raises(RuntimeError, match="already owned"):
            with coordinate.exclusive_lock(lock):
                pass


def test_serial_policy_lock_is_blocking_not_nonblocking(isolated, monkeypatch):
    flags = []
    monkeypatch.setattr(coordinate.fcntl, "flock", lambda fd, flag: flags.append(flag))
    with coordinate.serialized_lock(coordinate.STATUS_ROOT / "serial.lock"):
        pass
    assert flags == [coordinate.fcntl.LOCK_EX]


def test_failed_worker_and_complete_without_receipt_rejected(isolated):
    status = isolated.root / "status/gpu0.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"status": "FAILED"}))
    with pytest.raises(RuntimeError, match="FAILED"):
        coordinate.worker_state("04")
    status.write_text(json.dumps({"status": "COMPLETE"}))
    (isolated.root / "status/gpu1.json").write_text(json.dumps({"status": "COMPLETE"}))
    with pytest.raises(RuntimeError, match="without shard receipt"):
        coordinate.worker_state("04")


def test_output_collision_fails_preflight(isolated):
    complete_workers(isolated)
    target = coordinate.paths("04")["final"]
    target.parent.mkdir(parents=True)
    target.write_text("existing")
    with pytest.raises(RuntimeError, match="fresh"):
        coordinate.preflight("04", isolated.manifest, isolated.manifest_sha, isolated.rc)


def test_dangling_symlink_is_an_output_collision(isolated):
    complete_workers(isolated)
    target = coordinate.paths("04")["aggregate"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(target.parent / "missing.json")
    with pytest.raises(RuntimeError, match="fresh"):
        coordinate.preflight("04", isolated.manifest, isolated.manifest_sha, isolated.rc)


def test_commands_reuse_frozen_merge_and_production_arguments(isolated):
    merge = coordinate.merge_command("04", isolated.rc)
    final = coordinate.final_command("04", isolated.policy, isolated.rc)
    assert merge[1] == str(isolated.root / "tools/merge_shards.py")
    assert merge.count("--shard") == 2
    assert merge[merge.index("--output-target") + 1] == str(coordinate.paths("04")["raw"])
    assert final[1] == str(isolated.overlay / "final_only.py")
    assert final[final.index("--renderer") + 1] == "threaded"
    assert final[final.index("--compose-workers") + 1] == "4"
    assert final[final.index("--encoder") + 1] == "libx265"


def test_coordinate_happy_path_records_children_and_final_only_pass(isolated, monkeypatch):
    complete_workers(isolated)
    calls = []

    environments = []

    def fake_child(command, rc, on_started, on_finished, env_overrides=None):
        pid = 7000 + len(calls)
        on_started(pid)
        calls.append(command)
        environments.append(env_overrides)
        p = coordinate.paths("04")
        if "merge_shards.py" in command[1]:
            p["aggregate"].parent.mkdir(parents=True, exist_ok=True)
            p["aggregate"].write_text("{}")
        else:
            p["final"].parent.mkdir(parents=True, exist_ok=True)
            p["final"].write_text("video")
            stat = p["final"].stat()
            identity = {"path": str(p["final"].resolve()), "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns}
            output = {"width": 15360, "height": 8640, "frame_count": 1952,
                      "fps": 16.0, "decoded_frame_count_performed": False}
            p["manifest"].parent.mkdir(parents=True, exist_ok=True)
            p["manifest"].write_text(json.dumps({
                "status": "PASS", "video_id": "04",
                "encoding": {"raw_status": "SKIPPED", "raw_16k_intermediate_video": False,
                             "raw_skip_reason": "user_authorized_final_only_no_dependency",
                             "renderer": "threaded", "compose_workers": 4,
                             "frames_written": 1952},
                "output": output, "output_file": identity,
                "verification_policy": {
                    "full_16k_decode_performed": False,
                    "final_output": {"file": identity, "encoder_frames_written": 1952,
                                     "encoder_exit_code": 0, "audio_mux_exit_code": 0,
                                     "nonzero_file": True, "header_probe": output,
                                     "decoded_full_frame_count": False},
                },
            }))
        on_finished(0, None)
        return 0

    monkeypatch.setattr(coordinate, "run_child", fake_child)
    result = coordinate.coordinate("04", isolated.manifest, isolated.manifest_sha,
                                   poll_seconds=0, rc=isolated.rc)
    assert result["status"] == "PASS"
    assert len(calls) == 2
    assert environments == [None, {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}]
    status = json.loads((coordinate.STATUS_ROOT / "video04.json").read_text())
    assert status["status"] == "PASS"
    assert [(item["role"], item["pid"]) for item in status["children"]] == [
        ("merge", 7000), ("final_only", 7001)]
    assert status["postprocess_policy"] == isolated.policy
    assert not coordinate.paths("04")["raw"].exists()


def test_final_identity_mismatch_is_rejected(isolated):
    p = coordinate.paths("03")
    p["final"].parent.mkdir(parents=True, exist_ok=True)
    p["final"].write_bytes(b"real output")
    stat = p["final"].stat()
    identity = {"path": str(p["final"].resolve()), "size": stat.st_size + 1,
                "mtime_ns": stat.st_mtime_ns}
    output = {"width": 15360, "height": 8640, "frame_count": 1952, "fps": 16.0,
              "decoded_frame_count_performed": False}
    p["manifest"].parent.mkdir(parents=True, exist_ok=True)
    p["manifest"].write_text(json.dumps({
        "status": "PASS", "video_id": "03", "output": output, "output_file": identity,
        "encoding": {"raw_status": "SKIPPED", "raw_16k_intermediate_video": False,
                     "raw_skip_reason": "user_authorized_final_only_no_dependency",
                     "renderer": "threaded", "compose_workers": 4,
                     "frames_written": 1952},
        "verification_policy": {"full_16k_decode_performed": False,
                                "final_output": {"file": identity}},
    }))
    with pytest.raises(RuntimeError, match="identity differs"):
        coordinate.validate_final("03", isolated.policy)


def test_interrupted_child_cleans_only_owned_new_session_pgid(isolated, monkeypatch):
    events = []

    class Child:
        pid = 43210
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if timeout is None:
                raise coordinate.CoordinatorInterrupted("test interrupt")
            self.returncode = -signal.SIGTERM
            return self.returncode

    def fake_popen(command, **kwargs):
        events.append(("popen", kwargs))
        return Child()

    monkeypatch.setattr(coordinate.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(coordinate.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(coordinate.os, "killpg",
                        lambda pgid, signum: events.append(("killpg", pgid, signum)))
    finished = []
    with pytest.raises(coordinate.CoordinatorInterrupted):
        coordinate.run_child(["fake"], isolated.rc, lambda pid: events.append(("start", pid)),
                             lambda code, cleanup: finished.append((code, cleanup)))
    assert events[0][1]["start_new_session"] is True
    assert ("killpg", 43210, signal.SIGTERM) in events
    assert all(event[1] == 43210 for event in events if event[0] == "killpg")
    assert finished[0][0] == -signal.SIGTERM
    assert finished[0][1]["action"] == "SIGTERM_OWNED_CHILD_PGID"
