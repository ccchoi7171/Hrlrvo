#!/usr/bin/env python3
"""Self-pinned final-only coordinator for one completed 03/04 tile pair."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator


OVERLAY_ROOT = Path(__file__).resolve().parent
ROOT = OVERLAY_ROOT.parents[1]
REMOTE_CONTROL = ROOT / "tools/remote_control.py"
REMOTE_CONTROL_SHA256 = "36d55fb5045c274b241f5aad5a3d6f98d0d70d20923f5d251dd2a57ee4683298"
BASELINE_CODE_MANIFEST_SHA256 = "5f2a5784c6347db524d5665ada9447cd9dc4f809d7a31d5be46aafe67350fd60"
OVERLAY_SCHEMA = "a6000_final_only_overlay/v2"
STATUS_ROOT = ROOT / "status/final_only_v2"
GPU_ASSIGNMENTS = {"04": (0, 1), "03": (2, 3)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def inside(path: Path, root: Path | None = None) -> Path:
    root = ROOT if root is None else root
    resolved = path.resolve()
    require(resolved.is_relative_to(root.resolve()), f"Path escapes isolated run: {path}")
    return resolved


def load_remote_control() -> Any:
    require(digest(REMOTE_CONTROL) == REMOTE_CONTROL_SHA256,
            "Frozen remote controller SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("_a6000_frozen_remote_control_v1", REMOTE_CONTROL)
    require(spec is not None and spec.loader is not None, "Cannot load frozen remote controller")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == REMOTE_CONTROL.resolve(), "Wrong remote controller loaded")
    return module


def validate_policy(policy: Any) -> dict[str, Any]:
    require(isinstance(policy, dict), "Overlay manifest policy is missing")
    renderer = policy.get("renderer")
    workers = policy.get("compose_workers")
    maximum = policy.get("max_parallel_videos")
    require(renderer in {"original", "threaded"}, "Unsupported renderer policy")
    require(isinstance(workers, int) and not isinstance(workers, bool) and 1 <= workers <= 16,
            "compose_workers must be an integer in [1,16]")
    require(isinstance(maximum, int) and not isinstance(maximum, bool) and maximum in {1, 2},
            "max_parallel_videos must be integer 1 or 2")
    return {"renderer": renderer, "compose_workers": workers, "max_parallel_videos": maximum}


def verify_overlay(manifest_path: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    manifest_path = inside(manifest_path)
    require(len(expected_manifest_sha256) == 64, "Caller must pin the overlay manifest SHA256")
    require(digest(manifest_path) == expected_manifest_sha256, "Overlay manifest SHA256 mismatch")
    manifest = read(manifest_path)
    require(manifest.get("schema") == OVERLAY_SCHEMA, "Unsupported overlay manifest schema")
    require(manifest.get("baseline_code_manifest_sha256") == BASELINE_CODE_MANIFEST_SHA256,
            "Overlay cites the wrong frozen code manifest")
    code_manifest = ROOT / "verification/code_manifest.json"
    require(digest(code_manifest) == BASELINE_CODE_MANIFEST_SHA256,
            "Frozen code manifest changed")

    expected_files = {"final_only.py", "parallel_renderer.py", "coordinate_v2.py"}
    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == expected_files,
            "Overlay file proof must cover exactly the three v2 scripts")
    for name, wanted in files.items():
        require(isinstance(wanted, str) and len(wanted) == 64, f"Invalid overlay SHA: {name}")
        require(digest(OVERLAY_ROOT / name) == wanted, f"Overlay script changed: {name}")

    source = manifest.get("source_proof")
    require(isinstance(source, dict), "Overlay manifest lacks source proof")
    source_path = inside(ROOT / source.get("path", ""))
    require(source_path == (ROOT / "verification/source_transfer_verification.json").resolve(),
            "Overlay source proof path is not the approved transfer receipt")
    require(digest(source_path) == source.get("sha256"), "Source proof changed or is stale")
    require(read(source_path).get("status") == "PASS", "Source proof is not PASS")

    benchmark = manifest.get("benchmark")
    require(isinstance(benchmark, dict), "Overlay manifest lacks benchmark proof")
    benchmark_path = inside(ROOT / benchmark.get("path", ""))
    require(digest(benchmark_path) == benchmark.get("sha256"), "Benchmark proof changed or is stale")
    decision = read(benchmark_path)
    policy = validate_policy(manifest.get("policy"))
    require(decision.get("status") == "PASS", "Benchmark decision is not PASS")
    require(decision.get("pixel_equality_pass") is True, "Benchmark lacks exact pixel-equality PASS")
    require(decision.get("selected_policy") == policy,
            "Benchmark selected policy differs from overlay policy")
    evidence = decision.get("evidence")
    require(isinstance(evidence, dict), "Benchmark lacks measured evidence")
    concurrency = evidence.get("concurrent_video_comparison")
    require(isinstance(concurrency, dict), "Benchmark lacks concurrent-video comparison")
    serial = concurrency.get("serial_seconds")
    parallel = concurrency.get("parallel_seconds")
    gain = concurrency.get("makespan_gain_percent")
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (serial, parallel, gain)), "Benchmark concurrency values are not numeric")
    require(serial > 0 and parallel > 0, "Benchmark concurrency durations must be positive")
    calculated_gain = (float(serial) - float(parallel)) / float(serial) * 100.0
    require(abs(calculated_gain - float(gain)) <= 0.1,
            "Benchmark makespan gain is inconsistent with measured durations")
    if policy["max_parallel_videos"] == 2:
        require(float(gain) >= 10.0,
                "Parallel final encoding requires at least 10% measured makespan gain")
    limitations = decision.get("limitations")
    require(isinstance(limitations, list) and limitations,
            "Benchmark must record its short-fixture limitations")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "source_proof": {"path": str(source_path), "sha256": source["sha256"]},
        "benchmark": {"path": str(benchmark_path), "sha256": benchmark["sha256"]},
        "policy": policy,
        "files": files,
    }


def paths(video: str) -> dict[str, Path]:
    require(video in GPU_ASSIGNMENTS, "Video must be 03 or 04")
    base = ROOT / "outputs" / video
    return {
        "input": ROOT / f"package/data/02_Outpaint_4K_3840x2160/{video}/final/{video}_Outpaint_4K.mp4",
        "anchor": ROOT / f"package/data/00_Input_8K_7680x4320/{video}_Input_8K.mp4",
        "raw": ROOT / "runs" / video / f"raw_flashvsr/{video}_FlashVSR_raw_16K_16fps.mp4",
        "final": base / f"final/{video}_Certification_Final_16K_16fps.mp4",
        "manifest": base / "metadata/final_manifest.json",
        "work": ROOT / "runs" / video,
        "aggregate": ROOT / "runs" / video / "aggregate_manifest.json",
    }


def merge_command(video: str, rc: Any) -> list[str]:
    p = paths(video)
    command = [str(rc.PYTHON), str(ROOT / "tools/merge_shards.py"), "--video-id", video,
               "--work-dir", str(p["work"]), "--output-manifest", str(p["aggregate"]),
               "--output-target", str(p["raw"])]
    for gpu in GPU_ASSIGNMENTS[video]:
        command.extend(["--shard", str(p["work"] / f"gpu{gpu}/shard_receipt.json")])
    return command


def final_command(video: str, policy: dict[str, Any], rc: Any) -> list[str]:
    command = rc.postprocess_command(video)
    require(Path(command[1]).resolve() == (ROOT / "tools/postprocess_video.py").resolve(),
            "Frozen postprocess command changed unexpectedly")
    command[1] = str(OVERLAY_ROOT / "final_only.py")
    command.extend(["--renderer", policy["renderer"],
                    "--compose-workers", str(policy["compose_workers"])])
    return [str(item) for item in command]


def worker_state(video: str) -> tuple[str, list[dict[str, Any]]]:
    states = []
    for gpu in GPU_ASSIGNMENTS[video]:
        path = ROOT / "status" / f"gpu{gpu}.json"
        state = read(path) if path.is_file() else {}
        status = state.get("status")
        require(status != "FAILED", f"Video{video} GPU{gpu} shard FAILED")
        require(status in {None, "RUNNING", "COMPLETE"},
                f"Video{video} GPU{gpu} has an unknown status: {status!r}")
        if status == "COMPLETE":
            receipt = paths(video)["work"] / f"gpu{gpu}/shard_receipt.json"
            require(receipt.is_file(), f"Video{video} GPU{gpu} COMPLETE without shard receipt")
        states.append(state)
    return ("COMPLETE" if all(item.get("status") == "COMPLETE" for item in states)
            else "WAITING"), states


def require_fresh(video: str) -> None:
    p = paths(video)
    collisions = [p[name] for name in ("aggregate", "raw", "final", "manifest")
                  if p[name].exists() or p[name].is_symlink()]
    require(not collisions, "Final-only v2 requires fresh aggregate/raw/final/report targets: "
            + ", ".join(map(str, collisions)))


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Coordinator lock is already owned: {path}") from error
        yield stream
    finally:
        stream.close()


@contextlib.contextmanager
def serialized_lock(path: Path) -> Iterator[Any]:
    """Blocking global policy lock; normal cross-video contention is not failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield stream
    finally:
        stream.close()


class CoordinatorInterrupted(RuntimeError):
    pass


@contextlib.contextmanager
def coordinator_signal_handlers() -> Iterator[None]:
    previous: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise CoordinatorInterrupted(f"Coordinator received signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def terminate_owned_child(child: Any) -> dict[str, Any]:
    """Boundedly terminate only a child whose new session PGID equals its PID."""
    evidence: dict[str, Any] = {"owned_pid": child.pid, "action": "none"}
    if child.poll() is not None:
        evidence.update({"action": "already_exited", "exit_code": child.returncode})
        return evidence
    try:
        pgid = os.getpgid(child.pid)
        require(pgid == child.pid, "Refusing cleanup: child no longer owns its isolated process group")
        evidence["owned_pgid"] = pgid
        os.killpg(pgid, signal.SIGTERM)
        evidence["action"] = "SIGTERM_OWNED_CHILD_PGID"
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            require(os.getpgid(child.pid) == child.pid,
                    "Refusing SIGKILL: child process-group ownership changed")
            os.killpg(child.pid, signal.SIGKILL)
            evidence["action"] = "SIGTERM_THEN_SIGKILL_OWNED_CHILD_PGID"
            child.wait(timeout=10)
        evidence["exit_code"] = child.returncode
    except ProcessLookupError:
        evidence.update({"action": "exited_during_cleanup", "exit_code": child.poll()})
    return evidence


def run_child(command: list[str], rc: Any, on_started: Any,
              on_finished: Any,
              env_overrides: dict[str, str] | None = None) -> int:
    env = rc.runtime_env()
    if env_overrides:
        env.update(env_overrides)
    child = subprocess.Popen(command, env=env, start_new_session=True)
    on_started(child.pid)
    cleanup: dict[str, Any] | None = None
    original_error: BaseException | None = None
    try:
        return child.wait()
    except BaseException as error:
        original_error = error
        try:
            cleanup = terminate_owned_child(child)
        except BaseException as cleanup_error:
            cleanup = {"owned_pid": child.pid, "action": "cleanup_failed",
                       "error": f"{type(cleanup_error).__name__}: {cleanup_error}"}
        raise
    finally:
        try:
            on_finished(child.poll(), cleanup)
        except BaseException:
            if original_error is None:
                raise


def validate_final(video: str, policy: dict[str, Any]) -> dict[str, Any]:
    p = paths(video)
    require(p["manifest"].is_file() and not p["manifest"].is_symlink() and
            p["final"].is_file() and not p["final"].is_symlink(),
            "Final output/report was not published as regular files")
    require(not (p["raw"].exists() or p["raw"].is_symlink()),
            "Final-only path unexpectedly created the raw 16K intermediate")
    value = read(p["manifest"])
    encoding = value.get("encoding", {})
    verification = value.get("verification_policy", {})
    require(value.get("status") == "PASS" and value.get("video_id") == video,
            "Final manifest status/video mismatch")
    require(encoding.get("raw_status") == "SKIPPED" and
            encoding.get("raw_16k_intermediate_video") is False and
            encoding.get("raw_skip_reason") == "user_authorized_final_only_no_dependency",
            "Final manifest does not truthfully record final-only raw omission")
    require(encoding.get("renderer") == policy["renderer"] and
            encoding.get("compose_workers") == policy["compose_workers"],
            "Final manifest renderer policy differs from approved benchmark policy")
    require(verification.get("full_16k_decode_performed") is False,
            "Final-only manifest unexpectedly claims a full decode")
    stat = p["final"].stat()
    actual_identity = {"path": str(p["final"].resolve()), "size": stat.st_size,
                       "mtime_ns": stat.st_mtime_ns}
    require(actual_identity["size"] > 0 and value.get("output_file") == actual_identity,
            "Final output identity differs from the manifest")
    output = value.get("output", {})
    require((output.get("width"), output.get("height"), output.get("frame_count")) ==
            (15360, 8640, 1952) and float(output.get("fps", -1)) == 16.0 and
            output.get("decoded_frame_count_performed") is False,
            "Final output header evidence is incomplete or wrong")
    final_evidence = verification.get("final_output", {})
    require(final_evidence.get("file") == actual_identity and
            final_evidence.get("encoder_frames_written") == 1952 and
            final_evidence.get("encoder_exit_code") == 0 and
            final_evidence.get("audio_mux_exit_code") == 0 and
            final_evidence.get("nonzero_file") is True and
            final_evidence.get("header_probe") == output and
            final_evidence.get("decoded_full_frame_count") is False,
            "Bounded final-output evidence does not match the published file")
    require(encoding.get("frames_written") == 1952,
            "Final encoder producer frame counter is incomplete")
    return value


class Status:
    def __init__(self, video: str):
        self.path = STATUS_ROOT / f"video{video}.json"
        require(not self.path.exists(), f"Final-only v2 status already exists: {self.path}")
        self.value = {
            "schema": "a6000_final_only_coordinator/v2", "video": video,
            "coordinator_pid": os.getpid(), "parent_pid": os.getppid(),
            "started_utc": now(), "status": "RUNNING", "children": [],
        }
        write(self.path, self.value)

    def update(self, status: str, **fields: Any) -> None:
        self.value.update(fields)
        self.value["status"] = status
        self.value["updated_utc"] = now()
        write(self.path, self.value)

    def child(self, role: str, pid: int) -> None:
        self.value["children"].append({"role": role, "pid": pid, "started_utc": now()})
        self.update(self.value["status"], active_child_pid=pid, active_child_role=role)

    def child_finished(self, role: str, code: int | None,
                       cleanup: dict[str, Any] | None = None) -> None:
        for child in reversed(self.value["children"]):
            if child["role"] == role and "exit_code" not in child:
                child.update({"exit_code": code, "finished_utc": now()})
                if cleanup is not None:
                    child["cleanup"] = cleanup
                break
        self.update(self.value["status"], active_child_pid=None, active_child_role=None)


def preflight(video: str, manifest_path: Path, manifest_sha: str, rc: Any | None = None) -> dict[str, Any]:
    rc = rc or load_remote_control()
    with exclusive_lock(STATUS_ROOT / f"video{video}.lock"):
        require(not (STATUS_ROOT / f"video{video}.json").exists(),
                "Final-only v2 status already exists; inspect it instead of overwriting")
        proof = verify_overlay(manifest_path, manifest_sha)
        rc.check_gate(smoke=True)
        require_fresh(video)
        readiness, states = worker_state(video)
        return {"status": "PASS", "video": video, "readiness": readiness,
                "worker_states": states, "policy": proof["policy"], "overlay": proof}


def coordinate(video: str, manifest_path: Path, manifest_sha: str,
               *, poll_seconds: float = 20.0, rc: Any | None = None) -> dict[str, Any]:
    rc = rc or load_remote_control()
    with exclusive_lock(STATUS_ROOT / f"video{video}.lock"):
        status = Status(video)
        with coordinator_signal_handlers():
            try:
                proof = verify_overlay(manifest_path, manifest_sha)
                rc.check_gate(smoke=True)
                require_fresh(video)
                commands = {"merge": merge_command(video, rc),
                            "final_only": final_command(video, proof["policy"], rc)}
                status.update("RUNNING", overlay=proof, postprocess_policy=proof["policy"],
                              commands=commands)
                while True:
                    readiness, states = worker_state(video)
                    if readiness == "COMPLETE":
                        break
                    status.update("WAITING_FOR_TWO_SHARDS", worker_states=states)
                    time.sleep(poll_seconds)

                if proof["policy"]["max_parallel_videos"] == 1:
                    status.update("WAITING_FOR_SERIAL_POSTPROCESS_LOCK")
                lock = (serialized_lock(STATUS_ROOT / "parallel_policy_1.lock")
                        if proof["policy"]["max_parallel_videos"] == 1 else contextlib.nullcontext())
                with lock:
                    # Outputs and immutable gates are rechecked after any wait and
                    # immediately before merge/final publication.
                    require_fresh(video)
                    proof = verify_overlay(manifest_path, manifest_sha)
                    rc.check_gate(smoke=True)
                    status.update("MERGING_GLOBAL_170_TILE_LAYOUT")
                    code = run_child(
                        commands["merge"], rc, lambda pid: status.child("merge", pid),
                        lambda code, cleanup: status.child_finished("merge", code, cleanup),
                    )
                    require(code == 0 and paths(video)["aggregate"].is_file(),
                            f"Global shard merge failed, exit={code}")

                    proof = verify_overlay(manifest_path, manifest_sha)
                    rc.check_gate(smoke=True)
                    p = paths(video)
                    require(not any(p[name].exists() or p[name].is_symlink()
                                    for name in ("raw", "final", "manifest")),
                            "Final target collision appeared after aggregate merge")
                    status.update("ENCODING_FINAL_ONLY", aggregate=str(p["aggregate"]))
                    code = run_child(
                        commands["final_only"], rc,
                        lambda pid: status.child("final_only", pid),
                        lambda code, cleanup: status.child_finished("final_only", code, cleanup),
                        {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    )
                    require(code == 0, f"Final-only postprocess failed, exit={code}")
                    verify_overlay(manifest_path, manifest_sha)
                    rc.check_gate(smoke=True)
                    result = validate_final(video, proof["policy"])
                status.update("PASS", finished_utc=now(), output=str(paths(video)["final"]),
                              final_manifest=str(paths(video)["manifest"]), active_child_pid=None,
                              active_child_role=None)
                return result
            except BaseException as error:
                status.update("FAILED", finished_utc=now(),
                              error=f"{type(error).__name__}: {error}")
                raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=["preflight", "run"])
    value.add_argument("--video", choices=["03", "04"], required=True)
    value.add_argument("--overlay-manifest", type=Path, required=True)
    value.add_argument("--overlay-manifest-sha256", required=True)
    value.add_argument("--poll-seconds", type=float, default=20.0)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        if args.action == "preflight":
            result = preflight(args.video, args.overlay_manifest, args.overlay_manifest_sha256)
            print(json.dumps(result, ensure_ascii=False))
        else:
            coordinate(args.video, args.overlay_manifest, args.overlay_manifest_sha256,
                       poll_seconds=args.poll_seconds)
        return 0
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
