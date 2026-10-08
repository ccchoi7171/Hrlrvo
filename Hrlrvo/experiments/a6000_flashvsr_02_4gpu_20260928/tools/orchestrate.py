#!/usr/bin/env python3
"""Fail-closed four-GPU orchestration for isolated video-02 FlashVSR.

The completed 03/04 deployment is used read-only for its environment, packaged
model/runtime bytes, and proven A6000 compatibility.  This root owns every
request, status, log, scratch artifact, aggregate, and final output for video02.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = Path("/home/oem/hrlrvo/experiments/a6000_flashvsr_0304_20260926")
REMOTE_CONTROL = ORIGIN / "tools/remote_control.py"
REMOTE_CONTROL_SHA256 = "36d55fb5045c274b241f5aad5a3d6f98d0d70d20923f5d251dd2a57ee4683298"
PYTHON = ORIGIN / "env/bin/python"
SESSION = "flashvsr_02_4gpu_20260928"
VIDEO = "02"
BASE_SEED = 25621683
ASSIGNMENTS = {0: (0, 42), 1: (43, 85), 2: (86, 127), 3: (128, 169)}
EXPECTED_UPSCALE_SHA256 = "3db9c0524418070a87b26b948612f8988cd47f007c6e7739a89d3911ed0d703d"
MIN_FREE_BYTES = 500 * 1024**3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_origin_control() -> Any:
    require(digest(REMOTE_CONTROL) == REMOTE_CONTROL_SHA256, "03/04 controller bytes changed")
    spec = importlib.util.spec_from_file_location("_frozen_0304_remote_control", REMOTE_CONTROL)
    require(spec is not None and spec.loader is not None, "cannot import frozen controller")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paths() -> dict[str, Path]:
    return {
        "input": ROOT / "inputs/02_Outpaint_4K.mp4",
        "anchor": ROOT / "inputs/02_Input_8K.mp4",
        "work": ROOT / "runs/02",
        "aggregate": ROOT / "runs/02/aggregate_manifest.json",
        "raw": ROOT / "runs/02/raw_flashvsr/02_FlashVSR_raw_16K_16fps.mp4",
        "final": ROOT / "outputs/02/final/02_Certification_Final_16K_16fps.mp4",
        "manifest": ROOT / "outputs/02/metadata/final_manifest.json",
    }


def runtime_env() -> dict[str, str]:
    return load_origin_control().runtime_env()


def verify_deployment(*, full_environment_gate: bool) -> dict[str, Any]:
    deployment_path = ROOT / "deployment/deployment_manifest.json"
    equivalence_path = ROOT / "verification/final_only_equivalence.json"
    require(deployment_path.is_file(), "deployment manifest is missing")
    deployment = read(deployment_path)
    require(deployment.get("status") == "PASS" and deployment.get("video_id") == VIDEO,
            "deployment manifest is not PASS for video02")
    require(deployment.get("base_seed") == BASE_SEED, "deployment seed mismatch")
    require(deployment.get("assignments") == {str(k): list(v) for k, v in ASSIGNMENTS.items()},
            "deployment tile assignments differ")
    require(read(equivalence_path).get("status") == "PASS", "final-only equivalence gate is not PASS")
    evidence = deployment.get("equivalence_evidence", {})
    require(evidence.get("path") == str(equivalence_path)
            and evidence.get("sha256") == digest(equivalence_path),
            "final-only equivalence evidence identity changed")
    for name, path in (("input", paths()["input"]), ("anchor", paths()["anchor"])):
        identity = deployment[name]
        require(path.is_file() and path.stat().st_size == identity["size"], f"{name} size mismatch")
        require(digest(path) == identity["sha256"], f"{name} SHA256 mismatch")
    package_src = ROOT / "package/src"
    for name in ("flashvsr", "models", "video_outpainting"):
        link = package_src / name
        expected = ORIGIN / "package/src" / name
        require(link.is_symlink() and link.resolve() == expected.resolve(), f"wrong read-only package link: {link}")
    for link, expected in ((ROOT / "env", ORIGIN / "env"),
                           (ROOT / "runtime", ORIGIN / "runtime")):
        require(link.is_symlink() and link.resolve() == expected.resolve(), f"wrong read-only runtime link: {link}")
    upscale = ROOT / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py"
    require(digest(upscale) == EXPECTED_UPSCALE_SHA256, "approved upscale snapshot changed")
    code = deployment.get("code_sha256", {})
    for relative, wanted in code.items():
        require(digest(ROOT / relative) == wanted, f"deployed code changed: {relative}")
    free = shutil.disk_usage(ROOT).free
    require(free >= MIN_FREE_BYTES, f"free space {free} below 500 GiB launch floor")
    if full_environment_gate:
        load_origin_control().check_gate(smoke=True)
    return {"deployment": deployment, "free_bytes": free}


def worker_command(gpu: int) -> list[str]:
    start, end = ASSIGNMENTS[gpu]
    p = paths()
    return [
        str(PYTHON), "-I", str(ROOT / "tools/tile_worker.py"),
        "--video-id", VIDEO, "--input", str(p["input"]), "--anchor", str(p["anchor"]),
        "--gpu-index", str(gpu), "--tile-start", str(start), "--tile-end", str(end),
        "--work-dir", str(p["work"] / f"gpu{gpu}"), "--output-target", str(p["raw"]),
        "--mode", "full",
    ]


def merge_command() -> list[str]:
    p = paths()
    command = [str(PYTHON), "-I", str(ROOT / "tools/merge_shards.py"),
               "--video-id", VIDEO, "--work-dir", str(p["work"]),
               "--output-manifest", str(p["aggregate"]), "--output-target", str(p["raw"])]
    for gpu in ASSIGNMENTS:
        command.extend(["--shard", str(p["work"] / f"gpu{gpu}/shard_receipt.json")])
    return command


def final_command() -> list[str]:
    p = paths()
    return [
        str(PYTHON), "-I", str(ROOT / "final_only/final_only.py"),
        "--video-id", VIDEO, "--input-4k", str(p["input"]), "--original-8k", str(p["anchor"]),
        "--model-root", str(ROOT / "package/src/models/flashvsr_v1_1"),
        "--flashvsr-reference", str(ROOT / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py"),
        "--outpainting-config", str(ROOT / "package/src/video_outpainting/main/configs/inference_4k.yaml"),
        "--output", str(p["final"]), "--raw-output", str(p["raw"]),
        "--final-manifest", str(p["manifest"]), "--work-dir", str(p["work"]),
        "--expected-frames", "1952", "--tile-size", "256", "--tile-overlap", "32",
        "--min-tile-size", "256", "--sparse-ratio", "2.0", "--local-range", "11",
        "--seed", str(BASE_SEED), "--ffmpeg-threads", "32", "--encoder", "libx265",
        "--aggregate-manifest", str(p["aggregate"]), "--renderer", "threaded",
        "--compose-workers", "8",
    ]


def await_barrier(role: str, token: str) -> None:
    reservation = read(ROOT / "status/launch_reservation.json")
    require(token and reservation.get("token") == token, "launch token mismatch")
    write(ROOT / f"status/ready_{role}.json", {"pid": os.getpid(), "token": token, "time_utc": now()})
    deadline = time.monotonic() + 600
    barrier = ROOT / "status/start_barrier.json"
    while not barrier.is_file():
        require(time.monotonic() < deadline, "launch barrier timeout")
        time.sleep(0.5)
    require(read(barrier).get("token") == token, "barrier token mismatch")


def supervise(gpu: int, token: str) -> None:
    await_barrier(f"gpu{gpu}", token)
    command = worker_command(gpu)
    work = paths()["work"] / f"gpu{gpu}"
    work.mkdir(parents=True, exist_ok=False)
    record = {"video_id": VIDEO, "gpu": gpu, "assignment": list(ASSIGNMENTS[gpu]),
              "supervisor_pid": os.getpid(), "command": command, "started_utc": now()}
    with (work / "supervisor.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=runtime_env(),
                                 start_new_session=True)
        write(ROOT / f"status/gpu{gpu}.json", {**record, "status": "RUNNING", "worker_pid": child.pid})
        code = child.wait()
    passed = code == 0 and (work / "shard_receipt.json").is_file()
    write(ROOT / f"status/gpu{gpu}.json", {**record, "status": "COMPLETE" if passed else "FAILED",
          "worker_pid": child.pid, "exit_code": code, "finished_utc": now()})
    require(passed, f"GPU{gpu} shard failed with exit {code}")


def coordinate(token: str) -> None:
    await_barrier("coordinator", token)
    status_path = ROOT / "status/video02.json"
    record: dict[str, Any] = {"video_id": VIDEO, "coordinator_pid": os.getpid(),
                              "started_utc": now(), "status": "WAITING_FOR_FOUR_SHARDS"}
    write(status_path, record)
    try:
        while True:
            states = [read(ROOT / f"status/gpu{gpu}.json")
                      if (ROOT / f"status/gpu{gpu}.json").is_file() else {} for gpu in ASSIGNMENTS]
            failed = [state for state in states if state.get("status") == "FAILED"]
            require(not failed, "at least one shard failed; final stage is forbidden")
            if all(state.get("status") == "COMPLETE" for state in states):
                break
            record.update({"status": "WAITING_FOR_FOUR_SHARDS", "worker_states": states,
                           "updated_utc": now()})
            write(status_path, record)
            time.sleep(20)
        record.update({"status": "VALIDATING_AND_MERGING_170_TILES", "updated_utc": now()})
        write(status_path, record)
        merged = subprocess.run(merge_command(), env=runtime_env(), check=False)
        require(merged.returncode == 0 and paths()["aggregate"].is_file(), "four-shard merge failed")
        record.update({"status": "ENCODING_FINAL_ONLY", "updated_utc": now(),
                       "aggregate_manifest": str(paths()["aggregate"])})
        write(status_path, record)
        final = subprocess.run(final_command(), env={**runtime_env(), "OMP_NUM_THREADS": "1",
                               "MKL_NUM_THREADS": "1"}, check=False)
        require(final.returncode == 0, "final-only composition/encoding failed")
        manifest = read(paths()["manifest"])
        require(manifest.get("status") == "PASS" and manifest.get("video_id") == VIDEO,
                "final manifest is not PASS for video02")
        record.update({"status": "PASS", "finished_utc": now(), "updated_utc": now(),
                       "output": str(paths()["final"]), "final_manifest": str(paths()["manifest"])})
        write(status_path, record)
    except BaseException as error:
        record.update({"status": "FAILED", "finished_utc": now(), "updated_utc": now(),
                       "error": f"{type(error).__name__}: {error}"})
        write(status_path, record)
        raise


def launch() -> None:
    proof = verify_deployment(full_environment_gate=True)
    require(subprocess.run(["tmux", "has-session", "-t", SESSION], capture_output=True, check=False).returncode != 0,
            f"tmux session already exists: {SESSION}")
    p = paths()
    collisions = [path for path in (p["work"], p["aggregate"], p["raw"], p["final"], p["manifest"])
                  if path.exists() or path.is_symlink()]
    require(not collisions, "fresh-only launch collision: " + ", ".join(map(str, collisions)))
    (ROOT / "status").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    write(ROOT / "status/launch_reservation.json", {"token": token, "time_utc": now(),
          "launcher_pid": os.getpid(), "free_bytes": proof["free_bytes"]})
    roles = [(f"gpu{gpu}", ["supervise", "--gpu", str(gpu)]) for gpu in ASSIGNMENTS]
    roles.append(("coordinator", ["coordinate"]))
    launched = []
    for index, (name, subcommand) in enumerate(roles):
        command = [str(PYTHON), "-I", str(Path(__file__).resolve()), *subcommand,
                   "--launch-token", token]
        log = ROOT / "logs" / f"{name}.log"
        shell = shlex.join(command) + " >> " + shlex.quote(str(log)) + " 2>&1"
        tmux = ["tmux", "new-session", "-d", "-s", SESSION] if index == 0 else [
            "tmux", "new-window", "-t", SESSION]
        subprocess.run([*tmux, "-n", name, shell], check=True)
        subprocess.run(["tmux", "set-option", "-w", "-t", f"{SESSION}:{name}",
                        "remain-on-exit", "on"], check=True)
        launched.append({"window": f"{SESSION}:{name}", "command": command, "log": str(log)})
    deadline = time.monotonic() + 120
    ready = [ROOT / f"status/ready_{name}.json" for name, _ in roles]
    while not all(path.is_file() and read(path).get("token") == token for path in ready):
        require(time.monotonic() < deadline, "not all five windows reached launch barrier")
        time.sleep(0.5)
    verify_deployment(full_environment_gate=False)
    write(ROOT / "status/start_barrier.json", {"token": token, "released_utc": now()})
    deadline = time.monotonic() + 120
    while True:
        states = [read(ROOT / f"status/gpu{gpu}.json")
                  if (ROOT / f"status/gpu{gpu}.json").is_file() else {} for gpu in ASSIGNMENTS]
        require(not any(item.get("status") == "FAILED" for item in states), "worker failed at startup")
        if all(item.get("status") in {"RUNNING", "COMPLETE"} for item in states):
            break
        require(time.monotonic() < deadline, "workers did not reach RUNNING state")
        time.sleep(0.5)
    write(ROOT / "status/launch.json", {"status": "LAUNCHED", "time_utc": now(),
          "session": SESSION, "video_id": VIDEO, "base_seed": BASE_SEED,
          "assignments": {str(k): list(v) for k, v in ASSIGNMENTS.items()}, "windows": launched})
    print("VIDEO02_FOUR_GPU_LAUNCHED", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "launch", "supervise", "coordinate"])
    parser.add_argument("--gpu", type=int, choices=range(4))
    parser.add_argument("--launch-token")
    args = parser.parse_args()
    try:
        if args.action == "preflight":
            print(json.dumps(verify_deployment(full_environment_gate=True), default=str))
        elif args.action == "launch":
            launch()
        elif args.action == "supervise":
            require(args.gpu is not None, "--gpu is required")
            supervise(args.gpu, args.launch_token or "")
        else:
            coordinate(args.launch_token or "")
        return 0
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
