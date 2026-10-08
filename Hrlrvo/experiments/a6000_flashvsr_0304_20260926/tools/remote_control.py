#!/usr/bin/env python3
"""Fail-closed control plane for this isolated 03/04 A6000 run only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "env/bin/python"
SESSION = "flashvsr_0304_20260926"
SOURCE_SHA = "83be683005f3817e71c0083305371eca216634f7f38f6d84f213286fbee3ff74"
MODEL_COMMIT = "cf910c61a60733e610e9c6e8b607f80c3a6c202b"
CHECKPOINT_REVISION = "27561b186ded3402d7c975f4fd722e2885b6135f"
BSA_COMMIT = "49d6c39e4dc0303442cda3bb758b3925d4399c49"
UPSCALER_SHA = "3db9c0524418070a87b26b948612f8988cd47f007c6e7739a89d3911ed0d703d"
GPU_UUIDS = [
    "GPU-663910bc-7de7-0749-de7f-593439c81349",
    "GPU-fbc56149-7397-6edf-4b3e-3d3619470ec2",
    "GPU-1e943e02-7abf-2f73-9f52-dfb8202d7a46",
    "GPU-89722209-d8f0-c647-98cf-a60c3c4beb35",
]
ASSIGNMENTS = {0: ("04", 0, 84), 1: ("04", 85, 169),
               2: ("03", 0, 84), 3: ("03", 85, 169)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def same_gpu_uuid(actual, expected):
    # PyTorch 2.6 returns the bare UUID; nvidia-smi includes the GPU- prefix.
    return str(actual).removeprefix("GPU-") == str(expected).removeprefix("GPU-")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def identity(path):
    s = Path(path).stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns,
            "ctime_ns": s.st_ctime_ns, "inode": s.st_ino, "device": s.st_dev}


def inside(path, root=ROOT):
    resolved = Path(path).resolve()
    require(resolved.is_relative_to(root.resolve()), f"Path escapes isolated run: {path}")
    return resolved


def runtime_env():
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": ",".join(GPU_UUIDS),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8"})
    env["PATH"] = f"{ROOT}/runtime/ffmpeg-6.1.1/bin:{PYTHON.parent}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = str(ROOT / "env/lib")
    env.pop("PYTHONPATH", None)
    env.pop("LD_PRELOAD", None)
    return env


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True,
                          env=runtime_env(), text=True, **kwargs)


def verify_assets(expected_code_manifest_sha256, expected_runtime_manifest_sha256):
    source_path = ROOT / "verification/source_provenance.json"
    require(digest(source_path) == SOURCE_SHA, "Approved H200 provenance manifest changed")
    source = read(source_path)
    require(source["upstream_model_commit"] == MODEL_COMMIT, "Wrong model commit")
    require(source["block_sparse_commit"] == BSA_COMMIT, "Wrong BSA commit")
    expected = {"package/" + f["path"]: f for f in source["files"]}
    trusted_manifests = {"code_manifest.json": expected_code_manifest_sha256,
                         "ffmpeg_bundle_manifest.json": expected_runtime_manifest_sha256}
    for name, trusted_hash in trusted_manifests.items():
        require(isinstance(trusted_hash, str) and len(trusted_hash) == 64,
                f"Caller must supply trusted LOCAL digest for {name}")
        require(digest(ROOT / "verification" / name) == trusted_hash,
                f"Transferred deployment manifest differs from local approval: {name}")
        expected.update(read(ROOT / "verification" / name)["files"])
    checked = {}
    for relative, wanted in expected.items():
        path = inside(ROOT / relative)
        before = identity(path)
        require(before["size"] == wanted["bytes"], f"Wrong file size: {relative}")
        require(digest(path) == wanted["sha256"], f"SHA256 mismatch: {relative}")
        require(identity(path) == before, f"File changed during verification: {relative}")
        checked[relative] = {"identity": before, "sha256": wanted["sha256"]}
    info = read(ROOT / "package/src/models/flashvsr_v1_1/MODEL_INFO.json")
    require(info["repository_commit"] == MODEL_COMMIT, "MODEL_INFO commit mismatch")
    require(info["checkpoint_revision"] == CHECKPOINT_REVISION, "Checkpoint revision mismatch")
    require(info["native_dependency"]["commit"] == BSA_COMMIT, "MODEL_INFO BSA mismatch")
    for name, wanted_hash in info["checkpoint_sha256"].items():
        key = "package/src/models/flashvsr_v1_1/checkpoint/" + name
        require(checked[key]["sha256"] == wanted_hash, "Checkpoint identity mismatch")
    write(ROOT / "verification/source_transfer_verification.json", {
        "status": "PASS", "time_utc": now(), "source_manifest_sha256": SOURCE_SHA,
        "model_commit": MODEL_COMMIT, "checkpoint_revision": CHECKPOINT_REVISION,
        "block_sparse_commit": BSA_COMMIT, "files": checked,
        "manifests": {name: digest(ROOT / "verification" / name) for name in
                      ("source_provenance.json", "code_manifest.json", "ffmpeg_bundle_manifest.json")},
    })
    print(f"ASSET_SHA_PASS files={len(checked)}", flush=True)


def check_assets():
    proof = read(ROOT / "verification/source_transfer_verification.json")
    require(proof.get("status") == "PASS" and proof.get("source_manifest_sha256") == SOURCE_SHA,
            "Missing approved transfer verification")
    for name, wanted in proof["manifests"].items():
        require(digest(ROOT / "verification" / name) == wanted, f"Manifest changed: {name}")
    for relative, checked in proof["files"].items():
        require(identity(inside(ROOT / relative)) == checked["identity"],
                f"Verified asset changed: {relative}")
    return proof


def environment_preflight():
    check_assets()
    require(Path(sys.executable).resolve() == PYTHON.resolve(), "Controller must use the isolated packed Python")
    import torch
    import block_sparse_attn
    import block_sparse_attn_cuda
    for module in (torch, block_sparse_attn, block_sparse_attn_cuda):
        require(Path(module.__file__).resolve().is_relative_to((ROOT / "env").resolve()),
                f"Module imported outside isolated environment: {module.__file__}")
    native_build = read(ROOT / "verification/bsa_ampere_build.json")
    require(native_build["build_exit_code"] == 0 and
            native_build["source_provenance"]["claimed_commit"] == BSA_COMMIT,
            "Native Ampere build lacks approved source provenance")
    require(digest(Path(block_sparse_attn_cuda.__file__)) == native_build["extension"]["sha256"],
            "Installed attention extension differs from approved Ampere build")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "torchaudio", "triton", "block-sparse-attn")}
    require(versions == {"torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124",
                         "torchaudio": "2.6.0+cu124", "triton": "3.2.0",
                         "block-sparse-attn": "0.0.2"}, f"Environment mismatch: {versions}")
    all_versions = installed_versions()
    require(all_versions == read(ROOT / "verification/source_environment_versions.json"),
            "Complete installed package versions differ from the packed H200 environment")
    require(torch.cuda.device_count() == 4, "Exactly four assigned CUDA devices are required")
    gpu_rows = []
    for index in range(4):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        require(name == "NVIDIA RTX A6000" and capability == (8, 6), f"Wrong GPU: {index}/{name}")
        actual_uuid = str(torch.cuda.get_device_properties(index).uuid)
        require(same_gpu_uuid(actual_uuid, GPU_UUIDS[index]),
                f"CUDA physical/logical UUID mismatch: {index}/{actual_uuid}")
        with torch.cuda.device(index):
            require(torch.cuda.is_bf16_supported(), "BF16 not supported")
        gpu_rows.append({"logical_index": index, "uuid": GPU_UUIDS[index],
                         "name": name, "capability": list(capability)})
    ffmpeg = run(["ffmpeg", "-version"], capture_output=True).stdout.splitlines()[0]
    require("6.1.1-3ubuntu5" in ffmpeg, "Not the production FFmpeg bundle")
    sys.path.insert(0, str(ROOT / "tools"))
    from tile_worker import load_upscale
    reference = load_upscale(ROOT / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py")
    require(digest(Path(reference.__file__)) == UPSCALER_SHA, "Upscaler source mismatch")
    # Existing production capability probe encodes only ONE synthetic frame.
    common = ROOT / "package/src/flashvsr/architect/common"
    sys.path.insert(0, str(common))
    import media_utils
    encoder, attempts = media_utils.select_encoder(15360, 8640, threads=32,
        scratch_dir=ROOT / "verification/encoder_probe", preferred="libx265")
    write(ROOT / "verification/environment_preflight.json", {
        "status": "PASS", "time_utc": now(), "python": sys.version,
        "torch_cuda": torch.version.cuda, "versions": versions, "all_package_versions": all_versions,
        "environment_file_identities": environment_identities(), "gpus": gpu_rows,
        "bsa_import_path": block_sparse_attn.__file__, "ffmpeg": ffmpeg,
        "bsa_extension_sha256": native_build["extension"]["sha256"],
        "bsa_build_proof_sha256": digest(ROOT / "verification/bsa_ampere_build.json"),
        "encoder": encoder, "encoder_attempts": attempts,
        "transfer_receipt_sha256": digest(ROOT / "verification/source_transfer_verification.json"),
    })
    print("ENVIRONMENT_PASS", flush=True)


def installed_versions():
    return dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))


def environment_identities():
    return {str(p.relative_to(ROOT / "env")): identity(p) for p in (ROOT / "env").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def check_gate(smoke=False):
    check_assets()
    env = read(ROOT / "verification/environment_preflight.json")
    require(env.get("status") == "PASS", "Environment test has not passed")
    require(env.get("transfer_receipt_sha256") ==
            digest(ROOT / "verification/source_transfer_verification.json"), "Environment proof is stale")
    require(installed_versions() == env.get("all_package_versions"), "Installed packages changed after preflight")
    require(environment_identities() == env.get("environment_file_identities"),
            "Environment files changed after preflight")
    if smoke:
        value = read(ROOT / "verification/compatibility_test.json")
        require(value.get("status") == "PASS", "All four GPU smoke tests must pass first")
        require(value.get("environment_receipt_sha256") ==
                digest(ROOT / "verification/environment_preflight.json"), "Smoke proof is stale")
        workers = value.get("workers", [])
        require(len(workers) == 4 and {item.get("gpu") for item in workers} == set(ASSIGNMENTS),
                "Compatibility proof must cover exactly GPUs 0,1,2,3")
        require(len({item.get("output_directory") for item in workers}) == 4,
                "Compatibility workers must have four isolated outputs")
        for item in workers:
            gpu = item["gpu"]
            video, start, _ = ASSIGNMENTS[gpu]
            work = ROOT / "smoke" / f"gpu{gpu}"
            require(item.get("video") == video and item.get("global_tile") == start and
                    item.get("frames") == 29 and Path(item["receipt"]) == work / "shard_receipt.json",
                    "Compatibility record assignment/path differs from approved layout")
            inside(Path(item["output_directory"]), work)
            require(digest(Path(item["receipt"])) == item["sha256"], "Smoke receipt changed")


def paths(video):
    require(video in ("03", "04"), "Invalid video ID")
    base = ROOT / "outputs" / video
    return {
        "input": ROOT / f"package/data/02_Outpaint_4K_3840x2160/{video}/final/{video}_Outpaint_4K.mp4",
        "anchor": ROOT / f"package/data/00_Input_8K_7680x4320/{video}_Input_8K.mp4",
        "raw": ROOT / "runs" / video / f"raw_flashvsr/{video}_FlashVSR_raw_16K_16fps.mp4",
        "final": base / f"final/{video}_Certification_Final_16K_16fps.mp4",
        "manifest": base / "metadata/final_manifest.json",
        "work": ROOT / "runs" / video,
    }


def worker_command(gpu, mode):
    video, start, end = ASSIGNMENTS[gpu]
    p = paths(video)
    work = (ROOT / "smoke" / f"gpu{gpu}") if mode == "smoke" else p["work"] / f"gpu{gpu}"
    return [str(x) for x in [PYTHON, ROOT / "tools/tile_worker.py", "--video-id", video,
        "--input", p["input"], "--anchor", p["anchor"], "--gpu-index", gpu,
        "--tile-start", start, "--tile-end", end, "--work-dir", work,
        "--output-target", p["raw"], "--mode", mode]], work


def check_smoke_receipt(gpu, work):
    from PIL import Image
    sys.path.insert(0, str(ROOT / "tools"))
    from tile_worker import load_upscale
    video, start, _ = ASSIGNMENTS[gpu]
    receipt_path = work / "shard_receipt.json"
    receipt = read(receipt_path)
    require(receipt.get("state") == "SHARD_COMPLETE" and receipt.get("mode") == "smoke",
            f"GPU{gpu} did not finish smoke inference")
    require(receipt["subprocess"]["exit_code"] == 0 and receipt["processed_input_frames"] == 29,
            "Smoke child or frame count failed")
    require(receipt["video_id"] == video and receipt["gpu_index"] == gpu and
            receipt["global_tile_indices"] == [start], "Smoke global assignment changed")
    manifest = read(receipt["private_partial_manifest"])
    artifact = manifest["tile_artifacts"][0]
    ref = load_upscale(ROOT / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py")
    from dataclasses import asdict
    require(artifact["tile"] == asdict(ref.make_tiles(3840, 2160, 256, 32)[start]), "Smoke geometry changed")
    require(artifact["tile_seed"] == 71634910 + start, "Smoke global seed changed")
    directory = inside(Path(artifact["directory"]), work)
    require({p.name for p in directory.iterdir()} == {f"{i:09d}.png" for i in range(29)},
            "Smoke frames missing or unexpectedly reordered/named")
    cursor = 0
    for filename in artifact["temporal_unit_metadata"]:
        emission = read(inside(Path(filename), work))
        require(emission["tile_index"] == start and emission["frame_start"] == cursor,
                "Smoke causal frame sequence has a gap/overlap")
        cursor = emission["frame_end_exclusive"]
    require(cursor == 29, "Smoke temporal sequence incomplete")
    for frame in (0, 28):
        with Image.open(directory / f"{frame:09d}.png") as im:
            im.load()
            require(im.size == (1024, 1024), "Smoke output tile shape changed")
    return {"gpu": gpu, "video": video, "global_tile": start,
            "receipt": str(receipt_path), "sha256": digest(receipt_path),
            "output_directory": str(directory), "frames": 29,
            "gpu_metrics": manifest["partial_report"].get("gpu", {})}


def smoke_test():
    check_gate()
    require(not (ROOT / "smoke").exists(), "Smoke workspace already exists; inspect rather than overwrite")
    launched = []
    for gpu in ASSIGNMENTS:
        command, work = worker_command(gpu, "smoke")
        work.mkdir(parents=True)
        log = (work / "supervisor.log").open("x")
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=runtime_env())
        log.close()
        launched.append((gpu, work, child))
    write(ROOT / "verification/smoke_processes.json", {
        "started_utc": now(), "workers": [{"gpu": g, "pid": c.pid, "work": str(w)} for g, w, c in launched]})
    exits = {gpu: child.wait() for gpu, _, child in launched}
    require(all(code == 0 for code in exits.values()), f"Smoke failed; full launch forbidden: {exits}")
    workers = [check_smoke_receipt(gpu, work) for gpu, work, _ in launched]
    require(len({w["output_directory"] for w in workers}) == 4, "Smoke workers shared output directories")
    write(ROOT / "verification/compatibility_test.json", {
        "status": "PASS", "time_utc": now(), "workers": workers,
        "environment_receipt_sha256": digest(ROOT / "verification/environment_preflight.json"),
        "scope": "29-frame warm-up plus steady-state original causal inference per representative tile",
        "full_1952_frame_validation_run": False,
    })
    print("FOUR_GPU_COMPATIBILITY_PASS", flush=True)


def supervise(gpu):
    check_gate(smoke=True)
    command, work = worker_command(gpu, "full")
    work.mkdir(parents=True, exist_ok=True)
    status = ROOT / "status" / f"gpu{gpu}.json"
    require(not status.exists(), "Worker supervisor status already exists")
    record = {"gpu": gpu, "assignment": ASSIGNMENTS[gpu], "started_utc": now(),
              "supervisor_pid": os.getpid(), "work": str(work), "command": command}
    with (work / "supervisor.log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=runtime_env())
        write(status, {**record, "status": "RUNNING", "worker_pid": child.pid})
        code = child.wait()
    passed = code == 0 and (work / "shard_receipt.json").is_file()
    write(status, {**record, "status": "COMPLETE" if passed else "FAILED",
                   "worker_pid": child.pid, "exit_code": code, "finished_utc": now()})
    require(passed, f"GPU{gpu} failed, exit={code}; preserved artifacts for investigation")


def postprocess_command(video):
    p = paths(video)
    return [str(x) for x in [PYTHON, ROOT / "tools/postprocess_video.py", "--video-id", video,
        "--input-4k", p["input"], "--original-8k", p["anchor"],
        "--model-root", ROOT / "package/src/models/flashvsr_v1_1",
        "--flashvsr-reference", ROOT / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py",
        "--outpainting-config", ROOT / "package/src/video_outpainting/main/configs/inference_4k.yaml",
        "--output", p["final"], "--raw-output", p["raw"], "--final-manifest", p["manifest"],
        "--work-dir", p["work"], "--expected-frames", 1952, "--tile-size", 256,
        "--tile-overlap", 32, "--min-tile-size", 256, "--sparse-ratio", 2.0,
        "--local-range", 11, "--seed", 71634910, "--ffmpeg-threads", 32,
        "--encoder", "libx265", "--aggregate-manifest", p["work"] / "aggregate_manifest.json"]]


def coordinate(video):
    p = paths(video)
    status_path = ROOT / "status" / f"video{video}.json"
    write(status_path, {"status": "WAITING_FOR_TWO_SHARDS", "pid": os.getpid(), "time_utc": now()})
    gpus = [g for g, a in ASSIGNMENTS.items() if a[0] == video]
    while True:
        states = [read(ROOT / "status" / f"gpu{g}.json")
                  if (ROOT / "status" / f"gpu{g}.json").exists() else {} for g in gpus]
        require(not any(s.get("status") == "FAILED" for s in states), f"Video{video} shard failed")
        if all(s.get("status") == "COMPLETE" for s in states):
            break
        time.sleep(20)
    check_gate(smoke=True)
    merge = [PYTHON, ROOT / "tools/merge_shards.py", "--video-id", video,
             "--work-dir", p["work"], "--output-manifest", p["work"] / "aggregate_manifest.json",
             "--output-target", p["raw"]]
    for gpu in gpus:
        merge.extend(["--shard", p["work"] / f"gpu{gpu}/shard_receipt.json"])
    run(merge)
    write(status_path, {"status": "WAITING_FOR_CPU_POSTPROCESS_LOCK", "time_utc": now()})
    with (ROOT / "status/postprocess.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        check_gate(smoke=True)
        write(status_path, {"status": "ENCODING_AND_ORIGINAL_CENTER_COMPOSITE", "time_utc": now()})
        run(postprocess_command(video))
    result = read(p["manifest"])
    require(result.get("status") == "PASS" and result.get("video_id") == video, "Final manifest failed")
    write(status_path, {"status": "PASS", "time_utc": now(), "manifest": str(p["manifest"]),
                        "output": str(p["final"]), "full_decode_repeated": False})


def launch_full():
    check_gate(smoke=True)
    require(shutil.disk_usage(ROOT).free >= 1024**4, "Less than 1 TiB free before dual-video launch")
    existing = subprocess.run(["tmux", "has-session", "-t", SESSION], capture_output=True)
    require(existing.returncode != 0, "Dedicated production tmux session already exists")
    for video in ("04", "03"):
        p = paths(video)
        require(not any(p[key].exists() for key in ("work", "raw", "final", "manifest")),
                f"Video{video} production workspace not fresh")
    (ROOT / "status").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    reservation = ROOT / "status/launch_reservation.json"
    token = secrets.token_hex(16)
    with reservation.open("x") as stream:
        json.dump({"time_utc": now(), "pid": os.getpid(), "token": token}, stream)
    windows = [(f"gpu{g}_v{a[0]}", "supervise", "--gpu", g) for g, a in ASSIGNMENTS.items()]
    windows += [(f"post_v{v}", "coordinate", "--video", v) for v in ("04", "03")]
    launched = []
    for index, (window, action, flag, value) in enumerate(windows):
        command = [str(PYTHON), "-I", str(Path(__file__).resolve()), action, flag, str(value),
                   "--launch-token", token]
        log = ROOT / "logs" / f"{window}.log"
        shell = shlex.join(command) + " >> " + shlex.quote(str(log)) + " 2>&1"
        tmux = ["tmux", "new-session", "-d", "-s", SESSION] if index == 0 else [
            "tmux", "new-window", "-t", SESSION]
        run(tmux + ["-n", window, shell])
        run(["tmux", "set-option", "-w", "-t", f"{SESSION}:{window}", "remain-on-exit", "on"])
        launched.append({"window": f"{SESSION}:{window}", "command": command, "log": str(log)})
    # Every window must reach this barrier before ANY full inference can start.
    deadline = time.monotonic() + 120
    while True:
        ready = [ROOT / "status" / f"ready_{action}_{value}.json" for _, action, _, value in windows]
        if all(path.is_file() and read(path).get("token") == token for path in ready):
            break
        require(time.monotonic() < deadline, "Not all six production windows became ready; barrier remains closed")
        time.sleep(0.5)
    check_gate(smoke=True)
    write(ROOT / "status/start_barrier.json", {"token": token, "released_utc": now()})
    deadline = time.monotonic() + 120
    while True:
        states = [read(ROOT / "status" / f"gpu{g}.json")
                  if (ROOT / "status" / f"gpu{g}.json").exists() else {} for g in ASSIGNMENTS]
        require(not any(s.get("status") == "FAILED" for s in states), "A full worker failed during startup")
        if all(s.get("status") in ("RUNNING", "COMPLETE") for s in states):
            for state in states:
                if state["status"] == "RUNNING":
                    os.kill(state["worker_pid"], 0)  # existence check only; no signal delivered
            break
        require(time.monotonic() < deadline, "Released workers did not report startup; inspect preserved logs")
        time.sleep(0.5)
    write(ROOT / "status/launch.json", {"status": "LAUNCHED", "time_utc": now(),
        "gpu_uuids": GPU_UUIDS, "assignments": ASSIGNMENTS, "windows": launched,
        "compatibility_receipt_sha256": digest(ROOT / "verification/compatibility_test.json")})
    print("FOUR_GPU_FULL_RUN_LAUNCHED", flush=True)


def await_start_barrier(action, value, token):
    require(token and read(ROOT / "status/launch_reservation.json").get("token") == token,
            "Missing matching launch reservation")
    check_gate(smoke=True)
    write(ROOT / "status" / f"ready_{action}_{value}.json",
          {"token": token, "pid": os.getpid(), "time_utc": now()})
    deadline = time.monotonic() + 600
    barrier = ROOT / "status/start_barrier.json"
    while not barrier.exists():
        require(time.monotonic() < deadline, "Launch barrier was never released; no inference started")
        time.sleep(0.5)
    require(read(barrier).get("token") == token, "Wrong start barrier token")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify-assets", "environment", "smoke", "launch-full", "supervise", "coordinate"])
    parser.add_argument("--gpu", type=int, choices=range(4))
    parser.add_argument("--video", choices=["03", "04"])
    parser.add_argument("--expected-code-manifest-sha256")
    parser.add_argument("--expected-runtime-manifest-sha256")
    parser.add_argument("--launch-token")
    args = parser.parse_args()
    # Also set the current process environment before any CUDA import; subprocess
    # mapping alone is insufficient for a trustworthy logical/physical GPU proof.
    clean_env = runtime_env()
    os.environ.clear()
    os.environ.update(clean_env)
    try:
        if args.action == "verify-assets":
            verify_assets(args.expected_code_manifest_sha256, args.expected_runtime_manifest_sha256)
        elif args.action == "supervise":
            require(args.gpu is not None, "GPU required")
            await_start_barrier(args.action, args.gpu, args.launch_token)
            supervise(args.gpu)
        elif args.action == "coordinate":
            require(args.video is not None, "Video ID required")
            await_start_barrier(args.action, args.video, args.launch_token)
            coordinate(args.video)
        else:
            {"environment": environment_preflight, "smoke": smoke_test,
             "launch-full": launch_full}[args.action]()
    except BaseException as error:
        failure = {"status": "FAILED", "action": args.action, "gpu": args.gpu,
                   "video": args.video, "time_utc": now(), "error": repr(error)}
        name = f"{args.action}_{args.gpu if args.gpu is not None else args.video}"
        write(ROOT / "status" / f"failure_{name}.json", failure)
        if args.action == "coordinate":
            write(ROOT / "status" / f"video{args.video}.json", failure)
        if args.action == "supervise":
            write(ROOT / "status" / f"gpu{args.gpu}.json", failure)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
