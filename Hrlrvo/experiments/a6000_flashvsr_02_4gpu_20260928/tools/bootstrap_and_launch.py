#!/usr/bin/env python3
"""Autonomously gate, validate, and launch the isolated video02 production run."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = Path("/home/oem/hrlrvo/experiments/a6000_flashvsr_0304_20260926")
STATUS = ROOT / "status/bootstrap.json"
EXPECTED = {
    "input": {"path": ROOT / "inputs/02_Outpaint_4K.mp4", "size": 1962411248,
              "sha256": "3978d52e91d99d08296d1bb8e8768dfa8c76005b61aff979bfe6f3df4e0cad7f"},
    "anchor": {"path": ROOT / "inputs/02_Input_8K.mp4", "size": 694119810,
               "sha256": "c192abcb6426be964a9a258c0c06fb22a81317eb8771d46fd81085a2172981b1"},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("r+") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def update(status: str, **fields) -> None:
    value = {"schema": "a6000_video02_bootstrap/v1", "status": status,
             "pid": os.getpid(), "updated_utc": now(), **fields}
    write(STATUS, value)


def run(command: list[str], *, env=None, log=None) -> None:
    completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT if log else None,
                               check=False, text=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {command}")


def ensure_package_overlay() -> None:
    package_src = ROOT / "package/src"
    cert = package_src / "cert_pipeline"
    cert.mkdir(parents=True, exist_ok=True)
    target = cert / "flash_composite.py"
    override = ROOT / "package_override/cert_pipeline/flash_composite.py"
    if target.exists():
        if digest(target) != digest(override):
            raise RuntimeError("existing isolated compositor differs")
    else:
        shutil.copy2(override, target)
    for name in ("flashvsr", "models", "video_outpainting"):
        link = package_src / name
        expected = ORIGIN / "package/src" / name
        if link.is_symlink():
            if link.resolve() != expected.resolve():
                raise RuntimeError(f"wrong existing package link: {link}")
        elif link.exists():
            raise RuntimeError(f"refusing non-symlink package collision: {link}")
        else:
            link.symlink_to(expected)
    for name in ("env", "runtime"):
        link = ROOT / name
        expected = ORIGIN / name
        if link.is_symlink():
            if link.resolve() != expected.resolve():
                raise RuntimeError(f"wrong existing runtime link: {link}")
        elif link.exists():
            raise RuntimeError(f"refusing runtime collision: {link}")
        else:
            link.symlink_to(expected)


def capture_initial_health() -> dict:
    request_paths = [ROOT / f"runs/02/gpu{gpu}/shard_request.json" for gpu in range(4)]
    deadline = time.monotonic() + 300
    while not all(path.is_file() for path in request_paths):
        if time.monotonic() >= deadline:
            raise RuntimeError("four shard requests were not published within five minutes")
        time.sleep(2)
    requests = [json.loads(path.read_text()) for path in request_paths]
    assignments = {0: (0,42), 1: (43,85), 2: (86,127), 3: (128,169)}
    for gpu, request in enumerate(requests):
        expected = assignments[gpu]
        params = request.get("parameters", {})
        if (request.get("video_id") != "02" or request.get("gpu_index") != gpu
                or request.get("assignment") != {"tile_start": expected[0], "tile_end": expected[1]}
                or params.get("tile_size") != 256 or params.get("tile_overlap") != 32
                or params.get("base_seed") != 25621683 or params.get("sparse_ratio") != 2.0
                or params.get("local_range") != 11
                or params.get("precision") != "torch.bfloat16 (original implementation)"):
            raise RuntimeError(f"GPU{gpu} published an invalid production request")
    states = [json.loads((ROOT / f"status/gpu{gpu}.json").read_text()) for gpu in range(4)]
    for state in states:
        if state.get("status") not in {"RUNNING", "COMPLETE"}:
            raise RuntimeError(f"unexpected initial worker state: {state}")
        if state.get("status") == "RUNNING":
            os.kill(int(state["worker_pid"]), 0)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True,
    )
    errors = {}
    for gpu in range(4):
        texts = []
        for path in (ROOT / f"logs/gpu{gpu}.log", ROOT / f"runs/02/gpu{gpu}/supervisor.log",
                     ROOT / f"runs/02/gpu{gpu}/worker.log"):
            if path.is_file():
                texts.append(path.read_text(encoding="utf-8", errors="replace")[-200000:])
        joined = "\n".join(texts)
        hits = [word for word in ("CUDA out of memory", "Traceback", "RuntimeError")
                if word.lower() in joined.lower()]
        if re.search(r"(?i)\b(?:nan|inf)\b", joined):
            hits.append("NaN/Inf")
        if hits:
            errors[str(gpu)] = hits
    if errors:
        raise RuntimeError(f"initial worker logs contain error signatures: {errors}")
    evidence = {
        "schema": "a6000_video02_initial_health/v1", "status": "PASS", "time_utc": now(),
        "requests": requests, "worker_states": states, "nvidia_smi": query.stdout.splitlines(),
        "error_signature_scan": "PASS", "tile_size": 256, "tile_overlap": 32,
        "base_seed": 25621683, "global_assignments": {str(k): list(v) for k,v in assignments.items()},
    }
    write(ROOT / "verification/initial_health.json", evidence)
    return evidence


def main() -> int:
    if STATUS.exists():
        raise RuntimeError(f"bootstrap status already exists: {STATUS}")
    update("WAITING_FOR_VERIFIED_INPUT_COPY", started_utc=now())
    deadline = time.monotonic() + 3 * 3600
    while True:
        if all(item["path"].is_file() and item["path"].stat().st_size == item["size"]
               for item in EXPECTED.values()):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("input transfer did not complete within three hours")
        time.sleep(15)
    update("VERIFYING_INPUT_SHA256")
    for name, item in EXPECTED.items():
        actual = digest(item["path"])
        if actual != item["sha256"]:
            raise RuntimeError(f"{name} SHA256 mismatch: {actual}")
    ensure_package_overlay()
    from orchestrate import runtime_env
    env = runtime_env()
    update("RUNNING_CPU_FINAL_ONLY_EQUIVALENCE")
    with (ROOT / "verification/equivalence.log").open("x") as log:
        run([str(ORIGIN / "env/bin/python"), "-I", str(ROOT / "tools/equivalence_check.py")],
            env=env, log=log)
    equivalence = json.loads((ROOT / "verification/final_only_equivalence.json").read_text())
    if equivalence.get("status") != "PASS":
        raise RuntimeError("CPU final-only equivalence failed")
    update("RUNNING_STATIC_CHECKS")
    code_files = [
        "tools/tile_worker.py", "tools/merge_shards.py", "tools/orchestrate.py",
        "tools/equivalence_check.py", "final_only/final_only.py",
        "final_only/parallel_renderer.py", "package/src/cert_pipeline/flash_composite.py",
    ]
    run([str(ORIGIN / "env/bin/python"), "-m", "py_compile", *[str(ROOT / p) for p in code_files]],
        env=env)
    free = shutil.disk_usage(ROOT).free
    previous_sizes = {
        "video03_scratch_bytes": sum(p.stat().st_size for p in (ORIGIN / "runs/03").rglob("*") if p.is_file()),
        "video04_scratch_bytes": sum(p.stat().st_size for p in (ORIGIN / "runs/04").rglob("*") if p.is_file()),
    }
    required = max(previous_sizes.values()) + 64 * 1024**3
    if free < required:
        raise RuntimeError(f"insufficient disk: free={free}, required={required}")
    manifest = {
        "schema": "a6000_video02_deployment/v1", "status": "PASS", "video_id": "02",
        "created_utc": now(), "base_seed": 25621683,
        "assignments": {"0": [0,42], "1": [43,85], "2": [86,127], "3": [128,169]},
        "inference_contract": {"global_tiles": 170, "tile_size": 256, "tile_overlap": 32,
            "frames_per_tile": 1952, "temporal_split": False, "precision": "torch.bfloat16",
            "sparse_ratio": 2.0, "local_range": 11,
            "tile_seed": "25621683 + original_global_tile_index"},
        "input": {"path": str(EXPECTED["input"]["path"]), "size": EXPECTED["input"]["size"],
                  "sha256": EXPECTED["input"]["sha256"]},
        "anchor": {"path": str(EXPECTED["anchor"]["path"]), "size": EXPECTED["anchor"]["size"],
                   "sha256": EXPECTED["anchor"]["sha256"]},
        "code_sha256": {name: digest(ROOT / name) for name in code_files},
        "origin_read_only": str(ORIGIN), "origin_upscale_sha256":
            digest(ORIGIN / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py"),
        "equivalence_evidence": {"path": str(ROOT / "verification/final_only_equivalence.json"),
                                 "sha256": digest(ROOT / "verification/final_only_equivalence.json")},
        "disk": {"free_bytes": free, "required_bytes": required, **previous_sizes},
    }
    write(ROOT / "deployment/deployment_manifest.json", manifest)
    update("LAUNCHING_PRODUCTION")
    run([str(ORIGIN / "env/bin/python"), "-I", str(ROOT / "tools/orchestrate.py"), "launch"], env=env)
    launch = json.loads((ROOT / "status/launch.json").read_text())
    update("CAPTURING_INITIAL_HEALTH", production_launch=launch)
    health = capture_initial_health()
    update("LAUNCHED", production_launch=launch, initial_health=health, finished_utc=now())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        update("FAILED", error=f"{type(error).__name__}: {error}", finished_utc=now())
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
