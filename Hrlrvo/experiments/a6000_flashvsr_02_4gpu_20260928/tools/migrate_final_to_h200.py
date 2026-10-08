#!/usr/bin/env python3
"""Copy-only, idempotent A6000 video02 final migration into the H200 layout."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import time


EXPERIMENT = Path(__file__).resolve().parents[1]
PIPELINE = Path("/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K")
TARGET = PIPELINE / "data/03_Upscale_16K_15360x8640/02"
REMOTE_ROOT = PurePosixPath("/home/oem/hrlrvo/experiments/a6000_flashvsr_02_4gpu_20260928")
REMOTE_FINAL = REMOTE_ROOT / "outputs/02/final/02_Certification_Final_16K_16fps.mp4"
EXPECTED_FINAL_NAME = "02_Certification_Final_16K_16fps.mp4"
WATCH_STATUS = EXPERIMENT / "verification/h200_video02_migration_status.json"
SSH_BASE = [
    "ssh", "-T", "-p", "2222",
    "-i", "/home/work/ai-graphics_33/.ssh/codex_school_a6000_20260923_ed25519",
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile=/home/work/ai-graphics_33/.ssh/known_hosts_codex_school_a6000",
    "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=10",
    "oem@203.253.25.176",
]
RSYNC_SHELL = " ".join(shlex.quote(item) for item in SSH_BASE[:-1]).replace("ssh -T", "ssh")
REMOTE = SSH_BASE[-1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def watch_status(status: str, **fields) -> None:
    atomic_json(WATCH_STATUS, {"schema": "h200_video02_migration_watcher/v1",
                              "status": status, "pid": os.getpid(),
                              "updated_utc": now(), **fields})


def ssh(command: str, *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([*SSH_BASE, command], check=False, text=True,
                          capture_output=capture)


def wait_for_remote_pass() -> tuple[dict, dict]:
    deadline = time.monotonic() + 72 * 3600
    status_path = REMOTE_ROOT / "status/video02.json"
    manifest_path = REMOTE_ROOT / "outputs/02/metadata/final_manifest.json"
    while True:
        result = ssh(f"test -f {shlex.quote(str(status_path))} && cat {shlex.quote(str(status_path))}")
        if result.returncode == 0:
            status = json.loads(result.stdout)
            if status.get("status") == "FAILED":
                raise RuntimeError(f"remote video02 failed: {status.get('error')}")
            if status.get("status") == "PASS":
                manifest_result = ssh(f"cat {shlex.quote(str(manifest_path))}")
                if manifest_result.returncode:
                    raise RuntimeError("remote status PASS without final manifest")
                manifest = json.loads(manifest_result.stdout)
                if manifest.get("status") != "PASS" or manifest.get("video_id") != "02":
                    raise RuntimeError("remote final manifest is not PASS for video02")
                output = manifest.get("output", {})
                if ((output.get("width"), output.get("height"), output.get("frame_count"))
                        != (15360, 8640, 1952) or float(output.get("fps", -1)) != 16.0):
                    raise RuntimeError("remote final manifest output spec mismatch")
                if PurePosixPath(manifest["output_file"]["path"]) != REMOTE_FINAL:
                    raise RuntimeError("remote manifest final path differs from approved video02 target")
                return status, manifest
        if time.monotonic() >= deadline:
            raise RuntimeError("remote video02 did not reach PASS within 72 hours")
        time.sleep(60)


def remote_inventory() -> dict:
    code = r'''
from pathlib import Path
import hashlib, json
root=Path("/home/oem/hrlrvo/experiments/a6000_flashvsr_02_4gpu_20260928")
fixed=[
 "outputs/02/metadata/final_manifest.json", "runs/02/aggregate_manifest.json",
 "deployment/deployment_manifest.json", "verification/final_only_equivalence.json",
 "verification/initial_health.json", "status/bootstrap.json", "status/launch.json",
 "status/video02.json",
 "verification/4shard_final_recovery_20260928/patch_record.json",
 "verification/4shard_final_recovery_20260928/preflight.json",
 "verification/4shard_final_recovery_20260928/original_video02_failed_status.json",
 "tools/resume_final_only_4shard.py",
 "package/src/cert_pipeline/flash_composite.py",
 "package_override/cert_pipeline/flash_composite.py",
 "final_only/parallel_renderer.py",
]
patterns=["status/gpu*.json", "logs/*.log", "runs/02/gpu*/shard_request.json",
 "runs/02/gpu*/shard_receipt.json", "runs/02/gpu*/private_partial_stage_manifest.json",
 "runs/02/gpu*/supervisor.log", "runs/02/gpu*/worker.log"]
paths=[]
for rel in fixed:
 p=root/rel
 if p.is_file() and not p.is_symlink(): paths.append(p)
for pattern in patterns:
 paths.extend(p for p in root.glob(pattern) if p.is_file() and not p.is_symlink())
rows=[]
for p in sorted(set(paths)):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024), b""): h.update(b)
 rows.append({"path":str(p.relative_to(root)),"size":p.stat().st_size,"sha256":h.hexdigest()})
print(json.dumps({"schema":"a6000_video02_small_artifacts/v1","files":rows}))
'''
    result = ssh("python3 -c " + shlex.quote(code))
    if result.returncode:
        raise RuntimeError(f"remote artifact inventory failed: {result.stderr}")
    value = json.loads(result.stdout)
    required = {
        "outputs/02/metadata/final_manifest.json", "runs/02/aggregate_manifest.json",
        "deployment/deployment_manifest.json", "verification/final_only_equivalence.json",
        "verification/initial_health.json", "status/launch.json", "status/video02.json",
        "verification/4shard_final_recovery_20260928/patch_record.json",
        "verification/4shard_final_recovery_20260928/preflight.json",
        "verification/4shard_final_recovery_20260928/original_video02_failed_status.json",
        "tools/resume_final_only_4shard.py",
        "package/src/cert_pipeline/flash_composite.py",
        "package_override/cert_pipeline/flash_composite.py",
        "final_only/parallel_renderer.py",
        *{f"status/gpu{gpu}.json" for gpu in range(4)},
        *{f"runs/02/gpu{gpu}/shard_request.json" for gpu in range(4)},
        *{f"runs/02/gpu{gpu}/shard_receipt.json" for gpu in range(4)},
    }
    actual = {item["path"] for item in value["files"]}
    missing = sorted(required - actual)
    if missing:
        raise RuntimeError(f"remote PASS lacks required small artifacts: {missing}")
    forbidden = [path for path in actual if "/scratch/" in path or path.endswith(".png")
                 or "temporal_metadata" in path or "raw_flashvsr" in path]
    if forbidden:
        raise RuntimeError(f"inventory unexpectedly includes excluded intermediates: {forbidden}")
    return value


def remote_final_identity() -> dict:
    command = f"stat -c '%s' {shlex.quote(str(REMOTE_FINAL))} && sha256sum {shlex.quote(str(REMOTE_FINAL))}"
    result = ssh(command)
    if result.returncode:
        raise RuntimeError(f"remote final identity failed: {result.stderr}")
    lines = result.stdout.splitlines()
    return {"path": str(REMOTE_FINAL), "size": int(lines[0]), "sha256": lines[1].split()[0]}


def fetch_small_artifacts(inventory: dict, staging: Path) -> None:
    relative = [item["path"] for item in inventory["files"]]
    tar_path = staging / "actual_small_artifacts.tar"
    command = [*SSH_BASE, "tar -C " + shlex.quote(str(REMOTE_ROOT)) + " -cf - -- "
               + " ".join(shlex.quote(path) for path in relative)]
    with tar_path.open("wb") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise RuntimeError(f"remote small-artifact tar failed: {completed.stderr.decode(errors='replace')}")
    extracted = staging / "extracted"
    extracted.mkdir(exist_ok=True)
    with tarfile.open(tar_path, "r") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        if names != set(relative):
            raise RuntimeError("small-artifact tar member set differs from inventory")
        for member in archive.getmembers():
            if member.name.startswith("/") or ".." in PurePosixPath(member.name).parts:
                raise RuntimeError(f"unsafe tar member: {member.name}")
        archive.extractall(extracted, filter="data")
    identities = {item["path"]: item for item in inventory["files"]}
    for relative_path, identity in identities.items():
        path = extracted / relative_path
        if path.stat().st_size != identity["size"] or digest(path) != identity["sha256"]:
            raise RuntimeError(f"small artifact transfer mismatch: {relative_path}")


def published_artifact_path(relative_path: str) -> Path:
    is_log = relative_path.startswith("logs/") or relative_path.endswith(".log")
    root = TARGET / ("logs/a6000_remote_actual" if is_log else
                     "metadata/a6000_remote_actual")
    return root / relative_path


def verify_existing_artifact_tree(root: Path, inventory: dict, *, logs: bool) -> None:
    """Accept only a complete byte-identical tree from an interrupted publish."""
    for item in inventory["files"]:
        is_log = item["path"].startswith("logs/") or item["path"].endswith(".log")
        if is_log != logs:
            continue
        destination = root / item["path"]
        if (not destination.is_file() or destination.stat().st_size != item["size"]
                or digest(destination) != item["sha256"]):
            raise RuntimeError(
                f"existing interrupted publication is incomplete or differs: {destination}"
            )


def publish(staging: Path, final_identity: dict, inventory: dict,
            remote_status: dict) -> dict:
    final_destination = TARGET / "final" / EXPECTED_FINAL_NAME
    metadata_destination = TARGET / "metadata/a6000_remote_actual"
    logs_destination = TARGET / "logs/a6000_remote_actual"
    receipt_path = TARGET / "metadata/a6000_transfer_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") == "PASS" and final_destination.is_file()
                and final_destination.stat().st_size == final_identity["size"]
                and digest(final_destination) == final_identity["sha256"]):
            return receipt
        raise RuntimeError("existing migration receipt is not a valid idempotent PASS")

    staged_final = staging / EXPECTED_FINAL_NAME
    if staged_final.stat().st_size != final_identity["size"] or digest(staged_final) != final_identity["sha256"]:
        raise RuntimeError("staged final identity mismatch")
    staged_metadata = staging / "publish_metadata"
    staged_logs = staging / "publish_logs"
    for directory in (staged_metadata, staged_logs):
        directory.mkdir(exist_ok=True)
    extracted = staging / "extracted"
    for item in inventory["files"]:
        source = extracted / item["path"]
        is_log = item["path"].startswith("logs/") or item["path"].endswith(".log")
        base = staged_logs if is_log else staged_metadata
        destination = base / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    final_destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    logs_destination.parent.mkdir(parents=True, exist_ok=True)
    # Metadata/log trees are committed before the final name becomes visible.
    # The receipt is the final transaction marker. A crash after any rename is
    # safely recoverable by validating each already-published object byte-for-byte.
    if metadata_destination.exists() or metadata_destination.is_symlink():
        verify_existing_artifact_tree(metadata_destination, inventory, logs=False)
    else:
        os.replace(staged_metadata, metadata_destination)
    if logs_destination.exists() or logs_destination.is_symlink():
        verify_existing_artifact_tree(logs_destination, inventory, logs=True)
    else:
        os.replace(staged_logs, logs_destination)
    if final_destination.exists() or final_destination.is_symlink():
        if (not final_destination.is_file()
                or final_destination.stat().st_size != final_identity["size"]
                or digest(final_destination) != final_identity["sha256"]):
            raise RuntimeError("existing published final differs from remote video02 final")
    else:
        os.replace(staged_final, final_destination)
    receipt = {
        "schema": "h200_video02_a6000_migration/v1", "status": "PASS",
        "published_utc": now(), "video_id": "02", "source_root": str(REMOTE_ROOT),
        "destination_root": str(TARGET), "final": {"source": final_identity,
            "destination": {"path": str(final_destination), "size": final_destination.stat().st_size,
                            "sha256": digest(final_destination)}},
        "small_artifacts": inventory, "remote_status": remote_status,
        "remote_manifest_sha256": next(
            item["sha256"] for item in inventory["files"]
            if item["path"] == "outputs/02/metadata/final_manifest.json"
        ),
        "excluded": ["tile PNGs", "temporal intermediate tree", "raw FlashVSR 16K MP4"],
        "raw_flashvsr_created_or_copied": False,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    lock_path = EXPERIMENT / "verification/h200_video02_migration.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another video02 migration watcher owns the lock") from error
        receipt_path = TARGET / "metadata/a6000_transfer_receipt.json"
        final_path = TARGET / "final" / EXPECTED_FINAL_NAME
        if receipt_path.is_file() and final_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("status") == "PASS"
                    and digest(final_path) == receipt["final"]["destination"]["sha256"]):
                print(json.dumps({"status": "ALREADY_PASS", "receipt": str(receipt_path)}))
                return 0
        watch_status("WAITING_FOR_REMOTE_FINAL_PASS")
        remote_status, remote_manifest = wait_for_remote_pass()
        watch_status("INVENTORY_AND_IDENTITY_VERIFICATION")
        inventory = remote_inventory()
        final_identity = remote_final_identity()
        if final_identity["size"] <= 0:
            raise RuntimeError("remote final is empty")
        staging = TARGET / ".incoming_a6000_video02"
        staging.mkdir(parents=True, exist_ok=True)
        atomic_json(staging / "request.json", {"video_id": "02", "remote_final": final_identity,
                    "inventory": inventory, "created_utc": now()})
        staged_final = staging / EXPECTED_FINAL_NAME
        watch_status("TRANSFERRING_FINAL_MP4", remote_final=final_identity,
                     staging_path=str(staged_final))
        command = ["rsync", "-a", "--partial", "--append-verify", "-e", RSYNC_SHELL,
                   f"{REMOTE}:{REMOTE_FINAL}", str(staged_final)]
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise RuntimeError(f"final MP4 rsync failed with exit {completed.returncode}")
        watch_status("TRANSFERRING_SMALL_ACTUAL_ARTIFACTS")
        fetch_small_artifacts(inventory, staging)
        watch_status("PUBLISHING_ATOMICALLY")
        receipt = publish(staging, final_identity, inventory, remote_status)
        watch_status("PASS", receipt=str(receipt_path), final=receipt["final"])
        print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        watch_status("FAILED", error=f"{type(error).__name__}: {error}")
        raise
