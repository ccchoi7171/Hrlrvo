#!/usr/bin/env python3
"""Wait for the copy-only 02 transfer, verify remote SHA256, publish an atomic gate."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = ROOT / "verification/input_transfer_pass.json"
REMOTE_ROOT = "/home/oem/hrlrvo/experiments/a6000_flashvsr_02_4gpu_20260928"
SSH = [
    "ssh", "-T", "-p", "2222",
    "-i", "/home/work/ai-graphics_33/.ssh/codex_school_a6000_20260923_ed25519",
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile=/home/work/ai-graphics_33/.ssh/known_hosts_codex_school_a6000",
    "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=10",
    "oem@203.253.25.176",
]
EXPECTED = {
    "02_Outpaint_4K.mp4": {
        "size": 1962411248,
        "sha256": "3978d52e91d99d08296d1bb8e8768dfa8c76005b61aff979bfe6f3df4e0cad7f",
    },
    "02_Input_8K.mp4": {
        "size": 694119810,
        "sha256": "c192abcb6426be964a9a258c0c06fb22a81317eb8771d46fd81085a2172981b1",
    },
}


def remote(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*SSH, command], check=False, text=True, capture_output=True)


def publish(value: dict) -> None:
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    temporary = MARKER.with_name(f".{MARKER.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, MARKER)


def main() -> int:
    if MARKER.exists():
        raise RuntimeError(f"marker already exists: {MARKER}")
    while True:
        command = "stat -c '%n %s' " + " ".join(
            f"{REMOTE_ROOT}/inputs/{name}" for name in EXPECTED
        )
        result = remote(command)
        if result.returncode == 0:
            sizes = {Path(line.rsplit(" ", 1)[0]).name: int(line.rsplit(" ", 1)[1])
                     for line in result.stdout.splitlines() if line.strip()}
            if all(sizes.get(name) == identity["size"] for name, identity in EXPECTED.items()):
                break
        time.sleep(15)
    command = "sha256sum " + " ".join(f"{REMOTE_ROOT}/inputs/{name}" for name in EXPECTED)
    result = remote(command)
    if result.returncode != 0:
        raise RuntimeError(f"remote sha256sum failed: {result.stderr}")
    hashes = {Path(line.split()[1]).name: line.split()[0] for line in result.stdout.splitlines()}
    failures = {name: {"expected": value["sha256"], "actual": hashes.get(name)}
                for name, value in EXPECTED.items() if hashes.get(name) != value["sha256"]}
    if failures:
        publish({"schema": "a6000_video02_input_transfer/v1", "status": "FAIL",
                 "verified_utc": datetime.now(timezone.utc).isoformat(), "failures": failures})
        return 1
    publish({"schema": "a6000_video02_input_transfer/v1", "status": "PASS", "video_id": "02",
             "outpaint_sha256": EXPECTED["02_Outpaint_4K.mp4"]["sha256"],
             "anchor_sha256": EXPECTED["02_Input_8K.mp4"]["sha256"],
             "verified_utc": datetime.now(timezone.utc).isoformat(),
             "remote_root": REMOTE_ROOT,
             "files": {name: {**identity, "remote_path": f"{REMOTE_ROOT}/inputs/{name}"}
                       for name, identity in EXPECTED.items()}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
