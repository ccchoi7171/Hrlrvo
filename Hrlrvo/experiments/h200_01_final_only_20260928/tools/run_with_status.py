#!/usr/bin/env python3
"""Atomic status for the isolated, user-authorized video01 final-only handoff."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "status/final_only.json"
FINAL = Path(
    "/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/data/"
    "03_Upscale_16K_15360x8640/01/final/01_Certification_Final_16K_16fps.mp4"
)
MANIFEST = FINAL.parent.parent / "metadata/final_manifest.json"


def emit(status: str, **fields: object) -> None:
    value = {
        "schema": "h200_01_final_only_status/v1",
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **fields,
    }
    temporary = STATUS.with_name(f".{STATUS.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, STATUS)
    print(json.dumps(value, sort_keys=True), flush=True)


def main() -> int:
    command = sys.argv[1:]
    if not command:
        raise ValueError("final-only command is missing")
    if STATUS.exists():
        previous = json.loads(STATUS.read_text(encoding="utf-8"))
        if previous.get("status") in {"RUNNING", "PASS"}:
            raise RuntimeError(f"final-only already {previous['status']}; refusing duplicate run")
    emit("RUNNING", command=command)
    result = subprocess.run(command, check=False)
    if result.returncode:
        emit("FAILED", exit_code=result.returncode)
        return result.returncode
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or manifest.get("video_id") != "01":
            raise RuntimeError("final manifest is not PASS for video01")
        if not FINAL.is_file() or FINAL.stat().st_size <= 0:
            raise RuntimeError("final output is missing or empty")
        if manifest.get("output_file", {}).get("size") != FINAL.stat().st_size:
            raise RuntimeError("manifest final size mismatch")
    except Exception as error:
        emit("FAILED", reason=f"{type(error).__name__}: {error}")
        return 1
    emit("PASS", final=str(FINAL), size_bytes=FINAL.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
