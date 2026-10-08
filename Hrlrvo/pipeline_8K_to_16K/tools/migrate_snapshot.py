#!/usr/bin/env python3
"""Copy-only migration utility. Never writes anywhere under the reference root."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "pipeline"
STATE = ROOT / "state/migration"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def publish(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def main():
    members = list((SOURCE / "src").rglob("*"))
    files = [p for p in members if p.is_file()]
    for name in (
        "run_cert.sh",
        "run_cert.py",
        "PIPELINE_EXECUTION_AUDIT.md",
        "DIRECTORY_LAYOUT.md",
    ):
        if (SOURCE / name).is_file():
            files.append(SOURCE / name)
    files += list((SOURCE / "configs").rglob("*.json"))
    for folder in ("00_Input_8K_7680x4320", "01_PreProc_2K_1920x1080"):
        files += [p for p in (SOURCE / "data" / folder).iterdir() if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    free = shutil.disk_usage(ROOT).free
    publish(
        STATE / "capacity.json",
        {
            "copy_bytes": total,
            "free_bytes": free,
            "required_reserve_bytes": 10 * 1024**3,
        },
    )
    if free < total + 10 * 1024**3:
        raise RuntimeError("INCOMPLETE reason=insufficient_disk_space")
    # Input/reference probes precede copies; ffprobe does not use count_frames here.
    media = []
    for p in files:
        if p.suffix != ".mp4" or "/data/" not in str(p):
            continue
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(p),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        probe = json.loads(result.stdout)
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        is_pre = "01_PreProc" in str(p)
        expected = (1920, 1080, "24/1") if is_pre else (7680, 4320, "16/1")
        assert (video["width"], video["height"], video["avg_frame_rate"]) == expected, p
        media.append({"source": str(p), "probe": probe, "metadata_status": "PASS"})
    publish(STATE / "media_headers.json", media)
    copied = []
    # Small runtime files first; no caller may change destination until COPY COMPLETE.
    files.sort(key=lambda p: p.stat().st_size)
    journal = STATE / "copy_integrity.jsonl"
    with journal.open("w") as log:
        for i, p in enumerate(files):
            rel = p.relative_to(SOURCE)
            if str(rel).startswith("configs/"):
                dest = ROOT / "docs/reference_configs" / rel.relative_to("configs")
            elif (
                p.name in ("run_cert.py", "run_cert.sh", "DIRECTORY_LAYOUT.md")
                and p.parent == SOURCE
            ):
                dest = ROOT / "docs/reference_entrypoints" / p.name
            elif p.name == "PIPELINE_EXECUTION_AUDIT.md" and p.parent == SOURCE:
                dest = ROOT / "docs" / p.name
            else:
                dest = ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            before = p.stat()
            source_hash = sha(p)
            if dest.exists():
                if sha(dest) != source_hash:
                    raise RuntimeError(f"refusing overwrite: {dest}")
            else:
                temp = dest.with_name("." + dest.name + ".copy_partial")
                shutil.copy2(p, temp, follow_symlinks=True)
                if sha(temp) != source_hash:
                    raise RuntimeError(f"copy SHA mismatch {p}")
                os.replace(temp, dest)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError(f"reference changed during copy: {p}")
            entry = {
                "source": str(p),
                "destination": str(dest.relative_to(ROOT)),
                "size_bytes": after.st_size,
                "sha256": source_hash,
                "destination_sha256": source_hash,
                "status": "PASS",
                "source_mtime_ns": after.st_mtime_ns,
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            copied.append(entry)
            if i % 500 == 0 or after.st_size > 500 * 1024**2:
                print(
                    f"[COPY] {i + 1}/{len(files)} bytes={after.st_size} path={rel}",
                    flush=True,
                )
    publish(
        STATE / "copy_summary.json",
        {
            "status": "PASS",
            "files": len(copied),
            "bytes": total,
            "source": str(SOURCE),
            "destination": str(ROOT),
        },
    )
    # Whole preprocessed stream readability is CPU-only, strict FFmpeg decoder exits.
    decode = []
    for i in range(1, 6):
        p = ROOT / f"data/01_PreProc_2K_1920x1080/{i:02d}_PreProc_2K.mp4"
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-err_detect",
                "explode",
                "-threads",
                "8",
                "-i",
                str(p),
                "-map",
                "0:v:0",
                "-an",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
        decode.append(
            {
                "path": str(p.relative_to(ROOT)),
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        )
        publish(STATE / "preproc_decode.json", decode)
        if result.returncode:
            raise RuntimeError(f"PreProc decode unreadable: {p}")
        print(f"[PREPROC_DECODE] video={i:02d} status=PASS", flush=True)
    print("COPY COMPLETE + PREPROC DECODE PASS", flush=True)


if __name__ == "__main__":
    main()
