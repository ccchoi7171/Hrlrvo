"""CPU-only acceptance harness; preprocessing is ONLY the real shared stage.

Runs run_cert.sh --stage preprocess with validation-only configurations, then
compares full RGB24 framemd5 sequences. Never imports or invokes a model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/preprocess_reproduction"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def metadata(path: Path) -> dict:
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    data = json.loads(result)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    return {
        k: video.get(k)
        for k in (
            "width",
            "height",
            "avg_frame_rate",
            "nb_frames",
            "pix_fmt",
            "codec_name",
            "time_base",
        )
    }


def hashes(path: Path, output: Path) -> tuple[list[tuple[int, str]], list[str]]:
    command = [
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
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "framemd5",
        "-",
    ]
    with output.open("x") as stream:
        subprocess.run(command, stdout=stream, check=True, text=True)
    rows = [
        line.split(",")
        for line in output.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    return [(int(row[-2]), row[-1].strip()) for row in rows], command


def validate(video: str) -> dict:
    directory = OUTPUT / video
    directory.mkdir(parents=True, exist_ok=True)
    config = ROOT / f"configs/validation/preprocess_reproduction/{video}.json"
    request = json.loads(config.read_text())
    canonical = ROOT / f"data/01_PreProc_2K_1920x1080/{video}_PreProc_2K.mp4"
    source = ROOT / request["input"]["path"]
    generated = ROOT / request["artifacts"]["preprocess"]
    if not generated.is_relative_to(directory) or generated.exists():
        raise RuntimeError(
            f"Refusing an existing or non-validation output: {generated}"
        )
    provenance = json.loads(canonical.with_suffix(".json").read_text())
    source_sha, canonical_sha = sha(source), sha(canonical)
    if (
        source_sha != provenance["input_sha256"]
        or canonical_sha != provenance["output_sha256"]
    ):
        raise RuntimeError("Canonical source/PreProc pair failed provenance SHA check")
    expected = int(metadata(canonical)["nb_frames"])
    command = [
        str(ROOT / "run_cert.sh"),
        "--config",
        str(config),
        "--stage",
        "preprocess",
    ]
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        CERT_OUTPAINT_PYTHON=sys.executable,
    )
    print(f"[REPRODUCTION] video={video} action=shared_stage_start", flush=True)
    with (directory / "stage_terminal.log").open("x") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    a_meta, b_meta = metadata(canonical), metadata(generated)
    print(
        f"[REPRODUCTION] video={video} action=full_rgb24_hashes frames={expected}",
        flush=True,
    )
    a, a_command = hashes(canonical, directory / "canonical_full_rgb24.framemd5")
    b, b_command = hashes(generated, directory / "regenerated_full_rgb24.framemd5")
    mismatches = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    first65 = sum(x == y for x, y in zip(a[:65], b[:65]))
    matched = sum(x == y for x, y in zip(a, b))
    first_pass = len(a) >= 65 and len(b) >= 65 and first65 == 65
    full_pass = len(a) == len(b) == expected and not mismatches
    metadata_pass = (
        a_meta == b_meta
        and a_meta["width"] == 1920
        and a_meta["height"] == 1080
        and a_meta["avg_frame_rate"] == "24/1"
        and a_meta["pix_fmt"] == "yuv420p10le"
        and a_meta["codec_name"] == "hevc"
    )
    unchanged = sha(source) == source_sha and sha(canonical) == canonical_sha
    result = {
        "video_id": video,
        "source": str(source),
        "source_sha256": source_sha,
        "canonical": str(canonical),
        "canonical_sha256": canonical_sha,
        "regenerated": str(generated),
        "regenerated_sha256": sha(generated),
        "config": str(config),
        "config_sha256": sha(config),
        "stage_command": command,
        "historical_ffmpeg_command": provenance["ffmpeg_command"],
        "canonical_metadata": a_meta,
        "regenerated_metadata": b_meta,
        "metadata_match": metadata_pass,
        "comparison_format": "rgb24",
        "canonical_decode_command": a_command,
        "regenerated_decode_command": b_command,
        "first65": {
            "matched": first65,
            "expected": 65,
            "status": "PASS" if first_pass else "FAIL",
        },
        "full": {
            "matched": matched,
            "expected": expected,
            "canonical_count": len(a),
            "regenerated_count": len(b),
            "status": "PASS" if full_pass else "FAIL",
        },
        "first_mismatch_zero_based": mismatches[0]
        if mismatches
        else (min(len(a), len(b)) if len(a) != len(b) else None),
        "mismatched_indices": mismatches,
        "source_and_canonical_unchanged": unchanged,
        "gpu_executed": False,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS"
        if first_pass and full_pass and metadata_pass and unchanged
        else "FAIL",
    }
    atomic_json(directory / "comparison.json", result)
    print(
        f"[REPRODUCTION] video={video} first65={first65}/65 full={matched}/{expected} status={result['status']}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        nargs="+",
        choices=["01", "02", "03", "04", "05"],
        default=["01", "02", "03", "04", "05"],
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for video in args.ids:
        try:
            result = validate(video)
        except Exception as error:
            result = {
                "video_id": video,
                "status": "FAIL",
                "error": str(error),
                "gpu_executed": False,
            }
            traceback.print_exc()
            directory = OUTPUT / video
            directory.mkdir(exist_ok=True)
            atomic_json(directory / "failure.json", result)
        results.append(result)
        atomic_json(
            OUTPUT / "summary.json",
            {
                "results": results,
                "status": "PASS"
                if len(results) == 5 and all(r["status"] == "PASS" for r in results)
                else "INCOMPLETE",
                "gpu_executed": False,
            },
        )
    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
