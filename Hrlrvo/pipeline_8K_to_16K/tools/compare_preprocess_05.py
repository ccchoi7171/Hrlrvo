"""CPU-only decoded-frame identity gate. Never starts inference."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path):
    value = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
            text=True,
        )
    )
    v = next(s for s in value["streams"] if s["codec_type"] == "video")
    return {
        key: v.get(key)
        for key in (
            "width",
            "height",
            "avg_frame_rate",
            "nb_frames",
            "pix_fmt",
            "codec_name",
        )
    }


def frame_hashes(path, dest):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-threads",
        "8",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-frames:v",
        "65",
        "-pix_fmt",
        "rgb24",
        "-f",
        "framemd5",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    dest.write_text(result.stdout)
    return [
        line.split(",")[-1].strip()
        for line in result.stdout.splitlines()
        if line and not line.startswith("#")
    ]


def main():
    old = ROOT / "data/01_PreProc_2K_1920x1080/05_PreProc_2K.mp4"
    new = ROOT / "data/validation/05_preprocess_from_8k/05_PreProc_2K.mp4"
    output = ROOT / "data/validation"
    a = frame_hashes(old, output / "05_old_first65_rgb24.framemd5")
    b = frame_hashes(new, output / "05_new_first65_rgb24.framemd5")
    mismatches = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
    metadata_a, metadata_b = probe(old), probe(new)
    metadata_equal = all(
        metadata_a[k] == metadata_b[k]
        for k in ("width", "height", "avg_frame_rate", "nb_frames", "pix_fmt")
    )
    exact = len(a) == len(b) == 65 and not mismatches and metadata_equal
    result = {
        "seed": 71634910,
        "old_preproc": str(old),
        "regenerated_preproc": str(new),
        "old_preproc_sha256": sha(old),
        "regenerated_preproc_sha256": sha(new),
        "old_metadata": metadata_a,
        "new_metadata": metadata_b,
        "metadata_match": metadata_equal,
        "decoded_pixel_format": "rgb24",
        "old_decoded_frames": len(a),
        "new_decoded_frames": len(b),
        "matched_frames": sum(x == y for x, y in zip(a, b)),
        "first_mismatch_zero_based": mismatches[0] if mismatches else None,
        "mismatched_frames_zero_based": mismatches,
        "PREPROCESS_05_FIRST65_DECODED_IDENTITY": "PASS" if exact else "FAIL",
        "gpu_followup_allowed": exact,
        "outpaint_reproduction": "NOT_RUN"
        if exact
        else "STOPPED_PREPROCESS_IDENTITY_FAIL",
        "diagnosis": "Old exact preprocessing provenance says CRF18; audited Stage0001 uses CRF1. No encoder/model parameters changed to force agreement.",
    }
    (output / "reproducibility_05.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
