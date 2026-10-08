"""Assemble immutable bridges once and each independent middle exactly once."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def raw_array_digest(array: np.ndarray) -> str:
    """Hash C-order tensor bytes without copying a full FP32 artifact."""
    if not array.flags.c_contiguous:
        raise RuntimeError("Raw tensor hash requires a C-contiguous array")
    raw = memoryview(array).cast("B")
    hasher = hashlib.sha256()
    for offset in range(0, len(raw), RAW_HASH_CHUNK_BYTES):
        hasher.update(raw[offset : offset + RAW_HASH_CHUNK_BYTES])
    return hasher.hexdigest()


def verify_raw_artifact(
    path: Path,
    expected_sha256: str,
    expected_shape: tuple[int, ...],
    label: str,
) -> None:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError(f"{label} status lacks a raw tensor SHA-256")
    array = np.load(path, mmap_mode="r")
    if array.shape != expected_shape or array.dtype != np.float32:
        raise RuntimeError(
            f"Invalid {label} artifact: {array.shape}, {array.dtype}"
        )
    actual = raw_array_digest(array)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} raw tensor SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def verify_assembly_inputs(plan: dict) -> None:
    reference_path = ROOT / "reference_bank/reference_manifest.json"
    reference = json.loads(reference_path.read_text())
    b00 = plan["bridges"][0]
    b00_path = Path(b00["reuse_full_context"]).expanduser().resolve()
    if (
        b00.get("bridge_id") != "B00"
        or b00_path != Path(reference.get("source", "")).expanduser().resolve()
        or digest(b00_path) != reference.get("source_sha256")
    ):
        raise RuntimeError("B00 reuse context differs from the reference manifest source")
    b00_array = np.load(b00_path, mmap_mode="r")
    if b00_array.shape != (65, 3, 1080, 1920) or b00_array.dtype != np.float32:
        raise RuntimeError("Invalid B00 reference source artifact")

    for bridge in plan["bridges"][1:]:
        bridge_id = bridge["bridge_id"]
        folder = ROOT / "bridge_bank" / bridge_id
        status = json.loads((folder / "status.json").read_text())
        if status.get("status") != "PASS":
            raise RuntimeError(f"Bridge is incomplete: {bridge_id}")
        verify_raw_artifact(
            folder / "context_65f_fp32.npy",
            status.get("context_sha256"),
            (65, 3, 1080, 1920),
            f"bridge {bridge_id}",
        )

    for body in plan["bodies"]:
        body_id = body["body_id"]
        folder = ROOT / "bodies" / body_id
        status = json.loads((folder / "status.json").read_text())
        if status.get("status") != "PASS":
            raise RuntimeError(f"Body is incomplete: {body_id}")
        verify_raw_artifact(
            folder / "generated_81f_fp32.npy",
            status.get("output_sha256"),
            (81, 3, 1080, 1920),
            f"body {body_id}",
        )


def bridge_array(bridge: dict) -> np.ndarray:
    if bridge["bridge_id"] == "B00":
        path = Path(bridge["reuse_full_context"])
    else:
        path = ROOT / "bridge_bank" / bridge["bridge_id"] / "context_65f_fp32.npy"
    full = np.load(path, mmap_mode="r")
    begin, end = int(bridge["extract_start"]), int(bridge["extract_stop"])
    result = full[begin:end]
    if result.shape != (9, 3, 1080, 1920) or result.dtype != np.float32:
        raise RuntimeError(f"Invalid bridge artifact: {bridge['bridge_id']}")
    return result


def body_middle(body: dict) -> np.ndarray:
    path = ROOT / "bodies" / body["body_id"] / "generated_81f_fp32.npy"
    full = np.load(path, mmap_mode="r")
    result = full[9:73]
    if result.shape != (64, 3, 1080, 1920) or result.dtype != np.float32:
        raise RuntimeError(f"Invalid body artifact: {body['body_id']}")
    return result


def write_rgb(process, frames: np.ndarray, limit: int | None = None) -> int:
    count = len(frames) if limit is None else min(len(frames), limit)
    for index in range(count):
        frame = np.clip(np.rint(np.moveaxis(frames[index], 0, -1) * 255.0), 0, 255).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "generated.mp4")
    args = parser.parse_args()
    plan = json.loads((ROOT / "plan.json").read_text())
    if len(plan["bridges"]) != 15 or len(plan["bodies"]) != 14:
        raise ValueError("Assembly requires 15 bridges and 14 bodies")
    # Verify every producer's actual raw FP32 bytes before opening ffmpeg.  The
    # status hashes intentionally cover tensor bytes, not the NPY container.
    verify_assembly_inputs(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", "1920x1080", "-r", "24", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    emitted = 0
    lineage = []
    try:
        for index, body in enumerate(plan["bodies"]):
            bridge = plan["bridges"][index]
            bridge_frames = bridge_array(bridge)
            begin = emitted
            emitted += write_rgb(process, bridge_frames)
            lineage.append({"kind": "bridge", "id": bridge["bridge_id"], "output_range": [begin, emitted]})
            middle = body_middle(body)
            begin = emitted
            emitted += write_rgb(process, middle)
            lineage.append({"kind": "middle", "id": body["body_id"], "output_range": [begin, emitted]})
        final_bridge = bridge_array(plan["bridges"][-1])
        begin = emitted
        emitted += write_rgb(process, final_bridge, limit=1025 - emitted)
        lineage.append({"kind": "bridge_tail", "id": "B14", "output_range": [begin, emitted], "cropped_virtual_frames": 6})
        process.stdin.close()
        code = process.wait()
    except BaseException:
        process.kill()
        raise
    if code or emitted != 1025:
        raise RuntimeError(f"Assembly failed: ffmpeg={code}, frames={emitted}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(args.output)],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(probe.stdout)
    manifest = {
        "status": "PASS",
        "output": str(args.output),
        "output_sha256": digest(args.output),
        "frames": emitted,
        "fps": 24,
        "lineage": lineage,
        "ffprobe": metadata,
        "duplicate_count": 0,
        "gap_count": 0,
    }
    (ROOT / "assembly_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "experiment_1_assembled", "output": str(args.output), "sha256": manifest["output_sha256"]}))


if __name__ == "__main__":
    main()
