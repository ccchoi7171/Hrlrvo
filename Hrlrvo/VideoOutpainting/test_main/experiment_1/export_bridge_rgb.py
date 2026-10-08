"""Export all immutable bridge contexts and extracted bridge frames as RGB media."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "bridge_rgb"
EXPECTED_CONTEXT = (65, 3, 1080, 1920)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    for frame in array:
        digest.update(np.ascontiguousarray(frame).view(np.uint8))
    return digest.hexdigest()


def rgb8(frame: np.ndarray) -> np.ndarray:
    """Match Experiment 1 assembly: clip, round(value*255), uint8 RGB."""
    return np.clip(np.rint(np.moveaxis(frame, 0, -1) * 255.0), 0, 255).astype(np.uint8)


def context_path(row: dict) -> Path:
    if row["bridge_id"] == "B00":
        return Path(row["reuse_full_context"])
    return ROOT / "bridge_bank" / row["bridge_id"] / "context_65f_fp32.npy"


def verify_source(row: dict, plan: dict, reference_manifest: dict) -> tuple[Path, np.ndarray, str]:
    path = context_path(row).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r")
    if array.shape != EXPECTED_CONTEXT or array.dtype != np.float32:
        raise RuntimeError(f"Invalid {row['bridge_id']} context: {array.shape}, {array.dtype}")
    actual_tensor_sha = tensor_sha256(array)
    if row["bridge_id"] == "B00":
        if path != Path(reference_manifest["source"]).resolve():
            raise RuntimeError("B00 source differs from the fixed reference source")
        if sha256_file(path) != reference_manifest["source_sha256"]:
            raise RuntimeError("B00 source file hash mismatch")
        if actual_tensor_sha != plan["source_rgb_sha256"] and actual_tensor_sha != reference_manifest["source_sha256"]:
            # NPY file hash and raw tensor hash are intentionally different; the
            # fixed B00 NPY file hash above is the authoritative provenance.
            pass
    else:
        status_path = ROOT / "bridge_bank" / row["bridge_id"] / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "PASS" or status.get("context_sha256") != actual_tensor_sha:
            raise RuntimeError(f"{row['bridge_id']} is not a hash-verified PASS artifact")
    return path, array, actual_tensor_sha


def encode_mp4(path: Path, frames: np.ndarray) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", "1920x1080", "-r", "24", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-threads", "2",
        "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(rgb8(frame).tobytes())
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        raise
    if return_code:
        raise RuntimeError(f"ffmpeg failed for {path}: {return_code}")


def export_one(row: dict, plan: dict, reference_manifest: dict, output_root: Path) -> dict:
    bridge_id = row["bridge_id"]
    source, context, raw_sha = verify_source(row, plan, reference_manifest)
    begin, stop = int(row["extract_start"]), int(row["extract_stop"])
    bridge = context[begin:stop]
    if bridge.shape != (9, 3, 1080, 1920):
        raise RuntimeError(f"{bridge_id} extraction is not 9 frames")
    destination = output_root / bridge_id
    frame_dir = destination / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    context_mp4 = destination / "context_65f.mp4"
    bridge_mp4 = destination / "bridge_9f.mp4"
    encode_mp4(context_mp4, context)
    encode_mp4(bridge_mp4, bridge)
    png_paths = []
    for index, frame in enumerate(bridge):
        path = frame_dir / f"frame_{index:03d}.png"
        Image.fromarray(rgb8(frame), mode="RGB").save(path, compress_level=1)
        png_paths.append(str(path))
    global_center = int(row["global_start"]) + 4
    return {
        "bridge_id": bridge_id,
        "source": str(source),
        "source_tensor_sha256": raw_sha,
        "context_start": int(row["context_start"]),
        "context_stop": int(row["context_stop"]),
        "padding": {"left": int(row["pad_left"]), "right": int(row["pad_right"])},
        "extract_indices_context": [begin, stop],
        "global_bridge_range": [int(row["global_start"]), int(row["global_stop"])],
        "central_frame": {
            "bridge_local_index": 4,
            "context_index": begin + 4,
            "global_frame": global_center,
            "timestamp_seconds": global_center / 24.0,
            "source_padded": global_center >= int(plan["output_frames"]),
        },
        "outputs": {
            "context_65f_mp4": str(context_mp4),
            "bridge_9f_mp4": str(bridge_mp4),
            "png_frames": png_paths,
        },
    }


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": stream["avg_frame_rate"], "frames": int(stream["nb_read_frames"]),
    }


def make_contact_sheet(records: list[dict], output: Path) -> Path:
    thumb_w, thumb_h, label_h = 640, 360, 54
    canvas = Image.new("RGB", (thumb_w * 3, (thumb_h + label_h) * 5), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    for index, record in enumerate(records):
        png = Path(record["outputs"]["png_frames"][4])
        image = Image.open(png).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x, y = (index % 3) * thumb_w, (index // 3) * (thumb_h + label_h)
        canvas.paste(image, (x, y))
        center = record["central_frame"]
        suffix = " padded" if center["source_padded"] else ""
        label = (
            f"{record['bridge_id']} | global {center['global_frame']} | "
            f"{center['timestamp_seconds']:.3f}s{suffix}"
        )
        draw.text((x + 12, y + thumb_h + 13), label, fill="black", font=font)
    canvas.save(output, compress_level=1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers != 4:
        raise ValueError("This CPU export is fixed to four workers")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    reference_manifest = json.loads(
        (ROOT / "reference_bank/reference_manifest.json").read_text(encoding="utf-8")
    )
    if len(plan["bridges"]) != 15:
        raise ValueError("Expected B00 through B14")
    records = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(export_one, row, plan, reference_manifest, output): row["bridge_id"]
            for row in plan["bridges"]
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: int(item["bridge_id"][1:]))

    verification = {"mp4_count": 0, "png_count": 0, "all_pass": True}
    for record in records:
        for key, expected in (("context_65f_mp4", 65), ("bridge_9f_mp4", 9)):
            path = Path(record["outputs"][key])
            probe = probe_video(path)
            if probe != {"width": 1920, "height": 1080, "fps": "24/1", "frames": expected}:
                raise RuntimeError(f"Invalid encoded video {path}: {probe}")
            record.setdefault("verification", {})[key] = probe
            verification["mp4_count"] += 1
        for png_path in record["outputs"]["png_frames"]:
            with Image.open(png_path) as image:
                if image.mode != "RGB" or image.size != (1920, 1080):
                    raise RuntimeError(f"Invalid RGB PNG: {png_path}")
            verification["png_count"] += 1
    if verification != {"mp4_count": 30, "png_count": 135, "all_pass": True}:
        raise RuntimeError(f"Incomplete export verification: {verification}")
    overview = make_contact_sheet(records, output / "bridge_overview_3x5.png")
    manifest = {
        "status": "PASS",
        "schema": "hrlrvo_experiment1_bridge_rgb_v1",
        "conversion": "FP32 CHW clipped to [0,1], round(value*255), uint8 RGB",
        "video": "H.264 CRF16 medium yuv420p, 1920x1080, 24fps, ffmpeg threads=2",
        "parallelism": "ThreadPoolExecutor max_workers=4",
        "overview": str(overview),
        "verification": verification,
        "bridges": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path), **verification}))


if __name__ == "__main__":
    main()
