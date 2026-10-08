#!/usr/bin/env python3
"""Lanczos-downsample one 7680x4320 MP4 to 1920x1080.

Usage: python resize_8k_to_2k.py /path/to/01_Input_8K.mp4
The output and its provenance JSON are written to pipeline/data/01_Resize_2K_1920_1080.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "data" / "01_Resize_2K_1920_1080"
SCALE_FILTER = "scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1"
VIDEO_PRESET = "fast"
VIDEO_CRF = 18
EXPECTED_INPUT_SIZE = (7680, 4320)
OUTPUT_SIZE = (1920, 1080)


def probe(path: Path, ffprobe: str) -> tuple[dict, dict, list[dict]]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    videos = [stream for stream in data["streams"] if stream["codec_type"] == "video"]
    audios = [stream for stream in data["streams"] if stream["codec_type"] == "audio"]
    if len(videos) != 1:
        raise ValueError(f"Expected one video stream in {path}, found {len(videos)}")
    return data, videos[0], audios


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_all_frames(path: Path, ffmpeg: str) -> int:
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error",
            "-xerror", "-err_detect", "explode", "-threads", "8", "-i", str(path),
            "-map", "0:v:0", "-map", "0:a?", "-fps_mode", "passthrough",
            "-f", "null", "-", "-progress", "pipe:1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    progress = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if progress.get("progress") != "end" or "frame" not in progress:
        raise ValueError(f"Full decode did not finish for {path}")
    return int(progress["frame"])


def first_version_line(executable: str) -> str:
    result = subprocess.run(
        [executable, "-version"], check=True, capture_output=True, text=True
    )
    return result.stdout.splitlines()[0]


def resize(input_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    if not input_path.is_file() or input_path.suffix.lower() != ".mp4":
        raise ValueError(f"Input must be an existing MP4 file: {input_path}")
    if "Input_8K" not in input_path.name:
        raise ValueError("Input filename must contain 'Input_8K' for the required rename")

    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_path = output_dir / input_path.name.replace("Input_8K", "Resize_2K", 1)
    partial_path = output_path.with_name(output_path.stem + ".partial.mp4")
    if output_path == input_path or partial_path.exists() or (output_path.exists() and not overwrite):
        raise FileExistsError(f"Refusing to overwrite input, output, or partial file: {output_path}")

    input_data, input_video, input_audio = probe(input_path, ffprobe)
    if (input_video["width"], input_video["height"]) != EXPECTED_INPUT_SIZE:
        raise ValueError(f"Expected 7680x4320 input, got {input_video['width']}x{input_video['height']}")
    if input_video.get("color_range") != "pc" or input_video.get("color_space") != "bt709":
        raise ValueError("Expected full-range BT.709 input; refusing an incorrect color conversion")
    if input_video.get("pix_fmt") != "yuv420p10le":
        raise ValueError("Expected 10-bit yuv420p10le input")
    if not input_video.get("nb_frames"):
        raise ValueError("Input frame count is missing from ffprobe metadata")

    input_stat = input_path.stat()
    input_hash = sha256(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-xerror",
        "-err_detect", "explode", "-threads", "8", "-i", str(input_path),
        "-map", "0:v:0", "-map", "0:a?", "-vf", SCALE_FILTER,
        "-fps_mode", "passthrough", "-c:v", "libx265", "-preset", VIDEO_PRESET,
        "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p10le",
        "-x265-params", "log-level=error:pools=8", "-color_range", "pc",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-tag:v", "hvc1",
        "-c:a", "copy", "-map_metadata", "0", "-video_track_timescale", "16384",
        "-n", str(partial_path),
    ]
    print(f"Processing: {input_path}\nOutput: {output_path}\nCommand: {shlex.join(command)}", flush=True)
    try:
        subprocess.run(command, check=True)
        output_data, output_video, output_audio = probe(partial_path, ffprobe)
        if (output_video["width"], output_video["height"]) != OUTPUT_SIZE:
            raise ValueError("Output resolution is not 1920x1080")
        if Fraction(output_video["avg_frame_rate"]) != Fraction(input_video["avg_frame_rate"]):
            raise ValueError("Output FPS differs from input")
        if int(output_video["nb_frames"]) != int(input_video["nb_frames"]):
            raise ValueError("Output frame count differs from input")
        if abs(float(output_data["format"]["duration"]) - float(input_data["format"]["duration"])) > 0.001:
            raise ValueError("Output duration differs from input")
        if len(output_audio) != len(input_audio) or any(
            output["codec_name"] != source["codec_name"]
            for source, output in zip(input_audio, output_audio)
        ):
            raise ValueError("Output audio streams differ from input")
        decoded_frames = decode_all_frames(partial_path, ffmpeg)
        if decoded_frames != int(input_video["nb_frames"]):
            raise ValueError(f"Decoded {decoded_frames} frames, expected {input_video['nb_frames']}")
        if (input_path.stat().st_size, input_path.stat().st_mtime_ns) != (
            input_stat.st_size, input_stat.st_mtime_ns
        ):
            raise RuntimeError("Input file changed during processing")
        partial_path.replace(output_path)
    finally:
        partial_path.unlink(missing_ok=True)

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "input_sha256": input_hash,
        "output_sha256": sha256(output_path),
        "ffmpeg_version": first_version_line(ffmpeg),
        "ffprobe_version": first_version_line(ffprobe),
        "python_version": sys.version.split()[0],
        "ffmpeg_command": command,
        "filter": SCALE_FILTER,
        "video_codec": "libx265",
        "pix_fmt": "yuv420p10le",
        "preset": VIDEO_PRESET,
        "crf": VIDEO_CRF,
        "audio": "copy",
        "seed": None,
        "seed_note": "No random operation is used; a seed is not required.",
        "verification": {
            "resolution": list(OUTPUT_SIZE),
            "fps": output_video["avg_frame_rate"],
            "frame_count": int(output_video["nb_frames"]),
            "decoded_frames": decoded_frames,
            "duration_seconds": float(output_data["format"]["duration"]),
            "audio_streams": len(output_audio),
            "full_decode": "ok",
        },
    }
    manifest_path = output_path.with_suffix(".json")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(f"Verified: {output_path} (1920x1080, {decoded_frames} frames, {output_video['avg_frame_rate']} fps)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_video", type=Path, help="One 7680x4320 MP4 input")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Atomically replace an existing generated output after successful validation",
    )
    args = parser.parse_args()
    try:
        resize(args.input_video, args.output_dir, args.overwrite)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
