#!/usr/bin/env python3
"""Speed one 16 fps video up to 24 fps without changing its video frames.

The input's HEVC packets are copied byte-for-byte. Their integer timestamps are
kept while the video time base changes from 1/16384 to 1/24576. Audio is sped up
by 1.5x so it stays in sync with the shorter video.
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
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "data" / "01-1_FPS_2K_1920_1080_24fps"
INPUT_FPS = Fraction(16, 1)
OUTPUT_FPS = Fraction(24, 1)
INPUT_TIME_BASE = "1/16384"
OUTPUT_TIME_BASE = "1/24576"
VIDEO_BITSTREAM_FILTER = (
    "setts=pts=PTS:dts=DTS:duration=DURATION:time_base=1/24576"
)
AUDIO_FILTER = "atempo=1.5"
AUDIO_BITRATE = "192k"


def probe(path: Path, ffprobe: str) -> tuple[dict, dict, list[dict]]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    video = [stream for stream in data["streams"] if stream["codec_type"] == "video"]
    audio = [stream for stream in data["streams"] if stream["codec_type"] == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise ValueError(f"Expected one video and one audio stream in {path}")
    return data, video[0], audio


def video_packets(path: Path, ffprobe: str) -> list[dict]:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_packets",
            "-show_entries", "packet=pts,dts,duration,data_hash", "-show_data_hash", "sha256",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["packets"]


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
            "-map", "0:v:0", "-map", "0:a:0", "-fps_mode", "passthrough",
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
    return subprocess.run(
        [executable, "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]


def retime(input_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    if not input_path.is_file() or input_path.suffix.lower() != ".mp4":
        raise ValueError(f"Input must be an existing MP4 file: {input_path}")
    if "Resize_2K" not in input_path.name:
        raise ValueError("Input filename must contain 'Resize_2K'")

    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_path = output_dir / input_path.name.replace("Resize_2K", "Input_2K", 1)
    partial_path = output_path.with_name(output_path.stem + ".partial.mp4")
    if output_path == input_path or partial_path.exists() or (output_path.exists() and not overwrite):
        raise FileExistsError(f"Refusing to overwrite input, output, or partial file: {output_path}")

    input_data, input_video, input_audio = probe(input_path, ffprobe)
    if (input_video["width"], input_video["height"]) != (1920, 1080):
        raise ValueError("Expected 1920x1080 input")
    if Fraction(input_video["avg_frame_rate"]) != INPUT_FPS:
        raise ValueError("Expected 16 fps input")
    if input_video["time_base"] != INPUT_TIME_BASE:
        raise ValueError(f"Expected video time base {INPUT_TIME_BASE}")
    if not input_video.get("nb_frames"):
        raise ValueError("Input frame count is missing from ffprobe metadata")

    frame_count = int(input_video["nb_frames"])
    target_duration = Fraction(frame_count, OUTPUT_FPS)
    input_stat = input_path.stat()
    input_hash = sha256(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-xerror",
        "-err_detect", "explode", "-i", str(input_path),
        "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", "-copytb", "1",
        "-bsf:v", VIDEO_BITSTREAM_FILTER, "-fps_mode:v", "passthrough",
        "-af", AUDIO_FILTER, "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-tag:v", "hvc1", "-map_metadata", "0", "-video_track_timescale", "24576",
        "-n", str(partial_path),
    ]
    print(f"Processing: {input_path}\nOutput: {output_path}\nCommand: {shlex.join(command)}", flush=True)
    try:
        subprocess.run(command, check=True)
        output_data, output_video, output_audio = probe(partial_path, ffprobe)
        if (output_video["width"], output_video["height"]) != (1920, 1080):
            raise ValueError("Output resolution changed")
        if Fraction(output_video["avg_frame_rate"]) != OUTPUT_FPS:
            raise ValueError("Output is not exactly 24 fps")
        if output_video["time_base"] != OUTPUT_TIME_BASE:
            raise ValueError("Output video time base is incorrect")
        if int(output_video["nb_frames"]) != frame_count:
            raise ValueError("Output frame count differs from input")
        if abs(float(output_video["duration"]) - float(target_duration)) > 0.001:
            raise ValueError("Output video duration is incorrect")
        if video_packets(input_path, ffprobe) != video_packets(partial_path, ffprobe):
            raise ValueError("A video packet, its order, or its timestamp changed")
        if output_audio[0]["codec_name"] != "aac":
            raise ValueError("Output audio is not AAC")
        if abs(float(output_audio[0]["duration"]) - float(target_duration)) > 0.05:
            raise ValueError("Output audio duration is not synchronized with video")
        if abs(float(output_data["format"]["duration"]) - float(target_duration)) > 0.05:
            raise ValueError("Output container duration is incorrect")
        decoded_frames = decode_all_frames(partial_path, ffmpeg)
        if decoded_frames != frame_count:
            raise ValueError(f"Decoded {decoded_frames} frames, expected {frame_count}")
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
        "video_method": "stream copy with exact time-base change",
        "video_bitstream_filter": VIDEO_BITSTREAM_FILTER,
        "audio_filter": AUDIO_FILTER,
        "audio_codec": "aac",
        "audio_bitrate": AUDIO_BITRATE,
        "seed": None,
        "seed_note": "No random operation is used; a seed is not required.",
        "verification": {
            "resolution": [1920, 1080],
            "input_fps": str(INPUT_FPS),
            "output_fps": str(OUTPUT_FPS),
            "input_frame_count": frame_count,
            "output_frame_count": int(output_video["nb_frames"]),
            "decoded_frames": decoded_frames,
            "expected_video_duration_seconds": float(target_duration),
            "video_duration_seconds": float(output_video["duration"]),
            "audio_duration_seconds": float(output_audio[0]["duration"]),
            "container_duration_seconds": float(output_data["format"]["duration"]),
            "video_packets_and_order_identical": True,
            "full_decode": "ok",
        },
    }
    manifest_path = output_path.with_suffix(".json")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(
        f"Verified: {output_path} (24 fps, {frame_count} unchanged video frames, "
        f"{output_video['duration']} s video)",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_video", type=Path, help="One 16 fps 1920x1080 MP4 input")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Atomically replace an existing generated output after successful validation",
    )
    args = parser.parse_args()
    try:
        retime(args.input_video, args.output_dir, args.overwrite)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
