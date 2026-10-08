#!/usr/bin/env python3
"""Resize one 8K/16fps MP4 to 2K and retime it to 24fps in one stage.

The video is downsampled once with Lanczos and keeps exactly the same frames in
the same order. Timestamps are compressed by 1.5x, so 16fps becomes 24fps
without interpolation. Audio is sped up by 1.5x to remain synchronized.
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
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "data" / "01_PreProc_2K_1920x1080"
EXPECTED_INPUT_SIZE = (7680, 4320)
OUTPUT_SIZE = (1920, 1080)
INPUT_FPS = Fraction(16, 1)
OUTPUT_FPS = Fraction(24, 1)
OUTPUT_TIME_BASE = "1/24576"
VIDEO_FILTER = (
    "scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,"
    "setsar=1,setpts=PTS*2/3,fps=24"
)
AUDIO_FILTER = "atempo=1.5"
VIDEO_PRESET = "fast"
VIDEO_CRF = 18
AUDIO_BITRATE = "192k"


def probe(path: Path, ffprobe: str) -> tuple[dict, dict, list[dict]]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    videos = [stream for stream in data["streams"] if stream["codec_type"] == "video"]
    audios = [stream for stream in data["streams"] if stream["codec_type"] == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ValueError(f"Expected one video and one audio stream in {path}")
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
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel",
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
            "-map",
            "0:a:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "null",
            "-",
            "-progress",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    progress = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if progress.get("progress") != "end" or "frame" not in progress:
        raise ValueError(f"Full decode did not finish for {path}")
    return int(progress["frame"])


def first_version_line(executable: str) -> str:
    return subprocess.run(
        [executable, "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    if "Input_8K" not in input_path.name:
        raise ValueError("Input filename must contain 'Input_8K'")
    output_name = input_path.name.replace("Input_8K", "PreProc_2K", 1)
    return Path(output_dir) / output_name


def build_ffmpeg_command(
    input_path: Path, partial_path: Path, ffmpeg: str
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-threads",
        "8",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-vf",
        VIDEO_FILTER,
        "-fps_mode:v",
        "passthrough",
        "-c:v",
        "libx265",
        "-preset",
        VIDEO_PRESET,
        "-crf",
        str(VIDEO_CRF),
        "-pix_fmt",
        "yuv420p10le",
        "-x265-params",
        "log-level=error:pools=8",
        "-color_range",
        "pc",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-tag:v",
        "hvc1",
        "-af",
        AUDIO_FILTER,
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE,
        "-map_metadata",
        "0",
        "-video_track_timescale",
        "24576",
        "-n",
        str(partial_path),
    ]


def preprocess(input_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    if not input_path.is_file() or input_path.suffix.lower() != ".mp4":
        raise ValueError(f"Input must be an existing MP4 file: {input_path}")

    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_path = output_path_for(input_path, output_dir)
    partial_path = output_path.with_name(output_path.stem + ".partial.mp4")
    if (
        output_path == input_path
        or partial_path.exists()
        or (output_path.exists() and not overwrite)
    ):
        raise FileExistsError(
            f"Refusing to overwrite input, output, or partial file: {output_path}"
        )

    input_data, input_video, input_audio = probe(input_path, ffprobe)
    if (input_video["width"], input_video["height"]) != EXPECTED_INPUT_SIZE:
        raise ValueError(
            f"Expected 7680x4320 input, got {input_video['width']}x{input_video['height']}"
        )
    if Fraction(input_video["avg_frame_rate"]) != INPUT_FPS:
        raise ValueError(f"Expected 16fps input, got {input_video['avg_frame_rate']}")
    if (
        input_video.get("color_range") != "pc"
        or input_video.get("color_space") != "bt709"
    ):
        raise ValueError(
            "Expected full-range BT.709 input; refusing an incorrect color conversion"
        )
    if input_video.get("pix_fmt") != "yuv420p10le":
        raise ValueError("Expected 10-bit yuv420p10le input")
    if not input_video.get("nb_frames"):
        raise ValueError("Input frame count is missing from ffprobe metadata")

    frame_count = int(input_video["nb_frames"])
    target_duration = Fraction(frame_count, OUTPUT_FPS)
    input_stat = input_path.stat()
    input_hash = sha256(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(input_path, partial_path, ffmpeg)
    print(
        f"Processing: {input_path}\nOutput: {output_path}\nCommand: {shlex.join(command)}",
        flush=True,
    )
    try:
        subprocess.run(command, check=True)
        output_data, output_video, output_audio = probe(partial_path, ffprobe)
        if (output_video["width"], output_video["height"]) != OUTPUT_SIZE:
            raise ValueError("Output resolution is not 1920x1080")
        if Fraction(output_video["avg_frame_rate"]) != OUTPUT_FPS:
            raise ValueError("Output is not exactly 24fps")
        if output_video["time_base"] != OUTPUT_TIME_BASE:
            raise ValueError(f"Output video time base is not {OUTPUT_TIME_BASE}")
        if int(output_video["nb_frames"]) != frame_count:
            raise ValueError("Output frame count differs from input")
        if abs(float(output_video["duration"]) - float(target_duration)) > 0.001:
            raise ValueError("Output video duration is incorrect")
        if output_video.get("pix_fmt") != "yuv420p10le":
            raise ValueError("Output is not 10-bit yuv420p10le")
        if (
            output_video.get("color_range") != "pc"
            or output_video.get("color_space") != "bt709"
        ):
            raise ValueError("Output is not full-range BT.709")
        if output_audio[0]["codec_name"] != "aac":
            raise ValueError("Output audio is not AAC")
        if abs(float(output_audio[0]["duration"]) - float(target_duration)) > 0.05:
            raise ValueError("Output audio duration is not synchronized with video")
        if (
            abs(float(output_data["format"]["duration"]) - float(target_duration))
            > 0.05
        ):
            raise ValueError("Output container duration is incorrect")
        decoded_frames = decode_all_frames(partial_path, ffmpeg)
        if decoded_frames != frame_count:
            raise ValueError(f"Decoded {decoded_frames} frames, expected {frame_count}")
        if (input_path.stat().st_size, input_path.stat().st_mtime_ns) != (
            input_stat.st_size,
            input_stat.st_mtime_ns,
        ):
            raise RuntimeError("Input file changed during processing")
        partial_path.replace(output_path)
    finally:
        partial_path.unlink(missing_ok=True)

    manifest = {
        "stage": "8K_to_2K_24fps_preprocess",
        "input": str(input_path),
        "output": str(output_path),
        "input_sha256": input_hash,
        "output_sha256": sha256(output_path),
        "ffmpeg_version": first_version_line(ffmpeg),
        "ffprobe_version": first_version_line(ffprobe),
        "python_version": sys.version.split()[0],
        "ffmpeg_command": command,
        "operations": [
            "Lanczos resize from 7680x4320 to 1920x1080",
            "Retiming from 16fps to 24fps without frame interpolation",
            "Audio speed-up by 1.5x",
        ],
        "video_filter": VIDEO_FILTER,
        "video_codec": "libx265",
        "pix_fmt": "yuv420p10le",
        "preset": VIDEO_PRESET,
        "crf": VIDEO_CRF,
        "audio_filter": AUDIO_FILTER,
        "audio_codec": "aac",
        "audio_bitrate": AUDIO_BITRATE,
        "seed": None,
        "seed_note": "No random operation is used; a seed is not required.",
        "verification": {
            "input_resolution": list(EXPECTED_INPUT_SIZE),
            "output_resolution": list(OUTPUT_SIZE),
            "input_fps": str(INPUT_FPS),
            "output_fps": str(OUTPUT_FPS),
            "input_frame_count": frame_count,
            "output_frame_count": int(output_video["nb_frames"]),
            "decoded_frames": decoded_frames,
            "input_duration_seconds": float(input_data["format"]["duration"]),
            "expected_output_video_duration_seconds": float(target_duration),
            "output_video_duration_seconds": float(output_video["duration"]),
            "output_audio_duration_seconds": float(output_audio[0]["duration"]),
            "output_container_duration_seconds": float(
                output_data["format"]["duration"]
            ),
            "frame_interpolation": False,
            "full_decode": "ok",
            "input_audio_streams": len(input_audio),
            "output_audio_streams": len(output_audio),
        },
    }
    manifest_path = output_path.with_suffix(".json")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    os.replace(temporary_manifest, manifest_path)
    print(
        f"Verified: {output_path} (1920x1080, 24fps, {frame_count} unchanged frames)",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_video", type=Path, help="One 7680x4320, 16fps MP4 input")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output after successful validation",
    )
    args = parser.parse_args()
    try:
        preprocess(args.input_video, args.output_dir, args.overwrite)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
