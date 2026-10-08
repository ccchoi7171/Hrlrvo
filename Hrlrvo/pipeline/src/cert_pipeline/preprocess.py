"""Frame-preserving 8K/16 fps to FHD/24 fps preprocessing.

This module packages the behavior of ``src/00to01/preproc_8k_to_2k_24.py``
for the certification runner.  The source frames are resized once with
Lanczos and their timestamps are compressed by 1.5x.  No interpolated frame is
created: source frame ``k`` remains output frame ``k``.

The module performs work only when :func:`run_preprocess` is called.  Importing
it never invokes ffmpeg or ffprobe.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_INPUT_SIZE = (7680, 4320)
EXPECTED_INPUT_FPS = Fraction(16, 1)
OUTPUT_SIZE = (1920, 1080)
OUTPUT_FPS = Fraction(24, 1)
VIDEO_FILTER = (
    "scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,"
    "setsar=1,setpts=PTS*2/3,fps=24"
)
AUDIO_FILTER = "atempo=1.5"
VIDEO_CODEC = "libx265"
VIDEO_PIXEL_FORMAT = "yuv420p10le"
VIDEO_CRF = 1
VIDEO_PRESET = "fast"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"


class PreprocessError(RuntimeError):
    """Raised when preprocessing inputs or results violate the contract."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_identity(path: Path) -> dict[str, Any]:
    """Bind resume artifacts to the actual source bytes, not its filename."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PreprocessError(f"Input changed during checksum: {path}")
    return {
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _log(log_path: Path, message: str) -> None:
    line = f"{_utc_now()} [PREPROCESS] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _fraction(value: object, field: str) -> Fraction:
    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise PreprocessError(f"Invalid {field}: {value!r}") from exc
    if fraction <= 0:
        raise PreprocessError(f"Invalid {field}: {value!r}")
    return fraction


def _video_stream(probe_data: Mapping[str, Any]) -> Mapping[str, Any]:
    streams = probe_data.get("streams")
    if not isinstance(streams, list):
        raise PreprocessError("ffprobe output has no streams array")
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if len(videos) != 1:
        raise PreprocessError(f"Expected exactly one video stream, found {len(videos)}")
    return videos[0]


def _audio_streams(probe_data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    streams = probe_data.get("streams")
    if not isinstance(streams, list):
        raise PreprocessError("ffprobe output has no streams array")
    return [s for s in streams if s.get("codec_type") == "audio"]


def _frame_count(video_stream: Mapping[str, Any]) -> int:
    # count_frames supplies nb_read_frames even when the container omits
    # nb_frames.  Prefer it because the certification count must reflect the
    # stream that ffmpeg can actually read.
    for field in ("nb_read_frames", "nb_frames"):
        value = video_stream.get(field)
        if value not in (None, "", "N/A"):
            try:
                count = int(str(value))
            except ValueError as exc:
                raise PreprocessError(f"Invalid {field}: {value!r}") from exc
            if count <= 0:
                raise PreprocessError(f"Invalid {field}: {value!r}")
            return count
    raise PreprocessError("ffprobe did not report a video frame count")


def probe_media(path: Path, ffprobe: str) -> dict[str, Any]:
    """Return stream/container metadata and a decoded video frame count."""

    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise PreprocessError(f"ffprobe failed for {path}: {detail}") from exc
    except json.JSONDecodeError as exc:
        raise PreprocessError(f"ffprobe returned invalid JSON for {path}") from exc
    if not isinstance(value, dict):
        raise PreprocessError(f"ffprobe returned an invalid document for {path}")
    return value


def validate_input_probe(
    probe_data: Mapping[str, Any], expected_frames: int
) -> dict[str, Any]:
    """Pure validation of the source metadata used by this stage."""

    if isinstance(expected_frames, bool) or not isinstance(expected_frames, int):
        raise PreprocessError("expected_frames must be a positive integer")
    if expected_frames <= 0:
        raise PreprocessError("expected_frames must be a positive integer")
    video = _video_stream(probe_data)
    size = (int(video.get("width", 0)), int(video.get("height", 0)))
    if size != EXPECTED_INPUT_SIZE:
        raise PreprocessError(f"Expected 7680x4320 source, got {size[0]}x{size[1]}")
    fps = _fraction(video.get("avg_frame_rate"), "source avg_frame_rate")
    if fps != EXPECTED_INPUT_FPS:
        raise PreprocessError(f"Expected 16 fps source, got {fps}")
    detected_frames = _frame_count(video)
    if detected_frames != expected_frames:
        raise PreprocessError(
            f"Source frame count {detected_frames} differs from expected "
            f"{expected_frames}"
        )
    return {
        "width": size[0],
        "height": size[1],
        "fps": str(fps),
        "frames": detected_frames,
        "has_audio": bool(_audio_streams(probe_data)),
    }


def validate_output_probe(
    probe_data: Mapping[str, Any], expected_frames: int, expect_audio: bool
) -> dict[str, Any]:
    """Pure validation of an encoded preprocessing result."""

    video = _video_stream(probe_data)
    size = (int(video.get("width", 0)), int(video.get("height", 0)))
    if size != OUTPUT_SIZE:
        raise PreprocessError(f"Expected 1920x1080 output, got {size[0]}x{size[1]}")
    fps = _fraction(video.get("avg_frame_rate"), "output avg_frame_rate")
    if fps != OUTPUT_FPS:
        raise PreprocessError(f"Expected 24 fps output, got {fps}")
    frames = _frame_count(video)
    if frames != expected_frames:
        raise PreprocessError(
            f"Output frame count {frames} differs from expected {expected_frames}"
        )
    if video.get("pix_fmt") != VIDEO_PIXEL_FORMAT:
        raise PreprocessError(
            f"Expected {VIDEO_PIXEL_FORMAT} output, got {video.get('pix_fmt')!r}"
        )
    audio_count = len(_audio_streams(probe_data))
    if expect_audio and audio_count < 1:
        raise PreprocessError("Source has audio but output has no audio stream")
    if not expect_audio and audio_count:
        raise PreprocessError("Source has no audio but output unexpectedly has audio")
    return {
        "width": size[0],
        "height": size[1],
        "fps": str(fps),
        "frames": frames,
        "has_audio": bool(audio_count),
    }


def output_path_for(source_path: Path, output_dir: Path) -> Path:
    """Derive the stable certification filename from an input filename."""

    name = source_path.name
    if name.endswith("_Input_8K.mp4"):
        name = name[: -len("_Input_8K.mp4")] + "_Input_FHD_24fps.mp4"
    else:
        name = source_path.stem + "_Input_FHD_24fps.mp4"
    return output_dir / name


def build_ffmpeg_command(
    source_path: Path,
    temporary_output: Path,
    ffmpeg: str,
    has_audio: bool,
) -> list[str]:
    """Build the sole ffmpeg invocation used by preprocessing."""

    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-xerror",
        "-err_detect",
        "explode",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
    ]
    if has_audio:
        command.extend(["-map", "0:a:0"])
    command.extend(
        [
            "-vf",
            VIDEO_FILTER,
            "-fps_mode:v",
            "passthrough",
            "-c:v",
            VIDEO_CODEC,
            "-preset",
            VIDEO_PRESET,
            "-crf",
            str(VIDEO_CRF),
            "-pix_fmt",
            VIDEO_PIXEL_FORMAT,
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
        ]
    )
    if has_audio:
        command.extend(
            [
                "-af",
                AUDIO_FILTER,
                "-c:a",
                AUDIO_CODEC,
                "-b:a",
                AUDIO_BITRATE,
            ]
        )
    command.extend(
        [
            "-map_metadata",
            "0",
            "-video_track_timescale",
            "24576",
            "-y",
            str(temporary_output),
        ]
    )
    return command


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    """Run a command while mirroring its combined output to terminal and log."""

    with log_path.open("a", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_stream.write(line)
            log_stream.flush()
        return_code = process.wait()
    if return_code:
        raise PreprocessError(f"ffmpeg exited with status {return_code}")


def _alignment_document(
    source_path: Path,
    output_path: Path,
    frame_count: int,
    has_audio: bool,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "0001_Preprocess_FHD24",
        "source": str(source_path),
        "source_identity": dict(source_identity),
        "output": str(output_path),
        "source_frames": frame_count,
        "output_frames": frame_count,
        "source_fps": 16,
        "processing_fps": 24,
        "frame_index_mapping": "source[k] -> output[k]",
        "first_frame_mapping": {"source": 0, "output": 0},
        "last_frame_mapping": {
            "source": frame_count - 1,
            "output": frame_count - 1,
        },
        "frame_order_preserved": True,
        "frame_count_preserved": True,
        "interpolation": False,
        "frame_drop": False,
        "frame_duplication": False,
        "timestamp_operation": "setpts=PTS*2/3 followed by CFR labeling at 24 fps",
        "video_filter": VIDEO_FILTER,
        "audio_present": has_audio,
        "audio_operation": AUDIO_FILTER if has_audio else None,
        "status": "PASS",
    }


def run_preprocess(
    source_path: Path | str,
    output_dir: Path | str,
    metadata_dir: Path | str,
    log_path: Path | str,
    expected_frames: int,
    resume: bool = False,
) -> Path:
    """Create and verify the frame-preserving FHD/24 fps processing input.

    ``expected_frames`` must be the runtime-resolved source frame count.  It is
    checked against an ffprobe decoded-frame count before ffmpeg is invoked.
    With ``resume=True``, an existing output is reused only after the exact same
    media validation succeeds.
    """

    source_path = Path(source_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    metadata_dir = Path(metadata_dir).expanduser().resolve()
    log_path = Path(log_path).expanduser().resolve()
    if not source_path.is_file() or source_path.suffix.lower() != ".mp4":
        raise PreprocessError(f"Input must be an existing MP4 file: {source_path}")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise PreprocessError("ffmpeg and ffprobe must be available on PATH")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(source_path, output_dir)
    temporary_output = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.partial.mp4"
    )
    if output_path == source_path:
        raise PreprocessError("Refusing to overwrite the source video")
    if temporary_output.exists():
        raise PreprocessError(f"Temporary output already exists: {temporary_output}")

    _log(
        log_path,
        f"status=START input={source_path} output={output_path} "
        f"expected_frames={expected_frames} resume={resume}",
    )
    input_probe = probe_media(source_path, ffprobe)
    input_summary = validate_input_probe(input_probe, expected_frames)
    source_identity = _source_identity(source_path)

    if output_path.exists():
        if not resume:
            raise FileExistsError(f"Output already exists: {output_path}")
        alignment_path = metadata_dir / "frame_alignment.json"
        if not alignment_path.is_file():
            # Interrupted between MP4 publication and alignment commit. An
            # uncommitted file cannot be reused, but retain it for audit.
            orphan_dir = metadata_dir / "orphaned_outputs"
            orphan_dir.mkdir(parents=True, exist_ok=True)
            orphan = orphan_dir / (
                f"{output_path.stem}.{dt.datetime.now(dt.timezone.utc).timestamp():.6f}"
                f"{output_path.suffix}"
            )
            os.replace(output_path, orphan)
            _log(log_path, f"status=ORPHAN_ARCHIVED path={orphan}")
        else:
            previous_alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
            if previous_alignment.get("source_identity") != source_identity:
                raise PreprocessError(
                    "Source bytes changed since the preprocessing output was created"
                )
            _atomic_json(metadata_dir / "input_ffprobe.json", input_probe)
            output_probe = probe_media(output_path, ffprobe)
            output_summary = validate_output_probe(
                output_probe, expected_frames, input_summary["has_audio"]
            )
            _atomic_json(metadata_dir / "output_ffprobe.json", output_probe)
            _atomic_json(
                metadata_dir / "frame_alignment.json",
                _alignment_document(
                    source_path,
                    output_path,
                    expected_frames,
                    input_summary["has_audio"],
                    source_identity,
                ),
            )
            _log(
                log_path,
                f"status=RESUME_PASS frames={output_summary['frames']} "
                f"resolution=1920x1080 fps=24 output={output_path}",
            )
            return output_path

    _atomic_json(metadata_dir / "input_ffprobe.json", input_probe)
    command = build_ffmpeg_command(
        source_path,
        temporary_output,
        ffmpeg,
        input_summary["has_audio"],
    )
    _log(log_path, f"command={shlex.join(command)}")
    try:
        _run_logged(command, log_path)
        output_probe = probe_media(temporary_output, ffprobe)
        output_summary = validate_output_probe(
            output_probe, expected_frames, input_summary["has_audio"]
        )
        source_after = source_path.stat()
        if (source_after.st_size, source_after.st_mtime_ns) != (
            source_identity["size_bytes"],
            source_identity["mtime_ns"],
        ):
            raise PreprocessError("Source bytes changed during preprocessing")
        # Only a completely encoded and metadata-verified file is made visible
        # at its final path.
        os.replace(temporary_output, output_path)
        _atomic_json(metadata_dir / "output_ffprobe.json", output_probe)
        _atomic_json(
            metadata_dir / "frame_alignment.json",
            _alignment_document(
                source_path,
                output_path,
                expected_frames,
                input_summary["has_audio"],
                source_identity,
            ),
        )
    except Exception as exc:
        _log(log_path, f"status=FAIL error={exc}")
        raise
    finally:
        temporary_output.unlink(missing_ok=True)

    _log(
        log_path,
        f"status=PASS frames={output_summary['frames']} "
        f"resolution=1920x1080 fps=24 output={output_path}",
    )
    return output_path


__all__ = [
    "PreprocessError",
    "build_ffmpeg_command",
    "output_path_for",
    "probe_media",
    "run_preprocess",
    "validate_input_probe",
    "validate_output_probe",
]
