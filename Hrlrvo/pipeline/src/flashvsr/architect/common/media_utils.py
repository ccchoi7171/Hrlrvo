from __future__ import annotations

import contextlib
import errno
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: str
    fps_float: float
    frame_count: int
    duration: float
    pixel_format: str | None
    video_codec: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    time_base: str | None
    audio_present: bool
    audio_codec: str | None

    @property
    def expected_x4_width(self) -> int:
        return self.width * 4

    @property
    def expected_x4_height(self) -> int:
        return self.height * 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(command: Sequence[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _positive_fraction(*values: str | None) -> Fraction:
    for value in values:
        if not value or value in {"0/0", "N/A"}:
            continue
        try:
            fraction = Fraction(value)
        except (ValueError, ZeroDivisionError):
            continue
        if fraction > 0:
            return fraction
    raise MediaError("ffprobe did not report a positive video frame rate")


def _count_frames(path: Path) -> int:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise MediaError(f"frame counting failed: {result.stderr.strip()}")
    try:
        count = int(result.stdout.strip())
    except ValueError as exc:
        raise MediaError(f"invalid ffprobe frame count: {result.stdout!r}") from exc
    if count <= 0:
        raise MediaError(f"invalid frame count: {count}")
    return count


def probe_media(path: str | os.PathLike[str], *, count_frames: bool = True) -> VideoMetadata:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MediaError(f"input is not a file: {source}")
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(source),
        ]
    )
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed for {source}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe returned invalid JSON") from exc
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream found: {source}")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    fps_fraction = _positive_fraction(video.get("avg_frame_rate"), video.get("r_frame_rate"))
    frame_count_raw = video.get("nb_frames")
    try:
        frame_count = int(frame_count_raw)
    except (TypeError, ValueError):
        frame_count = 0
    # Container frame-count headers are frequently missing or stale.  When the
    # caller requests an authoritative count, always use ffprobe's decoded
    # packet/frame count instead of trusting nb_frames.
    if count_frames:
        frame_count = _count_frames(source)
    # Prefer the video stream duration.  Matroska often omits it, while the
    # container duration may instead reflect a longer copied audio stream.
    duration_raw = video.get("duration")
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError):
        if frame_count:
            duration = frame_count / float(fps_fraction)
        else:
            try:
                duration = float(payload.get("format", {}).get("duration"))
            except (TypeError, ValueError):
                duration = 0.0
    if frame_count <= 0:
        frame_count = max(1, round(duration * float(fps_fraction)))
    return VideoMetadata(
        path=str(source),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=f"{fps_fraction.numerator}/{fps_fraction.denominator}",
        fps_float=float(fps_fraction),
        frame_count=frame_count,
        duration=duration,
        pixel_format=video.get("pix_fmt"),
        video_codec=video.get("codec_name"),
        color_space=video.get("color_space"),
        color_transfer=video.get("color_transfer"),
        color_primaries=video.get("color_primaries"),
        time_base=video.get("time_base"),
        audio_present=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
    )


def metadata_lines(metadata: VideoMetadata) -> list[str]:
    return [
        f"Input path    : {metadata.path}",
        f"Resolution    : {metadata.width}x{metadata.height}",
        f"FPS           : {metadata.fps} ({metadata.fps_float:.9f})",
        f"Frame count   : {metadata.frame_count}",
        f"Duration      : {metadata.duration:.6f} sec",
        f"Pixel format  : {metadata.pixel_format or 'unknown'}",
        f"Audio presence: {'yes' if metadata.audio_present else 'no'}",
    ]


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def atomic_publish_file(
    source_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
) -> Path:
    """Publish a completed file atomically, including across filesystems."""
    source = Path(source_path)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, destination)
        return destination
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, length=16 * 1024 * 1024)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    source.unlink()
    return destination


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_video_region(
    path: str | os.PathLike[str],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    start_frame: int,
    frame_count: int,
    threads: int = 1,
) -> np.ndarray:
    if min(x, y, start_frame) < 0 or min(width, height, frame_count) <= 0:
        raise ValueError("invalid decode region or temporal range")
    end_frame = start_frame + frame_count
    filters = (
        f"trim=start_frame={start_frame}:end_frame={end_frame},"
        f"setpts=PTS-STARTPTS,crop={width}:{height}:{x}:{y}"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(max(1, threads)),
        "-i",
        str(Path(path).resolve()),
        "-map",
        "0:v:0",
        "-vf",
        filters,
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        raise MediaError("failed to open ffmpeg pipes")
    frame_bytes = width * height * 3
    frames: list[np.ndarray] = []
    stderr = ""
    return_code: int | None = None
    try:
        for _ in range(frame_count):
            raw = _read_exact(process.stdout, frame_bytes)
            if len(raw) != frame_bytes:
                break
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy())
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code != 0:
        raise MediaError(f"ffmpeg region decode failed: {stderr.strip()}")
    if len(frames) != frame_count:
        raise MediaError(
            f"decoded {len(frames)} frames, expected {frame_count} "
            f"for [{start_frame}, {end_frame})"
        )
    return np.stack(frames, axis=0)


def _select_expression(indices: Sequence[int]) -> str:
    if not indices:
        raise ValueError("at least one frame index is required")
    return "+".join(f"eq(n\\,{int(index)})" for index in indices)


def decode_selected_frames(
    path: str | os.PathLike[str],
    indices: Sequence[int],
    *,
    output_width: int | None = None,
    output_height: int | None = None,
    threads: int = 1,
) -> Iterator[np.ndarray]:
    metadata = probe_media(path, count_frames=False)
    width = output_width or metadata.width
    height = output_height or metadata.height
    filters = [f"select='{_select_expression(indices)}'"]
    if (width, height) != (metadata.width, metadata.height):
        filters.append(f"scale={width}:{height}:flags=lanczos")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(max(1, threads)),
        "-i",
        str(Path(path).resolve()),
        "-map",
        "0:v:0",
        "-vf",
        ",".join(filters),
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        raise MediaError("failed to open ffmpeg sample decoder")
    frame_bytes = width * height * 3
    yielded = 0
    stderr = ""
    return_code: int | None = None
    try:
        while yielded < len(indices):
            raw = _read_exact(process.stdout, frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise MediaError("ffmpeg returned a partial sampled frame")
            yielded += 1
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code != 0:
        raise MediaError(f"ffmpeg sampled decode failed: {stderr.strip()}")
    if yielded != len(indices):
        raise MediaError(f"sampled {yielded} frames, expected {len(indices)}")


def encoder_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "hevc_nvenc",
            "extension": ".mp4",
            "arguments": ["-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "16", "-b:v", "0", "-pix_fmt", "p010le", "-tag:v", "hvc1"],
        },
        {
            "name": "av1_nvenc",
            "extension": ".mkv",
            "arguments": ["-c:v", "av1_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "16", "-b:v", "0", "-pix_fmt", "p010le"],
        },
        {
            "name": "libx265",
            "extension": ".mp4",
            "arguments": ["-c:v", "libx265", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-x265-params", "log-level=error"],
        },
        {
            "name": "libsvtav1",
            "extension": ".mkv",
            "arguments": ["-c:v", "libsvtav1", "-preset", "8", "-crf", "14", "-pix_fmt", "yuv420p10le"],
        },
        {
            "name": "ffv1",
            "extension": ".mkv",
            "arguments": ["-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1", "-pix_fmt", "yuv444p10le"],
        },
    ]


def select_encoder(
    width: int,
    height: int,
    *,
    threads: int,
    scratch_dir: str | os.PathLike[str],
    preferred: str = "auto",
    timeout: float = 240.0,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if width <= 0 or height <= 0:
        raise ValueError("invalid encoder probe dimensions")
    candidates = [
        {**candidate, "arguments": list(candidate["arguments"])}
        for candidate in encoder_candidates()
    ]
    for candidate in candidates:
        if candidate["name"] == "libx265":
            parameter_index = candidate["arguments"].index("-x265-params") + 1
            worker_count = max(1, min(int(threads), 32))
            candidate["arguments"][parameter_index] = (
                # FFmpeg maps its generic -threads value to x265's
                # frameNumThreads.  The packaged libx265 rejects values above
                # its compile-time X265_MAX_FRAME_THREADS (16 here), even
                # though larger worker pools are valid.  Let x265 choose a
                # supported frame-thread count while retaining the requested
                # bounded worker pool for CPU parallelism.
                f"log-level=error:pools={worker_count}:frame-threads=0"
            )
    if preferred != "auto":
        candidates = [candidate for candidate in candidates if candidate["name"] == preferred]
        if not candidates:
            raise MediaError(f"unknown encoder: {preferred}")
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, str]] = []
    for candidate in candidates:
        output = scratch / f"encoder_probe_{candidate['name']}{candidate['extension']}"
        output.unlink(missing_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r=1",
            "-frames:v",
            "1",
            "-threads",
            str(max(1, min(threads, 32))),
            *candidate["arguments"],
            "-y",
            str(output),
        ]
        try:
            result = _run(command, timeout=timeout)
            reason = result.stderr.strip()[-2000:]
        except subprocess.TimeoutExpired:
            result = None
            reason = f"probe timed out after {timeout:.0f}s"
        passed = bool(result and result.returncode == 0 and output.is_file() and output.stat().st_size > 0)
        attempts.append({"encoder": candidate["name"], "result": "PASS" if passed else "FAIL", "reason": reason})
        output.unlink(missing_ok=True)
        if passed:
            return candidate, attempts
    details = "; ".join(f"{item['encoder']}: {item['reason']}" for item in attempts)
    raise MediaError(f"no encoder could preserve {width}x{height}; {details}")


class RawVideoEncoder:
    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        width: int,
        height: int,
        fps: str,
        frame_count: int,
        encoder: dict[str, Any],
        threads: int,
        source_metadata: VideoMetadata,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.frames_written = 0
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            fps,
            "-i",
            "pipe:0",
            "-an",
            "-frames:v",
            str(frame_count),
            "-threads",
            str(max(1, threads)),
            *encoder["arguments"],
        ]
        color_flags = {
            "-colorspace": source_metadata.color_space,
            "-color_trc": source_metadata.color_transfer,
            "-color_primaries": source_metadata.color_primaries,
        }
        for flag, value in color_flags.items():
            if value and value not in {"unknown", "reserved"}:
                command.extend([flag, value])
        command.extend(["-y", str(self.path)])
        self.command = command
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if self.process.stdin is None or self.process.stderr is None:
            raise MediaError("failed to open ffmpeg encoder pipes")

    def write(self, frame_or_rows: np.ndarray) -> None:
        array = np.ascontiguousarray(frame_or_rows, dtype=np.uint8)
        try:
            self.process.stdin.write(array.tobytes())
        except BrokenPipeError as exc:
            error = self.process.stderr.read().decode("utf-8", errors="replace")
            raise MediaError(f"encoder terminated early: {error.strip()}") from exc

    def frame_complete(self) -> None:
        self.frames_written += 1

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        error = self.process.stderr.read().decode("utf-8", errors="replace")
        return_code = self.process.wait()
        if return_code != 0:
            raise MediaError(f"video encoding failed: {error.strip()}")
        if self.frames_written != self.frame_count:
            raise MediaError(f"encoded {self.frames_written} frames, expected {self.frame_count}")

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.path.unlink(missing_ok=True)


def finalize_video_with_audio(
    video_only_path: str | os.PathLike[str],
    output_base: str | os.PathLike[str],
    *,
    input_path: str | os.PathLike[str],
    input_metadata: VideoMetadata,
    extension: str,
) -> Path:
    video_only = Path(video_only_path)
    base = Path(output_base)

    def container_path(path: Path, selected_extension: str) -> Path:
        # ``Path.with_suffix`` treats every final dot as an extension, so a
        # versioned extensionless base such as ``FlashVSR_v1.1_x4`` is
        # truncated to ``FlashVSR_v1.mp4``. Only replace a real known
        # container suffix; otherwise append the selected extension.
        for known in (".mp4", ".mkv"):
            if path.name.endswith(known):
                return path.with_name(path.name[: -len(known)] + selected_extension)
        return path.with_name(path.name + selected_extension)

    destination = container_path(base, extension)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not input_metadata.audio_present:
        return atomic_publish_file(video_only, destination)
    attempts = [destination]
    if extension != ".mkv":
        attempts.append(container_path(base, ".mkv"))
    errors: list[str] = []
    for candidate in attempts:
        candidate.unlink(missing_ok=True)
        result = _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_only),
                "-i",
                str(Path(input_path).resolve()),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-map_metadata",
                "1",
                "-c",
                "copy",
                "-y",
                str(candidate),
            ]
        )
        if result.returncode == 0 and candidate.is_file():
            video_only.unlink(missing_ok=True)
            return candidate
        errors.append(result.stderr.strip())
        candidate.unlink(missing_ok=True)
    raise MediaError("audio stream copy/mux failed: " + " | ".join(errors))


def uniform_sample_indices(frame_count: int, *, minimum: int = 12, maximum: int = 24) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= minimum:
        return list(range(frame_count))
    sample_count = max(minimum, min(maximum, round(2.0 * math.sqrt(frame_count))))
    return sorted({round(value) for value in np.linspace(0, frame_count - 1, sample_count)})


def executable_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for executable in ("ffmpeg", "ffprobe", "nvidia-smi"):
        path = shutil.which(executable)
        if not path:
            versions[executable] = "missing"
            continue
        result = _run([path, "--version"])
        text = (result.stdout or result.stderr).splitlines()
        versions[executable] = text[0] if text else "unknown"
    return versions
