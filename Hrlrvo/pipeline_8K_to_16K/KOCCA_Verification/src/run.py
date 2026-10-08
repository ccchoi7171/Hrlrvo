"""Run one KOCCA verification video using its frozen JSON settings.

The input may live anywhere. Its basename selects configs/01.json through
configs/05.json. Every fresh invocation gets isolated artifacts; --resume
continues the most recently started invocation for that video.
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import queue
import re
import select
import signal
import shutil
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

from display import StageDisplay, elapsed_text

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
INPUT_NAME = re.compile(r"^(0[1-5])_Input_8K\.mp4$")
RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{6}$")
HEARTBEAT_SECONDS = 30
FRAME_PLAN = {"outpaint": "all", "upscale": "all"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(
    path: Path, on_progress: Callable[[int, int], None] | None = None,
) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    processed = 0
    last_report = time.monotonic()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            processed += len(block)
            now = time.monotonic()
            if on_progress is not None and now - last_report >= HEARTBEAT_SECONDS:
                on_progress(processed, before.st_size)
                last_report = now
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while hashing: {path}")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def probe_input(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError("input has no video stream")
    frames = video.get("nb_frames")
    if frames in (None, "N/A"):
        counted = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)], capture_output=True, text=True, check=True,
        )
        frames = counted.stdout.strip()
    frame_count = int(frames)
    fps = Fraction(video["avg_frame_rate"])
    r_fps = Fraction(video["r_frame_rate"])
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError("input has invalid FPS or frame count")
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    return {
        "resolution": f"{video['width']}x{video['height']}",
        "fps": str(fps.numerator) if fps.denominator == 1 else str(fps),
        "frames": frame_count,
        "duration": f"{frame_count / float(fps):.3f}s",
        "pixel_format": video.get("pix_fmt", "unknown"),
        "audio": (
            f"{audio.get('codec_name')}/{audio.get('sample_rate')}Hz/{audio.get('channels')}ch"
            if audio else "none"
        ),
        "original_metadata": {
            "path": str(path),
            "width": int(video["width"]),
            "height": int(video["height"]),
            "avg_frame_rate": f"{fps.numerator}/{fps.denominator}",
            "r_frame_rate": f"{r_fps.numerator}/{r_fps.denominator}",
            "fps": float(fps),
            "frame_count": frame_count,
            "duration": float(video.get("duration") or payload["format"]["duration"]),
            "pix_fmt": video.get("pix_fmt"),
            "codec_name": video.get("codec_name"),
            "color_space": video.get("color_space"),
            "color_transfer": video.get("color_transfer"),
            "color_primaries": video.get("color_primaries"),
            "time_base": video.get("time_base"),
            "start_time": video.get("start_time"),
            "decoded_frame_count_performed": False,
        },
    }


def sha256_sequence(path: Path, frames: int) -> str:
    """Hash image names and bytes in frame order for reproduction comparison."""
    expected = {f"{index}.jpg" for index in range(frames)}
    if not path.is_dir() or {item.name for item in path.iterdir()} != expected:
        raise RuntimeError(f"final JPG sequence is incomplete: {path}")
    digest = hashlib.sha256()
    for index in range(frames):
        frame = path / f"{index}.jpg"
        if frame.stat().st_size == 0:
            raise RuntimeError(f"empty final JPG frame: {frame}")
        digest.update(frame.name.encode("ascii") + b"\0")
        with frame.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def publish_final(source: Path, destination: Path, frames: int, expected_sha256: str,
                  audit_root: Path) -> None:
    """Atomically publish the numbered sequence, preserving any older output."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"temporary output directory already exists: {temporary}")
    temporary.mkdir()
    prior: Path | None = None
    copied = False
    try:
        for index in range(frames):
            name = f"{index}.jpg"
            try:
                os.link(source / name, temporary / name)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(source / name, temporary / name)
                copied = True
        if copied and sha256_sequence(temporary, frames) != expected_sha256:
            raise RuntimeError("published JPG sequence SHA256 mismatch")
        if destination.exists():
            prior = audit_root / "previous_published_output"
            if prior.exists():
                raise RuntimeError(f"previous output audit path already exists: {prior}")
            os.replace(destination, prior)
        os.replace(temporary, destination)
    except BaseException:
        if prior is not None and prior.exists() and not destination.exists():
            os.replace(prior, destination)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class RunLog:
    def __init__(self, video_id: str) -> None:
        self.path = HERE / "log" / f"{video_id}.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8", buffering=1)
        self.active_progress = False

    def close(self) -> None:
        if self.active_progress:
            sys.stdout.write("\n")
        self.stream.close()

    def event(self, name: str, **fields: Any) -> None:
        details = " ".join(
            f"{key}={self._value(value)}"
            for key, value in fields.items()
        )
        self.emit(f"[{name}]" + (f" {details}" if details else ""))

    @staticmethod
    def _value(value: Any) -> str:
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str) and value and not any(c.isspace() for c in value):
            return value
        return json.dumps(value, ensure_ascii=False)

    def emit(self, line: str) -> None:
        if self.active_progress:
            sys.stdout.write("\n")
            self.active_progress = False
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self.stream.write(f"{utc_now()} {line}\n")
        self.stream.flush()

    def banner(self, label: str) -> None:
        width = max(60, shutil.get_terminal_size((80, 24)).columns)
        title = f"[ {label} ]"
        left = max(4, (width - len(title)) // 2)
        right = max(4, width - len(title) - left)
        self.emit("=" * left + title + "=" * right)

    def progress(
        self, name: str, completed: int, total: int, *, prefix: str = "",
        suffix: str = "", show_percent: bool = True,
    ) -> None:
        if total <= 0 or not 0 <= completed <= total:
            raise ValueError(f"invalid {name} progress: {completed}/{total}")
        bar_width = 30
        filled = min(bar_width, completed * bar_width // total)
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
        percent = f" {completed / total * 100:5.1f}%" if show_percent else ""
        parts = [f"[{name}]", prefix, bar, percent.strip(), suffix]
        line = " ".join(part for part in parts if part)
        if sys.stdout.isatty():
            sys.stdout.write("\r\033[K" + line)
            self.active_progress = True
        else:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self.stream.write(f"{utc_now()} {line}\n")
        self.stream.flush()

def environment_python(name: str, override: str) -> Path:
    selected = os.environ.get(override)
    candidates = [Path(selected).expanduser()] if selected else []
    candidates.append(ROOT.parent.parent / "miniconda3" / "envs" / name / "bin" / "python")
    if name == "Hrlrvo":
        candidates.append(Path(sys.executable))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RuntimeError(f"Python environment {name} not found; set {override}")


def load_settings(
    video_id: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    spec_path = HERE / "configs" / f"{video_id}.json"
    profile_path = HERE / "configs" / "profile.json"
    spec_bytes = spec_path.read_bytes()
    profile_bytes = profile_path.read_bytes()
    spec = json.loads(spec_bytes)
    profile = json.loads(profile_bytes)
    if not all(isinstance(value, dict) for value in (spec, profile)):
        raise TypeError("KOCCA JSON roots must be objects")
    if spec.get("schema_version") != 1 or spec.get("video_id") != video_id:
        raise ValueError(f"KOCCA video specification does not match {video_id}")
    allowed_spec_keys = {"schema_version", "video_id", "input_filename", "input_sha256", "seeds"}
    if set(spec) - allowed_spec_keys:
        raise ValueError("KOCCA video specification contains a video-specific processing option")
    if spec.get("input_filename") != f"{video_id}_Input_8K.mp4":
        raise ValueError("KOCCA input filename does not match video ID")
    if profile.get("schema_version") != 1 or "frames" in profile:
        raise ValueError("KOCCA profile must contain shared model settings only")
    if profile.get("upscale_chunk_frames") != 21:
        raise ValueError("KOCCA Upscale chunk size must be 21 frames")
    seeds = spec.get("seeds")
    if not isinstance(seeds, dict) or set(seeds) != {"outpainting", "upscale"}:
        raise TypeError("KOCCA seeds must contain only outpainting and upscale")
    for stage in ("outpainting", "upscale"):
        value = seeds.get(stage)
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError(f"KOCCA seeds.{stage} must be a 32-bit nonnegative integer")
    return (
        spec, profile,
        hashlib.sha256(spec_bytes).hexdigest(),
        hashlib.sha256(profile_bytes).hexdigest(),
    )


def build_request(
    video_id: str, run_id: str, input_path: Path,
    spec: dict[str, Any], profile: dict[str, Any],
    spec_sha256: str, profile_sha256: str,
) -> dict[str, Any]:
    job = f"{video_id}_kocca_{run_id}"
    base = f"KOCCA_Verification/Runs/{video_id}/{run_id}"
    request = {key: profile[key] for key in (
        "schema_version", "preprocess", "outpainting", "upscale",
        "final", "verification",
    )}
    request = json.loads(json.dumps(request))
    request.update({
        "job_name": job,
        "frames": dict(FRAME_PLAN),
        "upscale_chunk_frames": profile["upscale_chunk_frames"],
        "output_root": f"{base}/state",
        "input": {"video_id": video_id, "path": str(input_path)},
        "seed": spec["seeds"]["outpainting"],
        "upscale_seed": spec["seeds"]["upscale"],
        "kocca_spec_sha256": spec_sha256,
        "kocca_profile_sha256": profile_sha256,
        "artifacts": {
            "preprocess": f"{base}/preprocess/{video_id}_PreProc_2K.mp4",
            "preprocess_metadata": f"{base}/preprocess/metadata",
            "outpaint": f"{base}/outpaint",
            "upscale": f"{base}/upscale",
        },
    })
    request["final"]["filename"] = f"{video_id}_KOCCA_Final_16K_16fps_{{frames}}f"
    request["upscale"]["output_mode"] = "jpeg_sequence"
    return request


def new_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + os.urandom(3).hex()


def status(path: Path, stage: str, **fields: Any) -> None:
    atomic_json(path, {"stage": stage, "updated_utc": utc_now(), **fields})


class StopKey:
    """Read one stop key from the SSH terminal without taking child stdin."""

    def __init__(self) -> None:
        self.fd: int | None = None
        self.original: list[Any] | None = None

    def enable(self) -> bool:
        original: list[Any] | None = None
        try:
            fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            original = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            attributes = termios.tcgetattr(fd)
            attributes[0] &= ~(termios.IXON | termios.IXOFF)
            termios.tcsetattr(fd, termios.TCSANOW, attributes)
        except (OSError, termios.error):
            if "fd" in locals():
                if original is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, original)
                os.close(fd)
            return False
        self.fd = fd
        self.original = original
        return True

    def poll(self) -> bool:
        if self.fd is None or not select.select([self.fd], [], [], 0)[0]:
            return False
        try:
            return b"\x11" in os.read(self.fd, 1024)
        except BlockingIOError:
            return False

    def close(self) -> None:
        if self.fd is not None:
            try:
                if self.original is not None:
                    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)
            finally:
                os.close(self.fd)
                self.fd = None


def run_stage(
    name: str, request_path: Path, *, resume: bool,
    outpaint_python: Path, flash_python: Path, log: RunLog,
    status_path: Path, display: StageDisplay,
) -> None:
    command = [str(outpaint_python), str(HERE / "src" / "stages.py"), "--config", str(request_path), "--stage", name]
    if resume:
        command.append("--resume")
    environment = os.environ.copy()
    environment.update({
        "CERT_FLASH_PYTHON": str(flash_python),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    display.begin(name)
    status(status_path, name.upper(), pid=os.getpid())
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader_thread = threading.Thread(target=reader, name=f"{name}-log-reader", daemon=True)
    reader_thread.start()
    progress: dict[str, Any] = {}
    stop_key = StopKey()
    stop_ready = False
    stop_requested = False
    last_status_write = time.monotonic()
    recent: deque[str] = deque(maxlen=12)
    try:
        while True:
            can_stop = (
                (name == "outpaint" and 0 < display.outpaint_completed < display.source_frames)
                or (name == "upscale" and 0 < display.upscale_completed < display.outpaint_frames)
            )
            pressed = stop_key.poll() if stop_ready and not stop_requested else False
            if pressed and can_stop:
                stop_requested = True
                completed = (display.outpaint_completed if name == "outpaint"
                             else display.upscale_completed)
                atomic_json(display.run_dir / "state" / f"{name}_stop_request.json", {
                    "key": "ctrl+q", "requested_utc": utc_now(),
                    "completed_frames_observed": completed,
                })
                event = "STOP_REQUESTED" if name == "outpaint" else "UPSCALE_STOP_REQUESTED"
                log.event(event, key="ctrl+q", completed_frames=completed)
                status(status_path, f"{name.upper()}_STOP_REQUESTED", pid=os.getpid(), progress=progress)
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGUSR1)
                    except ProcessLookupError:
                        pass
            try:
                line = lines.get(timeout=0.2 if stop_ready and not stop_requested else HEARTBEAT_SECONDS)
            except queue.Empty:
                if time.monotonic() - last_status_write >= HEARTBEAT_SECONDS:
                    elapsed = int(time.monotonic() - started)
                    status(status_path, name.upper(), pid=os.getpid(), elapsed_seconds=elapsed, progress=progress)
                    last_status_write = time.monotonic()
                continue
            if line is None:
                break
            recent.append(line.rstrip("\n"))
            if name == "outpaint" and "[STOP_CONTROL_READY]" in line and not stop_ready:
                stop_ready = stop_key.enable()
                log.event("STOP_CONTROL", key="ctrl+q", available=stop_ready)
            if name == "upscale" and "[UPSCALE_STOP_CONTROL_READY]" in line and not stop_ready:
                stop_ready = stop_key.enable()
                log.event("UPSCALE_STOP_CONTROL", key="ctrl+q", available=stop_ready)
            updated = display.consume(name, line)
            if updated is not None:
                progress = updated
                now = time.monotonic()
                if now - last_status_write >= 1:
                    status(status_path, name.upper(), pid=os.getpid(), progress=progress)
                    last_status_write = now
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        raise
    finally:
        stop_key.close()
    reader_thread.join()
    return_code = process.wait()
    if return_code:
        for detail in recent:
            log.event("STAGE_ERROR_DETAIL", text=detail[:350])
        log.event("STAGE_RAW_LOG", path=display.run_dir / "logs" / "raw" / f"{display.request['job_name']}.log")
        raise RuntimeError(f"{name} stage exited with status {return_code}")
    status(status_path, name.upper(), pid=os.getpid(), progress=progress)
    stage_name = {"preprocess": "PreProcess", "outpaint": "VideoOutPainting", "upscale": "Upscale"}[name]
    if name == "preprocess":
        output = (ROOT / display.request["artifacts"]["preprocess"]).resolve()
    elif name == "outpaint":
        output = display.run_dir / "outpaint" / "final" / (
            f"{display.request['input']['video_id']}_Outpaint_4K.mp4"
        )
    else:
        upscale_receipt = read_json(
            display.run_dir / "upscale" / "metadata" / "stage_complete.json"
        )
        output = Path(upscale_receipt["output"])
    if not (output.is_dir() if name == "upscale" else output.is_file()):
        raise RuntimeError(f"{stage_name} success did not publish {output}")
    if name == "preprocess":
        log.event("PREPROCESS_DONE", resolution="1920x1080",
                  frames=display.source_frames, status="PASS")
    elif name == "outpaint":
        receipt = read_json(display.run_dir / "outpaint" / "metadata" / "stage_complete.json")
        display.outpaint_frames = int(receipt["frames"])
        display.upscale_frames = display.outpaint_frames
        log.event("OUTPAINT_DONE", resolution="3840x2160",
                  frames=display.outpaint_frames, chunks=receipt["chunks"], status="PASS")
    else:
        log.event("UPSCALE_DONE", resolution="15360x8640",
                  frames=display.upscale_frames, status="PASS")
    log.event("STAGE_COMPLETE", stage=stage_name,
              elapsed=elapsed_text(time.monotonic() - started))


def run(input_arg: str, resume: bool) -> int:
    match = INPUT_NAME.fullmatch(Path(input_arg).name)
    if match is None:
        raise ValueError("input filename must be 01_Input_8K.mp4 through 05_Input_8K.mp4")
    video_id = match[1]
    log = RunLog(video_id)
    lock_path = HERE / "log" / f"{video_id}.lock"
    try:
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                log.event("RUN_FAILED", video=video_id, reason="another KOCCA run is active")
                raise RuntimeError(f"another KOCCA run is active for {video_id}") from error
            return _run_locked(input_arg, video_id, resume, log)
    finally:
        log.close()


def _run_locked(input_arg: str, video_id: str, resume: bool, log: RunLog) -> int:
    started = time.monotonic()
    run_id = "unassigned"
    status_path: Path | None = None
    try:
        input_path = Path(input_arg).expanduser().resolve(strict=True)
        if not input_path.is_file():
            raise ValueError(f"input is not a file: {input_path}")
        run_fields: dict[str, Any] = {"video": video_id, "input": input_path}
        if resume:
            run_fields["resume"] = True
        log.event("RUN_START", **run_fields)
        log.event("LOG_FILE", path=log.path)
        spec, profile, spec_sha256, profile_sha256 = load_settings(video_id)
        input_details = probe_input(input_path)
        if input_details["resolution"] != "7680x4320" or input_details["fps"] != "16":
            raise ValueError("KOCCA input must be 7680x4320 at 16 fps")
        input_sha256 = sha256_file(input_path)
        log.event("INPUT_HASH", sha256=input_sha256)
        expected_input_sha256 = spec.get("input_sha256")
        if expected_input_sha256 is not None and input_sha256 != expected_input_sha256:
            raise ValueError("input SHA256 differs from the video specification")
        source_frames = input_details["frames"]
        for stage, checkpoint in (
            ("VideoOutPainting", ROOT / "src/models/outpaint_checkpoint_007500/adapter_model.safetensors"),
            ("Upscale", ROOT / "src/models/flashvsr_v1_1/checkpoint/diffusion_pytorch_model_streaming_dmd.safetensors"),
        ):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        video_root = HERE / "Runs" / video_id
        video_root.mkdir(parents=True, exist_ok=True)
        binding_path = video_root / "state" / "registered_input.json"
        if binding_path.is_file() and read_json(binding_path).get("sha256") != input_sha256:
            raise ValueError("input SHA256 differs from this video's registered successful run")
        latest_path = video_root / "latest_run.json"
        if resume:
            latest = read_json(latest_path)
            run_id = latest.get("run_id", "")
            if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
                raise ValueError("latest KOCCA run ID is invalid")
        else:
            run_id = new_run_id()
        run_dir = video_root / run_id
        request_path = run_dir / "request.json"
        status_path = run_dir / "status.json"
        request = build_request(
            video_id, run_id, input_path, spec, profile,
            spec_sha256, profile_sha256,
        )
        if resume:
            if read_json(request_path) != request:
                raise ValueError("resume request changed: input path, seeds, profile, or specification")
            identity = read_json(run_dir / "input_identity.json")
            if identity.get("sha256") != input_sha256:
                raise ValueError("resume input SHA256 changed")
        else:
            if run_dir.exists():
                raise FileExistsError(f"run directory already exists: {run_dir}")
            run_dir.mkdir(parents=True)
            atomic_json(request_path, request)
            atomic_json(run_dir / "input_identity.json", {
                "path": str(input_path), "sha256": input_sha256,
                "size_bytes": input_path.stat().st_size,
                "original_metadata": input_details["original_metadata"],
            })
            atomic_json(latest_path, {"run_id": run_id, "request": str(request_path)})
        status(status_path, "PREFLIGHT", run_id=run_id, input_sha256=input_sha256)
        outpaint_python = environment_python("Hrlrvo", "CERT_OUTPAINT_PYTHON")
        flash_python = environment_python("Hrlrvo-FlashVSR-py31115", "CERT_FLASH_PYTHON")
        log.event("PREFLIGHT_DONE", status="PASS")
        display = StageDisplay(log, request, run_dir, source_frames)
        for stage in ("preprocess", "outpaint", "upscale"):
            run_stage(
                stage, request_path, resume=resume, outpaint_python=outpaint_python,
                flash_python=flash_python, log=log, status_path=status_path,
                display=display,
            )
        outpaint_path = run_dir / "outpaint" / "final" / f"{video_id}_Outpaint_4K.mp4"
        outpaint_receipt = read_json(run_dir / "outpaint" / "metadata" / "stage_complete.json")
        upscale_receipt_path = run_dir / "upscale" / "metadata" / "stage_complete.json"
        upscale_receipt = read_json(upscale_receipt_path)
        final_frames = int(upscale_receipt["frames"])
        final_path = Path(upscale_receipt["output"])
        manifest = read_json(run_dir / "upscale" / "metadata" / "final_manifest.json")
        if (manifest.get("status") != "PASS" or not final_path.is_dir()
                or manifest.get("encoding", {}).get("frames_written") != final_frames):
            raise RuntimeError("final KOCCA output or PASS manifest is missing")
        resolved = read_json(run_dir / "state" / "resolved_config.json")
        frame_counts = resolved.get("frames")
        if not isinstance(frame_counts, dict) or not all(
            type(frame_counts.get(name)) is int and frame_counts[name] > 0
            for name in ("source", "outpaint", "upscale", "final")
        ):
            raise RuntimeError("resolved frame counts are missing from the stage receipt")
        if (frame_counts["outpaint"] != outpaint_receipt["frames"]
                or frame_counts["upscale"] != final_frames
                or frame_counts["final"] != final_frames
                or final_frames > outpaint_receipt["frames"]):
            raise RuntimeError("resolved frame counts do not match accepted chunks")
        outpaint_sha256 = sha256_file(outpaint_path)
        final_sha256 = sha256_sequence(final_path, final_frames)
        log.event("FINAL_HASH", sha256=final_sha256)
        published_path = HERE / "Output" / f"{video_id}_16K"
        receipt = {
            "status": "PASS", "video_id": video_id, "run_id": run_id,
            "input": str(input_path), "input_sha256": input_sha256,
            "spec_sha256": spec_sha256, "profile_sha256": profile_sha256,
            "frames": frame_counts,
            "outpaint_chunks": outpaint_receipt["chunks"],
            "outpaint_stop_requested": outpaint_receipt["stop_requested"],
            "upscale_chunks": upscale_receipt["chunks"],
            "upscale_stop_requested": upscale_receipt["stop_requested"],
            "outpainting_seed": spec["seeds"]["outpainting"],
            "upscale_seed": spec["seeds"]["upscale"],
            "outpaint_output": str(outpaint_path), "outpaint_sha256": outpaint_sha256,
            "final_output": str(final_path), "final_sha256": final_sha256,
            "published_output": str(published_path),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "verification": manifest.get("verification_policy"),
        }
        previous_path = video_root / "latest_success.json"
        verdict = "FIRST_RUN"
        if previous_path.is_file():
            previous = read_json(previous_path)
            comparable = all(previous.get(key) == receipt[key] for key in (
                "input_sha256", "spec_sha256", "profile_sha256",
                "outpainting_seed", "upscale_seed",
                "outpaint_chunks", "outpaint_stop_requested",
                "upscale_chunks", "upscale_stop_requested", "frames",
            ))
            if comparable:
                verdict = "MATCH" if (
                    previous.get("outpaint_sha256") == outpaint_sha256
                    and previous.get("final_sha256") == final_sha256
                ) else "DIFFERENT"
            else:
                verdict = "NOT_COMPARABLE"
        log.event("REPRO_CHECK", verdict=verdict)
        receipt["reproduction_comparison"] = verdict
        if verdict == "DIFFERENT":
            receipt["status"] = "REPRO_MISMATCH"
            if final_path == published_path:
                rejected = run_dir / "repro_mismatch_output"
                if rejected.exists():
                    raise RuntimeError(f"rejected output audit path already exists: {rejected}")
                os.replace(published_path, rejected)
                previous_output = run_dir / "previous_published_output"
                if previous_output.is_dir():
                    os.replace(previous_output, published_path)
                receipt["final_output"] = str(rejected)
            atomic_json(run_dir / "receipt.json", receipt)
            raise RuntimeError("same input and JSON produced different Outpainting or final JPG bytes")
        if final_path != published_path:
            # Existing runs may still use the previous isolated output layout.
            publish_final(final_path, published_path, final_frames, final_sha256, run_dir)
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 1)
        atomic_json(run_dir / "receipt.json", receipt)
        if not binding_path.exists():
            atomic_json(binding_path, {"sha256": input_sha256, "filename": input_path.name})
        atomic_json(previous_path, receipt)
        status(status_path, "DONE", run_id=run_id, frames=frame_counts,
               output=str(published_path), final_sha256=final_sha256)
        log.event("SAVED", path=published_path)
        complete_fields: dict[str, Any] = {
            "video": video_id, "frames": frame_counts["final"],
            "chunks": outpaint_receipt["chunks"], "status": "PASS",
            "elapsed": elapsed_text(time.monotonic() - started),
        }
        log.event("ALL_COMPLETE", **complete_fields)
        return 0
    except BaseException as error:
        if status_path is not None and status_path.parent.is_dir():
            status(status_path, "FAILED", run_id=run_id, error=f"{type(error).__name__}: {error}")
        failure_fields: dict[str, Any] = {"video": video_id}
        failure_fields.update(error=f"{type(error).__name__}: {error}",
                              elapsed=elapsed_text(time.monotonic() - started))
        log.event("RUN_FAILED", **failure_fields)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KOCCA per-video 8K to 16K runner")
    parser.add_argument("video", nargs="?", help="input video path; basename must be NN_Input_8K.mp4")
    parser.add_argument("--input", dest="input_option", help="same as positional video path")
    parser.add_argument("--resume", action="store_true", help="resume the latest run for this video")
    args = parser.parse_args(argv)
    if (args.video is None) == (args.input_option is None):
        parser.error("provide exactly one input video path")
    try:
        return run(args.video or args.input_option, args.resume)
    except (OSError, TypeError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        print(f"[KOCCA_FAIL] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
