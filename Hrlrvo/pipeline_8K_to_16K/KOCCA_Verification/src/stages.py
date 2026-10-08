"""KOCCA-only stage runner for preprocess, VideoOutPainting, and FlashVSR.

All model computation is delegated to the packaged reference implementations.
Nothing in this module executes at import time except defining the runner.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib
import json
import os
import random
import re
import signal
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

KOCCA_ROOT = Path(__file__).resolve().parent.parent
ROOT = KOCCA_ROOT.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

preprocess = importlib.import_module("cert_pipeline.preprocess")
flash_composite = importlib.import_module("cert_pipeline.flash_composite")

NEGATIVE_PROMPT = "people, person, human, staff, audience, crowd, backstage, seats"
FRAME_PLAN = {"outpaint": "all", "upscale": "all"}
FLASH_ASSETS = (
    "repository/diffsynth/pipelines/flashvsr_tiny_long.py",
    "repository/examples/WanVSR/utils/utils.py",
    "repository/examples/WanVSR/utils/TCDecoder.py",
    "repository/examples/WanVSR/prompt_tensor/posi_prompt.pth",
    "checkpoint/diffusion_pytorch_model_streaming_dmd.safetensors",
    "checkpoint/LQ_proj_in.ckpt",
    "checkpoint/TCDecoder.ckpt",
)


class ConfigError(ValueError):
    pass


class OutpaintStop(BaseException):
    """Raised only by the KOCCA OutPainting stop-key signal handler."""


class UpscaleStop(BaseException):
    """Raised only by the KOCCA Upscale stop-key signal handler."""


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def stage_log(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("a", encoding="utf-8") as stream,
        contextlib.redirect_stdout(Tee(sys.stdout, stream)),
        contextlib.redirect_stderr(Tee(sys.stderr, stream)),
    ):
        yield


def log(event: str, **fields: Any) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    print(
        f"{timestamp} [{event}] "
        + " ".join(f"{key}={value}" for key, value in fields.items()),
        flush=True,
    )


def absolute_package_path(value: Any, *, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ConfigError(f"package path must be nonempty and relative: {value!r}")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ConfigError(f"package path escapes package root: {value!r}")
    if must_exist and not path.exists():
        raise ConfigError(f"required package path does not exist: {path}")
    return path


def input_video_path(config: dict[str, Any]) -> Path:
    """KOCCA accepts an absolute input path at any location."""
    value = config["input"]["path"]
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ConfigError("KOCCA input.path must be an absolute file path")
    return Path(value).resolve(strict=True)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ConfigError(f"{label}: got {actual!r}, expected {expected!r}")


def resolve_limit(value: Any, available: int, name: str) -> int:
    resolved = available if value == "all" else value
    if type(resolved) is not int or resolved <= 0:
        raise ConfigError(f"frames.{name} must resolve to a positive integer")
    if resolved > available:
        raise ConfigError(f"frames.{name}={resolved} exceeds available {available}")
    return resolved


def validate_config(path: Path) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON: {path}: {error}") from error
    if not isinstance(config, dict):
        raise ConfigError("config root must be an object")
    require_equal(config.get("schema_version"), 1, "schema_version")
    source = config.get("input")
    video_id = source.get("video_id") if isinstance(source, dict) else None
    if not isinstance(video_id, str) or re.fullmatch(r"0[1-5]", video_id) is None:
        raise ConfigError(f"unsupported video_id: {video_id!r}")
    if "mode" in config or "kocca_mode_sha256" in config:
        raise ConfigError("KOCCA runs use one Ctrl+Q-controlled execution path")
    job = config.get("job_name")
    if not isinstance(job, str) or re.fullmatch(
        rf"{video_id}_kocca_[0-9]{{8}}T[0-9]{{6}}Z_[0-9a-f]{{6}}", job
    ) is None:
        raise ConfigError("invalid KOCCA job_name")
    spec_path = KOCCA_ROOT / "configs" / f"{video_id}.json"
    try:
        spec_bytes = spec_path.read_bytes()
        spec = json.loads(spec_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"invalid KOCCA video specification: {spec_path}") from error
    require_equal(spec.get("schema_version"), 1, "KOCCA schema_version")
    require_equal(spec.get("video_id"), video_id, "KOCCA video_id")
    require_equal(spec.get("input_filename"), f"{video_id}_Input_8K.mp4", "KOCCA input_filename")
    allowed_spec_keys = {"schema_version", "video_id", "input_filename", "input_sha256", "seeds"}
    if set(spec) - allowed_spec_keys:
        raise ConfigError("KOCCA video specification contains a video-specific processing option")
    seeds = spec.get("seeds", {})
    if not isinstance(seeds, dict) or set(seeds) != {"outpainting", "upscale"}:
        raise ConfigError("KOCCA seeds must contain only outpainting and upscale")
    for name in ("outpainting", "upscale"):
        value = seeds.get(name)
        if type(value) is not int or not 0 <= value < 2**32:
            raise ConfigError(f"KOCCA seeds.{name} must be a 32-bit nonnegative integer")
    require_equal(config.get("seed"), seeds.get("outpainting"), "KOCCA outpainting seed")
    require_equal(config.get("upscale_seed"), seeds.get("upscale"), "KOCCA upscale seed")
    import hashlib

    require_equal(
        config.get("kocca_spec_sha256"), hashlib.sha256(spec_bytes).hexdigest(),
        "KOCCA specification SHA256",
    )
    profile_path = KOCCA_ROOT / "configs" / "profile.json"
    try:
        profile_bytes = profile_path.read_bytes()
        profile = json.loads(profile_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"invalid KOCCA profile: {profile_path}") from error
    require_equal(
        config.get("kocca_profile_sha256"), hashlib.sha256(profile_bytes).hexdigest(),
        "KOCCA profile SHA256",
    )
    if not isinstance(profile, dict):
        raise ConfigError("KOCCA profile must be an object")
    require_equal(profile.get("schema_version"), 1, "KOCCA profile schema_version")
    if "frames" in profile:
        raise ConfigError("KOCCA profile must contain shared model settings only")
    require_equal(
        config.get("upscale_chunk_frames"),
        21,
        "KOCCA Upscale chunk frames",
    )
    require_equal(
        profile.get("upscale_chunk_frames"), 21,
        "KOCCA shared Upscale chunk frames",
    )
    require_equal(config.get("frames"), FRAME_PLAN, "KOCCA frames")
    for name in ("preprocess", "outpainting", "upscale", "verification"):
        require_equal(config.get(name), profile.get(name), f"KOCCA {name}")
    expected_final_name = f"{video_id}_KOCCA_Final_16K_16fps_{{frames}}f"
    require_equal(
        config.get("final"),
        {**profile["final"], "filename": expected_final_name},
        "KOCCA final",
    )
    source_path = input_video_path(config)
    require_equal(source_path.name, f"{video_id}_Input_8K.mp4", "KOCCA input filename")
    if not source_path.is_file():
        raise ConfigError(f"input is not a file: {source_path}")
    pre = config.get("preprocess", {})
    require_equal(pre, preprocess.PREPROCESS_CONTRACT, "preprocess")
    out = config.get("outpainting", {})
    from cert_pipeline.recurrence import validate_recurrence

    validate_recurrence(out.get("recurrence"))
    require_equal(
        out.get("runtime_config"),
        "src/video_outpainting/main/configs/inference_4k.yaml",
        "outpainting.runtime_config",
    )
    require_equal(
        out.get("checkpoint"),
        "src/models/outpaint_checkpoint_007500",
        "outpainting.checkpoint",
    )
    for key, expected in {
        "steps": 30,
        "positive_prompt": "",
        "negative_prompt": NEGATIVE_PROMPT,
        "input_resolution": [1920, 1080],
        "output_resolution": [3840, 2160],
        "alpha_feather_px": 16,
        "processing_fps": 24,
        "stored_fps": 16,
        "memory_optimization": {
            "video_only_gpu_weights": True,
            "vae_decode_streaming": True,
            "ffn_token_chunking": False,
        },
    }.items():
        require_equal(out.get(key), expected, f"outpainting.{key}")
    runtime_config = absolute_package_path(out.get("runtime_config"))
    checkpoint = absolute_package_path(out.get("checkpoint"))
    if (
        not runtime_config.is_file()
        or not (checkpoint / "adapter_model.safetensors").is_file()
    ):
        raise ConfigError(
            "outpainting runtime config or checkpoint-007500 adapter is missing"
        )
    up = config.get("upscale", {})
    require_equal(up.get("model"), "src/models/flashvsr_v1_1", "upscale.model")
    require_equal(
        up.get("runtime"),
        "src/flashvsr/architect/FlashVSR_v1.1/upscale.py",
        "upscale.runtime",
    )
    for key, expected in {
        "scale": 4,
        "input_resolution": [3840, 2160],
        "output_resolution": [15360, 8640],
        "fps": 16,
    }.items():
        require_equal(up.get(key), expected, f"upscale.{key}")
    model_root = absolute_package_path(up.get("model"))
    flash_runtime = absolute_package_path(up.get("runtime"))
    if not flash_runtime.is_file():
        raise ConfigError(f"FlashVSR runtime is not a file: {flash_runtime}")
    for member in FLASH_ASSETS:
        if not (model_root / member).is_file():
            raise ConfigError(f"FlashVSR asset missing: {model_root / member}")
    for key in ("min_tile_size", "local_range", "gpu_index", "ffmpeg_threads"):
        if type(up.get(key)) is not int:
            raise ConfigError(f"upscale.{key} must be integer")
    if up["min_tile_size"] <= 0 or up["ffmpeg_threads"] <= 0 or up["gpu_index"] < 0:
        raise ConfigError("invalid FlashVSR integer option")
    if (
        up.get("tile_size") != 320
        or up.get("tile_overlap") != 32
        or up.get("min_tile_size") != 320
    ):
        raise ConfigError("KOCCA FlashVSR tile settings must be fixed at 320px with 32px overlap")
    if up.get("sparse_ratio") not in (1.5, 2.0) or up["local_range"] not in (9, 11):
        raise ConfigError("unsupported FlashVSR temporal options")
    if up.get("encoder") not in ("libx265", "hevc_nvenc"):
        raise ConfigError("unsupported FlashVSR reference encoder setting")
    require_equal(up.get("output_mode"), "jpeg_sequence", "upscale.output_mode")
    final = config.get("final", {})
    for key, expected in {
        "fps": 16,
        "center_original_8k": True,
        "center_feather_px": 64,
        "audio_source": "none",
    }.items():
        require_equal(final.get(key), expected, f"final.{key}")
    if (
        not isinstance(final.get("filename"), str)
        or Path(final["filename"]).name != final["filename"]
    ):
        raise ConfigError("final.filename must be a plain directory name")
    output_root = absolute_package_path(config.get("output_root"), must_exist=False)
    run_id = job.removeprefix(f"{video_id}_kocca_")
    expected_run_root = KOCCA_ROOT / "Runs" / video_id / run_id / "state"
    require_equal(output_root, expected_run_root, "KOCCA output_root")
    if "validation" in config:
        raise ConfigError("KOCCA does not support validation-only jobs")
    artifacts = config.get("artifacts", {})
    for name in ("preprocess", "preprocess_metadata", "outpaint", "upscale"):
        target = absolute_package_path(artifacts.get(name), must_exist=False)
        allowed_root = KOCCA_ROOT / "Runs" / video_id / run_id
        if not target.is_relative_to(allowed_root):
            raise ConfigError(f"artifacts.{name} must be beneath {allowed_root}")
    if not artifacts["preprocess"].endswith(".mp4"):
        raise ConfigError("artifacts.preprocess must be MP4")
    validate_artifact_layout(config)
    verification = config.get("verification", {})
    if verification and type(verification.get("full_decode", False)) is not bool:
        raise ConfigError("verification.full_decode must be a boolean")
    # Inspect actual YAML settings without importing torch or loading a model.
    import yaml

    settings = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ConfigError("outpainting YAML is not an object")
    for key in ("base_model_path", "text_encoder_path", "adapter_path"):
        absolute_package_path(settings["model"][key])
    require_equal(
        settings["model"]["base_model_path"],
        "src/models/ltx_2_3/ltx-2.3-22b-dev.safetensors",
        "native LTX DEV weights",
    )
    require_equal(
        settings["model"]["text_encoder_path"],
        "src/models/gemma_3_12b",
        "native Gemma encoder",
    )
    require_equal(
        settings["model"]["adapter_path"],
        "src/models/outpaint_checkpoint_007500/adapter_model.safetensors",
        "native checkpoint-007500",
    )
    absolute_package_path(settings["ltx_repo"])
    require_equal(
        settings["video"]["boundary_blend_width_pixels"],
        out["alpha_feather_px"],
        "native outpainting feather",
    )
    require_equal(settings["video"]["fps"], 24.0, "native outpainting processing fps")
    return config, source_path, output_root, runtime_config, model_root


def preflight_source(source: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not shutil.which("ffmpeg"):
        raise ConfigError("ffmpeg and ffprobe must be on PATH")
    # Historical Stage0001 uses source nb_frames; the encoded result is fully decoded.
    probe = preprocess.probe_media(source, ffprobe, count_frames=False)
    video = preprocess._video_stream(probe)
    count = preprocess._frame_count(video)
    preprocess.validate_input_probe(probe, count)
    return count


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", dir=destination.parent, delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)


def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    import torch

    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log(
        "SEED",
        seed=seed,
        python="global",
        numpy="global",
        pytorch="global",
        flashvsr="per_tile_generator_plus_global_rng",
    )


def verify_video(
    path: Path, *, width: int, height: int, fps: int, frames: int
) -> dict[str, Any]:
    probe = preprocess.probe_media(path, shutil.which("ffprobe") or "ffprobe")
    video = preprocess._video_stream(probe)
    count = preprocess._frame_count(video)
    if (int(video["width"]), int(video["height"])) != (width, height):
        raise RuntimeError(f"{path}: resolution mismatch")
    if preprocess._fraction(video.get("avg_frame_rate"), "avg_frame_rate") != fps:
        raise RuntimeError(f"{path}: fps mismatch")
    if count != frames:
        raise RuntimeError(f"{path}: {count} decoded frames, expected {frames}")
    return probe


def verify_video_headers(
    path: Path, *, width: int, height: int, fps: int, frames: int
) -> dict[str, Any]:
    """Bounded final-only check; this does not claim a full decoded-frame check."""
    metadata = flash_composite.probe_video_headers(path)
    flash_composite.validate_video_spec(
        metadata,
        label="final-only 16K output",
        resolution=(width, height),
        fps=Fraction(fps, 1),
        frame_count=frames,
    )
    return metadata


def verify_jpeg_sequence(
    path: Path, frames: int, *, start_frame: int = 0, exact: bool = True
) -> None:
    """Check a numbered range without decoding the 16K images."""
    if not path.is_dir() or frames <= 0:
        raise RuntimeError(f"JPG sequence directory is missing: {path}")
    names = {item.name for item in path.iterdir()}
    expected = {f"{index}.jpg" for index in range(start_frame, start_frame + frames)}
    names_match = names == expected if exact else expected.issubset(names)
    if not names_match or any((path / name).stat().st_size == 0 for name in expected):
        raise RuntimeError(f"JPG sequence is incomplete: {path}")


def validated_probe_summary(
    probe: dict[str, Any],
    *,
    width: int,
    height: int,
    fps: int,
    frames: int,
    label: str,
) -> dict[str, Any]:
    """Validate a persisted ffprobe document produced by a completed stage."""
    video = preprocess._video_stream(probe)
    actual_size = (int(video["width"]), int(video["height"]))
    if actual_size != (width, height):
        raise RuntimeError(f"{label}: resolution mismatch: {actual_size}")
    actual_fps = preprocess._fraction(video.get("avg_frame_rate"), "avg_frame_rate")
    if actual_fps != fps:
        raise RuntimeError(f"{label}: fps mismatch: {actual_fps}")
    actual_frames = preprocess._frame_count(video)
    if actual_frames != frames:
        raise RuntimeError(
            f"{label}: frame-count mismatch: {actual_frames} != {frames}"
        )
    return {
        "resolution": f"{width}x{height}",
        "fps": fps,
        "frames": actual_frames,
        "pixel_format": video.get("pix_fmt", "unknown"),
    }


def run_logged(command: list[str], *, cwd: Path | None = None) -> None:
    log("COMMAND", argv=" ".join(command))
    process = subprocess.Popen(
        command,
        cwd=cwd,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        status = process.wait()
    except BaseException:
        if process.poll() is None:
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
    if status:
        raise RuntimeError(f"command exited with status {status}: {command[0]}")


def _run_flash_native(
    config: dict[str, Any],
    source_8k: Path,
    outpaint_4k: Path,
    frame_count: int,
    runtime_config: Path,
    model_root: Path,
    *,
    resume: bool,
    stage_override: Path | None = None,
    start_frame: int = 0,
    selected_range: bool = False,
) -> Path:
    stage = stage_override or artifact_path(config, "upscale")
    video_id = config["input"]["video_id"]
    up = config["upscale"]
    selected = outpaint_4k
    if selected_range:
        selected_probe = preprocess.probe_media(
            selected, shutil.which("ffprobe") or "ffprobe", count_frames=False
        )
        selected_video = preprocess._video_stream(selected_probe)
        if (
            int(selected_video["width"]) != 3840
            or int(selected_video["height"]) != 2160
            or preprocess._fraction(selected_video["avg_frame_rate"], "avg_frame_rate") != 16
            or start_frame < 0
            or start_frame + frame_count > preprocess._frame_count(selected_video)
        ):
            raise ConfigError("Upscale chunk lies outside the accepted 4K/16fps output")
    else:
        selected_probe = verify_video(
            selected, width=3840, height=2160, fps=16, frames=frame_count
        )
    selected_video = preprocess._video_stream(selected_probe)
    log(
        "UPSCALE_INPUT_READY",
        path=selected,
        resolution="3840x2160",
        fps=16,
        frames=frame_count,
        duration=selected_probe.get("format", {}).get("duration", "unknown"),
        pixel_format=selected_video.get("pix_fmt", "unknown"),
        status="PASS",
    )
    raw = stage / "raw_flashvsr" / f"{video_id}_FlashVSR_raw_16K_16fps.mp4"
    final = (
        KOCCA_ROOT / "Output" / f"{video_id}_16K"
        if selected_range else stage / "final" / "frames"
    )
    manifest = stage / "metadata" / "final_manifest.json"
    input_identity_path = artifact_path(config, "preprocess").parent.parent / "input_identity.json"
    input_identity = json.loads(input_identity_path.read_text(encoding="utf-8"))
    original_sha256 = input_identity.get("sha256")
    if (
        input_identity.get("path") != str(source_8k)
        or not isinstance(original_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", original_sha256) is None
    ):
        raise ConfigError("KOCCA original input identity is missing or invalid")
    wrapper = absolute_package_path("src/cert_pipeline/flash_composite.py")
    flash_runtime = absolute_package_path(up["runtime"])
    flash_python = os.environ.get("CERT_FLASH_PYTHON") or str(
        ROOT.parent.parent / "miniconda3" / "envs"
        / "Hrlrvo-FlashVSR-py31115" / "bin" / "python"
    )
    if not Path(flash_python).exists() and not shutil.which(flash_python):
        raise ConfigError(f"FlashVSR Python executable not found: {flash_python}")
    command = [
        flash_python,
        str(wrapper),
        "--video-id",
        video_id,
        "--input-4k",
        str(selected),
        "--original-8k",
        str(source_8k),
        "--preverified-original-sha256",
        original_sha256,
        "--model-root",
        str(model_root),
        "--flashvsr-reference",
        str(flash_runtime),
        "--outpainting-config",
        str(runtime_config),
        "--output",
        str(final),
        "--raw-output",
        str(raw),
        "--final-manifest",
        str(manifest),
        "--work-dir",
        str(stage / "segments"),
        "--expected-frames",
        str(frame_count),
        "--tile-size",
        str(up["tile_size"]),
        "--tile-overlap",
        str(up["tile_overlap"]),
        "--min-tile-size",
        str(up["min_tile_size"]),
        "--sparse-ratio",
        str(up["sparse_ratio"]),
        "--local-range",
        str(up["local_range"]),
        "--seed",
        str(config["upscale_seed"]),
        "--gpu-index",
        str(up["gpu_index"]),
        "--ffmpeg-threads",
        str(up["ffmpeg_threads"]),
        "--encoder",
        str(up["encoder"]),
        "--output-mode",
        "jpeg_sequence",
    ]
    if selected_range:
        command.extend([
            "--selected-range", "--start-frame", str(start_frame),
            "--omit-original-audio", "--direct-jpeg-output",
        ])
        if "original_metadata" in input_identity:
            command.extend(["--preverified-original-metadata-json", str(input_identity_path)])
    with stage_log(stage / "logs" / "flashvsr.log"):
        log(
            "UPSCALE",
            video=video_id,
            seed=config["upscale_seed"],
            frames=frame_count,
            source=selected,
            model="FlashVSR_v1.1",
            scale=4,
            seed_mode="per_tile_generator_plus_global_rng",
            output_mode="jpeg_sequence",
        )
        run_logged(command, cwd=ROOT)
        output_mode = "jpeg_sequence"
        final_record = json.loads(manifest.read_text(encoding="utf-8"))
        verify_jpeg_sequence(
            final, frame_count, start_frame=start_frame if selected_range else 0,
            exact=not selected_range,
        )
        if (final_record.get("status") != "PASS"
                or final_record.get("encoding", {}).get("output_mode") != "jpeg_sequence"
                or final_record.get("encoding", {}).get("frames_written") != frame_count):
            raise RuntimeError("JPG sequence manifest/frame count mismatch")
        atomic_json(
            stage / "metadata" / "flashvsr_config.json",
            {**up, "requested_output_mode": output_mode, "actual_output_mode": output_mode},
        )
        stage_manifest = stage / "segments" / "flashvsr_stage_manifest.json"
        if not stage_manifest.is_file():
            raise RuntimeError("FlashVSR native stage manifest missing")
        stage_record = json.loads(stage_manifest.read_text(encoding="utf-8"))
        stage_report = stage_record.get("partial_report", {})
        tile_configuration = (
            stage_report.get("configuration", {})
            if isinstance(stage_report, dict)
            else {}
        )
        tile_count = stage_record.get("tile_count")
        tile_size = tile_configuration.get("tile_size")
        tile_overlap = tile_configuration.get("tile_overlap")
        if all(type(value) is int for value in (tile_count, tile_size, tile_overlap)):
            log(
                "UPSCALE_TILE_PLAN",
                tile_size=tile_size,
                overlap=tile_overlap,
                tiles=tile_count,
                frames_per_tile=frame_count,
                status="PASS",
            )
        atomic_json(
            stage / "metadata" / "segment_plan.json",
            {
                "actual_unit": "TinyLong causal emissions per spatial tile",
                "source_manifest": str(stage_manifest),
                "temporal_unit_metadata": stage_record.get(
                    "temporal_unit_metadata", []
                ),
                "tile_count": stage_record.get("tile_count"),
                "frames": frame_count,
            },
        )
        if config.get("verification", {}).get("full_decode", False):
            raise ConfigError("full pixel decode is unavailable for KOCCA JPG output")
        log(
            "UPSCALE_DONE",
            temporal_units=len(stage_record.get("temporal_unit_metadata", [])),
            tiles=stage_record.get("tile_count"),
            frames=f"{frame_count}/{frame_count}",
            resolution="15360x8640",
            fps=16,
            status="PASS",
            output=final,
        )
    return final


def upscale_chunk_ranges(frame_count: int, chunk_frames: int) -> list[tuple[int, int]]:
    """Keep every independent FlashVSR segment at or above its 21-frame minimum."""
    if frame_count < chunk_frames or chunk_frames != 21:
        raise ConfigError("verification Upscale requires at least 21 frames")
    ranges: list[tuple[int, int]] = []
    start = 0
    while frame_count - start >= 2 * chunk_frames:
        ranges.append((start, chunk_frames))
        start += chunk_frames
    ranges.append((start, frame_count - start))
    return ranges


def prepare_direct_jpeg_output(
    stage: Path, video_id: str, request: dict[str, Any], *, resume: bool
) -> Path:
    """Give this run ownership of the public folder before writing any JPEGs."""
    output = KOCCA_ROOT / "Output" / f"{video_id}_16K"
    owner = stage / "metadata" / "output_owner.json"
    archive = stage.parent / "previous_published_output"
    output.parent.mkdir(parents=True, exist_ok=True)
    owner.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        if owner.is_file():
            record = json.loads(owner.read_text(encoding="utf-8"))
            if record != {"output": str(output), "request": request}:
                raise ConfigError("direct JPG output ownership does not match this run")
        elif archive.exists() or (output.is_dir() and not any(output.iterdir())):
            # Recover a crash between moving the old result and writing the marker.
            atomic_json(owner, {"output": str(output), "request": request})
        else:
            raise ConfigError("direct JPG output ownership is missing; refusing to overwrite it")
        if not output.exists():
            output.mkdir()
    else:
        if owner.exists() or archive.exists():
            raise ConfigError("direct JPG output was already prepared for this run")
        if output.exists():
            if not output.is_dir() or output.is_symlink():
                raise ConfigError(f"public JPG output is not a directory: {output}")
            os.replace(output, archive)
            log("UPSCALE_PREVIOUS_OUTPUT_SAVED", path=archive)
        output.mkdir()
        atomic_json(owner, {"output": str(output), "request": request})
    if not output.is_dir() or output.is_symlink():
        raise ConfigError(f"direct JPG output is not a directory: {output}")
    return output


def clear_uncommitted_jpeg_range(output: Path, start: int, count: int) -> None:
    """Remove only this chunk's files after an interrupted encode."""
    for index in range(start, start + count):
        (output / f"{index}.jpg").unlink(missing_ok=True)


def _run_flash_chunked(
    config: dict[str, Any],
    source_8k: Path,
    outpaint_4k: Path,
    outpaint_frames: int,
    runtime_config: Path,
    model_root: Path,
    *,
    resume: bool,
) -> Path:
    stage = artifact_path(config, "upscale")
    video_id = config["input"]["video_id"]
    plan = upscale_chunk_ranges(outpaint_frames, config["upscale_chunk_frames"])
    receipt_path = stage / "metadata" / "stage_complete.json"
    if resume and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        final = Path(receipt["output"])
        if receipt.get("status") != "PASS" or receipt.get("request") != stage_identity(config, "upscale"):
            raise ConfigError("completed Upscale receipt does not match this KOCCA request")
        verify_jpeg_sequence(final, int(receipt["frames"]))
        log("UPSCALE_ASSEMBLED", frames=receipt["frames"],
            chunks=receipt["chunks"], reused=True)
        return final

    final = prepare_direct_jpeg_output(
        stage, video_id, stage_identity(config, "upscale"), resume=resume
    )

    completed: list[tuple[int, int, Path]] = []
    current_chunk = 0
    stopped = False

    def on_stop(_signum: int, _frame: Any) -> None:
        raise UpscaleStop()

    previous_handler = signal.signal(signal.SIGUSR1, on_stop)
    try:
        log("UPSCALE_STOP_CONTROL_READY", key="ctrl+q")
        try:
            for index, (start, count) in enumerate(plan):
                current_chunk = index
                log("UPSCALE_CHUNK_START", chunk=index, frames=f"{start}-{start + count - 1}",
                    position=f"{index + 1}/{len(plan)}")
                chunk_stage = stage / "chunks" / f"chunk_{index:03d}"
                if not (chunk_stage / "metadata" / "final_manifest.json").is_file():
                    clear_uncommitted_jpeg_range(final, start, count)
                result = _run_flash_native(
                    config, source_8k, outpaint_4k, count, runtime_config,
                    model_root, resume=resume, stage_override=chunk_stage,
                    start_frame=start, selected_range=True,
                )
                completed.append((start, count, result))
                log("UPSCALE_CHUNK_DONE", chunk=index,
                    completed_frames=start + count, status="PASS")
        except UpscaleStop:
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            if not completed:
                start, count = plan[current_chunk]
                clear_uncommitted_jpeg_range(final, start, count)
                raise ConfigError("Ctrl+Q was pressed before Upscale chunk 0 completed")
            stopped = True
            if current_chunk >= len(completed):
                start, count = plan[current_chunk]
                clear_uncommitted_jpeg_range(final, start, count)
                log("UPSCALE_CHUNK_EXCLUDED", chunk=current_chunk, reason="not_committed")
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)

    accepted = sum(count for _, count, _ in completed)
    if accepted < 21 or completed[0][0] != 0:
        raise ConfigError("Upscale completed chunks do not contain a valid leading range")
    for previous, current in zip(completed, completed[1:]):
        if previous[0] + previous[1] != current[0]:
            raise ConfigError("Upscale completed chunks are not contiguous")

    verify_jpeg_sequence(final, accepted)

    atomic_json(stage / "metadata" / "final_manifest.json", {
        "schema_version": 1,
        "status": "PASS",
        "video_id": video_id,
        "request": stage_identity(config, "upscale"),
        "output": str(final),
        "outpaint_input": str(outpaint_4k),
        "temporal_mode": "independent_21_frame_segments",
        "chunks": [
            {
                "index": index, "start_frame": start, "frames": count,
                "output": str(path),
                "manifest": str(stage / "chunks" / f"chunk_{index:03d}" / "metadata" / "final_manifest.json"),
            }
            for index, (start, count, path) in enumerate(completed)
        ],
        "encoding": {"output_mode": "jpeg_sequence", "frames_written": accepted},
        "verification_policy": {
            "full_16k_pixel_decode_performed": False,
            "ffprobe_count_frames_performed": False,
            "final_frame_count_evidence": "numbered_jpg_files_and_completed_chunk_encoder_counters",
            "limitations": "JPG files were counted and checked for nonzero size; each chunk has its own PASS manifest",
        },
    })
    atomic_json(receipt_path, {
        "status": "PASS", "request": stage_identity(config, "upscale"),
        "frames": accepted, "chunks": len(completed),
        "stop_requested": stopped, "output": str(final),
    })
    log("UPSCALE_ASSEMBLED", frames=accepted, chunks=len(completed),
        stop_requested=stopped)
    return final


def validate_artifact_layout(config: dict[str, Any]) -> None:
    """Keep every KOCCA artifact inside this run's isolated output directory."""
    video = config["input"]["video_id"]
    run_id = config["job_name"].removeprefix(f"{video}_kocca_")
    base = KOCCA_ROOT / "Runs" / video / run_id
    expected = {
        "preprocess": base / "preprocess" / f"{video}_PreProc_2K.mp4",
        "preprocess_metadata": base / "preprocess" / "metadata",
        "outpaint": base / "outpaint",
        "upscale": base / "upscale",
    }
    for name, target in expected.items():
        actual = absolute_package_path(config["artifacts"][name], must_exist=False)
        require_equal(actual, target, f"artifacts.{name}")
    require_equal(
        absolute_package_path(config["output_root"], must_exist=False),
        base / "state", "output_root",
    )


def artifact_path(config: dict[str, Any], name: str) -> Path:
    return absolute_package_path(config["artifacts"][name], must_exist=False)


def preprocess_output_path(config: dict[str, Any]) -> Path:
    return artifact_path(config, "preprocess")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_identity(config: dict[str, Any], stage: str) -> dict[str, Any]:
    keys = ["input", "seed", "frames", "preprocess", "outpainting"]
    if stage == "upscale":
        keys += ["upscale", "final", "upscale_seed", "upscale_chunk_frames"]
    return {key: config[key] for key in keys}


@contextlib.contextmanager
def stage_lock(name: str) -> Iterator[None]:
    import fcntl

    directory = ROOT / "state" / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ConfigError(
                f"Another {name} process is running; no duplicate worker allowed"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stage_directories(name: str) -> tuple[str, ...]:
    return (
        ("final", "metadata", "logs", "chunks", "lineart")
        if name == "outpaint"
        else ("final", "metadata", "logs", "raw_flashvsr", "segments")
    )


def stage_artifact_files(stage: Path, name: str) -> list[Path]:
    return [
        p
        for sub in stage_directories(name)
        for p in (stage / sub).rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    ]


def prepare_stage(
    stage: Path, config: dict[str, Any], name: str, *, resume: bool
) -> None:
    request = stage / "metadata" / "stage_request.json"
    identity = stage_identity(config, name)
    meaningful = stage_artifact_files(stage, name)
    if meaningful:
        if not resume:
            raise ConfigError(f"{stage} contains artifacts: use --resume")
        if not request.is_file() or json.loads(request.read_text()) != identity:
            raise ConfigError(
                f"{name} resume request does not match current model/config/frame limits"
            )
    for sub in (
        (
            "final",
            "metadata",
            "logs",
            "chunks/rgb",
            "chunks/native",
            "chunks/latents",
            "chunks/manifests",
        )
        if name == "outpaint"
        else (
            "final",
            "metadata",
            "logs",
            "raw_flashvsr",
            "segments/input",
            "segments/metadata",
            "segments/tiles",
        )
    ):
        (stage / sub).mkdir(parents=True, exist_ok=True)
    # Layout placeholders are not native journal artifacts. The unchanged
    # SegmentedOutputStore requires a genuinely empty directory on first use.
    for sub in stage_directories(name):
        for marker in (stage / sub).rglob(".gitkeep"):
            if marker.is_symlink() or not marker.is_file() or marker.stat().st_size:
                raise ConfigError(
                    f"Unexpected nonempty/symlink layout marker: {marker}"
                )
            marker.unlink()
    atomic_json(request, identity)


def validate_preprocessed(config: dict[str, Any]) -> tuple[Path, int, str]:
    path = preprocess_output_path(config)
    if not path.is_file():
        raise ConfigError(
            f"Canonical PreProc is missing: {path}; run --stage preprocess first"
        )
    probe = preprocess.probe_media(path, shutil.which("ffprobe") or "ffprobe")
    video = preprocess._video_stream(probe)
    frames = preprocess._frame_count(video)
    digest = sha256_file(path)
    manifest = path.with_suffix(".json")
    if not manifest.is_file():
        raise ConfigError(f"PreProc provenance manifest required: {manifest}")
    provenance = json.loads(manifest.read_text())
    if provenance.get("output_sha256") != digest:
        raise ConfigError(f"PreProc provenance SHA mismatch: {manifest}")
    preprocess.validate_producer_provenance(provenance)
    if provenance.get("producer") == "run_preprocess_stage":
        current_producer = preprocess.producer_identity(
            shutil.which("ffmpeg") or "ffmpeg", shutil.which("ffprobe") or "ffprobe"
        )
        if provenance.get("producer_identity") != current_producer:
            raise ConfigError(
                "PreProc producer implementation/toolchain identity changed"
            )
    source = input_video_path(config)
    expected_source = provenance.get("input_sha256") or provenance.get(
        "source_identity", {}
    ).get("sha256")
    if expected_source != sha256_file(source):
        raise ConfigError(
            "PreProc is not bound to the configured original source bytes"
        )
    declared_source = provenance.get("source") or provenance.get("input", "")
    if Path(declared_source).name != source.name:
        raise ConfigError("PreProc source video ID/provenance mismatch")
    declared_frames = provenance.get("frames") or provenance.get(
        "verification", {}
    ).get("output_frame_count")
    input_frames = provenance.get("frames") or provenance.get("verification", {}).get(
        "input_frame_count"
    )
    if declared_frames != frames or input_frames != frames:
        raise ConfigError("PreProc provenance frame-count mismatch/truncation")
    # SHA binds this reported count to the verified migrated bytes; count-decode
    # only when no usable container count exists, never assume a fixed N.
    source_probe = json.loads(
        subprocess.check_output(
            [
                shutil.which("ffprobe") or "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(source),
            ],
            text=True,
        )
    )
    try:
        source_count = preprocess._frame_count(preprocess._video_stream(source_probe))
    except preprocess.PreprocessError:
        source_probe = preprocess.probe_media(
            source, shutil.which("ffprobe") or "ffprobe"
        )
        source_count = preprocess._frame_count(preprocess._video_stream(source_probe))
    preprocess.validate_input_probe(source_probe, source_count)
    preprocess.validate_output_probe(
        probe, frames, bool(preprocess._audio_streams(source_probe))
    )
    if source_count != frames:
        raise ConfigError("Canonical PreProc does not cover the full original source")
    log(
        "PREPROCESS_CACHE",
        path=path,
        frames=frames,
        sha256=digest,
        source_sha256=expected_source,
        producer_crf=provenance.get(
            "crf", provenance.get("preprocess_config", {}).get("crf")
        ),
        status="PASS",
    )
    return path, frames, digest


def snapshot_resolved_frames(
    config: dict[str, Any], available: int, stage: str
) -> None:
    """Persist resolved conditions before loading a model, including failed runs."""
    resolved = json.loads(json.dumps(config))
    outpaint = resolve_limit(config["frames"]["outpaint"], available, "outpaint")
    upscale = resolve_limit(config["frames"]["upscale"], outpaint, "upscale")
    resolved["executed_stage"] = stage
    resolved["frames"] = {
        "source": available,
        "outpaint": outpaint,
        "upscale": upscale,
        "final": upscale,
    }
    atomic_json(
        absolute_package_path(config["output_root"], must_exist=False)
        / "resolved_config.json",
        resolved,
    )


@contextlib.contextmanager
def preprocess_progress(expected_frames: int, metadata_dir: Path) -> Iterator[None]:
    """Expose FFmpeg's real frame counter for this KOCCA child process only.

    The content-producing command is unchanged. FFmpeg's ``-progress`` option
    only requests machine-readable status, and the exact executed command is
    retained beside the canonical producer record for audit.
    """
    original = preprocess._run_logged

    def run_with_progress(command: list[str], log_path: Path) -> None:
        actual = list(command)
        actual[3:3] = ["-progress", "pipe:1"]
        atomic_json(
            metadata_dir / "progress_instrumentation.json",
            {"canonical_ffmpeg_command": list(command), "executed_ffmpeg_command": actual},
        )
        latest = -1
        with log_path.open("a", encoding="utf-8") as details:
            process = subprocess.Popen(
                actual, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                details.write(line)
                details.flush()
                match = re.fullmatch(r"frame=\s*(\d+)\s*", line)
                if match:
                    frame = min(int(match.group(1)), expected_frames)
                    if frame > latest:
                        log("PREPROCESS_PROGRESS", frame=frame, total=expected_frames)
                        latest = frame
                elif line.strip() and not ("=" in line and line.split("=", 1)[0] in {
                    "fps", "stream_0_0_q", "bitrate", "total_size", "out_time_us",
                    "out_time_ms", "out_time", "dup_frames", "drop_frames", "speed", "progress",
                }):
                    print(line, end="", flush=True)
            return_code = process.wait()
        if return_code:
            raise preprocess.PreprocessError(f"ffmpeg exited with status {return_code}")

    preprocess._run_logged = run_with_progress
    try:
        yield
    finally:
        preprocess._run_logged = original


def run_preprocess_stage(
    config: dict[str, Any], *, resume: bool = False
) -> Path:
    """Create the isolated 2K preprocessing artifact (CPU only)."""
    source = input_video_path(config)
    destination = preprocess_output_path(config)
    metadata = artifact_path(config, "preprocess_metadata")
    with stage_lock(f"video_{config['input']['video_id']}"):
        source_frames = preflight_source(source)
        source_identity = preprocess._source_identity(source)
        producer = preprocess.producer_identity(
            shutil.which("ffmpeg") or "ffmpeg", shutil.which("ffprobe") or "ffprobe"
        )
        canonical_manifest = destination.with_suffix(".json")
        if resume and destination.is_file() and canonical_manifest.is_file():
            previous = json.loads(canonical_manifest.read_text())
            # Resume only when source, settings, and toolchain still match.
            if (
                previous.get("producer") == "run_preprocess_stage"
                and previous.get("preprocess_config") == config["preprocess"]
                and previous.get("source_identity") == source_identity
                and previous.get("producer_identity") == producer
            ):
                _, frames, _ = validate_preprocessed(config)
                if frames != source_frames:
                    raise ConfigError("Preprocess cache/source frame count mismatch")
                log("PREPROCESS_PROGRESS", frame=frames, total=source_frames)
                log("PREPROCESS_REUSE", path=destination, frames=frames, status="PASS")
                return destination
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        work = metadata / "runs" / stamp
        with preprocess_progress(source_frames, work / "metadata"):
            generated = preprocess.run_preprocess(
                source,
                work / "output",
                work / "metadata",
                metadata / "logs" / "preprocess.log",
                source_frames,
                resume=False,
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            preserved = metadata / "preserved" / stamp
            preserved.mkdir(parents=True, exist_ok=True)
            atomic_copy(destination, preserved / destination.name)
            if canonical_manifest.exists():
                atomic_copy(canonical_manifest, preserved / canonical_manifest.name)
        digest = sha256_file(generated)
        atomic_copy(generated, destination)
        if sha256_file(destination) != digest:
            raise RuntimeError("Preprocess publication copy SHA mismatch")
        atomic_json(
            canonical_manifest,
            {
                "producer": "run_preprocess_stage",
                "output_sha256": digest,
                "source_identity": source_identity,
                "source": str(source),
                "preprocess_config": config["preprocess"],
                "producer_identity": producer,
                "frames": source_frames,
                "audit_metadata": str(work / "metadata"),
                "generated": str(generated),
            },
        )
        # The packaged preprocessor has completed its strict full-frame decode
        # check at this point. Confirm the final count even if FFmpeg emitted
        # fewer progress snapshots during encoding.
        log("PREPROCESS_PROGRESS", frame=source_frames, total=source_frames)
        log(
            "PREPROCESS_STAGE_DONE",
            output=destination,
            frames=source_frames,
            status="PASS",
        )
        return destination


def assemble_stopped_outpaint(
    config: dict[str, Any], stage: Path, preprocessed: Path, input_sha: str,
    available: int,
) -> tuple[Path, int, int]:
    """Publish only the native journal's validated, contiguous committed prefix."""
    from cert_pipeline import outpaint as native
    outpaint_src = ROOT / "src" / "video_outpainting" / "main" / "src"
    if str(outpaint_src) not in sys.path:
        sys.path.insert(0, str(outpaint_src))
    from ltx_outpaint.streaming_video import SegmentedOutputStore, encode_segments, write_record_manifest

    video_id = config["input"]["video_id"]
    chunks = stage / "chunks"
    native_dir = chunks / "native"
    manifests = chunks / "manifests"
    metadata = stage / "metadata"
    log_path = stage / "logs" / "outpaint.log"
    processing = native_dir / f"{video_id}_Outpaint_4K_24fps.mp4"
    final = stage / "final" / f"{video_id}_Outpaint_4K.mp4"
    session_path = native_dir / "run.json"
    if not session_path.is_file():
        raise ConfigError("Ctrl+Q was pressed before the native OutPainting session was created")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    request = session.get("request", {})
    if (request.get("input_path") != str(preprocessed)
            or request.get("input_sha256") != input_sha
            or request.get("requested_frames") != available
            or request.get("seed") != config["seed"]
            or request.get("output_path") != str(processing)):
        raise ConfigError("stopped OutPainting journal does not match this request")
    records = native._load_native_records(native_dir)
    if not records:
        raise ConfigError("Ctrl+Q was pressed before chunk 0 committed; no output can be assembled")
    store = SegmentedOutputStore(native_dir)
    accepted = 0
    artifacts = []
    for index, record in enumerate(records):
        begin, end = map(int, record["new_output_frame_range"])
        planned_end = min(available, 65 + index * 48)
        if (begin != accepted or end != planned_end
                or int(record["chunk_index"]) != index
                or int(record["output_appended_count"]) != end - begin):
            raise ConfigError(f"committed chunk {index} breaks the source frame plan")
        if store.load_record(
            record, output_height=2160, output_width=3840,
            chunk_seed=config["seed"] + int(record["absolute_start_frame"]),
        ) is None:
            raise ConfigError(f"committed chunk {index} is missing")
        segment = native_dir / str(record["segment_file"])
        artifacts.append(native._persist_committed_chunk(
            record, segment, native_segment_dir=native_dir,
            mp4_dir=chunks / "rgb", latent_dir=chunks / "latents",
            log_path=log_path,
        ))
        accepted = end
    encode_segments(
        records, native_dir, processing, fps=24.0,
        width=3840, height=2160, expected_frames=accepted,
    )
    native._write_certification_manifests(
        records, artifacts, manifests_dir=manifests, frame_count=accepted,
    )
    write_record_manifest(
        manifests / "chunk_lineage.json",
        {"requested_frames": accepted, "original_requested_frames": available,
         "clip_frames": 65, "prefix_frames": 17, "stride_frames": 48,
         "gap_count": 0, "duplicate_count": 0, "stop_requested": True},
        records,
    )
    native._atomic_json(manifests / "inference_manifest.json", {
        "input_path": str(preprocessed), "output_path": str(processing),
        "original_requested_frames": available, "requested_frames": accepted,
        "chunk_count": len(records), "seed": config["seed"],
        "inference_steps": 30, "stop_requested": True,
        "segment_directory": str(native_dir),
    })
    native._atomic_json(metadata / "ffprobe_processing_24fps.json", native._probe(processing))
    native._retime_to_16fps(
        processing, final, frame_count=accepted, log_path=log_path, resume=False,
    )
    native._atomic_json(metadata / "ffprobe_final.json", native._probe(final))
    native._atomic_json(metadata / "inference_config.json", {
        "runtime_config": str(absolute_package_path(config["outpainting"]["runtime_config"])),
        "reference_entrypoint": "ltx_outpaint.inference.sample_long_range",
        "original_requested_frames": available, "requested_frames": accepted,
        "chunk_count": len(records), "stop_requested": True,
        "processing_fps": 24, "stored_fps": 16,
        "frame_count_preserved": True, "fps_resampling": False,
        "alpha_feather_px": 16,
    })
    native._atomic_json(metadata / "seed.json", {
        "seed": config["seed"],
        "seed_mode": "global_rng_and_reference_chunk_seed",
        "reference_chunk_seed": "base_seed + absolute_start_frame",
    })
    log("OUTPAINT_ASSEMBLED", frames=accepted, chunks=len(records), stop_requested=True)
    return final, accepted, len(records)


def run_outpaint_stage(
    config: dict[str, Any], *, resume: bool = False
) -> Path:
    """Run VideoOutPainting for the selected frame range."""
    with stage_lock("outpaint"), stage_lock(f"video_{config['input']['video_id']}"):
        preprocessed, available, input_sha = validate_preprocessed(config)
        log(
            "OUTPAINT_INPUT_CHECK",
            path=preprocessed,
            resolution="1920x1080",
            fps=24,
            frames=available,
            provenance="PASS",
            status="PASS",
        )
        frames = resolve_limit(config["frames"]["outpaint"], available, "outpaint")
        snapshot_resolved_frames(config, available, "outpaint")
        stage = artifact_path(config, "outpaint")
        prepare_stage(stage, config, "outpaint", resume=resume)
        identity = {"sha256": input_sha, "frames": available, "path": str(preprocessed)}
        if resume and (stage / "metadata" / "stage_complete.json").is_file():
            receipt = json.loads((stage / "metadata" / "stage_complete.json").read_text())
            if receipt.get("input") != identity:
                raise ConfigError("Outpainting completed artifact input identity changed")
            completed, completed_frames = validate_outpaint_artifact(config)
            log("OUTPAINT_ASSEMBLED", frames=completed_frames,
                chunks=receipt["chunks"],
                reused=True)
            return completed
        prior_input = stage / "metadata" / "input_identity.json"
        if (
            resume
            and prior_input.is_file()
            and json.loads(prior_input.read_text()) != identity
        ):
            raise ConfigError("Outpainting resume PreProc identity changed")
        atomic_json(prior_input, identity)
        with stage_log(stage / "logs" / "outpaint.log"):
            log(
                "OUTPAINT",
                video=config["input"]["video_id"],
                seed=config["seed"],
                frames=frames,
                steps=30,
                status="START",
            )
            seed_all(config["seed"])
            from cert_pipeline.outpaint import run_outpaint

            def on_stop(_signum: int, _frame: Any) -> None:
                raise OutpaintStop()

            stop_request = absolute_package_path(config["output_root"], must_exist=False) / "outpaint_stop_request.json"
            if resume and stop_request.is_file():
                stopped = True
                result, completed_frames, completed_chunks = assemble_stopped_outpaint(
                    config, stage, preprocessed, input_sha, available,
                )
            else:
                previous_handler = signal.signal(signal.SIGUSR1, on_stop)
                stopped = False
                try:
                    try:
                        log("STOP_CONTROL_READY", key="ctrl+q")
                        result = run_outpaint(ROOT, config, preprocessed, stage, frames, resume)
                        completed_frames = frames
                        chunk_plan = json.loads((stage / "chunks" / "manifests" / "chunk_plan.json").read_text())
                        completed_chunks = int(chunk_plan["chunk_count"])
                        log("OUTPAINT_ASSEMBLED", frames=completed_frames,
                            chunks=completed_chunks, stop_requested=False)
                    except OutpaintStop:
                        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
                        stopped = True
                        result, completed_frames, completed_chunks = assemble_stopped_outpaint(
                            config, stage, preprocessed, input_sha, available,
                        )
                finally:
                    signal.signal(signal.SIGUSR1, previous_handler)
            processing_probe_path = stage / "metadata" / "ffprobe_processing_24fps.json"
            final_probe_path = stage / "metadata" / "ffprobe_final.json"
            if not processing_probe_path.is_file() or not final_probe_path.is_file():
                raise RuntimeError("Outpainting completion probes are missing")
            validated_probe_summary(
                json.loads(processing_probe_path.read_text(encoding="utf-8")),
                width=3840,
                height=2160,
                fps=24,
                frames=completed_frames,
                label="Outpainting native assembly",
            )
            final_summary = validated_probe_summary(
                json.loads(final_probe_path.read_text(encoding="utf-8")),
                width=3840,
                height=2160,
                fps=16,
                frames=completed_frames,
                label="Outpainting final retime",
            )
            if completed_frames != int(final_summary["frames"]):
                raise RuntimeError("OutPainting accepted-frame count changed during retime")
            atomic_json(
                stage / "metadata" / "stage_complete.json",
                {
                    "status": "PASS",
                    "request": stage_identity(config, "outpaint"),
                    "input": identity,
                    "frames": completed_frames,
                    "chunks": completed_chunks,
                    "stop_requested": stopped,
                    "requested_frames": frames,
                    "output": str(result),
                    "output_sha256": sha256_file(result),
                    "long_range_domain_fps": 24,
                    "output_fps": 16,
                    "output_role": "final_16fps_deliverable",
                },
            )
            log(
                "OUTPAINT_DONE",
                video=config["input"]["video_id"],
                frames=f"{completed_frames}/{frames}",
                output=result,
                status="PASS",
            )
        return result


def validate_outpaint_artifact(config: dict[str, Any]) -> tuple[Path, int]:
    stage = artifact_path(config, "outpaint")
    path = stage / "final" / f"{config['input']['video_id']}_Outpaint_4K.mp4"
    receipt = stage / "metadata" / "stage_complete.json"
    if not receipt.is_file() or not path.is_file():
        raise ConfigError(
            "A committed Outpainting stage is required before --stage upscale"
        )
    record = json.loads(receipt.read_text())
    if record.get("status") != "PASS" or record.get("request") != stage_identity(
        config, "outpaint"
    ):
        raise ConfigError("Outpainting stage receipt/config mismatch")
    if sha256_file(path) != record.get("output_sha256"):
        raise ConfigError("Outpainting final SHA mismatch")
    manifests = stage / "chunks" / "manifests"
    for name in (
        "chunk_plan.json",
        "chunk_results.jsonl",
        "chunk_lineage.json",
        "inference_manifest.json",
    ):
        if not (manifests / name).is_file():
            raise ConfigError(f"Outpainting artifact manifest missing: {name}")
    frames = int(record["frames"])
    verify_video(path, width=3840, height=2160, fps=16, frames=frames)
    return path, frames


def run_upscale_stage(
    config: dict[str, Any], *, resume: bool = False
) -> Path:
    """Run FlashVSR and center composite to numbered JPG images."""
    with stage_lock("upscale"), stage_lock(f"video_{config['input']['video_id']}"):
        outpainted, outpaint_frames = validate_outpaint_artifact(config)
        log(
            "UPSCALE_INPUT_CHECK",
            path=outpainted,
            resolution="3840x2160",
            fps=16,
            frames=outpaint_frames,
            status="PASS",
        )
        frames = outpaint_frames
        stage = artifact_path(config, "upscale")
        prepare_stage(stage, config, "upscale", resume=resume)
        return _run_flash_chunked(
            config, input_video_path(config), outpainted, frames,
            absolute_package_path(config["outpainting"]["runtime_config"]),
            absolute_package_path(config["upscale"]["model"]),
            resume=resume,
        )


def execute_stages(
    config: dict[str, Any], stage: str, *, resume: bool = False
) -> Path:
    if stage == "preprocess":
        return run_preprocess_stage(config, resume=resume)
    if stage == "outpaint":
        return run_outpaint_stage(config, resume=resume)
    if stage == "upscale":
        return run_upscale_stage(config, resume=resume)
    raise ConfigError(f"Unknown stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="KOCCA isolated 8K to 16K stage execution"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("preprocess", "outpaint", "upscale"), required=True
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    config_path = args.config.expanduser().resolve()
    config, _, run, _, _ = validate_config(config_path)
    run.mkdir(parents=True, exist_ok=True)
    requested = run / "requested_config.json"
    if requested.exists() and requested.read_bytes() != config_path.read_bytes():
        raise ConfigError("Config bytes differ from existing requested_config.json")
    if not args.resume and args.stage in ("outpaint", "upscale"):
        target = artifact_path(config, args.stage)
        if stage_artifact_files(target, args.stage):
            raise ConfigError(f"{args.stage} target contains artifacts; use --resume")
    atomic_copy(config_path, requested)
    raw_log = run.parent / "logs" / "raw" / f"{config['job_name']}.log"
    with stage_log(raw_log):
        log(
            "CONFIG",
            job=config["job_name"],
            stage=args.stage,
            seed=config["seed"],
            config=config_path,
        )
        result = execute_stages(config, args.stage, resume=args.resume)
        resolved = json.loads(json.dumps(config))
        resolved["upscale"]["output_mode"] = config["upscale"].get(
            "output_mode", "jpeg_sequence"
        )
        resolved["executed_stage"] = args.stage
        selected_preproc = preprocess_output_path(config)
        resolved["resolved_preprocess_artifact"] = str(selected_preproc)
        if selected_preproc.with_suffix(".json").is_file():
            provenance = json.loads(selected_preproc.with_suffix(".json").read_text())
            available = provenance.get("frames") or provenance.get(
                "verification", {}
            ).get("output_frame_count")
            if available:
                out = resolve_limit(
                    config["frames"]["outpaint"], int(available), "outpaint"
                )
                receipt = artifact_path(config, "outpaint") / "metadata" / "stage_complete.json"
                if receipt.is_file():
                    out = int(json.loads(receipt.read_text(encoding="utf-8"))["frames"])
                up = out
                upscale_receipt = artifact_path(config, "upscale") / "metadata" / "stage_complete.json"
                if upscale_receipt.is_file():
                    up = int(json.loads(upscale_receipt.read_text(encoding="utf-8"))["frames"])
                    if up < 21 or up > out:
                        raise ConfigError("accepted Upscale frames exceed the Outpainting result")
                resolved["frames"] = {
                    "source": available,
                    "outpaint": out,
                    "upscale": up,
                    "final": up,
                }
        atomic_json(run / "resolved_config.json", resolved)
        log(
            "PIPELINE_DONE",
            job=config["job_name"],
            stage=args.stage,
            output=result,
            status="PASS",
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"[PIPELINE_FAIL] {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from error
