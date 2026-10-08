#!/usr/bin/env python3
"""One JSON-driven entrypoint for shared Full, Quick, stage-only and queue execution.

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
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

preprocess = importlib.import_module("cert_pipeline.preprocess")
flash_composite = importlib.import_module("cert_pipeline.flash_composite")

SEEDS = {"01": 50374151, "02": 25621683, "03": 33333333, "04": 71634910, "05": 71634910}
NEGATIVE_PROMPT = "people, person, human, staff, audience, crowd, backstage, seats"
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


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ConfigError(f"{label}: got {actual!r}, expected {expected!r}")


def check_limit(value: Any, name: str) -> None:
    if value != "all" and (type(value) is not int or value <= 0):
        raise ConfigError(f"frames.{name} must be 'all' or a positive integer")


def resolve_limit(value: Any, available: int, name: str) -> int:
    resolved = available if value == "all" else value
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
    source = config.get("input", {})
    video_id = source.get("video_id")
    if video_id not in SEEDS:
        raise ConfigError(f"unsupported video_id: {video_id!r}")
    require_equal(config.get("seed"), SEEDS[video_id], "seed")
    source_path = absolute_package_path(source.get("path"))
    require_equal(
        source_path,
        ROOT / f"data/00_Input_8K_7680x4320/{video_id}_Input_8K.mp4",
        "input.path",
    )
    if not source_path.is_file():
        raise ConfigError(f"input is not a file: {source_path}")
    frames = config.get("frames", {})
    check_limit(frames.get("outpaint"), "outpaint")
    check_limit(frames.get("upscale"), "upscale")
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
    if up.get("tile_size") != "auto" or up.get("tile_overlap") != "auto":
        raise ConfigError("current certified FlashVSR tile settings must be auto")
    if up.get("sparse_ratio") not in (1.5, 2.0) or up["local_range"] not in (9, 11):
        raise ConfigError("unsupported FlashVSR temporal options")
    if up.get("encoder") not in ("libx265", "hevc_nvenc"):
        raise ConfigError("unsupported FlashVSR MP4 encoder")
    if up.get("output_mode", "final_only") not in ("final_only", "legacy_raw"):
        raise ConfigError("upscale.output_mode must be final_only or legacy_raw")
    final = config.get("final", {})
    for key, expected in {
        "fps": 16,
        "center_original_8k": True,
        "center_feather_px": 64,
        "audio_source": "original_8k",
    }.items():
        require_equal(final.get(key), expected, f"final.{key}")
    if (
        not isinstance(final.get("filename"), str)
        or not final["filename"].endswith(".mp4")
        or Path(final["filename"]).name != final["filename"]
    ):
        raise ConfigError("final.filename must be a plain MP4 basename")
    output_root = absolute_package_path(config.get("output_root"), must_exist=False)
    if not output_root.is_relative_to(ROOT / "state" / "runs"):
        raise ConfigError("output_root must be beneath package state/runs/")
    job = config.get("job_name")
    if "validation" in config:
        if (config["validation"] != {"native_outpaint_only": True}
                or job not in (f"{video_id}_recurrence_c0", f"{video_id}_recurrence_c1")):
            raise ConfigError("Native-only validation is restricted to recurrence C0/C1 audit jobs")
    if (
        not isinstance(job, str)
        or not job.startswith(video_id + "_")
        or Path(job).name != job
    ):
        raise ConfigError("invalid job_name")
    artifacts = config.get("artifacts", {})
    for name in ("preprocess", "preprocess_metadata", "outpaint", "upscale"):
        target = absolute_package_path(artifacts.get(name), must_exist=False)
        if not target.is_relative_to(ROOT / "data"):
            raise ConfigError(f"artifacts.{name} must be beneath package data/")
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


def run_logged(command: list[str], *, cwd: Path | None = None) -> None:
    log("COMMAND", argv=" ".join(command))
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    status = process.wait()
    if status:
        raise RuntimeError(f"command exited with status {status}: {command[0]}")


def first_frame_checksums(path: Path, frames: int) -> list[str]:
    """Hash decoded RGB frames as FlashVSR reads them, by index rather than time."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frames),
        "-pix_fmt",
        "rgb24",
        "-f",
        "framemd5",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"frame checksum failed for {path}: {result.stderr}")
    checksums = [
        line.rsplit(",", 1)[-1].strip()
        for line in result.stdout.splitlines()
        if line and not line.startswith("#")
    ]
    if len(checksums) != frames:
        raise RuntimeError(
            f"frame checksum count mismatch for {path}: {len(checksums)} != {frames}"
        )
    return checksums


def upscale_input(
    outpaint_4k: Path,
    stage: Path,
    video_id: str,
    upscale_frames: int,
    outpaint_frames: int,
    *,
    resume: bool,
) -> Path:
    if upscale_frames == outpaint_frames:
        return outpaint_4k
    clipped = (
        stage
        / "segments"
        / "input"
        / f"{video_id}_Outpaint_4K_16fps_{upscale_frames}f.mp4"
    )
    manifest_path = stage / "metadata" / "upscale_input.json"
    source_stat = outpaint_4k.stat()
    identity = {
        "path": str(outpaint_4k),
        "size_bytes": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "selected_frame_range": [0, upscale_frames],
    }
    if clipped.exists():
        if not resume:
            raise RuntimeError("existing FlashVSR input clip requires --resume")
        if not manifest_path.is_file():
            orphan_dir = stage / "metadata" / "orphaned_inputs"
            orphan_dir.mkdir(parents=True, exist_ok=True)
            orphan = (
                orphan_dir
                / f"{clipped.stem}.{dt.datetime.now(dt.timezone.utc).timestamp():.6f}.mp4"
            )
            os.replace(clipped, orphan)
            log("UPSCALE_INPUT_ORPHAN_ARCHIVED", path=orphan)
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source") != identity:
                raise RuntimeError(
                    "existing FlashVSR input clip has no matching resume manifest"
                )
            verify_video(
                clipped, width=3840, height=2160, fps=16, frames=upscale_frames
            )
            checksums = first_frame_checksums(clipped, upscale_frames)
            if checksums != manifest.get(
                "decoded_frame_md5"
            ) or checksums != first_frame_checksums(outpaint_4k, upscale_frames):
                raise RuntimeError(
                    "resumed FlashVSR input clip is not pixel-identical to the first source frames"
                )
            return clipped
    temporary = clipped.with_name(f".{clipped.stem}.{os.getpid()}.partial.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-xerror",
        "-i",
        str(outpaint_4k),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"trim=end_frame={upscale_frames},setpts=N/(16*TB)",
        "-fps_mode",
        "passthrough",
        "-c:v",
        "libx265",
        "-x265-params",
        "lossless=1",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p10le",
        "-y",
        str(temporary),
    ]
    try:
        run_logged(command)
        probe = verify_video(
            temporary, width=3840, height=2160, fps=16, frames=upscale_frames
        )
        source_checksums = first_frame_checksums(outpaint_4k, upscale_frames)
        clip_checksums = first_frame_checksums(temporary, upscale_frames)
        if source_checksums != clip_checksums:
            raise RuntimeError("FlashVSR input clip changed decoded RGB pixel values")
        os.replace(temporary, clipped)
        atomic_json(
            manifest_path,
            {
                "source": identity,
                "clip": str(clipped),
                "probe": probe,
                "decoded_frame_md5": clip_checksums,
                "frame_policy": "decoded source frames [0,N), lossless HEVC, no fps filter/interpolation",
            },
        )
    finally:
        temporary.unlink(missing_ok=True)
    return clipped


def _run_flash_native(
    config: dict[str, Any],
    source_8k: Path,
    outpaint_4k: Path,
    run: Path,
    frame_count: int,
    outpaint_frames: int,
    runtime_config: Path,
    model_root: Path,
    *,
    resume: bool,
) -> Path:
    stage = artifact_path(config, "upscale")
    video_id = config["input"]["video_id"]
    up = config["upscale"]
    selected = upscale_input(
        outpaint_4k, stage, video_id, frame_count, outpaint_frames, resume=resume
    )
    raw = stage / "raw_flashvsr" / f"{video_id}_FlashVSR_raw_16K_16fps.mp4"
    final = stage / "final" / config["final"]["filename"]
    manifest = stage / "metadata" / "final_manifest.json"
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
        str(config["seed"]),
        "--gpu-index",
        str(up["gpu_index"]),
        "--ffmpeg-threads",
        str(up["ffmpeg_threads"]),
        "--encoder",
        str(up["encoder"]),
        "--output-mode",
        up.get("output_mode", "final_only"),
    ]
    if resume and "output_mode" not in up:
        command.append("--allow-existing-legacy")
    with stage_log(stage / "logs" / "flashvsr.log"):
        log(
            "UPSCALE",
            video=video_id,
            seed=config["seed"],
            frames=frame_count,
            source=selected,
            model="FlashVSR_v1.1",
            scale=4,
            seed_mode="per_tile_generator_plus_global_rng",
            output_mode=up.get("output_mode", "final_only"),
        )
        run_logged(command, cwd=ROOT)
        output_mode = up.get("output_mode", "final_only")
        final_record = json.loads(manifest.read_text(encoding="utf-8"))
        actual_mode = final_record.get("encoding", {}).get("output_mode", "legacy_raw")
        if actual_mode != output_mode:
            if not (resume and "output_mode" not in up
                    and output_mode == "final_only" and actual_mode == "legacy_raw"):
                raise RuntimeError("upscale output mode differs from requested mode")
            log("UPSCALE_REUSE_LEGACY", video=video_id, output=final)
        if actual_mode == "legacy_raw":
            final_probe = verify_video(
                final, width=15360, height=8640, fps=16, frames=frame_count
            )
            raw_probe = verify_video(
                raw, width=15360, height=8640, fps=16, frames=frame_count
            )
            atomic_json(stage / "metadata" / "ffprobe_raw.json", raw_probe)
        else:
            final_probe = verify_video_headers(
                final, width=15360, height=8640, fps=16, frames=frame_count
            )
            if (final_record.get("status") != "PASS"
                    or final_record.get("encoding", {}).get("output_mode") != "final_only"
                    or final_record.get("encoding", {}).get("frames_written") != frame_count):
                raise RuntimeError("final-only manifest/encoder frame count mismatch")
        atomic_json(stage / "metadata" / "ffprobe_final.json", final_probe)
        atomic_json(
            stage / "metadata" / "flashvsr_config.json",
            {**up, "requested_output_mode": output_mode, "actual_output_mode": actual_mode},
        )
        stage_manifest = stage / "segments" / "flashvsr_stage_manifest.json"
        if not stage_manifest.is_file():
            raise RuntimeError("FlashVSR native stage manifest missing")
        stage_record = json.loads(stage_manifest.read_text(encoding="utf-8"))
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
            run_logged(
                ["ffmpeg", "-v", "error", "-i", str(final), "-f", "null", "-"], cwd=ROOT
            )
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


def validate_artifact_layout(config: dict[str, Any]) -> None:
    """Prevent layout options/--force from aliasing source or upstream artifacts."""
    video = config["input"]["video_id"]
    job = config["job_name"]
    names = ("preprocess", "preprocess_metadata", "outpaint", "upscale")
    paths = {
        name: absolute_package_path(config["artifacts"][name], must_exist=False)
        for name in names
    }
    if job in (f"{video}_certification", f"{video}_quick"):
        suffix = "" if job.endswith("_certification") else "/quick"
        expected = {
            "preprocess": f"data/01_PreProc_2K_1920x1080/{video}_PreProc_2K.mp4",
            "preprocess_metadata": f"data/01_PreProc_2K_1920x1080/metadata/{video}",
            "outpaint": f"data/02_Outpaint_4K_3840x2160/{video}{suffix}",
            "upscale": f"data/03_Upscale_16K_15360x8640/{video}{suffix}",
        }
        if any(paths[name] != ROOT / expected[name] for name in names):
            raise ConfigError("Canonical stage layout does not match video/job")
    elif job == f"{video}_legacy_raw":
        expected = {
            "preprocess": f"data/01_PreProc_2K_1920x1080/{video}_PreProc_2K.mp4",
            "preprocess_metadata": f"data/01_PreProc_2K_1920x1080/metadata/{video}",
            "outpaint": f"data/02_Outpaint_4K_3840x2160/{video}",
            "upscale": f"data/03_Upscale_16K_15360x8640/{video}/legacy_raw",
        }
        if any(paths[name] != ROOT / expected[name] for name in names):
            raise ConfigError("Legacy raw stage must be isolated from canonical outputs")
        if config["upscale"].get("output_mode") != "legacy_raw":
            raise ConfigError("Legacy raw job requires upscale.output_mode=legacy_raw")
    elif job in (
        f"{video}_recurrence_baseline", f"{video}_recurrence_c0", f"{video}_recurrence_c1"
    ):
        canonical = ROOT / f"data/01_PreProc_2K_1920x1080/{video}_PreProc_2K.mp4"
        if paths["preprocess"] != canonical:
            raise ConfigError("Recurrence validation must use the canonical PreProc")
        validation_root = ROOT / "data/validation/recurrence_lineart" / job
        if any(not paths[name].is_relative_to(validation_root)
               for name in ("preprocess_metadata", "outpaint", "upscale")):
            raise ConfigError("Recurrence validation outputs must be isolated")
        for a in names:
            for b in names:
                if a != b and (paths[a] == paths[b] or paths[a].is_relative_to(paths[b])):
                    raise ConfigError("Recurrence validation artifact paths overlap")
    elif job in (f"{video}_validation", f"{video}_preprocess_reproduction"):
        if any(
            not path.is_relative_to(ROOT / "data/validation") for path in paths.values()
        ):
            raise ConfigError("Validation artifacts must remain under data/validation")
        for a in names:
            for b in names:
                if a != b and (
                    paths[a] == paths[b] or paths[a].is_relative_to(paths[b])
                ):
                    raise ConfigError("Validation artifact paths overlap")
    else:
        raise ConfigError(
            "Unsupported job name; use certification, quick, validation or preprocess_reproduction"
        )
    if (
        absolute_package_path(config["output_root"], must_exist=False)
        != ROOT / "state/runs" / job
    ):
        raise ConfigError("Run snapshot path must match job_name")


def artifact_path(config: dict[str, Any], name: str) -> Path:
    return absolute_package_path(config["artifacts"][name], must_exist=False)


def preprocess_output_path(config: dict[str, Any]) -> Path:
    """Keep imported canonical references immutable, including on future Full runs."""
    declared = artifact_path(config, "preprocess")
    video = config["input"]["video_id"]
    reference = ROOT / f"data/01_PreProc_2K_1920x1080/{video}_PreProc_2K.mp4"
    if declared == reference:
        return (
            artifact_path(config, "preprocess_metadata") / "generated" / declared.name
        )
    return declared  # Validation outputs are already isolated by config validation.


def preprocessed_input_path(config: dict[str, Any]) -> Path:
    """Use the latest explicit Stage0001 result, or the validated imported cache."""
    generated = preprocess_output_path(config)
    return generated if generated.is_file() else artifact_path(config, "preprocess")


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
        keys += ["upscale", "final"]
    if "validation" in config:
        keys += ["validation"]
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
    # Quick is nested below the ID directory but is a separate artifact identity.
    return [
        p
        for sub in stage_directories(name)
        for p in (stage / sub).rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    ]


def prepare_stage(
    stage: Path, config: dict[str, Any], name: str, *, resume: bool, force: bool
) -> None:
    request = stage / "metadata" / "stage_request.json"
    identity = stage_identity(config, name)
    meaningful = stage_artifact_files(stage, name)
    if meaningful and force:
        archive = (
            stage.parent
            / ".archive"
            / f"{stage.name}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        )
        archive.mkdir(parents=True, exist_ok=True)
        # Only a new-workspace stage can reach this branch; reference paths are rejected.
        for sub in stage_directories(name):
            if (stage / sub).exists():
                os.replace(stage / sub, archive / sub)
        log("ARCHIVE", stage=name, path=archive)
    elif meaningful:
        if not resume:
            raise ConfigError(f"{stage} contains artifacts: use --resume or --force")
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
    path = preprocessed_input_path(config)
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
    source = absolute_package_path(config["input"]["path"])
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


def run_preprocess_stage(
    config: dict[str, Any], *, resume: bool = False, force: bool = False
) -> Path:
    """The only Stage 0001 path used by Full and --stage preprocess (CPU only)."""
    source = absolute_package_path(config["input"]["path"])
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
            # Imported historical files are preserved, but fresh Full always regenerates.
            # A CRF1 draft receipt or changed producer/toolchain is never a cache hit.
            if (
                previous.get("producer") == "run_preprocess_stage"
                and previous.get("preprocess_config") == config["preprocess"]
                and previous.get("source_identity") == source_identity
                and previous.get("producer_identity") == producer
            ):
                _, frames, _ = validate_preprocessed(config)
                if frames != source_frames:
                    raise ConfigError("Preprocess cache/source frame count mismatch")
                log("PREPROCESS_REUSE", path=destination, frames=frames, status="PASS")
                return destination
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        work = metadata / "runs" / stamp
        # Fresh Full ALWAYS calls the audited Stage 0001, even if canonical exists.
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
        log(
            "PREPROCESS_STAGE_DONE",
            output=destination,
            frames=source_frames,
            status="PASS",
        )
        return destination


def run_outpaint_stage(
    config: dict[str, Any], *, resume: bool = False, force: bool = False
) -> Path:
    """One shared Outpainting stage; never runs preprocessing or FlashVSR."""
    with stage_lock("outpaint"), stage_lock(f"video_{config['input']['video_id']}"):
        preprocessed, available, input_sha = validate_preprocessed(config)
        frames = resolve_limit(config["frames"]["outpaint"], available, "outpaint")
        upscale_frames = resolve_limit(config["frames"]["upscale"], frames, "upscale")
        if upscale_frames < 21:
            raise ConfigError("FlashVSR input must contain at least 21 real frames")
        snapshot_resolved_frames(config, available, "outpaint")
        stage = artifact_path(config, "outpaint")
        prepare_stage(stage, config, "outpaint", resume=resume, force=force)
        prior_input = stage / "metadata" / "input_identity.json"
        identity = {"sha256": input_sha, "frames": available, "path": str(preprocessed)}
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

            result = run_outpaint(ROOT, config, preprocessed, stage, frames, resume)
            atomic_json(
                stage / "metadata" / "stage_complete.json",
                {
                    "status": "PASS",
                    "request": stage_identity(config, "outpaint"),
                    "input": identity,
                    "frames": frames,
                    "output": str(result),
                    "output_sha256": sha256_file(result),
                    "long_range_domain_fps": 24,
                    "output_fps": 24 if config.get("validation", {}).get("native_outpaint_only", False) else 16,
                    "output_role": "native_24fps_validation_master" if config.get("validation", {}).get("native_outpaint_only", False) else "final_16fps_deliverable",
                },
            )
            log(
                "OUTPAINT_DONE",
                video=config["input"]["video_id"],
                frames=f"{frames}/{frames}",
                output=result,
                status="PASS",
            )
        return result


def validate_outpaint_artifact(config: dict[str, Any]) -> tuple[Path, int]:
    if config.get("validation", {}).get("native_outpaint_only", False):
        raise ConfigError("Native24fps validation masters are not FlashVSR deliverables")
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
    config: dict[str, Any], *, resume: bool = False, force: bool = False
) -> Path:
    """One shared Flash/center/audio stage. Auto tiling and causal loop are untouched."""
    with stage_lock("upscale"), stage_lock(f"video_{config['input']['video_id']}"):
        outpainted, outpaint_frames = validate_outpaint_artifact(config)
        frames = resolve_limit(config["frames"]["upscale"], outpaint_frames, "upscale")
        if frames < 21:
            raise ConfigError("FlashVSR requires at least 21 real frames")
        stage = artifact_path(config, "upscale")
        prepare_stage(stage, config, "upscale", resume=resume, force=force)
        return _run_flash_native(
            config,
            absolute_package_path(config["input"]["path"]),
            outpainted,
            ROOT,
            frames,
            outpaint_frames,
            absolute_package_path(config["outpainting"]["runtime_config"]),
            absolute_package_path(config["upscale"]["model"]),
            resume=resume,
        )


def execute_stages(
    config: dict[str, Any], stage: str, *, resume: bool = False, force: bool = False
) -> Path:
    """Selection only: all interfaces dispatch to these SAME stage callables."""
    if config.get("validation", {}).get("native_outpaint_only", False) and stage != "outpaint":
        raise ConfigError("Native-only recurrence validation must use --stage outpaint")
    if config.get("job_name", "").endswith("_legacy_raw") and stage != "upscale":
        raise ConfigError("Legacy raw job reuses the canonical Outpainting; select --stage upscale")
    result = None
    if stage in ("all", "preprocess"):
        result = run_preprocess_stage(config, resume=resume, force=force)
    if stage in ("all", "outpaint"):
        result = run_outpaint_stage(config, resume=resume, force=force)
    if stage in ("all", "upscale"):
        result = run_upscale_stage(config, resume=resume, force=force)
    if result is None:
        raise ConfigError(f"Unknown stage: {stage}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="8K to 16K: shared Full / Quick / stage-only execution"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("preprocess", "outpaint", "upscale"), default="all"
    )
    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument("--resume", action="store_true")
    exclusive.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    config_path = args.config.expanduser().resolve()
    config, _, run, _, _ = validate_config(config_path)
    run.mkdir(parents=True, exist_ok=True)
    requested = run / "requested_config.json"
    if requested.exists() and requested.read_bytes() != config_path.read_bytes():
        if not args.force:
            raise ConfigError(
                "Config bytes differ from existing requested_config.json; use a separate job or --force"
            )
        archived = run / (
            "requested_config."
            + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            + ".json"
        )
        atomic_copy(requested, archived)
    if not args.resume and not args.force:
        for name in ("outpaint", "upscale"):
            if args.stage in ("all", name):
                target = artifact_path(config, name)
                if stage_artifact_files(target, name):
                    raise ConfigError(
                        f"{name} target contains artifacts; use --resume or --force"
                    )
    atomic_copy(config_path, requested)
    with stage_log(ROOT / "logs" / f"{config['job_name']}.log"):
        log(
            "CONFIG",
            job=config["job_name"],
            stage=args.stage,
            seed=config["seed"],
            config=config_path,
        )
        result = execute_stages(
            config, args.stage, resume=args.resume, force=args.force
        )
        resolved = json.loads(json.dumps(config))
        resolved["upscale"]["output_mode"] = config["upscale"].get(
            "output_mode", "final_only"
        )
        resolved["executed_stage"] = args.stage
        selected_preproc = preprocessed_input_path(config)
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
                up = resolve_limit(config["frames"]["upscale"], out, "upscale")
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
