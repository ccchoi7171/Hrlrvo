#!/usr/bin/env python3
"""One JSON-driven entrypoint for certification and short inference runs.

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
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

preprocess = importlib.import_module("cert_pipeline.preprocess")

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
    for key, expected in {
        "source_fps": 16,
        "processing_fps": 24,
        "source_resolution": [7680, 4320],
        "output_resolution": [1920, 1080],
        "resize": "lanczos",
        "video_codec": "libx265",
        "pixel_format": "yuv420p10le",
        "crf": 1,
        "audio_tempo": 1.5,
    }.items():
        require_equal(pre.get(key), expected, f"preprocess.{key}")
    out = config.get("outpainting", {})
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
    if not output_root.is_relative_to(ROOT / "data"):
        raise ConfigError("output_root must be beneath package data/")
    expected_output_suffix = f"video_{video_id}_seed{SEEDS[video_id]}"
    if output_root.name != expected_output_suffix or output_root.parent.name not in (
        "certification",
        "quick",
    ):
        raise ConfigError("output_root does not match the configured video ID/seed")
    require_equal(
        config.get("job_name"), f"{video_id}_{output_root.parent.name}", "job_name"
    )
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
    probe = preprocess.probe_media(source, ffprobe)
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


def make_layout(run: Path) -> None:
    folders = (
        "config",
        "logs",
        "0001_Preprocess_FHD24/output",
        "0001_Preprocess_FHD24/metadata",
        "0001_Preprocess_FHD24/logs",
        "0002_Outpaint_4K16/final",
        "0002_Outpaint_4K16/chunks/mp4",
        "0002_Outpaint_4K16/chunks/latents",
        "0002_Outpaint_4K16/chunks/native",
        "0002_Outpaint_4K16/chunks/manifests",
        "0002_Outpaint_4K16/metadata",
        "0002_Outpaint_4K16/logs",
        "0003_Upscale_FlashVSR_16K/segments/input",
        "0003_Upscale_FlashVSR_16K/segments/tiles",
        "0003_Upscale_FlashVSR_16K/segments/metadata",
        "0003_Upscale_FlashVSR_16K/raw_flashvsr",
        "0003_Upscale_FlashVSR_16K/final",
        "0003_Upscale_FlashVSR_16K/metadata",
        "0003_Upscale_FlashVSR_16K/logs",
    )
    for folder in folders:
        (run / folder).mkdir(parents=True, exist_ok=True)


def existing_artifacts(run: Path) -> list[Path]:
    if not run.exists():
        return []
    return [
        path for path in run.rglob("*") if path.is_file() and path.name != ".gitkeep"
    ]


def prepare_run(run: Path, *, resume: bool, force: bool) -> None:
    previous = existing_artifacts(run)
    if previous and force:
        archive_root = run.parent / ".archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = archive_root / f"{run.name}_{stamp}"
        os.replace(run, archived)
        print(f"[ARCHIVE] previous run retained at {archived}", flush=True)
    elif previous and not resume:
        raise ConfigError(
            f"run already contains artifacts; use --resume or --force: {run}"
        )
    make_layout(run)


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
            orphan = orphan_dir / f"{clipped.stem}.{dt.datetime.now(dt.timezone.utc).timestamp():.6f}.mp4"
            os.replace(clipped, orphan)
            log("UPSCALE_INPUT_ORPHAN_ARCHIVED", path=orphan)
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source") != identity:
                raise RuntimeError(
                    "existing FlashVSR input clip has no matching resume manifest"
                )
            verify_video(clipped, width=3840, height=2160, fps=16, frames=upscale_frames)
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


def run_flash(
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
    stage = run / "0003_Upscale_FlashVSR_16K"
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
    flash_python = os.environ.get("CERT_FLASH_PYTHON", sys.executable)
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
    ]
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
        )
        run_logged(command, cwd=ROOT)
        final_probe = verify_video(
            final, width=15360, height=8640, fps=16, frames=frame_count
        )
        raw_probe = verify_video(
            raw, width=15360, height=8640, fps=16, frames=frame_count
        )
        atomic_json(stage / "metadata" / "ffprobe_final.json", final_probe)
        atomic_json(stage / "metadata" / "ffprobe_raw.json", raw_probe)
        atomic_json(stage / "metadata" / "flashvsr_config.json", up)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Self-contained JSON-driven certification inference"
    )
    parser.add_argument("--config", type=Path, required=True)
    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument("--resume", action="store_true")
    exclusive.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    config_path = args.config.expanduser().resolve()
    # Every validation below precedes the first PyTorch/model import.
    config, source, run, runtime_config, model_root = validate_config(config_path)
    source_frames = preflight_source(source)
    outpaint_frames = resolve_limit(
        config["frames"]["outpaint"], source_frames, "outpaint"
    )
    upscale_frames = resolve_limit(
        config["frames"]["upscale"], outpaint_frames, "upscale"
    )
    if upscale_frames < 21:
        raise ConfigError("FlashVSR TinyLong requires at least 21 real input frames")
    prepare_run(run, resume=args.resume, force=args.force)
    requested_path = run / "config" / "requested_config.json"
    if (
        args.resume
        and requested_path.exists()
        and requested_path.read_bytes() != config_path.read_bytes()
    ):
        raise ConfigError(
            "resume config bytes differ from the original requested_config.json"
        )
    atomic_copy(config_path, requested_path)
    resolved = json.loads(json.dumps(config))
    resolved["frames"] = {
        "source": source_frames,
        "outpaint": outpaint_frames,
        "upscale": upscale_frames,
        "final": upscale_frames,
    }
    resolved["resolved_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(run / "config" / "resolved_config.json", resolved)
    with (
        (run / "logs" / "pipeline.log").open("a", encoding="utf-8") as stream,
        contextlib.redirect_stdout(Tee(sys.stdout, stream)),
        contextlib.redirect_stderr(Tee(sys.stderr, stream)),
    ):
        log(
            "PREFLIGHT",
            job=config["job_name"],
            video=config["input"]["video_id"],
            seed=config["seed"],
            input=source.name,
            resolution="7680x4320",
            fps=16,
            frames=source_frames,
            config=config_path,
            status="PASS",
        )
        seed_all(config["seed"])
        pre_stage = run / "0001_Preprocess_FHD24"
        # The preprocessing module mirrors its own FFmpeg output to this
        # stage file; the run-level Tee captures the same terminal stream.
        preprocessed = preprocess.run_preprocess(
            source,
            pre_stage / "output",
            pre_stage / "metadata",
            pre_stage / "logs" / "preprocess.log",
            source_frames,
            resume=args.resume,
        )
        from cert_pipeline.outpaint import run_outpaint

        out_stage = run / "0002_Outpaint_4K16"
        with stage_log(out_stage / "logs" / "outpaint.log"):
            log(
                "OUTPAINT",
                video=config["input"]["video_id"],
                seed=config["seed"],
                frames=outpaint_frames,
                steps=30,
                status="START",
            )
            outpainted = run_outpaint(
                ROOT, config, preprocessed, out_stage, outpaint_frames, args.resume
            )
            log(
                "OUTPAINT_DONE",
                video=config["input"]["video_id"],
                frames=f"{outpaint_frames}/{outpaint_frames}",
                status="PASS",
                output=outpainted,
            )
        final = run_flash(
            config,
            source,
            outpainted,
            run,
            upscale_frames,
            outpaint_frames,
            runtime_config,
            model_root,
            resume=args.resume,
        )
        log(
            "PIPELINE_DONE",
            job=config["job_name"],
            video=config["input"]["video_id"],
            seed=config["seed"],
            frames=upscale_frames,
            output=final,
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
