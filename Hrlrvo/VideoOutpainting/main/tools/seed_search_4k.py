#!/usr/bin/env python3
"""Run resumable 65-frame 4K step-10 predicted-clean seed preview search.

Each job keeps the native 30-step LTX schedule, evaluates through model step 10,
decodes predicted-clean frames 0/32/64, and saves three PNGs.  The scheduler
runs two seeds for one video concurrently, waits for both, then advances in
01 -> 02 -> 03 -> 04 -> 05 round-robin order.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

MAIN = Path(__file__).resolve().parents[1]
PROJECT = MAIN.parent
WORKSPACE = PROJECT.parent
PIPELINE = WORKSPACE / "pipeline"

DEFAULT_CONFIG = MAIN / "configs/inference_4k.yaml"
DEFAULT_CHECKPOINT = PROJECT / "runs/hrlrvo_main_train_v1/checkpoints/checkpoint-007500"
DEFAULT_INPUT_ROOT = PIPELINE / "data/01_PreProc_2K_1920x1080"
DEFAULT_OUTPUT_ROOT = PIPELINE / "data/02_seedSearch_4K_3840x2160"
DEFAULT_NEGATIVE_PROMPT = (
    "people, person, human, staff, audience, crowd, backstage, seats"
)
DEFAULT_VIDEOS = ("01", "02", "03", "04", "05")

OUTPUT_WIDTH = 3840
OUTPUT_HEIGHT = 2160
INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080
FPS = Fraction(24, 1)
FRAME_COUNT = 65
FULL_SCHEDULE_STEPS = 30
PREVIEW_STEP = 10
PREVIEW_FRAMES = (0, 32, 64)
SEED_MIN = 10_000_000
SEED_MAX = 99_999_999

MANIFEST_FIELDS = (
    "event_time",
    "video_id",
    "input_path",
    "seed",
    "checkpoint_path",
    "negative_prompt",
    "preview_step",
    "full_schedule_steps",
    "preview_frames",
    "output_preview_f000",
    "output_preview_f032",
    "output_preview_f064",
    "run_dir",
    "start_time",
    "end_time",
    "status",
    "exit_code",
    "round_index",
    "batch_index",
    "pid",
    "num_frames",
    "resolution",
    "config_path",
    "config_sha256",
    "context_fingerprint",
    "log_path",
    "message",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def emit(root: Path, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    log = root / "logs/scheduler.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def input_for(input_root: Path, video_id: str) -> Path:
    if video_id not in DEFAULT_VIDEOS:
        raise ValueError(f"Unsupported video ID {video_id!r}")
    return input_root / f"{video_id}_PreProc_2K.mp4"


def output_paths_for(
    output_root: Path, video_id: str, seed: int
) -> dict[int, Path]:
    return {
        frame: output_root
        / video_id
        / f"{video_id}_{seed}_step{PREVIEW_STEP}_f{frame:03d}.png"
        for frame in PREVIEW_FRAMES
    }


def output_for(output_root: Path, video_id: str, seed: int, frame: int) -> Path:
    return output_paths_for(output_root, video_id, seed)[frame]


def run_parent_for(output_root: Path, video_id: str, seed: int) -> Path:
    return output_root / "_runs" / f"{video_id}_{seed}"


def log_for(output_root: Path, video_id: str, seed: int) -> Path:
    return output_root / "logs" / f"{video_id}_{seed}.log"


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one selected video stream in {path}")
    stream = streams[0]
    count_text = stream.get("nb_frames")
    if not count_text or count_text == "N/A":
        raise ValueError(f"Could not determine frame count for {path}")
    return {
        "codec_name": stream.get("codec_name"),
        "resolution": [int(stream["width"]), int(stream["height"])],
        "fps": stream["avg_frame_rate"],
        "r_frame_rate": stream.get("r_frame_rate"),
        "frame_count": int(count_text),
        "duration_seconds": float(
            stream.get("duration") or payload.get("format", {}).get("duration") or 0
        ),
        "size_bytes": int(payload.get("format", {}).get("size") or path.stat().st_size),
    }


def validate_input(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    info = probe_video(path)
    if info["resolution"] != [INPUT_WIDTH, INPUT_HEIGHT]:
        raise ValueError(f"{path}: expected 1920x1080, got {info['resolution']}")
    if Fraction(info["fps"]) != FPS:
        raise ValueError(f"{path}: expected 24 fps, got {info['fps']}")
    if info["frame_count"] < FRAME_COUNT:
        raise ValueError(f"{path}: requires at least {FRAME_COUNT} frames")
    return info


def validate_preview_png(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        resolution = [int(image.width), int(image.height)]
        mode = image.mode
        image_format = image.format
    if resolution != [OUTPUT_WIDTH, OUTPUT_HEIGHT]:
        raise ValueError(f"{path}: expected 3840x2160, got {resolution}")
    if mode != "RGB" or image_format != "PNG":
        raise ValueError(f"{path}: expected RGB PNG, got {mode} {image_format}")
    return {
        "resolution": resolution,
        "mode": mode,
        "format": image_format,
        "size_bytes": path.stat().st_size,
    }


def validate_config(config: Path) -> tuple[dict[str, Any], str]:
    if not config.is_file():
        raise FileNotFoundError(config)
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise TypeError(f"Config is not a mapping: {config}")
    if settings.get("video", {}).get("canvas") != [OUTPUT_WIDTH, OUTPUT_HEIGHT]:
        raise ValueError("Inference config canvas must be 3840x2160")
    if Fraction(str(settings.get("video", {}).get("fps"))) != FPS:
        raise ValueError("Inference config FPS must be 24")
    if int(settings.get("long_range", {}).get("clip_frames", -1)) != FRAME_COUNT:
        raise ValueError("Inference config long_range.clip_frames must be 65")
    if int(settings.get("sampling", {}).get("inference_steps", -1)) != FULL_SCHEDULE_STEPS:
        raise ValueError("Inference config sampling.inference_steps must be 30")
    if int(settings.get("architecture", {}).get("architecture_version", -1)) != 8:
        raise ValueError("Expected architecture_version 8")
    if settings.get("sampling", {}).get("negative_prompt", "") != "":
        raise ValueError("YAML negative prompt must stay empty; runtime CLI owns it")
    return settings, file_sha256(config)


def validate_checkpoint(
    checkpoint: Path, settings: dict[str, Any]
) -> tuple[dict[str, Any], Path, str]:
    if checkpoint.name != "checkpoint-007500" or not checkpoint.is_dir():
        raise ValueError(f"A real checkpoint-007500 directory is required: {checkpoint}")
    adapter = checkpoint / "adapter_model.safetensors"
    manifest_path = checkpoint / "outpainting_manifest.json"
    if not adapter.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete outpainting checkpoint: {checkpoint}")
    manifest = read_json(manifest_path)
    if int(manifest.get("global_step", -1)) != 7500:
        raise ValueError(f"Checkpoint manifest is not global_step 7500: {checkpoint}")
    expected_arch = int(settings["architecture"]["architecture_version"])
    if int(manifest.get("architecture_version", -1)) != expected_arch:
        raise ValueError("Checkpoint/config architecture versions differ")
    if manifest.get("base_model_path") != settings.get("model", {}).get(
        "base_model_path"
    ):
        raise ValueError("Checkpoint/config base model paths differ")
    outpainting = manifest.get("outpainting", {})
    if outpainting.get("clip_frames") != FRAME_COUNT:
        raise ValueError("Checkpoint was not trained with 65-frame clips")
    if outpainting.get("canvas_pixels") != [OUTPUT_HEIGHT, OUTPUT_WIDTH]:
        raise ValueError("Checkpoint canvas is not 3840x2160")
    if outpainting.get("anchor_size_pixels") != [INPUT_HEIGHT, INPUT_WIDTH]:
        raise ValueError("Checkpoint anchor is not 1920x1080")
    return manifest, adapter, file_sha256(adapter)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps != FULL_SCHEDULE_STEPS:
        raise ValueError("Seed search must keep the full 30-step schedule")
    if args.preview_step != PREVIEW_STEP:
        raise ValueError("Seed search preview step must be 10")
    if tuple(args.preview_frames) != PREVIEW_FRAMES:
        raise ValueError("Seed search preview frames must be exactly 0,32,64")
    if args.negative_prompt != DEFAULT_NEGATIVE_PROMPT:
        raise ValueError("Seed search negative prompt must match the production string exactly")

    config = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    settings, config_sha = validate_config(config)
    checkpoint_manifest, adapter, adapter_sha = validate_checkpoint(
        checkpoint, settings
    )

    inputs: dict[str, Any] = {}
    for video_id in DEFAULT_VIDEOS:
        path = input_for(input_root, video_id).resolve()
        info = validate_input(path)
        inputs[video_id] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": file_sha256(path),
            "probe": info,
        }

    generation = {
        "schema_version": 2,
        "semantic": "seed_search_step10_predicted_clean_preview",
        "mode": "normal_long_range",
        "num_frames": FRAME_COUNT,
        "output_resolution": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
        "fps": 24,
        "full_schedule_steps": FULL_SCHEDULE_STEPS,
        "preview_step": PREVIEW_STEP,
        "preview_loop_index_zero_based": PREVIEW_STEP - 1,
        "preview_frames": list(PREVIEW_FRAMES),
        "preview_latent_source": (
            "guided_predicted_clean_after_cfg_stg_and_conditioning_before_euler"
        ),
        "full_mp4_generated": False,
        "config_path": str(config),
        "config_sha256": config_sha,
        "checkpoint_path": str(checkpoint),
        "checkpoint_global_step": checkpoint_manifest["global_step"],
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_sha,
        "negative_prompt": args.negative_prompt,
        "positive_prompt": "",
        "memory_optimization": {
            "video_only_gpu_weights": True,
            "vae_decode_streaming": True,
            "ffn_token_chunking": False,
        },
        "partial_decode": {
            "random_access_supported": False,
            "temporal_traversal_required_through_frame": max(PREVIEW_FRAMES),
            "full_decoded_video_materialized": False,
            "retained_frames_only": True,
        },
        "python_executable": str(Path(sys.executable).resolve()),
        "conda_environment": Path(sys.prefix).name,
        "inputs": inputs,
    }
    generation["context_fingerprint"] = value_sha256(generation)

    output_root.mkdir(parents=True, exist_ok=True)
    for video_id in DEFAULT_VIDEOS:
        (output_root / video_id).mkdir(parents=True, exist_ok=True)
    (output_root / "_runs").mkdir(exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)

    context_path = output_root / "run_context.json"
    if context_path.exists():
        existing = read_json(context_path)
        if existing.get("generation") != generation:
            raise RuntimeError(
                "Existing output root has a different generation identity. "
                "Archive the previous seed-search state before changing semantics."
            )
    else:
        atomic_json(context_path, {"created_at": utc_now(), "generation": generation})
    return generation


def append_manifest(output_root: Path, values: dict[str, Any]) -> None:
    path = output_root / "seed_search_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        empty = handle.tell() == 0
        writer = csv.DictWriter(
            handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore"
        )
        if empty:
            writer.writeheader()
        row = {field: values.get(field, "") for field in MANIFEST_FIELDS}
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def metadata_matches(
    metadata: dict[str, Any], *, video_id: str, seed: int, generation: dict[str, Any]
) -> bool:
    expected = {
        "status": "success",
        "semantic": generation["semantic"],
        "video_id": video_id,
        "seed": seed,
        "input_path": generation["inputs"][video_id]["path"],
        "negative_prompt": generation["negative_prompt"],
        "checkpoint_path": generation["checkpoint_path"],
        "num_frames": FRAME_COUNT,
        "full_schedule_steps": FULL_SCHEDULE_STEPS,
        "preview_step": PREVIEW_STEP,
        "preview_frames": list(PREVIEW_FRAMES),
        "context_fingerprint": generation["context_fingerprint"],
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def completed_job(
    output_root: Path, video_id: str, seed: int, generation: dict[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]] | None:
    run_parent = run_parent_for(output_root, video_id, seed)
    metadata_path = run_parent / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path)
    if not metadata_matches(
        metadata, video_id=video_id, seed=seed, generation=generation
    ):
        return None
    probes: dict[int, dict[str, Any]] = {}
    for frame, output in output_paths_for(output_root, video_id, seed).items():
        try:
            probes[frame] = validate_preview_png(output)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if metadata.get(f"output_preview_f{frame:03d}") != str(output):
            return None
    return probes, metadata


def inference_environment() -> dict[str, str]:
    env = os.environ.copy()
    source = str(MAIN / "src")
    env["PYTHONPATH"] = source + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def native_command(
    *,
    args: argparse.Namespace,
    generation: dict[str, Any],
    video_id: str,
    seed: int,
    run_dir: Path,
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-m",
        "ltx_outpaint.inference",
        "--config",
        generation["config_path"],
        "--checkpoint",
        generation["checkpoint_path"],
        "--steps",
        str(FULL_SCHEDULE_STEPS),
        "--video-only-gpu-weights",
        "--vae-decode-streaming",
        "--no-ffn-token-chunking",
        "--anchor-video",
        generation["inputs"][video_id]["path"],
        "--mode",
        "normal_long_range",
        "--num-frames",
        str(FRAME_COUNT),
        "--seed",
        str(seed),
        "--negative-prompt",
        generation["negative_prompt"],
        "--seed-search-output-dir",
        str(run_dir),
        "--preview-step",
        str(PREVIEW_STEP),
        "--preview-frames",
        *(str(frame) for frame in PREVIEW_FRAMES),
    ]


def write_resolved_config(
    run_dir: Path,
    generation: dict[str, Any],
    *,
    video_id: str,
    seed: int,
) -> None:
    base = yaml.safe_load(Path(generation["config_path"]).read_text(encoding="utf-8"))
    value = {
        "base_config": base,
        "seed_search_runtime": {
            "semantic": generation["semantic"],
            "mode": generation["mode"],
            "video_id": video_id,
            "input_path": generation["inputs"][video_id]["path"],
            "seed": seed,
            "checkpoint_path": generation["checkpoint_path"],
            "negative_prompt": generation["negative_prompt"],
            "num_frames": FRAME_COUNT,
            "full_schedule_steps": FULL_SCHEDULE_STEPS,
            "preview_step": PREVIEW_STEP,
            "preview_frames": list(PREVIEW_FRAMES),
            "full_mp4_generated": False,
            "context_fingerprint": generation["context_fingerprint"],
        },
    }
    atomic_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
    )


@dataclass
class RunningJob:
    video_id: str
    seed: int
    round_index: int
    batch_index: int
    process: subprocess.Popen[str]
    log_handle: Any
    log_path: Path
    run_dir: Path
    output_paths: dict[int, Path]
    command: list[str]
    started_at: str
    started_monotonic: float


def base_manifest_row(
    *,
    video_id: str,
    seed: int,
    round_index: int,
    batch_index: int,
    output_root: Path,
    generation: dict[str, Any],
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    outputs = output_paths_for(output_root, video_id, seed)
    return {
        "event_time": utc_now(),
        "video_id": video_id,
        "input_path": generation["inputs"][video_id]["path"],
        "seed": seed,
        "checkpoint_path": generation["checkpoint_path"],
        "negative_prompt": generation["negative_prompt"],
        "preview_step": PREVIEW_STEP,
        "full_schedule_steps": FULL_SCHEDULE_STEPS,
        "preview_frames": json.dumps(list(PREVIEW_FRAMES)),
        "output_preview_f000": str(outputs[0]),
        "output_preview_f032": str(outputs[32]),
        "output_preview_f064": str(outputs[64]),
        "run_dir": str(run_dir),
        "round_index": round_index,
        "batch_index": batch_index,
        "num_frames": FRAME_COUNT,
        "resolution": f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        "config_path": generation["config_path"],
        "config_sha256": generation["config_sha256"],
        "context_fingerprint": generation["context_fingerprint"],
        "log_path": str(log_path),
    }


def launch_job(
    args: argparse.Namespace,
    generation: dict[str, Any],
    *,
    video_id: str,
    seed: int,
    round_index: int,
    batch_index: int,
) -> RunningJob | None:
    output_root = args.output_root.resolve()
    run_dir = run_parent_for(output_root, video_id, seed)
    log_path = log_for(output_root, video_id, seed)
    existing = completed_job(output_root, video_id, seed, generation)
    if existing is not None:
        _probes, metadata = existing
        row = base_manifest_row(
            video_id=video_id,
            seed=seed,
            round_index=round_index,
            batch_index=batch_index,
            output_root=output_root,
            generation=generation,
            run_dir=run_dir,
            log_path=log_path,
        )
        row.update(
            {
                "status": "skipped",
                "exit_code": 0,
                "start_time": metadata.get("start_time", ""),
                "end_time": utc_now(),
                "message": "validated existing step-10 f000/f032/f064 previews",
            }
        )
        append_manifest(output_root, row)
        emit(output_root, f"SKIP video={video_id} seed={seed}: valid 3-PNG preview")
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    command = native_command(
        args=args,
        generation=generation,
        video_id=video_id,
        seed=seed,
        run_dir=run_dir,
    )
    atomic_text(run_dir / "command.txt", shlex.join(command) + "\n")
    write_resolved_config(run_dir, generation, video_id=video_id, seed=seed)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    run_log = run_dir / "inference.log"
    if not run_log.exists():
        try:
            os.link(log_path, run_log)
        except OSError:
            run_log.symlink_to(log_path)
    started_at = utc_now()
    log_handle.write(
        "\n"
        + json.dumps(
            {
                "event": "launcher_start",
                "started_at": started_at,
                "video_id": video_id,
                "seed": seed,
                "run_dir": str(run_dir),
                "command": command,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=MAIN,
        env=inference_environment(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    atomic_json(
        run_dir / "job_state.json",
        {
            "status": "running",
            "video_id": video_id,
            "seed": seed,
            "pid": process.pid,
            "start_time": started_at,
            "run_dir": str(run_dir),
            "context_fingerprint": generation["context_fingerprint"],
            "command": command,
        },
    )
    row = base_manifest_row(
        video_id=video_id,
        seed=seed,
        round_index=round_index,
        batch_index=batch_index,
        output_root=output_root,
        generation=generation,
        run_dir=run_dir,
        log_path=log_path,
    )
    row.update(
        {
            "status": "running",
            "start_time": started_at,
            "pid": process.pid,
            "message": "native step-10 predicted-clean preview inference launched",
        }
    )
    append_manifest(output_root, row)
    emit(
        output_root,
        f"LAUNCHED video={video_id} seed={seed} pid={process.pid} "
        f"round={round_index} batch={batch_index}",
    )
    return RunningJob(
        video_id=video_id,
        seed=seed,
        round_index=round_index,
        batch_index=batch_index,
        process=process,
        log_handle=log_handle,
        log_path=log_path,
        run_dir=run_dir,
        output_paths=output_paths_for(output_root, video_id, seed),
        command=command,
        started_at=started_at,
        started_monotonic=time.monotonic(),
    )


def _validate_native_metadata(
    metadata: dict[str, Any], job: RunningJob, generation: dict[str, Any]
) -> None:
    expected = {
        "status": "success",
        "semantic": generation["semantic"],
        "input_path": generation["inputs"][job.video_id]["path"],
        "checkpoint_path": generation["checkpoint_path"],
        "negative_prompt": generation["negative_prompt"],
        "seed": job.seed,
        "mode": "normal_long_range",
        "num_frames": FRAME_COUNT,
        "full_schedule_steps": FULL_SCHEDULE_STEPS,
        "preview_step": PREVIEW_STEP,
        "preview_frames": list(PREVIEW_FRAMES),
        "full_mp4_generated": False,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Native preview metadata mismatch: {mismatches}")
    sigmas = metadata.get("sigma_schedule")
    if not isinstance(sigmas, list) or len(sigmas) != FULL_SCHEDULE_STEPS + 1:
        raise ValueError("Native metadata does not contain the full 30-step sigma schedule")


def finish_job(
    args: argparse.Namespace,
    generation: dict[str, Any],
    job: RunningJob,
) -> str:
    output_root = args.output_root.resolve()
    exit_code = job.process.wait()
    job.log_handle.close()
    ended_at = utc_now()
    elapsed = time.monotonic() - job.started_monotonic
    status = "fail"
    message = ""
    probes: dict[int, dict[str, Any]] = {}
    try:
        if exit_code != 0:
            raise RuntimeError(f"native inference exited {exit_code}")
        native_metadata_path = job.run_dir / "metadata.json"
        native_metadata = read_json(native_metadata_path)
        _validate_native_metadata(native_metadata, job, generation)

        for frame in PREVIEW_FRAMES:
            native_output = job.run_dir / f"f{frame:03d}.png"
            validate_preview_png(native_output)
            final_output = job.output_paths[frame]
            final_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = final_output.with_name(
                f".{final_output.name}.{os.getpid()}.tmp"
            )
            try:
                shutil.copy2(native_output, temporary)
                validate_preview_png(temporary)
                os.replace(temporary, final_output)
            finally:
                temporary.unlink(missing_ok=True)
            probes[frame] = validate_preview_png(final_output)

        metadata = {
            **native_metadata,
            "status": "success",
            "video_id": job.video_id,
            "input_sha256": generation["inputs"][job.video_id]["sha256"],
            "adapter_path": generation["adapter_path"],
            "adapter_sha256": generation["adapter_sha256"],
            "config_path": generation["config_path"],
            "config_sha256": generation["config_sha256"],
            "context_fingerprint": generation["context_fingerprint"],
            "output_preview_f000": str(job.output_paths[0]),
            "output_preview_f032": str(job.output_paths[32]),
            "output_preview_f064": str(job.output_paths[64]),
            "output_preview_sha256": {
                f"f{frame:03d}": file_sha256(job.output_paths[frame])
                for frame in PREVIEW_FRAMES
            },
            "run_dir": str(job.run_dir),
            "log_path": str(job.log_path),
            "command_path": str(job.run_dir / "command.txt"),
            "resolved_config_path": str(job.run_dir / "resolved_config.yaml"),
            "command": job.command,
            "pid": job.process.pid,
            "start_time": job.started_at,
            "end_time": ended_at,
            "elapsed_seconds": elapsed,
            "exit_code": exit_code,
            "round_index": job.round_index,
            "batch_index": job.batch_index,
            "final_output_probe": {
                f"f{frame:03d}": probes[frame] for frame in PREVIEW_FRAMES
            },
        }
        atomic_json(job.run_dir / "metadata.json", metadata)
        status = "success"
        message = "validated and committed step-10 f000/f032/f064 previews"
    except Exception as error:  # noqa: BLE001 - record failure and continue scheduling
        message = f"{type(error).__name__}: {error}"
        with job.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"event": "launcher_failure", "error": message}) + "\n"
            )

    atomic_json(
        job.run_dir / "job_state.json",
        {
            "status": status,
            "video_id": job.video_id,
            "seed": job.seed,
            "pid": job.process.pid,
            "start_time": job.started_at,
            "end_time": ended_at,
            "elapsed_seconds": elapsed,
            "exit_code": exit_code,
            "run_dir": str(job.run_dir),
            "context_fingerprint": generation["context_fingerprint"],
            "message": message,
        },
    )
    row = base_manifest_row(
        video_id=job.video_id,
        seed=job.seed,
        round_index=job.round_index,
        batch_index=job.batch_index,
        output_root=output_root,
        generation=generation,
        run_dir=job.run_dir,
        log_path=job.log_path,
    )
    row.update(
        {
            "status": status,
            "start_time": job.started_at,
            "end_time": ended_at,
            "exit_code": exit_code,
            "pid": job.process.pid,
            "message": message,
        }
    )
    append_manifest(output_root, row)
    emit(
        output_root,
        f"FINISHED video={job.video_id} seed={job.seed} status={status} "
        f"exit={exit_code} elapsed={elapsed:.1f}s",
    )
    return status


def _interrupt_jobs(
    jobs: list[RunningJob], args: argparse.Namespace, generation: dict[str, Any]
) -> None:
    for job in jobs:
        if job.process.poll() is None:
            job.process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 30
    for job in jobs:
        while job.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.25)
        if job.process.poll() is None:
            job.process.terminate()
    for job in jobs:
        if not job.log_handle.closed:
            finish_job(args, generation, job)


def run_batch(
    args: argparse.Namespace,
    generation: dict[str, Any],
    *,
    video_id: str,
    seeds: list[int],
    round_index: int,
    batch_index: int,
) -> dict[str, int]:
    if len(seeds) > args.max_concurrent:
        raise ValueError("A batch cannot exceed --max-concurrent")
    running: list[RunningJob] = []
    statuses: dict[str, int] = {"success": 0, "fail": 0, "skipped": 0}
    for seed in seeds:
        try:
            job = launch_job(
                args,
                generation,
                video_id=video_id,
                seed=seed,
                round_index=round_index,
                batch_index=batch_index,
            )
            if job is None:
                statuses["skipped"] += 1
            else:
                running.append(job)
        except Exception as error:  # noqa: BLE001 - one launch must not stop scheduler
            statuses["fail"] += 1
            log_path = log_for(args.output_root, video_id, seed)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            message = f"launch failure: {type(error).__name__}: {error}"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
            row = base_manifest_row(
                video_id=video_id,
                seed=seed,
                round_index=round_index,
                batch_index=batch_index,
                output_root=args.output_root,
                generation=generation,
                run_dir=run_parent_for(args.output_root, video_id, seed),
                log_path=log_path,
            )
            row.update({"status": "fail", "end_time": utc_now(), "message": message})
            append_manifest(args.output_root, row)
            emit(
                args.output_root,
                f"FAILED TO LAUNCH video={video_id} seed={seed}: {error}",
            )

    if running:
        emit(
            args.output_root,
            f"WAITING video={video_id} parallel_jobs="
            f"{[(job.seed, job.process.pid) for job in running]}",
        )
    try:
        while any(job.process.poll() is None for job in running):
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        emit(args.output_root, "INTERRUPT received; stopping only active seed-search jobs")
        _interrupt_jobs(running, args, generation)
        raise
    for job in running:
        statuses[finish_job(args, generation, job)] += 1
    return statuses


def load_or_create_schedule(output_root: Path, start_seed: int) -> dict[str, Any]:
    path = output_root / "seed_schedule.json"
    if path.is_file():
        schedule = read_json(path)
        if schedule.get("start_seed") != start_seed:
            raise RuntimeError("Existing seed schedule uses a different --start-seed")
        return schedule
    schedule = {
        "schema_version": 2,
        "semantic": "step10_three_frame_preview",
        "created_at": utc_now(),
        "start_seed": start_seed,
        "seed_range_inclusive": [SEED_MIN, SEED_MAX],
        "entries": [],
    }
    atomic_json(path, schedule)
    return schedule


def allocate_seeds(schedule: dict[str, Any], count: int) -> list[int]:
    used = {
        int(seed)
        for entry in schedule.get("entries", [])
        for seed in entry.get("seeds", [])
    }
    result: list[int] = []
    start_seed = int(schedule["start_seed"])
    if not used:
        result.append(start_seed)
    while len(result) < count:
        seed = SEED_MIN + secrets.randbelow(SEED_MAX - SEED_MIN + 1)
        if seed not in used and seed not in result:
            result.append(seed)
    return result


def load_or_create_scheduler_state(
    output_root: Path, scheduler_contract: dict[str, Any]
) -> dict[str, Any]:
    contract_path = output_root / "scheduler_contract.json"
    if contract_path.is_file():
        existing = read_json(contract_path)
        if existing.get("contract") != scheduler_contract:
            raise RuntimeError(
                "Existing scheduler contract differs; refusing mixed scheduling"
            )
    else:
        atomic_json(
            contract_path, {"created_at": utc_now(), "contract": scheduler_contract}
        )

    path = output_root / "scheduler_state.json"
    if path.is_file():
        return read_json(path)
    state = {
        "schema_version": 2,
        "next_video_index": 0,
        "next_round_index": 1,
        "next_batch_index": 1,
        "current_batch": None,
        "updated_at": utc_now(),
    }
    atomic_json(path, state)
    return state


def persist_schedule(output_root: Path, schedule: dict[str, Any]) -> None:
    atomic_json(output_root / "seed_schedule.json", schedule)


def persist_state(output_root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json(output_root / "scheduler_state.json", state)


def acquire_scheduler_lock(output_root: Path) -> Any:
    path = output_root / ".seed_search.lock"
    handle = path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"Another seed-search scheduler owns {path}") from error
    handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
    handle.flush()
    return handle


def scheduler(args: argparse.Namespace, generation: dict[str, Any]) -> None:
    output_root = args.output_root.resolve()
    videos = tuple(part.strip() for part in args.videos.split(",") if part.strip())
    if not videos or any(video not in DEFAULT_VIDEOS for video in videos):
        raise ValueError(f"--videos must be a subset of {','.join(DEFAULT_VIDEOS)}")
    if len(set(videos)) != len(videos):
        raise ValueError("--videos contains duplicates")
    if args.max_concurrent != 2 or args.seeds_per_video_per_round != 2:
        raise ValueError(
            "This production pipeline requires exactly 2 concurrent jobs and 2 seeds/video"
        )
    if not SEED_MIN <= args.start_seed <= SEED_MAX:
        raise ValueError("--start-seed must be an 8-digit integer")

    scheduler_contract = {
        "schema_version": 2,
        "semantic": generation["semantic"],
        "videos": list(videos),
        "max_concurrent": args.max_concurrent,
        "seeds_per_video_per_round": args.seeds_per_video_per_round,
        "start_seed": args.start_seed,
        "round_robin": True,
        "num_frames": FRAME_COUNT,
        "full_schedule_steps": FULL_SCHEDULE_STEPS,
        "preview_step": PREVIEW_STEP,
        "preview_frames": list(PREVIEW_FRAMES),
    }
    schedule = load_or_create_schedule(output_root, args.start_seed)
    state = load_or_create_scheduler_state(output_root, scheduler_contract)
    batches_processed = 0
    emit(
        output_root,
        f"SCHEDULER START pid={os.getpid()} videos={list(videos)} "
        f"continuous={args.continuous} semantic={generation['semantic']}",
    )

    while True:
        current = state.get("current_batch")
        if current is None:
            video_index = int(state["next_video_index"])
            video_id = videos[video_index]
            seeds = allocate_seeds(schedule, args.seeds_per_video_per_round)
            current = {
                "batch_index": int(state["next_batch_index"]),
                "round_index": int(state["next_round_index"]),
                "video_index": video_index,
                "video_id": video_id,
                "seeds": seeds,
                "created_at": utc_now(),
            }
            schedule["entries"].append(dict(current))
            persist_schedule(output_root, schedule)
            state["current_batch"] = current
            persist_state(output_root, state)

        emit(
            output_root,
            f"BATCH START round={current['round_index']} batch={current['batch_index']} "
            f"video={current['video_id']} seeds={current['seeds']}",
        )
        statuses = run_batch(
            args,
            generation,
            video_id=current["video_id"],
            seeds=list(map(int, current["seeds"])),
            round_index=int(current["round_index"]),
            batch_index=int(current["batch_index"]),
        )
        emit(
            output_root,
            f"BATCH END round={current['round_index']} batch={current['batch_index']} "
            f"video={current['video_id']} statuses={statuses}",
        )

        next_index = (int(current["video_index"]) + 1) % len(videos)
        next_round = int(current["round_index"]) + (1 if next_index == 0 else 0)
        state.update(
            {
                "next_video_index": next_index,
                "next_round_index": next_round,
                "next_batch_index": int(current["batch_index"]) + 1,
                "current_batch": None,
                "last_completed_batch": {
                    **current,
                    "completed_at": utc_now(),
                    "statuses": statuses,
                },
            }
        )
        persist_state(output_root, state)
        batches_processed += 1
        if args.max_batches is not None and batches_processed >= args.max_batches:
            break
        if not args.continuous and next_index == 0:
            break
        time.sleep(args.between_batches_seconds)

    emit(output_root, f"SCHEDULER STOP batches_processed={batches_processed}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--steps", type=int, default=FULL_SCHEDULE_STEPS)
    parser.add_argument("--preview-step", type=int, default=PREVIEW_STEP)
    parser.add_argument(
        "--preview-frames", type=int, nargs="+", default=list(PREVIEW_FRAMES)
    )
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--seeds-per-video-per-round", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=33_333_333)
    parser.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--single-video", choices=DEFAULT_VIDEOS)
    parser.add_argument("--single-seed", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--between-batches-seconds", type=float, default=5.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.poll_seconds <= 0 or args.between_batches_seconds < 0:
        parser.error("poll/between-batches seconds are invalid")
    if (args.single_video is None) != (args.single_seed is None):
        parser.error("--single-video and --single-seed must be supplied together")
    if args.single_seed is not None and not SEED_MIN <= args.single_seed <= SEED_MAX:
        parser.error("--single-seed must be an 8-digit integer")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    generation = preflight(args)
    emit(
        args.output_root,
        "PREFLIGHT PASS "
        f"65f 3840x2160 full_schedule=30 preview_step=10 frames={list(PREVIEW_FRAMES)} "
        f"checkpoint={generation['checkpoint_path']} "
        f"negative_prompt={generation['negative_prompt']!r}",
    )
    if args.preflight_only:
        return 0
    lock = acquire_scheduler_lock(args.output_root)
    try:
        if args.single_video is not None:
            statuses = run_batch(
                args,
                generation,
                video_id=args.single_video,
                seeds=[args.single_seed],
                round_index=0,
                batch_index=0,
            )
            return 0 if statuses["fail"] == 0 else 1
        scheduler(args, generation)
        return 0
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(main())
