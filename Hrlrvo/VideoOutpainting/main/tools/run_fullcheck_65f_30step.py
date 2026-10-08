#!/usr/bin/env python3
"""Sequentially validate selected seed-search candidates with full 65f inference.

This runner deliberately invokes the production ``normal_long_range`` path and
never supplies any seed-preview CLI option.  It records one isolated run
directory per candidate, validates each final MP4 with ffprobe, and keeps going
to the next video if an individual job fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, TextIO

import yaml

REPO_ROOT = Path("/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/main")
CONFIG_PATH = REPO_ROOT / "configs/inference_4k.yaml"
CHECKPOINT_PATH = Path(
    "/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/"
    "runs/hrlrvo_main_train_v1/checkpoints/checkpoint-007500"
)
INPUT_ROOT = Path(
    "/home/work/ai-graphics_33/Hrlrvo/pipeline/data/01_PreProc_2K_1920x1080"
)
OUTPUT_ROOT = Path(
    "/home/work/ai-graphics_33/Hrlrvo/pipeline/data/03_fullCheck_65f_30step"
)
NEGATIVE_PROMPT = "people, person, human, staff, audience, crowd, backstage, seats"
NUM_FRAMES = 65
INFERENCE_STEPS = 30
WIDTH = 3840
HEIGHT = 2160
FPS = Fraction(24, 1)


@dataclass(frozen=True)
class Job:
    video_id: str
    input_path: Path
    seed: int
    negative_prompt: str


DEFAULT_JOBS = (
    Job("01", INPUT_ROOT / "01_PreProc_2K.mp4", 50374151, NEGATIVE_PROMPT),
    Job("02", INPUT_ROOT / "02_PreProc_2K.mp4", 52044206, NEGATIVE_PROMPT),
    Job("03", INPUT_ROOT / "03_PreProc_2K.mp4", 33333333, ""),
    Job("04", INPUT_ROOT / "04_PreProc_2K.mp4", 19379255, NEGATIVE_PROMPT),
    Job("05", INPUT_ROOT / "05_PreProc_2K.mp4", 71634910, NEGATIVE_PROMPT),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(message: str, *, root_log: TextIO | None = None) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    if root_log is not None:
        root_log.write(line + "\n")
        root_log.flush()


def run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def probe_video(path: Path, *, count_frames: bool) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
    command.extend(
        [
            "-show_entries",
            (
                "stream=codec_name,width,height,pix_fmt,avg_frame_rate,"
                "nb_frames,nb_read_frames:format=format_name,duration,size"
            ),
            "-of",
            "json",
            str(path),
        ]
    )
    result = run_json(command)
    streams = result.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected exactly one video stream in {path}")
    stream = streams[0]
    frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    return {
        "codec_name": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pix_fmt": stream.get("pix_fmt"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_frames": int(stream["nb_frames"])
        if str(stream.get("nb_frames", "")).isdigit()
        else None,
        "nb_read_frames": int(stream["nb_read_frames"])
        if str(stream.get("nb_read_frames", "")).isdigit()
        else None,
        "verified_frame_count": int(frame_value)
        if str(frame_value or "").isdigit()
        else None,
        "duration": float(result.get("format", {}).get("duration", 0.0)),
        "size_bytes": int(result.get("format", {}).get("size", path.stat().st_size)),
        "format_name": result.get("format", {}).get("format_name"),
    }


def validate_input(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Input video is missing or empty: {path}")
    probe = probe_video(path, count_frames=False)
    if (probe["width"], probe["height"]) != (1920, 1080):
        raise RuntimeError(f"Input must be 1920x1080: {path}: {probe}")
    if Fraction(probe["avg_frame_rate"]) != FPS:
        raise RuntimeError(f"Input must be 24 fps: {path}: {probe}")
    if probe["nb_frames"] is None or probe["nb_frames"] < NUM_FRAMES:
        raise RuntimeError(f"Input must contain at least 65 frames: {path}: {probe}")
    return probe


def validate_output(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Output MP4 is missing or empty: {path}")
    probe = probe_video(path, count_frames=True)
    failures: list[str] = []
    if (probe["width"], probe["height"]) != (WIDTH, HEIGHT):
        failures.append(f"resolution={probe['width']}x{probe['height']}")
    if Fraction(probe["avg_frame_rate"]) != FPS:
        failures.append(f"fps={probe['avg_frame_rate']}")
    if probe["verified_frame_count"] != NUM_FRAMES:
        failures.append(f"frames={probe['verified_frame_count']}")
    if failures:
        raise RuntimeError(f"Output contract failed for {path}: {', '.join(failures)}")
    return probe


def verify_checkpoint() -> dict[str, Any]:
    adapter = CHECKPOINT_PATH / "adapter_model.safetensors"
    manifest_path = CHECKPOINT_PATH / "outpainting_manifest.json"
    for path in (adapter, manifest_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Checkpoint artifact is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("global_step", -1)) != 7500:
        raise RuntimeError(
            "Checkpoint manifest does not identify global_step 7500: "
            f"{manifest.get('global_step')!r}"
        )
    return {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "global_step": 7500,
        "adapter_path": str(adapter),
        "adapter_size_bytes": adapter.stat().st_size,
        "adapter_sha256": sha256_file(adapter),
    }


def verify_config() -> dict[str, Any]:
    settings = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    checks = {
        "video.canvas": settings["video"]["canvas"] == [WIDTH, HEIGHT],
        "video.fps": Fraction(str(settings["video"]["fps"])) == FPS,
        "long_range.clip_frames": settings["long_range"]["clip_frames"] == NUM_FRAMES,
        "sampling.inference_steps": settings["sampling"]["inference_steps"]
        == INFERENCE_STEPS,
        "runtime.device": settings["runtime"]["device"] == "cuda",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Inference config contract failed: {', '.join(failed)}")
    return settings


def verify_gpu_idle() -> None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    active = completed.stdout.strip()
    if active:
        raise RuntimeError(
            "GPU already has compute processes; refusing to overlap this sequential "
            f"validation run:\n{active}"
        )


def preflight(
    jobs: tuple[Job, ...], *, require_idle_gpu: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    if Path.cwd().resolve() != REPO_ROOT:
        os.chdir(REPO_ROOT)
    for executable in ("ffprobe", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable is unavailable: {executable}")
    module_path = REPO_ROOT / "src/ltx_outpaint/inference.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Inference entrypoint is missing: {module_path}")
    settings = verify_config()
    checkpoint = verify_checkpoint()
    inputs = {job.video_id: validate_input(job.input_path) for job in jobs}
    if require_idle_gpu:
        verify_gpu_idle()
    return settings, {"checkpoint": checkpoint, "inputs": inputs}


def output_path_for(job: Job) -> Path:
    return OUTPUT_ROOT / job.video_id / f"{job.video_id}_{job.seed}_65f_30step.mp4"


def run_dir_for(job: Job) -> Path:
    return OUTPUT_ROOT / "_runs" / f"{job.video_id}_{job.seed}"


def build_command(job: Job, run_dir: Path, output_path: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ltx_outpaint.inference",
        "--config",
        str(CONFIG_PATH),
        "--checkpoint",
        str(CHECKPOINT_PATH),
        "--steps",
        str(INFERENCE_STEPS),
        "--video-only-gpu-weights",
        "--vae-decode-streaming",
        "--no-ffn-token-chunking",
        "--anchor-video",
        str(job.input_path),
        "--mode",
        "normal_long_range",
        "--num-frames",
        str(NUM_FRAMES),
        "--seed",
        str(job.seed),
        "--negative-prompt",
        job.negative_prompt,
        "--output",
        str(output_path),
        "--segment-dir",
        str(run_dir / "segments"),
    ]
    segment_manifest = run_dir / "segments/run.json"
    if segment_manifest.is_file():
        command.append("--resume")
    elif (run_dir / "segments").exists() and any((run_dir / "segments").iterdir()):
        raise RuntimeError(
            f"Nonempty segment directory has no resume manifest: {run_dir / 'segments'}"
        )
    return command


def write_resolved_config(
    run_dir: Path,
    settings: dict[str, Any],
    job: Job,
    output_path: Path,
) -> None:
    resolved = dict(settings)
    resolved["_fullcheck_runtime_cli"] = {
        "mode": "normal_long_range",
        "input_path": str(job.input_path),
        "output_path": str(output_path),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "seed": job.seed,
        "negative_prompt": job.negative_prompt,
        "num_frames": NUM_FRAMES,
        "inference_steps": INFERENCE_STEPS,
        "preview_options_supplied": False,
        "video_only_gpu_weights": True,
        "vae_decode_streaming": True,
        "ffn_token_chunking": False,
    }
    atomic_write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
    )


def base_metadata(job: Job, run_dir: Path, output_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "semantic": "selected_seed_full_65f_30step_validation",
        "video_id": job.video_id,
        "input_path": str(job.input_path),
        "seed": job.seed,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "num_frames": NUM_FRAMES,
        "source_frame_range": [0, NUM_FRAMES],
        "resolution": [WIDTH, HEIGHT],
        "fps": float(FPS),
        "inference_steps": INFERENCE_STEPS,
        "negative_prompt": job.negative_prompt,
        "negative_prompt_enabled": bool(job.negative_prompt),
        "preview_mode": False,
        "preview_step": None,
        "output_path": str(output_path),
        "run_dir": str(run_dir),
        "start_time": None,
        "end_time": None,
        "elapsed_seconds": None,
        "exit_code": None,
        "status": "pending",
    }


def copy_native_artifacts(output_path: Path, run_dir: Path) -> None:
    for name in ("inference_manifest.json", "chunk_lineage.json"):
        source = output_path.parent / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)


def update_context(context_path: Path, context: dict[str, Any]) -> None:
    context["updated_at"] = utc_now()
    atomic_write_json(context_path, context)


def stream_process(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== attempt {utc_now()} =====\n")
        handle.write(f"COMMAND: {shlex.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            return process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            raise


def run_job(
    job: Job,
    settings: dict[str, Any],
    context: dict[str, Any],
    context_path: Path,
    root_log: TextIO,
) -> bool:
    run_dir = run_dir_for(job)
    output_path = output_path_for(job)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    metadata = base_metadata(job, run_dir, output_path)
    write_resolved_config(run_dir, settings, job, output_path)

    try:
        existing_probe = validate_output(output_path)
    except FileNotFoundError:
        existing_probe = None
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        metadata.update(
            {
                "status": "invalid_existing_output",
                "end_time": utc_now(),
                "exit_code": 70,
                "error": str(exc),
            }
        )
        atomic_write_json(metadata_path, metadata)
        log(f"INVALID EXISTING OUTPUT video={job.video_id}: {exc}", root_log=root_log)
        return False
    if existing_probe is not None:
        metadata.update(
            {
                "status": "success",
                "skip_reason": "existing output passed ffprobe validation",
                "end_time": utc_now(),
                "exit_code": 0,
                "ffprobe": existing_probe,
                "output_sha256": sha256_file(output_path),
            }
        )
        atomic_write_json(metadata_path, metadata)
        copy_native_artifacts(output_path, run_dir)
        log(
            f"SKIP valid output video={job.video_id} seed={job.seed} path={output_path}",
            root_log=root_log,
        )
        return True

    command = build_command(job, run_dir, output_path)
    atomic_write_text(run_dir / "command.txt", shlex.join(command) + "\n")
    start_wall = time.monotonic()
    metadata.update(
        {
            "status": "running",
            "start_time": utc_now(),
            "command": command,
            "pid": os.getpid(),
        }
    )
    atomic_write_json(metadata_path, metadata)
    context["jobs"][job.video_id] = {
        "seed": job.seed,
        "status": "running",
        "run_dir": str(run_dir),
        "output_path": str(output_path),
        "negative_prompt_enabled": bool(job.negative_prompt),
    }
    update_context(context_path, context)
    log(
        "START "
        f"video={job.video_id} seed={job.seed} negative_prompt="
        f"{'enabled' if job.negative_prompt else 'disabled'}",
        root_log=root_log,
    )
    log(f"COMMAND {shlex.join(command)}", root_log=root_log)

    env = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        source_path
        if not env.get("PYTHONPATH")
        else f"{source_path}:{env['PYTHONPATH']}"
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        exit_code = stream_process(command, run_dir / "inference.log", env)
    except KeyboardInterrupt:
        metadata.update(
            {
                "status": "interrupted",
                "end_time": utc_now(),
                "elapsed_seconds": time.monotonic() - start_wall,
                "exit_code": 130,
            }
        )
        atomic_write_json(metadata_path, metadata)
        context["jobs"][job.video_id]["status"] = "interrupted"
        update_context(context_path, context)
        raise

    metadata.update(
        {
            "end_time": utc_now(),
            "elapsed_seconds": time.monotonic() - start_wall,
            "exit_code": exit_code,
        }
    )
    if exit_code != 0:
        metadata["status"] = "fail"
        atomic_write_json(metadata_path, metadata)
        context["jobs"][job.video_id].update({"status": "fail", "exit_code": exit_code})
        update_context(context_path, context)
        log(
            f"FAIL video={job.video_id} seed={job.seed} exit={exit_code}",
            root_log=root_log,
        )
        return False

    try:
        output_probe = validate_output(output_path)
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        metadata.update(
            {
                "status": "invalid_output",
                "exit_code": 70,
                "validation_error": str(exc),
            }
        )
        atomic_write_json(metadata_path, metadata)
        context["jobs"][job.video_id].update(
            {"status": "invalid_output", "exit_code": 70}
        )
        update_context(context_path, context)
        log(f"INVALID video={job.video_id}: {exc}", root_log=root_log)
        return False

    copy_native_artifacts(output_path, run_dir)
    metadata.update(
        {
            "status": "success",
            "exit_code": 0,
            "ffprobe": output_probe,
            "output_sha256": sha256_file(output_path),
        }
    )
    atomic_write_json(metadata_path, metadata)
    context["jobs"][job.video_id].update(
        {
            "status": "success",
            "exit_code": 0,
            "elapsed_seconds": metadata["elapsed_seconds"],
            "ffprobe": output_probe,
        }
    )
    update_context(context_path, context)
    log(
        "SUCCESS "
        f"video={job.video_id} seed={job.seed} "
        f"resolution={output_probe['width']}x{output_probe['height']} "
        f"frames={output_probe['verified_frame_count']} "
        f"elapsed={metadata['elapsed_seconds']:.1f}s path={output_path}",
        root_log=root_log,
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected seeds sequentially as full 65f/30-step MP4s"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config/checkpoint/inputs without creating outputs or using the GPU",
    )
    parser.add_argument(
        "--job",
        action="append",
        metavar="VIDEO_ID:SEED",
        help=(
            "Run an explicit video/seed pair. Repeat for sequential jobs; "
            "omitting this option uses the original five-job validation set."
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=NEGATIVE_PROMPT,
        help="Negative prompt applied to every explicit --job",
    )
    parser.add_argument(
        "--run-id",
        help="Safe identifier used to isolate the suite context and runner log",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Allow this run to share a GPU after the caller verifies VRAM headroom",
    )
    return parser.parse_args()


def resolve_jobs(args: argparse.Namespace) -> tuple[Job, ...]:
    if not args.job:
        return DEFAULT_JOBS
    inputs = {
        video_id: INPUT_ROOT / f"{video_id}_PreProc_2K.mp4"
        for video_id in ("01", "02", "03", "04", "05")
    }
    jobs: list[Job] = []
    seen: set[str] = set()
    for value in args.job:
        try:
            video_id, raw_seed = value.split(":", 1)
            seed = int(raw_seed)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --job {value!r}; expected VIDEO_ID:SEED"
            ) from exc
        if video_id not in inputs:
            raise ValueError(f"Unsupported video_id in --job: {video_id!r}")
        if video_id in seen:
            raise ValueError(f"Duplicate video_id in --job: {video_id}")
        if not 10_000_000 <= seed <= 99_999_999:
            raise ValueError(f"Seed must be an 8-digit integer: {seed}")
        seen.add(video_id)
        jobs.append(Job(video_id, inputs[video_id], seed, args.negative_prompt))
    return tuple(jobs)


def resolve_run_id(args: argparse.Namespace, jobs: tuple[Job, ...]) -> str | None:
    run_id = args.run_id
    if run_id is None and args.job:
        run_id = "_".join(f"{job.video_id}_{job.seed}" for job in jobs)
    if run_id is not None and re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("--run-id may contain only letters, digits, '_' and '-'")
    return run_id


def main() -> int:
    args = parse_args()
    jobs = resolve_jobs(args)
    run_id = resolve_run_id(args, jobs)
    settings, evidence = preflight(
        jobs,
        require_idle_gpu=not args.preflight_only and not args.allow_busy_gpu,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_pass",
                    "python": sys.executable,
                    "config_path": str(CONFIG_PATH),
                    "checkpoint": evidence["checkpoint"],
                    "jobs": [
                        {
                            **asdict(job),
                            "input_path": str(job.input_path),
                            "output_path": str(output_path_for(job)),
                            "negative_prompt_enabled": bool(job.negative_prompt),
                        }
                        for job in jobs
                    ],
                    "run_id": run_id,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_suffix = "" if run_id is None else f"_{run_id}"
    context_path = OUTPUT_ROOT / f"run_context{artifact_suffix}.json"
    context: dict[str, Any] = {
        "schema_version": 1,
        "semantic": "selected_seed_full_65f_30step_validation",
        "run_id": run_id,
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "ended_at": None,
        "python": sys.executable,
        "config_path": str(CONFIG_PATH),
        "checkpoint": evidence["checkpoint"],
        "execution": {
            "sequential": True,
            "max_concurrent": 1,
            "allow_busy_gpu": args.allow_busy_gpu,
            "mode": "normal_long_range",
            "num_frames": NUM_FRAMES,
            "source_frame_range": [0, NUM_FRAMES],
            "inference_steps": INFERENCE_STEPS,
            "preview_options_supplied": False,
        },
        "jobs": {},
    }
    update_context(context_path, context)
    suite_started = time.monotonic()
    results: list[bool] = []
    root_log_path = OUTPUT_ROOT / f"fullcheck_runner{artifact_suffix}.log"
    with root_log_path.open("a", encoding="utf-8") as root_log:
        log(
            "SUITE START sequential=1 max_concurrent=1 num_frames=65 "
            "steps=30 preview_options=none",
            root_log=root_log,
        )
        try:
            for job in jobs:
                results.append(run_job(job, settings, context, context_path, root_log))
        except KeyboardInterrupt:
            context.update(
                {
                    "status": "interrupted",
                    "ended_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - suite_started,
                }
            )
            update_context(context_path, context)
            log("SUITE INTERRUPTED", root_log=root_log)
            return 130

        successful = sum(results)
        context.update(
            {
                "status": "success" if successful == len(jobs) else "partial_failure",
                "ended_at": utc_now(),
                "elapsed_seconds": time.monotonic() - suite_started,
                "success_count": successful,
                "failure_count": len(jobs) - successful,
            }
        )
        update_context(context_path, context)
        log(
            f"SUITE END status={context['status']} success={successful}/{len(jobs)} "
            f"elapsed={context['elapsed_seconds']:.1f}s",
            root_log=root_log,
        )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
