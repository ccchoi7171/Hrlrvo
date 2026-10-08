"""Run one existing VideoOutpainting prefix_seed_search (one seed, B=1)."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from common import (
    CANVAS,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    MAIN,
    OUTPUT_ROOT,
    SOURCE_FRAMES,
    append_jsonl,
    atomic_json,
    check_video,
    inference_identity,
    input_for,
    output_for,
    runtime_identity,
    sha256_file,
    utc_now,
    validate_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, choices=("01", "02", "03", "04", "05"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--round-index", type=int)
    parser.add_argument("--slot-index", type=int)
    parser.add_argument("--schedule-sha256")
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--resume", action="store_true", help="Skip only a fully validated matching result")
    parser.add_argument("--verify-tensor-load", action="store_true", help="Smoke-test the continuation loader")
    return parser.parse_args()


def native_command(*, input_video: Path, seed: int, stage: Path, config: Path,
                   checkpoint: Path, steps: int) -> list[str]:
    return [
        sys.executable, "-m", "ltx_outpaint.inference",
        "--config", str(config), "--anchor-video", str(input_video),
        "--mode", "prefix_seed_search", "--prefix-output-dir", str(stage),
        "--prefix-seeds", str(seed), "--checkpoint", str(checkpoint),
        "--steps", str(steps), "--video-only-gpu-weights",
        "--vae-decode-streaming", "--no-ffn-token-chunking",
    ]


def main() -> None:
    args = parse_args()
    if not 0 <= args.seed < 2**31:
        raise SystemExit("seed must be in [0, 2**31)")
    input_video = input_for(args.video_id).resolve()
    root = args.output_root.expanduser().resolve()
    config = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    mp4, tensor, meta = output_for(args.video_id, args.seed, root)
    mp4.parent.mkdir(parents=True, exist_ok=True)
    manifest = root / "_manifests/run_manifest.jsonl"
    failure_manifest = root / "_manifests/failed_runs.jsonl"
    lock_path = mp4.parent / f".{args.video_id}_{args.seed}_4K.run.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _run_locked(args, input_video, root, config, checkpoint, mp4, tensor, meta,
                    manifest, failure_manifest)


def _run_locked(args: argparse.Namespace, input_video: Path, root: Path, config: Path,
                checkpoint: Path, mp4: Path, tensor: Path, meta: Path,
                manifest: Path, failure_manifest: Path) -> None:
    started_at = utc_now()
    started = time.monotonic()
    stage: Path | None = None
    native_log: Path | None = None
    command: list[str] | None = None
    native_pid: int | None = None
    try:
        input_info = check_video(input_video, resolution=(1920, 1080), min_frames=SOURCE_FRAMES,
                                 full_decode=False)
        input_sha = sha256_file(input_video)
        if args.expected_input_sha256 and args.expected_input_sha256 != input_sha:
            raise ValueError(f"Input SHA-256 changed since preflight: {input_video}")
        identity = inference_identity(config, checkpoint, args.steps)
        if meta.is_file():
            try:
                with meta.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
            except (OSError, ValueError):
                existing = None  # Incomplete/corrupt metadata is not a completion marker.
            if isinstance(existing, dict) and existing.get("status") == "success":
                try:
                    validate_candidate(
                        args.video_id, args.seed, root, expected_input_sha=input_sha,
                        expected_config_sha=identity["config_sha256"],
                        expected_adapter_sha=identity["adapter_sha256"],
                        expected_checkpoint=checkpoint, expected_steps=args.steps,
                        expected_schedule_sha=args.schedule_sha256,
                        expected_round=args.round_index, expected_slot=args.slot_index,
                    )
                except Exception as error:
                    # Never silently replace a committed good-seed result after a
                    # config/input change. Corrupt or partial data can be retried.
                    if (existing.get("input_video_sha256") != input_sha
                            or existing.get("config_sha256") != identity["config_sha256"]
                            or existing.get("adapter_sha256") != identity["adapter_sha256"]):
                        raise RuntimeError("Committed result belongs to a different input/config/checkpoint; "
                                           "use a separate --output-root") from error
                else:
                    if args.resume:
                        print(json.dumps({"status": "skipped_valid", "video_id": args.video_id,
                                          "seed": args.seed, "metadata": str(meta)}), flush=True)
                        return
                    raise FileExistsError(f"Complete result exists: {meta}; use another --output-root "
                                          "to actually rerun this seed, or --resume to validate/skip")

        stage = root / "_staging" / f"{args.video_id}_{args.seed}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        stage.mkdir(parents=True)
        log_dir = root / "_logs" / args.video_id
        log_dir.mkdir(parents=True, exist_ok=True)
        native_log = log_dir / f"{args.video_id}_{args.seed}_{stage.name}.native.log"
        command = native_command(input_video=input_video, seed=args.seed, stage=stage,
                                 config=config, checkpoint=checkpoint, steps=args.steps)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(MAIN / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with native_log.open("w", encoding="utf-8") as log:
            log.write("command: " + shlex.join(command) + "\n")
            log.write("working_directory: " + str(MAIN) + "\n")
            log.flush()
            native_process = subprocess.Popen(command, cwd=MAIN, env=env, stdout=log,
                                              stderr=subprocess.STDOUT)
            native_pid = native_process.pid
            log.write(f"native_pid: {native_pid}\n")
            log.flush()
            native_returncode = native_process.wait()
        if native_returncode:
            raise RuntimeError(f"Native prefix_seed_search exited {native_returncode}; see {native_log}")

        stem = f"seed_{args.seed}"
        native_mp4 = stage / f"{stem}_preview.mp4"
        native_tensor = stage / f"{stem}_prefix.pt"
        native_meta = stage / f"{stem}_meta.json"
        if not all(path.is_file() for path in (native_mp4, native_tensor, native_meta)):
            raise FileNotFoundError(f"Native mode did not produce all three artifacts in {stage}")
        with native_meta.open("r", encoding="utf-8") as handle:
            native = json.load(handle)
        if (native.get("seed") != args.seed or native.get("frame_count") != SOURCE_FRAMES
                or native.get("tensor_shape") != [SOURCE_FRAMES, 3, CANVAS[1], CANVAS[0]]
                or native.get("tensor_dtype") != "float32" or native.get("resolution") != list(CANVAS)
                or native.get("fps") != 24.0 or native.get("source_frame_range_inclusive") != [0, 16]
                or native.get("source_sha256") != input_sha
                or native.get("adapter_sha256") != identity["adapter_sha256"]
                or native.get("checkpoint_path") != str(checkpoint)
                or native.get("config_path") != str(config)
                or native.get("inference_steps") != args.steps):
            raise ValueError(f"Native artifact metadata did not match the requested run: {native_meta}")
        if native_tensor.stat().st_size != native.get("tensor_byte_size"):
            raise ValueError("Native prefix tensor byte size mismatch")
        if sha256_file(native_tensor) != native.get("tensor_sha256"):
            raise ValueError("Native prefix tensor SHA-256 mismatch")
        video_info = check_video(native_mp4, resolution=CANVAS, min_frames=SOURCE_FRAMES,
                                 exact_frames=SOURCE_FRAMES, full_decode=True)
        mp4_sha = sha256_file(native_mp4)
        runtime = runtime_identity()

        # The existing continuation loader accepts any *_prefix.pt basename as
        # long as matching *_meta.json has tensor_file/tensor_sha256. Rename only
        # artifacts, preserving all native lineage and exact FP32 tensor bytes.
        os.replace(native_tensor, tensor)
        os.replace(native_mp4, mp4)
        metadata = dict(native)
        metadata.update(identity)
        metadata.update(runtime)
        metadata.update({
            "status": "success", "video_id": args.video_id,
            "input_video": str(input_video), "input_video_sha256": input_sha,
            "input_resolution": input_info["resolution"], "input_fps": input_info["fps"],
            "input_frame_count": input_info["frame_count"],
            "source_frame_range": [0, SOURCE_FRAMES],
            "source_frame_range_inclusive": [0, SOURCE_FRAMES - 1],
            "round_index": args.round_index, "slot_index": args.slot_index,
            "schedule_sha256": args.schedule_sha256,
            "output_mp4_path": str(mp4), "output_prefix_path": str(tensor),
            "output_metadata_path": str(meta), "output_mp4_sha256": mp4_sha,
            "output_frame_count": SOURCE_FRAMES, "output_resolution": list(CANVAS),
            "output_fps": 24, "output_duration_seconds": video_info["duration_seconds"],
            "tensor_file": tensor.name, "preview_file": mp4.name,
            "native_command_argv": command, "exact_executed_command": shlex.join(command),
            "native_pythonpath": env["PYTHONPATH"], "working_directory": str(MAIN),
            "wrapper_invocation": [sys.executable, *sys.argv],
            "native_log_path": str(native_log), "native_pid": native_pid,
            "creation_timestamp": utc_now(), "start_time": started_at,
            "end_time": utc_now(), "runtime_seconds": time.monotonic() - started,
        })
        atomic_json(meta, metadata)  # Completion marker, deliberately written last.
        validate_candidate(args.video_id, args.seed, root, expected_input_sha=input_sha,
                           expected_config_sha=identity["config_sha256"],
                           expected_adapter_sha=identity["adapter_sha256"],
                           expected_checkpoint=checkpoint, expected_steps=args.steps,
                           expected_schedule_sha=args.schedule_sha256,
                           expected_round=args.round_index, expected_slot=args.slot_index)
        if args.verify_tensor_load:
            # Native loader checks shape, dtype, finite range and tensor digest.
            sys.path.insert(0, str(MAIN / "src"))
            from ltx_outpaint.prefix_io import load_prefix_artifact

            loaded, _ = load_prefix_artifact(meta.with_name(tensor.name), expected_frames=SOURCE_FRAMES,
                                             height=CANVAS[1], width=CANVAS[0])
            del loaded
        append_jsonl(manifest, {"event": "success", "video_id": args.video_id,
                                "seed": args.seed, "round_index": args.round_index,
                                "slot_index": args.slot_index, "metadata": str(meta),
                                "ended_at": utc_now()})
        native_meta.unlink(missing_ok=True)
        stage.rmdir()
        print(json.dumps({"status": "success", "video_id": args.video_id,
                          "seed": args.seed, "mp4": str(mp4), "prefix": str(tensor),
                          "metadata": str(meta), "native_log": str(native_log)}), flush=True)
    except Exception as error:
        # A partial/invalid metadata file must never act as a success marker.
        if meta.is_file():
            try:
                with meta.open("r", encoding="utf-8") as handle:
                    written = json.load(handle)
                if written.get("start_time") == started_at:
                    meta.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
        failure = {
            "status": "failed", "video_id": args.video_id, "seed": args.seed,
            "round_index": args.round_index, "slot_index": args.slot_index,
            "input_video": str(input_video), "start_time": started_at, "end_time": utc_now(),
            "runtime_seconds": time.monotonic() - started, "error": str(error),
            "traceback": traceback.format_exc(), "native_log_path": str(native_log) if native_log else None,
            "staging_directory": str(stage) if stage else None,
            "native_pid": native_pid,
            "exact_executed_command": shlex.join(command) if command else None,
        }
        append_jsonl(failure_manifest, failure)
        append_jsonl(manifest, failure)
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
