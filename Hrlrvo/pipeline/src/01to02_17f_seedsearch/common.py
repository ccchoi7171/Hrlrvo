"""Shared paths, input checks and artifact validation for the prefix-search wrapper.

No model or sampling logic lives here.  The actual generation is delegated to
``ltx_outpaint.inference --mode prefix_seed_search``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

PIPELINE = Path(__file__).resolve().parents[2]
PROJECT = PIPELINE.parent
MAIN = PROJECT / "VideoOutpainting/main"
INPUT_ROOT = PIPELINE / "data/01-1_FPS_2K_1920_1080_24fps"
OUTPUT_ROOT = PIPELINE / "data/02_seedSearch_4K_17f_3840_2160"
DEFAULT_CONFIG = MAIN / "configs/inference_4k.yaml"
DEFAULT_CHECKPOINT = PROJECT / "VideoOutpainting/runs/hrlrvo_main_train_v1/checkpoints/checkpoint-007500"
VIDEO_IDS = ("01", "02", "03", "04", "05")
SOURCE_FRAMES = 17
FPS = Fraction(24, 1)
CANVAS = (3840, 2160)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def input_for(video_id: str) -> Path:
    if video_id not in VIDEO_IDS:
        raise ValueError(f"Unknown video ID {video_id!r}; expected one of {VIDEO_IDS}")
    return INPUT_ROOT / f"{video_id}_Input_2K.mp4"


def output_for(video_id: str, seed: int, root: Path = OUTPUT_ROOT) -> tuple[Path, Path, Path]:
    if video_id not in VIDEO_IDS or type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("video ID or seed is invalid")
    base = Path(root) / video_id / f"{video_id}_{seed}_4K"
    return base.with_suffix(".mp4"), Path(f"{base}_prefix.pt"), Path(f"{base}_meta.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    if len(data.get("streams", [])) != 1:
        raise ValueError(f"Expected one selected video stream in {path}")
    stream = data["streams"][0]
    if stream.get("codec_type") != "video":
        raise ValueError(f"Video stream missing in {path}")
    count_text = stream.get("nb_read_frames") or stream.get("nb_frames")
    if not count_text or count_text == "N/A":
        raise ValueError(f"Could not count video frames in {path}")
    return {
        "resolution": [int(stream["width"]), int(stream["height"])],
        "fps": stream["avg_frame_rate"],
        "r_frame_rate": stream.get("r_frame_rate"),
        "frame_count": int(count_text),
        "duration_seconds": float(stream.get("duration") or data.get("format", {}).get("duration") or 0),
    }


def decode_video(path: Path) -> int:
    """Decode every video frame using bounded-memory FFmpeg; no RGB preload."""
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror",
        "-err_detect", "explode", "-threads", "2", "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
        "-f", "null", "-", "-progress", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Full video decode failed for {path}: {result.stderr.strip()[-1500:]}")
    frames = [line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("frame=")]
    if not frames:
        raise RuntimeError(f"FFmpeg did not report decoded frames for {path}")
    return int(frames[-1])


def check_video(path: Path, *, resolution: tuple[int, int], min_frames: int,
                exact_frames: int | None = None, full_decode: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    info = probe_video(path)
    if tuple(info["resolution"]) != resolution:
        raise ValueError(f"{path}: resolution {info['resolution']} != {list(resolution)}")
    if Fraction(info["fps"]) != FPS:
        raise ValueError(f"{path}: FPS {info['fps']} != 24/1")
    if info["frame_count"] < min_frames:
        raise ValueError(f"{path}: {info['frame_count']} frames < {min_frames}")
    if exact_frames is not None and info["frame_count"] != exact_frames:
        raise ValueError(f"{path}: {info['frame_count']} frames != {exact_frames}")
    if full_decode:
        info["decoded_frames"] = decode_video(path)
        if info["decoded_frames"] != info["frame_count"]:
            raise ValueError(f"{path}: decoded {info['decoded_frames']} != probed {info['frame_count']}")
    return info


def preflight_inputs(root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Check all five inputs, save evidence, and fail if any input is unsuitable."""
    records: dict[str, Any] = {}
    for video_id in VIDEO_IDS:
        path = input_for(video_id)
        record: dict[str, Any] = {"video_id": video_id, "path": str(path), "exists": path.is_file()}
        try:
            record.update(check_video(path, resolution=(1920, 1080), min_frames=SOURCE_FRAMES))
            record["sha256"] = sha256_file(path)
            record["decode_status"] = "pass"
            record["status"] = "pass"
        except Exception as error:  # noqa: BLE001 - collect all input failures before aborting
            record.update({"decode_status": "fail", "status": "fail", "error": str(error)})
        records[video_id] = record
    report = {"checked_at": utc_now(), "input_root": str(INPUT_ROOT), "videos": records,
              "status": "pass" if all(v["status"] == "pass" for v in records.values()) else "fail"}
    atomic_json(Path(root) / "_manifests/input_preflight.json", report)
    if report["status"] != "pass":
        raise RuntimeError("Input preflight failed: " + "; ".join(
            f"{key}: {value['error']}" for key, value in records.items() if value["status"] != "pass"
        ))
    return report


def inference_identity(config: Path, checkpoint: Path, steps: int) -> dict[str, Any]:
    config, checkpoint = config.resolve(), checkpoint.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    adapter = checkpoint / "adapter_model.safetensors"
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    if steps < 1:
        raise ValueError("steps must be positive")
    with config.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if Fraction(str(settings["video"]["fps"])) != FPS:
        raise ValueError(f"Inference config FPS is {settings['video']['fps']}, expected 24")
    if tuple(settings["video"]["canvas"]) != CANVAS:
        raise ValueError("Inference canvas must be 3840x2160")
    if tuple(settings["video"]["expected_anchor"]) != (1920, 1080):
        raise ValueError("Inference expected anchor must be 1920x1080")
    if int(settings["long_range"]["prefix_frames"]) != SOURCE_FRAMES:
        raise ValueError("Inference prefix must be exactly 17 frames")
    return {
        "config_path": str(config), "config_sha256": sha256_file(config),
        "checkpoint_path": str(checkpoint), "checkpoint_identifier": checkpoint.name,
        "adapter_path": str(adapter), "adapter_sha256": sha256_file(adapter),
        "steps": steps, "fps_setting": 24, "canvas": list(CANVAS),
        "source_frame_range_inclusive": [0, 16],
        "sampling_settings": settings.get("sampling"),
        "video_settings": settings.get("video"),
        "memory_optimization": {
            "video_only_gpu_weights": True, "vae_decode_streaming": True,
            "ffn_token_chunking": False,
        },
    }


def runtime_identity() -> dict[str, Any]:
    import torch

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    return {
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "conda_environment": Path(sys.prefix).name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pytorch_version": torch.__version__, "cuda_runtime_version": torch.version.cuda,
        "gpu_model": gpu.stdout.strip().splitlines()[0] if gpu.returncode == 0 and gpu.stdout.strip() else None,
    }


def validate_candidate(video_id: str, seed: int, root: Path = OUTPUT_ROOT,
                       *, expected_input_sha: str | None = None,
                       expected_config_sha: str | None = None,
                       expected_adapter_sha: str | None = None,
                       expected_checkpoint: Path | None = None,
                       expected_steps: int | None = None,
                       expected_schedule_sha: str | None = None,
                       expected_round: int | None = None,
                       expected_slot: int | None = None,
                       full_decode: bool = True) -> dict[str, Any]:
    mp4, tensor, meta = output_for(video_id, seed, root)
    if not all(path.is_file() for path in (mp4, tensor, meta)):
        raise FileNotFoundError(f"Incomplete output for {video_id}/{seed}: {mp4}, {tensor}, {meta}")
    with meta.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {
        "status": "success", "video_id": video_id, "seed": seed,
        "tensor_file": tensor.name, "preview_file": mp4.name,
        "tensor_shape": [SOURCE_FRAMES, 3, CANVAS[1], CANVAS[0]],
        "tensor_dtype": "float32", "frame_count": SOURCE_FRAMES,
        "resolution": list(CANVAS), "fps": 24.0,
        "output_mp4_path": str(mp4), "output_prefix_path": str(tensor),
        "source_frame_range_inclusive": [0, 16],
        "memory_optimization": {
            "video_only_gpu_weights": True, "vae_decode_streaming": True,
            "ffn_token_chunking": False,
        },
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{meta}: {key}={metadata.get(key)!r}, expected {expected!r}")
    for key, expected in (("input_video_sha256", expected_input_sha),
                          ("config_sha256", expected_config_sha),
                          ("adapter_sha256", expected_adapter_sha),
                          ("checkpoint_path", str(expected_checkpoint.resolve()) if expected_checkpoint else None),
                          ("steps", expected_steps),
                          ("schedule_sha256", expected_schedule_sha),
                          ("round_index", expected_round),
                          ("slot_index", expected_slot)):
        if expected is not None and metadata.get(key) != expected:
            raise ValueError(f"{meta}: {key} mismatch")
    if tensor.stat().st_size != metadata.get("tensor_byte_size"):
        raise ValueError(f"{tensor}: tensor byte size mismatch")
    if sha256_file(tensor) != metadata.get("tensor_sha256"):
        raise ValueError(f"{tensor}: tensor SHA-256 mismatch")
    if sha256_file(mp4) != metadata.get("output_mp4_sha256"):
        raise ValueError(f"{mp4}: MP4 SHA-256 mismatch")
    info = check_video(mp4, resolution=CANVAS, min_frames=SOURCE_FRAMES,
                       exact_frames=SOURCE_FRAMES, full_decode=full_decode)
    return {"metadata": metadata, "video": info}
