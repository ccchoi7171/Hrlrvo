#!/usr/bin/env python3
"""Official DOVE one-step pretrained video SR benchmark wrapper.

The neural path mirrors ``inference_script.py`` from the official DOVE
repository: bilinear x4 conditioning, CogVideoX VAE encode, one transformer
evaluation at timestep 399, scheduler velocity conversion, and CogVideoX VAE
decode.  Spatial and temporal overlaps are resolved by the same centre-valid
crop policy as upstream.  There is deliberately no interpolation-only output
fallback.

Unlike the research script, full-resolution conditioning and output frames are
disk-backed.  This avoids keeping the upstream full video, output, and integer
write-count tensors resident in host RAM at the same time.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

HERE = Path(__file__).resolve().parent
ARCHITECT = HERE.parent
ROOT = ARCHITECT.parent
MODEL_ROOT = ROOT / "models" / "DOVE"
REPOSITORY = MODEL_ROOT / "repository"
DEFAULT_MODEL_PATH = MODEL_ROOT / "checkpoint"
DEFAULT_INPUT = ROOT / "input" / "0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "benchmark" / "DOVE"
DEFAULT_RESULT_DIR = ROOT / "result" / "benchmark" / "DOVE"
MODEL_NAME = "DOVE"
SCALE = 4
MICRO_FRAMES = 9
MINI_FRAMES = 17
NATIVE_TEMPORAL_STRIDE = 8
SPATIAL_ALIGNMENT = 16
SR_TIMESTEP = 399
PINNED_REPOSITORY_REVISION = "0cd4240442cb5d122839c279977142cb6d648987"
PINNED_DOVE_MODEL_REVISION = "55ea86b7b2fbad77019a300e785b7fe1be065924"
PINNED_BASE_VAE_REVISION = "fdc5267c90b5c06492985b966e43aae984e189e0"

_MEDIA: Any | None = None


class DOVECancelled(RuntimeError):
    pass


class _OneShotSignalHandler:
    def __call__(self, signum: int, _frame: object) -> None:
        # A terminal signal and a launcher-forwarded signal can arrive nearly
        # together. Ignore duplicates so cleanup/report persistence is not
        # interrupted after the first cancellation request.
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            managed = getattr(signal, name, None)
            if managed is not None:
                signal.signal(managed, signal.SIG_IGN)
        raise DOVECancelled(f"received signal {signum}")


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{stamp}] [DOVE] {message}", flush=True)


def shared_media() -> Any:
    global _MEDIA
    if _MEDIA is None:
        name = "_hrlrvo_dove_media_utils"
        spec = importlib.util.spec_from_file_location(name, ARCHITECT / "common" / "media_utils.py")
        if spec is None or spec.loader is None:
            raise ImportError("cannot load architect/common/media_utils.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MEDIA = module
    return _MEDIA


@dataclasses.dataclass(frozen=True)
class Interval:
    start: int
    end: int
    valid_start: int
    valid_end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclasses.dataclass(frozen=True)
class SpatialTile:
    index: int
    y: Interval
    x: Interval


def padded_frame_count(frame_count: int) -> int:
    """Return the smallest DOVE/CogVideoX-valid 8n+1 frame count."""
    if frame_count <= 0:
        raise ValueError("frame count must be positive")
    return 1 + math.ceil((frame_count - 1) / NATIVE_TEMPORAL_STRIDE) * NATIVE_TEMPORAL_STRIDE


def benchmark_frame_range(
    mode: str, total_frames: int, start_frame: int, max_frames: int | None
) -> tuple[int, int]:
    if mode not in {"micro", "mini", "full"}:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    if not 0 <= start_frame < total_frames:
        raise ValueError(f"start frame {start_frame} is outside [0, {total_frames})")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    default = {"micro": MICRO_FRAMES, "mini": MINI_FRAMES, "full": total_frames - start_frame}[mode]
    count = min(default if max_frames is None else max_frames, total_frames - start_frame)
    if mode in {"micro", "mini"} and count < MICRO_FRAMES:
        raise ValueError(f"DOVE {mode} benchmark requires at least {MICRO_FRAMES} real frames")
    return start_frame, count


def _starts(length: int, window: int, overlap: int) -> list[int]:
    if length <= 0 or window <= 0:
        raise ValueError("length and window must be positive")
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must be non-negative and smaller than window")
    if window >= length:
        return [0]
    starts = list(range(0, length - window + 1, window - overlap))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def make_intervals(length: int, window: int, overlap: int) -> list[Interval]:
    """Make overlapping windows whose centre-valid regions cover once exactly."""
    starts = _starts(length, min(window, length), min(overlap, max(0, min(window, length) - 1)))
    ends = [min(start + window, length) for start in starts]
    boundaries = [0]
    for left_end, right_start in zip(ends[:-1], starts[1:]):
        if right_start >= left_end:
            raise ValueError("windows contain a gap; increase window or reduce stride")
        boundaries.append(right_start + (left_end - right_start) // 2)
    boundaries.append(length)
    return [
        Interval(start, end, boundaries[index], boundaries[index + 1])
        for index, (start, end) in enumerate(zip(starts, ends))
    ]


def make_temporal_chunks(frame_count: int, window: int, overlap: int) -> list[Interval]:
    if (frame_count - 1) % NATIVE_TEMPORAL_STRIDE:
        raise ValueError("DOVE temporal input must contain 8n+1 frames")
    window = min(window, frame_count)
    if (window - 1) % NATIVE_TEMPORAL_STRIDE:
        raise ValueError("--temporal-window must be 8n+1")
    chunks = make_intervals(frame_count, window, min(overlap, window - 1))
    if any((chunk.size - 1) % NATIVE_TEMPORAL_STRIDE for chunk in chunks):
        raise ValueError("every DOVE temporal chunk must contain 8n+1 frames")
    return chunks


def make_spatial_tiles(
    height: int, width: int, tile_height: int, tile_width: int, overlap: int
) -> list[SpatialTile]:
    for name, value in {
        "output height": height,
        "output width": width,
        "tile height": min(tile_height, height),
        "tile width": min(tile_width, width),
        "tile overlap": overlap,
    }.items():
        if value % SPATIAL_ALIGNMENT:
            raise ValueError(f"{name} must be divisible by {SPATIAL_ALIGNMENT}: {value}")
    ys = make_intervals(height, min(tile_height, height), min(overlap, height - 1))
    xs = make_intervals(width, min(tile_width, width), min(overlap, width - 1))
    if any(value % SPATIAL_ALIGNMENT for interval in (*ys, *xs) for value in (interval.start, interval.end)):
        raise ValueError("tile boundaries must be divisible by 16; use aligned tile/overlap values")
    return [SpatialTile(index, y, x) for index, (y, x) in enumerate((y, x) for y in ys for x in xs)]


def spatial_retry_plans(
    tile_height: int, tile_width: int, max_oom_retries: int
) -> list[tuple[int, int]]:
    """Return the bounded, declared H200 spatial fallback sequence."""
    if max_oom_retries < 0:
        raise ValueError("--max-oom-retries must be non-negative")
    candidates = [(tile_height, tile_width), (640, 1024), (512, 768)]
    plans: list[tuple[int, int]] = []
    for candidate in candidates:
        # A fallback must never increase either dimension supplied by the user.
        if candidate[0] <= tile_height and candidate[1] <= tile_width and candidate not in plans:
            plans.append(candidate)
    return plans[: max_oom_retries + 1]


def exact_coverage_voxels(
    chunks: Sequence[Interval], tiles: Sequence[SpatialTile], real_frames: int
) -> int:
    return sum(
        max(0, min(chunk.valid_end, real_frames) - min(chunk.valid_start, real_frames))
        * (tile.y.valid_end - tile.y.valid_start)
        * (tile.x.valid_end - tile.x.valid_start)
        for chunk in chunks
        for tile in tiles
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight_assets(model_path: Path) -> dict[str, Any]:
    required = [
        REPOSITORY / "inference_script.py",
        model_path / "model_index.json",
        model_path / "transformer" / "config.json",
        model_path / "vae" / "config.json",
        model_path / "scheduler" / "scheduler_config.json",
        model_path / "text_encoder" / "config.json",
        model_path / "tokenizer" / "tokenizer_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing official DOVE pipeline asset(s):\n" + "\n".join(missing))
    missing_weights: list[str] = []
    for component in ("transformer", "vae", "text_encoder"):
        directory = model_path / component
        indexes = sorted(directory.glob("*.index.json"))
        if indexes:
            for index in indexes:
                payload = json.loads(index.read_text(encoding="utf-8"))
                for filename in set(payload.get("weight_map", {}).values()):
                    if not (directory / filename).is_file():
                        missing_weights.append(str(directory / filename))
        elif not any(any(directory.glob(pattern)) for pattern in ("*.safetensors", "*.bin")):
            missing_weights.append(f"{directory}/*.safetensors|*.bin")
    if missing_weights:
        raise FileNotFoundError("missing DOVE model weight file(s):\n" + "\n".join(missing_weights))
    revision = "unknown"
    if (REPOSITORY / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
            check=False, text=True, capture_output=True,
        )
        if result.returncode == 0:
            revision = result.stdout.strip()
    if revision != PINNED_REPOSITORY_REVISION:
        raise RuntimeError(
            f"DOVE repository revision mismatch: {revision}; expected {PINNED_REPOSITORY_REVISION}"
        )
    provenance: dict[str, Any] | None = None
    for name in ("provenance.json", "download_manifest.json"):
        candidate = model_path / name
        if candidate.is_file():
            provenance = json.loads(candidate.read_text(encoding="utf-8"))
            break
    if provenance is None:
        raise FileNotFoundError(
            f"missing checkpoint provenance manifest in {model_path}; "
            "the composite author-DOVE/base-VAE checkpoint is not auditable"
        )
    provenance_text = json.dumps(provenance, sort_keys=True)
    for label, expected in (
        ("author DOVE model", PINNED_DOVE_MODEL_REVISION),
        ("official CogVideoX1.5-5B VAE", PINNED_BASE_VAE_REVISION),
    ):
        if expected not in provenance_text:
            raise RuntimeError(f"provenance manifest does not pin {label} revision {expected}")
    if provenance.get("schema_version") != 1 or not isinstance(provenance.get("files"), dict):
        raise RuntimeError("unsupported or incomplete DOVE provenance manifest schema")
    invalid_manifest_files: list[str] = []
    for relative, record in provenance["files"].items():
        candidate = model_path / relative
        expected_bytes = record.get("bytes") if isinstance(record, dict) else None
        expected_sha256 = record.get("sha256") if isinstance(record, dict) else None
        if (
            not candidate.is_file()
            or expected_bytes is None
            or candidate.stat().st_size != expected_bytes
            or not expected_sha256
            or _sha256(candidate) != expected_sha256
        ):
            invalid_manifest_files.append(relative)
    if invalid_manifest_files:
        raise RuntimeError(
            "DOVE checkpoint files do not match provenance size/SHA-256:\n"
            + "\n".join(invalid_manifest_files)
        )
    return {
        "official_repository": "https://github.com/zhengchen1999/DOVE",
        "repository_path": str(REPOSITORY),
        "repository_revision": revision,
        "model_path": str(model_path),
        "model_index_sha256": _sha256(model_path / "model_index.json"),
        "provenance_manifest": provenance,
    }


def parse_optional_number(text: str) -> float | None:
    value = text.strip().splitlines()[0].strip() if text.strip() else ""
    try:
        return float(value)
    except ValueError:
        return None


class GPUMonitor:
    def __init__(self, gpu_index: int, interval: float = 0.25) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[dict[str, float | None]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)

    def _loop(self) -> None:
        pid = str(os.getpid())
        while not self.stop.is_set():
            try:
                gpu = subprocess.run(
                    ["nvidia-smi", f"--id={self.gpu_index}", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                    check=True, text=True, capture_output=True, timeout=4,
                ).stdout.strip().split(",")
                utilization_result = subprocess.run(
                    ["nvidia-smi", f"--id={self.gpu_index}", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    check=False, text=True, capture_output=True, timeout=4,
                )
                utilization = parse_optional_number(utilization_result.stdout)
                apps = subprocess.run(
                    ["nvidia-smi", f"--id={self.gpu_index}", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                    check=True, text=True, capture_output=True, timeout=4,
                ).stdout.splitlines()
                process = 0.0
                for line in apps:
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 2 and fields[0] == pid:
                        process += float(fields[1])
                self.samples.append({
                    "total": float(gpu[0]), "capacity": float(gpu[1]),
                    "util": utilization, "process": process,
                })
            except (OSError, subprocess.SubprocessError, ValueError):
                self.stop.wait(self.interval)
                continue
            self.stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        utilization = [sample["util"] for sample in self.samples if sample["util"] is not None]
        return {
            "gpu_total_vram_mb": self.samples[-1]["capacity"],
            "peak_process_gpu_memory_mb": max(sample["process"] for sample in self.samples),
            "peak_total_gpu_used_memory_mb": max(sample["total"] for sample in self.samples),
            "average_gpu_utilization_percent": sum(utilization) / len(utilization) if utilization else None,
            "peak_gpu_utilization_percent": max(utilization) if utilization else None,
            "utilization_unavailable_reason": None if utilization else "nvidia-smi utilization.gpu returned [Not Found]",
            "gpu_sample_count": len(self.samples),
        }


def stage_conditioning(
    input_path: Path,
    start_frame: int,
    real_frames: int,
    padded_frames: int,
    output_height: int,
    output_width: int,
    path: Path,
    threads: int,
) -> Any:
    """Create official float32 bilinear conditioning as a disk memmap."""
    import numpy as np
    import torch

    staging = np.memmap(path, mode="w+", dtype=np.float32, shape=(padded_frames, 3, output_height, output_width))
    last: np.ndarray | None = None
    indices = list(range(start_frame, start_frame + real_frames))
    for index, rgb in enumerate(shared_media().decode_selected_frames(input_path, indices, threads=threads)):
        source = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0)
        expanded = torch.nn.functional.interpolate(
            source, size=(output_height, output_width), mode="bilinear", align_corners=False
        )[0].numpy()
        staging[index] = expanded
        last = expanded
        del source, expanded
    if last is None:
        raise RuntimeError("input decoder produced no frames")
    for index in range(real_frames, padded_frames):
        staging[index] = last
    staging.flush()
    return staging


def load_pipeline(model_path: Path, device: str, dtype_name: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    pipe = CogVideoXPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # Compute the official empty-prompt conditioning once, then release the
    # 4.7B T5 encoder before loading the SR transformer and VAE onto the GPU.
    pipe.text_encoder.to(device)
    tokens = pipe.tokenizer(
        "", padding="max_length", max_length=pipe.transformer.config.max_text_seq_length,
        truncation=True, add_special_tokens=True, return_tensors="pt",
    ).input_ids
    with torch.inference_mode():
        prompt = pipe.text_encoder(tokens.to(device))[0].to(dtype=dtype, device="cpu")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.transformer.to(device).eval()
    pipe.vae.to(device).eval()
    details = {
        "precision": dtype_name,
        "sr_noise_step": SR_TIMESTEP,
        "scheduler": "CogVideoXDPMScheduler(timestep_spacing=trailing)",
        "neural_path": "bilinear x4 -> VAE encode/sample -> one transformer call -> scheduler velocity -> VAE decode",
    }
    return pipe, prompt, details


def infer_tile(pipe: Any, prompt_cpu: Any, video: Any, device: str) -> tuple[Any, dict[str, float]]:
    """Run the exact official DOVE neural operations with split timings."""
    import torch

    # Upstream wraps process_video in @no_grad.  Inference mode is the stronger
    # equivalent and is essential here: retaining a 5B-transformer autograd
    # graph would invalidate both the fidelity benchmark and its VRAM result.
    with torch.inference_mode():
        return _infer_tile_impl(pipe, prompt_cpu, video, device)


def _infer_tile_impl(pipe: Any, prompt_cpu: Any, video: Any, device: str) -> tuple[Any, dict[str, float]]:
    import torch
    from diffusers.models.embeddings import get_3d_rotary_pos_embed

    timings = {"vae_encode_seconds": 0.0, "diffusion_sampling_seconds": 0.0, "decode_seconds": 0.0}
    # Upstream performs x / 255 * 2 - 1 while the decoded tensor is still
    # float32, then casts to the VAE dtype. Preserve that order: doing the
    # division after BF16 conversion measurably changes common 8-bit levels.
    video = video.div(127.5).sub(1.0).to(
        device=pipe.vae.device, dtype=pipe.vae.dtype
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    latent = pipe.vae.encode(video).latent_dist.sample() * pipe.vae.config.scaling_factor
    torch.cuda.synchronize()
    timings["vae_encode_seconds"] = time.perf_counter() - started

    patch_t = pipe.transformer.config.patch_size_t
    copied = 0
    if patch_t is not None:
        copied = latent.shape[2] % patch_t
        if copied:
            latent = torch.cat([latent[:, :, :1].repeat(1, 1, copied, 1, 1), latent], dim=2)
        if latent.shape[2] % patch_t:
            raise RuntimeError("DOVE latent temporal dimension is not patch aligned")
    _, _, frames, height, width = latent.shape
    hidden = latent.permute(0, 2, 1, 3, 4)
    prompt = prompt_cpu.to(device=hidden.device, dtype=hidden.dtype)
    timestep = torch.full((hidden.shape[0],), SR_TIMESTEP, dtype=torch.long, device=hidden.device)
    rotary = None
    if pipe.transformer.config.use_rotary_positional_embeddings:
        grid_h = height // pipe.transformer.config.patch_size
        grid_w = width // pipe.transformer.config.patch_size
        temporal = frames if patch_t is None else (frames + patch_t - 1) // patch_t
        rotary = get_3d_rotary_pos_embed(
            embed_dim=pipe.transformer.config.attention_head_dim,
            crops_coords=None, grid_size=(grid_h, grid_w), temporal_size=temporal,
            grid_type="slice", max_size=(grid_h, grid_w), device=hidden.device,
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    predicted = pipe.transformer(
        hidden_states=hidden, encoder_hidden_states=prompt, timestep=timestep,
        image_rotary_emb=rotary, return_dict=False,
    )[0]
    generated = pipe.scheduler.get_velocity(predicted, hidden, timestep)
    torch.cuda.synchronize()
    timings["diffusion_sampling_seconds"] = time.perf_counter() - started
    if copied:
        generated = generated[:, copied:]
    started = time.perf_counter()
    decoded = pipe.decode_latents(generated)
    decoded = (decoded * 0.5 + 0.5).clamp_(0, 1)
    torch.cuda.synchronize()
    timings["decode_seconds"] = time.perf_counter() - started
    return decoded, timings


def _output_base(input_path: Path, mode: str, start: int, count: int) -> str:
    return f"{input_path.stem}_{MODEL_NAME}_x4_{mode}_f{start:06d}_n{count}"


def _report_markdown(report: dict[str, Any]) -> str:
    timing, gpu = report.get("timing", {}), report.get("gpu", {})
    return "\n".join([
        f"# DOVE {report['benchmark_mode']} benchmark",
        "",
        f"- Status: **{report['status']}**",
        f"- Frames: {report.get('effective_input_frames_processed', 0)}",
        f"- Spatial tiles: {report.get('spatial_tile_count', 0)}",
        f"- Temporal chunks: {report.get('temporal_chunk_count', 0)}",
        f"- Total: {timing.get('total_wall_seconds')} s",
        f"- Neural: {timing.get('neural_inference_seconds')} s",
        f"- Decode: {timing.get('decode_seconds')} s",
        f"- End-to-end/input frame: {timing.get('end_to_end_seconds_per_input_frame')} s",
        f"- Peak torch allocated: {gpu.get('peak_torch_allocated_mib')} MiB",
        f"- Peak process GPU: {gpu.get('peak_process_gpu_memory_mb')} MiB",
        f"- Output: `{report.get('output_path')}`",
        "",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--report-json", help="explicit benchmark JSON path (orchestrator contract)")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--benchmark-mode", choices=("micro", "mini", "full"), default="micro")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--tile-height", type=int, default=768, help="target/output-domain tile height")
    parser.add_argument("--tile-width", type=int, default=1360, help="target/output-domain tile width")
    parser.add_argument("--tile-overlap", type=int, default=32, help="target/output-domain overlap")
    parser.add_argument("--max-oom-retries", type=int, default=2, help="bounded spatial tile reductions after CUDA OOM")
    parser.add_argument("--temporal-window", type=int, default=17)
    parser.add_argument("--temporal-overlap", type=int, default=8)
    parser.add_argument("--ffmpeg-threads", type=int, default=16)
    parser.add_argument("--encoder", default="auto")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument("--scratch-root", default=os.environ.get("DOVE_SCRATCH_ROOT", "/tmp"))
    parser.add_argument("--keep-scratch-on-failure", action="store_true")
    parser.add_argument("--probe-only", action="store_true", help="validate local official assets without loading weights")
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    wall_started = time.perf_counter()
    input_path, model_path = Path(args.input).resolve(), Path(args.model_path).resolve()
    output_dir, result_dir = Path(args.output_dir).resolve(), Path(args.result_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata = shared_media().probe_media(input_path, count_frames=True)
    start, real_frames = benchmark_frame_range(
        args.benchmark_mode, metadata.frame_count, args.start_frame, args.max_frames
    )
    padded_frames = padded_frame_count(real_frames)
    output_height, output_width = metadata.height * SCALE, metadata.width * SCALE
    chunks = make_temporal_chunks(padded_frames, min(args.temporal_window, padded_frames), args.temporal_overlap)
    retry_plans = spatial_retry_plans(args.tile_height, args.tile_width, args.max_oom_retries)
    if not retry_plans:
        raise ValueError("no valid spatial execution plan")
    initial_tiles = make_spatial_tiles(
        output_height, output_width, retry_plans[0][0], retry_plans[0][1], args.tile_overlap
    )
    provenance = preflight_assets(model_path)
    base = _output_base(input_path, args.benchmark_mode, start, real_frames)
    report_path = (
        Path(args.report_json).resolve()
        if args.report_json
        else result_dir / f"benchmark_{args.benchmark_mode}.json"
    )
    report_markdown_path = report_path.with_suffix(".md")
    Path(args.scratch_root).mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"dove_{args.benchmark_mode}_", dir=args.scratch_root))
    timings = {
        "model_load_seconds": 0.0, "input_decode_and_staging_seconds": 0.0,
        "vae_encode_seconds": 0.0, "diffusion_sampling_seconds": 0.0,
        "neural_inference_seconds": 0.0, "decode_seconds": 0.0,
        "merge_seconds": 0.0, "encode_seconds": 0.0, "total_wall_seconds": 0.0,
    }
    report: dict[str, Any] = {
        "model": MODEL_NAME, "status": "RUNNING", "pipeline_mode": "official_pretrained_dove_one_step",
        "benchmark_mode": args.benchmark_mode, "input": metadata.to_dict(), "output_path": None,
        "effective_input_frames_processed": real_frames, "effective_output_frames_written": 0,
        "frame_selection": {"start_frame": start, "real_frame_count": real_frames, "padded_frame_count": padded_frames},
        "temporal_chunk_count": len(chunks),
        "native_temporal_unit_count": (padded_frames - 1) // NATIVE_TEMPORAL_STRIDE,
        "spatial_tile_count": len(initial_tiles),
        "tile_count": len(chunks) * len(initial_tiles),
        # The common temporal analyzer expects transition-frame indices, not
        # [start, end] ranges. Frame zero is the initial chunk, not a boundary.
        "chunk_boundaries": [chunk.start for chunk in chunks[1:]],
        "temporal_chunk_ranges": [[chunk.start, chunk.end] for chunk in chunks],
        "spatial_tile_shape_output": [args.tile_height, args.tile_width],
        "spatial_overlap_output": args.tile_overlap,
        "merge_strategy": "official center-valid crop, exact single-write placement",
        "provenance": provenance, "timing": timings, "timings": timings,
        "timings_detail": {
            "intermediate_tile_staging_seconds": 0.0,
            "oom_retry_overhead_seconds": 0.0,
        },
        "spatial_attempts": [], "gpu": {},
    }
    encoder_process = None
    monitor: GPUMonitor | None = None
    try:
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu_index)
        torch.cuda.reset_peak_memory_stats(args.gpu_index)
        with GPUMonitor(args.gpu_index, args.gpu_sample_interval) as monitor:
            log(
                f"staging {real_frames} real / {padded_frames} padded frames "
                f"at {output_width}x{output_height}"
            )
            started = time.perf_counter()
            conditioning = stage_conditioning(
                input_path, start, real_frames, padded_frames, output_height, output_width,
                scratch / "conditioning.f32", args.ffmpeg_threads,
            )
            timings["input_decode_and_staging_seconds"] = time.perf_counter() - started

            log("loading pinned official DOVE pipeline")
            started = time.perf_counter()
            pipe, prompt, model_details = load_pipeline(model_path, f"cuda:{args.gpu_index}", args.dtype)
            timings["model_load_seconds"] = time.perf_counter() - started
            model_details["checkpoint_provenance"] = provenance["provenance_manifest"]
            report["model_details"] = model_details

            output_path = scratch / "output.rgb"
            output: np.memmap | None = None
            selected_tiles: list[SpatialTile] | None = None
            array = tensor = generated = None
            for attempt_index, (tile_height, tile_width) in enumerate(retry_plans):
                tiles = make_spatial_tiles(
                    output_height, output_width, tile_height, tile_width, args.tile_overlap
                )
                attempt = {
                    "attempt": attempt_index + 1, "tile_height": tile_height,
                    "tile_width": tile_width, "tile_overlap": args.tile_overlap,
                    "spatial_tile_count": len(tiles), "status": "RUNNING",
                }
                report["spatial_attempts"].append(attempt)
                output_path.unlink(missing_ok=True)
                output = np.memmap(
                    output_path, mode="w+", dtype=np.uint8,
                    shape=(padded_frames, output_height, output_width, 3),
                )
                torch.manual_seed(args.seed)
                array = tensor = generated = None
                completed_tiles = 0
                total_tiles = len(chunks) * len(tiles)
                attempt_started = time.perf_counter()
                component_keys = (
                    "vae_encode_seconds",
                    "diffusion_sampling_seconds",
                    "decode_seconds",
                    "merge_seconds",
                )
                component_before = {key: timings[key] for key in component_keys}
                staging_before = report["timings_detail"][
                    "intermediate_tile_staging_seconds"
                ]
                progress_interval = max(1, total_tiles // 20)
                log(
                    f"spatial attempt {attempt_index + 1}/{len(retry_plans)}: "
                    f"{tile_width}x{tile_height}, {len(tiles)} tiles/chunk, "
                    f"{len(chunks)} temporal chunk(s)"
                )
                try:
                    for chunk in chunks:
                        for tile in tiles:
                            staging_started = time.perf_counter()
                            array = np.asarray(
                                conditioning[
                                    chunk.start:chunk.end, :, tile.y.start:tile.y.end, tile.x.start:tile.x.end
                                ]
                            ).copy()
                            report["timings_detail"]["intermediate_tile_staging_seconds"] += (
                                time.perf_counter() - staging_started
                            )
                            tensor = torch.from_numpy(array).permute(1, 0, 2, 3).unsqueeze(0)
                            generated, item = infer_tile(pipe, prompt, tensor, f"cuda:{args.gpu_index}")
                            timings["vae_encode_seconds"] += item["vae_encode_seconds"]
                            timings["diffusion_sampling_seconds"] += item["diffusion_sampling_seconds"]
                            timings["decode_seconds"] += item["decode_seconds"]
                            merge_started = time.perf_counter()
                            frame0, frame1 = chunk.valid_start, min(chunk.valid_end, real_frames)
                            if frame1 > frame0:
                                local_t0, local_t1 = frame0 - chunk.start, frame1 - chunk.start
                                local_y0, local_y1 = tile.y.valid_start - tile.y.start, tile.y.valid_end - tile.y.start
                                local_x0, local_x1 = tile.x.valid_start - tile.x.start, tile.x.valid_end - tile.x.start
                                cropped = generated[0, :, local_t0:local_t1, local_y0:local_y1, local_x0:local_x1]
                                # Match upstream save_video: multiply, clamp,
                                # then uint8 truncation (not nearest rounding).
                                rgb = cropped.permute(1, 2, 3, 0).mul(255).to(torch.uint8).cpu().numpy()
                                output[
                                    frame0:frame1, tile.y.valid_start:tile.y.valid_end,
                                    tile.x.valid_start:tile.x.valid_end,
                                ] = rgb
                            timings["merge_seconds"] += time.perf_counter() - merge_started
                            completed_tiles += 1
                            if completed_tiles == total_tiles or completed_tiles % progress_interval == 0:
                                elapsed = time.perf_counter() - attempt_started
                                eta = elapsed / completed_tiles * (total_tiles - completed_tiles)
                                log(
                                    f"attempt {attempt_index + 1}: {completed_tiles}/{total_tiles} "
                                    f"tile-chunks, elapsed {elapsed:.1f}s, ETA {eta:.1f}s, "
                                    f"torch reserved {torch.cuda.memory_reserved(args.gpu_index) / 2**20:.0f} MiB"
                                )
                    output.flush()
                    expected_voxels = real_frames * output_height * output_width
                    coverage_voxels = exact_coverage_voxels(chunks, tiles, real_frames)
                    if coverage_voxels != expected_voxels:
                        raise RuntimeError(
                            f"invalid tile coverage: {coverage_voxels}, expected {expected_voxels}"
                        )
                    attempt["status"] = "PASS"
                    attempt["timing"] = {
                        key: timings[key] - component_before[key]
                        for key in component_keys
                    }
                    attempt["timing"]["intermediate_tile_staging_seconds"] = (
                        report["timings_detail"]["intermediate_tile_staging_seconds"]
                        - staging_before
                    )
                    attempt["timing"]["wall_seconds"] = (
                        time.perf_counter() - attempt_started
                    )
                    selected_tiles = tiles
                    report["selected_spatial_plan"] = {
                        "tile_height": tile_height, "tile_width": tile_width,
                        "tile_overlap": args.tile_overlap,
                    }
                    report["spatial_tile_count"] = len(tiles)
                    report["tile_count"] = len(chunks) * len(tiles)
                    report["spatial_tile_shape_output"] = [tile_height, tile_width]
                    report["coverage"] = {
                        "expected_frame_pixels": expected_voxels,
                        "written_frame_pixels": coverage_voxels,
                        "exact_single_write": True,
                    }
                    break
                except torch.cuda.OutOfMemoryError as oom:
                    failed_wall = time.perf_counter() - attempt_started
                    attempt.update({
                        "status": "OOM",
                        "error": str(oom),
                        "timing": {
                            **{
                                key: timings[key] - component_before[key]
                                for key in component_keys
                            },
                            "intermediate_tile_staging_seconds": (
                                report["timings_detail"][
                                    "intermediate_tile_staging_seconds"
                                ]
                                - staging_before
                            ),
                            "wall_seconds": failed_wall,
                        },
                    })
                    for key in component_keys:
                        timings[key] = component_before[key]
                    report["timings_detail"][
                        "intermediate_tile_staging_seconds"
                    ] = staging_before
                    report["timings_detail"][
                        "oom_retry_overhead_seconds"
                    ] += failed_wall
                    del tensor, generated, array
                    del output
                    output = None
                    output_path.unlink(missing_ok=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if attempt_index + 1 >= len(retry_plans):
                        raise
            if output is None or selected_tiles is None:
                raise RuntimeError("DOVE spatial execution did not select a successful plan")
            del tensor, generated, array  # noqa: F821 - set by the successful retry
            report["spatial_oom_retry_count"] = sum(
                attempt["status"] == "OOM" for attempt in report["spatial_attempts"]
            )
            output.flush()
            timings["neural_inference_seconds"] = timings["vae_encode_seconds"] + timings["diffusion_sampling_seconds"]

            encoder, attempts = shared_media().select_encoder(
                output_width, output_height, threads=args.ffmpeg_threads,
                scratch_dir=scratch, preferred=args.encoder,
            )
            report["encoder"] = {"selected": encoder["name"], "attempts": attempts}
            log(f"encoding {real_frames} frames with {encoder['name']}")
            video_only = scratch / f"video_only{encoder['extension']}"
            started = time.perf_counter()
            encoder_process = shared_media().RawVideoEncoder(
                video_only, width=output_width, height=output_height, fps=metadata.fps,
                frame_count=real_frames, encoder=encoder, threads=args.ffmpeg_threads,
                source_metadata=metadata,
            )
            for frame in range(real_frames):
                encoder_process.write(output[frame])
                encoder_process.frame_complete()
            encoder_process.close()
            encoder_process = None
            destination = output_dir / f"{base}{encoder['extension']}"
            shared_media().atomic_publish_file(video_only, destination)
            timings["encode_seconds"] = time.perf_counter() - started
            report["output_path"] = str(destination)
            report["effective_output_frames_written"] = real_frames
            report["decoded_source_frames"] = real_frames
        gpu = monitor.summary()
        gpu.update({
            "peak_torch_allocated_mib": torch.cuda.max_memory_allocated(args.gpu_index) / 2**20,
            "peak_torch_reserved_mib": torch.cuda.max_memory_reserved(args.gpu_index) / 2**20,
        })
        gpu["peak_process_gpu_mib"] = gpu.get("peak_process_gpu_memory_mb")
        gpu["peak_total_gpu_used_mib"] = gpu.get("peak_total_gpu_used_memory_mb")
        report["gpu"] = gpu
        timings["total_wall_seconds"] = time.perf_counter() - wall_started
        timings.update({
            "pure_inference_seconds_per_input_frame": (timings["neural_inference_seconds"] + timings["decode_seconds"]) / real_frames,
            "end_to_end_seconds_per_input_frame": timings["total_wall_seconds"] / real_frames,
            "pure_inference_seconds_per_output_frame": (timings["neural_inference_seconds"] + timings["decode_seconds"]) / real_frames,
            "end_to_end_seconds_per_output_frame": timings["total_wall_seconds"] / real_frames,
        })
        report["status"] = "PASS"
        for key, value in timings.items():
            report[key] = value
        report["peak_torch_allocated_mib"] = gpu.get("peak_torch_allocated_mib")
        report["peak_torch_reserved_mib"] = gpu.get("peak_torch_reserved_mib")
        report["peak_process_gpu_mib"] = gpu.get("peak_process_gpu_mib")
        report["peak_total_gpu_used_mib"] = gpu.get("peak_total_gpu_used_mib")
        shared_media().atomic_write_json(report_path, report)
        shared_media().atomic_write_text(report_markdown_path, _report_markdown(report))
        shutil.rmtree(scratch)
        return report
    except BaseException as error:
        if encoder_process is not None:
            with contextlib.suppress(Exception):
                encoder_process.abort()
        timings["total_wall_seconds"] = time.perf_counter() - wall_started
        if monitor is not None:
            report["gpu"].update(monitor.summary())
        with contextlib.suppress(Exception):
            report["gpu"].update({
                "peak_torch_allocated_mib": torch.cuda.max_memory_allocated(args.gpu_index) / 2**20,
                "peak_torch_reserved_mib": torch.cuda.max_memory_reserved(args.gpu_index) / 2**20,
            })
        report["gpu"]["peak_process_gpu_mib"] = report["gpu"].get(
            "peak_process_gpu_memory_mb"
        )
        report["gpu"]["peak_total_gpu_used_mib"] = report["gpu"].get(
            "peak_total_gpu_used_memory_mb"
        )
        for key, value in timings.items():
            report[key] = value
        report["peak_torch_allocated_mib"] = report["gpu"].get(
            "peak_torch_allocated_mib"
        )
        report["peak_torch_reserved_mib"] = report["gpu"].get(
            "peak_torch_reserved_mib"
        )
        report["peak_process_gpu_mib"] = report["gpu"].get(
            "peak_process_gpu_mib"
        )
        report["peak_total_gpu_used_mib"] = report["gpu"].get(
            "peak_total_gpu_used_mib"
        )
        report.update({"status": "FAIL", "error": str(error), "traceback": traceback.format_exc(), "scratch_path": str(scratch)})
        with contextlib.suppress(Exception):
            shared_media().atomic_write_json(report_path, report)
            shared_media().atomic_write_text(report_markdown_path, _report_markdown(report))
        if not args.keep_scratch_on_failure:
            shutil.rmtree(scratch, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    handler = _OneShotSignalHandler()
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        managed = getattr(signal, name, None)
        if managed is not None:
            signal.signal(managed, handler)
    args = build_parser().parse_args(argv)
    if args.probe_only:
        print(json.dumps(preflight_assets(Path(args.model_path).resolve()), indent=2, ensure_ascii=False))
        return 0
    report = run_benchmark(args)
    print(json.dumps({"status": report["status"], "output_path": report["output_path"], "timing": report["timing"], "gpu": report["gpu"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
