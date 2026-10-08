#!/usr/bin/env python3
"""FlashVSR v1.1 pretrained 4x video upscaler.

This is a production wrapper around OpenImagingLab's official
``FlashVSRTinyLongPipeline``.  It deliberately keeps the official continuous
causal/KV inference loop.  Spatial source tiles are independent continuous
video streams and are blended only after every tile has been written to the
runtime scratch directory; frames are never super-resolved independently.

No training or weight modification is performed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence


HERE = Path(__file__).resolve().parent
ARCHITECT = HERE.parent
ROOT = ARCHITECT.parent
# Packaged certification assets live beside ``flashvsr`` under ``src/models``.
# Only the asset root differs from the reference snapshot; the official model
# repository and checkpoint layout below remain unchanged.
PACKAGE_SRC = HERE.parents[2]
MODEL_ROOT = PACKAGE_SRC / "models" / "flashvsr_v1_1"
REPOSITORY = MODEL_ROOT / "repository"
CHECKPOINT = MODEL_ROOT / "checkpoint"
DEFAULT_INPUT = ROOT / "input" / "0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "test" / "FlashVSR_v1.1"
DEFAULT_RESULT_DIR = ROOT / "result" / "test" / "FlashVSR_v1.1"
MODEL_NAME = "FlashVSR_v1.1"
SCALE = 4
MODEL_ALIGNMENT = 128  # official LQ projection operates on 16x16 source blocks
STAGE_MANIFEST_SCHEMA = 1
MICRO_FRAMES = 21  # one official TinyLong warm-up/causal emission step
MINI_FRAMES = 29   # warm-up plus one steady-state 8-frame streaming step


def run(command: list[str], *, check: bool = True, text: bool = True, **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=text, **kwargs)


def read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def ffprobe(path: Path) -> dict[str, Any]:
    """Read authoritative stream metadata, counting decoded video frames."""
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-show_streams",
        "-show_format", "-of", "json", str(path),
    ]
    data = json.loads(run(command, capture_output=True).stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream: {path}")
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    rate_text = video.get("avg_frame_rate") or video.get("r_frame_rate")
    fps = Fraction(rate_text)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS reported by ffprobe: {rate_text!r}")
    frames_text = video.get("nb_read_frames") or video.get("nb_frames")
    if not frames_text or frames_text == "N/A":
        raise RuntimeError("ffprobe could not determine the input frame count")
    duration = video.get("duration") or data.get("format", {}).get("duration")
    result = {
        "path": str(path.resolve()),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": float(fps),
        "fps_fraction": str(fps),
        "frame_count": int(frames_text),
        "duration": float(duration) if duration not in (None, "N/A") else int(frames_text) / float(fps),
        "pixel_format": video.get("pix_fmt"),
        "audio_present": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
    }
    return result


def print_metadata(meta: dict[str, Any]) -> None:
    print(f"Input path    : {meta['path']}", flush=True)
    print(f"Resolution    : {meta['width']}x{meta['height']}", flush=True)
    print(f"FPS           : {meta['fps_fraction']} ({meta['fps']:.8f})", flush=True)
    print(f"Frame count   : {meta['frame_count']}", flush=True)
    print(f"Duration      : {meta['duration']:.6f} sec", flush=True)
    print(f"Pixel format  : {meta['pixel_format']}", flush=True)
    print(f"Audio presence: {meta['audio_present']}", flush=True)


def require_assets() -> None:
    required = [
        REPOSITORY / "diffsynth" / "pipelines" / "flashvsr_tiny_long.py",
        REPOSITORY / "examples" / "WanVSR" / "utils" / "utils.py",
        REPOSITORY / "examples" / "WanVSR" / "utils" / "TCDecoder.py",
        REPOSITORY / "examples" / "WanVSR" / "prompt_tensor" / "posi_prompt.pth",
        CHECKPOINT / "diffusion_pytorch_model_streaming_dmd.safetensors",
        CHECKPOINT / "LQ_proj_in.ckpt",
        CHECKPOINT / "TCDecoder.ckpt",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing official FlashVSR asset(s):\n" + "\n".join(missing))


def ceil_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def padded_model_frames(real_frames: int) -> int:
    """Smallest 8n+1 length whose official long pipeline emits >= real_frames."""
    # The long pipeline emits F-4 frames and needs at least F=25 for one step.
    return max(25, ceil_multiple(real_frames + 3, 8) + 1)


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile >= length:
        return [0]
    stride = tile - overlap
    starts = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


@dataclass(frozen=True)
class Tile:
    index: int
    x: int
    y: int
    width: int
    height: int


def make_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    xs = tile_starts(width, min(tile_size, width), overlap)
    ys = tile_starts(height, min(tile_size, height), overlap)
    tiles: list[Tile] = []
    for y in ys:
        for x in xs:
            tiles.append(Tile(len(tiles), x, y, min(tile_size, width - x), min(tile_size, height - y)))
    return tiles


class GPUMonitor:
    """Lightweight nvidia-smi sampler scoped to this inference process."""

    def __init__(self, gpu_index: int, interval: float):
        self.gpu_index = gpu_index
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "GPUMonitor":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(2.0, self.interval * 3))

    def _loop(self) -> None:
        pid = os.getpid()
        while not self.stop_event.is_set():
            try:
                gpu = run([
                    "nvidia-smi", f"--id={self.gpu_index}",
                    "--query-gpu=memory.used,utilization.gpu,memory.total,name",
                    "--format=csv,noheader,nounits",
                ], capture_output=True, timeout=5).stdout.strip().split(", ")
                apps = run([
                    "nvidia-smi", f"--id={self.gpu_index}",
                    "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits",
                ], capture_output=True, timeout=5).stdout.splitlines()
                process_mb = 0.0
                for line in apps:
                    fields = [v.strip() for v in line.split(",")]
                    if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
                        process_mb += float(fields[1])
                try:
                    utilization = float(gpu[1])
                except (ValueError, IndexError):
                    utilization = None
                self.samples.append({
                    "total_used_mb": float(gpu[0]), "utilization_percent": utilization,
                    "total_vram_mb": float(gpu[2]), "process_used_mb": process_mb,
                    "gpu_name": gpu[3] if len(gpu) > 3 else "unknown",
                })
            except Exception:
                pass
            self.stop_event.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        utilization = [
            float(sample["utilization_percent"])
            for sample in self.samples
            if sample["utilization_percent"] is not None
        ]
        return {
            "gpu_name": self.samples[-1]["gpu_name"],
            "gpu_total_vram_mb": self.samples[-1]["total_vram_mb"],
            "peak_process_gpu_memory_mb": max(s["process_used_mb"] for s in self.samples),
            "peak_total_gpu_used_memory_mb": max(s["total_used_mb"] for s in self.samples),
            "average_gpu_utilization_percent": sum(utilization) / len(utilization) if utilization else None,
            "peak_gpu_utilization_percent": max(utilization) if utilization else None,
            "utilization_unavailable_reason": None if utilization else "nvidia-smi utilization.gpu returned [Not Found]",
            "gpu_sample_count": len(self.samples),
        }


class TileFrameStore:
    """Disk-backed upscaled/padded conditioning frames for one spatial tile."""

    def __init__(self, path: Path, shape: tuple[int, int, int, int], real_frames: int):
        import numpy as np
        self.path = path
        self.shape = shape
        self.real_frames = real_frames
        self.array = np.memmap(path, mode="r", dtype=np.uint8, shape=shape)

    def get(self, start: int, end: int, *, device: str, dtype: Any) -> Any:
        import numpy as np
        import torch
        data = np.asarray(self.array[start:end]).copy()
        tensor = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0)
        return tensor.to(device=device, dtype=dtype).div_(127.5).sub_(1.0)


def build_tile_store(
    input_path: Path,
    tile: Tile,
    total_frames: int,
    padded_frames: int,
    scratch: Path,
    ffmpeg_threads: int,
    start_frame: int = 0,
) -> TileFrameStore:
    import cv2
    import numpy as np

    cv2.setNumThreads(max(1, ffmpeg_threads))
    out_h, out_w = tile.height * SCALE, tile.width * SCALE
    pad_h, pad_w = ceil_multiple(out_h, MODEL_ALIGNMENT), ceil_multiple(out_w, MODEL_ALIGNMENT)
    store_path = scratch / f"conditioning_{tile.index:04d}.rgb"
    array = np.memmap(store_path, mode="w+", dtype=np.uint8, shape=(padded_frames, pad_h, pad_w, 3))
    command = [
        "ffmpeg", "-v", "error", "-threads", str(max(1, ffmpeg_threads)),
        "-i", str(input_path), "-map", "0:v:0",
        "-vf", f"select=gte(n\\,{start_frame}),crop={tile.width}:{tile.height}:{tile.x}:{tile.y}",
        "-vsync", "0", "-frames:v", str(total_frames),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    source_bytes = tile.width * tile.height * 3
    last = None
    try:
        for frame_index in range(total_frames):
            raw = read_exact(process.stdout, source_bytes)
            if len(raw) != source_bytes:
                stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
                raise RuntimeError(f"ffmpeg ended at frame {frame_index}/{total_frames}: {stderr}")
            source = np.frombuffer(raw, np.uint8).reshape(tile.height, tile.width, 3)
            scaled = cv2.resize(source, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            if pad_h != out_h or pad_w != out_w:
                scaled = np.pad(scaled, ((0, pad_h - out_h), (0, pad_w - out_w), (0, 0)), mode="edge")
            array[frame_index] = scaled
            last = scaled
        if last is None:
            raise RuntimeError("Input contains no decodable frames")
        for frame_index in range(total_frames, padded_frames):
            array[frame_index] = last
        array.flush()
    finally:
        if process.stdout:
            process.stdout.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if process.returncode not in (0, None):
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        raise RuntimeError(f"ffmpeg tile decode failed ({process.returncode}): {stderr}")
    del array
    return TileFrameStore(store_path, (padded_frames, pad_h, pad_w, 3), total_frames)


def initialize_pipeline(device: str) -> tuple[Any, dict[str, Any]]:
    """Load only official v1.1 pretrained modules (never train/fine-tune)."""
    require_assets()
    sys.path.insert(0, str(REPOSITORY))
    sys.path.insert(0, str(REPOSITORY / "examples" / "WanVSR"))
    import torch
    from diffsynth import FlashVSRTinyLongPipeline, ModelManager
    from utils.TCDecoder import build_tcdecoder
    from utils.utils import Causal_LQ4x_Proj

    started = time.perf_counter()
    manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    manager.load_models([str(CHECKPOINT / "diffusion_pytorch_model_streaming_dmd.safetensors")])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(manager, device=device)
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(
        device, dtype=torch.bfloat16
    )
    pipe.denoising_model().LQ_proj_in.load_state_dict(
        torch.load(CHECKPOINT / "LQ_proj_in.ckpt", map_location="cpu", weights_only=True), strict=True
    )
    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
    incompatible = pipe.TCDecoder.load_state_dict(
        torch.load(CHECKPOINT / "TCDecoder.ckpt", map_location="cpu", weights_only=True), strict=False
    )
    pipe.to(device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    context = torch.load(
        REPOSITORY / "examples" / "WanVSR" / "prompt_tensor" / "posi_prompt.pth",
        map_location="cpu", weights_only=True,
    )
    pipe.init_cross_kv(context_tensor=context)
    pipe.load_models_to_device(["dit", "vae"])
    return pipe, {
        "seconds": time.perf_counter() - started,
        "tcdecoder_missing_keys": list(incompatible.missing_keys),
        "tcdecoder_unexpected_keys": list(incompatible.unexpected_keys),
    }


def stream_official_tile(
    pipe: Any,
    store: TileFrameStore,
    padded_frames: int,
    real_frames: int,
    seed: int,
    sparse_ratio: float,
    local_range: int,
    callback: Callable[[int, Any], None],
    timing: dict[str, float] | None = None,
    emission_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Run one complete causal tile timeline without constructing autograd graphs."""
    import torch

    # The upstream pipeline's public call is decorated with @torch.no_grad().
    # This callback-based adaptation must preserve that inference invariant;
    # otherwise recurrent KV/LQ/decoder caches retain graphs across the video.
    with torch.inference_mode():
        return _stream_official_tile_impl(
            pipe,
            store,
            padded_frames,
            real_frames,
            seed,
            sparse_ratio,
            local_range,
            callback,
            timing,
            emission_callback,
        )


def _stream_official_tile_impl(
    pipe: Any,
    store: TileFrameStore,
    padded_frames: int,
    real_frames: int,
    seed: int,
    sparse_ratio: float,
    local_range: int,
    callback: Callable[[int, Any], None],
    timing: dict[str, float] | None = None,
    emission_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Official long-video loop with a callback instead of accumulating frames.

    The DiT KV cache, causal LQ projection cache and tiny decoder cache remain
    alive for the complete tile timeline.  Only conditioning slices and decoded
    output chunks are materialized, avoiding a full 16K sequence in RAM/VRAM.
    """
    import torch
    from diffsynth.models.wan_video_dit import precompute_freqs_cis_3d
    from diffsynth.pipelines.flashvsr_tiny_long import model_fn_wan_video

    device, dtype = pipe.device, pipe.torch_dtype
    out_h, out_w = store.shape[1], store.shape[2]
    process_total = (padded_frames - 1) // 8 - 2
    if process_total < 1:
        raise RuntimeError(f"Internal padded frame count is too short: {padded_frames}")
    # Upstream precomputes 1024 RoPE positions. Extend them deterministically
    # instead of imposing an input-length limit when a longer video needs more.
    required_rope = max((padded_frames - 1) // 4 + 1, out_h // 16 + 1, out_w // 16 + 1)
    if pipe.dit.freqs[0].shape[0] < required_rope:
        block = pipe.dit.blocks[0]
        pipe.dit.freqs = precompute_freqs_cis_3d(block.dim // block.num_heads, end=required_rope)
    pipe.dit.LQ_proj_in.clear_cache()
    pipe.TCDecoder.clean_mem()
    generator = torch.Generator(device=device).manual_seed(seed)
    pre_cache_k = [None] * len(pipe.dit.blocks)
    pre_cache_v = [None] * len(pipe.dit.blocks)
    lq_previous = 0
    written = 0

    for process_index in range(process_total):
        emission_started = time.perf_counter()
        emission_frame_start = written
        lq_latents = None
        if process_index == 0:
            slices = [(max(0, i * 4 - 3), (i + 1) * 4 - 3) for i in range(7)]
            latent_t = 6
            lq_current = 21
        else:
            slices = [
                (process_index * 8 + 17 + i * 4, process_index * 8 + 21 + i * 4)
                for i in range(2)
            ]
            latent_t = 2
            lq_current = process_index * 8 + 21
        for start, end in slices:
            condition = store.get(start, end, device=device, dtype=dtype)
            projection_started = time.perf_counter()
            current = pipe.dit.LQ_proj_in.stream_forward(condition)
            if timing is not None:
                timing["neural_inference_seconds"] = timing.get("neural_inference_seconds", 0.0) + (
                    time.perf_counter() - projection_started
                )
            del condition
            if current is None:
                continue
            if lq_latents is None:
                lq_latents = current
            else:
                for layer_index in range(len(lq_latents)):
                    lq_latents[layer_index] = torch.cat((lq_latents[layer_index], current[layer_index]), dim=1)

        latent = torch.randn(
            (1, 16, latent_t, out_h // 8, out_w // 8), generator=generator,
            device=device, dtype=dtype,
        )
        neural_started = time.perf_counter()
        predicted, pre_cache_k, pre_cache_v = model_fn_wan_video(
            pipe.dit, x=latent, timestep=pipe.timestep, context=None, tea_cache=None,
            use_unified_sequence_parallel=False, LQ_latents=lq_latents,
            is_full_block=False, is_stream=True, pre_cache_k=pre_cache_k,
            pre_cache_v=pre_cache_v,
            topk_ratio=sparse_ratio * 768 * 1280 / (out_h * out_w),
            kv_ratio=3.0, cur_process_idx=process_index, t_mod=pipe.t_mod,
            t=pipe.t, local_range=local_range,
        )
        latent.sub_(predicted)
        if timing is not None:
            timing["neural_inference_seconds"] = timing.get("neural_inference_seconds", 0.0) + (
                time.perf_counter() - neural_started
            )
        condition = store.get(lq_previous, lq_current, device=device, dtype=dtype)
        decode_started = time.perf_counter()
        decoded = pipe.TCDecoder.decode_video(
            latent.transpose(1, 2), parallel=False, show_progress_bar=False, cond=condition
        ).transpose(1, 2).mul_(2).sub_(1)
        try:
            decoded = pipe.ColorCorrector(
                decoded, condition, clip_range=(-1, 1), chunk_size=None, method="adain"
            )
        except torch.cuda.OutOfMemoryError:
            raise
        chunk = decoded[0].permute(1, 2, 3, 0).float().add_(1).mul_(127.5).clamp_(0, 255).byte().cpu()
        if timing is not None:
            timing["decode_seconds"] = timing.get("decode_seconds", 0.0) + (
                time.perf_counter() - decode_started
            )
        for local_index in range(chunk.shape[0]):
            if written >= real_frames:
                break
            callback(written, chunk[local_index].numpy())
            written += 1
        if emission_callback is not None and written > emission_frame_start:
            emission_callback({
                "emission_index": process_index,
                "frame_start": emission_frame_start,
                "frame_end_exclusive": written,
                "frame_end_inclusive": written - 1,
                "frame_count": written - emission_frame_start,
                "duration_seconds": time.perf_counter() - emission_started,
            })
        del predicted, latent, decoded, chunk, condition, lq_latents
        lq_previous = lq_current
    if written != real_frames:
        raise RuntimeError(f"FlashVSR produced {written} frames, expected {real_frames}")
    return written


def save_tile_frame(directory: Path, tile: Tile, frame_index: int, frame: Any) -> None:
    import cv2
    height, width = tile.height * SCALE, tile.width * SCALE
    cropped = frame[:height, :width]
    path = directory / f"{frame_index:09d}.png"
    if not cv2.imwrite(str(path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError(f"Failed to stage tile frame: {path}")


def feather_axis(length: int, left_overlap: int, right_overlap: int) -> Any:
    import numpy as np
    weight = np.ones(length, np.float32)
    if left_overlap:
        weight[:left_overlap] = np.linspace(0.0, 1.0, left_overlap, endpoint=False, dtype=np.float32)
    if right_overlap:
        weight[-right_overlap:] = np.minimum(
            weight[-right_overlap:], np.linspace(1.0, 0.0, right_overlap, endpoint=False, dtype=np.float32)
        )
    return np.maximum(weight, np.float32(1e-4))


def tile_weight(tile: Tile, tiles: list[Tile], frame_w: int, frame_h: int) -> Any:
    # Actual overlaps are derived from geometry, including an end-anchored last tile.
    same_row = [t for t in tiles if t.y == tile.y]
    same_col = [t for t in tiles if t.x == tile.x]
    left = max((t.x + t.width - tile.x for t in same_row if t.x < tile.x), default=0) * SCALE
    right = max((tile.x + tile.width - t.x for t in same_row if t.x > tile.x), default=0) * SCALE
    top = max((t.y + t.height - tile.y for t in same_col if t.y < tile.y), default=0) * SCALE
    bottom = max((tile.y + tile.height - t.y for t in same_col if t.y > tile.y), default=0) * SCALE
    return feather_axis(frame_h, top, bottom)[:, None] * feather_axis(frame_w, left, right)[None, :]


def available_encoders() -> list[str]:
    result = run(["ffmpeg", "-hide_banner", "-encoders"], check=False, capture_output=True)
    text = (result.stdout or "") + (result.stderr or "")
    return [name for name in ("ffv1", "libaom-av1", "libsvtav1", "av1_nvenc", "hevc_nvenc", "libx265") if name in text]


def with_container_extension(path: Path, extension: str) -> Path:
    """Replace only a known container extension, preserving dots in model names."""
    for known in (".mp4", ".mkv"):
        if path.name.endswith(known):
            return path.with_name(path.name[: -len(known)] + extension)
    return path.with_name(path.name + extension)


def encode_tiles(
    input_path: Path, output_path: Path, meta: dict[str, Any], tiles: list[Tile], tile_dirs: dict[int, Path],
    ffmpeg_threads: int, encoder_name: str, scratch: Path,
) -> tuple[Path, float, list[str], dict[str, Any], list[dict[str, str]], dict[str, float]]:
    """Blend one full-resolution frame at a time into a proven 16K-capable stream."""
    import cv2
    import numpy as np

    if str(ARCHITECT) not in sys.path:
        sys.path.insert(0, str(ARCHITECT))
    from common.media_utils import VideoMetadata, finalize_video_with_audio, select_encoder

    started = time.perf_counter()
    selected, encoder_attempts = select_encoder(
        meta["width"] * SCALE,
        meta["height"] * SCALE,
        threads=ffmpeg_threads,
        scratch_dir=scratch,
        preferred=encoder_name,
    )
    output_path = with_container_extension(output_path, selected["extension"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent = scratch / f"flashvsr_video_only{selected['extension']}"
    out_w, out_h = meta["width"] * SCALE, meta["height"] * SCALE
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{out_w}x{out_h}", "-framerate", meta["fps_fraction"], "-i", "pipe:0",
        "-an", "-threads", str(ffmpeg_threads), *selected["arguments"], str(silent),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    weights = {tile.index: tile_weight(tile, tiles, tile.width * SCALE, tile.height * SCALE) for tile in tiles}
    merge_seconds = 0.0
    try:
        for frame_index in range(meta["frame_count"]):
            merge_started = time.perf_counter()
            accumulator = np.zeros((out_h, out_w, 3), np.float32)
            normalization = np.zeros((out_h, out_w), np.float32)
            for tile in tiles:
                image = cv2.imread(str(tile_dirs[tile.index] / f"{frame_index:09d}.png"), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Missing staged tile {tile.index}, frame {frame_index}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                y0, x0 = tile.y * SCALE, tile.x * SCALE
                h, w = image.shape[:2]
                weight = weights[tile.index]
                accumulator[y0:y0+h, x0:x0+w] += image.astype(np.float32) * weight[..., None]
                normalization[y0:y0+h, x0:x0+w] += weight
            frame = np.rint(accumulator / np.maximum(normalization[..., None], 1e-6)).clip(0, 255).astype(np.uint8)
            merge_seconds += time.perf_counter() - merge_started
            process.stdin.write(frame.tobytes())
    except BaseException:
        with contextlib.suppress(BrokenPipeError):
            process.stdin.close()
        if process.stderr:
            process.stderr.read()
        process.wait()
        silent.unlink(missing_ok=True)
        raise
    else:
        with contextlib.suppress(BrokenPipeError):
            process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(
            f"exact-resolution {selected['name']} encoding failed ({code}): {stderr}"
        )
    input_metadata = VideoMetadata(
        path=meta["path"], width=meta["width"], height=meta["height"],
        fps=meta["fps_fraction"], fps_float=meta["fps"], frame_count=meta["frame_count"],
        duration=meta["duration"], pixel_format=meta.get("pixel_format"), video_codec=None,
        color_space=None, color_transfer=None, color_primaries=None, time_base=None,
        audio_present=meta["audio_present"], audio_codec=meta.get("audio_codec"),
    )
    final_path = finalize_video_with_audio(
        silent,
        output_path.with_suffix(""),
        input_path=str(input_path),
        input_metadata=input_metadata,
        extension=selected["extension"],
    )
    total_seconds = time.perf_counter() - started
    return (
        final_path, total_seconds, available_encoders(), selected, encoder_attempts,
        # Include codec capability probing and setup in encode, while keeping
        # the explicitly measured weighted pixel merge disjoint.
        {"merge_seconds": merge_seconds, "encode_seconds": max(0.0, total_seconds - merge_seconds)},
    )


def git_revision() -> str | None:
    result = run(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], check=False, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def report_markdown(report: dict[str, Any]) -> str:
    timings, gpu = report["timings"], report.get("gpu", {})
    average_utilization = gpu.get("average_gpu_utilization_percent")
    peak_utilization = gpu.get("peak_gpu_utilization_percent")
    utilization_text = (
        f"{average_utilization:.1f}% / {peak_utilization:.1f}%"
        if isinstance(average_utilization, (int, float))
        and isinstance(peak_utilization, (int, float))
        else "N/A (nvidia-smi utilization counters unavailable)"
    )
    return f"""# FlashVSR v1.1 Benchmark Report

- Result: **{report['status']}**
- Input: `{report['input']['path']}`
- Output: `{report.get('output_path')}`
- Resolution: {report['input']['width']}x{report['input']['height']} -> {report['expected_output']['width']}x{report['expected_output']['height']} (4x)
- Frames / FPS: {report['input']['frame_count']} / {report['input']['fps_fraction']}
- Official continuous causal/KV path: yes
- Source tiling: {report['configuration']['tile_size']} px, overlap {report['configuration']['tile_overlap']} px

## Timing

| Metric | Value |
|---|---:|
| Model initialization / checkpoint loading | {timings.get('model_load_seconds', 0):.3f} s |
| Neural inference (DiT) | {timings.get('neural_inference_seconds', timings.get('inference_seconds', 0)):.3f} s |
| TCDecoder + color correction | {timings.get('decode_seconds', 0):.3f} s |
| Spatial weighted merge | {timings.get('merge_seconds', 0):.3f} s |
| Video encode / mux | {timings.get('encode_seconds', timings.get('encoding_mux_seconds', 0)):.3f} s |
| Total wall-clock | {timings.get('total_wall_seconds', 0):.3f} s |
| Pure inference sec/input frame | {timings.get('pure_inference_seconds_per_input_frame', 0):.6f} |
| End-to-end sec/input frame | {timings.get('end_to_end_seconds_per_input_frame', timings.get('total_seconds_per_frame', 0)):.6f} |
| Pure inference sec/output frame | {timings.get('pure_inference_seconds_per_output_frame', 0):.6f} |
| End-to-end sec/output frame | {timings.get('end_to_end_seconds_per_output_frame', timings.get('total_seconds_per_frame', 0)):.6f} |
| Effective inference FPS | {timings.get('effective_inference_fps', 0):.6f} |
| Effective end-to-end FPS | {timings.get('effective_end_to_end_fps', 0):.6f} |

## GPU

| Metric | Value |
|---|---:|
| GPU | {gpu.get('gpu_name', 'unknown')} |
| Total VRAM | {gpu.get('gpu_total_vram_mb', 0):.0f} MiB |
| Peak torch allocated | {gpu.get('peak_torch_allocated_mb', 0):.0f} MiB |
| Peak torch reserved | {gpu.get('peak_torch_reserved_mb', 0):.0f} MiB |
| Peak process memory | {gpu.get('peak_process_gpu_memory_mb', 0):.0f} MiB |
| Peak total GPU used | {gpu.get('peak_total_gpu_used_memory_mb', 0):.0f} MiB |
| Average / peak utilization | {utilization_text} |
"""


def write_report(report: dict[str, Any], result_dir: Path, explicit_json: Path | None = None) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = explicit_json or result_dir / "report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not report.get("probe_only") and "input" in report and "timings" in report:
        json_path.with_suffix(".md").write_text(report_markdown(report))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Durably publish a JSON sentinel without exposing a partial document."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tile_commit_path(scratch: Path, tile: Tile) -> Path:
    return scratch / "tile_commits" / f"tile_{tile.index:04d}.json"


def _validate_tile_commit(
    scratch: Path, tile: Tile, frame_count: int, seed: int, request_sha256: str,
) -> list[Path] | None:
    """Reuse only a fully committed independent spatial tile, not a partial causal stream."""
    path = _tile_commit_path(scratch, tile)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("state") != "COMPLETE" or value.get("tile") != asdict(tile)
            or value.get("frame_count") != frame_count
            or value.get("tile_seed") != seed
            or value.get("request_sha256") != request_sha256):
        raise RuntimeError(f"FlashVSR completed tile metadata differs from this run: {path}")
    directory = (scratch / "tiles" / f"tile_{tile.index:04d}").resolve()
    images = value.get("png_files")
    if not isinstance(images, list) or len(images) != frame_count:
        raise RuntimeError(f"FlashVSR completed tile has an incomplete PNG manifest: {path}")
    for index, entry in enumerate(images):
        image = (directory / f"{index:09d}.png").resolve()
        if (entry.get("path") != str(image) or not image.is_file()
                or image.stat().st_size != entry.get("size_bytes")
                or _file_sha256(image) != entry.get("sha256")):
            raise RuntimeError(f"FlashVSR committed tile PNG is missing or corrupt: {image}")
    metadata = [Path(item).resolve() for item in value.get("temporal_unit_metadata", [])]
    metadata_hashes = value.get("temporal_unit_hashes")
    if (not metadata or not isinstance(metadata_hashes, list)
            or len(metadata) != len(metadata_hashes)
            or any(not item.is_file() for item in metadata)):
        raise RuntimeError(f"FlashVSR tile emission metadata is missing: {path}")
    for item, expected_hash in zip(metadata, metadata_hashes):
        if _file_sha256(item) != expected_hash:
            raise RuntimeError(f"FlashVSR tile emission metadata is corrupt: {item}")
    return metadata


def _commit_tile(
    scratch: Path, tile: Tile, frame_count: int, seed: int,
    request_sha256: str, metadata: list[Path],
) -> None:
    directory = (scratch / "tiles" / f"tile_{tile.index:04d}").resolve()
    images = []
    for index in range(frame_count):
        path = directory / f"{index:09d}.png"
        if not path.is_file():
            raise RuntimeError(f"FlashVSR tile is missing an emitted frame: {path}")
        images.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": _file_sha256(path)})
    write_json_atomic(_tile_commit_path(scratch, tile), {
        "schema_version": 1,
        "state": "COMPLETE",
        "tile": asdict(tile),
        "tile_seed": seed,
        "frame_count": frame_count,
        "request_sha256": request_sha256,
        "png_files": images,
        "temporal_unit_metadata": [str(path.resolve()) for path in metadata],
        "temporal_unit_hashes": [_file_sha256(path) for path in metadata],
    })


def stage_manifest_path(args: argparse.Namespace) -> Path:
    if not args.stage_manifest:
        raise ValueError(f"--stage-manifest is required for --stage-mode {args.stage_mode}")
    return Path(args.stage_manifest).resolve()


def read_stage_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"FlashVSR stage manifest does not exist: {path}") from error
    if manifest.get("schema_version") != STAGE_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported FlashVSR stage manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("model") != MODEL_NAME or manifest.get("state") != "COMPLETE":
        raise ValueError("FlashVSR stage manifest is not a complete inference artifact")
    return manifest


def execute_encode_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Encode staged frames without importing torch or creating a CUDA context."""
    started = time.perf_counter()
    manifest_path = stage_manifest_path(args)
    manifest = read_stage_manifest(manifest_path)
    scratch = Path(manifest["scratch_path"]).resolve()
    if not scratch.is_dir():
        raise FileNotFoundError(f"FlashVSR stage scratch is missing: {scratch}")
    meta = manifest["input"]
    input_path = Path(meta["path"]).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Staged input is missing: {input_path}")
    tiles: list[Tile] = []
    tile_dirs: dict[int, Path] = {}
    for artifact in manifest.get("tile_artifacts", []):
        tile = Tile(**artifact["tile"])
        directory = Path(artifact["directory"]).resolve()
        try:
            directory.relative_to(scratch)
        except ValueError as error:
            raise ValueError(f"Tile directory escapes stage scratch: {directory}") from error
        if not directory.is_dir():
            raise FileNotFoundError(f"Staged tile directory is missing: {directory}")
        expected_frames = int(manifest["processed_input_frames"])
        pngs = list(directory.glob("*.png"))
        if len(pngs) != expected_frames:
            raise RuntimeError(
                f"Staged tile {tile.index} has {len(pngs)} frames, expected {expected_frames}"
            )
        tiles.append(tile)
        tile_dirs[tile.index] = directory
    if not tiles or len(tiles) != int(manifest["tile_count"]):
        raise RuntimeError("FlashVSR stage manifest contains an incomplete tile set")

    # Stage-owned values are authoritative; only CPU scheduling/encoder selection
    # may be changed when resuming the encode phase.
    output_arg = Path(manifest["output_target"])
    succeeded = False
    try:
        output_path, encoding_seconds, encoders, selected, attempts, encode_components = encode_tiles(
            input_path, output_arg, meta, tiles, tile_dirs, args.ffmpeg_threads,
            args.encoder, scratch,
        )
        report = dict(manifest["partial_report"])
        report["status"] = "PASS"
        report["stage_mode"] = "encode"
        report["output_path"] = str(output_path.resolve())
        report["encoder"] = selected["name"]
        report["container_extension"] = output_path.suffix
        report["encoder_arguments"] = selected["arguments"]
        report["encoder_probe_attempts"] = attempts
        report["available_relevant_encoders"] = encoders
        inference_timings = dict(report["timings"])
        staging_seconds = float(report.get("timings_detail", {}).get("intermediate_tile_staging_seconds", 0))
        total_work_seconds = float(inference_timings["total_wall_seconds"]) + encoding_seconds
        frames = max(1, int(manifest["processed_input_frames"]))
        inference_timings.update({
            "encoding_mux_seconds": staging_seconds + encoding_seconds,
            "merge_seconds": encode_components["merge_seconds"],
            "encode_seconds": encode_components["encode_seconds"],
            "total_wall_seconds": total_work_seconds,
            "total_seconds_per_frame": total_work_seconds / frames,
            "end_to_end_seconds_per_input_frame": total_work_seconds / frames,
            "end_to_end_seconds_per_output_frame": total_work_seconds / frames,
            "effective_end_to_end_fps": frames / total_work_seconds,
        })
        report["timings"] = inference_timings
        report["timing"] = inference_timings
        for metric in (
            "model_load_seconds", "neural_inference_seconds", "decode_seconds", "merge_seconds",
            "encode_seconds", "total_wall_seconds", "pure_inference_seconds_per_input_frame",
            "end_to_end_seconds_per_input_frame", "pure_inference_seconds_per_output_frame",
            "end_to_end_seconds_per_output_frame",
        ):
            report[metric] = inference_timings.get(metric, 0.0)
        report["timings_detail"] = {
            **report.get("timings_detail", {}),
            "final_encoding_mux_seconds": encoding_seconds,
            "stage_resume_wall_seconds": time.perf_counter() - started,
        }
        report["stage_manifest"] = str(manifest_path)
        succeeded = True
        return report
    finally:
        if succeeded and not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def choose_auto_tile(gpu_index: int) -> int:
    result = run([
        "nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ], check=False, capture_output=True)
    try:
        free_mb = float(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        free_mb = 0
    # Conservative source-space defaults; model sees four times these dimensions.
    return 320 if free_mb >= 100_000 else 256 if free_mb >= 60_000 else 192


def is_cuda_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def benchmark_frame_selection(
    total_frames: int, mode: str, start_frame: int, max_frames: int | None,
) -> tuple[int, int]:
    """Return (start, count), preserving the legacy full-video default."""
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"--start-frame must be in [0, {total_frames - 1}]")
    available = total_frames - start_frame
    default_count = {"micro": MICRO_FRAMES, "mini": MINI_FRAMES, "full": available}[mode]
    count = min(available, default_count, max_frames if max_frames is not None else available)
    if mode != "full" and count < MICRO_FRAMES:
        raise ValueError(
            f"FlashVSR TinyLong needs at least {MICRO_FRAMES} real frames for a meaningful "
            f"{mode} causal benchmark; selected range has {count}"
        )
    return start_frame, count


def streaming_step_count(frame_count: int) -> int:
    """Number of meaningful Tiny Long causal emissions for real input frames."""
    if frame_count <= 0:
        return 0
    if frame_count <= MICRO_FRAMES:
        return 1
    return 1 + math.ceil((frame_count - MICRO_FRAMES) / 8)


def estimate_flash_times(
    *,
    model_load_seconds: float,
    inference_work_seconds: float,
    merge_seconds: float,
    encode_seconds: float,
    staging_seconds: float,
    measured_frames: int,
    frame_targets: Sequence[int] = (10, 100, 1000, 1920),
) -> dict[str, dict[str, float | int]]:
    """Project cost from one bounded run; combined micro/mini fitting is reported separately."""
    measured_steps = streaming_step_count(measured_frames)
    inference_per_step = inference_work_seconds / max(1, measured_steps)
    cpu_io_per_frame = (staging_seconds + merge_seconds + encode_seconds) / max(1, measured_frames)
    estimates: dict[str, dict[str, float | int]] = {}
    for frames in frame_targets:
        steps = streaming_step_count(frames)
        estimates[str(frames)] = {
            "frames": frames,
            "streaming_steps": steps,
            "estimated_total_seconds": (
                model_load_seconds + inference_per_step * steps + cpu_io_per_frame * frames
            ),
        }
    return estimates


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage_mode == "encode":
        return execute_encode_stage(args)

    import random
    import numpy as np
    import torch

    # Keep the production per-tile generator derivation unchanged.  Seed the
    # remaining global RNGs as well for certification reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[SEED] model=FlashVSR_v1.1 base_seed={args.seed} seed_mode=per_tile_generator_plus_global_rng", flush=True)

    started = time.perf_counter()
    input_path = Path(args.input).resolve()
    meta = ffprobe(input_path)
    print_metadata(meta)
    if args.scale != SCALE:
        raise ValueError("FlashVSR official pretrained inference is fixed to exact 4x; --scale must be 4")
    if args.probe:
        start_frame, real_frames = 0, min(meta["frame_count"], args.probe_frames)
    else:
        start_frame, real_frames = benchmark_frame_selection(
            meta["frame_count"], args.benchmark_mode, args.start_frame, args.max_frames,
        )
    source_meta = dict(meta)
    meta = dict(meta)
    meta["frame_count"] = real_frames
    meta["duration"] = real_frames / meta["fps"]
    # A benchmark range is a video sample, not a remux of the complete source
    # audio timeline. Full legacy runs retain the original audio behavior.
    if start_frame or real_frames != source_meta["frame_count"]:
        meta["audio_present"] = False
        meta["audio_codec"] = None
    model_frames = padded_model_frames(real_frames)
    tile_size = choose_auto_tile(args.gpu_index) if args.tile_size == "auto" else int(args.tile_size)
    tile_size = min(tile_size, meta["width"], meta["height"])
    tile_size = max(64, (tile_size // 32) * 32)
    requested_overlap = 32 if args.tile_overlap == "auto" else int(args.tile_overlap)
    overlap = min(requested_overlap, max(0, tile_size - 32))
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile overlap must be non-negative and smaller than tile size")

    if args.output:
        output_arg = Path(args.output)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
        sample_suffix = "" if args.benchmark_mode == "full" and start_frame == 0 and real_frames == source_meta["frame_count"] else (
            f"_{args.benchmark_mode}_f{start_frame:06d}_n{real_frames}"
        )
        output_arg = output_dir / f"{input_path.stem}_{MODEL_NAME}_x4{sample_suffix}.mkv"
    scratch_parent = Path(args.scratch_dir) if args.scratch_dir else Path("/tmp")
    scratch_parent.mkdir(parents=True, exist_ok=True)
    request_sha256 = ""
    if args.scratch_path:
        if args.stage_mode != "inference" or not args.stage_request:
            raise ValueError("--scratch-path requires staged inference and --stage-request")
        request_sha256 = _file_sha256(Path(args.stage_request).resolve())
        scratch = Path(args.scratch_path).resolve()
        identity_path = scratch / "certification_identity.json"
        if scratch.exists() and not identity_path.is_file() and any(scratch.iterdir()):
            raise RuntimeError("FlashVSR scratch contains artifacts without a run identity")
        scratch.mkdir(parents=True, exist_ok=True)
        if identity_path.is_file():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if (identity.get("request_sha256") != request_sha256
                    or identity.get("input_path") != str(input_path)
                    or identity.get("frame_count") != real_frames):
                raise RuntimeError("FlashVSR scratch identity differs from this certification request")
            tile_size = int(identity["tile_size"])
            overlap = int(identity["tile_overlap"])
            print(f"[RESUME] FlashVSR tile journal={scratch} tile_size={tile_size} overlap={overlap}", flush=True)
        else:
            write_json_atomic(identity_path, {
                "schema_version": 1,
                "request_sha256": request_sha256,
                "input_path": str(input_path),
                "frame_count": real_frames,
                "tile_size": tile_size,
                "tile_overlap": overlap,
            })
    else:
        scratch = Path(tempfile.mkdtemp(prefix="hrlrvo_flashvsr_", dir=scratch_parent))
    artifact_metadata_root = (
        Path(args.artifact_metadata_dir).resolve()
        if args.artifact_metadata_dir
        else (Path(args.result_dir).resolve() / "temporal_units" / scratch.name)
    )
    artifact_metadata_root.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu_index)
    torch.cuda.reset_peak_memory_stats(args.gpu_index)
    monitor = GPUMonitor(args.gpu_index, args.gpu_sample_interval)
    report: dict[str, Any] = {
        "model": MODEL_NAME, "status": "RUNNING", "probe_only": args.probe,
        "input": meta,
        "source_input": source_meta,
        "expected_output": {"width": meta["width"] * SCALE, "height": meta["height"] * SCALE, "scale": SCALE},
        "configuration": {
            "tile_size": tile_size, "tile_overlap": overlap, "padded_model_frames": model_frames,
            "processed_input_frames": real_frames, "sparse_ratio": args.sparse_ratio,
            "local_range": args.local_range, "gpu_index": args.gpu_index,
            "temporal_mode": "official_continuous_causal_kv_no_independent_chunks",
            "benchmark_mode": args.benchmark_mode,
            "start_frame": start_frame,
            "effective_input_frames_processed": real_frames,
            "artifact_metadata_root": str(artifact_metadata_root),
        },
        # Spatial tiles each use one uninterrupted causal stream; there are no
        # independent temporal chunk joins to inspect.
        "chunk_boundaries": [],
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "pytorch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
            "safetensors": importlib.metadata.version("safetensors"),
            "block_sparse_attention": importlib.metadata.version("block-sparse-attn"),
            "modelscope": importlib.metadata.version("modelscope"),
            "repository_commit": git_revision(), "repository": str(REPOSITORY),
            "block_sparse_attention_commit": "49d6c39e4dc0303442cda3bb758b3925d4399c49",
            "checkpoint": str(CHECKPOINT / "diffusion_pytorch_model_streaming_dmd.safetensors"),
            "checkpoint_revision": "27561b186ded3402d7c975f4fd722e2885b6135f",
        },
    }
    retain_stage = False
    try:
        with monitor:
            pipe, load_info = initialize_pipeline(f"cuda:{args.gpu_index}")
            report["model_load"] = load_info
            print(
                f"[MODEL_LOAD] model={MODEL_NAME} gpu={args.gpu_index} "
                f"seed={args.seed} status=READY elapsed={load_info['seconds']:.1f}s",
                flush=True,
            )
            inference_seconds = 0.0
            component_timing = {"neural_inference_seconds": 0.0, "decode_seconds": 0.0}
            staging_seconds = 0.0
            attempts = 0
            while True:
                attempts += 1
                try:
                    attempt_metadata_dir = artifact_metadata_root / f"attempt_{attempts:03d}_{time.time_ns()}"
                    attempt_metadata_dir.mkdir(parents=True, exist_ok=True)
                    tiles = make_tiles(meta["width"], meta["height"], tile_size, overlap)
                    if args.probe:
                        tiles = tiles[:1]
                    tile_dirs: dict[int, Path] = {}
                    tile_temporal_metadata: dict[int, list[Path]] = {}
                    for tile in tiles:
                        tile_dir = scratch / "tiles" / f"tile_{tile.index:04d}"
                        if args.scratch_path:
                            prior = _validate_tile_commit(
                                scratch, tile, real_frames, args.seed + tile.index, request_sha256,
                            )
                            if prior is not None:
                                tile_dirs[tile.index] = tile_dir
                                tile_temporal_metadata[tile.index] = prior
                                print(
                                    f"[UPSCALE_TILE_RESUME] tile={tile.index + 1}/{len(tiles)} "
                                    f"frames={real_frames}/{real_frames} status=PASS",
                                    flush=True,
                                )
                                continue
                            if tile_dir.exists() and any(tile_dir.iterdir()):
                                abandoned = scratch / "incomplete_tiles" / (
                                    f"tile_{tile.index:04d}_{time.time_ns()}"
                                )
                                abandoned.parent.mkdir(parents=True, exist_ok=True)
                                tile_dir.rename(abandoned)
                        tile_dir.mkdir(parents=True, exist_ok=True)
                        tile_dirs[tile.index] = tile_dir
                        tile_temporal_metadata[tile.index] = []
                        store = build_tile_store(
                            input_path, tile, real_frames, model_frames,
                            scratch, args.ffmpeg_threads, start_frame,
                        )
                        tile_started = time.perf_counter()
                        tile_seed = args.seed + tile.index
                        def stage_output(
                            frame_index: int,
                            frame: Any,
                            directory: Path = tile_dir,
                            tile_spec: Tile = tile,
                        ) -> None:
                            nonlocal staging_seconds
                            stage_started = time.perf_counter()
                            save_tile_frame(directory, tile_spec, frame_index, frame)
                            staging_seconds += time.perf_counter() - stage_started
                            completed = frame_index + 1
                            if completed == 1 or completed % 10 == 0 or completed == real_frames:
                                elapsed = time.perf_counter() - tile_started
                                eta = elapsed / completed * (real_frames - completed)
                                print(
                                    f"[UPSCALE] tile={tile_spec.index + 1}/{len(tiles)} "
                                    f"frame={completed}/{real_frames} "
                                    f"progress={completed / real_frames * 100:.1f}% "
                                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                                    flush=True,
                                )

                        def record_emission(
                            emission: dict[str, Any],
                            directory: Path = tile_dir,
                            tile_spec: Tile = tile,
                            effective_seed: int = tile_seed,
                        ) -> None:
                            frame_start = int(emission["frame_start"])
                            frame_end_exclusive = int(emission["frame_end_exclusive"])
                            frame_end_inclusive = int(emission["frame_end_inclusive"])
                            pngs = [
                                str((directory / f"{index:09d}.png").resolve())
                                for index in range(frame_start, frame_end_exclusive)
                            ]
                            metadata_path = attempt_metadata_dir / (
                                f"tile_{tile_spec.index:04d}_emission_"
                                f"{int(emission['emission_index']):04d}_frames_"
                                f"{frame_start:09d}_{frame_end_inclusive:09d}.json"
                            )
                            payload = {
                                "schema_version": 1,
                                "model": MODEL_NAME,
                                "temporal_unit": "TinyLong causal emission",
                                "tile": asdict(tile_spec),
                                "tile_index": tile_spec.index,
                                "base_seed": args.seed,
                                "tile_seed": effective_seed,
                                "emission_index": int(emission["emission_index"]),
                                "frame_start": frame_start,
                                "frame_end_exclusive": frame_end_exclusive,
                                "frame_end_inclusive": frame_end_inclusive,
                                "actual_start_frame": start_frame + frame_start,
                                "actual_end_frame": start_frame + frame_end_inclusive,
                                "frame_count": int(emission["frame_count"]),
                                "duration_seconds": float(emission["duration_seconds"]),
                                "png_directory": str(directory.resolve()),
                                "png_frame_range": {
                                    "pattern": "%09d.png",
                                    "start": frame_start,
                                    "end_exclusive": frame_end_exclusive,
                                },
                                "png_files": pngs,
                                "state": "COMPLETE",
                            }
                            write_json_atomic(metadata_path, payload)
                            tile_temporal_metadata[tile_spec.index].append(metadata_path)
                            print(
                                f"[UPSCALE_SEGMENT_DONE] tile={tile_spec.index + 1}/{len(tiles)} "
                                f"emission={int(emission['emission_index']) + 1} "
                                f"frames={frame_start}-{frame_end_inclusive} "
                                f"artifact={metadata_path} "
                                f"elapsed={float(emission['duration_seconds']):.1f}s",
                                flush=True,
                            )

                        staging_before = staging_seconds
                        inference_started = time.perf_counter()
                        try:
                            stream_official_tile(
                                pipe, store, model_frames, real_frames,
                                tile_seed, args.sparse_ratio,
                                args.local_range, stage_output,
                                component_timing,
                                record_emission,
                            )
                        finally:
                            elapsed = time.perf_counter() - inference_started
                            inference_seconds += max(
                                0.0,
                                elapsed - (staging_seconds - staging_before),
                            )
                        if args.scratch_path:
                            _commit_tile(
                                scratch, tile, real_frames, tile_seed,
                                request_sha256, tile_temporal_metadata[tile.index],
                            )
                            print(
                                f"[UPSCALE_TILE_DONE] tile={tile.index + 1}/{len(tiles)} "
                                f"frames={real_frames}/{real_frames} status=PASS",
                                flush=True,
                            )
                        store.path.unlink(missing_ok=True)
                        del store
                        torch.cuda.empty_cache()
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    if not isinstance(error, torch.cuda.OutOfMemoryError) and not is_cuda_oom(error):
                        raise
                    torch.cuda.empty_cache()
                    shutil.rmtree(scratch / "tiles", ignore_errors=True)
                    if args.scratch_path:
                        shutil.rmtree(scratch / "tile_commits", ignore_errors=True)
                    # This attempt's PNGs were discarded, so its metadata must
                    # not survive as apparently complete artifact references.
                    shutil.rmtree(attempt_metadata_dir, ignore_errors=True)
                    if tile_size <= args.min_tile_size:
                        raise RuntimeError(
                            f"FlashVSR OOM persisted at minimum source tile {tile_size}px; "
                            "increase available VRAM or lower --min-tile-size"
                        )
                    previous = tile_size
                    tile_size = max(args.min_tile_size, ((tile_size - 32) // 32) * 32)
                    overlap = min(overlap, tile_size - 32)
                    print(f"CUDA OOM at tile {previous}px; retrying complete run with {tile_size}px tiles", flush=True)
                    report["configuration"]["tile_size"] = tile_size
                    report["configuration"]["tile_overlap"] = overlap
                    if args.scratch_path:
                        write_json_atomic(scratch / "certification_identity.json", {
                            "schema_version": 1,
                            "request_sha256": request_sha256,
                            "input_path": str(input_path),
                            "frame_count": real_frames,
                            "tile_size": tile_size,
                            "tile_overlap": overlap,
                        })
            encoding_seconds = staging_seconds
            final_components = {"merge_seconds": 0.0, "encode_seconds": 0.0}
            output_path: Path | None = None
            encoders: list[str] = available_encoders()
            if not args.probe and args.stage_mode == "all":
                output_path, final_encoding_seconds, encoders, selected_encoder, encoder_attempts, final_components = encode_tiles(
                    input_path, output_arg, meta, tiles, tile_dirs, args.ffmpeg_threads,
                    args.encoder, scratch,
                )
                encoding_seconds += final_encoding_seconds
                report["output_path"] = str(output_path.resolve())
                report["encoder"] = selected_encoder["name"]
                report["container_extension"] = output_path.suffix
                report["encoder_arguments"] = selected_encoder["arguments"]
                report["encoder_probe_attempts"] = encoder_attempts
            report["timings_detail"] = {
                "intermediate_tile_staging_seconds": staging_seconds,
            }
            report["available_relevant_encoders"] = encoders
            report["configuration"]["oom_retry_attempts"] = attempts - 1
            report["temporal_unit_metadata"] = [
                str(path.resolve())
                for tile in tiles
                for path in tile_temporal_metadata[tile.index]
            ]
        gpu = monitor.summary()
        gpu.update({
            "peak_torch_allocated_mb": torch.cuda.max_memory_allocated(args.gpu_index) / 2**20,
            "peak_torch_reserved_mb": torch.cuda.max_memory_reserved(args.gpu_index) / 2**20,
        })
        total_seconds = time.perf_counter() - started
        denominator = max(1, real_frames)
        report["gpu"] = gpu
        report["peak_vram_mb"] = max(
            gpu.get("peak_process_gpu_memory_mb", 0), gpu.get("peak_torch_reserved_mb", 0)
        )
        report["peak_torch_allocated_mb"] = gpu.get("peak_torch_allocated_mb", 0)
        report["peak_torch_reserved_mb"] = gpu.get("peak_torch_reserved_mb", 0)
        report["peak_process_gpu_memory_mb"] = gpu.get("peak_process_gpu_memory_mb", 0)
        report["peak_total_gpu_used_memory_mb"] = gpu.get("peak_total_gpu_used_memory_mb", 0)
        report["peak_torch_allocated_mib"] = gpu.get("peak_torch_allocated_mb", 0)
        report["peak_torch_reserved_mib"] = gpu.get("peak_torch_reserved_mb", 0)
        report["peak_process_gpu_mib"] = gpu.get("peak_process_gpu_memory_mb", 0)
        report["peak_total_gpu_used_mib"] = gpu.get("peak_total_gpu_used_memory_mb", 0)
        report["effective_input_frames_processed"] = real_frames
        report["effective_output_frames_written"] = real_frames
        report["tile_count"] = len(tiles)
        pure_inference = (
            component_timing["neural_inference_seconds"]
            + component_timing["decode_seconds"]
        )
        report["timings"] = {
            "model_load_seconds": load_info["seconds"], "inference_seconds": inference_seconds,
            "encoding_mux_seconds": encoding_seconds, "total_wall_seconds": total_seconds,
            "neural_inference_seconds": component_timing["neural_inference_seconds"],
            "decode_seconds": component_timing["decode_seconds"],
            "merge_seconds": final_components["merge_seconds"],
            "encode_seconds": final_components["encode_seconds"],
            "inference_seconds_per_frame": inference_seconds / denominator,
            "total_seconds_per_frame": total_seconds / denominator,
            "pure_inference_seconds_per_input_frame": pure_inference / denominator,
            "end_to_end_seconds_per_input_frame": total_seconds / denominator,
            "pure_inference_seconds_per_output_frame": pure_inference / denominator,
            "end_to_end_seconds_per_output_frame": total_seconds / denominator,
            "effective_inference_fps": denominator / inference_seconds if inference_seconds else 0,
            "effective_end_to_end_fps": denominator / total_seconds if total_seconds else 0,
        }
        report["timing"] = report["timings"]
        report["streaming_step_count"] = streaming_step_count(real_frames)
        report["time_estimation"] = {
            "formula": (
                "model_load_once + full_sr_inference_seconds_per_streaming_step * "
                "streaming_step_count(N) + staging_merge_encode_seconds_per_frame * N"
            ),
            "estimates": estimate_flash_times(
                model_load_seconds=float(load_info["seconds"]),
                inference_work_seconds=inference_seconds,
                merge_seconds=float(final_components["merge_seconds"]),
                encode_seconds=float(final_components["encode_seconds"]),
                staging_seconds=float(staging_seconds),
                measured_frames=real_frames,
            ),
        }
        for metric in (
            "model_load_seconds", "neural_inference_seconds", "decode_seconds", "merge_seconds",
            "encode_seconds", "total_wall_seconds", "pure_inference_seconds_per_input_frame",
            "end_to_end_seconds_per_input_frame", "pure_inference_seconds_per_output_frame",
            "end_to_end_seconds_per_output_frame",
        ):
            report[metric] = report["timings"][metric]
        if args.probe:
            report["recommended_tile_size"] = tile_size
            report["recommended_tile_overlap"] = overlap
        if args.stage_mode == "inference":
            manifest_path = stage_manifest_path(args)
            report["status"] = "STAGED"
            report["stage_mode"] = "inference"
            report["stage_manifest"] = str(manifest_path)
            manifest = {
                "schema_version": STAGE_MANIFEST_SCHEMA,
                "model": MODEL_NAME,
                "state": "COMPLETE",
                "created_at_epoch_seconds": time.time(),
                "input": meta,
                "output_target": str(output_arg.resolve()),
                "scratch_path": str(scratch.resolve()),
                "processed_input_frames": real_frames,
                "padded_model_frames": model_frames,
                "tile_count": len(tiles),
                "tile_artifacts": [
                    {
                        "tile": asdict(tile),
                        "directory": str(tile_dirs[tile.index].resolve()),
                        "tile_seed": args.seed + tile.index,
                        "temporal_unit_metadata": [
                            str(path.resolve())
                            for path in tile_temporal_metadata[tile.index]
                        ],
                    }
                    for tile in tiles
                ],
                "temporal_unit_metadata": report["temporal_unit_metadata"],
                "encode_defaults": {
                    "ffmpeg_threads": args.ffmpeg_threads,
                    "encoder": args.encoder,
                },
                "partial_report": report,
            }
            write_json_atomic(manifest_path, manifest)
            retain_stage = True
        else:
            report["status"] = "PASS"
        return report
    finally:
        if not args.keep_scratch and not retain_stage and not args.scratch_path:
            shutil.rmtree(scratch, ignore_errors=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Official pretrained FlashVSR v1.1 exact 4x video inference")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output", help="Output base path; selected codec determines .mp4 or .mkv")
    p.add_argument("--output-dir", help="Directory for the standard model-named output")
    p.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    p.add_argument("--report-json", "--probe-json", dest="report_json", help="Override report JSON path (useful for VRAM probes)")
    p.add_argument("--scale", type=int, default=4, choices=[4])
    p.add_argument("--tile-size", default="auto", help="Source-space tile side in pixels, multiple of 32, or auto")
    p.add_argument("--tile-overlap", default="auto", help="Source-space overlap in pixels, or auto")
    p.add_argument("--min-tile-size", type=int, default=128, help="Smallest OOM retry tile")
    p.add_argument("--sparse-ratio", type=float, default=2.0, choices=[1.5, 2.0])
    p.add_argument("--local-range", type=int, default=11, choices=[9, 11])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-index", type=int, default=int(os.environ.get("CUDA_VISIBLE_DEVICE_INDEX", "0")))
    p.add_argument("--gpu-sample-interval", type=float, default=0.5)
    p.add_argument("--ffmpeg-threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument(
        "--encoder", default="auto",
        choices=["auto", "ffv1", "libx265", "libsvtav1", "hevc_nvenc", "av1_nvenc"],
        help="16K encoder preference; auto performs a real one-frame capability probe",
    )
    p.add_argument("--scratch-dir", help="Parent for automatically cleaned runtime scratch")
    p.add_argument("--scratch-path", help=argparse.SUPPRESS)
    p.add_argument("--stage-request", help=argparse.SUPPRESS)
    p.add_argument("--keep-scratch", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--stage-mode", choices=["all", "inference", "encode"], default="all")
    p.add_argument("--stage-manifest", help="Atomic handoff manifest for split inference/encode stages")
    p.add_argument(
        "--artifact-metadata-dir",
        help="Directory for atomic per-tile TinyLong temporal-emission metadata",
    )
    p.add_argument("--probe", action="store_true", help="Run minimum valid causal inference and emit peak VRAM JSON")
    p.add_argument("--probe-only", action="store_true", dest="probe", help=argparse.SUPPRESS)
    p.add_argument("--probe-frames", type=int, default=21, help="Real leading frames used by --probe")
    p.add_argument(
        "--benchmark-mode", choices=["micro", "mini", "full"], default="full",
        help=f"micro={MICRO_FRAMES} frames, mini={MINI_FRAMES} frames, full=all selected frames",
    )
    p.add_argument("--max-frames", type=int, help="Cap selected real input frames (minimum 21)")
    p.add_argument("--start-frame", type=int, default=0, help="Zero-based first decoded input frame")
    return p


def main() -> int:
    args = parser().parse_args()
    report_path = Path(args.report_json) if args.report_json else None
    try:
        if args.probe and args.stage_mode != "all":
            raise ValueError("--probe is supported only with --stage-mode all")
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        if args.probe and (args.benchmark_mode != "full" or args.max_frames is not None or args.start_frame):
            raise ValueError("--probe cannot be combined with benchmark range options")
        report = execute(args)
        write_report(report, Path(args.result_dir), report_path)
        print(json.dumps({
            "status": report["status"], "output_path": report.get("output_path"),
            "report_json": str(report_path or Path(args.result_dir) / "report.json"),
            "peak_torch_allocated_mb": report.get("gpu", {}).get("peak_torch_allocated_mb"),
            "peak_process_gpu_memory_mb": report.get("gpu", {}).get("peak_process_gpu_memory_mb"),
        }, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        failure = {"model": MODEL_NAME, "status": "FAIL", "error": f"{type(error).__name__}: {error}", "probe_only": args.probe}
        with contextlib.suppress(Exception):
            write_report(failure, Path(args.result_dir), report_path)
        print(f"ERROR: {failure['error']}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
