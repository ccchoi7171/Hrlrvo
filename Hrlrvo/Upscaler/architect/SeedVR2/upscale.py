#!/usr/bin/env python3
"""SeedVR2-3B video super-resolution runner.

This is an inference-only adapter around ByteDance's official
``VideoDiffusionInfer``.  It keeps the official one-step video model (rather
than replacing it with an image SR model), and adds bounded temporal/spatial
execution needed for 4K -> 16K processing on one GPU.

Heavy dependencies are imported lazily so ``--help`` and orchestration
preflight checks work even before the SeedVR2 conda environment is active.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from typing import Any, Iterator, Sequence


HERE = pathlib.Path(__file__).resolve().parent
ARCHITECT_ROOT = HERE.parent
PROJECT_ROOT = ARCHITECT_ROOT.parent
if str(ARCHITECT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHITECT_ROOT))
DEFAULT_INPUT = PROJECT_ROOT / "input" / "0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4"
MODEL_ROOT = PROJECT_ROOT / "models" / "SeedVR2"
REPOSITORY = MODEL_ROOT / "repository"
CHECKPOINT_ROOT = MODEL_ROOT / "checkpoint"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "test"
    / "SeedVR2"
    / "0a8556b7-296c-4d33-90fe-b596ce5eba74_SeedVR2_x4.mkv"
)
DEFAULT_RESULT_DIR = PROJECT_ROOT / "result" / "test" / "SeedVR2"
SCALE = 4
MINIMUM_TEMPORAL_UNIT = 5
MINI_TEMPORAL_UNIT = 9
_SHARED_MEDIA: Any | None = None


def _shared_media() -> Any:
    """Load benchmark media helpers without colliding with SeedVR's `common`."""
    global _SHARED_MEDIA
    if _SHARED_MEDIA is None:
        name = "_hrlrvo_seed_shared_media_utils"
        spec = importlib.util.spec_from_file_location(name, ARCHITECT_ROOT / "common" / "media_utils.py")
        if spec is None or spec.loader is None:
            raise ImportError("cannot load architect/common/media_utils.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _SHARED_MEDIA = module
    return _SHARED_MEDIA


class RetryPlan(RuntimeError):
    """Signals that an OOM-safe smaller execution plan is required."""


@dataclasses.dataclass(frozen=True)
class MediaInfo:
    path: str
    width: int
    height: int
    fps: str
    fps_float: float
    frame_count: int
    duration: float
    pixel_format: str
    has_audio: bool


@dataclasses.dataclass(frozen=True)
class Tile:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int


@dataclasses.dataclass
class ChunkStage:
    index: int
    start: int
    valid: int
    paths: list[pathlib.Path]


def _run(command: Sequence[str], *, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), check=check, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout,
    )


def probe_media(path: pathlib.Path) -> MediaInfo:
    metadata = _shared_media().probe_media(path, count_frames=True)
    return MediaInfo(
        path=metadata.path, width=metadata.width, height=metadata.height,
        fps=metadata.fps, fps_float=metadata.fps_float, frame_count=metadata.frame_count,
        duration=metadata.duration, pixel_format=metadata.pixel_format or "unknown",
        has_audio=metadata.audio_present,
    )


def print_media(info: MediaInfo) -> None:
    # ``info.frame_count`` comes from an authoritative decoded-frame count.
    # Do not re-read a potentially stale container ``nb_frames`` header just
    # for logging, because the printed startup metadata is part of the final
    # benchmark record.
    print(
        "\n".join(
            [
                f"Input path    : {info.path}",
                f"Resolution    : {info.width}x{info.height}",
                f"FPS           : {info.fps} ({info.fps_float:.9f})",
                f"Frame count   : {info.frame_count}",
                f"Duration      : {info.duration:.6f} sec",
                f"Pixel format  : {info.pixel_format}",
                f"Audio presence: {'yes' if info.has_audio else 'no'}",
            ]
        ),
        flush=True,
    )


def _align_down(value: int, alignment: int) -> int:
    return max(alignment, value - value % alignment)


def _valid_window(value: int) -> int:
    value = max(MINIMUM_TEMPORAL_UNIT, value)
    return value - ((value - 1) % 4)


def benchmark_frame_range(
    mode: str,
    total_frames: int,
    start_frame: int,
    max_frames: int | None,
) -> tuple[int, int]:
    if mode not in {"micro", "mini", "full"}:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    if total_frames <= 0:
        raise ValueError("input has no video frames")
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start frame {start_frame} is outside [0, {total_frames})")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    default_count = {
        "micro": MINIMUM_TEMPORAL_UNIT,
        "mini": MINI_TEMPORAL_UNIT,
        "full": total_frames - start_frame,
    }[mode]
    count = min(max_frames if max_frames is not None else default_count, total_frames - start_frame)
    if mode in {"micro", "mini"} and count < MINIMUM_TEMPORAL_UNIT:
        raise ValueError(
            f"SeedVR2 {mode} benchmark requires at least {MINIMUM_TEMPORAL_UNIT} real frames"
        )
    return start_frame, count


def temporal_chunk_count(frame_count: int, window: int, overlap: int) -> int:
    if frame_count <= 0:
        return 0
    if frame_count <= window:
        return 1
    return 1 + math.ceil((frame_count - window) / (window - overlap))


def estimate_seed_times(
    *,
    model_load_seconds: float,
    inference_work_seconds: float,
    merge_seconds: float,
    encode_seconds: float,
    measured_frames: int,
    measured_tile_chunks: int,
    tile_count: int,
    window: int,
    overlap: int,
    frame_targets: Sequence[int] = (10, 100, 1000, 1920),
) -> dict[str, dict[str, float | int]]:
    inference_per_tile_chunk = inference_work_seconds / max(1, measured_tile_chunks)
    cpu_per_frame = (merge_seconds + encode_seconds) / max(1, measured_frames)
    estimates: dict[str, dict[str, float | int]] = {}
    for frames in frame_targets:
        chunks = temporal_chunk_count(frames, window, overlap)
        inference_units = chunks * tile_count
        estimates[str(frames)] = {
            "frames": frames,
            "temporal_chunks": chunks,
            "tile_chunk_inference_units": inference_units,
            "estimated_total_seconds": (
                model_load_seconds
                + inference_per_tile_chunk * inference_units
                + cpu_per_frame * frames
            ),
        }
    return estimates


def resolve_plan(args: argparse.Namespace, free_bytes: int) -> tuple[int, int, int, int]:
    # Conservative starting points, scaled by *measured* free memory.  Both are
    # CLI-overridable and subsequently reduced on CUDA OOM.
    free_gib = free_bytes / 2**30
    if args.temporal_window == "auto":
        window = (
            MINIMUM_TEMPORAL_UNIT
            if args.benchmark_mode in {"micro", "mini"}
            else 13 if free_gib >= 110 else 9 if free_gib >= 70 else 5
        )
    else:
        window = _valid_window(int(args.temporal_window))
    overlap_t = (5 if window >= 9 else 1) if args.temporal_overlap == "auto" else int(args.temporal_overlap)
    if overlap_t < 1 or overlap_t >= window or (window - overlap_t) % 4:
        raise ValueError("temporal overlap must be >=1, < window, and window-overlap must be divisible by 4")

    if args.tile_size == "auto":
        automatic_tile = 768 if free_gib >= 110 else 640 if free_gib >= 70 else 512
        # The retained H200 probe proved 488/60 at temporal 5/1.  Minimal
        # benchmarks start from that known-safe geometry instead of repeating
        # the already-documented 768 -> 612 -> 488 OOM search.
        tile = min(automatic_tile, 488) if args.benchmark_mode in {"micro", "mini"} else automatic_tile
    else:
        tile = _align_down(int(args.tile_size), 4)
    overlap_s = _align_down(tile // 8 if args.tile_overlap == "auto" else int(args.tile_overlap), 4)
    if overlap_s < 4 or overlap_s * 2 >= tile:
        raise ValueError("tile overlap must be >=4, aligned to 4, and smaller than half the source tile")
    return window, overlap_t, tile, overlap_s


def reduce_plan(window: int, overlap_t: int, tile: int, overlap_s: int) -> tuple[int, int, int, int]:
    if window > 5:
        new_window = window - 4
        new_overlap = min(overlap_t, new_window - 4)
        new_overlap = max(1, new_window - 4 * max(1, (new_window - new_overlap) // 4))
        return new_window, new_overlap, tile, min(overlap_s, tile // 3)
    new_tile = _align_down(int(tile * 0.8), 4)
    if new_tile < 128:
        raise RuntimeError("CUDA OOM persisted at the minimum safe temporal window and source tile size")
    new_overlap_s = min(overlap_s, _align_down(max(4, new_tile // 8), 4))
    return window, overlap_t, new_tile, new_overlap_s


def _starts(length: int, size: int, overlap: int) -> list[int]:
    if size >= length:
        return [0]
    stride = size - overlap
    starts = list(range(0, max(1, length - size + 1), stride))
    last = length - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tiles(width: int, height: int, size: int, overlap: int) -> tuple[list[Tile], int, int]:
    padded_w = math.ceil(width / 4) * 4
    padded_h = math.ceil(height / 4) * 4
    tile_w, tile_h = min(size, padded_w), min(size, padded_h)
    xs, ys = _starts(padded_w, tile_w, overlap), _starts(padded_h, tile_h, overlap)
    tiles = []
    for y0 in ys:
        for x0 in xs:
            tiles.append(Tile(len(tiles), x0, y0, min(x0 + tile_w, padded_w), min(y0 + tile_h, padded_h)))
    return tiles, padded_w, padded_h


def decode_chunks(
    info: MediaInfo,
    window: int,
    overlap: int,
    threads: int,
    *,
    source_start_frame: int = 0,
) -> Iterator[tuple[int, Any]]:
    import numpy as np

    end_frame = source_start_frame + info.frame_count
    filters = (
        f"trim=start_frame={source_start_frame}:end_frame={end_frame},"
        "setpts=PTS-STARTPTS"
    )
    command = [
        "ffmpeg", "-v", "error", "-threads", str(max(1, threads)),
        "-i", info.path, "-map", "0:v:0", "-vf", filters,
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    frame_bytes = info.width * info.height * 3
    buffer: deque[Any] = deque()
    start = 0
    eof = False

    def read_frame() -> Any | None:
        raw = process.stdout.read(frame_bytes)
        if not raw:
            return None
        if len(raw) != frame_bytes:
            raise RuntimeError(f"Truncated raw frame: {len(raw)} of {frame_bytes} bytes")
        return np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3).copy()

    try:
        while True:
            while len(buffer) < window and not eof:
                frame = read_frame()
                if frame is None:
                    eof = True
                else:
                    buffer.append(frame)
            if not buffer:
                break
            decoded_extent = start + len(buffer)
            if eof and decoded_extent < info.frame_count:
                raise RuntimeError(
                    f"ffmpeg ended at decoded frame {decoded_extent}, expected {info.frame_count}"
                )
            yield start, np.stack(buffer, axis=0)
            # An exact window-sized source does not set EOF until another read
            # is attempted.  Stop from the authoritative selected frame count
            # instead of emitting the retained overlap as a spurious terminal
            # chunk (5 frames previously became starts [0, 4]).
            if decoded_extent >= info.frame_count:
                break
            stride = window - overlap
            for _ in range(min(stride, len(buffer))):
                buffer.popleft()
            start += stride
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code and not sys.exc_info()[0]:
            raise RuntimeError(f"ffmpeg decode failed ({return_code}): {stderr[-2000:]}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def repository_context() -> Iterator[None]:
    old_cwd = pathlib.Path.cwd()
    old_path = list(sys.path)
    os.chdir(REPOSITORY)
    sys.path.insert(0, str(REPOSITORY))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def setup_distributed() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(_free_port())
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


class GPUSampler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        pid = str(os.getpid())
        while not self.stop_event.is_set():
            try:
                query = _run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
                fields = [x.strip() for x in query.stdout.splitlines()[0].split(",")]
                used = float(fields[0])
                try:
                    util = float(fields[1])
                except (ValueError, IndexError):
                    util = None
                process_mem = 0.0
                apps = _run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], check=False)
                for line in apps.stdout.splitlines():
                    fields = [x.strip() for x in line.split(",")]
                    if fields and fields[0] == pid:
                        process_mem += float(fields[1])
                self.samples.append({"total_used_mib": used, "utilization_percent": util, "process_mib": process_mem})
            except Exception:
                pass
            self.stop_event.wait(self.interval)

    def __enter__(self) -> "GPUSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 2)

    def summary(self) -> dict[str, float | None]:
        if not self.samples:
            return {"peak_process_gpu_mib": None, "peak_total_gpu_used_mib": None, "average_gpu_utilization_percent": None, "peak_gpu_utilization_percent": None}
        utilization = [
            float(sample["utilization_percent"])
            for sample in self.samples
            if sample["utilization_percent"] is not None
        ]
        return {
            "peak_process_gpu_mib": max(s["process_mib"] for s in self.samples),
            "peak_total_gpu_used_mib": max(s["total_used_mib"] for s in self.samples),
            "average_gpu_utilization_percent": sum(utilization) / len(utilization) if utilization else None,
            "peak_gpu_utilization_percent": max(utilization) if utilization else None,
            "utilization_unavailable_reason": None if utilization else "nvidia-smi utilization.gpu returned [Not Found]",
        }


def gpu_identity() -> dict[str, Any]:
    query = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
    fields = [part.strip() for part in query.stdout.splitlines()[0].split(",")]
    return {"name": fields[0], "total_mib": float(fields[1]), "free_mib_before": float(fields[2])}


def configure_runner() -> tuple[Any, dict[str, str]]:
    import torch
    from omegaconf import OmegaConf

    setup_distributed()
    # Both the benchmark and upstream repository have a top-level package
    # named ``common``. Shared utility callables above remain valid after their
    # modules are removed from the import cache; clearing only our package here
    # ensures upstream absolute imports resolve to SeedVR's own implementation.
    for module_name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", "") or ""
        if (module_name == "common" or module_name.startswith("common.")) and str(ARCHITECT_ROOT / "common") in module_file:
            del sys.modules[module_name]
    with repository_context():
        from common.config import load_config
        from common.distributed import init_torch
        from projects.video_diffusion_sr.infer import VideoDiffusionInfer

        config = load_config(str(REPOSITORY / "configs_3b" / "main.yaml"))
        OmegaConf.set_readonly(config, False)
        config.vae.checkpoint = str(CHECKPOINT_ROOT / "ema_vae.pth")
        config.diffusion.cfg.scale = 1.0
        config.diffusion.cfg.rescale = 0.0
        config.diffusion.timesteps.sampling.steps = 1
        runner = VideoDiffusionInfer(config)
        init_torch(cudnn_benchmark=False)
        runner.configure_dit_model(device="cuda", checkpoint=str(CHECKPOINT_ROOT / "seedvr2_ema_3b.pth"))
        runner.configure_vae_model()
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
        runner.configure_diffusion()
    versions = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": str(torch.version.cuda), "seedvr_git_commit": _run(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"]).stdout.strip(),
        "checkpoint": str(CHECKPOINT_ROOT / "seedvr2_ema_3b.pth"),
        "checkpoint_revision": "37255ff8cccfb01071b87f635a5948ca8d53117c",
        "transformers": importlib.metadata.version("transformers"),
        "diffusers": importlib.metadata.version("diffusers"),
        "flash_attn": importlib.metadata.version("flash-attn"),
        "apex": importlib.metadata.version("apex"),
    }
    return runner, versions


def _seed_for(base: int, chunk: int, tile: Tile) -> int:
    digest = hashlib.sha256(f"{base}:{chunk}:{tile.x0}:{tile.y0}".encode()).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**31)


def infer_tile(
    runner: Any,
    frames: Any,
    tile: Tile,
    valid: int,
    seed: int,
    phase_timing: dict[str, float] | None = None,
) -> Any:
    import numpy as np
    import torch
    import torch.nn.functional as F

    crop = frames[:, tile.y0:tile.y1, tile.x0:tile.x1]
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).to("cuda", non_blocking=True)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    target_size = ((tile.y1 - tile.y0) * SCALE, (tile.x1 - tile.x0) * SCALE)
    tensor = F.interpolate(tensor, size=target_size, mode="bicubic", align_corners=False, antialias=True)
    tensor = tensor.clamp_(0, 1).sub_(0.5).div_(0.5).permute(1, 0, 2, 3).contiguous()

    # The temporal VAE requires 4n+1 samples. Replication is trimmed exactly
    # after inference and never changes output frame count.
    padded_length = max(5, math.ceil((valid - 1) / 4) * 4 + 1)
    if tensor.shape[1] < padded_length:
        tensor = torch.cat([tensor, tensor[:, -1:].expand(-1, padded_length - tensor.shape[1], -1, -1)], dim=1)

    runner.dit.to("cpu")
    runner.vae.to("cuda")
    torch.cuda.synchronize()
    vae_encode_started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        latents = runner.vae_encode([tensor])[0]
    torch.cuda.synchronize()
    if phase_timing is not None:
        phase_timing["vae_encode_seconds"] = phase_timing.get("vae_encode_seconds", 0.0) + (
            time.perf_counter() - vae_encode_started
        )
    runner.vae.to("cpu")
    runner.dit.to("cuda")
    del tensor

    generator = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device, generator=generator)
    condition = runner.get_condition(noise, task="sr", latent_blur=latents)
    pos = torch.load(CHECKPOINT_ROOT / "pos_emb.pt", map_location="cuda", weights_only=True)
    neg = torch.load(CHECKPOINT_ROOT / "neg_emb.pt", map_location="cuda", weights_only=True)
    original_vae_decode = runner.vae_decode
    decode_elapsed = 0.0

    def timed_vae_decode(latents_to_decode: Any) -> Any:
        nonlocal decode_elapsed
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        decoded = original_vae_decode(latents_to_decode)
        torch.cuda.synchronize()
        decode_elapsed += time.perf_counter() - decode_started
        return decoded

    runner.vae_decode = timed_vae_decode
    torch.cuda.synchronize()
    diffusion_and_decode_started = time.perf_counter()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            samples = runner.inference(
                noises=[noise], conditions=[condition], texts_pos=[pos], texts_neg=[neg],
                cfg_scale=1.0, dit_offload=True,
            )
        torch.cuda.synchronize()
    finally:
        runner.vae_decode = original_vae_decode
    diffusion_and_decode = time.perf_counter() - diffusion_and_decode_started
    if phase_timing is not None:
        phase_timing["vae_decode_seconds"] = phase_timing.get("vae_decode_seconds", 0.0) + decode_elapsed
        phase_timing["diffusion_sampling_seconds"] = phase_timing.get(
            "diffusion_sampling_seconds", 0.0
        ) + max(0.0, diffusion_and_decode - decode_elapsed)
    runner.dit.to("cpu")
    sample = samples[0][:, :valid].permute(1, 2, 3, 0)
    sample = sample.clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8).cpu().numpy()
    del samples, condition, noise, latents, pos, neg
    torch.cuda.empty_cache()
    return sample


def _spatial_weight(tile: Tile, padded_w: int, padded_h: int, overlap: int) -> Any:
    import numpy as np

    h, w = (tile.y1 - tile.y0) * SCALE, (tile.x1 - tile.x0) * SCALE
    ov = overlap * SCALE
    wy, wx = np.ones(h, np.float32), np.ones(w, np.float32)
    ramp = (1.0 - np.cos(np.linspace(0, math.pi, ov, dtype=np.float32))) * 0.5
    if tile.x0 > 0:
        wx[:ov] = ramp
    if tile.x1 < padded_w:
        wx[-ov:] = ramp[::-1]
    if tile.y0 > 0:
        wy[:ov] = ramp
    if tile.y1 < padded_h:
        wy[-ov:] = ramp[::-1]
    return wy[:, None] * wx[None, :]


def _temporal_weight(local: int, valid: int, overlap: int, first: bool, last: bool) -> float:
    import numpy as np

    weight = 1.0
    if not first and local < overlap:
        weight *= float((1 - np.cos(math.pi * (local + 1) / (overlap + 1))) * 0.5)
    if not last and local >= valid - overlap:
        reverse = valid - local
        weight *= float((1 - np.cos(math.pi * reverse / (overlap + 1))) * 0.5)
    return weight


class VideoEncoder:
    def __init__(self, output: pathlib.Path, info: MediaInfo, encoder: dict[str, Any], threads: int):
        self.requested = output
        self.codec = str(encoder["name"])
        self.frame_count = info.frame_count
        self.frames_written = 0
        self.silent_path = output.with_name(output.stem + ".silent" + output.suffix)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-v", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{info.width*SCALE}x{info.height*SCALE}", "-r", info.fps, "-i", "-",
            "-an", "-threads", str(max(1, threads)), *encoder["arguments"], str(self.silent_path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: Any) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes(order="C"))
        self.frames_written += 1

    def close(self, input_path: pathlib.Path, *, copy_audio: bool = True) -> pathlib.Path:
        assert self.process.stdin is not None
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
        code = self.process.wait()
        if code:
            raise RuntimeError(f"video encoding failed ({code}): {stderr[-4000:]}")
        if self.frames_written != self.frame_count:
            raise RuntimeError(
                f"encoded {self.frames_written} frames, expected {self.frame_count}"
            )
        media = _shared_media()
        source = media.probe_media(input_path, count_frames=False)
        if not copy_audio:
            source = dataclasses.replace(source, audio_present=False, audio_codec=None)
        # This helper stream-copies audio and retries Matroska when the selected
        # codec's primary container cannot carry the original audio stream.
        return media.finalize_video_with_audio(
            self.silent_path,
            self.requested.with_suffix(""),
            input_path=input_path,
            input_metadata=source,
            extension=self.requested.suffix,
        )

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.silent_path.unlink(missing_ok=True)


def select_codec_for_media(requested: pathlib.Path, codec: str, scratch: pathlib.Path, threads: int, info: MediaInfo) -> tuple[pathlib.Path, dict[str, Any], list[dict[str, str]]]:
    # A real one-frame probe is run at the exact output dimensions. Unsupported
    # 16K codecs are skipped; resolution is never reduced as a fallback.
    encoder, attempts = _shared_media().select_encoder(
        info.width * SCALE, info.height * SCALE,
        threads=threads, scratch_dir=scratch, preferred=codec,
    )
    return requested.with_suffix(encoder["extension"]), encoder, attempts


def render_frames(
    stages: list[ChunkStage], tiles: list[Tile], padded_w: int, padded_h: int,
    info: MediaInfo, overlap_t: int, overlap_s: int, begin: int, end: int,
    encoder: VideoEncoder, scratch: pathlib.Path,
) -> tuple[float, float]:
    import numpy as np

    out_h, out_w = info.height * SCALE, info.width * SCALE
    acc_path, weight_path = scratch / "frame_acc.f32", scratch / "frame_weight.f32"
    spatial = {tile.index: _spatial_weight(tile, padded_w, padded_h, overlap_s) for tile in tiles}
    merge_seconds = 0.0
    encoder_feed_seconds = 0.0
    for frame_index in range(begin, min(end, info.frame_count)):
        merge_started = time.perf_counter()
        acc = np.memmap(acc_path, mode="w+", dtype=np.float32, shape=(out_h, out_w, 3))
        denom = np.memmap(weight_path, mode="w+", dtype=np.float32, shape=(out_h, out_w))
        acc[:] = 0
        denom[:] = 0
        for stage in stages:
            local = frame_index - stage.start
            if local < 0 or local >= stage.valid:
                continue
            tw = _temporal_weight(local, stage.valid, overlap_t, stage.start == 0, stage.start + stage.valid >= info.frame_count)
            for tile, tile_path in zip(tiles, stage.paths):
                values = np.load(tile_path, mmap_mode="r")[local]
                y0, y1, x0, x1 = tile.y0*SCALE, min(tile.y1*SCALE, out_h), tile.x0*SCALE, min(tile.x1*SCALE, out_w)
                if y0 >= out_h or x0 >= out_w:
                    continue
                weights = spatial[tile.index][:(y1-y0), :(x1-x0)] * tw
                acc[y0:y1, x0:x1] += values[:(y1-y0), :(x1-x0)].astype(np.float32) * weights[..., None]
                denom[y0:y1, x0:x1] += weights
        if np.any(denom <= 0):
            raise RuntimeError(f"Aggregation left uncovered pixels in output frame {frame_index}")
        frame = np.clip(acc / denom[..., None], 0, 255).round().astype(np.uint8)
        merge_seconds += time.perf_counter() - merge_started
        feed_started = time.perf_counter()
        encoder.write(frame)
        encoder_feed_seconds += time.perf_counter() - feed_started
        del frame, acc, denom
        acc_path.unlink(missing_ok=True)
        weight_path.unlink(missing_ok=True)
    return merge_seconds, encoder_feed_seconds


def run_attempt(
    runner: Any, info: MediaInfo, args: argparse.Namespace, plan: tuple[int, int, int, int],
    scratch: pathlib.Path, *, probe_only: bool,
) -> tuple[dict[str, Any], pathlib.Path | None]:
    import numpy as np
    import torch

    window, overlap_t, tile_size, overlap_s = plan
    tiles, padded_w, padded_h = make_tiles(info.width, info.height, tile_size, overlap_s)
    print(f"SeedVR2 plan: temporal={window}/{overlap_t}, source_tile={tile_size}/{overlap_s}, tiles={len(tiles)}", flush=True)
    inference_seconds = 0.0
    phase_timing = {
        "vae_encode_seconds": 0.0,
        "diffusion_sampling_seconds": 0.0,
        "vae_decode_seconds": 0.0,
        "merge_seconds": 0.0,
        "encode_seconds": 0.0,
    }
    stages: list[ChunkStage] = []
    emitted = 0
    output_path: pathlib.Path | None = None
    encoder = None
    encoder_attempts: list[dict[str, str]] = []
    encoder_name: str | None = None
    encoder_arguments: list[str] | None = None
    container_extension: str | None = None
    chunk_boundaries: list[int] = []
    decoded_end = 0
    if not probe_only:
        encoder_setup_started = time.perf_counter()
        resolved, encoder_spec, encoder_attempts = select_codec_for_media(pathlib.Path(args.output), args.encoder, scratch, args.ffmpeg_threads, info)
        output_path = resolved
        encoder_name = str(encoder_spec["name"])
        encoder_arguments = list(encoder_spec["arguments"])
        container_extension = resolved.suffix
        encoder = VideoEncoder(resolved, info, encoder_spec, args.ffmpeg_threads)
        phase_timing["encode_seconds"] += time.perf_counter() - encoder_setup_started

    try:
        for chunk_index, (start, frames) in enumerate(
            decode_chunks(
                info,
                window,
                overlap_t,
                args.ffmpeg_threads,
                source_start_frame=args.start_frame,
            )
        ):
            chunk_boundaries.append(start)
            valid = int(frames.shape[0])
            decoded_end = max(decoded_end, start + valid)
            if padded_h != info.height or padded_w != info.width:
                frames = np.pad(frames, ((0,0),(0,padded_h-info.height),(0,padded_w-info.width),(0,0)), mode="edge")
            chunk_dir = scratch / f"chunk_{chunk_index:06d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            paths: list[pathlib.Path] = []
            for tile in (tiles[:1] if probe_only else tiles):
                started = time.perf_counter()
                try:
                    result = infer_tile(
                        runner,
                        frames,
                        tile,
                        valid,
                        _seed_for(args.seed, chunk_index, tile),
                        phase_timing,
                    )
                    torch.cuda.synchronize()
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    raise RetryPlan(str(exc)) from exc
                inference_seconds += time.perf_counter() - started
                path = chunk_dir / f"tile_{tile.index:05d}.npy"
                np.save(path, result, allow_pickle=False)
                paths.append(path)
                del result
            if probe_only:
                break
            stages.append(ChunkStage(chunk_index, start, valid, paths))
            next_start = start + (window - overlap_t)
            assert encoder is not None
            merge_seconds, feed_seconds = render_frames(
                stages, tiles, padded_w, padded_h, info, overlap_t, overlap_s,
                emitted, next_start, encoder, scratch,
            )
            phase_timing["merge_seconds"] += merge_seconds
            phase_timing["encode_seconds"] += feed_seconds
            emitted = min(next_start, info.frame_count)
            # Once a chunk cannot contribute to an un-emitted frame, delete it.
            keep = []
            for stage in stages:
                if stage.start + stage.valid <= emitted:
                    shutil.rmtree(stage.paths[0].parent, ignore_errors=True)
                else:
                    keep.append(stage)
            stages = keep
        if not probe_only:
            if decoded_end != info.frame_count:
                raise RuntimeError(f"decoded frame extent {decoded_end} does not match ffprobe frame count {info.frame_count}")
            assert encoder is not None
            merge_seconds, feed_seconds = render_frames(
                stages, tiles, padded_w, padded_h, info, overlap_t, overlap_s,
                emitted, info.frame_count, encoder, scratch,
            )
            phase_timing["merge_seconds"] += merge_seconds
            phase_timing["encode_seconds"] += feed_seconds
            close_started = time.perf_counter()
            output_path = encoder.close(
                pathlib.Path(info.path),
                copy_audio=args.copy_audio,
            )
            phase_timing["encode_seconds"] += time.perf_counter() - close_started
            container_extension = output_path.suffix
            encoder = None
    finally:
        if encoder is not None:
            encoder.abort()
    neural_seconds = (
        phase_timing["vae_encode_seconds"]
        + phase_timing["diffusion_sampling_seconds"]
    )
    return {
        "sr_inference_seconds": inference_seconds,
        "neural_inference_seconds": neural_seconds,
        "conditioning_vae_encode_seconds": phase_timing["vae_encode_seconds"],
        "diffusion_sampling_seconds": phase_timing["diffusion_sampling_seconds"],
        "decode_seconds": phase_timing["vae_decode_seconds"],
        "merge_seconds": phase_timing["merge_seconds"],
        "encode_seconds": phase_timing["encode_seconds"],
        "encoding_mux_seconds": phase_timing["merge_seconds"] + phase_timing["encode_seconds"],
        "encoder": encoder_name,
        "encoder_arguments": encoder_arguments,
        "container_extension": container_extension,
        "encoder_probe_attempts": encoder_attempts,
        "chunk_boundaries": chunk_boundaries,
        "tile_count": len(tiles),
        "temporal_chunk_count": len(chunk_boundaries),
        "tile_chunk_inference_units": len(chunk_boundaries) * (1 if probe_only else len(tiles)),
        "effective_input_frames_processed": info.frame_count,
        "effective_output_frames_written": 0 if probe_only else info.frame_count,
    }, output_path


def validate_assets() -> None:
    required = [
        REPOSITORY / "projects" / "video_diffusion_sr" / "infer.py",
        REPOSITORY / "configs_3b" / "main.yaml", CHECKPOINT_ROOT / "seedvr2_ema_3b.pth",
        CHECKPOINT_ROOT / "ema_vae.pth", CHECKPOINT_ROOT / "pos_emb.pt", CHECKPOINT_ROOT / "neg_emb.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing official SeedVR2 assets:\n" + "\n".join(missing))
    remote = _run(["git", "-C", str(REPOSITORY), "remote", "get-url", "origin"]).stdout.strip()
    if "ByteDance-Seed/SeedVR" not in remote and "bytedance-seed/SeedVR" not in remote:
        raise RuntimeError(f"SeedVR repository is not the official ByteDance upstream: {remote}")


def write_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    media = _shared_media()
    path.parent.mkdir(parents=True, exist_ok=True)
    media.atomic_write_json(path, report)
    md = path.with_suffix(".md")
    timing = report.get("timing", {})
    gpu = report.get("gpu", {})
    lines = [
        "# SeedVR2 3B Inference Report", "",
        f"- Status: **{report.get('status', 'unknown')}**",
        f"- Benchmark mode: `{report.get('benchmark_mode', 'full')}`",
        f"- Input: `{report.get('input')}`", f"- Output: `{report.get('output')}`",
        f"- Scale: {SCALE}x", f"- Frames: {report.get('input_metadata', {}).get('frame_count')}",
        f"- Model load: {timing.get('model_load_seconds')} sec",
        f"- Neural inference total: {timing.get('neural_inference_seconds')} sec",
        f"- Conditioning VAE encode: {timing.get('conditioning_vae_encode_seconds')} sec",
        f"- Diffusion sampling: {timing.get('diffusion_sampling_seconds')} sec",
        f"- VAE decode: {timing.get('decode_seconds')} sec",
        f"- Spatial/temporal merge: {timing.get('merge_seconds')} sec",
        f"- Encode/mux: {timing.get('encode_seconds')} sec",
        f"- Total wall: {timing.get('total_wall_seconds')} sec",
        f"- Tile count: {report.get('tile_count')}",
        f"- Temporal chunks: {report.get('temporal_chunk_count')}",
        f"- Pure inference sec/input frame: {timing.get('pure_inference_seconds_per_input_frame')}",
        f"- End-to-end sec/input frame: {timing.get('end_to_end_seconds_per_input_frame')}",
        f"- Peak torch allocated: {gpu.get('peak_torch_allocated_mib')} MiB",
        f"- Peak torch reserved: {gpu.get('peak_torch_reserved_mib')} MiB",
        f"- Peak process GPU: {gpu.get('peak_process_gpu_mib')} MiB",
        f"- Peak total GPU used: {gpu.get('peak_total_gpu_used_mib')} MiB",
        "", "This run uses the official ByteDance SeedVR2-3B one-step `VideoDiffusionInfer` in BF16 autocast. No training or fine-tuning is performed.",
    ]
    media.atomic_write_text(md, "\n".join(lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official SeedVR2-3B one-step temporal video SR with OOM-safe x4 tiling")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT.parent))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--scratch-dir", default="/tmp", help="Parent for auto-cleaned runtime scratch")
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--report-json", help=argparse.SUPPRESS)
    parser.add_argument("--probe-json", help="Optional separate probe result JSON path")
    parser.add_argument("--probe-only", action="store_true", help="Run one real minimal temporal/tile inference and exit without a video")
    parser.add_argument(
        "--benchmark-mode",
        choices=["micro", "mini", "full"],
        default="full",
        help="micro=5 frames, mini=9 frames, full=all remaining frames",
    )
    parser.add_argument("--max-frames", type=int, help="Override the mode's frame limit")
    parser.add_argument("--start-frame", type=int, default=0, help="Zero-based decoded source start frame")
    parser.add_argument("--temporal-window", default="auto", help="4n+1 frame window or 'auto' (default: auto)")
    parser.add_argument("--temporal-overlap", default="auto", help="Temporal overlap or 'auto'; window-overlap must be divisible by 4")
    parser.add_argument("--tile-size", default="auto", help="Source-resolution square tile or 'auto' (default: auto)")
    parser.add_argument("--tile-overlap", default="auto", help="Source-pixel spatial overlap or 'auto'")
    parser.add_argument("--ffmpeg-threads", type=int, default=max(1, min(16, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--encoder", choices=["auto", "ffv1", "libx265", "libsvtav1", "hevc_nvenc", "av1_nvenc"], default="auto")
    parser.add_argument("--seed", type=int, default=666)
    # Reaching the 128px minimum from the largest automatic H200 plan can
    # require ten reductions (first temporal, then spatial).  Keep enough
    # retries to guarantee that the advertised minimum is actually attempted.
    parser.add_argument("--max-oom-retries", type=int, default=16)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.probe_only and not args.probe_json:
        raise SystemExit("--probe-only requires --probe-json so the normal benchmark report is not overwritten")
    input_path = pathlib.Path(args.input).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    report_path = pathlib.Path(args.probe_json or args.report_json or (pathlib.Path(args.result_dir) / "report.json")).resolve()
    started = time.perf_counter()
    report: dict[str, Any] = {
        "model": "SeedVR2-3B",
        "scale": SCALE,
        "input": str(input_path),
        "output": None,
        "probe_only": args.probe_only,
        "benchmark_mode": args.benchmark_mode,
    }
    try:
        validate_assets()
        source_info = probe_media(input_path)
        start_frame, selected_frames = benchmark_frame_range(
            args.benchmark_mode,
            source_info.frame_count,
            args.start_frame,
            args.max_frames,
        )
        args.start_frame = start_frame
        args.copy_audio = start_frame == 0 and selected_frames == source_info.frame_count
        info = dataclasses.replace(
            source_info,
            frame_count=selected_frames,
            duration=selected_frames / source_info.fps_float,
            has_audio=source_info.has_audio and args.copy_audio,
        )
        mode_suffix = "" if args.benchmark_mode == "full" and args.copy_audio else (
            f"_{args.benchmark_mode}_start{start_frame}_frames{selected_frames}"
        )
        args.output = args.output or str(
            output_dir / f"{input_path.stem}_SeedVR2_x4{mode_suffix}.mkv"
        )
        print_media(info)
        report["input_metadata"] = dataclasses.asdict(info)
        report["source_metadata"] = dataclasses.asdict(source_info)
        report["frame_selection"] = {
            "start_frame": start_frame,
            "selected_frames": selected_frames,
            "minimum_valid_temporal_unit": MINIMUM_TEMPORAL_UNIT,
        }
        identity = gpu_identity()
        report["gpu"] = identity
        plan = resolve_plan(args, int(identity["free_mib_before"] * 2**20))
        import torch
        torch.cuda.reset_peak_memory_stats()
        with GPUSampler(args.gpu_sample_interval) as sampler:
            load_started = time.perf_counter()
            runner, versions = configure_runner()
            load_seconds = time.perf_counter() - load_started
            report["environment"] = versions
            scratch_root = pathlib.Path(args.scratch_dir).resolve()
            scratch_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f"hrlrvo_seedvr2_{os.getpid()}_", dir=scratch_root
            ) as temp:
                scratch = pathlib.Path(temp)
                for attempt in range(args.max_oom_retries + 1):
                    shutil.rmtree(scratch, ignore_errors=True)
                    scratch.mkdir(parents=True)
                    try:
                        timing, output = run_attempt(
                            runner, info, args, plan, scratch,
                            probe_only=args.probe_only,
                        )
                        break
                    except RetryPlan:
                        if attempt >= args.max_oom_retries:
                            raise
                        old = plan
                        plan = reduce_plan(*plan)
                        print(
                            f"CUDA OOM: reducing SeedVR2 plan {old} -> {plan} and retrying",
                            flush=True,
                        )
                else:
                    raise RuntimeError("unreachable OOM retry state")
        gpu_metrics = sampler.summary()
        gpu_metrics.update({
            "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_torch_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        })
        report["gpu"].update(gpu_metrics)
        measured_peaks = [
            float(value) for value in (
                gpu_metrics.get("peak_process_gpu_mib"),
                gpu_metrics.get("peak_torch_reserved_mib"),
            ) if value is not None
        ]
        report["execution_plan"] = {"temporal_window": plan[0], "temporal_overlap": plan[1], "source_tile_size": plan[2], "source_tile_overlap": plan[3]}
        report.update({
            "recommended_temporal_window": plan[0],
            "recommended_temporal_overlap": plan[1],
            "recommended_tile_size": plan[2],
            "recommended_tile_overlap": plan[3],
            "peak_vram_mb": max(measured_peaks, default=None),
            "chunk_boundaries": timing.get("chunk_boundaries", []),
            "encoder": timing.get("encoder"),
            "encoder_arguments": timing.get("encoder_arguments"),
            "container_extension": timing.get("container_extension"),
            "encoder_probe_attempts": timing.get("encoder_probe_attempts", []),
            "tile_count": timing.get("tile_count"),
            "temporal_chunk_count": timing.get("temporal_chunk_count"),
            "tile_chunk_inference_units": timing.get("tile_chunk_inference_units"),
            "effective_input_frames_processed": timing.get("effective_input_frames_processed"),
            "effective_output_frames_written": timing.get("effective_output_frames_written"),
        })
        total = time.perf_counter() - started
        frames = int(timing.get("effective_input_frames_processed", info.frame_count))
        output_frames = int(timing.get("effective_output_frames_written", 0))
        neural_seconds = float(timing.get("neural_inference_seconds", timing["sr_inference_seconds"]))
        pure_inference_seconds = neural_seconds + float(timing.get("decode_seconds", 0.0))
        report["timing"] = {
            "model_load_seconds": load_seconds, **timing, "total_wall_seconds": total,
            "input_frame_count": info.frame_count,
            "inference_seconds_per_frame": timing["sr_inference_seconds"] / max(frames, 1),
            "total_seconds_per_frame": total / max(frames, 1),
            "effective_inference_fps": frames / max(timing["sr_inference_seconds"], 1e-9),
            "effective_end_to_end_fps": frames / max(total, 1e-9),
            "pure_inference_seconds_per_input_frame": pure_inference_seconds / max(frames, 1),
            "end_to_end_seconds_per_input_frame": total / max(frames, 1),
            "pure_inference_seconds_per_output_frame": pure_inference_seconds / max(output_frames or frames, 1),
            "end_to_end_seconds_per_output_frame": total / max(output_frames or frames, 1),
        }
        report["time_estimation"] = {
            "formula": (
                "model_load_once + full_sr_inference_seconds_per_tile_chunk * "
                "tile_count * temporal_chunk_count(N, window, overlap) + "
                "merge_encode_seconds_per_frame * N"
            ),
            "estimates": estimate_seed_times(
                model_load_seconds=load_seconds,
                inference_work_seconds=float(timing["sr_inference_seconds"]),
                merge_seconds=float(timing.get("merge_seconds", 0.0)),
                encode_seconds=float(timing.get("encode_seconds", 0.0)),
                measured_frames=frames,
                measured_tile_chunks=int(timing.get("tile_chunk_inference_units", 1)),
                tile_count=int(timing.get("tile_count", 1)),
                window=plan[0],
                overlap=plan[1],
            ),
        }
        report["output"] = str(output) if output else None
        report["status"] = "probe_pass" if args.probe_only else "inference_complete"
        write_report(report_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["timing"] = {"total_wall_seconds": time.perf_counter() - started}
        write_report(report_path, report)
        print(report["traceback"], file=sys.stderr)
        return 1
    finally:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
