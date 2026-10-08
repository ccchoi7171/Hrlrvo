#!/usr/bin/env python3
"""Run one isolated, global-index-preserving FlashVSR spatial tile shard.

This launcher deliberately does not modify the packaged production
``upscale.py``.  A child process imports that snapshot, replaces only its
``make_tiles`` result with a disjoint selection of the original global Tile
objects, and runs the original staged-inference implementation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from fractions import Fraction
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSCALE = ROOT / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py"
BASE_SEED = 71634910
TILE_SIZE = 256
TILE_OVERLAP = 32
SPARSE_RATIO = 2.0
LOCAL_RANGE = 11
FFMPEG_THREADS = 32
ENCODER = "libx265"
FULL_FRAMES = 1952
APPROVED_UPSCALE_SHA256 = "3db9c0524418070a87b26b948612f8988cd47f007c6e7739a89d3911ed0d703d"
ASSIGNMENTS = {
    ("04", 0): (0, 84),
    ("04", 1): (85, 169),
    ("03", 2): (0, 84),
    ("03", 3): (85, 169),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_approved_upscale(path: Path) -> str:
    actual = sha256_file(path)
    if actual != APPROVED_UPSCALE_SHA256:
        raise RuntimeError(
            f"FlashVSR upscale snapshot is not the approved production code: {path} "
            f"(expected {APPROVED_UPSCALE_SHA256}, got {actual})"
        )
    return actual


def load_upscale(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("a6000_isolated_flashvsr_upscale", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import FlashVSR snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_video_paths(video_id: str, input_path: Path, anchor_path: Path) -> None:
    expected_input = f"{video_id}_Outpaint_4K.mp4"
    expected_anchor = f"{video_id}_Input_8K.mp4"
    if input_path.name != expected_input:
        raise ValueError(f"Video {video_id} input basename must be {expected_input}, got {input_path.name}")
    if anchor_path.name != expected_anchor:
        raise ValueError(f"Video {video_id} anchor basename must be {expected_anchor}, got {anchor_path.name}")
    if input_path.stem.split("_", 1)[0] != anchor_path.stem.split("_", 1)[0]:
        raise ValueError("Outpaint input and 8K anchor video IDs do not match")


def validate_assignment(video_id: str, gpu_index: int, tile_start: int, tile_end: int) -> None:
    expected = ASSIGNMENTS.get((video_id, gpu_index))
    if expected is None:
        raise ValueError(f"GPU {gpu_index} is not assigned to video {video_id}")
    if (tile_start, tile_end) != expected:
        raise ValueError(
            f"Video {video_id} GPU {gpu_index} must own global tiles {expected[0]}-{expected[1]}, "
            f"got {tile_start}-{tile_end}"
        )


def select_global_tiles(
    original_make_tiles: Callable[[int, int, int, int], list[Any]],
    tile_start: int,
    tile_end: int,
) -> Callable[[int, int, int, int], list[Any]]:
    """Return a make_tiles wrapper retaining original Tile objects and indices."""
    def selected(width: int, height: int, tile_size: int, overlap: int) -> list[Any]:
        all_tiles = original_make_tiles(width, height, tile_size, overlap)
        if len(all_tiles) != 170:
            raise RuntimeError(f"Expected the production 170-tile layout, got {len(all_tiles)}")
        if tile_start < 0 or tile_end >= len(all_tiles) or tile_start > tile_end:
            raise ValueError(f"Invalid global tile selection {tile_start}-{tile_end}")
        result = all_tiles[tile_start:tile_end + 1]
        if [tile.index for tile in result] != list(range(tile_start, tile_end + 1)):
            raise RuntimeError("Global tile indices changed while selecting the shard")
        return result
    return selected


def cheap_header_ffprobe(upscale: Any, path: Path) -> dict[str, Any]:
    """Read container counters without the production full-decode count pass.

    The transferred 03/04 inputs have authoritative constant-rate ``nb_frames``
    counters.  Missing counters fail closed instead of silently estimating.
    """
    command = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ]
    data = json.loads(upscale.run(command, capture_output=True).stdout)
    video = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream: {path}")
    audio = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"), None)
    rate_text = video.get("avg_frame_rate") or video.get("r_frame_rate")
    fps = Fraction(rate_text)
    frames_text = video.get("nb_frames")
    if fps <= 0 or not frames_text or frames_text == "N/A":
        raise RuntimeError("Input lacks authoritative header fps/frame counters; refusing an estimated shard run")
    duration = video.get("duration") or data.get("format", {}).get("duration")
    return {
        "path": str(path.resolve()),
        "width": int(video["width"]), "height": int(video["height"]),
        "fps": float(fps), "fps_fraction": str(fps), "frame_count": int(frames_text),
        "duration": float(duration) if duration not in (None, "N/A") else int(frames_text) / float(fps),
        "pixel_format": video.get("pix_fmt"), "audio_present": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
    }


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    anchor_path = Path(args.anchor).resolve()
    assert_video_paths(args.video_id, input_path, anchor_path)
    validate_assignment(args.video_id, args.gpu_index, args.tile_start, args.tile_end)
    if not input_path.is_file() or not anchor_path.is_file():
        raise FileNotFoundError("Input and anchor must both exist before a shard can start")
    selected_start, selected_end = args.tile_start, args.tile_end
    max_frames = None
    benchmark_mode = "full"
    if args.mode == "smoke":
        smoke_tile = args.smoke_global_tile
        if smoke_tile is None:
            smoke_tile = selected_start
        if not selected_start <= smoke_tile <= selected_end:
            raise ValueError("Smoke global tile must belong to this GPU's approved assignment")
        selected_start = selected_end = smoke_tile
        max_frames = 29
        benchmark_mode = "mini"
    work_dir = Path(args.work_dir).resolve()
    upscale_path = Path(args.upscale_path).resolve()
    if not upscale_path.is_file():
        raise FileNotFoundError(f"Packaged upscale snapshot is missing: {upscale_path}")
    upscale_sha256 = require_approved_upscale(upscale_path)
    return {
        "schema": "a6000_flashvsr_shard_request/v1",
        "video_id": args.video_id,
        "mode": args.mode,
        "input": str(input_path),
        "anchor": str(anchor_path),
        "input_identity": {
            "path": str(input_path), "size": input_path.stat().st_size, "sha256": sha256_file(input_path),
        },
        "anchor_identity": {
            "path": str(anchor_path), "size": anchor_path.stat().st_size, "sha256": sha256_file(anchor_path),
        },
        "gpu_index": args.gpu_index,
        "assignment": {"tile_start": args.tile_start, "tile_end": args.tile_end},
        "selection": {"tile_start": selected_start, "tile_end": selected_end},
        "work_dir": str(work_dir),
        "output_target": str(Path(args.output_target).resolve()),
        "upscale_path": str(upscale_path),
        "upscale_sha256": upscale_sha256,
        "parameters": {
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "base_seed": BASE_SEED,
            "sparse_ratio": SPARSE_RATIO,
            "local_range": LOCAL_RANGE,
            "precision": "torch.bfloat16 (original implementation)",
            "ffmpeg_threads": FFMPEG_THREADS,
            "encoder": ENCODER,
            "benchmark_mode": benchmark_mode,
            "max_frames": max_frames,
        },
    }


def validate_request_contract(request: dict[str, Any]) -> None:
    if request.get("schema") != "a6000_flashvsr_shard_request/v1":
        raise ValueError("Unsupported shard request schema")
    video_id = request.get("video_id")
    gpu_index = int(request.get("gpu_index", -1))
    assignment = request.get("assignment", {})
    tile_start = int(assignment.get("tile_start", -1))
    tile_end = int(assignment.get("tile_end", -1))
    validate_assignment(video_id, gpu_index, tile_start, tile_end)
    assert_video_paths(video_id, Path(request["input"]), Path(request["anchor"]))
    expected_common = {
        "tile_size": TILE_SIZE, "tile_overlap": TILE_OVERLAP, "base_seed": BASE_SEED,
        "sparse_ratio": SPARSE_RATIO, "local_range": LOCAL_RANGE,
        "precision": "torch.bfloat16 (original implementation)",
        "ffmpeg_threads": FFMPEG_THREADS, "encoder": ENCODER,
    }
    params = request.get("parameters", {})
    if any(params.get(key) != value for key, value in expected_common.items()):
        raise ValueError("Shard request changes an approved FlashVSR inference parameter")
    selection = request.get("selection", {})
    selected_start = int(selection.get("tile_start", -1))
    selected_end = int(selection.get("tile_end", -1))
    mode = request.get("mode")
    if mode == "full":
        if ((selected_start, selected_end) != (tile_start, tile_end)
                or params.get("benchmark_mode") != "full" or params.get("max_frames") is not None):
            raise ValueError("Full shard request does not cover its exact approved assignment")
    elif mode == "smoke":
        if (selected_start != selected_end or not tile_start <= selected_start <= tile_end
                or params.get("benchmark_mode") != "mini" or params.get("max_frames") != 29):
            raise ValueError("Smoke shard request is not one approved global tile over 29 frames")
    else:
        raise ValueError(f"Unsupported shard mode: {mode!r}")


def _internal_run(request_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    validate_request_contract(request)
    work_dir = Path(request["work_dir"]).resolve()
    upscale_path = Path(request["upscale_path"]).resolve()
    if request.get("upscale_sha256") != APPROVED_UPSCALE_SHA256:
        raise RuntimeError("Shard request does not cite the approved FlashVSR upscale snapshot")
    # Rehash immediately before import so code changed after request creation
    # cannot execute under an earlier approved identity.
    require_approved_upscale(upscale_path)
    upscale = load_upscale(upscale_path)
    original_make_tiles = upscale.make_tiles
    start = int(request["selection"]["tile_start"])
    end = int(request["selection"]["tile_end"])
    upscale.make_tiles = select_global_tiles(original_make_tiles, start, end)
    upscale.ffprobe = lambda path: cheap_header_ffprobe(upscale, path)

    scratch = work_dir / "scratch"
    result_dir = work_dir / "result"
    metadata_dir = work_dir / "temporal_metadata"
    partial_manifest = work_dir / "private_partial_stage_manifest.json"
    params = request["parameters"]
    argv = [
        "--input", request["input"],
        "--output", request["output_target"],
        "--result-dir", str(result_dir),
        "--report-json", str(result_dir / "report.json"),
        "--tile-size", str(params["tile_size"]),
        "--tile-overlap", str(params["tile_overlap"]),
        "--min-tile-size", str(params["tile_size"]),
        "--sparse-ratio", str(params["sparse_ratio"]),
        "--local-range", str(params["local_range"]),
        "--seed", str(params["base_seed"]),
        "--gpu-index", str(request["gpu_index"]),
        "--ffmpeg-threads", str(params["ffmpeg_threads"]),
        "--encoder", params["encoder"],
        "--scratch-path", str(scratch),
        "--stage-request", str(request_path),
        "--stage-mode", "inference",
        "--stage-manifest", str(partial_manifest),
        "--artifact-metadata-dir", str(metadata_dir),
        "--benchmark-mode", params["benchmark_mode"],
    ]
    if params["max_frames"] is not None:
        argv.extend(["--max-frames", str(params["max_frames"])])
    parsed = upscale.parser().parse_args(argv)
    report = upscale.execute(parsed)
    upscale.write_report(report, result_dir, result_dir / "report.json")
    if report.get("status") != "STAGED":
        raise RuntimeError(f"Original FlashVSR inference did not stage successfully: {report.get('status')}")
    return 0


def _validate_private_manifest(request: dict[str, Any], manifest: dict[str, Any]) -> None:
    start = int(request["selection"]["tile_start"])
    end = int(request["selection"]["tile_end"])
    expected = list(range(start, end + 1))
    actual = [int(item["tile"]["index"]) for item in manifest.get("tile_artifacts", [])]
    if manifest.get("state") != "COMPLETE" or actual != expected:
        raise RuntimeError("Private partial stage manifest does not exactly match the selected global tiles")
    expected_frames = 29 if request["mode"] == "smoke" else FULL_FRAMES
    if int(manifest.get("processed_input_frames", -1)) != expected_frames:
        raise RuntimeError("Private partial stage manifest has the wrong processed frame count")
    for artifact in manifest["tile_artifacts"]:
        index = int(artifact["tile"]["index"])
        if int(artifact["tile_seed"]) != BASE_SEED + index:
            raise RuntimeError(f"Tile {index} did not retain its global-index seed")


def run_worker(args: argparse.Namespace) -> Path:
    request = build_request(args)
    work_dir = Path(request["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / ".worker.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another worker already owns this isolated work directory: {work_dir}") from error
        request_path = work_dir / "shard_request.json"
        if request_path.exists():
            prior = json.loads(request_path.read_text(encoding="utf-8"))
            if prior != request:
                raise RuntimeError(f"Existing shard request differs from this run: {request_path}")
        else:
            write_json_atomic(request_path, request)
        log_path = work_dir / "worker.log"
        command = [sys.executable, str(Path(__file__).resolve()), "_internal", "--request", str(request_path)]
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"FlashVSR shard child failed with exit {completed.returncode}; see {log_path}")
    manifest_path = work_dir / "private_partial_stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_private_manifest(request, manifest)
    receipt = {
        "schema": "a6000_flashvsr_tile_shard/v1",
        "state": "SHARD_COMPLETE",
        "completion_scope": "PARTIAL_SPATIAL_TILES",
        "video_id": request["video_id"],
        "mode": request["mode"],
        "input_path": request["input"],
        "anchor_path": request["anchor"],
        "input_identity": request["input_identity"],
        "anchor_identity": request["anchor_identity"],
        "gpu_index": request["gpu_index"],
        "assignment": request["assignment"],
        "selection": request["selection"],
        "parameters": request["parameters"],
        "work_dir": str(work_dir.resolve()),
        "request_path": str(request_path.resolve()),
        "request_sha256": sha256_file(request_path),
        "private_partial_manifest": str(manifest_path.resolve()),
        "private_partial_manifest_sha256": sha256_file(manifest_path),
        "upscale_path": request["upscale_path"],
        "upscale_sha256": request["upscale_sha256"],
        "subprocess": {"exit_code": completed.returncode, "log_path": str(log_path.resolve())},
        "global_tile_indices": [
            int(item["tile"]["index"]) for item in manifest["tile_artifacts"]
        ],
        "processed_input_frames": int(manifest["processed_input_frames"]),
        "model": manifest["model"],
        "repository_commit": manifest["partial_report"]["environment"].get("repository_commit"),
    }
    receipt_path = work_dir / "shard_receipt.json"
    write_json_atomic(receipt_path, receipt)
    return receipt_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-id", required=True, choices=["03", "04"])
    p.add_argument("--input", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--gpu-index", required=True, type=int, choices=range(4))
    p.add_argument("--tile-start", required=True, type=int)
    p.add_argument("--tile-end", required=True, type=int)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--output-target", required=True)
    p.add_argument("--mode", choices=["smoke", "full"], default="full")
    p.add_argument("--smoke-global-tile", type=int)
    p.add_argument("--upscale-path", default=str(DEFAULT_UPSCALE))
    return p


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_internal":
        internal = argparse.ArgumentParser()
        internal.add_argument("--request", required=True)
        args = internal.parse_args(sys.argv[2:])
        return _internal_run(Path(args.request).resolve())
    args = parser().parse_args()
    receipt = run_worker(args)
    print(json.dumps({"status": "SHARD_COMPLETE", "receipt": str(receipt)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
