#!/usr/bin/env python3
"""Bounded, explicitly non-production full-layout CPU/encoder fixture benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "package/src"))
sys.path.insert(0, str(HERE))
from cert_pipeline import flash_composite as fc  # noqa: E402
import parallel_renderer  # noqa: E402


class HashEncoder:
    def __init__(self, path, *, frame_count, **_kwargs):
        self.frames_written = 0
        self.expected_frames = frame_count
        self.hashes = []
        self.current = hashlib.sha256()

    def write(self, rows):
        self.current.update(memoryview(rows).cast("B"))

    def frame_complete(self):
        self.frames_written += 1
        self.hashes.append(self.current.hexdigest())
        self.current = hashlib.sha256()

    def close(self):
        if self.frames_written != self.expected_frames:
            raise RuntimeError("fixture encoder frame count mismatch")

    def abort(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", choices=["03", "04"], required=True)
    parser.add_argument("--renderer", choices=["original", "threaded"], required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--frames", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if not out.is_relative_to(ROOT / "verification/final_only_v2_bench"):
        raise RuntimeError("benchmark may only write its dedicated verification namespace")
    out.mkdir(parents=True, exist_ok=False)
    reference_path = ROOT / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py"
    fc.require_approved_reference(reference_path)
    reference = fc.load_reference_module(reference_path)
    media_utils = fc.load_media_utils(reference_path)
    tiles = reference.make_tiles(3840, 2160, 256, 32)
    if len(tiles) != 170:
        raise RuntimeError("fixture requires original full170 layout")
    gpus = (0, 1) if args.video == "04" else (2, 3)
    sources = []
    for gpu in gpus:
        work = ROOT / "runs" / args.video / f"gpu{gpu}"
        for commit in sorted((work / "scratch/tile_commits").glob("*.json"))[:4]:
            record = json.loads(commit.read_text())
            if record["state"] != "COMPLETE" or record["frame_count"] != 1952:
                raise RuntimeError("fixture source tile is not fully committed")
            sources.append(work / "scratch/tiles" / f"tile_{record['tile']['index']:04d}")
    if len(sources) < 4:
        raise RuntimeError("at least four genuine committed source tiles required")
    directories = {}
    for tile in tiles:
        directory = out / "fixture_tiles" / f"tile_{tile.index:04d}"
        directory.mkdir(parents=True)
        for frame in range(args.frames):
            # The same immutable fixture is used for each renderer. Full-layout
            # geometry is genuine, but repeated tile content is NOT a final video.
            source = sources[tile.index % len(sources)] / f"{100 + frame:09d}.png"
            if not source.is_file():
                raise RuntimeError(f"missing committed fixture source: {source}")
            (directory / f"{frame:09d}.png").symlink_to(source)
        directories[tile.index] = directory
    anchor = ROOT / f"package/data/00_Input_8K_7680x4320/{args.video}_Input_8K.mp4"
    input_path = ROOT / f"package/data/02_Outpaint_4K_3840x2160/{args.video}/final/{args.video}_Outpaint_4K.mp4"
    fc.assert_video_identity_pair(args.video, input_path, anchor)
    metadata = fc.probe_video_headers(input_path)
    hashes = []
    selected = {
        "name": "libx265", "extension": ".mp4",
        "arguments": ["-c:v", "libx265", "-preset", "fast", "-crf", "14",
                      "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
                      "-x265-params", "log-level=error:pools=32:frame-threads=0"],
    }
    original_encoder = media_utils.RawVideoEncoder

    class CaptureEncoder(HashEncoder):
        def close(self):
            super().close()
            hashes.extend(self.hashes)

    if not args.encode:
        media_utils = SimpleNamespace(VideoMetadata=media_utils.VideoMetadata,
                                      RawVideoEncoder=CaptureEncoder)
    else:
        class CaptureRealEncoder(original_encoder):
            def __init__(self, *pos, **kw):
                super().__init__(*pos, **kw)
                self.digest = hashlib.sha256()

            def write(self, rows):
                self.digest.update(memoryview(rows).cast("B"))
                super().write(rows)

            def frame_complete(self):
                hashes.append(self.digest.hexdigest())
                self.digest = hashlib.sha256()
                super().frame_complete()

        media_utils = SimpleNamespace(VideoMetadata=media_utils.VideoMetadata,
                                      RawVideoEncoder=CaptureRealEncoder)
    call_args = SimpleNamespace(
        original_8k=anchor, expected_frames=args.frames, ffmpeg_threads=32,
        video_id=args.video, compose_workers=args.workers,
    )
    started = time.perf_counter()
    renderer = fc.encode_staged_tiles if args.renderer == "original" else parallel_renderer.encode_staged_tiles
    metrics = renderer(
        args=call_args, reference=reference, media_utils=media_utils, input_metadata=metadata,
        tiles=tiles, tile_dirs=directories, selected_encoder=selected,
        partial_output=out / "fixture_NOT_PRODUCTION.mp4", feather_width=64,
    )
    wall = time.perf_counter() - started
    headers = None
    if args.encode:
        headers = fc.probe_video_headers(out / "fixture_NOT_PRODUCTION.mp4")
        fc.validate_video_spec(headers, label="short fixture encode", resolution=fc.EXPECTED_16K,
                               fps=fc.EXPECTED_OUTPUT_FPS, frame_count=args.frames)
    result = {
        "status": "PASS", "fixture_only": True,
        "scope": "full170 genuine geometry with repeated committed tile PNGs and actual same-ID anchor",
        "limitations": ["repeated PNG source content/cache is not full video storage workload",
                        "short clip does not measure steady-state codec/GOP rate"],
        "video": args.video, "renderer": args.renderer, "workers": args.workers,
        "frames": args.frames, "real_encode": args.encode, "wall_seconds": wall,
        "metrics": metrics, "rgb_sha256": hashes, "headers": headers,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "encoder": selected, "anchor": str(anchor),
    }
    fc.atomic_write_json(out / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
