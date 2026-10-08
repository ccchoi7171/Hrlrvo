"""Bounded CPU scheduling of the unchanged production spatial/center pixel math.

No temporal chunking, GPU operations, codec changes, or raw intermediate.
Each band retains the production tile accumulation order and float32 operations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

# This 02 overlay lives directly under the experiment root (unlike the
# 03/04 overlay at optimizations/final_only_v2/).
ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMPOSITE = ROOT / "package/src/cert_pipeline/flash_composite.py"
FROZEN_COMPOSITE_SHA256 = "8d1af95298e3f11d7433f6b539069fb5627112bbdbcc42bb8f3c1a98d0301a4e"


def _load_frozen_composite():
    if hashlib.sha256(FROZEN_COMPOSITE.read_bytes()).hexdigest() != FROZEN_COMPOSITE_SHA256:
        raise RuntimeError("frozen compositor SHA256 changed")
    name = "_a6000_frozen_flash_composite_v1"
    module = sys.modules.get(name)
    if module is not None:
        if Path(getattr(module, "__file__", "")).resolve() != FROZEN_COMPOSITE:
            raise RuntimeError("cached frozen compositor resolved to an unexpected file")
        return module
    spec = importlib.util.spec_from_file_location(name, FROZEN_COMPOSITE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen compositor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fc = _load_frozen_composite()


class FrameComposer:
    def __init__(self, reference, tiles, tile_dirs, *, workers, feather_width):
        import numpy as np

        if not isinstance(workers, int) or not 1 <= workers <= 16:
            raise ValueError("compose workers must be an integer in [1, 16]")
        self.tiles = tiles
        self.tile_dirs = tile_dirs
        self.feather_width = feather_width
        self.width, self.height = fc.EXPECTED_16K
        self.bands = []
        if len({tile.index for tile in tiles}) != len(tiles):
            raise fc.PipelineError("duplicate global tile index")
        axes = {tile.index: fc.tile_weight_axes(tile, tiles, reference) for tile in tiles}
        for start, end, active in fc.make_vertical_bands(tiles, self.height):
            normalization = np.zeros((end - start, self.width), dtype=np.float32)
            contributions = []
            for tile in active:
                x0, x1 = tile.x * fc.SCALE, (tile.x + tile.width) * fc.SCALE
                y0, y1 = start - tile.y * fc.SCALE, end - tile.y * fc.SCALE
                y_axis, x_axis = axes[tile.index]
                weight = y_axis[y0:y1, None] * x_axis[None, :]
                # Identical addition order/dtype to production, computed once
                # because positions and weights do not depend on frame pixels.
                normalization[:, x0:x1] += weight
                weight.setflags(write=False)
                contributions.append((tile.index, x0, x1, y0, y1, weight))
            if np.any(normalization <= 0):
                raise fc.PipelineError(f"zero tile normalization in rows [{start}, {end})")
            normalization.setflags(write=False)
            self.bands.append((start, end, contributions, normalization))
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spatial")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def render(self, frame_index, original):
        import cv2
        import numpy as np

        def read_tile(tile):
            path = self.tile_dirs[tile.index] / f"{frame_index:09d}.png"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise fc.PipelineError(f"missing/corrupt staged tile {tile.index}, frame {frame_index}: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            expected = (tile.height * fc.SCALE, tile.width * fc.SCALE, 3)
            if image.shape != expected:
                raise fc.PipelineError(f"staged tile {tile.index}, frame {frame_index} has shape {image.shape}, expected {expected}")
            return tile.index, image

        # Bounded to one frame's 170 PNGs, not the 1952-frame sequence.
        images = dict(self.pool.map(read_tile, self.tiles))
        output = np.empty((self.height, self.width, 3), dtype=np.uint8)

        def render_band(band):
            start, end, contributions, normalization = band
            accumulator = np.zeros((end - start, self.width, 3), dtype=np.float32)
            for index, x0, x1, y0, y1, weight in contributions:
                pixels = images[index][y0:y1].astype(np.float32)
                pixels *= weight[..., None]
                accumulator[:, x0:x1] += pixels
            rows = np.rint(accumulator / normalization[..., None]).clip(0, 255).astype(np.uint8)
            fc.composite_original_center_rows(
                rows, original, canvas_row_start=start, feather_width=self.feather_width,
            )
            # Disjoint row ranges: scheduling cannot change addition order or
            # introduce a concurrent write to any pixel.
            output[start:end] = rows

        list(self.pool.map(render_band, self.bands))
        return output


def encode_staged_tiles(*, args, reference, media_utils, input_metadata,
                        tiles, tile_dirs, selected_encoder, partial_output, feather_width):
    import cv2

    # Avoid 32 OpenCV inner threads per outer task; conversion remains the same
    # uint8 BGR->RGB operation. Codec threads and all quality flags are untouched.
    cv2.setNumThreads(1)
    workers = args.compose_workers
    started = time.perf_counter()
    reader = encoder = composer = prefetch = None
    component_seconds = {"original_decode_seconds": 0.0, "frame_compose_seconds": 0.0}
    try:
        composer = FrameComposer(reference, tiles, tile_dirs, workers=workers, feather_width=feather_width)
        reader = fc.OriginalFrameReader(
            args.original_8k, width=fc.EXPECTED_8K[0], height=fc.EXPECTED_8K[1],
            frame_count=args.expected_frames, threads=args.ffmpeg_threads,
        )
        encoder = media_utils.RawVideoEncoder(
            partial_output, width=fc.EXPECTED_16K[0], height=fc.EXPECTED_16K[1],
            fps=f"{fc.EXPECTED_OUTPUT_FPS.numerator}/{fc.EXPECTED_OUTPUT_FPS.denominator}",
            frame_count=args.expected_frames, encoder=selected_encoder,
            threads=args.ffmpeg_threads,
            source_metadata=fc.source_video_metadata(media_utils, input_metadata),
        )
        prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="next-frame")

        def prepare(index):
            begin = time.perf_counter()
            original = reader.read(index)
            component_seconds["original_decode_seconds"] += time.perf_counter() - begin
            begin = time.perf_counter()
            frame = composer.render(index, original)
            component_seconds["frame_compose_seconds"] += time.perf_counter() - begin
            return frame

        future = prefetch.submit(prepare, 0)
        for index in range(args.expected_frames):
            frame = future.result()
            if index + 1 < args.expected_frames:
                future = prefetch.submit(prepare, index + 1)
            # Exactly one encoder, in original temporal order. At most one next
            # frame is prepared while this write blocks on the encoder pipe.
            encoder.write(frame)
            encoder.frame_complete()
            del frame
            completed = index + 1
            if completed == 1 or completed % 10 == 0 or completed == args.expected_frames:
                elapsed = time.perf_counter() - started
                print(f"[ENCODE_FINAL_ONLY] video_id={args.video_id} frame={completed}/{args.expected_frames} "
                      f"elapsed={elapsed:.3f}s eta={elapsed / completed * (args.expected_frames - completed):.3f}s",
                      flush=True)
        reader.close()
        encoder.close()
        return {
            "seconds": time.perf_counter() - started,
            "frames_written": encoder.frames_written,
            "vertical_band_count": len(composer.bands),
            "memory_mode": "one-frame PNG cache, immutable global weights, at most one next RGB frame",
            "composition_workers": workers,
            "prefetch_frames": 1,
            "opencv_internal_threads": 1,
            "pixel_math": "unchanged production float32 tile order/rint/center helper",
            **component_seconds,
        }
    except BaseException:
        if reader is not None:
            reader.abort()
        if encoder is not None:
            encoder.abort()
        raise
    finally:
        if prefetch is not None:
            prefetch.shutdown(wait=True, cancel_futures=True)
        if composer is not None:
            composer.close()
