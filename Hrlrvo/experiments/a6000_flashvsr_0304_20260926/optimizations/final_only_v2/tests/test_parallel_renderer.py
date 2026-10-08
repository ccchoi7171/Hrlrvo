from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

OVERLAY = Path(__file__).resolve().parents[1]
ROOT = OVERLAY.parents[1]
sys.path.insert(0, str(ROOT / "package/src"))
sys.path.insert(0, str(OVERLAY))
import parallel_renderer  # noqa: E402
fc = parallel_renderer.fc


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "EXPECTED_16K", (64, 48))
    monkeypatch.setattr(fc, "EXPECTED_8K", (32, 24))
    monkeypatch.setattr(fc, "CENTER_BBOX", (12, 16, 36, 48))
    reference = fc.load_reference_module(
        ROOT / "package/src/flashvsr/architect/FlashVSR_v1.1/upscale.py"
    )
    tiles = reference.make_tiles(16, 12, 8, 2)
    rng = np.random.default_rng(71634910)
    tile_dirs = {}
    for tile in tiles:
        directory = tmp_path / f"tile_{tile.index}"
        directory.mkdir()
        tile_dirs[tile.index] = directory
        for frame in range(3):
            pixels = rng.integers(0, 256, (tile.height * 4, tile.width * 4, 3), dtype=np.uint8)
            assert cv2.imwrite(str(directory / f"{frame:09d}.png"), pixels)
    originals = [rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(3)]
    return reference, tiles, tile_dirs, originals


def encode_with_fakes(monkeypatch, scene, function, workers, feather):
    reference, tiles, tile_dirs, originals = scene
    frames = []
    read_indices = []
    closes = []

    class Reader:
        def __init__(self, *args, **kwargs):
            self.next = 0

        def read(self, index):
            assert index == self.next
            self.next += 1
            read_indices.append(index)
            return originals[index]

        def close(self):
            assert self.next == 3
            closes.append("reader")

        def abort(self):
            closes.append("reader_abort")

    class Encoder:
        def __init__(self, *args, **kwargs):
            self.rows = []
            self.frames_written = 0

        def write(self, rows):
            self.rows.append(rows.copy())

        def frame_complete(self):
            frames.append(np.concatenate(self.rows, axis=0))
            self.rows.clear()
            self.frames_written += 1

        def close(self):
            assert self.frames_written == 3
            closes.append("encoder")

        def abort(self):
            closes.append("encoder_abort")

    monkeypatch.setattr(fc, "OriginalFrameReader", Reader)
    monkeypatch.setattr(fc, "source_video_metadata", lambda *args: None)
    metrics = function(
        args=SimpleNamespace(
            original_8k=Path("03_Input_8K.mp4"), expected_frames=3,
            ffmpeg_threads=1, video_id="03", compose_workers=workers,
        ),
        reference=reference, media_utils=SimpleNamespace(RawVideoEncoder=Encoder),
        input_metadata={}, tiles=tiles, tile_dirs=tile_dirs, selected_encoder={},
        partial_output=Path("unused.mp4"), feather_width=feather,
    )
    assert metrics["frames_written"] == 3
    assert read_indices == [0, 1, 2]
    assert sorted(closes) == ["encoder", "reader"]
    return frames


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("feather", [0, 2, 4])
def test_exact_production_rgb_order_overlap_and_anchor(scene, monkeypatch, workers, feather):
    renderer = importlib.import_module("parallel_renderer")
    expected = encode_with_fakes(monkeypatch, scene, fc.encode_staged_tiles, 1, feather)
    actual = encode_with_fakes(monkeypatch, scene, renderer.encode_staged_tiles, workers, feather)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)


def test_missing_tile_is_not_silently_ignored(scene):
    renderer = importlib.import_module("parallel_renderer")
    reference, tiles, directories, originals = scene
    (directories[tiles[0].index] / "000000001.png").unlink()
    with renderer.FrameComposer(reference, tiles, directories, workers=2, feather_width=4) as composer:
        with pytest.raises(fc.PipelineError, match="missing/corrupt"):
            composer.render(1, originals[1])


def test_wrong_tile_shape_is_rejected(scene):
    renderer = importlib.import_module("parallel_renderer")
    reference, tiles, directories, originals = scene
    cv2.imwrite(str(directories[tiles[0].index] / "000000001.png"), np.zeros((4, 4, 3), np.uint8))
    with renderer.FrameComposer(reference, tiles, directories, workers=2, feather_width=4) as composer:
        with pytest.raises(fc.PipelineError, match="shape"):
            composer.render(1, originals[1])


@pytest.mark.parametrize("workers", [0, -1, 17])
def test_worker_count_is_bounded(scene, workers):
    renderer = importlib.import_module("parallel_renderer")
    reference, tiles, directories, _ = scene
    with pytest.raises(ValueError, match="workers"):
        renderer.FrameComposer(reference, tiles, directories, workers=workers, feather_width=4)


def test_full_geometry_is_required(scene):
    renderer = importlib.import_module("parallel_renderer")
    reference, tiles, directories, _ = scene
    with pytest.raises(fc.PipelineError, match="normalization|cover"):
        renderer.FrameComposer(reference, tiles[:1], directories, workers=1, feather_width=4)


def test_public_package_module_poisoning_cannot_replace_frozen_helper(monkeypatch):
    monkeypatch.setitem(sys.modules, "cert_pipeline.flash_composite", SimpleNamespace(__file__="/tmp/wrong.py"))
    assert parallel_renderer._load_frozen_composite() is fc


def test_cached_private_module_path_poisoning_is_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "_a6000_frozen_flash_composite_v1", SimpleNamespace(__file__="/tmp/wrong.py"))
    with pytest.raises(RuntimeError, match="unexpected file"):
        parallel_renderer._load_frozen_composite()
