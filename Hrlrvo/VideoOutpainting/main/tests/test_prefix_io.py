from __future__ import annotations

import json
from pathlib import Path

import av
import pytest
import torch

from ltx_outpaint.prefix_io import load_prefix_artifact, write_prefix_candidate
from ltx_outpaint.streaming_video import file_sha256


def _prefix(frames: int = 17) -> torch.Tensor:
    return torch.linspace(0, 1, frames * 3 * 4 * 6, dtype=torch.float32).reshape(
        frames, 3, 4, 6
    )


def _metadata() -> dict[str, object]:
    return {
        "source_video": "/input/source.mp4",
        "checkpoint": "/weights/checkpoint-007500",
        "config": "/config/inference_4k.yaml",
        "source_frame_range": [0, 16],
        "generation_timestamp": "2026-09-19T00:00:00Z",
        "memory_optimization": {
            "video_only_gpu_weights": True,
            "vae_decode_streaming": True,
            "ffn_token_chunking": False,
        },
    }


def test_candidate_roundtrip_preview_and_metadata(tmp_path: Path) -> None:
    original = _prefix()
    prefix_path, preview_path, meta_path, saved = write_prefix_candidate(
        original,
        tmp_path,
        seed=33333333,
        fps=24.0,
        metadata=_metadata(),
    )
    actual, loaded = load_prefix_artifact(
        prefix_path, expected_frames=17, height=4, width=6
    )

    assert prefix_path.name == "seed_33333333_prefix.pt"
    assert preview_path.name == "seed_33333333_preview.mp4"
    assert meta_path.name == "seed_33333333_meta.json"
    assert torch.equal(actual, original)
    assert actual.dtype == torch.float32
    assert actual.shape == (17, 3, 4, 6)
    assert loaded == saved == json.loads(meta_path.read_text())
    assert loaded["source_frame_range"] == [0, 16]
    assert loaded["memory_optimization"]["vae_decode_streaming"] is True
    assert loaded["tensor_sha256"] == file_sha256(prefix_path)
    with av.open(str(preview_path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (6, 4)
        assert sum(1 for _ in container.decode(video=0)) == 17
    assert not list(tmp_path.glob(".*.partial"))
    assert not list(tmp_path.glob("*.npy"))


def test_contiguous_view_does_not_serialize_full_backing(tmp_path: Path) -> None:
    backing = _prefix(40)
    view = backing[:17]
    assert view.is_contiguous() and view.untyped_storage().nbytes() > view.numel() * 4
    prefix_path, _, _, _ = write_prefix_candidate(
        view, tmp_path, seed=1, fps=24.0, metadata=_metadata()
    )
    assert prefix_path.stat().st_size < backing.numel() * 4
    actual, _ = load_prefix_artifact(prefix_path, expected_frames=17, height=4, width=6)
    assert torch.equal(actual, view)


def test_rejects_corrupt_or_incomplete_artifact(tmp_path: Path) -> None:
    prefix_path, _, meta_path, _ = write_prefix_candidate(
        _prefix(), tmp_path, seed=2, fps=24.0, metadata=_metadata()
    )
    with prefix_path.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 255]))
    with pytest.raises(ValueError, match="SHA-256"):
        load_prefix_artifact(prefix_path, expected_frames=17, height=4, width=6)
    meta_path.unlink()
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_prefix_artifact(prefix_path, expected_frames=17, height=4, width=6)


def test_rejects_bad_geometry_dtype_and_unusable_pixels(tmp_path: Path) -> None:
    original = _prefix()
    with pytest.raises(ValueError, match="shape"):
        write_prefix_candidate(original[0], tmp_path, seed=0, fps=24.0, metadata={})
    with pytest.raises(TypeError, match="FP32"):
        write_prefix_candidate(original.half(), tmp_path, seed=0, fps=24.0, metadata={})
    with pytest.raises(ValueError, match="NaN or Inf"):
        bad = original.clone()
        bad[0, 0, 0, 0] = float("nan")
        write_prefix_candidate(bad, tmp_path, seed=0, fps=24.0, metadata={})
    with pytest.raises(ValueError, match="range"):
        write_prefix_candidate(original + 1, tmp_path, seed=0, fps=24.0, metadata={})

    prefix_path, _, _, _ = write_prefix_candidate(
        original, tmp_path, seed=0, fps=24.0, metadata=_metadata()
    )
    with pytest.raises(ValueError, match="geometry"):
        load_prefix_artifact(prefix_path, expected_frames=16, height=4, width=6)
    with pytest.raises(ValueError, match="geometry"):
        load_prefix_artifact(prefix_path, expected_frames=17, height=8, width=6)
    with pytest.raises(FileExistsError, match="complete"):
        write_prefix_candidate(original, tmp_path, seed=0, fps=24.0, metadata=_metadata())
