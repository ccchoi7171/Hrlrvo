"""CPU-only checks that a selected FP32 prefix enters chunk zero's condition."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import pytest
import torch

from ltx_outpaint import inference
from ltx_outpaint.prefix_io import write_prefix_candidate
from ltx_outpaint.streaming_video import file_sha256, iter_segment_frames

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path, count: int) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = 16
        stream.height = 8
        stream.pix_fmt = "yuv420p"
        for index in range(count):
            pixels = np.full((8, 16, 3), index % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _settings(adapter: Path, base: Path) -> dict[str, Any]:
    return {
        "ltx_repo": str(ROOT.parent / "models/LTX2.3"),
        "model": {
            "base_model_path": str(base),
            "video_vae_path": None,
            "adapter_path": str(adapter),
        },
        "provenance": {"base_sha256": "synthetic-test-only"},
        "architecture": {"architecture_version": 8},
        "video": {
            "canvas": [16, 8],
            "fps": 24.0,
            "fps_tolerance": 0.001,
            "expected_anchor": [16, 8],
            "enforce_expected_anchor": True,
            "boundary_blend_width_pixels": 0,
        },
        "long_range": {
            "clip_frames": 65,
            "prefix_frames": 17,
            "stride_frames": 48,
            "final_chunk_policy": "repeat_last",
        },
        "sampling": {
            "negative_prompt": "",
            "seed": 33333333,
            "inference_steps": 30,
            "cfg_scale": 3.0,
            "stg_scale": 1.0,
            "stg_blocks": [28],
            "guidance_rescale": 0.7,
        },
        "runtime": {"device": "cpu", "dtype": "bfloat16"},
    }


def _artifact(tmp_path: Path, source: Path, adapter: Path) -> tuple[Path, torch.Tensor]:
    prefix = torch.linspace(0, 1, 17 * 3 * 8 * 16).reshape(17, 3, 8, 16)
    path, _, _, _ = write_prefix_candidate(
        prefix,
        tmp_path / "candidates",
        seed=33333333,
        fps=24.0,
        metadata={
            "source_path": str(source.resolve()),
            "source_sha256": file_sha256(source),
            "source_frame_range": [0, 17],
            "adapter_sha256": file_sha256(adapter),
        },
    )
    return path, prefix


def test_chunk_zero_uses_selected_prefix_then_recurrent_prefix_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    _source(source, 114)
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"synthetic-adapter")
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"synthetic-base")
    selected_path, selected = _artifact(tmp_path, source, adapter)
    settings = _settings(adapter, base)
    inference._install_official_sources(ROOT.parent / "models/LTX2.3")
    from ltx_trainer import model_loader

    monkeypatch.setattr(
        model_loader,
        "read_video_scale_factors",
        lambda _: SimpleNamespace(time=8, height=8, width=8),
    )
    load_calls: list[int] = []
    monkeypatch.setattr(
        inference,
        "_load_models",
        lambda *args, **kwargs: (
            load_calls.append(1) or object(), object(), object(), {"mode": "mock"}
        ),
    )
    prefix_inputs: list[torch.Tensor | None] = []

    def fake_chunk(**kwargs: Any) -> torch.Tensor:
        source_chunk = kwargs["anchor_frames"]
        start = kwargs["absolute_start_frame"]
        previous = kwargs["previous_prefix"]
        prefix_inputs.append(None if previous is None else previous.clone())
        result = torch.empty((len(source_chunk), 3, 8, 16), dtype=torch.float32)
        for index in range(len(source_chunk)):
            result[index].fill_((start + index) / 255.0)
        if previous is not None:
            result[: len(previous)] = previous
        return result

    monkeypatch.setattr(inference, "sample_chunk", fake_chunk)
    output = tmp_path / "output" / "video.mp4"
    segment_dir = tmp_path / "segments"
    result = inference.sample_long_range(
        settings,
        source,
        output,
        num_frames=114,
        checkpoint_path=adapter,
        segment_dir=segment_dir,
        keep_intermediates=True,
        initial_prefix_path=selected_path,
    )
    assert result == output.resolve()
    assert len(load_calls) == 1
    assert len(prefix_inputs) == 3
    assert torch.equal(prefix_inputs[0], selected)
    assert prefix_inputs[1] is not None
    assert torch.equal(
        prefix_inputs[1],
        torch.stack([torch.full_like(selected[0], (48 + i) / 255.0) for i in range(17)]),
    )
    assert torch.equal(
        prefix_inputs[2],
        torch.stack([torch.full_like(selected[0], (96 + i) / 255.0) for i in range(17)]),
    )
    manifest = json.loads((output.parent / "inference_manifest.json").read_text())
    records = json.loads((output.parent / "chunk_lineage.json").read_text())["chunks"]
    assert manifest["initial_prefix_bitwise_equal"] is True
    assert records[0]["initial_prefix_bitwise_equal"] is True
    assert manifest["prefix_gap_count"] == manifest["prefix_duplicate_count"] == 0
    assert [(r["new_output_frame_range"]) for r in records] == [
        [0, 65], [65, 113], [113, 114]
    ]
    expected_first = selected.mul(255).permute(0, 2, 3, 1).to(torch.uint8).numpy()
    segment_frames = iter_segment_frames(records, segment_dir)
    for expected in expected_first:
        assert np.array_equal(next(segment_frames), expected)
    with av.open(str(output)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 114

    # A fully committed resume reassembles without loading a model or replacing
    # the exact selected artifact's run identity.
    result2 = inference.sample_long_range(
        settings,
        source,
        output,
        num_frames=114,
        checkpoint_path=adapter,
        segment_dir=segment_dir,
        resume=True,
        keep_intermediates=True,
        initial_prefix_path=selected_path,
    )
    assert result2 == result
    assert len(load_calls) == 1

    # Simulate a crash before chunk 2's JSON completion marker. Resume must
    # restore chunk 1's recurrent prefix/RNG and regenerate only the tail.
    (segment_dir / "chunk_000002.json").unlink()
    prefix_inputs.clear()
    result3 = inference.sample_long_range(
        settings,
        source,
        output,
        num_frames=114,
        checkpoint_path=adapter,
        segment_dir=segment_dir,
        resume=True,
        keep_intermediates=True,
        initial_prefix_path=selected_path,
    )
    assert result3 == result
    assert len(load_calls) == 2
    assert len(prefix_inputs) == 1
    assert torch.equal(
        prefix_inputs[0],
        torch.stack([torch.full_like(selected[0], (96 + i) / 255.0) for i in range(17)]),
    )


def test_existing_65f_mp4_bootstraps_chunk_zero_and_generation_starts_at_48(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    bootstrap = tmp_path / "bootstrap.mp4"
    _source(source, 114)
    _source(bootstrap, 65)
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"synthetic-adapter")
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"synthetic-base")
    settings = _settings(adapter, base)
    inference._install_official_sources(ROOT.parent / "models/LTX2.3")
    from ltx_trainer import model_loader

    monkeypatch.setattr(
        model_loader,
        "read_video_scale_factors",
        lambda _: SimpleNamespace(time=8, height=8, width=8),
    )
    load_calls: list[int] = []
    monkeypatch.setattr(
        inference,
        "_load_models",
        lambda *args, **kwargs: (
            load_calls.append(1) or object(), object(), object(), {"mode": "mock"}
        ),
    )
    sampled_starts: list[int] = []
    prefix_inputs: list[torch.Tensor] = []

    def fake_chunk(**kwargs: Any) -> torch.Tensor:
        source_chunk = kwargs["anchor_frames"]
        start = kwargs["absolute_start_frame"]
        previous = kwargs["previous_prefix"]
        assert previous is not None
        sampled_starts.append(start)
        prefix_inputs.append(previous.clone())
        result = torch.empty((len(source_chunk), 3, 8, 16), dtype=torch.float32)
        for index in range(len(source_chunk)):
            result[index].fill_((start + index) / 255.0)
        result[: len(previous)] = previous
        return result

    monkeypatch.setattr(inference, "sample_chunk", fake_chunk)
    output = tmp_path / "output" / "video.mp4"
    segment_dir = tmp_path / "segments"

    result = inference.sample_long_range(
        settings,
        source,
        output,
        num_frames=114,
        checkpoint_path=adapter,
        segment_dir=segment_dir,
        keep_intermediates=True,
        bootstrap_first_chunk_path=bootstrap,
        manifest_dir=segment_dir,
    )

    records = json.loads((segment_dir / "chunk_lineage.json").read_text())["chunks"]
    segment0 = np.load(segment_dir / "segment_000000.npy", allow_pickle=False)
    expected_prefix = (
        torch.from_numpy(segment0[-17:].copy())
        .float()
        .div(255.0)
        .permute(0, 3, 1, 2)
    )
    manifest = json.loads((segment_dir / "inference_manifest.json").read_text())

    assert result == output.resolve()
    assert sampled_starts == [48, 96]
    assert torch.equal(prefix_inputs[0], expected_prefix)
    assert records[0]["new_output_frame_range"] == [0, 65]
    assert records[0]["runtime"]["model_inference_skipped"] is True
    assert records[0]["bootstrap_first_chunk"]["path"] == str(bootstrap.resolve())
    assert manifest["completed_chunks_resumed"] == 0
    assert manifest["bootstrap_first_chunk_artifact"]["path"] == str(
        bootstrap.resolve()
    )
    assert manifest["manifest_directory"] == str(segment_dir.resolve())
    assert not (output.parent / "inference_manifest.json").exists()

    sampled_starts.clear()
    result2 = inference.sample_long_range(
        settings,
        source,
        output,
        num_frames=114,
        checkpoint_path=adapter,
        segment_dir=segment_dir,
        resume=True,
        keep_intermediates=True,
        bootstrap_first_chunk_path=bootstrap,
        manifest_dir=segment_dir,
    )
    assert result2 == result
    assert sampled_starts == []
    assert load_calls == [1]


@pytest.mark.parametrize("mutate", ["source", "adapter", "frames", "resolution", "dtype"])
def test_rejects_wrong_prefix_lineage_or_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: str
) -> None:
    source = tmp_path / "source.mp4"
    _source(source, 17)
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base")
    path, _ = _artifact(tmp_path, source, adapter)
    settings = _settings(adapter, base)
    if mutate == "source":
        with source.open("ab") as handle:
            handle.write(b"changed")
    elif mutate == "adapter":
        adapter.write_bytes(b"different")
    elif mutate == "frames":
        settings["long_range"]["prefix_frames"] = 9
        settings["long_range"]["stride_frames"] = 56
    elif mutate == "resolution":
        settings["video"]["canvas"] = [24, 8]
    else:
        tensor = torch.load(path, weights_only=True)
        torch.save(tensor.half(), path)
        meta_path = path.with_name("seed_33333333_meta.json")
        meta = json.loads(meta_path.read_text())
        meta["tensor_sha256"] = file_sha256(path)
        meta["tensor_byte_size"] = path.stat().st_size
        meta_path.write_text(json.dumps(meta))
    inference._install_official_sources(ROOT.parent / "models/LTX2.3")
    from ltx_trainer import model_loader

    monkeypatch.setattr(
        model_loader,
        "read_video_scale_factors",
        lambda _: SimpleNamespace(time=8, height=8, width=8),
    )
    with pytest.raises((ValueError, TypeError)):
        inference.sample_long_range(
            settings,
            source,
            tmp_path / "output.mp4",
            num_frames=17,
            checkpoint_path=adapter,
            initial_prefix_path=path,
        )
