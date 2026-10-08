"""CPU-only prefix-search orchestration tests; production model is never loaded."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Self

import pytest
import torch

from ltx_outpaint import inference, prefix_io, prefix_modes, streaming_video


@pytest.mark.parametrize(
    ("base", "explicit", "count", "expected"),
    [
        (13, None, None, [13]),
        (13, None, 3, [13, 14, 15]),
        (13, [7, 2], None, [7, 2]),
        (13, [7, 2], 2, [7, 2]),
    ],
)
def test_seed_resolution(base, explicit, count, expected):
    assert prefix_modes.resolve_prefix_seeds(
        base_seed=base, prefix_seeds=explicit, num_prefix_samples=count
    ) == expected


@pytest.mark.parametrize(
    ("base", "explicit", "count"),
    [(-1, None, None), (True, None, None), (0, [], None), (0, [1, 1], None),
     (0, [1], 2), (0, [False], None), (0, None, 0)],
)
def test_invalid_seed_requests_fail_before_model_load(base, explicit, count):
    with pytest.raises(ValueError):
        prefix_modes.resolve_prefix_seeds(
            base_seed=base, prefix_seeds=explicit, num_prefix_samples=count
        )


def _settings(tmp_path: Path) -> dict[str, Any]:
    return {
        "ltx_repo": str(tmp_path / "ltx"),
        "model": {
            "base_model_path": str(tmp_path / "base.safetensors"),
            "adapter_path": str(tmp_path / "checkpoint-007500"),
        },
        "video": {
            "fps": 24.0,
            "canvas": [8, 4],
            "expected_anchor": [4, 2],
            "enforce_expected_anchor": True,
            "boundary_blend_width_pixels": 1,
        },
        "long_range": {
            "clip_frames": 65,
            "prefix_frames": 17,
            "stride_frames": 48,
            "final_chunk_policy": "repeat_last",
        },
        "sampling": {"seed": 33333333, "inference_steps": 30},
        "runtime": {"device": "cpu", "dtype": "bfloat16"},
        "memory_optimization": {
            "video_only_gpu_weights": True,
            "vae_decode_streaming": True,
            "ffn_token_chunking": False,
            "ffn_chunk_tokens": 8192,
        },
    }


@pytest.mark.parametrize("runtime_device", ["cpu", "cuda"])
def test_prefix_search_uses_first_17_source_frames_and_one_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime_device: str
) -> None:
    settings = _settings(tmp_path)
    settings["runtime"]["device"] = runtime_device
    reset_calls: list[torch.device] = []
    if runtime_device == "cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_calls.append)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 2 * 1024**3)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 3 * 1024**3)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    config = tmp_path / "inference.yaml"
    config.write_text("test: true")
    checkpoint = tmp_path / "checkpoint-007500"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")

    monkeypatch.setattr(inference, "_install_official_sources", lambda _: None)
    fake_loader = ModuleType("ltx_trainer.model_loader")
    fake_loader.read_video_scale_factors = lambda _: SimpleNamespace(time=8, height=8, width=8)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ltx_trainer.model_loader", fake_loader)
    monkeypatch.setattr(
        streaming_video,
        "probe_video",
        lambda _: streaming_video.VideoMetadata(32, 24.0, 4, 2),
    )
    monkeypatch.setattr(streaming_video, "file_sha256", lambda _: "fake-sha")

    read_requests: list[dict[str, Any]] = []

    class FakeReader:
        def __init__(self, path: Path, *, stride_frames: int) -> None:
            assert path == video.resolve()
            assert stride_frames == 48

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def read_chunk(self, record: dict[str, Any]) -> torch.Tensor:
            read_requests.append(record)
            return torch.zeros((17, 3, 2, 4), dtype=torch.float32)

    monkeypatch.setattr(streaming_video, "StreamingAnchorReader", FakeReader)
    monkeypatch.setattr(
        inference,
        "_make_validation_config",
        lambda *args, **kwargs: SimpleNamespace(seed=0, inference_steps=30),
    )
    loaded: list[int] = []
    runner = SimpleNamespace(_config=None)

    def fake_load_models(_settings, validation_config, _boundary, **kwargs):
        loaded.append(validation_config.seed)
        runner._config = validation_config
        assert kwargs["memory_optimization"]["vae_decode_streaming"] is True
        return object(), runner, object(), {}

    monkeypatch.setattr(inference, "_load_models", fake_load_models)
    sampled: list[tuple[int, int, bool, torch.Size]] = []

    def fake_sample_chunk(**kwargs) -> torch.Tensor:
        sampled.append(
            (kwargs["runner"]._config.seed, kwargs["absolute_start_frame"],
             kwargs["previous_prefix"] is None, kwargs["anchor_frames"].shape)
        )
        assert kwargs["vae_decode_streaming"] is True
        assert kwargs["boundary_blend_width_pixels"] == 1
        return torch.zeros((17, 3, 4, 8), dtype=torch.float32)

    monkeypatch.setattr(inference, "sample_chunk", fake_sample_chunk)
    written: list[tuple[int, dict[str, Any]]] = []

    def fake_write(prefix, output_dir, *, seed, fps, metadata):
        assert prefix.dtype == torch.float32
        assert prefix.shape == (17, 3, 4, 8)
        assert fps == 24
        written.append((seed, metadata))
        return (
            output_dir / f"seed_{seed}_prefix.pt",
            output_dir / f"seed_{seed}_preview.mp4",
            output_dir / f"seed_{seed}_meta.json",
            {**metadata, "tensor_sha256": "output-sha"},
        )

    monkeypatch.setattr(prefix_io, "write_prefix_candidate", fake_write)
    results = prefix_modes.sample_prefix_seed_search(
        settings, video, tmp_path / "out", config,
        prefix_seeds=[33333333, 33333334],
    )
    assert read_requests == [{"absolute_start_frame": 0, "real_input_frame_count": 17}]
    assert loaded == [33333333]
    assert sampled == [
        (33333333, 0, True, torch.Size((17, 3, 2, 4))),
        (33333334, 0, True, torch.Size((17, 3, 2, 4))),
    ]
    assert [item["seed"] for item in results] == [33333333, 33333334]
    assert all(item["tensor_sha256"] == "output-sha" for item in results)
    for seed, metadata in written:
        assert seed in (33333333, 33333334)
        assert metadata["source_path"] == str(video.resolve())
        assert metadata["source_sha256"] == "fake-sha"
        assert metadata["checkpoint_path"] == str(checkpoint.resolve())
        assert metadata["adapter_sha256"] == "fake-sha"
        assert metadata["config_path"] == str(config.resolve())
        assert metadata["source_frame_range"] == [0, 17]
        assert metadata["source_frame_range_inclusive"] == [0, 16]
        assert metadata["memory_optimization"]["video_only_gpu_weights"] is True
        assert metadata["memory_optimization"]["vae_decode_streaming"] is True
        assert metadata["memory_optimization"]["ffn_token_chunking"] is False
        assert metadata["generation_timestamp"]
        assert metadata["sample_runtime_seconds"] >= 0
        if runtime_device == "cuda":
            assert metadata["torch_max_allocated_gib"] == 2
            assert metadata["torch_max_reserved_gib"] == 3
        else:
            assert "torch_max_allocated_gib" not in metadata
    assert reset_calls == ([torch.device("cuda")] * 2 if runtime_device == "cuda" else [])


def test_prefix_search_rejects_short_source_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(inference, "_install_official_sources", lambda _: None)
    fake_loader = ModuleType("ltx_trainer.model_loader")
    fake_loader.read_video_scale_factors = lambda _: SimpleNamespace(time=8, height=8, width=8)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ltx_trainer.model_loader", fake_loader)
    monkeypatch.setattr(
        streaming_video,
        "probe_video",
        lambda _: streaming_video.VideoMetadata(16, 24.0, 4, 2),
    )
    with pytest.raises(ValueError, match="requires 17"):
        prefix_modes.sample_prefix_seed_search(
            settings, source, tmp_path / "out", tmp_path / "config.yaml",
        )
