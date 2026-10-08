"""CPU-only equivalence checks for the inference-specific VAE output path."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from ltx_outpaint.vae_decode_streaming import (
    decode_video_frames_streaming,
    decode_video_streaming,
)


@dataclass
class _State:
    latent: torch.Tensor


class _Decoder:
    def __init__(self, temporal_lengths: tuple[int, ...]) -> None:
        self.temporal_lengths = temporal_lengths
        self.devices: list[str] = []
        self.tiling_configs: list[object] = []

    def to(self, device: torch.device | str) -> _Decoder:
        self.devices.append(str(device))
        return self

    def tiled_decode(self, latent: torch.Tensor, *, tiling_config: object):
        self.tiling_configs.append(tiling_config)
        position = 0
        for length in self.temporal_lengths:
            yield latent[:, :, position : position + length]
            position += length
        assert position == latent.shape[2]


class _Runner:
    def __init__(self, lengths: tuple[int, ...]) -> None:
        self._vae_decoder = _Decoder(lengths)
        self.tile_plan = object()
        self.planner_inputs: list[tuple[torch.dtype, str]] = []

    def _decode_tiling_config(self, latent: torch.Tensor, device: torch.device):
        self.planner_inputs.append((latent.dtype, str(device)))
        return self.tile_plan


def _legacy(state: _State, lengths: tuple[int, ...]) -> torch.Tensor:
    chunks = list(_Decoder(lengths).tiled_decode(state.latent.bfloat16(), tiling_config=None))
    decoded = torch.cat(chunks, dim=2)
    decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
    return decoded[0].float().cpu()


@pytest.mark.parametrize(
    "lengths",
    [(1,), (16, 49), (24, 24, 17), (8, 8, 1), (65,)],
)
def test_reconstructed_temporal_regions_match_legacy_bitwise(lengths: tuple[int, ...]):
    frames = sum(lengths)
    # Frame identity remains visible after BF16 normalization, including values
    # near the clipping bounds and a non-full final temporal group.
    values = torch.arange(frames * 3 * 2 * 4, dtype=torch.float32).reshape(
        1, 3, frames, 2, 4
    )
    state = _State((values.remainder(511) / 255 - 1.0).bfloat16())
    runner = _Runner(lengths)

    actual = decode_video_streaming(runner, state, torch.device("cpu"))
    expected = _legacy(state, lengths)

    assert actual.shape == expected.shape == (3, frames, 2, 4)
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert runner.planner_inputs == [(torch.bfloat16, "cpu")]
    assert runner._vae_decoder.tiling_configs == [runner.tile_plan]
    assert runner._vae_decoder.devices == ["cpu", "cpu"]


def test_no_completed_region_is_reassembled_on_gpu(monkeypatch: pytest.MonkeyPatch):
    original_cat = torch.cat

    def cpu_only_cat(tensors, *, dim):
        assert all(tensor.device.type == "cpu" for tensor in tensors)
        assert dim == 1
        return original_cat(tensors, dim=dim)

    monkeypatch.setattr(torch, "cat", cpu_only_cat)
    state = _State(torch.arange(7, dtype=torch.bfloat16).reshape(1, 1, 7, 1, 1))
    result = decode_video_streaming(_Runner((3, 4)), state, torch.device("cpu"))
    assert result.shape == (1, 7, 1, 1)


def test_empty_decoder_output_is_rejected():
    state = _State(torch.empty(1, 1, 0, 1, 1, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="no temporal regions"):
        decode_video_streaming(_Runner(()), state, torch.device("cpu"))


def test_selected_frames_match_full_decode_without_materializing_full_output():
    state = _State(
        torch.linspace(-1.0, 1.0, 3 * 65 * 2 * 4, dtype=torch.float32)
        .reshape(1, 3, 65, 2, 4)
        .bfloat16()
    )
    full = _legacy(state, (16, 24, 25))
    metrics = {}
    selected = decode_video_frames_streaming(
        _Runner((16, 24, 25)),
        state,
        torch.device("cpu"),
        (0, 32, 64),
        metrics=metrics,
    )

    assert selected.shape == (3, 3, 2, 4)
    assert torch.equal(selected, full[:, [0, 32, 64]])
    assert metrics == {
        "vae_temporal_frames_traversed": 65,
        "vae_temporal_regions_traversed": 3,
        "vae_frames_retained": [0, 32, 64],
        "vae_random_access_decode": False,
        "full_decoded_video_materialized": False,
    }


@pytest.mark.parametrize("indices", [(), (32, 0), (0, 0), (-1, 0)])
def test_selected_frame_indices_are_strict(indices: tuple[int, ...]):
    state = _State(torch.zeros(1, 3, 65, 1, 1, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        decode_video_frames_streaming(
            _Runner((65,)), state, torch.device("cpu"), indices
        )
