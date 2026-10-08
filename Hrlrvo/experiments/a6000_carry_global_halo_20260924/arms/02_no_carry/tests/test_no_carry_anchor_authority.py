from __future__ import annotations

from types import SimpleNamespace

import torch

from ltx_outpaint import inference
from ltx_outpaint.geometry import Rect


class _LatentShape:
    batch = 1
    channels = 2
    frames = 3
    height = 3
    width = 3

    @staticmethod
    def to_torch_shape() -> tuple[int, int, int, int, int]:
        return (1, 2, 3, 3, 3)


def test_rgb_vae_prefix_reuses_current_authoritative_anchor(monkeypatch) -> None:
    """No-carry changes only the generated-prefix source, not anchor authority."""
    scale_factors = SimpleNamespace(time=8, height=32, width=32)
    boundary = inference._vae_boundary(96, 96, 32, 32)
    anchor = torch.empty(17, 3, 32, 32)
    previous_prefix = torch.empty(17, 3, 96, 96)
    anchor_latent = torch.full((1, 2, 3, 1, 1), 7.0)
    rgb_prefix_latent = torch.full((1, 2, 3, 3, 3), 3.0)
    calls: list[str] = []

    monkeypatch.setattr(inference, "_reflect_pad_rgb", lambda frames, **_: frames)

    def fake_encode(_frames, *_args, training_aligned_anchor=False, **_kwargs):
        calls.append("anchor" if training_aligned_anchor else "rgb_prefix")
        return anchor_latent if training_aligned_anchor else rgb_prefix_latent

    monkeypatch.setattr(inference, "_encode_condition_rgb", fake_encode)
    condition, rect = inference._condition_latents(
        anchor,
        previous_prefix,
        boundary,
        (32, 32, 64, 64),
        _LatentShape(),
        scale_factors,
        None,
        None,
        torch.device("cpu"),
        torch.float32,
        prefix_latent_override=None,
        generated_prefix_strength=1.0,
    )

    assert calls == ["anchor", "rgb_prefix"]
    assert rect == Rect(1, 1, 2, 2)
    assert torch.equal(condition[:, :, :, 1:2, 1:2], anchor_latent)
    generated_region = torch.ones_like(condition, dtype=torch.bool)
    generated_region[:, :, :, 1:2, 1:2] = False
    assert torch.equal(
        condition[generated_region], rgb_prefix_latent[generated_region]
    )


def test_no_carry_path_never_requests_causal_carry_keyframe(monkeypatch) -> None:
    scale_factors = SimpleNamespace(time=8, height=32, width=32)
    boundary = inference._vae_boundary(96, 96, 32, 32)
    encoded_frame_counts: list[int] = []

    monkeypatch.setattr(inference, "_reflect_pad_rgb", lambda frames, **_: frames)

    def fake_encode(frames, *_args, training_aligned_anchor=False, **_kwargs):
        encoded_frame_counts.append(int(frames.shape[0]))
        spatial = 1 if training_aligned_anchor else 3
        return torch.zeros(1, 2, 3, spatial, spatial)

    monkeypatch.setattr(inference, "_encode_condition_rgb", fake_encode)
    inference._condition_latents(
        torch.empty(17, 3, 32, 32),
        torch.empty(17, 3, 96, 96),
        boundary,
        (32, 32, 64, 64),
        _LatentShape(),
        scale_factors,
        None,
        None,
        torch.device("cpu"),
        torch.float32,
        prefix_latent_override=None,
    )

    assert encoded_frame_counts == [17, 17]
