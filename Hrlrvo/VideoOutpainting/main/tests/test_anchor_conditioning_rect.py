from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape
from ltx_trainer.training_strategies.flexible import (
    FlexibleStrategyConfig,
    ModalityConfig,
)
from ltx_trainer.validation_runner import ValidationRunner

from ltx_outpaint import inference
from ltx_outpaint.geometry import Rect, adaptive_layout, local_windows
from ltx_outpaint.train import OutpaintingStrategy, _latent_anchor_rect

RECORD_ID = "0002e965-b236-4ec5-846d-4d1b919cf3f0.pt"
GT_PATH = Path("/home/work/ai-graphics_33/Hrlrvo/cache_Data/train/latents") / RECORD_ID
ANCHOR_PATH = (
    Path("/home/work/ai-graphics_33/Hrlrvo/cache_Data_Anchor_2K/train/latents")
    / RECORD_ID
)


class _FixedTimestepSampler:
    def sample_for(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (latents.shape[0],),
            0.5,
            dtype=latents.dtype,
            device=latents.device,
        )


def _cached_training_batch() -> tuple[dict, dict]:
    gt = torch.load(GT_PATH, map_location="cpu", weights_only=True)
    anchor = torch.load(ANCHOR_PATH, map_location="cpu", weights_only=True)
    provenance = anchor["provenance"]
    anchor_latents = anchor["latents"]
    target = gt["latents"][:, : provenance["latent_frame_count"]].unsqueeze(0)
    condition = torch.zeros_like(target)
    y0, x0, y1, x1 = provenance["canonical_latent_rect_yxyx"]
    condition[0, :, :, y0:y1, x0:x1] = anchor_latents
    metadata = {
        "num_frames": torch.tensor([target.shape[2]]),
        "height": torch.tensor([target.shape[3]]),
        "width": torch.tensor([target.shape[4]]),
        "fps": torch.tensor([float(gt["fps"])]),
    }
    batch = {
        "video_latents": {"latents": target, **metadata},
        "anchor_latents": {"latents": condition, **metadata},
        "anchor_provenance": [provenance],
        "conditions": {
            "video_prompt_embeds": torch.zeros(1, 1, 1),
            "prompt_attention_mask": torch.ones(1, 1, dtype=torch.bool),
        },
    }
    return batch, anchor


def test_latent_anchor_rect_uses_cached_anchor_shape_for_extent() -> None:
    _, anchor = _cached_training_batch()
    provenance = anchor["provenance"]
    rect = _latent_anchor_rect(provenance, height=68, width=120)
    anchor_height, anchor_width = anchor["latents"].shape[-2:]

    assert tuple(anchor["latents"].shape[-2:]) == (34, 60)
    assert rect == (17, 30, 51, 90)
    assert rect[2] - rect[0] == anchor_height
    assert rect[3] - rect[1] == anchor_width


def test_cached_anchor_placement_known_timestep_and_loss_masks_match() -> None:
    batch, anchor = _cached_training_batch()
    provenance = anchor["provenance"]
    config = FlexibleStrategyConfig(
        video=ModalityConfig(is_generated=True, latents_dir="latents", conditions=[]),
    )
    strategy = OutpaintingStrategy(
        config,
        {
            "anchor_rect_pixels": provenance["anchor_rect_canonical_yxyx"],
            "prefix_frames": 17,
            "prefix_probability": 0.0,
        },
    )
    strategy.video_scale_factors = SimpleNamespace(time=8, height=32, width=32)
    anchor_tokens = strategy._patchify_latent_data(batch["anchor_latents"], "video").latents
    inputs = strategy.prepare_training_inputs(batch, _FixedTimestepSampler())

    y0, x0, y1, x1 = (17, 30, 51, 90)
    expected = torch.zeros(1, 9, 68, 120, dtype=torch.bool)
    expected[:, :, y0:y1, x0:x1] = True
    expected = expected.flatten(1)
    known = ~inputs.video_loss_mask

    assert torch.equal(known, expected)
    assert int(known.sum()) == 9 * 34 * 60
    assert int(torch.logical_xor(known, expected).sum()) == 0
    assert bool(torch.all(inputs.video.timesteps[known] == 0))
    assert bool(torch.all(inputs.video.timesteps[~known] == 0.5))
    assert bool(torch.all(~inputs.video_loss_mask[known]))
    assert bool(torch.all(inputs.video_loss_mask[~known]))
    assert torch.equal(inputs.video.latent[known], anchor_tokens[known])
    assert inputs.video.geometry.anchors[0].height == anchor["latents"].shape[-2]
    assert inputs.video.geometry.anchors[0].width == anchor["latents"].shape[-1]
    assert bool(known.reshape(1, 9, 68, 120)[:, :, 50, 30:90].all())


@pytest.mark.parametrize("prefix_frames", [0, 17])
def test_inference_cached_anchor_mask_geometry_and_real_noiser_match_training(
    monkeypatch: pytest.MonkeyPatch, prefix_frames: int,
) -> None:
    """Run production chunk preparation and native noising, stopping before DiT.

    Only RGB padding/encoding are shape-only stubs returning the real cache.
    No VAE, Transformer, decode, backward or CUDA execution is performed.
    """
    batch, cached = _cached_training_batch()
    provenance = cached["provenance"]
    sf = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    shape = VideoLatentShape(1, 128, 9, 68, 120)
    boundary = inference._vae_boundary(2160, 3840, sf.height, sf.width)
    pixel_rect = inference._center_anchor(boundary, 1080, 1920)
    expected_rect = Rect(17, 30, 51, 90)
    condition = batch["anchor_latents"]["latents"].clone()
    if prefix_frames:
        condition[:, :, :3] = batch["video_latents"]["latents"][:, :, :3]

    strategy = OutpaintingStrategy(
        FlexibleStrategyConfig(video=ModalityConfig(is_generated=True, latents_dir="latents", conditions=[])),
        {"anchor_rect_pixels": list(pixel_rect), "prefix_frames": 17,
         "prefix_probability": float(bool(prefix_frames))},
    )
    strategy.video_scale_factors = sf
    training = strategy.prepare_training_inputs(batch, _FixedTimestepSampler())
    expected_known = ~training.video_loss_mask
    patchifier = VideoLatentPatchifier(1)
    expected_tokens = patchifier.patchify(condition)

    pad_calls = []

    def pad(frames, **padding):
        pad_calls.append(padding)
        return torch.empty(
            (1, 3, frames.shape[0], frames.shape[-2] + padding["top"] + padding["bottom"],
             frames.shape[-1] + padding["left"] + padding["right"]), device="meta",
        )

    def encode(frames, *_args, **_kwargs):
        if frames.shape[2] == 65:
            assert tuple(frames.shape) == (1, 3, 65, 1088, 1920)
            return cached["latents"].unsqueeze(0)
        assert prefix_frames == 17 and tuple(frames.shape) == (1, 3, 17, 2176, 3840)
        return batch["video_latents"]["latents"][:, :, :3]

    monkeypatch.setattr(inference, "_reflect_pad_rgb", pad)
    monkeypatch.setattr(inference, "_encode_condition_rgb", encode)
    original_condition = inference._condition_latents
    original_mask = inference._known_mask
    original_geometry = inference._canvas_geometry
    rect_sources = {}

    def capture_condition(*args):
        result, rect = original_condition(*args)
        rect_sources["placement"] = rect
        assert rect == expected_rect
        assert torch.equal(result, condition)
        return result, rect

    def capture_mask(**kwargs):
        assert kwargs["anchor_rect"] is rect_sources["placement"]
        rect_sources["mask"] = kwargs["anchor_rect"]
        return original_mask(**kwargs)

    def capture_geometry(latent_shape, rect):
        assert rect is rect_sources["placement"]
        result = original_geometry(latent_shape, rect)
        assert result.anchors[0] is rect
        rect_sources["geometry"] = rect
        return result

    monkeypatch.setattr(inference, "_condition_latents", capture_condition)
    monkeypatch.setattr(inference, "_known_mask", capture_mask)
    monkeypatch.setattr(inference, "_canvas_geometry", capture_geometry)
    runner = SimpleNamespace(
        _video_scale_factors=sf,
        _config=SimpleNamespace(frame_rate=24, seed=42, negative_prompt="", samples=[SimpleNamespace(prompt="")]),
        _cached_embeddings=[None],
        _get_prompt_embeddings=lambda *_: (torch.zeros(1, 1, 1), None, torch.zeros(1, 1, 1), None),
    )

    class NoDenoiserExecution(Exception):
        pass

    def inspect_noised_state(**kwargs):
        state, clean = kwargs["video_state"], kwargs["video_clean"]
        actual_known = state.denoise_mask.squeeze(-1) == 0
        missing = int((expected_known & ~actual_known).sum())
        noisy = patchifier.unpatchify(state.latent, shape)
        row_error = float((noisy[:, :, :, 50, 30:90] - condition[:, :, :, 50, 30:90]).abs().max())
        print({"prefix_frames": prefix_frames, "missing_known_tokens": missing,
               "last_anchor_row_noiser_max_error": row_error,
               "inference_geometry": vars(runner.geometry.anchors[0])})
        assert missing == 0
        assert torch.equal(actual_known, expected_known)
        assert runner.geometry.anchors == training.video.geometry.anchors == (expected_rect,)
        assert torch.equal(state.latent[actual_known], expected_tokens[actual_known])
        assert torch.equal(clean.clean_latent[actual_known], expected_tokens[actual_known])
        assert row_error == 0.0
        assert not torch.equal(state.latent[~actual_known], expected_tokens[~actual_known])
        assert bool((state.denoise_mask.squeeze(-1)[~actual_known] == 1).all())
        modality = ValidationRunner._modality_from_latent_state(state, torch.zeros(1, 1, 1), torch.tensor(0.5))
        assert bool((modality.timesteps[actual_known] == 0).all())
        assert bool((modality.timesteps[~actual_known] == 0.5).all())
        assert bool((training.video.timesteps[actual_known] == 0).all())
        assert torch.equal(training.video.latent[actual_known], expected_tokens[actual_known])
        assert torch.isclose(strategy.compute_loss(training.video_targets.float() + 1, None, training), torch.tensor(1.0))
        assert kwargs["audio_state"] is None and kwargs["audio_clean"] is None
        for k in (4, 5):
            for target in local_windows(68, 120, k):
                assert adaptive_layout(runner.geometry.anchors[0], target, 68, 120, 4) == adaptive_layout(
                    training.video.geometry.anchors[0], target, 68, 120, 4,
                )
        raise NoDenoiserExecution

    runner._run_denoising = inspect_noised_state
    with pytest.raises(NoDenoiserExecution):
        inference.sample_chunk(
            transformer=torch.nn.Linear(1, 1, dtype=torch.bfloat16), runner=runner, vae_encoder=None,
            anchor_frames=torch.empty((65, 3, 1080, 1920), device="meta"),
            previous_prefix=torch.empty((17, 3, 2160, 3840), device="meta") if prefix_frames else None,
            boundary=boundary, anchor_rect=pixel_rect, absolute_start_frame=0,
        )
    assert pad_calls[0] == {"top": 4, "bottom": 4, "left": 0, "right": 0}
    assert list(pixel_rect) == provenance["anchor_rect_canonical_yxyx"]
    assert set(rect_sources) == {"placement", "mask", "geometry"}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("pixel_rect", [(0, 0, 32, 64), (33, 67, 68, 118), (77, 92, 148, 185)])
def test_encoded_extent_is_not_a_hardcoded_center_or_floor_end(monkeypatch, pixel_rect):
    sf = SpatioTemporalScaleFactors(8, 32, 32)
    boundary = inference._vae_boundary(160, 192, 32, 32)
    shape = VideoLatentShape(1, 2, 3, 5, 6)
    y0, x0, y1, x1 = pixel_rect
    h, w = y1 - y0, x1 - x0
    encoded = torch.randn(1, 2, 3, (h + y0 % 32 + 31) // 32, (w + x0 % 32 + 31) // 32)
    monkeypatch.setattr(inference, "_reflect_pad_rgb", lambda frames, **_: frames)
    monkeypatch.setattr(inference, "_encode_condition_rgb", lambda *_, **__: encoded)
    condition, rect = inference._condition_latents(
        torch.empty(17, 3, h, w, device="meta"), None, boundary, pixel_rect,
        shape, sf, None, None, torch.device("cpu"), torch.float32,
    )
    assert rect == Rect(y0 // 32, x0 // 32, y0 // 32 + encoded.shape[-2], x0 // 32 + encoded.shape[-1])
    assert torch.equal(condition[:, :, :, rect.y0:rect.y1, rect.x0:rect.x1], encoded)
    geometry = inference._canvas_geometry(shape, rect)
    mask = inference._known_mask(
        batch=1, latent_frames=3, latent_height=5, latent_width=6,
        anchor_rect=rect, prefix_frames=0, temporal_factor=8, device=torch.device("cpu"),
    )
    assert geometry.anchors[0] is rect
    assert int(mask.sum()) == 3 * encoded.shape[-2] * encoded.shape[-1]


def test_invalid_encoder_extent_and_outside_mask_fail_closed(monkeypatch):
    sf = SpatioTemporalScaleFactors(8, 32, 32)
    shape = VideoLatentShape(1, 2, 3, 5, 6)
    monkeypatch.setattr(inference, "_reflect_pad_rgb", lambda frames, **_: frames)
    monkeypatch.setattr(inference, "_encode_condition_rgb", lambda *_, **__: torch.zeros(1, 2, 3, 1, 1))
    with pytest.raises(RuntimeError, match="does not map to canonical"):
        inference._condition_latents(
            torch.empty(17, 3, 35, 51, device="meta"), None,
            inference._vae_boundary(160, 192, 32, 32), (33, 67, 68, 118),
            shape, sf, None, None, torch.device("cpu"), torch.float32,
        )
    outside = Rect(4, 5, 6, 7)
    with pytest.raises(ValueError, match="outside"):
        inference._known_mask(
            batch=1, latent_frames=3, latent_height=5, latent_width=6,
            anchor_rect=outside, prefix_frames=0, temporal_factor=8, device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="outside"):
        inference._canvas_geometry(shape, outside)
