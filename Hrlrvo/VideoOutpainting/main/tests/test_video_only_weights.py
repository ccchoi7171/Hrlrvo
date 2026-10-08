"""CPU-only guards for the inference-only Audio/AV residency selector."""

import json

import pytest
import torch
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel, LTXModelType

from ltx_outpaint.video_only_weights import (
    offload_unused_audio_modules,
    place_video_only_weights_on_gpu,
)


def _model(*, cross_attention_adaln: bool = True) -> LTXModel:
    model = LTXModel(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=16,
        in_channels=4,
        out_channels=4,
        audio_in_channels=4,
        audio_out_channels=4,
        cross_attention_adaln=cross_attention_adaln,
    ).eval()
    # The constructor leaves AdaLN tables uninitialized until checkpoint load.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    return model


@pytest.mark.parametrize("cross_attention_adaln", [False, True])
def test_audio_offload_keeps_video_and_shared_weights_and_strict_state(
    cross_attention_adaln: bool,
) -> None:
    model = _model(cross_attention_adaln=cross_attention_adaln)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    video_weight = model.patchify_proj.weight
    video_block_weight = model.transformer_blocks[0].ff.net[0].proj.weight
    video_head_weight = model.proj_out.weight

    report = offload_unused_audio_modules(model)

    json.dumps(report)
    assert report["parameter_count"] > 0
    assert report["parameter_bytes"] == report["parameter_count"] * 4  # CPU FP32 model
    assert report["gpu_bytes_released"] == 0
    assert len(report["module_names"]) == (8 if cross_attention_adaln else 7) + 5 * model.num_blocks
    assert "transformer_blocks.0.audio_to_video_attn" in report["module_names"]
    assert "transformer_blocks.0.video_to_audio_attn" in report["module_names"]
    assert "transformer_blocks.0.attn2" not in report["module_names"]
    assert "transformer_blocks.0.ff" not in report["module_names"]
    assert "scale_shift_table" not in report["direct_parameter_names"]
    assert model.patchify_proj.weight is video_weight
    assert model.transformer_blocks[0].ff.net[0].proj.weight is video_block_weight
    assert model.proj_out.weight is video_head_weight
    for name, old in before.items():
        assert torch.equal(model.state_dict()[name], old), name

    # No parameter is dropped from the state dict; checkpoint strictness remains possible.
    restored = _model(cross_attention_adaln=cross_attention_adaln)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert tuple(restored.state_dict()) == tuple(before)


def test_shared_parameter_is_rejected_before_any_move() -> None:
    model = _model()
    model.audio_patchify_proj = model.patchify_proj

    with pytest.raises(ValueError, match="shared with the video"):
        offload_unused_audio_modules(model)


def test_video_only_forward_is_bitwise_equal_after_offload() -> None:
    model = _model()
    positions = torch.zeros(1, 3, 3, 2)
    positions[..., 1] = 1
    video = Modality(
        latent=torch.randn(1, 3, 4),
        sigma=torch.tensor([0.5]),
        timesteps=torch.full((1, 3), 0.5),
        positions=positions,
        context=torch.randn(1, 4, 16),
    )
    with torch.no_grad():
        before, _ = model(video, None, None)
        offload_unused_audio_modules(model)
        after, _ = model(video, None, None)
    assert torch.equal(before, after)


def test_rejects_training_and_non_av_model() -> None:
    model = _model()
    model.train()
    with pytest.raises(ValueError, match="inference-only"):
        offload_unused_audio_modules(model)

    model.eval()
    model.model_type = LTXModelType.VideoOnly
    with pytest.raises(TypeError, match="AudioVideo"):
        offload_unused_audio_modules(model)


def test_cpu_first_video_placement_matches_full_gpu_model() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable for the optional placement smoke test")

    torch.manual_seed(789)
    source = _model()
    state = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    device = torch.device("cuda")
    reference = _model().to(device=device, dtype=torch.bfloat16).eval()
    reference.load_state_dict(state, strict=True)
    optimized = _model().to(dtype=torch.bfloat16).eval()
    optimized.load_state_dict(state, strict=True)

    report = place_video_only_weights_on_gpu(optimized, device, torch.bfloat16)
    positions = torch.zeros(1, 3, 3, 2, device=device)
    positions[..., 1] = 1
    video = Modality(
        latent=torch.randn(1, 3, 4, device=device, dtype=torch.bfloat16),
        sigma=torch.tensor([0.5], device=device),
        timesteps=torch.full((1, 3), 0.5, device=device),
        positions=positions,
        context=torch.randn(1, 4, 16, device=device, dtype=torch.bfloat16),
    )
    with torch.inference_mode():
        expected, _ = reference(video, None, None)
        actual, _ = optimized(video, None, None)

    assert report["gpu_bytes_avoided"] > 0
    assert torch.equal(actual, expected)
    assert optimized.audio_patchify_proj.weight.device.type == "cpu"
    assert optimized.transformer_blocks[0].audio_ff.net[0].proj.weight.device.type == "cpu"
    assert optimized.patchify_proj.weight.device.type == "cuda"
    assert optimized.transformer_blocks[0].ff.net[0].proj.weight.device.type == "cuda"
