import pytest
import torch
from ltx_core.model.transformer.feed_forward import FeedForward

from ltx_outpaint.ffn_token_chunking import install_ffn_token_chunking
from ltx_outpaint.lora import install_linear_lora


class VideoBlock(torch.nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.ff = FeedForward(dim, dim)
        self.audio_ff = FeedForward(dim, dim)


class VideoTransformer(torch.nn.Module):
    def __init__(self, count: int = 2) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([VideoBlock() for _ in range(count)])


@pytest.mark.parametrize("tokens,chunk_tokens", [
    (0, 4), (1, 4), (4, 4), (5, 4), (8, 4), (9, 4),
    (17, 8), (73, 16), (100, 37),
])
def test_video_ffn_token_boundaries_match_native_cpu(tokens: int, chunk_tokens: int) -> None:
    torch.manual_seed(123)
    model = VideoTransformer().eval()
    x = torch.randn(2, tokens, 8)
    expected = [block.ff(x) for block in model.transformer_blocks]
    original_audio = model.transformer_blocks[0].audio_ff.forward

    report = install_ffn_token_chunking(model, chunk_tokens)

    assert report == {"video_ffn_blocks": 2, "chunk_tokens": chunk_tokens}
    assert model.transformer_blocks[0].audio_ff.forward == original_audio
    with torch.no_grad():
        for block, baseline in zip(model.transformer_blocks, expected, strict=True):
            actual = block.ff(x)
            assert actual.shape == baseline.shape
            assert actual.dtype == baseline.dtype
            assert torch.allclose(actual, baseline, rtol=1e-6, atol=1e-6)


def test_installed_lora_path_is_preserved_with_final_partial_chunk() -> None:
    torch.manual_seed(987)
    model = VideoTransformer(count=1).eval()
    ff = model.transformer_blocks[0].ff
    ff.requires_grad_(False)
    for linear in (ff.net[0].proj, ff.net[2]):
        install_linear_lora(linear, "ffn_lora", rank=2, alpha=2)
        with torch.no_grad():
            linear.ffn_lora.lora_B.weight.normal_(std=0.03)
    x = torch.randn(2, 19, 8)
    with torch.no_grad():
        expected = ff(x)
    keys_before = tuple(model.state_dict())

    install_ffn_token_chunking(model, chunk_tokens=8)
    with torch.no_grad():
        actual = ff(x)

    assert tuple(model.state_dict()) == keys_before
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_invalid_or_duplicate_install_does_not_modify_other_blocks() -> None:
    model = VideoTransformer().eval()
    for chunk_tokens in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            install_ffn_token_chunking(model, chunk_tokens)
    assert all(not hasattr(block.ff, "_hrlrvo_ffn_chunk_tokens") for block in model.transformer_blocks)

    install_ffn_token_chunking(model, 8)
    with pytest.raises(ValueError, match="already installed"):
        install_ffn_token_chunking(model, 4)


def test_training_model_rejected() -> None:
    model = VideoTransformer()
    with pytest.raises(ValueError, match="inference-only"):
        install_ffn_token_chunking(model, 8)
    assert all(not hasattr(block.ff, "_hrlrvo_ffn_chunk_tokens") for block in model.transformer_blocks)
