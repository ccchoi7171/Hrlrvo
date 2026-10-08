import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ltx_outpaint.geometry import CanvasGeometry, Rect, global_cells
from ltx_outpaint.lineart_global_prior import (
    LineArtGlobalPrior,
    build_attention_bias,
    build_fine_write_mask,
    current_prior,
)


def sigmas():
    # Unique values deliberately include a nonuniform schedule.
    return tuple(float(x) for x in torch.linspace(1.0, 0.01, 30))


def context(sigma, anchor=Rect(9, 15, 27, 45), frames=9):
    return SimpleNamespace(
        sigma=torch.tensor(sigma, dtype=torch.float32),
        geometry=CanvasGeometry(frames, 36, 60, (anchor,)),
    )


def prior(scale=0.5, enabled=True):
    strength = torch.arange(135, dtype=torch.float32).reshape(9, 15) / 134
    return LineArtGlobalPrior(strength, sigmas(), enabled=enabled, bias_scale=scale)


def test_schedule_boundaries_and_repeated_cfg_do_not_advance_step():
    p = prior()
    expected = {9: 0.5, 10: 0.3, 19: 0.3, 20: 0.1, 29: 0.1}
    with p.chunk_scope(1):
        for step, value in expected.items():
            active = current_prior(context([sigmas()[step], sigmas()[step]]))
            assert active[2] == step
            assert active[3] == pytest.approx(value)
        # A repeat is another model/block call at the same scheduler step, not
        # an implicit counter increment.
        assert current_prior(context([sigmas()[10], sigmas()[10]]))[2] == 10
        assert p.summary()["calls_by_step"][10] == 2
    assert p.summary()["active_calls"] == 6


def test_unknown_and_mixed_sigma_fail():
    p = prior()
    with p.chunk_scope(1):
        with pytest.raises(ValueError, match="mixed"):
            current_prior(context([sigmas()[3], sigmas()[4]]))
        with pytest.raises(ValueError, match="not in"):
            current_prior(context([100.0, 100.0]))


def test_bias_math_time_major_token_order_and_query_gating():
    p = prior()
    ctx = context([sigmas()[0]], anchor=Rect(1, 1, 35, 59), frames=9)
    cells = global_cells(36, 60, 4)
    with p.chunk_scope(1, prefix_latent_frames=3):
        active = current_prior(ctx)
        bias, eligible = build_attention_bias(p, active[1], ctx, cells, torch.float32, torch.device("cpu"), active[3])
        g = p.token_strength.flatten().repeat(9)
        assert bias.shape == (1, 1215, 1215)
        # Prefix rows are untouched, while their tokens are still full-strength keys.
        assert not eligible[0, : 3 * 135].any()
        assert torch.count_nonzero(bias[0, : 3 * 135]) == 0
        # C4(0,0) overlaps but is not wholly inside the anchor: generated-any query.
        q = 3 * 135 + 1
        assert eligible[0, q]
        assert torch.allclose(bias[0, q], 0.5 * g[q] * g)
        # C4(4,4) is wholly anchor-owned and cannot be a biased query.
        contained = 3 * 135 + 1 * 15 + 1
        assert not eligible[0, contained]
        assert torch.count_nonzero(bias[0, contained]) == 0
        # Exact row-major spatial ordering followed by time-major repetition.
        assert g[0] == p.token_strength[0, 0]
        assert g[14] == p.token_strength[0, 14]
        assert g[15] == p.token_strength[1, 0]
        assert g[135] == p.token_strength[0, 0]


def test_fine_write_mask_preserves_prefix_and_anchor_exactly():
    p = prior()
    ctx = context([sigmas()[0]], anchor=Rect(9, 15, 27, 45), frames=9)
    with p.chunk_scope(1, prefix_latent_frames=3):
        active = current_prior(ctx)
        mask = build_fine_write_mask(active[1], ctx, torch.device("cpu"))
        assert mask.shape == (1, 9, 36, 60, 1)
        assert not mask[:, :3].any()
        assert not mask[:, 3:, 9:27, 15:45].any()
        assert mask[:, 3:, :, :15].all()
        assert mask[:, 3:, :, 45:].all()


class FakeCompressor(nn.Module):
    def forward(self, grid, cells):
        return grid[:, :, ::4, ::4, :]


class LeakyDecompressor(nn.Module):
    def forward(self, tokens, compressed_shape, target_shape):
        b, t, n, d = tokens.shape
        x = tokens.reshape(b, t, 9, 15, d).repeat_interleave(4, 2).repeat_interleave(4, 3)
        # Learned Conv3D-like support: eligible changes bleed one step/cell into
        # otherwise unselected positions unless the final fine mask restores them.
        y = torch.nn.functional.avg_pool3d(
            x.permute(0, 4, 1, 2, 3), kernel_size=3, stride=1, padding=1
        )
        return y.permute(0, 2, 3, 4, 1)


class FakeAttention:
    scale = 4
    heads = 1
    shared_outpainting_lora = {"q": None, "k": None, "v": None}
    global_lora = {"q": None, "k": None, "v": None}

    def __init__(self):
        self.compressors = {"4": FakeCompressor()}
        self.decompressors = {"4": LeakyDecompressor()}
        self.to_q = self.to_k = self.to_v = nn.Identity()
        self.original_calls = 0
        self.masked_calls = 0

    def preattention_function(self, q, k, attention, mask, pe, k_pe):
        assert mask is None and k_pe is None
        return q, k

    def attention_function(self, q, k, v, heads):
        self.original_calls += 1
        return torch.zeros_like(v)

    def masked_attention_function(self, q, k, v, heads, mask):
        self.masked_calls += 1
        assert mask.ndim == 3
        return torch.ones_like(v)


def install_test_shims(monkeypatch):
    module = importlib.import_module("ltx_outpaint.global_attention")
    monkeypatch.setattr(module, "phase_rope", lambda context, cells, scale, dtype: object())
    monkeypatch.setattr(
        module,
        "project_with_loras",
        lambda linear, x, loras: linear(x),
    )
    monkeypatch.setattr(
        module, "attend",
        lambda attention, x, pe, lora: attention.attention_function(x, x, x, attention.heads),
    )
    return module


@pytest.mark.parametrize("mode", ["unbound", "disabled", "scale0", "c0"])
def test_off_routes_exact_original_backend_without_mask(monkeypatch, mode):
    module = install_test_shims(monkeypatch)
    attn = FakeAttention()
    grid = torch.randn(1, 9, 36, 60, 1)
    ctx = context([sigmas()[0]])
    if mode == "unbound":
        got = module.global_attention(attn, grid, ctx)
    else:
        p = prior(enabled=mode != "disabled", scale=0.0 if mode == "scale0" else 0.5)
        with p.chunk_scope(0 if mode == "c0" else 1):
            got = module.global_attention(attn, grid, ctx)
    assert attn.original_calls == 1
    assert attn.masked_calls == 0
    assert torch.count_nonzero(got) == 0


def test_dual_decompression_restores_prefix_and_anchor_after_leakage(monkeypatch):
    module = install_test_shims(monkeypatch)
    attn = FakeAttention()
    grid = torch.zeros(1, 9, 36, 60, 1)
    ctx = context([sigmas()[0]])
    p = prior()
    with p.chunk_scope(1, prefix_latent_frames=3):
        got = module.global_attention(attn, grid, ctx)
        summary = p.summary()
    assert attn.original_calls == 1 and attn.masked_calls == 1
    # Leaky decompressor would contaminate t=2 and anchor borders. Final restore
    # guarantees canonical Global output there anyway.
    assert torch.count_nonzero(got[:, :3]) == 0
    assert torch.count_nonzero(got[:, 3:, 9:27, 15:45]) == 0
    assert got[:, 3:, :, :15].max() > 0
    assert summary["configured_original_backend"] == "method"
    assert summary["configured_biased_backend"] == "method"


def test_zero_line_map_is_exact_unmasked_route(monkeypatch):
    module = install_test_shims(monkeypatch)
    attn = FakeAttention()
    grid = torch.randn(1, 9, 36, 60, 1)
    empty = LineArtGlobalPrior(torch.zeros(9, 15), sigmas(), bias_scale=0.5)
    with empty.chunk_scope(1):
        got = module.global_attention(attn, grid, context([sigmas()[0]]))
    assert attn.original_calls == 1
    assert attn.masked_calls == 0
    assert torch.count_nonzero(got) == 0


def test_sigma_lookup_cached_by_held_context_identity(monkeypatch):
    p = prior()
    ctx = context([sigmas()[7], sigmas()[7]])
    calls = 0
    original = p._step_for_sigma

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(p, "_step_for_sigma", counted)
    with p.chunk_scope(1):
        assert current_prior(ctx)[2] == 7
        assert current_prior(ctx)[2] == 7
        assert calls == 1
        # Equal contents in a different ViewContext must be independently checked.
        assert current_prior(context([sigmas()[7], sigmas()[7]]))[2] == 7
        assert calls == 2
