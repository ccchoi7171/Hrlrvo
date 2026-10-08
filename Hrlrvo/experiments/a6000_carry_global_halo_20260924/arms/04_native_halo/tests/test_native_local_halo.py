from types import SimpleNamespace

import torch
from torch import nn

from ltx_outpaint.geometry import Rect
from ltx_outpaint.native_local_attention import native_local, select_rope


class _Zero(nn.Module):
    def forward(self, value):
        return torch.zeros_like(value)


class _Attention:
    def __init__(self, radius=1):
        self.native_local_halo_radius = radius
        self.rope_type = SimpleNamespace(value="split")
        self.to_q = nn.Identity()
        self.to_k = nn.Identity()
        self.to_v = nn.Identity()
        self.shared_outpainting_lora = {key: _Zero() for key in ("q", "k", "v")}
        self.heads = 1
        self.pre_calls = []
        self.attention_calls = []

    def preattention_function(self, q, k, attention, mask, pe, k_pe):
        self.pre_calls.append((q.clone(), k.clone(), pe, k_pe))
        return q, k

    def attention_function(self, q, k, v, heads):
        self.attention_calls.append((q.clone(), k.clone(), v.clone()))
        # Preserve query length while making the result depend on the read set.
        return q + k.mean(dim=1, keepdim=True) + v.mean(dim=1, keepdim=True)


def _context(t, h, w):
    positions = torch.arange(t * h * w, dtype=torch.float32).reshape(1, 1, -1, 1)
    return SimpleNamespace(native_pe=(positions,))


def _legacy_native_local(attention, grid, window, context):
    """The exact pre-halo implementation, retained as a regression oracle."""
    b, t, h, w, d = grid.shape
    indices = (torch.arange(t)[:, None, None] * h * w
               + torch.arange(window.y0, window.y1)[None, :, None] * w
               + torch.arange(window.x0, window.x1)[None, None, :]).flatten()
    x = grid[:, :, window.y0:window.y1, window.x0:window.x1].reshape(b, -1, d)
    pe = select_rope(context.native_pe, indices, attention.rope_type)
    shared = attention.shared_outpainting_lora
    q = attention.to_q(x) + shared["q"](x)
    k = attention.to_k(x) + shared["k"](x)
    v = attention.to_v(x) + shared["v"](x)
    q, k = attention.preattention_function(q, k, attention, None, pe, None)
    output = attention.attention_function(q, k, v, attention.heads)
    return output.reshape(b, t, window.height, window.width, d)


def test_default_halo_keeps_query_and_output_shape_but_expands_kv():
    grid = torch.arange(1 * 2 * 6 * 8 * 1, dtype=torch.float32).reshape(1, 2, 6, 8, 1)
    window = Rect(2, 3, 4, 6)
    attention = _Attention()

    output = native_local(attention, grid, window, _context(2, 6, 8))

    q, k, v = attention.attention_calls[-1]
    assert output.shape == (1, 2, 2, 3, 1)
    assert q.shape[1] == 2 * 2 * 3
    # radius 1: y=[1,5), x=[2,7), over both temporal positions.
    assert k.shape[1] == v.shape[1] == 2 * 4 * 5


def test_clipped_halo_uses_canonical_temporal_indices_without_wrap_or_duplicates():
    t, h, w = 2, 5, 7
    grid = torch.arange(t * h * w, dtype=torch.float32).reshape(1, t, h, w, 1)
    window = Rect(0, 0, 2, 2)
    attention = _Attention()

    native_local(attention, grid, window, _context(t, h, w))

    _, k, q_pe, k_pe = attention.pre_calls[-1]
    q_indices = q_pe[0].flatten().to(torch.long).tolist()
    support_indices = k_pe[0].flatten().to(torch.long).tolist()
    expected_q = [0, 1, 7, 8, 35, 36, 42, 43]
    expected_support = [
        y * w + x + frame * h * w
        for frame in range(t)
        for y in range(3)
        for x in range(3)
    ]
    assert q_indices == expected_q
    assert support_indices == expected_support
    assert len(support_indices) == len(set(support_indices))
    assert k.flatten().to(torch.long).tolist() == expected_support
    # Clipping at x=0 must not wrap around to the right edge.
    assert all((index % w) < 3 for index in support_indices)


def test_halo_can_read_anchor_strip_outside_core_but_never_expands_output():
    grid = torch.zeros(1, 1, 6, 8, 1)
    window = Rect(2, 4, 4, 6)
    grid[0, 0, 3, 3, 0] = 99  # Simulated anchor token in the left halo.
    attention = _Attention()

    output = native_local(attention, grid, window, _context(1, 6, 8))

    q, k, v = attention.attention_calls[-1]
    assert 99 not in q
    assert 99 in k and 99 in v
    assert output.shape == (1, 1, 2, 2, 1)
    assert torch.count_nonzero(output) == output.numel()


def test_radius_zero_is_bit_exact_with_pre_halo_path():
    torch.manual_seed(7)
    grid = torch.randn(2, 3, 5, 7, 4)
    window = Rect(1, 2, 4, 6)
    context = _context(3, 5, 7)
    expected_attention = _Attention(radius=0)
    actual_attention = _Attention(radius=0)

    expected = _legacy_native_local(expected_attention, grid, window, context)
    actual = native_local(actual_attention, grid, window, context)

    assert torch.equal(actual, expected)
    assert actual_attention.pre_calls[-1][-1] is None


def test_invalid_halo_radius_is_rejected():
    grid = torch.zeros(1, 1, 3, 3, 1)
    with torch.no_grad():
        for radius in (-1, 1.5, True):
            attention = _Attention(radius=radius)
            try:
                native_local(attention, grid, Rect(0, 0, 2, 2), _context(1, 3, 3))
            except ValueError as error:
                assert "non-negative integer" in str(error)
            else:
                raise AssertionError(f"radius {radius!r} was accepted")
