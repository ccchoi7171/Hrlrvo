"""The only full-canvas peer of the Local System; no W_O in this view."""

import torch

from .geometry import global_cells
from .lineart_global_prior import (
    build_attention_bias,
    build_fine_write_mask,
    current_prior,
    record_backends,
)
from .lora import project_with_loras
from .native_local_attention import attend
from .position import phase_rope


def global_attention(attention, grid, context):
    b, t, h, w, d = grid.shape
    s = attention.scale
    cells = global_cells(h, w, s)
    compressed = attention.compressors[str(s)](grid, cells)
    pe = phase_rope(context, cells, s, grid.dtype)
    active = current_prior(context)
    if active is None:
        # Exact pre-experiment route: no masked backend, duplicate decompression,
        # masks, QKV projection changes, or floating-point selection.
        output = attend(attention, compressed.reshape(b, -1, d), pe, attention.global_lora)
        return attention.decompressors[str(s)](
            output.reshape(b, t, len(cells), d), (h // s, w // s), (h, w))

    prior, state, _step, strength = active
    x = compressed.reshape(b, -1, d)
    shared = attention.shared_outpainting_lora
    q, k, v = (
        project_with_loras(
            getattr(attention, f"to_{name}"), x, (shared[name], attention.global_lora[name])
        )
        for name in ("q", "k", "v")
    )
    q, k = attention.preattention_function(q, k, attention, None, pe, None)

    # Preserve the configured fast/unmasked backend for the canonical result.
    original_tokens = attention.attention_function(q, k, v, attention.heads)
    bias, eligible_rows = build_attention_bias(
        prior, state, context, cells, q.dtype, q.device, strength
    )
    biased_tokens = attention.masked_attention_function(q, k, v, attention.heads, bias)
    record_backends(state, attention.attention_function, attention.masked_attention_function)

    # Select only eligible compressed query rows before learned decompression.
    selected_tokens = torch.where(eligible_rows[..., None], biased_tokens, original_tokens)
    decompressor = attention.decompressors[str(s)]
    shape = (b, t, len(cells), d)
    original_full = decompressor(
        original_tokens.reshape(shape), (h // s, w // s), (h, w)
    )
    selected_full = decompressor(
        selected_tokens.reshape(shape), (h // s, w // s), (h, w)
    )
    # Conv3D refinement can leak selected token changes across time/space. Restore
    # the canonical Global output at every prefix/anchor fine position afterward.
    fine_write = build_fine_write_mask(state, context, grid.device)
    return torch.where(fine_write, selected_full, original_full)
