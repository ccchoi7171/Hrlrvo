"""Native Local reads exactly one canonical window; it never applies W_O."""

import torch

from .lora import project_with_loras


def select_rope(pe, indices, rope_type):
    axis = 2 if rope_type.value == "split" else 1
    return tuple(part.index_select(axis, indices) for part in pe)


def attend(attention, x, pe, lora=None):
    shared = attention.shared_outpainting_lora
    q, k, v = (
        project_with_loras(getattr(attention, f"to_{key}"), x,
                           (shared[key],) if lora is None else (shared[key], lora[key]))
        for key in ("q", "k", "v")
    )
    q, k = attention.preattention_function(q, k, attention, None, pe, None)
    return attention.attention_function(q, k, v, attention.heads)


def native_local(attention, grid, window, context):
    b, t, h, w, d = grid.shape
    indices = (torch.arange(t, device=grid.device)[:, None, None] * h * w
               + torch.arange(window.y0, window.y1, device=grid.device)[None, :, None] * w
               + torch.arange(window.x0, window.x1, device=grid.device)[None, None, :]).flatten()
    x = grid[:, :, window.y0:window.y1, window.x0:window.x1].reshape(b, -1, d)
    pe = select_rope(context.native_pe, indices, attention.rope_type)
    return attend(attention, x, pe).reshape(b, t, window.height, window.width, d)
