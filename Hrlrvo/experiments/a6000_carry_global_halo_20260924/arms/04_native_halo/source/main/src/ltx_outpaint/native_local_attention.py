"""Native Local attention with a read-only spatial K/V halo; never applies W_O.

This isolated experiment keeps each canonical Local window as the exact query
and output support.  Keys and values may read exactly one canonical Native
spatial token beyond it (roughly 32 RGB pixels for the current patch-1 latent
grid).  The support is clipped to the real canvas: it is never padded or
wrapped to the opposite edge.
"""

import torch

from .lora import project_with_loras


# The experiment protocol requires exactly one Native spatial token.  A
# runtime attribute is accepted only to make the radius-zero legacy path
# directly testable without altering model state.
NATIVE_LOCAL_HALO_RADIUS = 1


def select_rope(pe, indices, rope_type):
    axis = 2 if rope_type.value == "split" else 1
    return tuple(part.index_select(axis, indices) for part in pe)


def attend(attention, x, pe, lora=None, *, kv_x=None, k_pe=None):
    """Attend with optional distinct K/V tokens and rotary positions.

    Existing callers retain the original self-attention behavior when ``kv_x``
    is omitted.  The halo arm supplies a larger ``kv_x`` while keeping ``x`` as
    the canonical query/output sequence.
    """
    shared = attention.shared_outpainting_lora
    projections = (shared["q"],) if lora is None else (shared["q"], lora["q"])
    q = project_with_loras(attention.to_q, x, projections)
    source = x if kv_x is None else kv_x
    projections = (shared["k"],) if lora is None else (shared["k"], lora["k"])
    k = project_with_loras(attention.to_k, source, projections)
    projections = (shared["v"],) if lora is None else (shared["v"], lora["v"])
    v = project_with_loras(attention.to_v, source, projections)
    q, k = attention.preattention_function(q, k, attention, None, pe, k_pe)
    return attention.attention_function(q, k, v, attention.heads)


def _canonical_indices(t, h, w, y0, y1, x0, x1, device):
    return (torch.arange(t, device=device)[:, None, None] * h * w
            + torch.arange(y0, y1, device=device)[None, :, None] * w
            + torch.arange(x0, x1, device=device)[None, None, :]).flatten()


def native_local(attention, grid, window, context):
    b, t, h, w, d = grid.shape
    radius = getattr(attention, "native_local_halo_radius", NATIVE_LOCAL_HALO_RADIUS)
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError("native_local_halo_radius must be a non-negative integer")
    indices = _canonical_indices(
        t, h, w, window.y0, window.y1, window.x0, window.x1, grid.device)
    x = grid[:, :, window.y0:window.y1, window.x0:window.x1].reshape(b, -1, d)
    pe = select_rope(context.native_pe, indices, attention.rope_type)
    if radius == 0:
        # Preserve the exact pre-experiment projection/RoPE/attention call path.
        output = attend(attention, x, pe)
    else:
        y0, y1 = max(0, window.y0 - radius), min(h, window.y1 + radius)
        x0, x1 = max(0, window.x0 - radius), min(w, window.x1 + radius)
        support_indices = _canonical_indices(t, h, w, y0, y1, x0, x1, grid.device)
        support = grid[:, :, y0:y1, x0:x1].reshape(b, -1, d)
        k_pe = select_rope(context.native_pe, support_indices, attention.rope_type)
        output = attend(attention, x, pe, kv_x=support, k_pe=k_pe)
    return output.reshape(b, t, window.height, window.width, d)
