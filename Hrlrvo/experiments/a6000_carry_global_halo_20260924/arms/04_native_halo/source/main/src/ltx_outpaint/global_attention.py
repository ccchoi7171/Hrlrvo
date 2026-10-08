"""The only full-canvas peer of the Local System; no W_O in this view."""

from .geometry import global_cells
from .native_local_attention import attend
from .position import phase_rope


def global_attention(attention, grid, context):
    b, t, h, w, d = grid.shape
    s = attention.scale
    cells = global_cells(h, w, s)
    compressed = attention.compressors[str(s)](grid, cells)
    pe = phase_rope(context, cells, s, grid.dtype)
    output = attend(attention, compressed.reshape(b, -1, d), pe, attention.global_lora)
    return attention.decompressors[str(s)](
        output.reshape(b, t, len(cells), d), (h // s, w // s), (h, w))
