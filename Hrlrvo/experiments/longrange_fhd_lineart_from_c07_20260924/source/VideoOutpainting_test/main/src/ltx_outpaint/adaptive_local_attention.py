"""Shared train/inference Adaptive path: C4-first promotion, Target-only write."""

from collections import defaultdict

import torch

from .geometry import adaptive_layout
from .lora import project_with_loras
from .native_local_attention import select_rope
from .position import target_query_rope


def mixed_attention(attention, tokens, layout, context):
    """Compute only Target-touching C2 query rows, retaining all full C2/C4 K/V.

    These are the same Adaptive QKV projections as self-attention, not a new
    cross-attention module. Query-eligible cells lead each frame's sequence;
    their full C2 read footprint can include known A or contextual B pixels.
    """
    b, t, n, d = tokens.shape
    if n != len(layout.cells) or layout.target_count == 0:
        raise ValueError("Packed Adaptive tokens do not match their support layout")
    x = tokens.reshape(b, t * n, d)
    target = tokens[:, :, :layout.target_count].reshape(b, t * layout.target_count, d)
    lora = attention.adaptive_lora
    shared = attention.shared_outpainting_lora
    # Never project non-Target-touching queries. Every query remains in K/V.
    q = project_with_loras(attention.to_q, target, (shared["q"], lora["q"]))
    k = project_with_loras(attention.to_k, x, (shared["k"], lora["k"]))
    v = project_with_loras(attention.to_v, x, (shared["v"], lora["v"]))
    qi = (torch.arange(t, device=x.device)[:, None] * n
          + torch.arange(layout.target_count, device=x.device)[None, :]).flatten()
    pe = target_query_rope(context, layout.cells, attention.scale // 2, x.dtype)
    q, k = attention.preattention_function(
        q, k, attention, None, select_rope(pe, qi, attention.rope_type), pe,
    )
    out = attention.attention_function(q, k, v, attention.heads)
    return out.reshape(b, t, layout.target_count, d)


def adaptive_local(attention, grid, target, context):
    """Return (Local-coordinate contribution, Target write mask) from one layout.

    Read support may extend outside the Local window. The returned mask never
    does; it also gates the Local fusion delta so absent Adaptive output cannot
    attenuate Native features outside resolved Target.
    """
    b, t, h, w, d = grid.shape
    # LTX's AttentionCallable has no varlen/cu_seqlens interface. Geometry/shape
    # buckets batch identical receptive fields with no sequence or spatial pad.
    buckets = defaultdict(list)
    for i, anchor in enumerate(context.geometry.batch_anchors(b)):
        buckets[anchor].append(i)
    # Zeros here mean no Adaptive OUTPUT contribution, never fake input tokens.
    result = grid.new_zeros(b, t, target.height, target.width, d)
    write_mask = torch.zeros((b, 1, target.height * target.width, 1), device=grid.device, dtype=torch.bool)
    for anchor, batch_indices in buckets.items():
        layout = adaptive_layout(anchor, target, h, w, attention.scale)
        if layout.target_count == 0:
            continue  # No compression, QKV, attention or R2 for resolved T=empty.
        indices = torch.tensor(batch_indices, device=grid.device)
        source = grid if len(batch_indices) == b else grid.index_select(0, indices)
        packed = grid.new_empty(len(batch_indices), t, len(layout.cells), d)
        for scale in (attention.scale // 2, attention.scale):
            locations = [i for i, c in enumerate(layout.cells) if c.scale == scale]
            if not locations:
                continue
            cells = tuple(layout.cells[i] for i in locations)
            if any(cell.valid_count != scale * scale or cell.offsets != tuple(range(scale * scale)) for cell in cells):
                raise RuntimeError("Adaptive compression accepts only complete, real C2/C4 footprints")
            values = attention.compressors[str(scale)](source, cells)
            packed = packed.index_copy(2, torch.tensor(locations, device=grid.device), values)
        target_tokens = mixed_attention(attention, packed, layout, context.subset(indices))
        # Restore each query's complete C2 footprint using the SAME existing R2.
        # Query cells are not a dense rectangle: treating cell index as space or
        # filling missing query locations would fabricate spatial evidence.
        # Cell-as-batch preserves T and the native temporal Conv3D refinement.
        batch, queries = len(batch_indices), layout.target_count
        per_cell = target_tokens.permute(0, 2, 1, 3).reshape(batch * queries, t, 1, d)
        restored = attention.decompressors[str(attention.scale // 2)](
            per_cell, (1, 1), (2, 2))
        restored = restored.reshape(batch, queries, t, 4, d).permute(0, 2, 1, 3, 4).reshape(batch, t, queries * 4, d)
        # A/B are read context, never Adaptive write owners. This also preserves
        # every odd-edge T pixel without cropping a query footprint or remainder.
        selected = restored.index_select(2, torch.tensor(layout.write_source_indices, device=grid.device))
        destination = torch.tensor(layout.write_target_indices, device=grid.device)
        local = grid.new_zeros(batch, t, target.height * target.width, d).index_copy(
            2, destination, selected)
        result = result.index_copy(0, indices, local.reshape(batch, t, target.height, target.width, d))
        bucket_mask = torch.zeros((1, 1, target.height * target.width, 1), device=grid.device, dtype=torch.bool)
        bucket_mask.index_fill_(2, destination, True)
        write_mask.index_copy_(0, indices, bucket_mask.expand(batch, -1, -1, -1))
    return result, write_mask.reshape(b, 1, target.height, target.width, 1)
