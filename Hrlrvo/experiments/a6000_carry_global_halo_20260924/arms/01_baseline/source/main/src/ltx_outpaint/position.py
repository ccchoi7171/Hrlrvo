"""Query-reference phases using the checkpoint's unmodified LTX RoPE.

References: mixed-res-dit@0825c5a wan/transformer_wan.py:134-255,668-701;
foveated_diffusion@694f493 src/diffsynth_fov/dit_video.py:83-126.
Adaptive uses v2 nominal-grid positions for full C2/C4 footprints, including
promoted mixed-role cells. Global's existing phase path is kept unchanged.
"""

from dataclasses import dataclass

import torch
from ltx_core.model.transformer.rope import (
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)


@dataclass(frozen=True)
class ViewContext:
    geometry: object
    positions: torch.Tensor
    sigma: torch.Tensor
    native_pe: tuple
    rope: object
    early_global_boost_scale: float | None = None

    def subset(self, indices):
        # Only the Adaptive path uses this; native_pe remains unused in subsets.
        return ViewContext(self.geometry, self.positions.index_select(0, indices),
                           self.sigma.index_select(0, indices), (), self.rope,
                           self.early_global_boost_scale)


@dataclass(frozen=True)
class RopeSettings:
    dim: int
    heads: int
    theta: float
    max_pos: tuple
    middle: bool
    double_precision: bool
    rope_type: object
    timestep_multiplier: float

    @classmethod
    def from_preprocessor(cls, processor):
        p = getattr(processor, "simple_preprocessor", processor)
        return cls(p.inner_dim, p.num_attention_heads, p.positional_embedding_theta,
                   tuple(p.max_pos), p.use_middle_indices_grid, p.double_precision_rope,
                   p.rope_type, p.timestep_scale_multiplier)


def target_query_positions(context, cells, query_scale):
    """Native time bounds plus v2 nominal-grid phases at Target-query scale.

    A cell at canonical origin o and nominal scale r_k has native position
    P_k = o / r_k + 1/2. PMA maps it to the query scale r_q = S/2:
    (P_k - 1/2) * (r_k / r_q) + 1/2 = o / r_q + 1/2.
    Pixel ownership and the fraction of Target in a full C2 footprint do not
    move its phase. Expanded/translated support uses the actual canonical origin.

    The two spatial endpoints both store the phase point, not an invented
    rectangular support. Native midpoint extraction therefore reads exactly
    that point. Temporal bounds, seconds, causal-first-frame convention and
    the native choice of temporal start/midpoint are left untouched.
    """
    if query_scale < 1 or not cells:
        raise ValueError("Target-query positions require nonempty support and a positive query scale")
    g, p = context.geometry, context.positions
    representatives = torch.tensor([cell.indices[0] for cell in cells], device=p.device)
    indices = (torch.arange(g.frames, device=p.device)[:, None] * g.height * g.width
               + representatives[None, :]).flatten()
    # index_select already owns its output; no additional clone is needed.
    mapped = p.index_select(2, indices)
    nominal = p.new_tensor([[cell.y / cell.scale + 0.5, cell.x / cell.scale + 0.5] for cell in cells])
    ratios = p.new_tensor([cell.scale / query_scale for cell in cells])[:, None]
    phases = (nominal - 0.5) * ratios + 0.5
    for axis, coordinate in ((1, 0), (2, 1)):
        phase = phases[:, coordinate].expand(g.frames, -1).reshape(1, -1)
        stride = (p[:, axis, 0, 1] - p[:, axis, 0, 0]).reshape(-1, 1)
        origin = p[:, axis, 0, 0].reshape(-1, 1)
        point = origin + phase * stride
        mapped[:, axis, :, 0] = point
        mapped[:, axis, :, 1] = point
    return mapped


def target_query_rope(context, cells, query_scale, dtype):
    """Rotate Target Q and every A/B/T K in the same query-native scale."""
    rope = context.rope
    return precompute_freqs_cis(
        target_query_positions(context, cells, query_scale),
        dim=rope.dim, out_dtype=dtype, theta=rope.theta, max_pos=list(rope.max_pos),
        use_middle_indices_grid=rope.middle, num_attention_heads=rope.heads,
        rope_type=rope.rope_type,
        freq_grid_generator=generate_freq_grid_np if rope.double_precision else generate_freq_grid_pytorch,
    )


def reference_origin(cells, scale):
    """Origin of the coarse cell lattice, including shifted Adaptive rectangles."""
    coarse = [cell for cell in cells if cell.scale == scale]
    return (coarse[0].y % scale, coarse[0].x % scale) if coarse else (0, 0)


def phase_rope(context, cells, reference_scale, dtype):
    """Two coordinate maps, not one common midpoint-RoPE for all mixed tokens.

    A query at scale r uses r-specific key phases. Fine keys sharing a coarse
    cell have identical coarse-query phase (PMA repeat rule); the attention
    caller retains one fine K/V representative per coarse cell for this pass.
    Canonical cell support lives in `cells`; only the RoPE reference changes.
    LTX pixel stride/midpoint convention, temporal seconds/bounds, frequencies,
    rotary ordering and heads remain native. No learned position parameters.
    """
    g, p = context.geometry, context.positions
    device = p.device
    representatives = torch.tensor([c.indices[0] for c in cells], device=device)
    indices = (torch.arange(g.frames, device=device)[:, None] * g.height * g.width
               + representatives[None, :]).flatten()
    mapped = p.index_select(2, indices).clone()
    coarse_origin = reference_origin(cells, reference_scale)
    for axis, start_index, end_index in ((1, 0, 2), (2, 1, 3)):
        starts, ends = [], []
        for cell in cells:
            start, end = cell.bounds[start_index], cell.bounds[end_index]
            if cell.scale < reference_scale:
                # The retained fine K/V represents a coarse context cell in this
                # query pass, as in PMA's one-top-left-key-per-2x2-group rule.
                origin = coarse_origin[axis - 1]
                start = origin + ((start - origin) // reference_scale) * reference_scale
                end = start + reference_scale
            starts.append(start / reference_scale)
            ends.append(end / reference_scale)
            if cell.scale > reference_scale and len(cell.indices) == cell.scale**2:
                # Public PMA rescales Wan ORIGIN indices, not cell centres.
                # Conjugate u -> r*u by LTX's P=u+1/2 convention:
                # P_key(query) = (P_key-1/2)*r + 1/2, r=s_key/s_query.
                # For aligned 2x full cells this fixes the extra half-fine-cell
                # phase in fine-Q/coarse-K. Same-scale and time are unchanged.
                shift = 0.5 * (1 - cell.scale / reference_scale)
                starts[-1] += shift
                ends[-1] += shift
        low = torch.tensor(starts, device=device, dtype=p.dtype).repeat(g.frames)[None, :]
        high = torch.tensor(ends, device=device, dtype=p.dtype).repeat(g.frames)[None, :]
        stride = (p[:, axis, 0, 1] - p[:, axis, 0, 0]).reshape(-1, 1)
        origin = p[:, axis, 0, 0].reshape(-1, 1)
        # Exact physical support stays in `cells`; these are phase bounds only.
        # Partial/irregular-cell centres retain the existing extension: public
        # PMA does not define them. Do not silently invent a new boundary rule;
        # their source-equivalence remains an explicit audit limitation.
        mapped[:, axis, :, 0] = origin + low * stride
        mapped[:, axis, :, 1] = origin + high * stride
    rope = context.rope
    return precompute_freqs_cis(
        mapped, dim=rope.dim, out_dtype=dtype, theta=rope.theta, max_pos=list(rope.max_pos),
        use_middle_indices_grid=rope.middle, num_attention_heads=rope.heads,
        rope_type=rope.rope_type,
        freq_grid_generator=generate_freq_grid_np if rope.double_precision else generate_freq_grid_pytorch,
    )
