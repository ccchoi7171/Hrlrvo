from __future__ import annotations

from dataclasses import replace
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import (
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)
from torch import nn

from ltx_outpaint import adaptive_local_attention
from ltx_outpaint import model as outpaint_model
from ltx_outpaint.compression import SpatialCompressor, SpatialDecompressor
from ltx_outpaint.geometry import (
    ANCHOR,
    BRIDGE,
    TARGET,
    CanvasGeometry,
    Cell,
    Rect,
    adaptive_layout,
    align_adaptive_support,
    build_adaptive_multires_tokens,
    global_cells,
    local_windows,
    resolve_adaptive_roles,
)
from ltx_outpaint.lora import project_with_loras
from ltx_outpaint.model import (
    ArchitectureConfig,
    MultiViewSelfAttention,
    _native_sa_gate,
)
from ltx_outpaint.position import (
    RopeSettings,
    ViewContext,
    target_query_positions,
    target_query_rope,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("box", "height", "width", "expected"),
    [
        (Rect(0, 0, 4, 4), 8, 8, Rect(0, 0, 4, 4)),
        # A divisible box keeps its existing origin, including odd origins.
        (Rect(1, 1, 5, 5), 8, 8, Rect(1, 1, 5, 5)),
        (Rect(0, 0, 3, 3), 8, 8, Rect(0, 0, 4, 4)),
        (Rect(5, 5, 8, 8), 8, 8, Rect(4, 4, 8, 8)),
        # Exact centre ties expand toward bottom/right.
        (Rect(3, 3, 5, 5), 8, 8, Rect(3, 3, 7, 7)),
        (Rect(1, 2, 5, 5), 8, 8, Rect(1, 2, 5, 6)),
        (Rect(2, 1, 5, 5), 8, 8, Rect(2, 1, 6, 5)),
        # If the centre-facing boundary is exhausted, only the remainder goes
        # to the opposite side; the box is never shifted onto a global lattice.
        (Rect(1, 1, 6, 6), 8, 8, Rect(0, 0, 8, 8)),
        (Rect(2, 2, 7, 7), 8, 8, Rect(0, 0, 8, 8)),
        (Rect(2, 3, 6, 6), 9, 9, Rect(2, 3, 6, 7)),
        (Rect(4, 4, 9, 9), 10, 10, Rect(1, 1, 9, 9)),
        (Rect(17, 30, 51, 90), 68, 120, Rect(17, 30, 53, 90)),
    ],
)
def test_centerward_minimal_c4_alignment(
    box: Rect, height: int, width: int, expected: Rect,
) -> None:
    aligned = align_adaptive_support(box, height, width)
    assert aligned == expected
    assert aligned.height == ceil(box.height / 4) * 4
    assert aligned.width == ceil(box.width / 4) * 4
    assert aligned.y0 <= box.y0 < box.y1 <= aligned.y1
    assert aligned.x0 <= box.x0 < box.x1 <= aligned.x1
    assert aligned.y1 <= height and aligned.x1 <= width


def test_alignment_rejects_when_next_c4_extent_cannot_fit_canvas() -> None:
    with pytest.raises(ValueError):
        align_adaptive_support(Rect(0, 0, 3, 3), 3, 3)


def test_role_resolution_is_row_major_and_anchor_then_target_then_bridge() -> None:
    support = Rect(0, 0, 4, 4)
    anchor = Rect(0, 0, 2, 2)
    target = Rect(1, 1, 4, 4)
    roles = resolve_adaptive_roles(anchor, target, support)

    assert len(roles) == 16
    assert roles.count(ANCHOR) == 4
    assert roles.count(TARGET) == 8
    assert roles.count(BRIDGE) == 4
    assert roles[1 * 4 + 1] == ANCHOR  # A wins an A/T overlap.
    assert roles[2 * 4 + 2] == TARGET
    assert roles[0 * 4 + 3] == BRIDGE


@pytest.mark.parametrize("role", [ANCHOR, TARGET])
def test_non_bridge_c4_parent_is_promoted_to_four_full_c2(role: int) -> None:
    cells = build_adaptive_multires_tokens(Rect(0, 0, 4, 4), (role,) * 16, width=4)
    assert len(cells) == 4
    assert all(cell.scale == 2 for cell in cells)
    assert all(cell.valid_count == 4 and cell.offsets == (0, 1, 2, 3) for cell in cells)
    assert all(cell.pixel_roles == (role,) * 4 for cell in cells)
    assert sum(bool(cell.role & TARGET) for cell in cells) == (4 if role == TARGET else 0)


def test_pure_bridge_parent_stays_one_full_c4() -> None:
    cells = build_adaptive_multires_tokens(Rect(0, 0, 4, 4), (BRIDGE,) * 16, width=4)
    assert len(cells) == 1
    cell = cells[0]
    assert cell.scale == 4 and cell.role == BRIDGE
    assert cell.indices == tuple(range(16))
    assert cell.offsets == tuple(range(16))
    assert cell.pixel_roles == (BRIDGE,) * 16


def test_thirteen_of_sixteen_bridge_is_promoted_not_partially_compressed() -> None:
    roles = (ANCHOR, ANCHOR, ANCHOR) + (BRIDGE,) * 13
    cells = build_adaptive_multires_tokens(Rect(0, 0, 4, 4), roles, width=4)

    assert len(cells) == 4
    assert all(cell.scale == 2 and cell.valid_count == 4 for cell in cells)
    assert not any(cell.scale == 4 for cell in cells)
    assert sorted(index for cell in cells for index in cell.indices) == list(range(16))


@pytest.mark.parametrize(
    ("touch_role", "expected_query_count"),
    [(ANCHOR, 0), (TARGET, 1)],
)
def test_one_non_bridge_pixel_promotes_entire_c4_parent(
    touch_role: int, expected_query_count: int,
) -> None:
    roles = [BRIDGE] * 16
    roles[0] = touch_role
    cells = build_adaptive_multires_tokens(Rect(0, 0, 4, 4), tuple(roles), width=4)

    assert len(cells) == 4
    assert all(cell.scale == 2 and cell.valid_count == 4 for cell in cells)
    assert sum(cell.pixel_roles == (BRIDGE,) * 4 for cell in cells) == 3
    query_count = sum(bool(cell.role & TARGET) for cell in cells)
    assert query_count == expected_query_count
    assert all(bool(cells[index].role & TARGET) for index in range(query_count))
    assert all(not (cell.role & TARGET) for cell in cells[query_count:])


def test_mixed_c2_uses_bitwise_role_and_query_cells_sort_first() -> None:
    # The first parent is pure Bridge. The second is promoted and has one
    # mixed A/T query cell; query-first ordering must move it ahead of the C4.
    support = Rect(0, 0, 4, 8)
    roles = [BRIDGE] * 32
    roles[4] = TARGET
    roles[5] = ANCHOR
    roles[12] = ANCHOR
    roles[13] = ANCHOR
    cells = build_adaptive_multires_tokens(support, tuple(roles), width=8)

    query_count = sum(bool(cell.role & TARGET) for cell in cells)
    assert query_count == 1
    assert bool(cells[0].role & TARGET)
    assert cells[0].role == (ANCHOR | TARGET)
    assert all(not (cell.role & TARGET) for cell in cells[query_count:])
    assert any(cell.scale == 4 and cell.role == BRIDGE for cell in cells)
    assert all(cell.valid_count == cell.scale**2 for cell in cells)


def _assert_layout_write_map(anchor: Rect, target: Rect, height: int, width: int) -> None:
    layout = adaptive_layout(anchor, target, height, width, scale=4)
    assert layout.rectangle is not None
    assert layout.rectangle.height % 4 == layout.rectangle.width % 4 == 0
    assert layout.target_count == sum(bool(cell.role & TARGET) for cell in layout.cells)
    assert all(cell.scale in (2, 4) for cell in layout.cells)
    assert all(cell.valid_count == cell.scale**2 for cell in layout.cells)
    assert all(bool(cell.role & TARGET) for cell in layout.cells[: layout.target_count])
    assert all(not (cell.role & TARGET) for cell in layout.cells[layout.target_count :])

    expected_targets = {
        (y - target.y0) * target.width + (x - target.x0)
        for y in range(target.y0, target.y1)
        for x in range(target.x0, target.x1)
        if not anchor.contains(y, x)
    }
    assert len(layout.write_source_indices) == len(layout.write_target_indices)
    assert set(layout.write_target_indices) == expected_targets
    assert len(layout.write_target_indices) == len(set(layout.write_target_indices))
    for source, destination in zip(
        layout.write_source_indices, layout.write_target_indices, strict=True,
    ):
        query_index, tap = divmod(source, 4)
        cell = layout.cells[query_index]
        canonical = cell.indices[tap]
        y, x = divmod(canonical, width)
        assert cell.scale == 2 and cell.pixel_roles[tap] == TARGET
        assert destination == (y - target.y0) * target.width + (x - target.x0)


@pytest.mark.parametrize(
    ("anchor", "target", "height", "width"),
    [
        (Rect(4, 4, 8, 8), Rect(0, 0, 2, 2), 8, 8),
        (Rect(4, 4, 8, 8), Rect(3, 3, 5, 5), 8, 8),
        (Rect(0, 0, 4, 4), Rect(6, 6, 8, 8), 8, 8),
        (Rect(2, 2, 6, 6), Rect(0, 3, 3, 6), 8, 8),
        (Rect(2, 2, 6, 6), Rect(5, 0, 8, 3), 8, 8),
        (Rect(4, 4, 8, 8), Rect(0, 4, 3, 7), 9, 9),
        (Rect(4, 4, 8, 8), Rect(4, 0, 7, 3), 9, 9),
        (Rect(4, 4, 8, 8), Rect(7, 7, 9, 9), 9, 9),
        (Rect(4, 4, 8, 8), Rect(0, 0, 3, 3), 10, 10),
        (Rect(4, 4, 8, 8), Rect(8, 0, 10, 2), 10, 10),
        (Rect(2, 2, 10, 10), Rect(0, 8, 4, 12), 12, 12),
        (Rect(2, 2, 10, 10), Rect(8, 8, 12, 12), 12, 12),
    ],
)
def test_layout_query_and_write_maps_cover_resolved_target_once(
    anchor: Rect, target: Rect, height: int, width: int,
) -> None:
    _assert_layout_write_map(anchor, target, height, width)


@pytest.mark.parametrize(
    ("anchor", "target"),
    [
        (Rect(0, 0, 8, 8), Rect(2, 2, 4, 4)),
        (Rect(0, 0, 8, 8), Rect(0, 0, 8, 8)),
        (Rect(1, 1, 7, 7), Rect(3, 3, 5, 5)),
    ],
)
def test_fully_known_target_returns_empty_before_alignment(
    anchor: Rect, target: Rect,
) -> None:
    layout = adaptive_layout(anchor, target, 8, 8, scale=4)
    assert layout.target == target
    assert layout.rectangle is None
    assert layout.cells == () and layout.roles == ()
    assert layout.target_count == 0
    assert layout.write_source_indices == () and layout.write_target_indices == ()


def test_global_c4_rejects_partial_boundary_geometry() -> None:
    with pytest.raises(ValueError, match="divisible"):
        global_cells(5, 5, 4)


class _RecordingDecompressor(SpatialDecompressor):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.calls: list[tuple[tuple[int, ...], tuple[int, int], tuple[int, int]]] = []

    def forward(
        self,
        tokens: torch.Tensor,
        compressed_shape: tuple[int, int],
        target_shape: tuple[int, int],
    ) -> torch.Tensor:
        self.calls.append((tuple(tokens.shape), compressed_shape, target_shape))
        return super().forward(tokens, compressed_shape, target_shape)


def test_runtime_restores_each_query_cell_and_writes_resolved_target_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, target = Rect(4, 4, 8, 8), Rect(3, 3, 5, 5)
    geometry = CanvasGeometry(2, 8, 8, (anchor,))
    context = SimpleNamespace(geometry=geometry, subset=lambda _indices: context)
    decompressor = _RecordingDecompressor(1)
    attention = SimpleNamespace(
        scale=4,
        compressors=nn.ModuleDict({"2": SpatialCompressor(1, 2), "4": SpatialCompressor(1, 4)}),
        decompressors=nn.ModuleDict({"2": decompressor}),
    )
    grid = torch.arange(1, 1 + 2 * 8 * 8, dtype=torch.float32).reshape(1, 2, 8, 8, 1)

    def identity_queries(_attention, packed, layout, _context):
        return packed[:, :, : layout.target_count]

    monkeypatch.setattr(adaptive_local_attention, "mixed_attention", identity_queries)
    output, write_mask = adaptive_local_attention.adaptive_local(
        attention, grid, target, context,
    )
    layout = adaptive_layout(anchor, target, 8, 8, 4)

    assert output.shape == (1, 2, target.height, target.width, 1)
    expected_written = torch.tensor(
        [[not anchor.contains(y, x) for x in range(target.x0, target.x1)]
         for y in range(target.y0, target.y1)],
    )
    assert torch.equal(output[0, 0, :, :, 0] != 0, expected_written)
    assert torch.equal(write_mask[0, 0, :, :, 0], expected_written)
    assert len(decompressor.calls) == 1
    shape, compressed_shape, restored_shape = decompressor.calls[0]
    assert shape == (layout.target_count, 2, 1, 1)
    assert compressed_shape == (1, 1) and restored_shape == (2, 2)


@pytest.mark.parametrize(
    "pixel_roles",
    [
        (ANCHOR, TARGET, TARGET, TARGET),  # AT / TT
        (BRIDGE, TARGET, TARGET, TARGET),  # BT / TT
        (ANCHOR, ANCHOR, ANCHOR, TARGET),  # AA / AT
    ],
)
def test_mixed_query_cell_restores_full_c2_but_writes_only_target_taps(
    monkeypatch: pytest.MonkeyPatch, pixel_roles: tuple[int, ...],
) -> None:
    cell = Cell(
        scale=2,
        role=pixel_roles[0] | pixel_roles[1] | pixel_roles[2] | pixel_roles[3],
        y=0,
        x=0,
        indices=(0, 1, 2, 3),
        offsets=(0, 1, 2, 3),
        bounds=(0, 0, 2, 2),
        pixel_roles=pixel_roles,
    )
    target_taps = tuple(i for i, role in enumerate(pixel_roles) if role == TARGET)
    layout = SimpleNamespace(
        cells=(cell,),
        target_count=1,
        write_source_indices=target_taps,
        write_target_indices=target_taps,
    )
    monkeypatch.setattr(adaptive_local_attention, "adaptive_layout", lambda *_args: layout)
    monkeypatch.setattr(
        adaptive_local_attention,
        "mixed_attention",
        lambda _attention, packed, _layout, _context: packed,
    )
    context = SimpleNamespace(
        geometry=CanvasGeometry(1, 2, 2, (Rect(0, 0, 1, 1),)),
        subset=lambda _indices: context,
    )
    attention = SimpleNamespace(
        scale=4,
        compressors=nn.ModuleDict({"2": SpatialCompressor(1, 2), "4": SpatialCompressor(1, 4)}),
        decompressors=nn.ModuleDict({"2": SpatialDecompressor(1)}),
    )
    grid = torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 2, 2, 1)

    output, write_mask = adaptive_local_attention.adaptive_local(
        attention, grid, Rect(0, 0, 2, 2), context,
    )
    expected = torch.tensor([role == TARGET for role in pixel_roles])

    assert torch.equal(write_mask.flatten(), expected)
    assert torch.equal(output.flatten() != 0, expected)


def test_runtime_fully_known_target_exits_without_compression_or_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, target = Rect(0, 0, 8, 8), Rect(2, 2, 4, 4)
    context = SimpleNamespace(
        geometry=CanvasGeometry(2, 8, 8, (anchor,)),
        subset=lambda _indices: pytest.fail("Empty T must not subset the attention context"),
    )
    attention = SimpleNamespace(
        scale=4,
        compressors=nn.ModuleDict({
            "2": nn.Identity(),
            "4": nn.Identity(),
        }),
        decompressors=nn.ModuleDict({"2": nn.Identity()}),
    )
    monkeypatch.setattr(
        adaptive_local_attention,
        "mixed_attention",
        lambda *_args, **_kwargs: pytest.fail("Empty T must not execute QKV/attention"),
    )
    grid = torch.randn(1, 2, 8, 8, 3)
    output, write_mask = adaptive_local_attention.adaptive_local(
        attention, grid, target, context,
    )
    assert output.shape == (1, 2, 2, 2, 3)
    assert torch.count_nonzero(output) == 0
    assert not bool(write_mask.any())


def test_r2_cell_batch_keeps_batch_time_and_query_samples_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, target = Rect(4, 4, 8, 8), Rect(3, 3, 5, 5)
    geometry = CanvasGeometry(3, 8, 8, (anchor,))
    context = SimpleNamespace(geometry=geometry, subset=lambda _indices: context)
    attention = SimpleNamespace(
        scale=4,
        compressors=nn.ModuleDict({"2": SpatialCompressor(1, 2), "4": SpatialCompressor(1, 4)}),
        decompressors=nn.ModuleDict({"2": SpatialDecompressor(1)}),
    )

    def labelled_queries(_attention, packed, layout, _context):
        batch, frames = packed.shape[:2]
        labels = torch.empty(batch, frames, layout.target_count, 1)
        for batch_index in range(batch):
            for frame in range(frames):
                for query in range(layout.target_count):
                    labels[batch_index, frame, query, 0] = 1000 * batch_index + 100 * frame + query + 1
        return labels

    monkeypatch.setattr(adaptive_local_attention, "mixed_attention", labelled_queries)
    grid = torch.randn(2, 3, 8, 8, 1)
    output, write_mask = adaptive_local_attention.adaptive_local(
        attention, grid, target, context,
    )
    output = output.flatten(2, 3)
    layout = adaptive_layout(anchor, target, 8, 8, 4)

    for batch_index in range(2):
        for frame in range(3):
            for source, destination in zip(
                layout.write_source_indices, layout.write_target_indices, strict=True,
            ):
                query = source // 4
                assert output[batch_index, frame, destination, 0] == (
                    1000 * batch_index + 100 * frame + query + 1
                )
    expected_write = torch.zeros(target.height * target.width, dtype=torch.bool)
    expected_write[list(layout.write_target_indices)] = True
    assert torch.equal(write_mask[0, 0].flatten(), expected_write)
    assert torch.equal(write_mask[1, 0].flatten(), expected_write)


def test_runtime_never_reads_outside_aligned_adaptive_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, target = Rect(4, 4, 8, 8), Rect(0, 0, 2, 2)
    layout = adaptive_layout(anchor, target, 8, 8, 4)
    context = SimpleNamespace(
        geometry=CanvasGeometry(2, 8, 8, (anchor,)),
        subset=lambda _indices: context,
    )
    attention = SimpleNamespace(
        scale=4,
        compressors=nn.ModuleDict({"2": SpatialCompressor(1, 2), "4": SpatialCompressor(1, 4)}),
        decompressors=nn.ModuleDict({"2": SpatialDecompressor(1)}),
    )
    grid = torch.full((1, 2, 8, 8, 1), torch.nan)
    box = layout.rectangle
    assert box is not None
    grid[:, :, box.y0 : box.y1, box.x0 : box.x1] = 1.0
    monkeypatch.setattr(
        adaptive_local_attention,
        "mixed_attention",
        lambda _attention, packed, current, _context: packed[:, :, : current.target_count],
    )

    output, write_mask = adaptive_local_attention.adaptive_local(
        attention, grid, target, context,
    )
    assert bool(torch.isfinite(output).all())
    assert write_mask.dtype == torch.bool


def _attention_and_context(
    *, frames: int, height: int, width: int, anchor: Rect, k: int = 3,
) -> tuple[MultiViewSelfAttention, ViewContext]:
    native = Attention(query_dim=32, heads=2, dim_head=16)
    native.requires_grad_(False)
    attention = MultiViewSelfAttention(native, ArchitectureConfig(k=k), block_index=0)
    positions = _native_positions(frames, height, width)
    rope = RopeSettings.from_preprocessor(
        SimpleNamespace(
            inner_dim=32,
            num_attention_heads=2,
            positional_embedding_theta=10_000.0,
            max_pos=[20, 2048, 2048],
            use_middle_indices_grid=True,
            double_precision_rope=False,
            rope_type=native.rope_type,
            timestep_scale_multiplier=1000,
        ),
    )
    native_pe = precompute_freqs_cis(
        positions,
        dim=32,
        out_dtype=torch.float32,
        theta=rope.theta,
        max_pos=list(rope.max_pos),
        use_middle_indices_grid=rope.middle,
        num_attention_heads=rope.heads,
        rope_type=rope.rope_type,
        freq_grid_generator=generate_freq_grid_pytorch,
    )
    context = ViewContext(
        CanvasGeometry(frames, height, width, (anchor,)),
        positions,
        torch.tensor([0.5]),
        native_pe,
        rope,
    )
    return attention, context


def test_target_query_attention_equals_full_attention_target_rows() -> None:
    torch.manual_seed(31)
    attention, context = _attention_and_context(
        frames=2, height=8, width=8, anchor=Rect(4, 4, 8, 8), k=4,
    )
    layout = adaptive_layout(Rect(4, 4, 8, 8), Rect(0, 0, 2, 2), 8, 8, 4)
    tokens = torch.randn(1, 2, len(layout.cells), 32)

    restricted = adaptive_local_attention.mixed_attention(attention, tokens, layout, context)
    packed = tokens.reshape(1, -1, 32)
    loras = attention.adaptive_lora
    shared = attention.shared_outpainting_lora
    q = project_with_loras(attention.to_q, packed, (shared["q"], loras["q"]))
    k = project_with_loras(attention.to_k, packed, (shared["k"], loras["k"]))
    v = project_with_loras(attention.to_v, packed, (shared["v"], loras["v"]))
    pe = target_query_rope(context, layout.cells, 2, packed.dtype)
    q, k = attention.preattention_function(q, k, attention, None, pe, pe)
    full = attention.attention_function(q, k, v, attention.heads)
    rows = (
        torch.arange(2)[:, None] * len(layout.cells)
        + torch.arange(layout.target_count)[None, :]
    ).flatten()
    expected = full.index_select(1, rows).reshape_as(restricted)

    assert torch.equal(restricted, expected)


def _nonzero_finite_grad(parameter: torch.Tensor) -> bool:
    return (
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool(torch.count_nonzero(parameter.grad))
    )


class _ConstantGate(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        return torch.full_like(gate, self.value)


def _stub_global_local_branches(
    attention: MultiViewSelfAttention, monkeypatch: pytest.MonkeyPatch
) -> tuple[torch.Tensor, torch.Tensor]:
    attention.to_out = nn.Identity()
    attention.local_fusion = _ConstantGate(0.25)
    attention.global_fusion = _ConstantGate(0.5)

    def fake_native(_attention, grid, window, _context):
        return torch.ones(
            grid.shape[0],
            grid.shape[1],
            window.height,
            window.width,
            grid.shape[-1],
            device=grid.device,
            dtype=grid.dtype,
        )

    def fake_adaptive(_attention, grid, window, _context):
        shape = (
            grid.shape[0],
            grid.shape[1],
            window.height,
            window.width,
            grid.shape[-1],
        )
        mask = torch.zeros(
            (grid.shape[0], 1, window.height, window.width, 1),
            device=grid.device,
            dtype=torch.bool,
        )
        return torch.zeros(shape, device=grid.device, dtype=grid.dtype), mask

    monkeypatch.setattr(outpaint_model, "native_local", fake_native)
    monkeypatch.setattr(outpaint_model, "adaptive_local", fake_adaptive)
    monkeypatch.setattr(
        outpaint_model,
        "global_attention",
        lambda _attention, grid, _context: torch.zeros_like(grid),
    )
    hidden = torch.zeros(1, 8 * 8, 32)
    return hidden, torch.zeros_like(hidden)


def test_early_global_boost_off_is_exact_legacy_lerp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention, context = _attention_and_context(
        frames=1, height=8, width=8, anchor=Rect(2, 2, 6, 6), k=4,
    )
    hidden, gate = _stub_global_local_branches(attention, monkeypatch)

    def reject_boost(*_args):
        pytest.fail("OFF path invoked Early Global Boost gate arithmetic")

    monkeypatch.setattr(outpaint_model, "_apply_early_global_boost_", reject_boost)
    token = _native_sa_gate.set((attention, gate))
    try:
        with torch.inference_mode():
            output = attention(hidden, pe=context)
    finally:
        _native_sa_gate.reset(token)

    expected = torch.lerp(torch.zeros_like(hidden), torch.ones_like(hidden), 0.5)
    assert torch.equal(output, expected)


def test_early_global_boost_forward_uses_learned_global_contribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention, context = _attention_and_context(
        frames=1, height=8, width=8, anchor=Rect(2, 2, 6, 6), k=4,
    )
    context = ViewContext(
        context.geometry,
        context.positions,
        context.sigma,
        context.native_pe,
        context.rope,
        early_global_boost_scale=1.4,
    )
    hidden, gate = _stub_global_local_branches(attention, monkeypatch)
    token = _native_sa_gate.set((attention, gate))
    try:
        with torch.inference_mode():
            output = attention(hidden, pe=context)
    finally:
        _native_sa_gate.reset(token)

    learned_weight = torch.full_like(hidden, 0.5)
    effective_weight = 1.0 - torch.clamp(1.4 * (1.0 - learned_weight), 0.0, 1.0)
    expected = torch.lerp(torch.zeros_like(hidden), torch.ones_like(hidden), effective_weight)
    assert torch.equal(output, expected)


def test_early_global_boost_reuses_gate_and_never_changes_learned_state() -> None:
    attention, _context = _attention_and_context(
        frames=1, height=8, width=8, anchor=Rect(2, 2, 6, 6), k=4,
    )
    state_before = {name: value.clone() for name, value in attention.state_dict().items()}
    with torch.inference_mode():
        learned_weight = attention.global_fusion(torch.randn(1, 1, 32))
        learned_copy = learned_weight.clone()
        data_ptr = learned_weight.data_ptr()
        effective_weight = outpaint_model._apply_early_global_boost_(
            learned_weight, 1.4
        )

    expected = 1.0 - torch.clamp(1.4 * (1.0 - learned_copy), 0.0, 1.0)
    assert effective_weight.data_ptr() == data_ptr
    assert torch.equal(effective_weight, expected)
    assert torch.all((1.0 - effective_weight) >= (1.0 - learned_copy))
    assert bool(((0.0 <= effective_weight) & (effective_weight <= 1.0)).all())
    assert set(attention.state_dict()) == set(state_before)
    assert all(
        torch.equal(value, state_before[name])
        for name, value in attention.state_dict().items()
    )
    assert "early_global_boost" not in ArchitectureConfig.__dataclass_fields__
    assert not any("early_global_boost" in name for name in attention.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_early_global_boost_preserves_gate_dtype_and_range(dtype: torch.dtype) -> None:
    learned_weight = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=dtype)
    learned_copy = learned_weight.clone()
    with torch.inference_mode():
        effective_weight = outpaint_model._apply_early_global_boost_(
            learned_weight, 1.4
        )
    expected = 1.0 - torch.clamp(1.4 * (1.0 - learned_copy), 0.0, 1.0)

    assert effective_weight.dtype == dtype
    assert torch.equal(effective_weight, expected)
    assert bool(((0.0 <= effective_weight) & (effective_weight <= 1.0)).all())


def test_early_global_boost_context_is_immutable_across_cfg_and_batch_split() -> None:
    geometry = CanvasGeometry(1, 2, 2, (Rect(0, 0, 2, 2),))
    base = Modality(
        latent=torch.zeros(2, 4, 8),
        sigma=torch.ones(2),
        timesteps=torch.ones(2, 4),
        positions=torch.zeros(2, 3, 4, 2),
        context=torch.zeros(2, 3, 8),
    )
    tagged = outpaint_model.attach_geometry(
        base, geometry, early_global_boost_scale=1.4
    )
    cfg_copy = replace(tagged, context=torch.ones_like(tagged.context))

    assert tagged.early_global_boost_scale == 1.4
    assert cfg_copy.early_global_boost_scale == 1.4
    assert all(
        chunk.early_global_boost_scale == 1.4 for chunk in tagged.split([1, 1])
    )
    assert outpaint_model.attach_geometry(base, geometry).early_global_boost_scale is None


def test_early_global_boost_rejects_gradient_enabled_execution() -> None:
    with pytest.raises(RuntimeError, match="inference-only"):
        outpaint_model._apply_early_global_boost_(torch.full((2,), 0.5), 1.4)


def test_local_fusion_is_exact_native_passthrough_outside_target_write_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention, context = _attention_and_context(
        frames=1, height=8, width=8, anchor=Rect(2, 2, 6, 6), k=4,
    )
    attention.to_out = nn.Identity()
    attention.local_fusion = _ConstantGate(0.25)
    attention.global_fusion = _ConstantGate(1.0)
    adaptive_value = torch.tensor(100.0, requires_grad=True)

    def fake_native(_attention, grid, window, _context):
        return grid[:, :, window.y0:window.y1, window.x0:window.x1]

    def fake_adaptive(_attention, grid, window, _context):
        shape = (grid.shape[0], grid.shape[1], window.height, window.width, grid.shape[-1])
        output = adaptive_value.expand(shape)
        mask = torch.zeros(
            (grid.shape[0], 1, window.height, window.width, 1), dtype=torch.bool,
        )
        mask[:, :, 0, 0] = True
        return output, mask

    monkeypatch.setattr(outpaint_model, "native_local", fake_native)
    monkeypatch.setattr(outpaint_model, "adaptive_local", fake_adaptive)
    monkeypatch.setattr(
        outpaint_model, "global_attention", lambda _attention, grid, _context: torch.zeros_like(grid),
    )
    hidden = torch.arange(1, 1 + 8 * 8 * 32, dtype=torch.float32).reshape(1, 8 * 8, 32)
    hidden.requires_grad_()
    gate = torch.zeros_like(hidden)
    token = _native_sa_gate.set((attention, gate))
    try:
        output = attention(hidden, pe=context).reshape(1, 1, 8, 8, 32)
    finally:
        _native_sa_gate.reset(token)

    write_mask = torch.zeros((1, 1, 8, 8, 1), dtype=torch.bool)
    for window in local_windows(8, 8, 4):
        write_mask[:, :, window.y0, window.x0] = True
    expanded_mask = write_mask.expand_as(output)
    input_grid = hidden.reshape_as(output)
    assert torch.equal(output[~expanded_mask], input_grid[~expanded_mask])
    expected_target = torch.lerp(
        adaptive_value.expand_as(input_grid), input_grid, 0.25,
    )
    assert torch.equal(output[expanded_mask], expected_target[expanded_mask])

    output.masked_select(~expanded_mask).sum().backward()
    assert adaptive_value.grad is None or adaptive_value.grad == 0
    assert torch.equal(hidden.grad.reshape_as(output)[~expanded_mask], torch.ones_like(output[~expanded_mask]))
    assert torch.count_nonzero(hidden.grad.reshape_as(output)[expanded_mask]) == 0


def test_adaptive_c2_c4_lora_r2_and_fusions_receive_gradients() -> None:
    torch.manual_seed(47)
    attention, context = _attention_and_context(
        frames=1, height=12, width=12, anchor=Rect(0, 0, 4, 4), k=3,
    )
    # This real Local is far enough to require coarse Bridge context.
    far = adaptive_layout(Rect(0, 0, 4, 4), Rect(8, 8, 12, 12), 12, 12, 4)
    assert any(cell.scale == 4 and cell.role == BRIDGE for cell in far.cells)
    hidden = torch.randn(1, 12 * 12, 32)
    gate = torch.randn_like(hidden).add_(0.25)
    token = _native_sa_gate.set((attention, gate))
    try:
        output = attention(hidden, pe=context)
    finally:
        _native_sa_gate.reset(token)
    output.square().mean().backward()

    assert all(
        _nonzero_finite_grad(module.lora_B.weight)
        for module in attention.adaptive_lora.values()
    )
    # PEFT zero-B initialization makes first-step A gradients zero; B must learn.
    assert all(module.lora_A.weight.grad is not None for module in attention.adaptive_lora.values())
    assert _nonzero_finite_grad(attention.compressors["2"].conv.weight)
    assert _nonzero_finite_grad(attention.compressors["4"].conv.weight)
    assert _nonzero_finite_grad(attention.decompressors["2"].refine.weight)
    for fusion in (attention.local_fusion, attention.global_fusion):
        assert _nonzero_finite_grad(fusion.beta)
        assert _nonzero_finite_grad(fusion.gamma)


def test_direct_adaptive_path_alone_trains_pure_bridge_c4_context() -> None:
    torch.manual_seed(53)
    anchor = Rect(0, 0, 4, 4)
    far_target = Rect(8, 8, 12, 12)
    attention, context = _attention_and_context(
        frames=1, height=12, width=12, anchor=anchor, k=3,
    )
    layout = adaptive_layout(anchor, far_target, 12, 12, 4)
    assert any(cell.scale == 4 and cell.role == BRIDGE for cell in layout.cells)

    grid = torch.randn(1, 1, 12, 12, 32)
    output, write_mask = adaptive_local_attention.adaptive_local(
        attention, grid, far_target, context,
    )
    assert bool(write_mask.any())
    output.square().mean().backward()

    assert all(
        _nonzero_finite_grad(module.lora_B.weight)
        for module in attention.adaptive_lora.values()
    )
    assert _nonzero_finite_grad(attention.compressors["2"].conv.weight)
    assert _nonzero_finite_grad(attention.compressors["4"].conv.weight)
    assert _nonzero_finite_grad(attention.decompressors["2"].refine.weight)


def _native_positions(frames: int, height: int, width: int) -> torch.Tensor:
    result = torch.empty(1, 3, frames * height * width, 2)
    index = 0
    for frame in range(frames):
        for y in range(height):
            for x in range(width):
                result[0, 0, index] = torch.tensor([frame / 24, (frame + 1) / 24])
                result[0, 1, index] = torch.tensor([y, y + 1])
                result[0, 2, index] = torch.tensor([x, x + 1])
                index += 1
    return result


def test_promoted_cells_keep_v2_nominal_pma_and_native_time() -> None:
    frames, height, width = 2, 8, 8
    support = Rect(0, 0, 4, 8)
    roles = [BRIDGE] * 32
    roles[4] = TARGET
    cells = build_adaptive_multires_tokens(support, tuple(roles), width)
    context = ViewContext(
        CanvasGeometry(frames, height, width, (Rect(2, 2, 6, 6),)),
        _native_positions(frames, height, width),
        torch.tensor([0.5]),
        (),
        RopeSettings(32, 2, 10_000.0, (20, 2048, 2048), True, False, None, 1000),
    )
    mapped = target_query_positions(context, cells, query_scale=2)

    for frame in range(frames):
        for cell_index, cell in enumerate(cells):
            output_index = frame * len(cells) + cell_index
            expected_y = cell.y / 2 + 0.5
            expected_x = cell.x / 2 + 0.5
            assert torch.equal(mapped[0, 1, output_index], torch.tensor([expected_y, expected_y]))
            assert torch.equal(mapped[0, 2, output_index], torch.tensor([expected_x, expected_x]))
            representative = frame * height * width + cell.indices[0]
            assert torch.equal(mapped[0, 0, output_index], context.positions[0, 0, representative])


def test_train_and_inference_architecture_metadata_are_identical() -> None:
    train = yaml.safe_load((ROOT / "configs/train_4k.yaml").read_text())
    inference = yaml.safe_load((ROOT / "configs/inference_4k.yaml").read_text())
    train_arch = dict(train["outpainting"]["architecture"])
    inference_arch = dict(inference["architecture"])
    inference_arch["k"] = inference_arch.pop("local_partition_k")
    inference_arch["scale"] = inference_arch.pop("compression_scale")

    assert train_arch == inference_arch
    assert train_arch["architecture_version"] == 8
    ArchitectureConfig(**train_arch)
