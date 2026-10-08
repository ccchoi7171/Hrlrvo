"""CPU-only C4 contract tests; no LTX forward, training, or optimizer steps."""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from ltx_core.model.transformer.attention import Attention
from torch import nn

from ltx_outpaint.compression import SpatialCompressor
from ltx_outpaint.geometry import (
    BRIDGE,
    TARGET,
    Rect,
    adaptive_layout,
    cells_for_rect,
    global_cells,
    local_windows,
)
from ltx_outpaint.global_attention import global_attention
from ltx_outpaint.model import ArchitectureConfig, MultiViewSelfAttention


@pytest.mark.parametrize("scale", [2, 4])
def test_full_cell_learned_convolution_and_input_order_are_preserved(scale):
    torch.manual_seed(23)
    compressor = SpatialCompressor(3, scale).double()
    with torch.no_grad():
        compressor.conv.weight.normal_()
        compressor.conv.bias.normal_()
    grid = torch.randn(2, 3, 8, 8, 3, dtype=torch.float64)
    cells = tuple(reversed(cells_for_rect(Rect(0, 0, 8, 8), 8, scale, BRIDGE)))
    actual = compressor(grid, cells)
    expected = []
    for cell in cells:
        patch = grid[:, :, cell.y:cell.y + scale, cell.x:cell.x + scale]
        patch = patch.permute(0, 1, 4, 2, 3).reshape(-1, 3, scale, scale)
        expected.append(compressor.conv(patch).reshape(2, 3, 3))
    torch.testing.assert_close(actual, torch.stack(expected, dim=2), rtol=1e-12, atol=1e-12)
    assert actual.shape == (2, 3, len(cells), 3)


def test_canonical_global_c4_grid_is_exact_full_17_by_30():
    cells = global_cells(68, 120, 4)
    assert len(cells) == 17 * 30
    coverage = Counter(index for cell in cells for index in cell.indices)
    assert set(coverage) == set(range(68 * 120))
    assert set(coverage.values()) == {1}
    assert all(cell.valid_count == 16 and cell.offsets == tuple(range(16)) for cell in cells)
    compressor = SpatialCompressor(1, 4)
    grid = torch.arange(68 * 120, dtype=torch.float32).reshape(1, 1, 68, 120, 1)
    actual = compressor(grid, cells).reshape(17, 30)
    expected = grid.reshape(17, 4, 30, 4).mean(dim=(1, 3))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("height,width", [(5, 5), (17, 20), (67, 120), (68, 119), (1, 4)])
def test_global_nondivisible_geometry_is_rejected_before_cells(height, width):
    with pytest.raises(ValueError, match="divisible"):
        global_cells(height, width, 4)


def test_invalid_global_input_fails_before_compressor_attention_or_restore():
    # No projection/compressor/R4 members are supplied: geometry must reject
    # before accessing any of them. This invokes no neural forward operation.
    with pytest.raises(ValueError, match="divisible"):
        global_attention(SimpleNamespace(scale=4), torch.ones(1, 1, 5, 8, 1), None)


@pytest.mark.parametrize("scale", [2, 4])
def test_rectangular_cell_builder_rejects_remainder(scale):
    with pytest.raises(ValueError, match="divisible"):
        cells_for_rect(Rect(0, 0, scale + 1, scale), scale, scale, BRIDGE)


@pytest.mark.parametrize("malformation", ["partial", "masked", "duplicate", "bounds", "outside", "scale"])
def test_compressor_rejects_non_full_or_noncanonical_support_before_conv(malformation):
    cell = global_cells(8, 8, 4)[0]
    if malformation == "partial":
        cell = replace(cell, indices=cell.indices[:12], offsets=cell.offsets[:12], bounds=(0, 0, 3, 4))
    elif malformation == "masked":
        cell = replace(cell, offsets=(0,) * 16)
    elif malformation == "duplicate":
        cell = replace(cell, indices=(0,) * 16)
    elif malformation == "bounds":
        cell = replace(cell, bounds=(0, 0, 3, 4))
    elif malformation == "outside":
        cell = replace(cell, y=6, bounds=(6, 0, 10, 4))
    else:
        cell = replace(cell, scale=2)
    compressor = SpatialCompressor(1, 4)
    calls = []
    handle = compressor.conv.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises(ValueError):
            compressor(torch.ones(1, 1, 8, 8, 1), (cell,))
    finally:
        handle.remove()
    assert calls == []


@pytest.mark.parametrize("scale", [2, 4])
def test_empty_cell_list_emits_no_fake_token(scale):
    compressor = SpatialCompressor(2, scale)
    assert compressor(torch.ones(1, 3, 8, 8, 2), ()).shape == (1, 3, 0, 2)


def _block(index):
    native = Attention(query_dim=8, heads=2, dim_head=4)
    native.requires_grad_(False)
    # Construction only: this module's attention forward is never called.
    return MultiViewSelfAttention(native, ArchitectureConfig(), block_index=index)


def test_per_block_c4_ownership_optimizer_selection_and_state_dict():
    blocks = nn.ModuleList([_block(0), _block(1)])
    for block in blocks:
        adaptive_c4 = block.compressors[str(block.scale)]
        global_c4 = block.compressors["4"]
        assert adaptive_c4 is global_c4
        assert adaptive_c4.conv.weight is global_c4.conv.weight
        assert adaptive_c4.conv.bias is global_c4.conv.bias
        assert block.compressors["2"] is not global_c4
        assert block.compressors["2"].conv.weight is not global_c4.conv.weight
        assert global_c4.conv.kernel_size == global_c4.conv.stride == (4, 4)
        assert global_c4.conv.padding == (0, 0) and global_c4.conv.groups == 8
        assert torch.equal(global_c4.conv.weight, torch.full_like(global_c4.conv.weight, 1 / 16))
        assert torch.count_nonzero(global_c4.conv.bias) == 0
    assert blocks[0].compressors["4"].conv.weight is not blocks[1].compressors["4"].conv.weight
    named = list(blocks.named_parameters(remove_duplicate=False))
    assert len(named) == len({id(parameter) for _, parameter in named})
    trainable = [parameter for parameter in blocks.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)  # Selection only; no step or optimizer state allocation.
    selected = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(selected) == len({id(p) for p in selected})
    assert {id(p) for p in selected} == {id(p) for p in trainable}
    assert not optimizer.state
    for i, block in enumerate(blocks):
        c4 = block.compressors["4"]
        assert sum(p is c4.conv.weight for p in selected) == 1
        assert sum(p is c4.conv.bias for p in selected) == 1
        assert not any(p is block.to_q.weight for p in selected)
        keys = [key for key in blocks.state_dict() if key.startswith(f"{i}.compressors.4.")]
        assert set(keys) == {f"{i}.compressors.4.conv.weight", f"{i}.compressors.4.conv.bias"}


def test_same_c4_primitive_consumers_have_same_values_and_additive_gradients():
    torch.manual_seed(51)
    block = _block(0)
    c4 = block.compressors["4"].double()
    with torch.no_grad():
        c4.conv.weight.normal_()
        c4.conv.bias.normal_()
    grid = torch.randn(1, 2, 12, 12, 8, dtype=torch.float64)
    layout = adaptive_layout(Rect(0, 0, 4, 4), Rect(8, 8, 12, 12), 12, 12)
    adaptive_cells = tuple(cell for cell in layout.cells if cell.scale == 4)
    assert adaptive_cells and all(cell.pixel_roles == (BRIDGE,) * 16 for cell in adaptive_cells)
    global_layout = global_cells(12, 12, 4)
    adaptive = block.compressors[str(block.scale)](grid, adaptive_cells)
    global_ = block.compressors["4"](grid, global_layout)
    indices = [next(i for i, g in enumerate(global_layout) if g.indices == a.indices) for a in adaptive_cells]
    torch.testing.assert_close(adaptive, global_[:, :, indices], rtol=1e-12, atol=1e-12)
    # Small compressor-only consumer losses, NOT a model/production training loss.
    parameters = tuple(c4.parameters())
    a_loss, g_loss = adaptive.square().mean(), global_.square().mean()
    a_grad = torch.autograd.grad(a_loss, parameters, retain_graph=True)
    g_grad = torch.autograd.grad(g_loss, parameters, retain_graph=True)
    combined = torch.autograd.grad(a_loss + g_loss, parameters)
    for a, g, total in zip(a_grad, g_grad, combined, strict=True):
        assert torch.isfinite(a).all() and torch.count_nonzero(a)
        assert torch.isfinite(g).all() and torch.count_nonzero(g)
        torch.testing.assert_close(total, a + g, rtol=1e-12, atol=1e-12)
    assert all(p.grad is None for p in block.parameters() if not p.requires_grad)


@pytest.mark.parametrize("k,expected", [(4, (3616, 224, 1620, 4)), (5, (4456, 362, 1590, 1))])
def test_all_41_adaptive_layouts_remain_full_and_lossless(k, expected):
    anchor = Rect(17, 30, 51, 90)
    totals = [0, 0, 0, 0]
    for target in local_windows(68, 120, k):
        layout = adaptive_layout(anchor, target, 68, 120)
        if not layout.target_count:
            totals[3] += 1
            assert layout.rectangle is None and not layout.cells
            continue
        support = layout.rectangle
        assert support.height % 4 == support.width % 4 == 0
        coverage = Counter(index for cell in layout.cells for index in cell.indices)
        expected_support = {y * 120 + x for y in range(support.y0, support.y1)
                            for x in range(support.x0, support.x1)}
        assert set(coverage) == expected_support and set(coverage.values()) == {1}
        assert all(cell.valid_count == cell.scale**2 and cell.offsets == tuple(range(cell.scale**2))
                   for cell in layout.cells)
        assert all(cell.scale == 2 and TARGET in cell.pixel_roles for cell in layout.cells[:layout.target_count])
        assert all(cell.pixel_roles == (BRIDGE,) * 16 for cell in layout.cells if cell.scale == 4)
        targets = {(y - target.y0) * target.width + x - target.x0
                   for y in range(target.y0, target.y1) for x in range(target.x0, target.x1)
                   if not anchor.contains(y, x)}
        assert set(layout.write_target_indices) == targets
        assert len(layout.write_target_indices) == len(targets)
        totals[0] += sum(cell.scale == 2 for cell in layout.cells)
        totals[1] += sum(cell.scale == 4 for cell in layout.cells)
        totals[2] += layout.target_count
    assert tuple(totals) == expected
