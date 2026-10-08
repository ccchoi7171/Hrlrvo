from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

MODULE_PATH = Path(__file__).parents[1] / "experiment_1" / "reference_memory.py"
SPEC = importlib.util.spec_from_file_location("experiment_1_reference_memory", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ZeroDelta(nn.Module):
    def forward(self, value):
        return torch.zeros_like(value)


class MeanCompressor(nn.Module):
    def forward(self, grid, cells):
        values = []
        flat = grid.flatten(2, 3)
        for cell in cells:
            values.append(flat[:, :, list(cell.indices)].mean(dim=2))
        return torch.stack(values, dim=2)


class CaptureAttention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.scale = 4
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.compressors = nn.ModuleDict({"4": MeanCompressor()})
        self.shared_outpainting_lora = nn.ModuleDict({"k": ZeroDelta(), "v": ZeroDelta()})
        self.global_lora = nn.ModuleDict({"k": ZeroDelta(), "v": ZeroDelta()})

    def forward(self, x, *, pe):
        return x


class FakeTransformer(nn.Module):
    def __init__(self, blocks=3):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [nn.ModuleDict({"attn1": CaptureAttention()}) for _ in range(blocks)]
        )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_four_artifacts_are_fixed_hashed_and_separately_encoded(tmp_path):
    refs = []
    for index in range(4):
        path = tmp_path / f"reference_{index}.png"
        path.write_bytes(f"image-{index}".encode())
        refs.append(MODULE.ReferenceArtifact(str(path), digest(path), index * 16))
    validated = MODULE.validate_reference_artifacts(refs)
    assert len(validated) == 4
    bad = list(refs)
    bad[3] = MODULE.ReferenceArtifact(bad[3].path, bad[3].sha256, 48, False)
    with pytest.raises(ValueError, match="own one-frame"):
        MODULE.validate_reference_artifacts(bad)


def test_reference_rope_repeats_four_spatial_lattices_per_target_time():
    # Split RoPE token axis is 2: [B,H,T*N,D].
    component = torch.arange(2 * 3, dtype=torch.float32).reshape(1, 1, 6, 1)
    cos, sin = MODULE._repeat_reference_rope(
        (component, component + 100), frames=2, spatial_tokens=3,
        reference_count=4, split_rope=True,
    )
    assert cos.shape == (1, 1, 24, 1)
    assert cos.flatten().tolist() == [0, 1, 2] * 4 + [3, 4, 5] * 4
    assert torch.equal(sin - cos, torch.full_like(cos, 100))


def test_clean_stream_caches_every_block_on_cpu_and_is_immutable():
    from ltx_outpaint.geometry import CanvasGeometry, Rect

    transformer = FakeTransformer().eval()
    handle = MODULE.install_reference_memory(transformer)
    hidden = torch.randn(4, 16, 8)
    context = SimpleNamespace(
        geometry=CanvasGeometry(1, 4, 4, (Rect(1, 1, 3, 3),)),
        sigma=torch.zeros(4),
    )

    def run_clean_stream():
        value = hidden
        for block in transformer.transformer_blocks:
            value = block.attn1(value, pe=context)
        return value

    report = handle.build_cache(run_clean_stream)
    assert report["blocks"] == 3
    assert report["reference_count"] == 4
    assert handle.cache_bytes > 0
    snapshot = handle.cache[0].key.clone()
    hidden.add_(1000)
    assert handle.cache[0].key.device.type == "cpu"
    assert torch.equal(handle.cache[0].key, snapshot)


def test_generated_mask_preserves_anchor_and_explicit_endpoint_tokens():
    from ltx_outpaint.geometry import CanvasGeometry, Rect

    transformer = FakeTransformer(1).eval()
    handle = MODULE.install_reference_memory(transformer)
    grid = torch.zeros(1, 3, 4, 4, 8)
    context = SimpleNamespace(geometry=CanvasGeometry(3, 4, 4, (Rect(1, 1, 3, 3),)))
    generated = handle._generated_mask(context, grid)
    assert not generated[:, :, 1:3, 1:3].any()
    assert generated[:, :, 0, 0].all()

    known = torch.zeros(1, 3, 4, 4, 1, dtype=torch.bool)
    known[:, (0, 2)] = True
    handle._known_mask = known
    generated = handle._generated_mask(context, grid)
    assert not generated[:, (0, 2)].any()
    assert generated[:, 1].all()
    expanded = handle._generated_mask(context, grid.expand(3, -1, -1, -1, -1))
    assert expanded.shape == (3, 3, 4, 4, 1)
    assert not expanded[:, (0, 2)].any()


def test_alpha_zero_returns_original_global_result_bit_exact():
    transformer = FakeTransformer(1).eval()
    block = transformer.transformer_blocks[0]
    handle = MODULE.ReferenceMemoryHandle(transformer, alpha=0.0)
    sentinel = torch.randn(1, 1, 4, 4, 8)
    handle._original_global = lambda attention, grid, context: sentinel
    handle._active = True
    actual = handle._reference_global(block.attn1, torch.randn_like(sentinel), object())
    assert actual.data_ptr() == sentinel.data_ptr()
