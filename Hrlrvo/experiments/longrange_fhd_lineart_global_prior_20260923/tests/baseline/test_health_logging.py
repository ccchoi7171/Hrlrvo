import json
import math

import pytest
import torch
from torch import nn

from ltx_outpaint.geometry import CanvasGeometry, Rect
from ltx_outpaint.health_logging import (
    GRADIENT_GROUPS,
    TrainingHealthLogger,
    parameter_health_group,
)


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, logits):
        return torch.sigmoid(logits + self.bias)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.partition_count = 2
        self.scale = 4
        self.shared_outpainting_lora = nn.Linear(2, 2, bias=False)
        self.adaptive_lora = nn.Linear(2, 2, bias=False)
        self.global_lora = nn.Linear(2, 2, bias=False)
        self.compressors = nn.ModuleDict({"2": nn.Linear(2, 2), "4": nn.Linear(2, 2)})
        self.decompressors = nn.ModuleDict({"2": nn.Linear(2, 2), "4": nn.Linear(2, 2)})
        self.local_fusion = _Gate()
        self.global_fusion = _Gate()


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _Attention()
        self.ffn_lora = nn.Linear(2, 2, bias=False)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(), _Block()])
        self.frozen_base = nn.Parameter(torch.ones(2), requires_grad=False)


def _all_trainables_loss(model):
    return sum(parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad)


def test_parameter_grouping_matches_v8_names(tmp_path):
    model = _ToyModel()
    health = TrainingHealthLogger(model, tmp_path / "metrics.jsonl", "grouping", expected_blocks=2)
    try:
        assert not health.unclassified_trainable_parameters
        assert set(health.grouped_named_parameters) == set(GRADIENT_GROUPS)
        assert all(health.grouped_named_parameters[group] for group in GRADIENT_GROUPS)
        assert parameter_health_group(
            "transformer_blocks.0.attn1.to_out.0.shared_outpainting_lora.lora_B.weight"
        ) == "op_lora"
        assert parameter_health_group("transformer_blocks.0.ff.net.0.proj.ffn_lora.lora_A.weight") == "f_lora"
        assert parameter_health_group("transformer_blocks.0.attn1.compressors.4.conv.weight") == "c4_shared"
    finally:
        health.close()


def test_gradient_health_uses_group_reductions_and_detects_frozen_violation(tmp_path):
    model = _ToyModel()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.fill_(0.25)
    health = TrainingHealthLogger(model, tmp_path / "metrics.jsonl", "grad", expected_blocks=2)
    try:
        _all_trainables_loss(model).backward()
        metrics = health.capture_pre_clip_gradients()
        assert metrics["nan_gradient_count"] == 0
        assert metrics["inf_gradient_count"] == 0
        assert metrics["frozen_parameter_grad_violation_count"] == 0
        assert metrics["zero_gradient_trainable_groups"] == []
        assert set(metrics["nonzero_gradient_trainable_groups"]) == set(GRADIENT_GROUPS)
        expected_norm = math.sqrt(sum(float((parameter.grad.float() ** 2).sum())
                                      for parameter in model.parameters() if parameter.grad is not None))
        assert metrics["grad_norm_global"] == pytest.approx(expected_norm)
        for group in GRADIENT_GROUPS:
            assert metrics[f"grad_norm_{group}"] > 0
            assert metrics["gradient_group_details"][group]["finite"]

        model.frozen_base.grad = torch.ones_like(model.frozen_base)
        assert health.capture_pre_clip_gradients()["frozen_parameter_grad_violation_count"] == 1
    finally:
        health.close()


def test_fusion_hooks_capture_actual_alpha_all_windows_and_exclude_recompute(tmp_path):
    model = _ToyModel()
    health = TrainingHealthLogger(model, tmp_path / "metrics.jsonl", "fusion", expected_blocks=2)
    try:
        assert health.set_geometry(CanvasGeometry(1, 8, 8, (Rect(2, 2, 6, 6),)), batch_size=1)
        health.begin_optimizer_step()
        # Two Local windows per block, one Global invocation per block.
        local_logits = ((0.0, math.log(4.0)), (math.log(0.25), 0.0))
        global_logits = (math.log(3.0), math.log(1.0 / 3.0))
        for block, values, global_value in zip(model.transformer_blocks, local_logits, global_logits, strict=True):
            for value in values:
                block.attn1.local_fusion(torch.full((1, 1, 4, 4, 1), value))
            block.attn1.global_fusion(torch.full((2, 3), global_value))
        health.end_forward()
        # Simulated gradient-checkpoint recomputation must not be counted.
        model.transformer_blocks[0].attn1.local_fusion(torch.full((1, 1, 4, 4, 1), 20.0))

        metrics = health.fusion_metrics()
        assert metrics["local_native_ratio_by_block"] == pytest.approx([0.65, 0.35])
        assert metrics["local_adaptive_ratio_by_block"] == pytest.approx([0.35, 0.65])
        assert metrics["global_local_ratio_by_block"] == pytest.approx([0.75, 0.25])
        assert metrics["global_global_ratio_by_block"] == pytest.approx([0.25, 0.75])
        assert metrics["local_fusion_forward_calls_by_block"] == [2, 2]
        assert metrics["global_fusion_forward_calls_by_block"] == [1, 1]
        assert metrics["local_fusion_missing_blocks"] == []
        assert metrics["global_fusion_missing_blocks"] == []
        assert not metrics["local_fusion_nonfinite_detected"]
        assert not metrics["global_fusion_nonfinite_detected"]
        assert metrics["local_fusion_target_mask_configured"]
    finally:
        health.close()


def test_local_fusion_ratio_excludes_anchor_native_passthrough_sites(tmp_path):
    model = _ToyModel()
    health = TrainingHealthLogger(model, tmp_path / "metrics.jsonl", "masked", expected_blocks=2)
    try:
        assert health.set_geometry(CanvasGeometry(1, 8, 8, (Rect(2, 2, 6, 6),)), batch_size=1)
        health.begin_optimizer_step()
        logits = torch.zeros((1, 1, 4, 4, 1))
        # First window intersects Anchor at its bottom-right 2x2. These values
        # are Native passthrough sites and must not influence the logged alpha.
        logits[:, :, 2:, 2:] = 20
        model.transformer_blocks[0].attn1.local_fusion(logits)
        health.end_forward()
        metrics = health.fusion_metrics()
        assert metrics["local_native_ratio_by_block"][0] == pytest.approx(0.5)
        assert metrics["local_adaptive_ratio_by_block"][0] == pytest.approx(0.5)
        assert metrics["local_native_ratio_by_block"][1] is None
        assert not metrics["fusion_observer_errors"]
    finally:
        health.close()


def test_missing_geometry_observer_is_fail_safe(tmp_path, caplog):
    model = _ToyModel()
    health = TrainingHealthLogger(model, tmp_path / "metrics.jsonl", "fail-safe", expected_blocks=2)
    try:
        health.begin_optimizer_step()
        # Instrumentation errors are recorded, never raised through model forward.
        output = model.transformer_blocks[0].attn1.local_fusion(torch.zeros(1, 1, 4, 4, 1))
        assert torch.equal(output, torch.full_like(output, 0.5))
        metrics = health.fusion_metrics()
        assert metrics["fusion_observer_errors"]
        assert metrics["local_fusion_missing_blocks"] == [0, 1]
        assert "Set CanvasGeometry" in caplog.text
    finally:
        health.close()


def test_accumulation_forward_resume_jsonl_and_representative_updates(tmp_path):
    path = tmp_path / "metrics.jsonl"
    model = _ToyModel()
    health = TrainingHealthLogger(model, path, "run-1", expected_blocks=2)
    try:
        assert health.set_geometry(CanvasGeometry(1, 8, 8, (Rect(2, 2, 6, 6),)), batch_size=1)
        health.begin_optimizer_step()
        model.transformer_blocks[0].attn1.local_fusion(torch.zeros(1, 1, 4, 4, 1))
        health.end_forward()
        health.begin_forward()
        model.transformer_blocks[1].attn1.local_fusion(torch.zeros(1, 1, 4, 4, 1))
        health.end_forward()
        metrics = health.fusion_metrics()
        assert metrics["local_fusion_observed_blocks"] == [0, 1]

        before = health.snapshot_representatives()
        with torch.no_grad():
            for named in health.grouped_named_parameters.values():
                if named:
                    named[0][1].add_(1)
        updates = health.representative_updates(before)
        assert updates and all(updates.values())
        assert health.log_step({"optimizer_step": 7, "loss": 1.25})
    finally:
        health.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["run_id"] == "run-1"
    assert record["optimizer_step"] == 7
    assert record["loss"] == 1.25
    assert record["timestamp"].endswith("+00:00")


def test_update_observer_does_not_scalarize_device_indices(tmp_path):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoLocalScalar(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func != torch.ops.aten._local_scalar_dense.default
            return func(*args, **(kwargs or {}))

    model = _ToyModel()
    health = TrainingHealthLogger(model, tmp_path / 'metrics.jsonl', 'no_scalar_sync', expected_blocks=2)
    try:
        _all_trainables_loss(model).backward()
        with NoLocalScalar():
            before = health.snapshot_representatives()
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(0.01)
            updates = health.representative_updates(before)
        assert all(updates.values())
    finally:
        health.close()
