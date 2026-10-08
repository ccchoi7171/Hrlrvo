from __future__ import annotations

import inspect

import torch

from ltx_outpaint import inference


def test_first_ten_steps_use_fixed_global_boost() -> None:
    assert [inference._early_global_boost_scale(index) for index in range(10)] == [
        1.4
    ] * 10
    assert inference._early_global_boost_metadata(True) == {
        "enabled": True,
        "start_scale": 1.4,
        "duration_steps": 10,
        "curve": "constant_then_legacy",
    }


def test_step_ten_and_later_bypass_boost_gate() -> None:
    runtime = inference._EarlyGlobalBoostRuntime(tuple(float(i) for i in range(30)))

    assert runtime.active_scale_for_sigma(torch.tensor([0.0])) == 1.4
    assert runtime.active_scale_for_sigma(torch.tensor([9.0])) == 1.4
    assert runtime.active_scale_for_sigma(torch.tensor([10.0])) is None
    assert runtime.active_scale_for_sigma(torch.tensor([29.0])) is None


def test_boost_remains_explicit_and_defaults_off() -> None:
    assert inference._early_global_boost_metadata(False) == {"enabled": False}
    assert inspect.signature(inference._load_models).parameters[
        "early_global_boost"
    ].default is False
    assert inspect.signature(inference.GeometryValidationRunner.create).parameters[
        "early_global_boost"
    ].default is False
