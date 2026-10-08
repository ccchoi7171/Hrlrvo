"""Delta-only LTX/PEFT-style Linear LoRA, without duplicating frozen modules."""

import math
from types import MethodType

import torch
from torch import nn


class LinearLoRA(nn.Module):
    """Bias-free Kaiming A / zero B with alpha/rank scaling and FP32 masters."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        if not isinstance(rank, int) or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive and finite")
        self.lora_A = nn.Linear(in_features, rank, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, out_features, bias=False, dtype=torch.float32)
        self.multiplier = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Native autocast still determines Linear compute precision. No frozen
        # base is stored here; only the low-rank residual owns parameters.
        return self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.multiplier


def project_with_loras(base_projection, x, loras=()):
    """base(x) + sum(delta(x)), never delta(delta(x)); restore base result dtype."""
    result = base_projection(x)
    dtype = result.dtype
    for lora in loras:
        result = result + lora(x)
    return result.to(dtype)


def _forward_with_lora(self, x):
    return project_with_loras(
        lambda value: nn.Linear.forward(self, value), x,
        (getattr(self, self._hrlrvo_lora_name),),
    )


def install_linear_lora(linear: nn.Linear, name: str, rank: int, alpha: float):
    """Add a residual to the existing Linear; preserve object, weight/bias keys.

    Used only on shared O and the two shared native FFN Linear targets. The
    enclosing Sequential/GELU/FFN still executes its unchanged native forward.
    """
    if type(linear) is not nn.Linear or hasattr(linear, "_hrlrvo_lora_name"):
        raise TypeError("LoRA installation requires an unwrapped native Linear")
    if any(parameter.requires_grad for parameter in linear.parameters()):
        raise ValueError("Freeze the pretrained Linear before installing its LoRA")
    delta = LinearLoRA(linear.in_features, linear.out_features, rank, alpha)
    linear.add_module(name, delta.to(device=linear.weight.device))
    linear._hrlrvo_lora_name = name
    linear.forward = MethodType(_forward_with_lora, linear)
