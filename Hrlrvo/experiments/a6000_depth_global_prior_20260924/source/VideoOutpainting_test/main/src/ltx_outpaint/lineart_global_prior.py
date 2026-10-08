"""Inference-only structural bias for the Global outpainting attention view.

The prior is deliberately scoped with a ContextVar so importing this module, or
constructing a disabled prior, cannot change the model's normal attention path.
It owns no learned parameters and never changes latent/prefix conditioning.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
from typing import Iterator

import torch

_CURRENT: ContextVar[tuple["LineArtGlobalPrior", "_ChunkState"] | None] = ContextVar(
    "ltx_outpaint_lineart_global_prior", default=None
)


@dataclass
class _ChunkState:
    chunk_index: int
    prefix_latent_frames: int
    cache: dict = field(default_factory=dict)
    calls: int = 0
    active_calls: int = 0
    calls_by_step: dict[int, int] = field(default_factory=dict)
    bias_min: float | None = None
    bias_max: float | None = None
    original_backend: str | None = None
    biased_backend: str | None = None
    sigma_context: object | None = None
    sigma_step: int | None = None


class LineArtGlobalPrior:
    """Fixed C4 line-strength prior used only by Global self-attention.

    ``sigma_by_step`` must contain the exact 30 scheduler sigmas used by the
    runner. A call is rejected rather than silently assigned to the wrong step
    when its batch contains mixed sigmas or no scheduler match.
    """

    def __init__(
        self,
        token_strength: torch.Tensor,
        sigma_by_step: tuple[float, ...],
        enabled: bool = True,
        bias_scale: float = 0.5,
    ) -> None:
        if not isinstance(token_strength, torch.Tensor):
            raise TypeError("token_strength must be a CPU float32 torch.Tensor")
        if token_strength.device.type != "cpu" or token_strength.dtype != torch.float32:
            raise TypeError("token_strength must be CPU float32")
        if token_strength.shape != (9, 15):
            raise ValueError("FHD Global C4 token_strength must have shape [9,15]")
        if not torch.isfinite(token_strength).all():
            raise ValueError("token_strength contains NaN/Inf")
        if (token_strength < 0).any() or (token_strength > 1).any():
            raise ValueError("token_strength must be normalized to [0,1]")
        if len(sigma_by_step) != 30 or not all(math.isfinite(float(x)) for x in sigma_by_step):
            raise ValueError("sigma_by_step must contain 30 finite scheduler values")
        if len(set(float(x) for x in sigma_by_step)) != len(sigma_by_step):
            raise ValueError("sigma_by_step values must be unique")
        if not math.isfinite(float(bias_scale)) or bias_scale < 0:
            raise ValueError("bias_scale must be finite and nonnegative")
        self.token_strength = token_strength.detach().clone().contiguous()
        self._has_lines = bool(torch.count_nonzero(self.token_strength))
        self.sigma_by_step = tuple(float(x) for x in sigma_by_step)
        self.enabled = bool(enabled)
        self.bias_scale = float(bias_scale)
        self._last_summary: dict = {
            "chunk_index": None,
            "enabled": self.enabled,
            "bias_scale": self.bias_scale,
            "calls": 0,
            "active_calls": 0,
        }

    @contextmanager
    def chunk_scope(self, chunk_index: int, prefix_latent_frames: int = 3) -> Iterator["LineArtGlobalPrior"]:
        if type(chunk_index) is not int or chunk_index < 0:
            raise ValueError("chunk_index must be a nonnegative integer")
        if type(prefix_latent_frames) is not int or prefix_latent_frames < 0:
            raise ValueError("prefix_latent_frames must be a nonnegative integer")
        if _CURRENT.get() is not None:
            raise RuntimeError("LineArtGlobalPrior chunk scopes cannot be nested")
        state = _ChunkState(chunk_index, prefix_latent_frames)
        token = _CURRENT.set((self, state))
        try:
            yield self
        finally:
            # Never retain device tensors (or an autograd graph) across chunks.
            state.cache.clear()
            self._last_summary = self._state_summary(state)
            _CURRENT.reset(token)

    def summary(self) -> dict:
        current = _CURRENT.get()
        if current is not None and current[0] is self:
            return self._state_summary(current[1])
        return dict(self._last_summary)

    def _state_summary(self, state: _ChunkState) -> dict:
        return {
            "chunk_index": state.chunk_index,
            "enabled": self.enabled,
            "bias_scale": self.bias_scale,
            "schedule": {
                "steps_0_9": self.bias_scale,
                "steps_10_19": self.bias_scale * 0.6,
                "steps_20_29": self.bias_scale * 0.2,
            },
            "calls": state.calls,
            "active_calls": state.active_calls,
            "calls_by_step": dict(sorted(state.calls_by_step.items())),
            "bias_min": state.bias_min,
            "bias_max": state.bias_max,
            "configured_original_backend": state.original_backend,
            "configured_biased_backend": state.biased_backend,
            "runtime_kernel_note": "configured callable identity only; fused runtime kernel requires profiler evidence",
        }

    def _step_for_sigma(self, sigma: torch.Tensor) -> int:
        values = sigma.detach().float().reshape(-1).cpu()
        if values.numel() == 0 or not torch.isfinite(values).all():
            raise ValueError("Global prior received empty or nonfinite sigma")
        first = float(values[0])
        # CFG duplicates are valid; different sigma values in one batch are not.
        tolerance = max(1e-7, abs(first) * 2e-6)
        if not torch.allclose(values, values.new_full(values.shape, first), rtol=2e-6, atol=tolerance):
            raise ValueError("Global prior received mixed sigma values in one batch")
        differences = [abs(first - candidate) for candidate in self.sigma_by_step]
        step = min(range(len(differences)), key=differences.__getitem__)
        candidate = self.sigma_by_step[step]
        if differences[step] > max(1e-7, abs(candidate) * 2e-6):
            raise ValueError(f"Global prior sigma {first!r} is not in the configured scheduler")
        return step

    def _lambda_for_step(self, step: int) -> float:
        # Immutable zero-based 30-step inference schedule.
        if step <= 9:
            return self.bias_scale
        if step <= 19:
            return self.bias_scale * 0.6
        return self.bias_scale * 0.2


def current_prior(context):
    """Return an active call descriptor, or ``None`` for the exact old route."""
    bound = _CURRENT.get()
    if bound is None:
        return None
    prior, state = bound
    state.calls += 1
    # C0 defines the reference and is intentionally byte-for-byte baseline.
    if (not prior.enabled or prior.bias_scale == 0.0 or state.chunk_index == 0
            or not prior._has_lines):
        return None
    # Every transformer block in one denoiser forward shares this immutable
    # ViewContext. Hold the object itself (not only id) to avoid id reuse and
    # synchronize its GPU sigma only once per forward.
    if state.sigma_context is context:
        step = state.sigma_step
    else:
        step = prior._step_for_sigma(context.sigma)
        state.sigma_context = context
        state.sigma_step = step
    state.calls_by_step[step] = state.calls_by_step.get(step, 0) + 1
    strength = prior._lambda_for_step(step)
    if strength == 0.0:
        return None
    state.active_calls += 1
    return prior, state, step, strength


def _backend_name(callable_) -> str:
    label = getattr(callable_, "label", None)
    return str(label) if label is not None else type(callable_).__name__


def build_attention_bias(prior, state, context, cells, dtype, device, strength):
    """Build additive [B,N,N] bias and compressed-row eligibility.

    Flattening is time-major, identical to ``compressed.reshape(B,-1,D)``.
    A coarse query cell is eligible when it contains any generated pixel, while
    every spatial/time token remains available as a key.
    """
    g = context.geometry
    if g.frames <= state.prefix_latent_frames:
        raise ValueError("prefix_latent_frames leaves no newly generated latent slices")
    if len(cells) != 9 * 15 or (g.height, g.width) != (36, 60):
        raise ValueError("LineArtGlobalPrior is validated only for FHD latent 36x60 / Global C4 9x15")
    batch_size = context.sigma.reshape(-1).numel()
    key = ("bias", device.type, device.index, dtype, batch_size, g.frames, tuple(g.anchors), strength)
    cached = state.cache.get(key)
    if cached is not None:
        return cached

    spatial = prior.token_strength.flatten().to(device=device, dtype=dtype)
    temporal = spatial.repeat(g.frames)  # [t0 cells..., t1 cells...] exactly.
    query_rows = []
    for anchor in g.batch_anchors(batch_size):
        generated_cell = torch.tensor(
            [any(not anchor.contains(y, x)
                 for y in range(cell.y, cell.y + cell.scale)
                 for x in range(cell.x, cell.x + cell.scale))
             for cell in cells],
            device=device,
            dtype=torch.bool,
        )
        time_ok = torch.arange(g.frames, device=device) >= state.prefix_latent_frames
        query_rows.append((time_ok[:, None] & generated_cell[None, :]).flatten())
    eligible = torch.stack(query_rows)
    # A zero-strength query has an all-zero additive row. Keep it on the exact
    # configured unmasked backend instead of accepting masked-backend rounding.
    line_query = (spatial > 0).repeat(g.frames)
    eligible &= line_query[None, :]
    bias = strength * temporal[None, :, None] * temporal[None, None, :]
    bias = bias.expand(eligible.shape[0], -1, -1).clone()
    bias.masked_fill_(~eligible[:, :, None], 0)
    state.bias_min = min(state.bias_min if state.bias_min is not None else 0.0, float(bias.min().item()))
    state.bias_max = max(state.bias_max if state.bias_max is not None else 0.0, float(bias.max().item()))
    result = bias, eligible
    state.cache[key] = result
    return result


def build_fine_write_mask(state, context, device):
    """[B,T,H,W,1] true only for new, generated fine latent positions."""
    g = context.geometry
    batch_size = context.sigma.reshape(-1).numel()
    key = ("fine", device.type, device.index, batch_size, g.frames, g.height, g.width, tuple(g.anchors))
    cached = state.cache.get(key)
    if cached is not None:
        return cached
    masks = []
    for anchor in g.batch_anchors(batch_size):
        spatial = torch.ones(g.height, g.width, device=device, dtype=torch.bool)
        spatial[anchor.y0:anchor.y1, anchor.x0:anchor.x1] = False
        temporal = torch.arange(g.frames, device=device) >= state.prefix_latent_frames
        masks.append(temporal[:, None, None] & spatial[None, :, :])
    result = torch.stack(masks)[..., None]
    state.cache[key] = result
    return result


def record_backends(state, original, biased):
    state.original_backend = _backend_name(original)
    state.biased_backend = _backend_name(biased)
