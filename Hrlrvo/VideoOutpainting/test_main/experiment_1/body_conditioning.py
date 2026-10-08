"""Experiment-only suffix conditioning and endpoint protection for 9+64+8."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EndpointCapture:
    known_mask: torch.Tensor | None = None
    suffix_latent_sha256: str | None = None
    anchor_rect: Any = None


def _tensor_sha256(value: torch.Tensor) -> str:
    import hashlib

    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


class EndpointConditioner:
    """Patch one inference module while preserving its canonical target grid."""

    def __init__(self, inference: Any, runner: Any, reference_handle: Any) -> None:
        self.inference = inference
        self.runner = runner
        self.reference_handle = reference_handle
        self.original_condition = inference._condition_latents
        self.original_known = inference._known_mask
        self.original_denoise = runner._run_denoising
        self.capture = EndpointCapture()
        self.suffix_context: torch.Tensor | None = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("Endpoint conditioner is already installed")

        def condition(*args, **kwargs):
            values, rect = self.original_condition(*args, **kwargs)
            if self.suffix_context is None:
                raise RuntimeError("A phased suffix context is required for every body")
            boundary = args[2] if len(args) > 2 else kwargs["boundary"]
            vae_encoder = args[6] if len(args) > 6 else kwargs["vae_encoder"]
            runner = args[7] if len(args) > 7 else kwargs["runner"]
            device = args[8] if len(args) > 8 else kwargs["device"]
            suffix_input = self.inference._reflect_pad_rgb(
                self.suffix_context,
                top=boundary.top,
                bottom=boundary.encoded_height - boundary.bottom,
                left=boundary.left,
                right=boundary.encoded_width - boundary.right,
            )
            suffix_all = self.inference._encode_condition_rgb(
                suffix_input, vae_encoder, runner, device
            ).to(device=values.device, dtype=values.dtype)
            if suffix_all.ndim != 5 or suffix_all.shape[2] != 5:
                raise RuntimeError(
                    f"The 33-frame suffix context must encode to five latent frames, got {tuple(suffix_all.shape)}"
                )
            suffix = suffix_all[:, :, -1:]
            if suffix.shape[:2] != values.shape[:2] or suffix.shape[-2:] != values.shape[-2:]:
                raise RuntimeError("Suffix latent does not match the 81-frame target canvas")
            # The original condition has already reasserted the separately encoded
            # source Anchor over the left prefix. Preserve that same Anchor at the
            # right endpoint while replacing only its generated exterior.
            anchor_last = values[
                :, :, -1:, rect.y0:rect.y1, rect.x0:rect.x1
            ].clone()
            values[:, :, -1:] = suffix
            values[:, :, -1:, rect.y0:rect.y1, rect.x0:rect.x1] = anchor_last
            self.capture.suffix_latent_sha256 = _tensor_sha256(suffix)
            self.capture.anchor_rect = rect
            return values, rect

        def known(**kwargs):
            mask = self.original_known(**kwargs)
            if mask.ndim != 4 or mask.shape[1] != 11:
                raise RuntimeError(f"9+64+8 requires an 11-frame latent mask, got {tuple(mask.shape)}")
            mask[:, -1] = 1.0
            self.capture.known_mask = mask.detach().to(device="cpu", dtype=torch.bool).unsqueeze(-1)
            return mask

        def denoise(*args, **kwargs):
            if self.capture.known_mask is None:
                raise RuntimeError("Known endpoint/Anchor mask was not captured before denoising")
            with self.reference_handle.target_scope(self.capture.known_mask):
                return self.original_denoise(*args, **kwargs)

        self.inference._condition_latents = condition
        self.inference._known_mask = known
        self.runner._run_denoising = denoise
        self._installed = True

    @contextmanager
    def body(self, suffix_context: torch.Tensor) -> Iterator[EndpointCapture]:
        if not self._installed:
            raise RuntimeError("Install the endpoint conditioner before sampling")
        if self.suffix_context is not None:
            raise RuntimeError("Nested body conditioning is unsupported")
        if tuple(suffix_context.shape) != (33, 3, 1080, 1920):
            raise ValueError(f"Suffix context must be [33,3,1080,1920], got {tuple(suffix_context.shape)}")
        self.suffix_context = suffix_context
        self.capture = EndpointCapture()
        try:
            yield self.capture
        finally:
            self.suffix_context = None

    def remove(self) -> None:
        if not self._installed:
            return
        self.inference._condition_latents = self.original_condition
        self.inference._known_mask = self.original_known
        self.runner._run_denoising = self.original_denoise
        self.suffix_context = None
        self._installed = False


def endpoint_mask_contract(mask: torch.Tensor, anchor_rect: Any) -> None:
    """Validate union(anchor, left two slices, right one slice)."""
    if tuple(mask.shape[:2]) != (1, 11) or mask.ndim != 5 or mask.shape[-1] != 1:
        raise ValueError("Endpoint mask must be [1,11,H,W,1]")
    grid = mask[..., 0].bool()
    if not bool(grid[:, :2].all()) or not bool(grid[:, -1:].all()):
        raise ValueError("Both endpoint conditions must be full-canvas known")
    if not bool(grid[:, 2:-1, anchor_rect.y0:anchor_rect.y1, anchor_rect.x0:anchor_rect.x1].all()):
        raise ValueError("Original Anchor must remain known through generated latent times")
    outside = grid[:, 2:-1].clone()
    outside[:, :, anchor_rect.y0:anchor_rect.y1, anchor_rect.x0:anchor_rect.x1] = False
    if bool(outside.any()):
        raise ValueError("Middle exterior tokens must remain generated")
