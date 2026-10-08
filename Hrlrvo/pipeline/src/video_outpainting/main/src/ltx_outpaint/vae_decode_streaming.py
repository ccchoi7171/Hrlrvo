"""Inference-only, bounded-GPU-output variant of ValidationRunner._decode_video.

The decoder's ``tiled_decode`` generator performs the existing spatial and
temporal overlap reconstruction.  Its yields are *completed temporal regions*,
not raw tiles.  Move each completed region to CPU before requesting the next
one, rather than retaining all GPU regions for a full-video ``torch.cat``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ltx_core.types import LatentState
    from ltx_trainer.validation_runner import ValidationRunner


def decode_video_streaming(
    runner: ValidationRunner,
    video_state: LatentState,
    device: torch.device,
) -> torch.Tensor:
    """Return CPU FP32 ``[C, T, H, W]`` pixels using the legacy tile plan.

    The BF16 normalization is intentionally identical to the legacy expression
    and runs before FP32 conversion.  Only the final CPU assembly differs: it
    concatenates normalized, already-transferred temporal regions on CPU.
    This avoids the full GPU BF16 cat and full GPU FP32 output allocation.
    """
    runner._vae_decoder.to(device)
    latent = video_state.latent.to(dtype=torch.bfloat16)
    tiling_config = runner._decode_tiling_config(latent, device)

    cpu_regions: list[torch.Tensor] = []
    for region in runner._vae_decoder.tiled_decode(latent, tiling_config=tiling_config):
        # Each yielded region has already undergone the decoder's spatial and
        # temporal overlap reconstruction.  Preserve the legacy BF16 operation
        # order, then transfer only this region's FP32 values.
        normalized = ((region + 1.0) / 2.0).clamp(0.0, 1.0)
        cpu_regions.append(normalized[0].float().cpu())
        del region, normalized

    runner._vae_decoder.to("cpu")
    if not cpu_regions:
        raise RuntimeError("VAE tiled_decode produced no temporal regions")
    if len(cpu_regions) == 1:
        return cpu_regions[0]
    return torch.cat(cpu_regions, dim=1)


def decode_video_frames_streaming(
    runner: ValidationRunner,
    video_state: LatentState,
    device: torch.device,
    frame_indices: tuple[int, ...],
    *,
    metrics: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Decode and retain selected output frames as CPU FP32 ``[C, N, H, W]``.

    The current VAE exposes sequential completed temporal regions rather than
    random-access frame decoding.  Consequently it must traverse regions up to
    the greatest requested frame (frame 64 means the complete 65-frame
    temporal traversal), but unrequested pixels are neither normalized nor
    transferred to CPU nor assembled into a full-video tensor.
    """
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    if tuple(sorted(set(frame_indices))) != frame_indices or frame_indices[0] < 0:
        raise ValueError("frame_indices must be unique, sorted, non-negative integers")

    runner._vae_decoder.to(device)
    latent = video_state.latent.to(dtype=torch.bfloat16)
    tiling_config = runner._decode_tiling_config(latent, device)
    wanted = set(frame_indices)
    selected: dict[int, torch.Tensor] = {}
    temporal_cursor = 0
    yielded_regions = 0

    try:
        for region in runner._vae_decoder.tiled_decode(
            latent, tiling_config=tiling_config
        ):
            if region.ndim != 5 or region.shape[0] != 1:
                raise ValueError(
                    "VAE tiled_decode regions must have shape [1,C,T,H,W]"
                )
            yielded_regions += 1
            region_frames = int(region.shape[2])
            region_end = temporal_cursor + region_frames
            for frame_index in sorted(wanted):
                if temporal_cursor <= frame_index < region_end:
                    local_index = frame_index - temporal_cursor
                    normalized = ((region[0, :, local_index] + 1.0) / 2.0).clamp(
                        0.0, 1.0
                    )
                    selected[frame_index] = normalized.float().cpu()
                    del normalized
            temporal_cursor = region_end
            del region
    finally:
        runner._vae_decoder.to("cpu")

    missing = [index for index in frame_indices if index not in selected]
    if missing:
        raise IndexError(
            f"Requested decoded frames {missing} exceed VAE output length "
            f"{temporal_cursor}"
        )
    if metrics is not None:
        metrics.update(
            {
                "vae_temporal_frames_traversed": temporal_cursor,
                "vae_temporal_regions_traversed": yielded_regions,
                "vae_frames_retained": list(frame_indices),
                "vae_random_access_decode": False,
                "full_decoded_video_materialized": False,
            }
        )
    return torch.stack([selected[index] for index in frame_indices], dim=1)
