"""Shared full-cell spatial compression and fixed spatial reconstruction.

C2/C4 use one depthwise convolution primitive on real full cells only.
Hrlrvo initializes C as an average and R's refinement as identity.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SpatialCompressor(nn.Module):
    def __init__(self, dim: int, scale: int):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(dim, dim, kernel_size=scale, stride=scale, padding=0, groups=dim)
        nn.init.constant_(self.conv.weight, 1.0 / (scale * scale))
        nn.init.zeros_(self.conv.bias)

    def forward(self, grid: torch.Tensor, cells: tuple) -> torch.Tensor:
        """[B,T,H,W,D] -> [B,T,N,D] over explicit canonical cell footprints.

        Every cell executes the same stride-s depthwise Conv2d. Adaptive and
        Global reuse this primitive and its learned weights without role-based
        modes. Partial, masked, or noncanonical cell descriptors are rejected.
        """
        b, t, h, w, d = grid.shape
        if not cells:
            return grid.new_empty(b, t, 0, d)
        s = self.scale
        for cell in cells:
            if (cell.scale != s or cell.valid_count != s * s
                    or cell.offsets != tuple(range(s * s))):
                raise ValueError(f"C{s} accepts only complete {s}x{s} cells with every kernel tap")
            if (cell.y < 0 or cell.x < 0 or cell.y + s > h or cell.x + s > w
                    or cell.bounds != (cell.y, cell.x, cell.y + s, cell.x + s)):
                raise ValueError(f"C{s} cell footprint must lie inside the current hidden canvas")
            expected = tuple((cell.y + dy) * w + cell.x + dx for dy in range(s) for dx in range(s))
            if cell.indices != expected:
                raise ValueError(f"C{s} must read exactly its full canonical spatial footprint")
        indices = torch.tensor([cell.indices for cell in cells], device=grid.device)
        values = grid.flatten(2, 3)[:, :, indices, :]
        spatial = values.permute(0, 1, 2, 4, 3).reshape(-1, d, s, s)
        return self.conv(spatial).reshape(b, t, len(cells), d).to(grid.dtype)


class SpatialDecompressor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # UltraGen: bilinear torchvision Resize then depthwise 3D refinement.
        # Hrlrvo identity initialization preserves the upsampled feature at start;
        # all temporal/spatial taps remain trainable, including off-center taps.
        self.refine = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        nn.init.zeros_(self.refine.weight)
        nn.init.ones_(self.refine.weight[:, :, 1, 1, 1])
        nn.init.zeros_(self.refine.bias)

    def forward(self, tokens, compressed_shape, target_shape):
        b, t, n, d = tokens.shape
        ch, cw = compressed_shape
        h, w = target_shape
        if n != ch * cw:
            raise ValueError("Decompression requires the exact rectangular target-token map")
        spatial = tokens.reshape(b * t, ch, cw, d).permute(0, 3, 1, 2)
        # torchvision Resize's tensor default: bilinear, align_corners=False,
        # antialias=True. Float32 also supports bf16 inputs on CPU backends.
        spatial = F.interpolate(spatial.float(), size=(h, w), mode="bilinear",
                                align_corners=False, antialias=True).to(tokens.dtype)
        volume = spatial.reshape(b, t, d, h, w).permute(0, 2, 1, 3, 4)
        return self.refine(volume).permute(0, 2, 3, 4, 1)
