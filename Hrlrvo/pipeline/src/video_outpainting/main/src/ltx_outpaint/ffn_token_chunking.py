"""Opt-in, inference-only token batching for the native video FFN.

This wraps only ``transformer_blocks[*].ff.forward``. The native FFN (and
the LoRA-installed Linear forwards inside it) still runs on every token;
no weights, activation, attention, or audio FFN are replaced. Different GEMM
shapes may choose different CUDA kernels, so BF16 bitwise equality must be
checked before enabling this optimization for production inference.
"""

from types import MethodType
from typing import cast

import torch


def _chunked_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    chunk_tokens = cast(int, self._hrlrvo_ffn_chunk_tokens)
    native_forward = type(self).forward
    if x.shape[1] <= chunk_tokens:
        return native_forward(self, x)

    result: torch.Tensor | None = None
    for start in range(0, x.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, x.shape[1])
        # Keep the native FFN and its installed Linear LoRAs unchanged. Copying
        # the finished token slice is exact and avoids retaining FFN outputs.
        part = native_forward(self, x[:, start:end])
        if result is None:
            result = part.new_empty(x.shape)
        result[:, start:end].copy_(part)
        del part
    if result is None:
        raise RuntimeError("FFN token chunking produced no output")
    return result


def install_ffn_token_chunking(transformer: torch.nn.Module, chunk_tokens: int) -> dict[str, int]:
    """Wrap video FFNs after checkpoint/LoRA loading; leave audio FFNs intact.

    Call only on an inference model (``eval`` and ``torch.no_grad`` context).
    An all-or-nothing preflight prevents partially patched transformer blocks.
    """
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, int) or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    if transformer.training:
        raise ValueError("FFN token chunking is inference-only; call eval() first")
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None or len(blocks) == 0:
        raise TypeError("transformer must have nonempty video transformer_blocks")
    for block in blocks:
        ff = getattr(block, "ff", None)
        if not isinstance(ff, torch.nn.Module):
            raise TypeError("every video transformer block must have a native ff module")
        if hasattr(ff, "_hrlrvo_ffn_chunk_tokens"):
            raise ValueError("FFN token chunking is already installed")
        if getattr(ff.forward, "__func__", None) is not type(ff).forward:
            raise TypeError("FFN forward was already overridden; refusing to replace it")

    for block in blocks:
        ff = block.ff
        ff._hrlrvo_ffn_chunk_tokens = chunk_tokens
        ff.forward = MethodType(_chunked_forward, ff)
    return {"video_ffn_blocks": len(blocks), "chunk_tokens": chunk_tokens}
