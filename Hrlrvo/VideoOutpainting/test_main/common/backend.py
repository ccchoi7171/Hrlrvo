"""Experiment-only exact-dtype CPU block streaming for VideoOutpainting.

This module does not alter production inference.  It keeps the transformer and
all adapter weights in their configured dtype, places non-block video modules
on the target CUDA device, and transfers one transformer block to CUDA for its
forward before returning it to CPU.  No quantization, activation chunking, or
shape changes are performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def install_relocation_checks(
    inference_module: Any,
    original_source_repo: str,
    relocated_repo: str,
    *,
    original_base_model_path: str | None = None,
    original_base_sha256: str | None = None,
) -> None:
    """Keep native provenance/adapter checks while accepting remote paths.

    The relocated checkout is still validated by the native checker (Git HEAD,
    clean package tree, imported ``ltx_core`` location).  The original
    provenance dict is returned unchanged so adapter metadata is never rewritten.
    """
    import copy
    from pathlib import Path
    import ltx_outpaint.model as model_module
    import ltx_outpaint.outpainting_adapter as adapter_module

    native_provenance = adapter_module.checkpoint_provenance
    original_base = Path(original_base_model_path).expanduser().resolve() if original_base_model_path else None

    def provenance(base_path: Any, declared: dict) -> dict:
        relocated = copy.deepcopy(declared)
        relocated["ltx_source_repo"] = str(Path(relocated_repo).expanduser().resolve())
        checked = native_provenance(base_path, relocated)
        if checked["ltx_source_revision"] != declared["ltx_source_revision"]:
            raise ValueError("relocated LTX revision differs from original provenance")
        return dict(declared)

    adapter_module.checkpoint_provenance = provenance

    native_load_adapter = model_module.load_adapter

    def load_adapter(model: Any, path: Any, base_model_path: Any) -> Any:
        if original_base is None:
            raise ValueError("original_base_model_path is required for adapter relocation")
        runtime = Path(base_model_path).expanduser().resolve()
        if runtime.name != original_base.name:
            raise ValueError("relocated base checkpoint filename differs from original identity")
        expected_sha = original_base_sha256
        if expected_sha is None:
            if not original_base.exists():
                raise FileNotFoundError("original base path unavailable; provide original_base_sha256")
            expected_sha = adapter_module.file_sha256(original_base)
        if adapter_module.file_sha256(runtime) != expected_sha:
            raise ValueError("relocated base checkpoint SHA-256 differs from original identity")
        return native_load_adapter(model, path, original_base)

    model_module.load_adapter = load_adapter


@dataclass
class BlockStreamHandle:
    transformer: torch.nn.Module
    blocks: tuple[torch.nn.Module, ...]
    device: torch.device
    hooks: tuple[Any, ...]
    cpu_storage: tuple[tuple[torch.nn.Parameter, torch.Tensor], ...] = ()
    h2d_weight_bytes: int = 0

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        for parameter, cpu_tensor in self.cpu_storage:
            parameter.data = cpu_tensor


def _find_blocks(transformer: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("transformer must expose transformer_blocks")
    result = tuple(blocks)
    if not result or any(not isinstance(block, torch.nn.Module) for block in result):
        raise ValueError("transformer_blocks must be a non-empty module sequence")
    return result


def install_block_streaming(
    transformer: torch.nn.Module,
    device: torch.device | str,
    *,
    expected_dtype: torch.dtype | None = None,
    video_only_streaming: bool = False,
) -> BlockStreamHandle:
    """Install sequential CPU↔GPU block movement hooks.

    The caller must have loaded the model on CPU first.  Hooks are deliberately
    attached only to transformer blocks; nested children move with their block.
    """
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("block streaming target must be CUDA")
    blocks = _find_blocks(transformer)
    if expected_dtype is not None:
        bad = [p.dtype for p in transformer.parameters() if p.dtype != expected_dtype]
        if bad:
            raise TypeError(f"transformer contains parameters outside {expected_dtype}")
    transformer.to(device="cpu")
    hooks: list[Any] = []
    audio_names = {"audio_attn1", "audio_attn2", "audio_ff", "audio_to_video_attn", "video_to_audio_attn"}
    audio_params = {"audio_scale_shift_table", "audio_prompt_scale_shift_table", "scale_shift_table_a2v_ca_audio", "scale_shift_table_a2v_ca_video"}
    cpu_by_block = {}
    for block in blocks:
        pairs = []
        for name, parameter in block.named_parameters():
            if video_only_streaming and (name.split(".", 1)[0] in audio_names or name in audio_params):
                continue
            pairs.append((parameter, parameter.data))
        cpu_by_block[id(block)] = tuple(pairs)
    cpu_storage = tuple(pair for pairs in cpu_by_block.values() for pair in pairs)
    buffers_by_block = {
        id(block): tuple((owner, name, buffer) for owner in block.modules()
                         for name, buffer in owner.named_buffers(recurse=False) if buffer is not None)
        for block in blocks
    }

    def before(module: torch.nn.Module, _args: tuple[Any, ...]) -> None:
        for parameter, cpu_tensor in cpu_by_block[id(module)]:
            parameter.data = cpu_tensor.to(device=target)
        for owner, name, cpu_tensor in buffers_by_block[id(module)]:
            owner._buffers[name] = cpu_tensor.to(device=target)

    def after(module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        for parameter, cpu_tensor in cpu_by_block[id(module)]:
            parameter.data = cpu_tensor
        for owner, name, cpu_tensor in buffers_by_block[id(module)]:
            owner._buffers[name] = cpu_tensor
        return output

    for block in blocks:
        hooks.append(block.register_forward_pre_hook(before))
        try:
            hooks.append(block.register_forward_hook(after, always_call=True))
        except TypeError:
            hooks.append(block.register_forward_hook(after))

    # Keep embeddings/heads and other non-block parameters resident on target.
    # Moving the ModuleList container would recursively move every block, so it
    # is explicitly excluded. Direct parameters/buffers are handled separately.
    block_ids = {id(block) for block in blocks}
    top_audio_modules = {
        "audio_patchify_proj", "audio_caption_projection", "audio_adaln_single",
        "audio_prompt_adaln_single", "audio_proj_out", "av_ca_video_scale_shift_adaln_single",
        "av_ca_audio_scale_shift_adaln_single", "av_ca_a2v_gate_adaln_single", "av_ca_v2a_gate_adaln_single",
    }
    for child_name, child in transformer.named_children():
        if child is getattr(transformer, "transformer_blocks"):
            continue
        if video_only_streaming and child_name in top_audio_modules:
            continue
        if id(child) not in block_ids:
            child.to(device=target)
    for name, parameter in transformer.named_parameters(recurse=False):
        if not (video_only_streaming and name == "audio_scale_shift_table"):
            parameter.data = parameter.data.to(device=target)
    for name, buffer in transformer.named_buffers(recurse=False):
        if buffer is not None:
            transformer._buffers[name] = buffer.to(device=target)
    h2d_bytes = sum(t.numel() * t.element_size() for _, t in cpu_storage)
    return BlockStreamHandle(transformer, blocks, target, tuple(hooks), cpu_storage, h2d_bytes)


def load_with_block_streaming(
    load_transformer: Any,
    checkpoint: Any,
    *,
    target_device: torch.device | str,
    dtype: torch.dtype,
    **kwargs: Any,
) -> tuple[torch.nn.Module, BlockStreamHandle]:
    """Load on CPU, then install streaming while returning a CUDA-rooted model.

    ``load_transformer`` is the repository's native loader.  Keeping loading
    on CPU avoids a transient full-model CUDA allocation before hooks exist.
    """
    transformer = load_transformer(checkpoint, device=torch.device("cpu"), dtype=dtype, **kwargs)
    handle = install_block_streaming(transformer, target_device, expected_dtype=dtype)
    first = next(transformer.parameters(), None)
    if first is None or first.device != torch.device(target_device):
        handle.close()
        raise RuntimeError("streamed transformer root is not CUDA-resident")
    return transformer, handle


def assert_cpu_stream_state(handle: BlockStreamHandle) -> None:
    """Raise unless all streamed blocks currently reside on CPU."""
    for index, block in enumerate(handle.blocks):
        devices = {parameter.device.type for parameter in block.parameters()}
        if devices and devices != {"cpu"}:
            raise AssertionError(f"block {index} is not CPU-resident: {devices}")
