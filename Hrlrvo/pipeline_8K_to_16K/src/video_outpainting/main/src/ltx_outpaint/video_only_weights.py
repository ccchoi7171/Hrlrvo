"""Inference-only residency control for the audio-disabled LTX AudioVideo model.

The checkpoint and module graph stay intact.  The caller must complete the base
and outpainting-adapter loads before calling this function, and must pass no audio
modality to subsequent forwards.  In that path, the upstream preprocessors return
before their A/V AdaLN modules, and ``BasicAVTransformerBlock.forward`` skips both
the audio and A/V branches.  Only those exact, enumerated weights move to CPU.
"""

import torch
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

# See LTXModel._init_audio/_init_audio_video and LTXModel.forward.  The video
# patchify, text projection, timestep AdaLN, blocks and output head are excluded.
_TOP_MODULES = (
    "audio_patchify_proj",
    "audio_caption_projection",
    "audio_adaln_single",
    "audio_prompt_adaln_single",
    "audio_proj_out",
    "av_ca_video_scale_shift_adaln_single",
    "av_ca_audio_scale_shift_adaln_single",
    "av_ca_a2v_gate_adaln_single",
    "av_ca_v2a_gate_adaln_single",
)
_TOP_PARAMETERS = ("audio_scale_shift_table",)
_BLOCK_MODULES = (
    "audio_attn1",
    "audio_attn2",
    "audio_ff",
    "audio_to_video_attn",
    "video_to_audio_attn",
)
_BLOCK_PARAMETERS = (
    "audio_scale_shift_table",
    "audio_prompt_scale_shift_table",
    "scale_shift_table_a2v_ca_audio",
    "scale_shift_table_a2v_ca_video",
)


def offload_unused_audio_modules(transformer: LTXModel) -> dict[str, object]:
    """Move only unreachable audio/A-V weights to CPU after full model loading.

    This deliberately does not replace/delete modules or mutate the checkpoint.
    It refuses models whose selected parameters alias a retained video parameter.
    Calling ``transformer.to(cuda)`` afterwards would undo the optimization.
    """
    if type(transformer) is not LTXModel or transformer.model_type is not LTXModelType.AudioVideo:
        raise TypeError("Video-only residency requires an LTX AudioVideo transformer")
    if transformer.training:
        raise ValueError("Video-only residency is inference-only; call eval() first")

    module_names: list[str] = []
    direct_names: list[str] = []
    for name in _TOP_MODULES:
        module = getattr(transformer, name, None)
        if module is not None:
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"Expected module at {name}")
            module_names.append(name)
    for name in _TOP_PARAMETERS:
        if not isinstance(getattr(transformer, name, None), torch.nn.Parameter):
            raise TypeError(f"Expected parameter at {name}")
        direct_names.append(name)

    for index, block in enumerate(transformer.transformer_blocks):
        if type(block) is not BasicAVTransformerBlock:
            raise TypeError(f"Unexpected transformer block type at {index}")
        prefix = f"transformer_blocks.{index}."
        for name in _BLOCK_MODULES:
            if not isinstance(getattr(block, name, None), torch.nn.Module):
                raise TypeError(f"Expected module at {prefix}{name}")
            module_names.append(prefix + name)
        for name in _BLOCK_PARAMETERS:
            parameter = getattr(block, name, None)
            # Prompt AdaLN tables are absent for some older checkpoint configs.
            if name == "audio_prompt_scale_shift_table" and parameter is None:
                continue
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(f"Expected parameter at {prefix}{name}")
            direct_names.append(prefix + name)

    prefixes = tuple(name + "." for name in module_names)
    direct = frozenset(direct_names)

    def selected(name: str) -> bool:
        return name in direct or name.startswith(prefixes)

    selected_parameters: dict[int, torch.nn.Parameter] = {}
    retained_ids: set[int] = set()
    for name, parameter in transformer.named_parameters(remove_duplicate=False):
        if selected(name):
            selected_parameters[id(parameter)] = parameter
        else:
            retained_ids.add(id(parameter))
    if selected_parameters.keys() & retained_ids:
        raise ValueError("An audio/A-V parameter is shared with the video forward path")

    selected_buffers: dict[int, torch.Tensor] = {}
    retained_buffer_ids: set[int] = set()
    for name, buffer in transformer.named_buffers(remove_duplicate=False):
        if selected(name):
            selected_buffers[id(buffer)] = buffer
        else:
            retained_buffer_ids.add(id(buffer))
    if selected_buffers.keys() & retained_buffer_ids:
        raise ValueError("An audio/A-V buffer is shared with the video forward path")

    parameter_bytes = sum(p.numel() * p.element_size() for p in selected_parameters.values())
    buffer_bytes = sum(b.numel() * b.element_size() for b in selected_buffers.values())
    released = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*selected_parameters.values(), *selected_buffers.values())
        if tensor.device.type == "cuda"
    )
    with torch.no_grad():
        for name in module_names:
            transformer.get_submodule(name).to(device="cpu")
        for name in direct_names:
            parent_name, _, leaf = name.rpartition(".")
            owner = transformer.get_submodule(parent_name) if parent_name else transformer
            parameter = getattr(owner, leaf)
            parameter.data = parameter.data.to(device="cpu")

    return {
        "parameter_count": sum(p.numel() for p in selected_parameters.values()),
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "gpu_bytes_released": released,
        "module_names": module_names,
        "direct_parameter_names": direct_names,
    }


def place_video_only_weights_on_gpu(
    transformer: LTXModel, device: torch.device, dtype: torch.dtype
) -> dict[str, object]:
    """Place only the video forward graph on CUDA after full CPU checkpoint load.

    Loading the whole model on CUDA and *then* offloading audio lowers live
    allocation but leaves its historical blocks in the caching allocator. The
    CPU-first path never reserves those Audio/AV weight blocks on CUDA. It still
    keeps every checkpoint parameter and adapter installed in the same module.
    """
    if device.type != "cuda":
        raise ValueError("Video-only GPU placement requires a CUDA device")
    if any(parameter.device.type != "cpu" for parameter in transformer.parameters()):
        raise ValueError("Video-only placement requires a wholly CPU-loaded transformer")
    report = offload_unused_audio_modules(transformer)
    module_names = report["module_names"]
    direct_names = report["direct_parameter_names"]
    parameter_bytes = report["parameter_bytes"]
    buffer_bytes = report["buffer_bytes"]
    if (
        not isinstance(module_names, list)
        or not isinstance(direct_names, list)
        or not isinstance(parameter_bytes, int)
        or not isinstance(buffer_bytes, int)
    ):
        raise TypeError("Unexpected Video-only residency report")
    prefixes = tuple(name + "." for name in module_names)
    direct = frozenset(direct_names)

    def excluded(name: str) -> bool:
        return name in direct or name.startswith(prefixes)

    moved_parameters: set[int] = set()
    video_parameter_count = 0
    with torch.no_grad():
        for name, parameter in transformer.named_parameters(remove_duplicate=False):
            if excluded(name) or id(parameter) in moved_parameters:
                continue
            parameter.data = parameter.data.to(device=device, dtype=dtype)
            moved_parameters.add(id(parameter))
            video_parameter_count += parameter.numel()
        moved_buffers: set[int] = set()
        for name, buffer in transformer.named_buffers(remove_duplicate=False):
            if excluded(name) or id(buffer) in moved_buffers:
                continue
            destination_dtype = dtype if buffer.is_floating_point() else buffer.dtype
            buffer.data = buffer.data.to(device=device, dtype=destination_dtype)
            moved_buffers.add(id(buffer))

    if any(
        parameter.device.type != "cpu"
        for name, parameter in transformer.named_parameters()
        if excluded(name)
    ) or any(
        parameter.device.type != device.type
        or (device.index is not None and parameter.device.index != device.index)
        for name, parameter in transformer.named_parameters()
        if not excluded(name)
    ):
        raise RuntimeError("Video-only parameter placement is incomplete")

    return {
        **report,
        "video_gpu_parameter_count": video_parameter_count,
        "gpu_bytes_avoided": parameter_bytes + buffer_bytes,
    }
