"""Install two-level multi-view SA without replacing the native LTX block.

The residual, AdaLN/RMSNorm, text Cross-Attention, FFN, output head, checkpoint
loop and diffusion state remain native. Only video's `attn1` is replaced;
shared native O/FFN Linear objects receive additive, delta-only LoRA.
"""

import json
import math
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import MethodType

import torch
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from .adaptive_local_attention import adaptive_local
from .compression import SpatialCompressor, SpatialDecompressor
from .geometry import CanvasGeometry, local_windows
from .global_attention import global_attention
from .lora import LinearLoRA, install_linear_lora, project_with_loras
from .native_local_attention import native_local
from .position import RopeSettings, ViewContext

# Scope the already-computed native gate to one block invocation. A ContextVar
# isolates concurrent calls; finally/reset also covers checkpoint recomputation
# and its early-stop exceptions. No gate is retained on a module or in RoPE.
_native_sa_gate: ContextVar[tuple[nn.Module, torch.Tensor | None] | None] = ContextVar("ltx_outpaint_native_sa_gate", default=None)


def _forward_with_native_gate(self, video, audio):
    if torch.compiler.is_compiling():
        raise RuntimeError("Native fusion gate capture requires eager blocks; compiled capture is not supported")
    token = _native_sa_gate.set((self.attn1, None))
    try:
        return BasicAVTransformerBlock.forward(self, video, audio)
    finally:
        _native_sa_gate.reset(token)


def _capture_native_ada_values(self, scale_shift_table, batch_size, timestep, indices):
    values = BasicAVTransformerBlock.get_ada_values(self, scale_shift_table, batch_size, timestep, indices)
    scope = _native_sa_gate.get()
    if (scope is not None and scope[0] is self.attn1
            and scale_shift_table is self.scale_shift_table and indices == slice(0, 3)):
        # These are the SAME values unpacked as vshift_msa/vscale_msa/vgate_msa
        # by the native forward and later used by its unchanged residual path.
        _native_sa_gate.set((self.attn1, values[2]))
    return values


@dataclass(frozen=True)
class ArchitectureConfig:
    architecture_version: int = 8
    fusion_type: str = "ltx_native_vgate"
    adaptive_attention_type: str = "target_touching_c2_query_only"
    position_type: str = "pma_v2_nominal_grid"
    adaptive_support_policy: str = "c4_first_c2_promotion"
    adaptive_role_priority: str = "anchor_target_bridge"
    adaptive_bbox_alignment: str = "canvas_center_minimal_expand"
    adaptive_restore_policy: str = "per_query_full_cell_target_write"
    adaptive_fusion_policy: str = "target_masked_native_passthrough"
    lora_type: str = "shared_outpainting_plus_branch_specialization"
    lora_conditioning: str = "sentinel_free_native_hrlrvo"
    lora_initialization: str = "peft_kaiming_a_zero_b"
    lora_targets: str = "shared_qkvo_branch_qkv_ffn"
    lora_bias: bool = False
    lora_dropout: float = 0.0
    shared_outpainting_lora_enabled: bool = True
    shared_outpainting_lora_rank: int = 32
    shared_outpainting_lora_alpha: float = 32.0
    native_lora_enabled: bool = False
    adaptive_lora_enabled: bool = True
    adaptive_lora_rank: int = 64
    adaptive_lora_alpha: float = 64.0
    global_lora_enabled: bool = True
    global_lora_rank: int = 64
    global_lora_alpha: float = 64.0
    ffn_lora_enabled: bool = True
    ffn_lora_rank: int = 32
    ffn_lora_alpha: float = 32.0
    official_outpainting_ic_lora_enabled: bool = False
    rgb_sentinel_conditioning: bool = False
    k: int = 4
    scale: int = 4
    local_fusion_init_native_weight: float = 0.5
    global_fusion_init_local_weight: float = 0.5
    decompressor_refine_initialization: str = "depthwise_identity"

    def __post_init__(self):
        if self.architecture_version != 8 or self.fusion_type != "ltx_native_vgate":
            raise ValueError("Unsupported architecture version; pre-v8 partial-cell adapters cannot be converted implicitly")
        if self.decompressor_refine_initialization != "depthwise_identity":
            raise ValueError("Decompressor refinement requires depthwise_identity initialization")
        if self.adaptive_attention_type != "target_touching_c2_query_only":
            raise ValueError("Unsupported Adaptive attention type")
        if self.position_type != "pma_v2_nominal_grid":
            raise ValueError("Unsupported PMA coordinate-convention version")
        if self.adaptive_support_policy != "c4_first_c2_promotion":
            raise ValueError("Unsupported Adaptive support policy")
        if (self.adaptive_role_priority != "anchor_target_bridge"
                or self.adaptive_bbox_alignment != "canvas_center_minimal_expand"
                or self.adaptive_restore_policy != "per_query_full_cell_target_write"
                or self.adaptive_fusion_policy != "target_masked_native_passthrough"):
            raise ValueError("Adaptive requires A > T > B, centerward full-cell expansion and Target-only write")
        if (self.lora_type != "shared_outpainting_plus_branch_specialization"
                or self.lora_conditioning != "sentinel_free_native_hrlrvo"
                or self.lora_initialization != "peft_kaiming_a_zero_b"
                or self.lora_targets != "shared_qkvo_branch_qkv_ffn"):
            raise ValueError("Unsupported branch LoRA architecture/conditioning/initialization")
        if self.lora_bias is not False or self.lora_dropout != 0.0:
            raise ValueError("Hrlrvo branch LoRA requires bias=False and dropout=0")
        if any(value is not True for value in
               (self.shared_outpainting_lora_enabled, self.adaptive_lora_enabled,
                self.global_lora_enabled, self.ffn_lora_enabled)) or self.native_lora_enabled is not False:
            raise ValueError("v6 requires shared Outpainting + Adaptive/Global + FFN LoRA, without Native LoRA")
        if any(value is not False for value in
               (self.official_outpainting_ic_lora_enabled, self.rgb_sentinel_conditioning)):
            raise ValueError("Conditioning protocol incompatible: official IC-LoRA and RGB sentinels are unsupported")
        if self.k < 1:
            raise ValueError("K must be positive")
        for role in ("shared_outpainting", "adaptive", "global", "ffn"):
            rank, alpha = getattr(self, f"{role}_lora_rank"), getattr(self, f"{role}_lora_alpha")
            if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
                raise ValueError(f"{role} LoRA rank and alpha must be positive and finite")
        if self.scale != 4:
            raise ValueError("C4-first/C2-promotion requires S=4; Local K remains independent")
        for name in ("local_fusion_init_native_weight", "global_fusion_init_local_weight"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")


@dataclass(frozen=True)
class OutpaintingModality(Modality):
    geometry: CanvasGeometry | None = None
    early_global_boost_scale: float | None = None

    def split(self, sizes):
        base = Modality(**{f.name: getattr(self, f.name) for f in fields(Modality)})
        chunks = base.split(sizes)
        anchors = self.geometry.batch_anchors(sum(sizes))
        start, result = 0, []
        for chunk, size in zip(chunks, sizes, strict=True):
            geometry = replace(self.geometry, anchors=anchors[start:start + size])
            if self.early_global_boost_scale is None:
                result.append(attach_geometry(chunk, geometry))
            else:
                result.append(
                    attach_geometry(
                        chunk,
                        geometry,
                        early_global_boost_scale=self.early_global_boost_scale,
                    )
                )
            start += size
        return result


def attach_geometry(
    video: Modality,
    geometry: CanvasGeometry,
    *,
    early_global_boost_scale: float | None = None,
) -> OutpaintingModality:
    if early_global_boost_scale is not None and (
        type(early_global_boost_scale) is not float
        or not math.isfinite(early_global_boost_scale)
        or early_global_boost_scale <= 1.0
    ):
        raise ValueError("Active Early Global Boost scale must be a finite float above 1.0")
    return OutpaintingModality(
        **{f.name: getattr(video, f.name) for f in fields(Modality)},
        geometry=geometry,
        early_global_boost_scale=early_global_boost_scale,
    )


class OutpaintingPreprocessor:
    """Carry immutable geometry inside checkpointed args, not mutable global state."""
    def __init__(self, native):
        self.native = native
        self.rope = RopeSettings.from_preprocessor(native)

    def prepare(self, modality, cross_modality=None):
        if cross_modality is not None:
            raise ValueError("This outpainting entrypoint is video-only; do not supply an audio modality")
        if not isinstance(modality, OutpaintingModality) or modality.geometry is None:
            raise ValueError("Use attach_geometry(video, CanvasGeometry(...)) for every model call, including CFG/STG")
        geometry = modality.geometry
        b, n, _ = modality.latent.shape
        if n != geometry.frames * geometry.height * geometry.width:
            raise ValueError("Canonical token count does not match supplied runtime geometry")
        geometry.batch_anchors(b)
        args = self.native.prepare(modality, cross_modality)
        positions = modality.positions
        if positions.ndim != 4 or positions.shape[1:] != (3, n, 2):
            raise ValueError("Expected native LTX-2.3 [B,3,N,2] position bounds")
        positions = positions.expand(b, -1, -1, -1)
        sigma = modality.sigma.reshape(-1).expand(b)
        if modality.early_global_boost_scale is None:
            context = ViewContext(
                geometry, positions, sigma, args.positional_embeddings, self.rope
            )
        else:
            context = ViewContext(
                geometry,
                positions,
                sigma,
                args.positional_embeddings,
                self.rope,
                early_global_boost_scale=modality.early_global_boost_scale,
            )
        return replace(args, positional_embeddings=context)


class NativeGateFusion(nn.Module):
    """Per-block channel affine of native vgate_msa; no extra timestep encoder."""
    def __init__(self, dim, initial_weight):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.full((dim,), math.log(initial_weight / (1 - initial_weight))))

    def forward(self, native_gate):
        # Eager addcmul promotes in its kernel: preserve FP32 master/affine
        # semantics without a full FP32 gate copy or separate product tensor.
        affine = torch.addcmul(self.beta, native_gate, self.gamma)
        # This fresh non-leaf buffer has no aliases. addcmul backward needs its
        # inputs, sigmoid backward needs its result; no canonical state changes.
        return affine.sigmoid_()


def _apply_early_global_boost_(
    learned_local_weight: torch.Tensor, global_scale: float
) -> torch.Tensor:
    """Reuse a fresh learned gate tensor as qG, then restore effective Local weight."""
    if torch.is_grad_enabled():
        raise RuntimeError("Early Global Boost is inference-only and cannot run with gradients enabled")
    if (
        type(global_scale) is not float
        or not math.isfinite(global_scale)
        or global_scale <= 1.0
    ):
        raise ValueError("Active Early Global Boost scale must be a finite float above 1.0")
    # learned_local_weight is the fresh sigmoid activation wG. Transform it in
    # place so no second full gate tensor or full-feature residual is allocated:
    # qG = 1-wG; qG_eff = clamp(scale*qG, 0, 1); wG_eff = 1-qG_eff.
    return (
        learned_local_weight.neg_()
        .add_(1.0)
        .mul_(global_scale)
        .clamp_(0.0, 1.0)
        .neg_()
        .add_(1.0)
    )


class MultiViewSelfAttention(nn.Module):
    def __init__(self, native, config, block_index):
        super().__init__()
        # Reuse the actual pretrained module objects, preserving their state keys.
        for name in ("to_q", "to_k", "to_v", "to_out", "q_norm", "k_norm", "to_gate_logits"):
            setattr(self, name, getattr(native, name))
        for name in ("heads", "dim_head", "rope_type", "attention_function", "masked_attention_function",
                     "preattention_function", "gated_attention_function"):
            setattr(self, name, getattr(native, name))
        dim = self.to_q.in_features
        if self.to_q.out_features != dim:
            raise ValueError("LTX hidden and inner SA dimensions must match for canonical channel fusion")
        self.scale = config.scale
        self.partition_count = config.k + block_index % 2
        for role in ("shared_outpainting", "adaptive", "global"):
            rank, alpha = getattr(config, f"{role}_lora_rank"), getattr(config, f"{role}_lora_alpha")
            setattr(self, f"{role}_lora", nn.ModuleDict({
                key: LinearLoRA(getattr(self, f"to_{key}").in_features,
                                getattr(self, f"to_{key}").out_features, rank, alpha)
                for key in ("q", "k", "v")
            }))
        # No per-view O: the same native Sequential still runs after BOTH fusions.
        install_linear_lora(self.to_out[0], "shared_outpainting_lora",
                            config.shared_outpainting_lora_rank, config.shared_outpainting_lora_alpha)
        # Exactly one C_s per scale per block: pure-B parents/Global share C4;
        # every promoted C2 child shares C2. All windows share these modules.
        self.compressors = nn.ModuleDict({str(s): SpatialCompressor(dim, s)
                                          for s in (config.scale // 2, config.scale)})
        self.decompressors = nn.ModuleDict({str(s): SpatialDecompressor(dim)
                                            for s in (config.scale // 2, config.scale)})
        self.local_fusion = NativeGateFusion(dim, config.local_fusion_init_native_weight)
        self.global_fusion = NativeGateFusion(dim, config.global_fusion_init_local_weight)

    def forward(self, x, context=None, mask=None, pe=None, k_pe=None,
                perturbation_mask=None, all_perturbed=False):
        if context is not None or k_pe is not None:
            raise ValueError("Multi-view replacement is for self-attention only")
        if mask is not None:
            raise ValueError("Dense pairwise SA masks are not supported; use official latent conditioning masks")
        if not isinstance(pe, ViewContext):
            raise TypeError("Missing immutable outpainting attention context")
        captured = _native_sa_gate.get()
        _native_sa_gate.set(None)  # Do not retain the gate through native CA/FFN.
        if all_perturbed:
            # Official STG bypass still uses V/head gate/O, now with shared task deltas.
            fused = project_with_loras(self.to_v, x, (self.shared_outpainting_lora["v"],))
        else:
            b, _, d = x.shape
            g = pe.geometry
            grid = x.reshape(b, g.frames, g.height, g.width, d)
            windows = local_windows(g.height, g.width, self.partition_count)
            if captured is None or captured[0] is not self or captured[1] is None:
                raise RuntimeError("Fusion requires this native block's already-computed vgate_msa")
            native_gate = captured[1]
            if (native_gate.ndim != 3 or native_gate.shape[0] != b or native_gate.shape[2] != d
                    or native_gate.shape[1] not in (1, x.shape[1])):
                raise ValueError("Native vgate_msa must be [B,1,D] or canonical [B,T*H*W,D]")
            gate_grid = native_gate.expand(b, x.shape[1], d).reshape_as(grid)
            fused_windows = []
            for window in windows:
                native = native_local(self, grid, window, pe)
                adaptive, write_mask = adaptive_local(self, grid, window, pe)
                if native.shape != adaptive.shape:
                    raise RuntimeError("Local fusion spatial supports differ")
                if write_mask.dtype != torch.bool or write_mask.shape != (b, 1, window.height, window.width, 1):
                    raise RuntimeError("Adaptive write mask must use the original Native Local coordinates")
                # Slice native gate at exactly the restored Target coordinates;
                # gate values never pass through any spatial compressor.
                local_weight = self.local_fusion(
                    gate_grid[:, :, window.y0:window.y1, window.x0:window.x1]
                ).to(native.dtype)
                # Zero-filled Adaptive output must not attenuate Native outside
                # resolved T. Within T the original channel-wise lerp is intact.
                fused_windows.append(torch.where(write_mask, torch.lerp(adaptive, native, local_weight), native))
                # Autograd retains what it needs; Python need not retain the
                # previous window's alpha/features into the next evaluation.
                del local_weight, native, adaptive, write_mask

            # Assemble only the already fused, identical-support local windows.
            local_full = x.new_zeros(b, g.frames, g.height * g.width, d)
            count = x.new_zeros(g.height * g.width)
            for window, local in zip(windows, fused_windows, strict=True):
                indices = (torch.arange(window.y0, window.y1, device=x.device)[:, None] * g.width
                           + torch.arange(window.x0, window.x1, device=x.device)[None, :]).flatten()
                local_full.index_add_(2, indices, local.flatten(2, 3))
                count.index_add_(0, indices, torch.ones_like(indices, dtype=count.dtype))
            local_full = (local_full / count[None, None, :, None]).reshape_as(grid)
            global_full = global_attention(self, grid, pe)
            global_weight = self.global_fusion(gate_grid).to(x.dtype)
            if pe.early_global_boost_scale is not None:
                global_weight = _apply_early_global_boost_(
                    global_weight, pe.early_global_boost_scale
                )
            fused = torch.lerp(global_full, local_full, global_weight).reshape_as(x)
            if perturbation_mask is not None:
                bypass = project_with_loras(self.to_v, x, (self.shared_outpainting_lora["v"],))
                fused = fused * perturbation_mask + bypass * (1 - perturbation_mask)
        if self.to_gate_logits is not None:
            fused = self.gated_attention_function(x, fused, self)
        # The ONLY W_O call. Native block subsequently runs SA residual, CA, FFN.
        return self.to_out(fused)


def install_outpainting(model, config: ArchitectureConfig, provenance: dict | None = None):
    if isinstance(model.video_args_preprocessor, OutpaintingPreprocessor):
        raise TypeError("Outpainting modules are already installed")
    model.requires_grad_(False)
    for index, block in enumerate(model.transformer_blocks):
        if type(block) is not BasicAVTransformerBlock:
            raise TypeError("Native gate capture requires the unmodified LTX BasicAVTransformerBlock")
        base = block.attn1
        replacement = MultiViewSelfAttention(base, config, index)
        # New trainables stay FP32 like official LoRA training; Accelerate owns
        # mixed precision. Inference casts the complete model to its chosen dtype.
        replacement.to(device=base.to_q.weight.device)
        block.attn1 = replacement
        # Preserve the shared native FFN/GELU/Sequential and pretrained keys.
        for linear in (block.ff.net[0].proj, block.ff.net[2]):
            install_linear_lora(linear, "ffn_lora", config.ffn_lora_rank, config.ffn_lora_alpha)
        # Preserve native class identity, parameter keys and FSDP wrap policy.
        # The original block forward still owns residual, Cross-Attention and FFN.
        block.forward = MethodType(_forward_with_native_gate, block)
        block.get_ada_values = MethodType(_capture_native_ada_values, block)
    model.video_args_preprocessor = OutpaintingPreprocessor(model.video_args_preprocessor)
    model._outpainting_config = config
    model._outpainting_provenance = provenance
    model._outpainting_parameter_names = frozenset(name for name, p in model.named_parameters() if p.requires_grad)
    return model


def parameter_counts(model):
    counts = {name: 0 for name in ("shared_outpainting_lora", "adaptive_lora", "global_lora", "ffn_lora",
                                   "compressors", "decompressors", "local_fusion", "global_fusion", "frozen_base")}
    for name, parameter in model.named_parameters():
        if ".native_lora." in name or (".attn2." in name and "lora" in name):
            raise RuntimeError(f"Unsupported v6 adapter parameter: {name}")
        group = next((key for key in counts if f".{key}." in name), "frozen_base")
        if parameter.requires_grad != (group != "frozen_base"):
            raise RuntimeError(f"v6 trainable/frozen ownership mismatch: {name}")
        counts[group] += parameter.numel()
    counts["trainable_total"] = sum(counts.values()) - counts["frozen_base"]
    counts["total"] = counts["trainable_total"] + counts["frozen_base"]
    counts["blocks"] = len(model.transformer_blocks)
    return counts


def adapter_state_dict(model, state_dict=None):
    state = model.state_dict() if state_dict is None else state_dict
    return {name: state[name] for name in model._outpainting_parameter_names}


def load_adapter(model, path, base_model_path):
    """Fail closed on partial/mismatched adapters rather than leaving random branches."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    declared_base = metadata.get("base_model_path")
    if declared_base is None or Path(declared_base).resolve() != Path(base_model_path).resolve():
        raise ValueError("Adapter base_model_path is missing or differs from the selected pretrained checkpoint")
    architecture = metadata.get("architecture")
    if architecture is None or json.loads(architecture) != asdict(model._outpainting_config):
        raise ValueError("Adapter architecture metadata differs from the installed modules")
    provenance = metadata.get("provenance")
    expected_provenance = model._outpainting_provenance
    if expected_provenance is None or provenance is None or json.loads(provenance) != expected_provenance:
        raise ValueError("Adapter base/source provenance is missing or differs from the validated runtime")
    state = load_file(str(path), device="cpu")
    expected = model._outpainting_parameter_names
    if state.keys() != expected:
        raise ValueError(f"Incomplete adapter: missing={sorted(expected - state.keys())}, "
                         f"unexpected={sorted(state.keys() - expected)}")
    parameters = dict(model.named_parameters())
    for name, tensor in state.items():
        if tensor.shape != parameters[name].shape or not tensor.is_floating_point():
            raise ValueError(f"Adapter tensor shape/type mismatch: {name}")
    model.load_state_dict(state, strict=False)
