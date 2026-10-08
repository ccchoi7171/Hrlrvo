"""Frozen-base provenance and a hard-disabled historical IC-LoRA audit utility.

The released In/Outpainting IC-LoRA needs a green-masked RGB reference encoded
and appended as extra tokens. It is NOT a drop-in prior for this model's single
canonical latent-replacement input. No IC artifact is part of active model
loading or checkpoint provenance. Never partially load or merge these weights.
The v6 scratch-initialized shared Outpainting LoRA is a separate, supported
delta-only adaptation owned by ArchitectureConfig, not this artifact interface.
"""

import hashlib
import math
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import ltx_core
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from safetensors import safe_open


@dataclass(frozen=True)
class OutpaintingAdapterConfig:
    supported_for_hrlrvo: ClassVar[bool] = False
    enabled: bool
    path: str
    source: str
    revision: str
    sha256: str
    scale: float = 1.0
    trainable: bool = False
    strict_compatibility: bool = True
    conditioning_protocol: str = "ic_lora_reference_tokens_green_mask_v1"

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.trainable) is not bool:
            raise ValueError("OP enabled/trainable must be booleans")
        if self.strict_compatibility is not True:
            raise ValueError("Partial/non-strict OP adapter loading is not supported")
        if not math.isfinite(self.scale):
            raise ValueError("OP adapter scale must be finite; it is not inferred from its filename")
        if not self.path or not self.source or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("OP adapter requires a path, source and pinned full revision")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("OP adapter requires a SHA-256 integrity identifier")

    def require_supported_conditioning(self):
        if self.enabled:
            raise ValueError(
                "Conditioning protocol incompatible: official IC-LoRA reference-token conditioning "
                "is unsupported for Hrlrvo. Its scratch shared Outpainting/branch/FFN LoRA "
                "does not load or merge official IC-LoRA artifacts."
            )


def reject_unsupported_conditioning(settings: dict) -> None:
    """Reject official IC/reference/sentinel activation, not scratch shared LoRA."""
    for name in ("outpainting_adapter", "ic_lora", "official_outpainting_ic_lora",
                 "rgb_sentinel_conditioning",
                 "outpainting_adapter_enabled", "ic_lora_enabled",
                 "official_outpainting_ic_lora_enabled"):
        if name not in settings:
            continue
        value = settings[name]
        enabled = value.get("enabled") if isinstance(value, dict) else value
        if enabled is not False or (isinstance(value, dict) and value.get("supported_for_hrlrvo", False) is not False):
            raise ValueError(
                f"Conditioning protocol incompatible: {name} is unsupported for Hrlrvo; "
                "use sentinel-free independent Anchor/prefix conditioning with scratch v6 LoRA"
            )
    shared = settings.get("shared_outpainting_lora")
    if isinstance(shared, dict) and any(key in shared for key in ("path", "checkpoint", "source", "sha256")):
        raise ValueError("Shared Outpainting LoRA is scratch/zero-delta only, not an external IC-LoRA artifact")
    video = settings.get("video", {})
    if isinstance(video, dict) and "neutral_value" in video:
        raise ValueError("RGB sentinel conditioning is unsupported: remove obsolete video.neutral_value")


@lru_cache(maxsize=8)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = Path(path).stat()
    if (after.st_size, after.st_mtime_ns) != (size, mtime_ns):
        raise ValueError(f"Checkpoint changed while being hashed: {path}")
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return _file_sha256(str(path), stat.st_size, stat.st_mtime_ns)


def inspect_outpainting_adapter(base_path: str | Path, config: OutpaintingAdapterConfig) -> dict:
    """Inspect every A/B pair and native base shape without loading tensor data.

    Only the pinned official Comfy prefix mapping is used, never ad-hoc key
    renaming, reshaping, partial loading or splitting in/outpainting weights.
    A weight-shape match does not override the conditioning incompatibility.
    """
    adapter_path = Path(config.path).expanduser().resolve()
    if file_sha256(adapter_path) != config.sha256:
        raise ValueError("OP adapter SHA-256 mismatch")
    with safe_open(str(base_path), framework="pt", device="cpu") as base, safe_open(
        str(adapter_path), framework="pt", device="cpu"
    ) as adapter:
        metadata = adapter.metadata() or {}
        if metadata.get("model_version") != "2.3":
            raise ValueError("OP adapter does not declare LTX-2.3")
        keys = adapter.keys()
        shapes = {key: tuple(adapter.get_slice(key).get_shape()) for key in keys}
        base_keys = set(base.keys())
        consumed, ranks, modules = set(), set(), {}
        for key, shape_a in shapes.items():
            if not key.endswith(".lora_A.weight"):
                continue
            pair = key.removesuffix(".lora_A.weight") + ".lora_B.weight"
            if pair not in shapes:
                raise ValueError(f"OP adapter is missing paired tensor: {pair}")
            shape_b = shapes[pair]
            native_key = LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key(key)
            module = native_key.removesuffix(".lora_A.weight")
            base_key = f"model.diffusion_model.{module}.weight"
            if len(shape_a) != 2 or len(shape_b) != 2 or shape_a[0] != shape_b[1]:
                raise ValueError(f"Invalid OP A/B shapes: {key}, {pair}")
            if base_key not in base_keys or tuple(base.get_slice(base_key).get_shape()) != (shape_b[0], shape_a[1]):
                raise ValueError(f"OP tensor cannot compose with native base: {base_key}")
            consumed.update((key, pair))
            ranks.add(shape_a[0])
            modules[module] = (shape_b[0], shape_a[1])
        if not modules or consumed != shapes.keys():
            raise ValueError(f"OP checkpoint has unsupported/unpaired tensors: {sorted(shapes.keys() - consumed)}")
    return {
        "supported_for_hrlrvo": False,
        "weight_shapes_compatible": True,
        "conditioning_compatible": False,
        "tensor_count": len(shapes),
        "parameters": sum(math.prod(shape) for shape in shapes.values()),
        "ranks": sorted(ranks),
        "modules": modules,
        "model_version": metadata["model_version"],
        "reference_downscale_factor": metadata.get("reference_downscale_factor"),
        # Native fuse_loras._products_for_sd_key uses coefficient * B @ A.
        "scaling": "explicit scale * B @ A; no filename or implicit alpha/rank multiplier",
    }


def validate_provenance_config(provenance: dict) -> None:
    required = {"base_model_id", "base_checkpoint", "base_sha256", "ltx_source_repo", "ltx_source_revision"}
    if set(provenance) != required:
        raise ValueError(f"Checkpoint provenance requires exactly {sorted(required)}")
    if not provenance["base_model_id"] or not re.fullmatch(r"[0-9a-f]{64}", provenance["base_sha256"]):
        raise ValueError("A base model identifier and SHA-256 are required")
    if not re.fullmatch(r"[0-9a-f]{40}", provenance["ltx_source_revision"]):
        raise ValueError("A pinned full LTX source revision is required")


def checkpoint_provenance(base_path: str | Path, provenance: dict) -> dict:
    """Validate frozen-base/source identity before loading models or resuming.

    Only the frozen LTX base and pinned source are runtime dependencies. The
    rejected official IC artifact is absent from both loading and provenance.
    """
    validate_provenance_config(provenance)
    base = Path(base_path).expanduser().resolve()
    repo = Path(provenance["ltx_source_repo"]).expanduser().resolve()
    if not Path(ltx_core.__file__).resolve().is_relative_to(repo):
        raise ValueError("Imported ltx_core is not from the configured pinned source checkout")
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != provenance["ltx_source_revision"]:
        raise ValueError("LTX source revision differs from checkpoint provenance")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no", "--", "packages"], text=True
    )
    if dirty:
        raise ValueError("Pinned LTX package sources have local modifications")
    if base.name != provenance["base_checkpoint"] or file_sha256(base) != provenance["base_sha256"]:
        raise ValueError("Base checkpoint filename/SHA-256 differs from declared identity")
    return dict(provenance)
