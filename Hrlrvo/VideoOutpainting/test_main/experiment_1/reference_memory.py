"""Inference-only, read-only four-image memory for Hrlrvo Global attention.

The references never enter the canonical target token grid.  Four separately
VAE-encoded, one-frame full-canvas images are passed once through the frozen
checkpoint at clean sigma.  A forward pre-hook captures each block's actual
AdaLN-normalized self-attention input, compresses it with that block's trained
Global C4 compressor, and stores projected K/V on CPU.  During target sampling,
only Global target queries read that immutable memory.  Local and Adaptive
attention retain their native Hrlrvo paths.

Temporal-neutral policy: every reference spatial lattice is repeated at each
target latent time and receives that target time's RoPE coordinate.  Thus the
memory carries spatial stage evidence without claiming a reference timestamp.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ReferenceArtifact:
    path: str
    sha256: str
    source_frame: int
    vae_encoded_separately: bool = True


@dataclass(frozen=True)
class _BlockMemory:
    key: torch.Tensor
    value: torch.Tensor
    spatial_tokens: int


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference_artifacts(artifacts: Sequence[ReferenceArtifact]) -> tuple[ReferenceArtifact, ...]:
    """Require exactly four fixed, distinct, content-hashed full-canvas images."""
    refs = tuple(artifacts)
    if len(refs) != 4:
        raise ValueError("Experiment 1 requires exactly four reference images")
    if len({ref.source_frame for ref in refs}) != 4:
        raise ValueError("Reference source frames must be distinct")
    if len({ref.sha256 for ref in refs}) != 4:
        raise ValueError("Reference image contents must be distinct")
    for ref in refs:
        path = Path(ref.path).expanduser().resolve()
        if not path.is_file() or _sha256(path) != ref.sha256:
            raise ValueError(f"Reference image hash mismatch: {path}")
        if not ref.vae_encoded_separately:
            raise ValueError("Each reference must be VAE-encoded as its own one-frame sample")
    return refs


def write_reference_manifest(path: str | Path, artifacts: Sequence[ReferenceArtifact]) -> Path:
    refs = validate_reference_artifacts(artifacts)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "hrlrvo_four_image_reference_memory_v1",
        "reference_count": 4,
        "selection": "one fixed sharp frame from each temporal stratum of first independent 65f result",
        "vae_policy": "four separate one-frame encodes; never a four-frame video encode",
        "stream": "frozen checkpoint clean-timestep stream",
        "attention": "target Q reads immutable reference K/V in Global branch only",
        "temporal_policy": "reference spatial positions use each target query time coordinate",
        "alpha": 0.1,
        "artifacts": [ref.__dict__ for ref in refs],
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _repeat_reference_rope(pe: tuple[torch.Tensor, torch.Tensor], *, frames: int,
                           spatial_tokens: int, reference_count: int,
                           split_rope: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat each target-time spatial lattice once per timeless reference."""
    axis = 2 if split_rope else 1
    result = []
    for component in pe:
        if component.shape[axis] != frames * spatial_tokens:
            raise ValueError("Target Global RoPE token count changed")
        shape = list(component.shape)
        prefix, suffix = shape[:axis], shape[axis + 1:]
        view = component.reshape(*prefix, frames, spatial_tokens, *suffix)
        view = view.unsqueeze(axis + 1).expand(
            *prefix, frames, reference_count, spatial_tokens, *suffix
        )
        result.append(view.reshape(*prefix, frames * reference_count * spatial_tokens, *suffix))
    return tuple(result)  # type: ignore[return-value]


class ReferenceMemoryHandle:
    """Installed reference-memory lifecycle; one handle belongs to one transformer."""

    def __init__(self, transformer: torch.nn.Module, *, alpha: float = 0.1) -> None:
        if type(alpha) is not float or not 0.0 <= alpha <= 1.0:
            raise ValueError("reference alpha must be a float in [0,1]")
        blocks = getattr(transformer, "transformer_blocks", None)
        if not blocks:
            raise TypeError("transformer must expose nonempty transformer_blocks")
        self.transformer = transformer
        self.alpha = alpha
        self.blocks = tuple(blocks)
        self.cache: tuple[_BlockMemory, ...] = ()
        self._active = False
        self._known_mask: torch.Tensor | None = None
        self._original_global = None
        self._model_module = None

    @property
    def cache_bytes(self) -> int:
        return sum(item.key.numel() * item.key.element_size()
                   + item.value.numel() * item.value.element_size() for item in self.cache)

    def build_cache(self, run_clean_reference_stream: Callable[[], object], *,
                    expected_reference_count: int = 4) -> dict[str, object]:
        """Run a caller-built B=4, T=1, sigma=0 reference modality exactly once.

        The callback owns VAE encoding and construction of the native modality,
        including the same empty-text conditioning as target inference.  It must
        call this transformer's frozen forward once.  Hooks validate the clean,
        one-frame, four-sample contract and capture every block.
        """
        if self._active:
            raise RuntimeError("Deactivate reference reads before rebuilding the cache")
        if expected_reference_count != 4:
            raise ValueError("The agreed experiment has exactly four references")
        captured: list[_BlockMemory | None] = [None] * len(self.blocks)
        hooks = []

        def make_hook(index: int):
            def hook(module, args, kwargs):
                if captured[index] is not None:
                    raise RuntimeError(f"Reference block {index} was evaluated more than once")
                if not args:
                    raise RuntimeError("Reference attention hook did not receive hidden states")
                hidden = args[0]
                context = kwargs.get("pe")
                if context is None and len(args) > 3:
                    context = args[3]
                geometry = getattr(context, "geometry", None)
                sigma = getattr(context, "sigma", None)
                if hidden.ndim != 3 or hidden.shape[0] != 4:
                    raise ValueError("Reference stream must be one batch of four independent images")
                if geometry is None or geometry.frames != 1:
                    raise ValueError("Each reference must have exactly one latent frame")
                if sigma is None or sigma.shape[0] != 4 or not bool((sigma == 0).all().item()):
                    raise ValueError("Reference stream must use the clean timestep sigma=0")
                if hidden.shape[1] != geometry.height * geometry.width:
                    raise ValueError("Reference stream must use the full canonical canvas grid")
                from ltx_outpaint.geometry import global_cells
                from ltx_outpaint.lora import project_with_loras

                cells = global_cells(geometry.height, geometry.width, module.scale)
                grid = hidden.reshape(4, 1, geometry.height, geometry.width, hidden.shape[-1])
                compressed = module.compressors[str(module.scale)](grid, cells).reshape(4, -1, hidden.shape[-1])
                shared, branch = module.shared_outpainting_lora, module.global_lora
                key = project_with_loras(module.to_k, compressed, (shared["k"], branch["k"]))
                value = project_with_loras(module.to_v, compressed, (shared["v"], branch["v"]))
                # CPU ownership makes concurrent jobs pay only the current block's GPU copy.
                key = key.detach().to("cpu").contiguous()
                value = value.detach().to("cpu").contiguous()
                if torch.cuda.is_available():
                    key, value = key.pin_memory(), value.pin_memory()
                captured[index] = _BlockMemory(key, value, len(cells))
            return hook

        try:
            for index, block in enumerate(self.blocks):
                hooks.append(block.attn1.register_forward_pre_hook(make_hook(index), with_kwargs=True))
            with torch.inference_mode():
                run_clean_reference_stream()
        finally:
            for hook in hooks:
                hook.remove()
        if any(item is None for item in captured):
            missing = [index for index, item in enumerate(captured) if item is None]
            raise RuntimeError(f"Reference stream did not traverse every video block: {missing}")
        self.cache = tuple(item for item in captured if item is not None)
        digest = hashlib.sha256()
        for item in self.cache:
            digest.update(item.key.view(torch.uint8).numpy().tobytes())
            digest.update(item.value.view(torch.uint8).numpy().tobytes())
        return {
            "blocks": len(self.cache), "reference_count": 4,
            "cache_bytes": self.cache_bytes, "cache_device": "cpu_pinned_if_cuda_available",
            "cache_sha256": digest.hexdigest(), "alpha": self.alpha,
            "temporal_policy": "target-time replicated; spatial coordinates preserved",
        }

    def _reference_global(self, attention, grid, context):
        from ltx_outpaint.geometry import global_cells
        from ltx_outpaint.lora import project_with_loras
        from ltx_outpaint.position import phase_rope

        if not self._active or self.alpha == 0.0:
            return self._original_global(attention, grid, context)
        try:
            block_index = next(i for i, block in enumerate(self.blocks) if block.attn1 is attention)
        except StopIteration as exc:
            raise RuntimeError("Reference Global attention received an unknown block") from exc
        memory = self.cache[block_index]
        b, frames, height, width, dim = grid.shape
        cells = global_cells(height, width, attention.scale)
        if len(cells) != memory.spatial_tokens:
            raise ValueError("Target and reference full-canvas spatial grids differ")
        compressed = attention.compressors[str(attention.scale)](grid, cells)
        target = compressed.reshape(b, -1, dim)
        shared, branch = attention.shared_outpainting_lora, attention.global_lora
        from ltx_outpaint.native_local_attention import attend
        target_pe = phase_rope(context, cells, attention.scale, grid.dtype)
        target_output = attend(attention, target, target_pe, branch)
        base = attention.decompressors[str(attention.scale)](
            target_output.reshape(b, frames, len(cells), dim),
            (height // attention.scale, width // attention.scale), (height, width),
        )
        query = project_with_loras(attention.to_q, target, (shared["q"], branch["q"]))
        key = memory.key.to(device=query.device, dtype=query.dtype, non_blocking=True)
        value = memory.value.to(device=query.device, dtype=query.dtype, non_blocking=True)
        key = key.unsqueeze(0).unsqueeze(0).expand(b, frames, 4, -1, -1).reshape(b, -1, key.shape[-1])
        value = value.unsqueeze(0).unsqueeze(0).expand(b, frames, 4, -1, -1).reshape(b, -1, value.shape[-1])
        split_rope = attention.rope_type.value == "split"
        reference_pe = _repeat_reference_rope(
            target_pe, frames=frames, spatial_tokens=len(cells),
            reference_count=4, split_rope=split_rope,
        )
        query, key = attention.preattention_function(
            query, key, attention, None, target_pe, reference_pe
        )
        reference = attention.attention_function(query, key, value, attention.heads)
        mixed = torch.lerp(target_output, reference, self.alpha)
        mixed = attention.decompressors[str(attention.scale)](
            mixed.reshape(b, frames, len(cells), dim),
            (height // attention.scale, width // attention.scale), (height, width),
        )
        generated = self._generated_mask(context, grid)
        return torch.where(generated, mixed, base)

    def _generated_mask(self, context, grid: torch.Tensor) -> torch.Tensor:
        b, frames, height, width, _ = grid.shape
        if self._known_mask is None:
            mask = torch.zeros((b, frames, height, width, 1), dtype=torch.bool, device=grid.device)
            for index, anchor in enumerate(context.geometry.batch_anchors(b)):
                mask[index, :, anchor.y0:anchor.y1, anchor.x0:anchor.x1] = True
        else:
            mask = self._known_mask.to(device=grid.device, dtype=torch.bool)
            if mask.shape == (1, frames, height, width, 1) and b > 1:
                mask = mask.expand(b, -1, -1, -1, -1)
            if mask.shape != (b, frames, height, width, 1):
                raise ValueError("known_token_mask must be [B,T,H,W,1] for this target call")
        return ~mask

    def activate(self) -> None:
        if len(self.cache) != len(self.blocks):
            raise RuntimeError("Build the complete four-reference cache before activation")
        if self._active:
            return
        import ltx_outpaint.model as model_module
        if getattr(model_module, "_reference_memory_owner", None) not in (None, self):
            raise RuntimeError("Another reference-memory handle is active in this process")
        self._model_module = model_module
        self._original_global = model_module.global_attention
        model_module.global_attention = self._reference_global
        model_module._reference_memory_owner = self
        self._active = True

    def deactivate(self) -> None:
        if not self._active:
            return
        self._model_module.global_attention = self._original_global
        self._model_module._reference_memory_owner = None
        self._active = False
        self._known_mask = None

    @contextlib.contextmanager
    def target_scope(self, known_token_mask: torch.Tensor | None = None) -> Iterator[None]:
        """Activate reads for one target call; mask includes Anchor and endpoint tokens."""
        if self._active:
            raise RuntimeError("Nested reference target scopes are unsupported")
        self._known_mask = known_token_mask
        self.activate()
        try:
            yield
        finally:
            self.deactivate()

    def remove(self) -> None:
        self.deactivate()
        self.cache = ()


def install_reference_memory(transformer: torch.nn.Module, *, alpha: float = 0.1) -> ReferenceMemoryHandle:
    """Create a reference-memory handle without changing checkpoint parameters."""
    if transformer.training:
        raise ValueError("Reference memory is inference-only; call transformer.eval() first")
    return ReferenceMemoryHandle(transformer, alpha=alpha)
