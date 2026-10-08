"""Official-LTX based long-range video outpainting inference.

The diffusion loop, scheduler, CFG/STG guidance, latent state handling, VAE
encoding/decoding, and media I/O are reused from the LTX-2.3 trainer.  This
module only constructs the combined spatial-anchor/temporal-prefix condition
and supplies the canonical canvas geometry required by the outpainting
self-attention installed in :mod:`ltx_outpaint.model`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

if TYPE_CHECKING:
    from ltx_outpaint.geometry import Rect

# Support the documented direct-file invocation as well as ``python -m``.
_PROJECT_SRC = str(Path(__file__).resolve().parents[1])
if _PROJECT_SRC not in sys.path:
    sys.path.insert(0, _PROJECT_SRC)


@dataclass(frozen=True)
class PixelBoundary:
    """Requested canvas embedded in a VAE-aligned pixel canvas."""

    requested_height: int
    requested_width: int
    encoded_height: int
    encoded_width: int
    top: int
    left: int

    @property
    def bottom(self) -> int:
        return self.top + self.requested_height

    @property
    def right(self) -> int:
        return self.left + self.requested_width


@dataclass(frozen=True)
class LongRangeSpec:
    clip_frames: int = 65
    prefix_frames: int = 17
    stride_frames: int = 48
    final_chunk_policy: str = "repeat_last"

    def validate(self, temporal_factor: int) -> None:
        if self.clip_frames != self.prefix_frames + self.stride_frames:
            raise ValueError("clip_frames must equal prefix_frames + stride_frames")
        if (self.clip_frames - 1) % temporal_factor:
            raise ValueError(
                "clip_frames must be aligned to the LTX video VAE temporal factor"
            )
        if (self.prefix_frames - 1) % temporal_factor:
            raise ValueError(
                "prefix_frames must be aligned to the LTX video VAE temporal factor"
            )
        if self.final_chunk_policy not in {"repeat_last", "reject"}:
            raise ValueError("final_chunk_policy must be 'repeat_last' or 'reject'")


_EARLY_GLOBAL_BOOST_START_SCALE = 1.4
_EARLY_GLOBAL_BOOST_DURATION_STEPS = 10
_EARLY_GLOBAL_BOOST_CURVE = "smoothstep"
_SEED_PREVIEW_FULL_STEPS = 30
_SEED_PREVIEW_STEP = 10
_SEED_PREVIEW_FRAMES = (0, 32, 64)


def _early_global_boost_scale(step_index: int) -> float | None:
    """Return the reported schedule scale for one zero-based denoise step."""
    if type(step_index) is not int or step_index < 0:
        raise ValueError("step_index must be a non-negative integer")
    if step_index >= _EARLY_GLOBAL_BOOST_DURATION_STEPS:
        return None
    if step_index == _EARLY_GLOBAL_BOOST_DURATION_STEPS - 1:
        # Keep the final scheduled value exact, then bypass gate arithmetic.
        return 1.0
    progress = step_index / (_EARLY_GLOBAL_BOOST_DURATION_STEPS - 1)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    return _EARLY_GLOBAL_BOOST_START_SCALE + (
        1.0 - _EARLY_GLOBAL_BOOST_START_SCALE
    ) * smooth


def _early_global_boost_metadata(enabled: bool) -> dict[str, Any]:
    if type(enabled) is not bool:
        raise TypeError("early_global_boost must be a boolean")
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "start_scale": _EARLY_GLOBAL_BOOST_START_SCALE,
        "duration_steps": _EARLY_GLOBAL_BOOST_DURATION_STEPS,
        "curve": _EARLY_GLOBAL_BOOST_CURVE,
    }


@dataclass(frozen=True)
class _EarlyGlobalBoostRuntime:
    """Map official scheduler sigma values to immutable per-step gate scales."""

    sigma_by_step: tuple[float, ...]

    @classmethod
    def from_inference_steps(cls, inference_steps: int) -> _EarlyGlobalBoostRuntime:
        if type(inference_steps) is not int or inference_steps < 1:
            raise ValueError("inference_steps must be a positive integer")
        from ltx_core.components.schedulers import LTX2Scheduler

        sigmas = LTX2Scheduler().execute(steps=inference_steps)
        sigma_by_step = tuple(float(value) for value in sigmas[:-1])
        if len(sigma_by_step) != inference_steps:
            raise RuntimeError("LTX2 scheduler did not return one sigma per denoise step")
        if len(set(sigma_by_step)) != len(sigma_by_step):
            raise RuntimeError(
                "Early Global Boost requires unique scheduler sigmas per denoise step"
            )
        return cls(sigma_by_step)

    def active_scale_for_sigma(self, sigma: torch.Tensor) -> float | None:
        """Resolve the outer scheduler step; repeated CFG/STG forwards do not advance it."""
        if sigma.numel() != 1 or sigma.dtype != torch.float32:
            raise ValueError(
                "Early Global Boost requires the official scalar float32 scheduler sigma"
            )
        value = float(sigma.detach().item())
        try:
            step_index = self.sigma_by_step.index(value)
        except ValueError as error:
            raise RuntimeError(
                "Current sigma does not belong to the configured denoising schedule"
            ) from error
        scale = _early_global_boost_scale(step_index)
        # A scheduled scale of exactly 1.0 must use the unmodified legacy branch.
        return scale if scale is not None and scale > 1.0 else None


def _resolve_adapter_path(path: Path) -> Path:
    """Resolve one explicit training checkpoint or adapter file, without fallback."""
    resolved = path.expanduser().resolve()
    adapter = resolved / "adapter_model.safetensors" if resolved.is_dir() else resolved
    if not adapter.is_file():
        raise FileNotFoundError(
            f"Inference adapter not found: {adapter}. Pass a checkpoint directory "
            "containing adapter_model.safetensors or the adapter file itself."
        )
    return adapter


def _adapter_category(name: str) -> str:
    for category in (
        "shared_outpainting_lora",
        "adaptive_lora",
        "global_lora",
        "ffn_lora",
        "compressors",
        "decompressors",
        "local_fusion",
        "global_fusion",
    ):
        if f".{category}." in name:
            return category
    return "unclassified"


def _adapter_key_report(model: Any, adapter_path: Path | None) -> dict[str, Any]:
    expected = set(model._outpainting_parameter_names)
    expected_categories: dict[str, int] = {}
    for name in expected:
        category = _adapter_category(name)
        expected_categories[category] = expected_categories.get(category, 0) + 1
    if adapter_path is None:
        return {
            "mode": "initialized_adapters",
            "adapter_path": None,
            "expected_key_count": len(expected),
            "loaded_key_count": 0,
            "missing_key_count": None,
            "unexpected_key_count": None,
            "shape_mismatch_count": None,
            "keys_by_category": expected_categories,
        }

    from safetensors import safe_open

    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        loaded = set(handle.keys())
    return {
        "mode": "checkpoint",
        "adapter_path": str(adapter_path),
        "expected_key_count": len(expected),
        "loaded_key_count": len(loaded),
        # load_adapter is the authoritative shape/metadata validator and has
        # already completed before this report is emitted.
        "missing_key_count": len(expected - loaded),
        "unexpected_key_count": len(loaded - expected),
        "shape_mismatch_count": 0,
        "keys_by_category": expected_categories,
    }


def _iter_long_range_chunks(
    requested_frames: int, spec: LongRangeSpec, *, temporal_factor: int = 8
) -> Iterator[dict[str, Any]]:
    """Describe recurrent chunks with the shortest VAE-aligned working lengths."""
    if requested_frames <= 0:
        raise ValueError("requested_frames must be positive")
    if temporal_factor <= 0:
        raise ValueError("temporal_factor must be positive")
    chunk_index = 0
    start = 0
    while start == 0 or start + spec.prefix_frames < requested_frames:
        real_count = min(spec.clip_frames, requested_frames - start)
        if real_count < spec.clip_frames and spec.final_chunk_policy == "reject":
            raise ValueError(
                f"Final chunk has {real_count} frames; configure final_chunk_policy=repeat_last "
                f"or request a length of {spec.clip_frames} + N*{spec.stride_frames} frames"
            )
        working_count = 1 + math.ceil((real_count - 1) / temporal_factor) * temporal_factor
        padded = working_count - real_count
        if start == 0:
            append_start, append_count = 0, real_count
            prefix_range = None
        else:
            append_start = start + spec.prefix_frames
            append_count = min(
                spec.stride_frames, max(0, requested_frames - append_start)
            )
            prefix_range = [start, start + spec.prefix_frames]
        yield {
            "chunk_index": chunk_index,
            "absolute_start_frame": start,
            "anchor_source_frame_range": [start, start + real_count],
            "prefix_output_frame_range": prefix_range,
            "new_output_frame_range": [append_start, append_start + append_count],
            "output_appended_count": append_count,
            "real_input_frame_count": real_count,
            "working_frame_count": working_count,
            "repeat_last_padding_count": padded,
            "final_partial_repeat_last": bool(padded),
        }
        chunk_index += 1
        if start == 0:
            if requested_frames <= spec.clip_frames:
                break
            start = spec.stride_frames
        else:
            start += spec.stride_frames


def _plan_long_range_chunks(
    requested_frames: int, spec: LongRangeSpec, *, temporal_factor: int = 8
) -> list[dict[str, Any]]:
    """Materialize the chunk plan for diagnostics/tests, not production inference."""
    return list(
        _iter_long_range_chunks(
            requested_frames, spec, temporal_factor=temporal_factor
        )
    )


def _lineage_health(records: list[dict[str, Any]], requested_frames: int) -> tuple[int, int]:
    covered: list[int] = []
    for record in records:
        begin, end = record["new_output_frame_range"]
        covered.extend(range(begin, end))
    expected = set(range(requested_frames))
    actual = set(covered)
    gaps = len(expected - actual)
    duplicates = len(covered) - len(actual)
    return gaps, duplicates


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _SamplingProgress:
    """Observe native Euler-step completions without changing the denoising loop."""

    def __init__(
        self, *, chunk_index: int | None = None, chunk_count: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        self.chunk_index = chunk_index
        self.chunk_count = chunk_count
        self.total_steps = total_steps
        self.completed = 0
        self.started = time.monotonic()

    def advance_step(self) -> None:
        self.completed += 1
        if self.total_steps is not None:
            elapsed = time.monotonic() - self.started
            eta = elapsed * (self.total_steps - self.completed) / self.completed
            print(json.dumps({
                "event": "inference_denoise_step",
                "chunk_index": self.chunk_index,
                "chunk_count": self.chunk_count,
                "step": self.completed,
                "total_steps": self.total_steps,
                "elapsed_seconds": round(elapsed, 3),
                "eta_seconds": round(eta, 3),
            }, sort_keys=True), flush=True)


class GeometryValidationRunner:
    """ValidationRunner facade that tags every CFG/STG model input with geometry.

    The class is constructed dynamically after the official source paths have
    been installed.  Keeping this facade small makes the native
    ``ValidationRunner._run_denoising`` the sole sampling implementation.
    """

    @staticmethod
    def create(
        validation_config: Any,
        model_path: Path,
        text_encoder_path: Path,
        video_vae_path: Path | None,
        *,
        early_global_boost: bool = False,
    ) -> Any:
        _validate_inference_text_conditioning(validation_config)
        if type(early_global_boost) is not bool:
            raise TypeError("early_global_boost must be a boolean")
        from ltx_trainer.validation_runner import ValidationRunner

        from ltx_outpaint.model import attach_geometry

        boost_runtime = (
            _EarlyGlobalBoostRuntime.from_inference_steps(
                int(validation_config.inference_steps)
            )
            if early_global_boost
            else None
        )

        class _Runner(ValidationRunner):
            geometry = None
            predicted_clean_capture = None

            def _modality_from_latent_state(
                self, state: Any, context: torch.Tensor, sigma: torch.Tensor
            ) -> Any:
                modality = ValidationRunner._modality_from_latent_state(
                    state, context, sigma
                )
                if self.geometry is None:
                    raise RuntimeError("CanvasGeometry must be set before denoising")
                if boost_runtime is None:
                    return attach_geometry(modality, self.geometry)
                boost_scale = boost_runtime.active_scale_for_sigma(sigma)
                if boost_scale is None:
                    return attach_geometry(modality, self.geometry)
                return attach_geometry(
                    modality,
                    self.geometry,
                    early_global_boost_scale=boost_scale,
                )

            def _run_denoising(
                self, *args: Any, stop_after_step: int | None = None, **kwargs: Any
            ) -> tuple[Any, Any]:
                """Use the native loop, optionally intercepting its Euler x0 input.

                ``ValidationRunner`` passes the fully guided, condition-composited
                predicted-clean tensor into ``EulerDiffusionStep.step``.  A
                process-local temporary stepper captures that exact input before
                the requested Euler update and unwinds the native loop.  The
                vendored LTX checkout remains byte-for-byte clean, which preserves
                its pinned-source provenance validation.
                """
                if stop_after_step is None:
                    return ValidationRunner._run_denoising(self, *args, **kwargs)
                if not 1 <= stop_after_step <= int(self._config.inference_steps):
                    raise ValueError(
                        "stop_after_step must be within the configured schedule"
                    )
                video_state = kwargs.get("video_state")
                audio_state = kwargs.get("audio_state")
                if video_state is None or audio_state is not None:
                    raise ValueError(
                        "Predicted-clean seed preview requires video-only denoising"
                    )

                import ltx_trainer.validation_runner as validation_module

                original_stepper = validation_module.EulerDiffusionStep

                class _PredictedCleanCaptured(RuntimeError):
                    def __init__(
                        self,
                        denoised: torch.Tensor,
                        sigmas: torch.Tensor,
                        step_index: int,
                    ) -> None:
                        super().__init__("predicted-clean preview captured")
                        self.denoised = denoised
                        self.sigmas = sigmas
                        self.step_index = step_index

                class _CapturingEuler(original_stepper):
                    def step(
                        self,
                        sample: torch.Tensor,
                        denoised_sample: torch.Tensor,
                        sigmas: torch.Tensor,
                        step_index: int,
                        **step_kwargs: Any,
                    ) -> torch.Tensor:
                        if step_index + 1 == stop_after_step:
                            raise _PredictedCleanCaptured(
                                denoised_sample, sigmas, step_index
                            )
                        return super().step(
                            sample,
                            denoised_sample,
                            sigmas,
                            step_index,
                            **step_kwargs,
                        )

                validation_module.EulerDiffusionStep = _CapturingEuler
                try:
                    return ValidationRunner._run_denoising(self, *args, **kwargs)
                except _PredictedCleanCaptured as capture:
                    self.predicted_clean_capture = {
                        "step": capture.step_index + 1,
                        "loop_index_zero_based": capture.step_index,
                        "sigma_schedule": capture.sigmas.detach().float().cpu(),
                        "source": (
                            "EulerDiffusionStep.denoised_sample_after_cfg_stg_"
                            "and_conditioning_before_euler"
                        ),
                    }
                    return replace(video_state, latent=capture.denoised), None
                finally:
                    validation_module.EulerDiffusionStep = original_stepper

        return _Runner(
            config=validation_config,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            video_vae_path=video_vae_path,
            load_text_encoder_in_8bit=False,
        )


def _install_official_sources(repo: Path) -> None:
    for package in ("ltx-core", "ltx-pipelines", "ltx-trainer"):
        source = repo / "packages" / package / "src"
        if not source.is_dir():
            raise FileNotFoundError(f"Missing official LTX package source: {source}")
        source_string = str(source)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)


def _vae_boundary(
    height: int, width: int, height_factor: int, width_factor: int
) -> PixelBoundary:
    encoded_height = math.ceil(height / height_factor) * height_factor
    encoded_width = math.ceil(width / width_factor) * width_factor
    # Explicit symmetric pixel padding is only a VAE-boundary policy.  It is
    # removed after decode and is never used to pad self-attention geometry.
    top = (encoded_height - height) // 2
    left = (encoded_width - width) // 2
    return PixelBoundary(height, width, encoded_height, encoded_width, top, left)


def _center_anchor(
    boundary: PixelBoundary, anchor_height: int, anchor_width: int
) -> tuple[int, int, int, int]:
    if (
        anchor_height > boundary.requested_height
        or anchor_width > boundary.requested_width
    ):
        raise ValueError(
            f"Anchor {anchor_width}x{anchor_height} exceeds requested canvas "
            f"{boundary.requested_width}x{boundary.requested_height}; silent resizing is forbidden"
        )
    y0 = boundary.top + (boundary.requested_height - anchor_height) // 2
    x0 = boundary.left + (boundary.requested_width - anchor_width) // 2
    return y0, x0, y0 + anchor_height, x0 + anchor_width


def _validate_empty_text_conditioning(validation_config: Any) -> None:
    """Keep the training/validation path fixed-empty without bypassing native text CFG."""
    if validation_config.negative_prompt != "" or any(
        sample.prompt != "" for sample in validation_config.samples
    ):
        raise ValueError(
            "Hrlrvo production text conditioning is fixed-empty: positive='' and negative=''"
        )


def _normalize_inference_prompt(value: str | None, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return value


def _validate_inference_text_conditioning(validation_config: Any) -> None:
    """Inference allows arbitrary text; native positive/negative contexts require strings."""
    if not isinstance(validation_config.negative_prompt, str):
        raise TypeError("negative_prompt must be a string")
    for sample in validation_config.samples:
        if not isinstance(sample.prompt, str):
            raise TypeError("prompt must be a string")


def _make_validation_config(
    settings: dict[str, Any],
    prompt: str | None,
    boundary: PixelBoundary,
    clip_frames: int,
    *,
    negative_prompt: str | None = "",
) -> Any:
    from ltx_trainer.config import ValidationConfig, ValidationSample

    sampling = settings["sampling"]
    if sampling.get("negative_prompt", "") != "":
        raise ValueError("sampling.negative_prompt must be empty; use --negative-prompt for inference")
    prompt = _normalize_inference_prompt(prompt, "prompt")
    negative_prompt = _normalize_inference_prompt(negative_prompt, "negative_prompt")
    config = ValidationConfig(
        samples=[ValidationSample(prompt=prompt)],
        negative_prompt=negative_prompt,
        video_dims=(boundary.encoded_width, boundary.encoded_height, clip_frames),
        frame_rate=float(settings["video"]["fps"]),
        seed=int(sampling["seed"]),
        inference_steps=int(sampling["inference_steps"]),
        video_cfg_scale=float(sampling["cfg_scale"]),
        video_stg_scale=float(sampling["stg_scale"]),
        stg_blocks=list(sampling["stg_blocks"]),
        guidance_rescale=float(sampling["guidance_rescale"]),
        video_modality_guidance_scale=1.0,
        generate_video=True,
        generate_audio=False,
        interval=None,
    )
    _validate_inference_text_conditioning(config)
    return config


def _memory_optimization_settings(settings: dict[str, Any]) -> dict[str, bool | int]:
    """Validate inference-only switches; absent settings preserve the legacy path."""
    raw = settings.get("memory_optimization", {})
    if not isinstance(raw, dict):
        raise TypeError("memory_optimization must be a mapping")
    names = ("video_only_gpu_weights", "vae_decode_streaming", "ffn_token_chunking")
    unknown = set(raw) - {*names, "ffn_chunk_tokens"}
    if unknown:
        raise ValueError(f"Unknown memory_optimization options: {sorted(unknown)}")
    result: dict[str, bool | int] = {}
    for name in names:
        value = raw.get(name, False)
        if type(value) is not bool:
            raise TypeError(f"memory_optimization.{name} must be boolean")
        result[name] = value
    chunk_tokens = raw.get("ffn_chunk_tokens", 8192)
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("memory_optimization.ffn_chunk_tokens must be a positive integer")
    result["ffn_chunk_tokens"] = chunk_tokens
    return result


def _load_models(
    settings: dict[str, Any],
    validation_config: Any,
    boundary: PixelBoundary,
    *,
    checkpoint_path: Path | None = None,
    initialized_adapters: bool = False,
    early_global_boost: bool = False,
    memory_optimization: dict[str, bool | int] | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    from ltx_trainer.model_loader import (
        load_transformer,
        load_video_vae_encoder,
        read_video_scale_factors,
    )

    from ltx_outpaint.model import ArchitectureConfig, install_outpainting, load_adapter
    from ltx_outpaint.outpainting_adapter import (
        checkpoint_provenance,
        reject_unsupported_conditioning,
    )

    model_cfg = settings["model"]
    reject_unsupported_conditioning(settings)
    architecture_kwargs = dict(settings["architecture"])
    architecture_kwargs["k"] = architecture_kwargs.pop("local_partition_k")
    architecture_kwargs["scale"] = architecture_kwargs.pop("compression_scale")
    architecture = ArchitectureConfig(**architecture_kwargs)
    checkpoint = Path(model_cfg["base_model_path"]).expanduser().resolve()
    provenance = checkpoint_provenance(checkpoint, settings["provenance"])
    vae_path_value = model_cfg.get("video_vae_path")
    vae_path = (
        Path(vae_path_value).expanduser().resolve() if vae_path_value else checkpoint
    )
    text_encoder_path = Path(model_cfg["text_encoder_path"]).expanduser().resolve()
    if checkpoint_path is not None and initialized_adapters:
        raise ValueError("Choose either a checkpoint or initialized adapters, not both")
    if initialized_adapters:
        adapter_path = None
    else:
        configured = checkpoint_path or Path(model_cfg["adapter_path"])
        adapter_path = _resolve_adapter_path(configured)
    for required in (checkpoint, vae_path, text_encoder_path):
        if not required.exists():
            raise FileNotFoundError(required)

    device = torch.device(settings["runtime"]["device"])
    dtype = (
        torch.bfloat16 if settings["runtime"]["dtype"] == "bfloat16" else torch.float32
    )
    scale_factors = read_video_scale_factors(vae_path)
    if (
        boundary.encoded_height % scale_factors.height
        or boundary.encoded_width % scale_factors.width
    ):
        raise ValueError(
            "Encoded canvas is not aligned to the checkpoint's VAE spatial factors"
        )

    # Match the official ValidationRunner lifecycle: cache text embeddings and
    # unload Gemma before placing the 22B transformer on the accelerator.
    runner = GeometryValidationRunner.create(
        validation_config,
        checkpoint,
        text_encoder_path,
        vae_path,
        early_global_boost=early_global_boost,
    )
    vae_encoder = load_video_vae_encoder(
        vae_path, device="cpu", dtype=dtype
    ).requires_grad_(False)

    # The scratch adapter path is an explicit diagnostic mode. Seed before
    # adapter installation so its Kaiming-A/zero-B initialization is repeatable.
    torch.manual_seed(int(validation_config.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(validation_config.seed))
    memory_optimization = memory_optimization or {}
    video_only_gpu_weights = bool(memory_optimization.get("video_only_gpu_weights", False))
    # The OFF branch keeps the exact original accelerator load sequence. In the
    # ON branch, the unchanged base loader and validated adapter load finish on
    # CPU so unreachable Audio/AV weights never reserve CUDA allocator blocks.
    load_device = torch.device("cpu") if video_only_gpu_weights else device
    transformer = load_transformer(checkpoint, device=load_device, dtype=dtype)
    transformer = install_outpainting(
        transformer,
        architecture,
        provenance,
    )
    # install_outpainting deliberately creates trainable modules in FP32 for
    # training. In inference the complete model must have one coherent dtype.
    transformer.to(device=load_device, dtype=dtype)
    if adapter_path is not None:
        load_adapter(transformer, adapter_path, base_model_path=checkpoint)
    adapter_report = _adapter_key_report(transformer, adapter_path)
    transformer.eval()
    if video_only_gpu_weights:
        from ltx_outpaint.video_only_weights import place_video_only_weights_on_gpu

        report = place_video_only_weights_on_gpu(transformer, device, dtype)
        print(json.dumps({"event": "video_only_gpu_weights", **report}, sort_keys=True), flush=True)
    if memory_optimization.get("ffn_token_chunking", False):
        from ltx_outpaint.ffn_token_chunking import install_ffn_token_chunking

        ffn_report = install_ffn_token_chunking(
            transformer, int(memory_optimization["ffn_chunk_tokens"])
        )
        print(json.dumps({"event": "ffn_token_chunking", **ffn_report}, sort_keys=True), flush=True)
    return transformer, runner, vae_encoder, adapter_report


def _reflect_pad_rgb(
    frames: torch.Tensor, *, top: int, bottom: int, left: int, right: int
) -> torch.Tensor:
    """Apply the lazy Anchor cache's canonical leading/trailing reflect padding."""
    from ltx_pipelines.utils.media_io.resize import resize_and_reflect_pad

    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("Condition RGB must be [F,3,H,W]")
    if min(top, bottom, left, right) < 0:
        raise ValueError("Canonical reflect padding cannot be negative")
    height, width = frames.shape[-2:]
    if max(top, bottom) >= height or max(left, right) >= width:
        raise ValueError("Condition is too small for canonical reflect padding")
    rgb = frames.permute(0, 2, 3, 1)
    if top or left:
        leading = resize_and_reflect_pad(
            rgb.flip((1, 2)), height + top, width + left
        )
        rgb = leading[0].permute(1, 2, 3, 0).flip((1, 2))
    padded = resize_and_reflect_pad(
        rgb, height + top + bottom, width + left + right
    )
    if padded.shape != (1, 3, frames.shape[0], height + top + bottom, width + left + right):
        raise RuntimeError("Official reflect padding changed condition geometry")
    return padded


@torch.inference_mode()
def _encode_condition_rgb(
    frames: torch.Tensor, vae_encoder: torch.nn.Module, runner: Any, device: torch.device,
    *, training_aligned_anchor: bool = False,
) -> torch.Tensor:
    if not torch.isfinite(frames).all().item():
        raise ValueError("Condition RGB contains non-finite values")
    if frames.numel() and (frames.min().item() < 0.0 or frames.max().item() > 1.0):
        raise ValueError("Condition RGB must use the official [0,1] input range")
    vae_encoder.eval().requires_grad_(False)
    try:
        if training_aligned_anchor:
            from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig

            # Anchor-only override matching the training cache: spatial 512/128,
            # with frames omitted to keep temporal tiling OFF. Prefix stays native.
            spatial = DimensionSizeConfig(tile_size=512, overlap=128)
            tiling = TileSizeConfig(height=spatial, width=spatial)
            vae_encoder.to(device)
            return vae_encoder.tiled_encode(  # type: ignore[operator]
                frames.to(device=device).mul(2).sub(1).to(dtype=torch.bfloat16), tiling
            ).cpu()
        return runner._encode_video(
            frames.to(device=device).mul(2).sub(1), vae_encoder, device
        )
    finally:
        vae_encoder.to("cpu")


def _condition_latents(
    anchor: torch.Tensor,
    previous_prefix: torch.Tensor | None,
    boundary: PixelBoundary,
    anchor_rect: tuple[int, int, int, int],
    latent_shape: Any,
    scale_factors: Any,
    vae_encoder: torch.nn.Module,
    runner: Any,
    device: torch.device,
    dtype: torch.dtype,
    *,
    prefix_latent_override: torch.Tensor | None = None,
    generated_prefix_strength: float = 1.0,
) -> tuple[torch.Tensor, Rect]:
    """Return observed-only conditioning and its one actual encoded Anchor extent."""
    from ltx_core.model.video_vae.video_vae import (
        map_temporal_interval_to_latent,
    )
    from ltx_pipelines.utils.media_io.resize import ResizeMode, align_resolution

    from ltx_outpaint.geometry import Rect

    if anchor.ndim != 4 or anchor.shape[1] != 3:
        raise ValueError("Anchor must be [F,3,H,W]")
    if min(anchor.shape[0], *anchor.shape[-2:]) < 1 or (anchor.shape[0] - 1) % scale_factors.time:
        raise ValueError("Anchor must contain a positive, VAE-aligned frame count; implicit temporal crop is forbidden")
    anchor_time, _ = map_temporal_interval_to_latent(0, anchor.shape[0], 0, 0, scale_factors.time)
    if (latent_shape.batch != 1 or latent_shape.frames != anchor_time.stop
            or latent_shape.height * scale_factors.height != boundary.encoded_height
            or latent_shape.width * scale_factors.width != boundary.encoded_width):
        raise ValueError("Condition and canonical latent canvas dimensions differ")
    if previous_prefix is not None:
        if previous_prefix.ndim != 4 or previous_prefix.shape[1] != 3:
            raise ValueError("Previous prefix must be [F,3,H,W]")
        if not 1 <= previous_prefix.shape[0] <= anchor.shape[0]:
            raise ValueError("Clean prefix must be nonempty and no longer than the current chunk")
        if tuple(previous_prefix.shape[-2:]) != (boundary.requested_height, boundary.requested_width):
            raise ValueError("Previous prefix must cover the entire requested canvas")
        if (previous_prefix.shape[0] - 1) % scale_factors.time:
            raise ValueError("Previous prefix frame count is not VAE-temporally aligned")
    if prefix_latent_override is not None and previous_prefix is None:
        raise ValueError("Latent carry requires an RGB prefix for exact visible output")
    if not math.isfinite(generated_prefix_strength) or not 0.0 <= generated_prefix_strength <= 1.0:
        raise ValueError("generated_prefix_strength must be finite and in [0,1]")
    y0, x0, y1, x1 = anchor_rect
    if (y1 - y0, x1 - x0) != tuple(anchor.shape[-2:]):
        raise ValueError("Anchor rectangle and RGB dimensions differ; resizing is forbidden")
    if y0 < 0 or x0 < 0 or y1 > boundary.encoded_height or x1 > boundary.encoded_width:
        raise ValueError("Anchor rectangle is outside the encoded canvas")

    sy, sx = scale_factors.height, scale_factors.width
    leading_top, leading_left = y0 % sy, x0 % sx
    anchor_height, anchor_width = anchor.shape[-2:]
    encoded_height = align_resolution(
        anchor_width, anchor_height + leading_top, ResizeMode.REFLECT_PAD, sy
    )[1]
    encoded_width = align_resolution(
        anchor_width + leading_left, anchor_height, ResizeMode.REFLECT_PAD, sx
    )[0]
    bottom = encoded_height - anchor_height - leading_top
    right = encoded_width - anchor_width - leading_left
    encoded_y0, encoded_x0 = y0 - leading_top, x0 - leading_left
    encoded_y1, encoded_x1 = encoded_y0 + encoded_height, encoded_x0 + encoded_width
    if encoded_y1 > boundary.encoded_height or encoded_x1 > boundary.encoded_width:
        raise ValueError("Aligned Anchor encoding extends beyond the canonical canvas")

    anchor_input = _reflect_pad_rgb(
        anchor, top=leading_top, bottom=bottom, left=leading_left, right=right
    )
    anchor_latent = _encode_condition_rgb(
        anchor_input, vae_encoder, runner, device, training_aligned_anchor=True
    ).to(device=device, dtype=dtype)
    expected_anchor = (
        1,
        latent_shape.channels,
        latent_shape.frames,
        encoded_height // sy,
        encoded_width // sx,
    )
    if tuple(anchor_latent.shape) != expected_anchor:
        raise RuntimeError("Independent Anchor latent does not map to canonical geometry")
    # The aligned origin follows the native VAE lattice. Ends come ONLY from
    # the actual encoded tensor, never from floor-rounded RGB content ends.
    latent_y0, latent_x0 = encoded_y0 // sy, encoded_x0 // sx
    anchor_h, anchor_w = anchor_latent.shape[-2:]
    latent_anchor_rect = Rect(
        latent_y0, latent_x0, latent_y0 + anchor_h, latent_x0 + anchor_w
    )

    condition = torch.zeros(
        latent_shape.to_torch_shape(), device=device, dtype=dtype
    )
    condition[
        :, :, :, latent_anchor_rect.y0 : latent_anchor_rect.y1,
        latent_anchor_rect.x0 : latent_anchor_rect.x1,
    ] = anchor_latent

    if previous_prefix is not None:
        prefix_slice, _ = map_temporal_interval_to_latent(
            0, previous_prefix.shape[0], 0, 0, scale_factors.time
        )
        expected_prefix = (
            1,
            condition.shape[1],
            prefix_slice.stop - prefix_slice.start,
            latent_shape.height,
            latent_shape.width,
        )
        if prefix_latent_override is None:
            prefix_input = _reflect_pad_rgb(
                previous_prefix,
                top=boundary.top,
                bottom=boundary.encoded_height - boundary.bottom,
                left=boundary.left,
                right=boundary.encoded_width - boundary.right,
            )
            prefix_latent = _encode_condition_rgb(
                prefix_input, vae_encoder, runner, device
            ).to(device=device, dtype=dtype)
        else:
            if prefix_latent_override.device.type != "cpu" or prefix_latent_override.dtype != dtype:
                raise TypeError("Carried latent must be CPU and match inference dtype")
            prefix_latent = prefix_latent_override.to(device=device)
        if tuple(prefix_latent.shape) != expected_prefix:
            raise RuntimeError("Independent prefix latent does not map to canonical geometry")
        if prefix_latent_override is not None:
            # The first latent of a causal VAE clip is a single-frame keyframe.
            # The corresponding carried slice covers eight frames, so only this
            # first slice must be encoded afresh; the other two remain direct
            # final-clean-latent carry with matching temporal support.
            first_input = _reflect_pad_rgb(
                previous_prefix[:1],
                top=boundary.top,
                bottom=boundary.encoded_height - boundary.bottom,
                left=boundary.left,
                right=boundary.encoded_width - boundary.right,
            )
            first_latent = _encode_condition_rgb(
                first_input, vae_encoder, runner, device
            ).to(device=device, dtype=dtype)
            if tuple(first_latent.shape) != (*expected_prefix[:2], 1, *expected_prefix[3:]):
                raise RuntimeError("Single-frame causal prefix keyframe shape mismatch")
            prefix_latent = prefix_latent.clone()
            prefix_latent[:, :, :1] = first_latent
        condition[:, :, prefix_slice] = prefix_latent
        if prefix_latent_override is not None:
            # Hybrid carry: generated region is direct final latent; the exact
            # current anchor encoding remains authoritative inside its known box.
            condition[
                :, :, prefix_slice,
                latent_anchor_rect.y0:latent_anchor_rect.y1,
                latent_anchor_rect.x0:latent_anchor_rect.x1,
            ] = anchor_latent[:, :, prefix_slice]
        if generated_prefix_strength != 1.0:
            # Upstream strength=0.8 would set denoise_mask=0.2 and regenerate
            # the prefix. Attenuate only the fixed side-context latent instead.
            condition[:, :, prefix_slice, :, :latent_anchor_rect.x0].mul_(generated_prefix_strength)
            condition[:, :, prefix_slice, :, latent_anchor_rect.x1:].mul_(generated_prefix_strength)
    return condition, latent_anchor_rect


def _known_mask(
    *,
    batch: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    anchor_rect: Rect,
    prefix_frames: int,
    temporal_factor: int,
    device: torch.device,
) -> torch.Tensor:
    """Official mask semantics: 1 clean/known, 0 noised/generated.

    Reuse the actual encoded Anchor placement; RGB content and padded latent
    support have different ends and must not be converted independently here.
    """
    if anchor_rect.y1 > latent_height or anchor_rect.x1 > latent_width:
        raise ValueError("Anchor rectangle is outside the latent mask canvas")
    mask = torch.zeros(
        (batch, latent_frames, latent_height, latent_width), device=device
    )
    mask[:, :, anchor_rect.y0 : anchor_rect.y1, anchor_rect.x0 : anchor_rect.x1] = 1.0
    if prefix_frames:
        prefix_latent_frames = (prefix_frames - 1) // temporal_factor + 1
        mask[:, :prefix_latent_frames] = 1.0
    return mask


def _canvas_geometry(
    latent_shape: Any, anchor_rect: Rect
) -> Any:
    from ltx_outpaint.geometry import CanvasGeometry

    _, _, frames, height, width = latent_shape.to_torch_shape()
    return CanvasGeometry(frames=frames, height=height, width=width, anchors=(anchor_rect,))


def _inference_boundary_blend_width(settings: dict[str, Any]) -> int:
    """Read the inference-only RGB feather width; missing config defaults to 16px."""
    width = settings["video"].get("boundary_blend_width_pixels", 16)
    if type(width) is not int or width < 0:
        raise ValueError("video.boundary_blend_width_pixels must be a non-negative integer")
    return width


def _inward_anchor_boundary_alpha(
    *,
    anchor_bbox: tuple[int, int, int, int],
    canvas_height: int,
    canvas_width: int,
    boundary_blend_width_pixels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return one 2D smoothstep alpha that feathers only exposed Anchor edges inward."""
    if type(boundary_blend_width_pixels) is not int or boundary_blend_width_pixels <= 0:
        raise ValueError("boundary_blend_width_pixels must be a positive integer")
    if not dtype.is_floating_point:
        raise TypeError("Anchor boundary alpha requires a floating-point output dtype")

    y0, x0, y1, x1 = anchor_bbox
    if (
        canvas_height < 1
        or canvas_width < 1
        or y0 < 0
        or x0 < 0
        or y1 <= y0
        or x1 <= x0
        or y1 > canvas_height
        or x1 > canvas_width
    ):
        raise ValueError("Anchor output bbox must be a nonempty rectangle inside the canvas")

    anchor_height, anchor_width = y1 - y0, x1 - x0
    active_top = y0 > 0
    active_bottom = y1 < canvas_height
    active_left = x0 > 0
    active_right = x1 < canvas_width
    if not any((active_top, active_bottom, active_left, active_right)):
        return torch.ones((anchor_height, anchor_width), device=device, dtype=dtype)

    # Compute in FP32, then return in the decode output dtype. One nearest-edge
    # distance field avoids order-dependent or repeated blending at corners.
    rows = torch.arange(anchor_height, device=device, dtype=torch.float32)[:, None]
    columns = torch.arange(anchor_width, device=device, dtype=torch.float32)[None, :]
    distance = torch.full(
        (anchor_height, anchor_width), float("inf"), device=device, dtype=torch.float32
    )
    if active_top:
        distance = torch.minimum(distance, rows)
    if active_bottom:
        distance = torch.minimum(distance, anchor_height - 1 - rows)
    if active_left:
        distance = torch.minimum(distance, columns)
    if active_right:
        distance = torch.minimum(distance, anchor_width - 1 - columns)

    normalized = (distance / boundary_blend_width_pixels).clamp_(0.0, 1.0)
    alpha = normalized.square() * (3.0 - 2.0 * normalized)
    return alpha.to(dtype=dtype)


def _composite_anchor_rgb(
    decoded: torch.Tensor,
    anchor_frames: torch.Tensor,
    *,
    anchor_bbox: tuple[int, int, int, int],
    boundary_blend_width_pixels: int,
) -> torch.Tensor:
    """Composite original Anchor RGB, preserving the exact legacy path at width zero."""
    y0, x0, y1, x1 = anchor_bbox
    anchor_region = decoded[:, :, y0:y1, x0:x1]
    if boundary_blend_width_pixels == 0:
        # Keep the production-training validation default bit-for-bit equivalent
        # to the historical hard composite.
        anchor_region[...] = anchor_frames
        return decoded
    if type(boundary_blend_width_pixels) is not int or boundary_blend_width_pixels < 0:
        raise ValueError("boundary_blend_width_pixels must be a non-negative integer")
    if decoded.ndim != 4 or anchor_frames.ndim != 4 or anchor_region.shape != anchor_frames.shape:
        raise ValueError("Anchor RGB and output bbox must have identical [F,C,H,W] shapes")

    source = anchor_frames.to(device=decoded.device, dtype=decoded.dtype)
    alpha = _inward_anchor_boundary_alpha(
        anchor_bbox=anchor_bbox,
        canvas_height=decoded.shape[-2],
        canvas_width=decoded.shape[-1],
        boundary_blend_width_pixels=boundary_blend_width_pixels,
        device=decoded.device,
        dtype=decoded.dtype,
    )
    blend_pixels = alpha < 1.0
    if not bool(blend_pixels.any().item()):
        anchor_region.copy_(source)
        return decoded

    # Advanced indexing snapshots only the narrow band before the exact Anchor
    # assignment. The single lerp uses the one 2D alpha for all frames/channels.
    blended_band = torch.lerp(
        anchor_region[..., blend_pixels],
        source[..., blend_pixels],
        alpha[blend_pixels],
    )
    anchor_region.copy_(source)
    anchor_region[..., blend_pixels] = blended_band
    return decoded


@torch.inference_mode()
def sample_chunk(
    *,
    transformer: Any,
    runner: Any,
    vae_encoder: torch.nn.Module,
    anchor_frames: torch.Tensor,
    previous_prefix: torch.Tensor | None,
    boundary: PixelBoundary,
    anchor_rect: tuple[int, int, int, int],
    absolute_start_frame: int,
    runtime_metrics: dict[str, Any] | None = None,
    boundary_blend_width_pixels: int = 0,
    allow_nonempty_text: bool = False,
    vae_decode_streaming: bool = False,
    preview_step: int | None = None,
    preview_frame_indices: tuple[int, ...] | None = None,
    progress_context: dict[str, int] | None = None,
    final_latent_observer: Callable[[torch.Tensor], None] | None = None,
    rebase_temporal_position: bool = False,
    prefix_latent_override: torch.Tensor | None = None,
    generated_prefix_strength: float = 1.0,
    latent_carry_out: list[torch.Tensor] | None = None,
    latent_carry_frames: int | None = None,
) -> torch.Tensor:
    """Sample one config-sized chunk through the official LTX Euler/CFG/STG loop.

    A preview request keeps the configured full inference schedule but stops at
    its one-based ``preview_step`` and decodes only the requested frames from
    that step's guided predicted-clean latent.  Normal inference is unchanged.
    """
    if (preview_step is None) != (preview_frame_indices is None):
        raise ValueError(
            "preview_step and preview_frame_indices must be supplied together"
        )
    if allow_nonempty_text:
        _validate_inference_text_conditioning(runner._config)
    else:
        _validate_empty_text_conditioning(runner._config)
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.mask_cond import VideoConditionByMask
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape, VideoPixelShape

    device = next(transformer.parameters()).device
    dtype = next(transformer.parameters()).dtype
    sf = runner._video_scale_factors
    frames = anchor_frames.shape[0]
    pixel_shape = VideoPixelShape(
        batch=1,
        frames=frames,
        height=boundary.encoded_height,
        width=boundary.encoded_width,
        fps=runner._config.frame_rate,
    )
    latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, scale_factors=sf)
    condition_args = (
        anchor_frames, previous_prefix, boundary, anchor_rect, latent_shape,
        sf, vae_encoder, runner, device, dtype,
    )
    if prefix_latent_override is None and generated_prefix_strength == 1.0:
        condition_latent, latent_anchor_rect = _condition_latents(*condition_args)
    else:
        condition_latent, latent_anchor_rect = _condition_latents(
            *condition_args,
            prefix_latent_override=prefix_latent_override,
            generated_prefix_strength=generated_prefix_strength,
        )
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=latent_shape,
        fps=runner._config.frame_rate,
        scale_factors=sf,
        causal_fix=True,
    )
    state = tools.create_initial_state(device=device, dtype=dtype)
    prefix_count = 0 if previous_prefix is None else previous_prefix.shape[0]
    mask = _known_mask(
        batch=condition_latent.shape[0],
        latent_frames=latent_shape.frames,
        latent_height=latent_shape.height,
        latent_width=latent_shape.width,
        anchor_rect=latent_anchor_rect,
        prefix_frames=prefix_count,
        temporal_factor=sf.time,
        device=device,
    )
    state = VideoConditionByMask(condition_latent, mask, strength=1.0).apply_to(
        state, tools
    )
    # LTX positions are pixel-coordinate bounds in seconds. Preserve absolute
    # source time across LongRange chunks instead of restarting each chunk at 0.
    # C0 retains even the legacy +0 operation; only C1+ may rebase.
    if absolute_start_frame == 0 or not rebase_temporal_position:
        state.positions[:, 0] += absolute_start_frame / runner._config.frame_rate
    clean_state = state
    seed = int(runner._config.seed) + absolute_start_frame
    state = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seed))(
        state, noise_scale=1.0
    )

    runner.geometry = _canvas_geometry(latent_shape, latent_anchor_rect)
    embeddings = runner._cached_embeddings[0]
    v_pos, _a_pos, v_neg, _a_neg = runner._get_prompt_embeddings(embeddings, device)
    if preview_step is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    denoise_started = time.monotonic()
    state, _ = runner._run_denoising(
        transformer=transformer,
        video_state=state,
        audio_state=None,
        video_clean=clean_state,
        audio_clean=None,
        video_frozen=False,
        audio_frozen=False,
        v_ctx_pos=v_pos,
        a_ctx_pos=torch.empty(0, device=device, dtype=dtype),
        v_ctx_neg=v_neg,
        a_ctx_neg=None,
        device=device,
        sampling_ctx=_SamplingProgress(
            chunk_index=progress_context.get("chunk_index") if progress_context else None,
            chunk_count=progress_context.get("chunk_count") if progress_context else None,
            total_steps=int(runner._config.inference_steps) if progress_context else None,
        ),
        stop_after_step=preview_step,
    )
    if preview_step is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    denoise_elapsed = time.monotonic() - denoise_started
    capture_metadata: dict[str, Any] = {}
    if preview_step is not None:
        capture = runner.predicted_clean_capture
        if capture is None or capture.get("step") != preview_step:
            raise RuntimeError("Native denoising loop did not capture the requested x0")
        captured_sigmas = capture["sigma_schedule"]
        if len(captured_sigmas) != int(runner._config.inference_steps) + 1:
            raise RuntimeError("Captured x0 did not use the full configured sigma schedule")
        capture_metadata = {
            "preview_loop_index_zero_based": capture["loop_index_zero_based"],
            "preview_sigma": float(captured_sigmas[preview_step - 1]),
            "preview_next_sigma": float(captured_sigmas[preview_step]),
            "captured_sigma_schedule_length": len(captured_sigmas),
            "preview_capture_source": capture["source"],
        }
    latent_finite = bool(torch.isfinite(state.latent).all().item())
    if not latent_finite:
        raise FloatingPointError("Denoised video latent contains NaN or Inf")
    if final_latent_observer is not None:
        final_latent_observer(state.latent)
    if latent_carry_out is not None:
        prefix_frames = latent_carry_frames if latent_carry_frames is not None else (
            0 if previous_prefix is None else previous_prefix.shape[0]
        )
        if prefix_frames < 1 or prefix_frames > anchor_frames.shape[0] or (prefix_frames - 1) % sf.time:
            raise ValueError("Carry frame count must be nonempty, aligned and within the working clip")
        prefix_latent_frames = (prefix_frames - 1) // sf.time + 1
        final_clean = tools.unpatchify(tools.clear_conditioning(state)).latent
        carry = final_clean[:, :, -prefix_latent_frames:].detach().to("cpu").clone()
        if carry.dtype != dtype or not bool(torch.isfinite(carry).all().item()):
            raise RuntimeError("Final clean latent carry has incompatible dtype or values")
        latent_carry_out.append(carry)
        del final_clean, carry
    preview_decode_metrics: dict[str, Any] = {}
    decode_fn = runner._decode_video
    if preview_frame_indices is not None:
        from ltx_outpaint.vae_decode_streaming import decode_video_frames_streaming

        decode_fn = lambda video_state, target_device: decode_video_frames_streaming(
            runner,
            video_state,
            target_device,
            preview_frame_indices,
            metrics=preview_decode_metrics,
        )
    elif vae_decode_streaming:
        from ltx_outpaint.vae_decode_streaming import decode_video_streaming

        decode_fn = lambda video_state, target_device: decode_video_streaming(
            runner, video_state, target_device
        )
    if preview_step is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_started = time.monotonic()
    decoded = runner._finalize_modality(state, tools, decode_fn, device).permute(1, 0, 2, 3)
    if preview_step is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_elapsed = time.monotonic() - decode_started
    decoded = decoded[
        :, :, boundary.top : boundary.bottom, boundary.left : boundary.right
    ]

    # Restore observed RGB only at the final output. Production inference may
    # feather exposed edges inward; the shared training-validation default stays
    # the historical width-zero hard composite. Transformer hidden is unchanged.
    out_y0 = anchor_rect[0] - boundary.top
    out_x0 = anchor_rect[1] - boundary.left
    out_y1 = out_y0 + anchor_frames.shape[-2]
    out_x1 = out_x0 + anchor_frames.shape[-1]
    composite_source = (
        anchor_frames
        if preview_frame_indices is None
        else anchor_frames[list(preview_frame_indices)]
    )
    decoded = _composite_anchor_rgb(
        decoded,
        composite_source,
        anchor_bbox=(out_y0, out_x0, out_y1, out_x1),
        boundary_blend_width_pixels=boundary_blend_width_pixels,
    )
    if previous_prefix is not None and preview_frame_indices is None:
        decoded[: previous_prefix.shape[0]] = previous_prefix
    elif previous_prefix is not None and preview_frame_indices is not None:
        for output_index, frame_index in enumerate(preview_frame_indices):
            if frame_index < previous_prefix.shape[0]:
                decoded[output_index] = previous_prefix[frame_index]
    decoded_finite = bool(torch.isfinite(decoded).all().item())
    if not decoded_finite:
        raise FloatingPointError("Decoded video chunk contains NaN or Inf")
    if runtime_metrics is not None:
        runtime_metrics.update(
            {
                "anchor_rgb_shape": list(anchor_frames.shape),
                "condition_latent_shape": list(condition_latent.shape),
                "full_generation_latent_shape": list(latent_shape.to_torch_shape()),
                "known_anchor_rect_yxyx": [
                    latent_anchor_rect.y0,
                    latent_anchor_rect.x0,
                    latent_anchor_rect.y1,
                    latent_anchor_rect.x1,
                ],
                "known_mask_shape": list(mask.shape),
                "known_mask_elements": int(mask.sum().item()),
                "conditioning_prefix_frames": prefix_count,
                "rebase_temporal_position": bool(rebase_temporal_position and absolute_start_frame > 0),
                "latent_carry_applied": prefix_latent_override is not None,
                "generated_prefix_strength": generated_prefix_strength,

                "chunk_seed": seed,
                "denoised_latent_finite": latent_finite,
                "decoded_chunk_shape": list(decoded.shape),
                "decoded_chunk_finite": decoded_finite,
                "boundary_blend_width_pixels": boundary_blend_width_pixels,
                "denoise_elapsed_seconds": denoise_elapsed,
                "vae_decode_elapsed_seconds": decode_elapsed,
                "preview_step": preview_step,
                "preview_frame_indices": (
                    None
                    if preview_frame_indices is None
                    else list(preview_frame_indices)
                ),
                "preview_latent_source": (
                    None
                    if preview_step is None
                    else "guided_predicted_clean_after_cfg_stg_and_conditioning_before_euler"
                ),
                **capture_metadata,
                **preview_decode_metrics,
            }
        )
    return decoded.cpu()


def _atomic_save_preview_png(frame: torch.Tensor, path: Path) -> None:
    """Commit one CPU RGB preview without exposing a partial PNG."""
    from ltx_trainer.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    try:
        save_image(frame, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def sample_seed_preview(
    settings: dict[str, Any],
    anchor_video: Path,
    output_dir: Path,
    *,
    prompt: str | None = "",
    num_frames: int,
    checkpoint_path: Path | None = None,
    initialized_adapters: bool = False,
    seed: int | None = None,
    inference_steps: int | None = None,
    preview_step: int = _SEED_PREVIEW_STEP,
    preview_frames: tuple[int, ...] = _SEED_PREVIEW_FRAMES,
    early_global_boost: bool = False,
    negative_prompt: str | None = "",
) -> Path:
    """Save step-10 x0 previews for one exact 65-frame seed-search job.

    This is the seed-search terminal path: it constructs the normal 30-step
    scheduler, evaluates only its first ten model steps, decodes frame
    0/32/64 from the step-10 guided predicted-clean latent, and never creates
    video segments or an MP4.
    """
    prompt = _normalize_inference_prompt(prompt, "prompt")
    negative_prompt = _normalize_inference_prompt(negative_prompt, "negative_prompt")
    if inference_steps != _SEED_PREVIEW_FULL_STEPS:
        raise ValueError(
            "Seed preview requires --steps 30; reducing num_inference_steps is forbidden"
        )
    if preview_step != _SEED_PREVIEW_STEP:
        raise ValueError("Seed preview requires preview_step=10")
    if preview_frames != _SEED_PREVIEW_FRAMES:
        raise ValueError("Seed preview requires frames 0, 32, and 64")

    memory_optimization = _memory_optimization_settings(settings)
    if not memory_optimization["vae_decode_streaming"]:
        raise ValueError("Seed preview requires streaming VAE decode")
    _install_official_sources(Path(settings["ltx_repo"]).expanduser().resolve())
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_trainer.model_loader import read_video_scale_factors

    from ltx_outpaint.streaming_video import (
        StreamingAnchorReader,
        file_sha256,
        probe_video,
    )

    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    anchor_video = anchor_video.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_metadata = probe_video(anchor_video)
    expected_fps = float(settings["video"]["fps"])
    tolerance = float(settings["video"].get("fps_tolerance", 1e-3))
    if abs(input_metadata.fps - expected_fps) > tolerance:
        raise ValueError(
            f"Input FPS {input_metadata.fps:g} does not match configured FPS "
            f"{expected_fps:g}; resampling is disabled"
        )

    vae_path = Path(
        settings["model"].get("video_vae_path")
        or settings["model"]["base_model_path"]
    ).expanduser().resolve()
    scale_factors = read_video_scale_factors(vae_path)
    width, height = map(int, settings["video"]["canvas"])
    boundary = _vae_boundary(
        height, width, scale_factors.height, scale_factors.width
    )
    anchor_rect = _center_anchor(
        boundary, input_metadata.height, input_metadata.width
    )
    expected_anchor = settings["video"].get("expected_anchor")
    if settings["video"].get("enforce_expected_anchor", False) and expected_anchor != [
        input_metadata.width,
        input_metadata.height,
    ]:
        raise ValueError(
            f"Input anchor is {input_metadata.width}x{input_metadata.height}, "
            f"expected {expected_anchor}; resizing is disabled"
        )

    long_range = LongRangeSpec(**settings["long_range"])
    long_range.validate(scale_factors.time)
    if num_frames != long_range.clip_frames or num_frames != 65:
        raise ValueError("Seed preview requires normal_long_range with exactly 65 frames")
    if input_metadata.frame_count < num_frames:
        raise ValueError(
            f"Anchor video has {input_metadata.frame_count} frames but 65 are required"
        )

    validation_config = _make_validation_config(
        settings,
        prompt,
        boundary,
        long_range.clip_frames,
        negative_prompt=negative_prompt,
    )
    if seed is not None:
        validation_config.seed = seed
    validation_config.inference_steps = inference_steps

    # This is the same scheduler class and the same configured step count used
    # inside ValidationRunner._run_denoising.  It deliberately contains 31
    # sigma boundaries for 30 model evaluations.
    sigmas = (
        LTX2Scheduler()
        .execute(steps=int(validation_config.inference_steps))
        .float()
        .cpu()
    )
    if len(sigmas) != _SEED_PREVIEW_FULL_STEPS + 1:
        raise RuntimeError("The 30-step LTX schedule did not produce 31 sigmas")

    model_load_started = time.monotonic()
    transformer, runner, vae_encoder, adapter_report = _load_models(
        settings,
        validation_config,
        boundary,
        checkpoint_path=checkpoint_path,
        initialized_adapters=initialized_adapters,
        early_global_boost=early_global_boost,
        memory_optimization=memory_optimization,
    )
    model_load_seconds = time.monotonic() - model_load_started
    print(
        json.dumps(
            {
                "event": "seed_preview_model_ready",
                "seed": int(validation_config.seed),
                "num_frames": num_frames,
                "full_schedule_steps": int(validation_config.inference_steps),
                "preview_step": preview_step,
                "preview_frames": list(preview_frames),
                "negative_prompt": validation_config.negative_prompt,
                "adapter": adapter_report,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    record = next(
        _iter_long_range_chunks(
            num_frames, long_range, temporal_factor=scale_factors.time
        )
    )
    with StreamingAnchorReader(
        anchor_video, stride_frames=long_range.stride_frames
    ) as anchor_reader:
        anchor_frames = anchor_reader.read_chunk(record)
    if anchor_frames.shape[0] != num_frames:
        raise RuntimeError("Seed preview source reader did not return exactly 65 frames")

    runtime_metrics: dict[str, Any] = {}
    preview_tensor = sample_chunk(
        transformer=transformer,
        runner=runner,
        vae_encoder=vae_encoder,
        anchor_frames=anchor_frames,
        previous_prefix=None,
        boundary=boundary,
        anchor_rect=anchor_rect,
        absolute_start_frame=0,
        runtime_metrics=runtime_metrics,
        boundary_blend_width_pixels=_inference_boundary_blend_width(settings),
        allow_nonempty_text=True,
        vae_decode_streaming=True,
        preview_step=preview_step,
        preview_frame_indices=preview_frames,
    )
    if not torch.equal(runner.predicted_clean_capture["sigma_schedule"], sigmas):
        raise RuntimeError(
            "Captured preview schedule differs from the configured 30-step schedule"
        )
    if tuple(preview_tensor.shape) != (len(preview_frames), 3, height, width):
        raise RuntimeError(
            "Preview tensor has unexpected shape "
            f"{tuple(preview_tensor.shape)}; expected "
            f"({len(preview_frames)}, 3, {height}, {width})"
        )

    save_started = time.monotonic()
    output_records: dict[str, Any] = {}
    for tensor_index, frame_index in enumerate(preview_frames):
        path = output_dir / f"f{frame_index:03d}.png"
        _atomic_save_preview_png(preview_tensor[tensor_index], path)
        output_records[f"f{frame_index:03d}"] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    png_save_seconds = time.monotonic() - save_started

    checkpoint_value = checkpoint_path or Path(settings["model"]["adapter_path"])
    checkpoint_value = checkpoint_value.expanduser().resolve()
    metadata = {
        "schema_version": 1,
        "status": "success",
        "semantic": "seed_search_step10_predicted_clean_preview",
        "input_path": str(anchor_video),
        "input_sha256": file_sha256(anchor_video),
        "checkpoint_path": str(checkpoint_value),
        "adapter_load": adapter_report,
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": int(validation_config.seed),
        "mode": "normal_long_range",
        "num_frames": num_frames,
        "resolution": [width, height],
        "fps": expected_fps,
        "full_schedule_steps": int(validation_config.inference_steps),
        "sigma_schedule": [float(value) for value in sigmas.tolist()],
        "preview_step": preview_step,
        "preview_loop_index_zero_based": preview_step - 1,
        "preview_sigma": float(sigmas[preview_step - 1]),
        "preview_next_sigma": float(sigmas[preview_step]),
        "preview_frames": list(preview_frames),
        "preview_latent_source": (
            "guided_predicted_clean_after_cfg_stg_and_conditioning_before_euler"
        ),
        "outputs": output_records,
        "full_mp4_generated": False,
        "partial_decode": {
            "random_access_supported": False,
            "temporal_traversal_required_through_frame": max(preview_frames),
            "full_decoded_video_materialized": False,
            "retained_frames_only": True,
        },
        "runtime": {
            "started_at": started_at,
            "ended_at": datetime.now(UTC).isoformat(),
            "model_load_seconds": model_load_seconds,
            "png_save_seconds": png_save_seconds,
            "wall_time_seconds": time.monotonic() - started,
            **runtime_metrics,
        },
    }
    _write_json_atomic(output_dir / "metadata.json", metadata)
    print(
        json.dumps(
            {
                "event": "seed_preview_complete",
                "output_dir": str(output_dir),
                "seed": int(validation_config.seed),
                "outputs": {
                    key: value["path"] for key, value in output_records.items()
                },
                "wall_time_seconds": metadata["runtime"]["wall_time_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return output_dir


def _capture_post_chunk_rng(
    device_name: str,
) -> tuple[torch.Tensor, torch.Tensor | None, str | None]:
    """Capture RNG consumed by denoising/VAE decode without changing seed semantics."""
    cpu_state = torch.get_rng_state().clone()
    device = torch.device(device_name)
    if device.type != "cuda":
        return cpu_state, None, None
    cuda_state = torch.cuda.get_rng_state(device).clone()
    return cpu_state, cuda_state, str(device)


def _restore_post_chunk_rng(
    cpu_state: torch.Tensor,
    cuda_state: torch.Tensor | None,
    cuda_device: str | None,
    *,
    expected_device: str,
) -> None:
    """Restore the exact post-chunk stream before the next resumed chunk."""
    torch.set_rng_state(cpu_state)
    device = torch.device(expected_device)
    if device.type == "cuda":
        if cuda_state is None or cuda_device != str(device):
            raise RuntimeError("Resume CUDA RNG state does not match runtime.device")
        torch.cuda.set_rng_state(cuda_state, device)
    elif cuda_state is not None or cuda_device is not None:
        raise RuntimeError("Resume state contains CUDA RNG for a non-CUDA run")


def _streaming_run_identity(
    *,
    settings: dict[str, Any],
    anchor_video: Path,
    output_path: Path,
    input_frame_count: int,
    input_fps: float,
    input_width: int,
    input_height: int,
    requested_frames: int,
    checkpoint_path: Path | None,
    initialized_adapters: bool,
    prompt: str,
    negative_prompt: str,
    seed: int,
    inference_steps: int,
    early_global_boost: bool,
    initial_prefix_artifact: dict[str, Any] | None = None,
    bootstrap_first_chunk_path: Path | None = None,
) -> dict[str, Any]:
    from ltx_outpaint.streaming_video import file_sha256, session_fingerprint

    stat = anchor_video.stat()
    generation_settings = {
        key: value
        for key, value in settings.items()
        if key not in ("long_range_io", "memory_optimization")
    }
    memory_optimization = _memory_optimization_settings(settings)
    if any(memory_optimization[name] for name in (
        "video_only_gpu_weights", "vae_decode_streaming", "ffn_token_chunking"
    )):
        generation_settings["memory_optimization"] = memory_optimization
    model_settings = settings["model"]
    base_path = Path(model_settings["base_model_path"]).expanduser().resolve()
    vae_value = model_settings.get("video_vae_path")
    vae_path = (
        Path(vae_value).expanduser().resolve() if vae_value else base_path
    )
    adapter_path = None
    if not initialized_adapters:
        configured_adapter = checkpoint_path or Path(model_settings["adapter_path"])
        adapter_path = _resolve_adapter_path(configured_adapter)

    def artifact(
        path: Path | None, *, content_sha256: str | None = None
    ) -> dict[str, Any] | None:
        if path is None:
            return None
        artifact_stat = path.stat()
        return {
            "path": str(path),
            "size_bytes": artifact_stat.st_size,
            "mtime_ns": artifact_stat.st_mtime_ns,
            "content_sha256": content_sha256,
        }

    base_sha256 = str(settings["provenance"]["base_sha256"])
    vae_sha256 = base_sha256 if vae_path == base_path else file_sha256(vae_path)

    request = {
        "settings": generation_settings,
        "input_path": str(anchor_video),
        "input_size_bytes": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "input_sha256": file_sha256(anchor_video),
        "input_frame_count": input_frame_count,
        "input_fps": input_fps,
        "input_width": input_width,
        "input_height": input_height,
        "requested_frames": requested_frames,
        "output_path": str(output_path),
        "checkpoint_argument": (
            None if checkpoint_path is None else str(checkpoint_path.expanduser().resolve())
        ),
        "base_model_artifact": artifact(base_path, content_sha256=base_sha256),
        "video_vae_artifact": artifact(vae_path, content_sha256=vae_sha256),
        "adapter_artifact": artifact(
            adapter_path,
            content_sha256=(None if adapter_path is None else file_sha256(adapter_path)),
        ),
        "initialized_adapters": initialized_adapters,
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "inference_steps": inference_steps,
        "early_global_boost": early_global_boost,
    }
    if initial_prefix_artifact is not None:
        # Absent for normal LongRange: retain the exact pre-prefix resume identity.
        request["initial_prefix_artifact"] = initial_prefix_artifact
    if bootstrap_first_chunk_path is not None:
        request["bootstrap_first_chunk_artifact"] = artifact(
            bootstrap_first_chunk_path,
            content_sha256=file_sha256(bootstrap_first_chunk_path),
        )
    return {
        "request_fingerprint": session_fingerprint(request),
        "request": request,
    }


def _online_lineage_end(
    records: Iterator[dict[str, Any]], *, expected_start: int = 0
) -> int:
    """Validate contiguous output intervals with constant memory."""
    next_frame = expected_start
    for record in records:
        begin, end = map(int, record["new_output_frame_range"])
        if begin != next_frame or end - begin != int(record["output_appended_count"]):
            raise RuntimeError(
                f"LongRange lineage is not contiguous at chunk {record['chunk_index']}"
            )
        next_frame = end
    return next_frame


def _save_certification_latent(
    path: Path, latent: torch.Tensor, *, record: dict[str, Any],
    seed: int, checkpoint: Path,
) -> None:
    """Persist the *actual* final denoised latent before the native VAE decode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    cpu_latent = latent.detach().to("cpu", copy=True)
    try:
        with temporary.open("wb") as stream:
            torch.save(cpu_latent, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256()
        with temporary.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        os.replace(temporary, path)
        _write_json_atomic(path.parent / "metadata.json", {
            "artifact": path.name,
            "sha256": digest.hexdigest(),
            "chunk_index": record["chunk_index"],
            "absolute_start_frame": record["absolute_start_frame"],
            "new_output_frame_range": record["new_output_frame_range"],
            "latent_shape": list(cpu_latent.shape),
            "latent_dtype": str(cpu_latent.dtype).removeprefix("torch."),
            "seed": seed,
            "checkpoint": str(checkpoint),
            "source": "sample_chunk.state.latent after native denoising, before native VAE decode",
        })
    finally:
        temporary.unlink(missing_ok=True)
        del cpu_latent


@torch.inference_mode()
def sample_long_range(
    settings: dict[str, Any],
    anchor_video: Path,
    output_path: Path,
    prompt: str | None = "",
    num_frames: int | None = None,
    checkpoint_path: Path | None = None,
    initialized_adapters: bool = False,
    seed: int | None = None,
    inference_steps: int | None = None,
    early_global_boost: bool = False,
    negative_prompt: str | None = "",
    *,
    segment_dir: Path | None = None,
    resume: bool = False,
    keep_intermediates: bool = False,
    initial_prefix_path: Path | None = None,
    bootstrap_first_chunk_path: Path | None = None,
    manifest_dir: Path | None = None,
    certification_latent_dir: Path | None = None,
    chunk_committed_observer: Callable[[dict[str, Any], Path], None] | None = None,
    equivalence_observer: Callable[
        [dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor], None
    ]
    | None = None,
) -> Path:
    """Run memory-bounded LongRange with legacy C0 and configured C1+ recurrence.

    ``equivalence_observer`` is an optional future-GPU diagnostic hook invoked
    before each chunk is released with the record, full generated FP32 tensor,
    next FP32 prefix, and emitted FP32 view. Its RNG effects are rolled back.
    The default production path does not retain diagnostic tensors.
    """
    from cert_pipeline.recurrence import DEFAULT_RECURRENCE, validate_recurrence
    from ltx_outpaint.recurrence_artifacts import load_carry, save_carry

    recurrence = validate_recurrence(settings.get("recurrence", DEFAULT_RECURRENCE))
    settings = dict(settings, recurrence=recurrence)
    enhanced_recurrence = (
        recurrence["rebase_temporal_position"] or recurrence["use_latent_carry"]
        or recurrence["lineart_global_prior"] or recurrence["generated_prefix_strength"] != 1.0
    )
    if enhanced_recurrence and (initial_prefix_path is not None or bootstrap_first_chunk_path is not None):
        raise ValueError("Enhanced recurrence starts from original C0; external prefix/bootstrap is unsupported")
    prompt = _normalize_inference_prompt(prompt, "prompt")
    negative_prompt = _normalize_inference_prompt(negative_prompt, "negative_prompt")
    memory_optimization = _memory_optimization_settings(settings)
    early_global_boost_metadata = _early_global_boost_metadata(early_global_boost)
    boundary_blend_width_pixels = _inference_boundary_blend_width(settings)
    _install_official_sources(Path(settings["ltx_repo"]).expanduser().resolve())
    from ltx_trainer.model_loader import read_video_scale_factors

    from ltx_outpaint.streaming_video import (
        SegmentedOutputStore,
        StreamingAnchorReader,
        encode_segments,
        probe_video,
        write_record_manifest,
    )

    started_at = time.monotonic()
    anchor_video = anchor_video.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_dir is None:
        manifest_dir = output_path.parent
    else:
        manifest_dir = manifest_dir.expanduser().resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    input_metadata = probe_video(anchor_video)
    input_total_frames = input_metadata.frame_count
    input_fps = input_metadata.fps
    expected_fps = float(settings["video"]["fps"])
    tolerance = float(settings["video"].get("fps_tolerance", 1e-3))
    if abs(input_fps - expected_fps) > tolerance:
        raise ValueError(
            f"Input FPS {input_fps:g} does not match configured FPS {expected_fps:g}; resampling is disabled"
        )
    requested_frames = input_total_frames if num_frames is None else num_frames
    if requested_frames <= 0 or input_total_frames < requested_frames:
        raise ValueError(
            f"Anchor video has {input_total_frames} frames but {requested_frames} were requested"
        )

    vae_path = Path(
        settings["model"].get("video_vae_path") or settings["model"]["base_model_path"]
    ).expanduser().resolve()
    sf = read_video_scale_factors(vae_path)
    width, height = map(int, settings["video"]["canvas"])
    boundary = _vae_boundary(height, width, sf.height, sf.width)
    anchor_rect = _center_anchor(
        boundary, input_metadata.height, input_metadata.width
    )
    expected_anchor = settings["video"].get("expected_anchor")
    if settings["video"].get("enforce_expected_anchor", False) and expected_anchor != [
        input_metadata.width,
        input_metadata.height,
    ]:
        raise ValueError(
            f"Input anchor is {input_metadata.width}x{input_metadata.height}, "
            f"expected {expected_anchor}; resizing is disabled"
        )

    long_range = LongRangeSpec(**settings["long_range"])
    long_range.validate(sf.time)
    if initial_prefix_path is not None and bootstrap_first_chunk_path is not None:
        raise ValueError(
            "initial_prefix_path and bootstrap_first_chunk_path are mutually exclusive"
        )
    if bootstrap_first_chunk_path is not None:
        if requested_frames < long_range.clip_frames:
            raise ValueError(
                "Bootstrap continuation requires at least one complete first chunk"
            )
        bootstrap_first_chunk_path = (
            bootstrap_first_chunk_path.expanduser().resolve()
        )
        if not bootstrap_first_chunk_path.is_file():
            raise FileNotFoundError(
                f"Bootstrap first chunk does not exist: {bootstrap_first_chunk_path}"
            )
    initial_prefix: torch.Tensor | None = None
    initial_prefix_artifact: dict[str, Any] | None = None
    if initial_prefix_path is not None:
        if requested_frames < long_range.prefix_frames:
            raise ValueError("Prefix continuation requires at least prefix_frames source frames")
        from ltx_outpaint.prefix_io import load_prefix_artifact
        from ltx_outpaint.streaming_video import file_sha256

        initial_prefix_path = initial_prefix_path.expanduser().resolve()
        initial_prefix, prefix_meta = load_prefix_artifact(
            initial_prefix_path,
            expected_frames=long_range.prefix_frames,
            height=height,
            width=width,
        )
        if prefix_meta.get("source_path") != str(anchor_video):
            raise ValueError("Initial prefix was generated from a different source path")
        if prefix_meta.get("source_sha256") != file_sha256(anchor_video):
            raise ValueError("Initial prefix source video hash does not match")
        if prefix_meta.get("source_frame_range") != [0, long_range.prefix_frames]:
            raise ValueError("Initial prefix must represent source frames [0,prefix_frames)")
        if float(prefix_meta.get("fps", -1)) != expected_fps:
            raise ValueError("Initial prefix FPS does not match inference config")
        configured_adapter = checkpoint_path or Path(settings["model"]["adapter_path"])
        if not initialized_adapters:
            adapter_sha256 = file_sha256(_resolve_adapter_path(configured_adapter))
            if prefix_meta.get("adapter_sha256") != adapter_sha256:
                raise ValueError("Initial prefix checkpoint adapter hash does not match")
        initial_prefix_artifact = {
            "path": str(initial_prefix_path),
            "tensor_sha256": prefix_meta["tensor_sha256"],
            "seed": prefix_meta["seed"],
            "source_sha256": prefix_meta["source_sha256"],
            "adapter_sha256": prefix_meta.get("adapter_sha256"),
        }
    validation_config = _make_validation_config(
        settings, prompt, boundary, long_range.clip_frames,
        negative_prompt=negative_prompt,
    )
    if seed is not None:
        validation_config.seed = seed
    if inference_steps is not None:
        if inference_steps < 1:
            raise ValueError("inference_steps must be positive")
        validation_config.inference_steps = inference_steps

    io_settings = settings.get("long_range_io", {})
    if not isinstance(io_settings, dict):
        raise TypeError("long_range_io must be a mapping when configured")
    for key in ("resume", "keep_intermediates"):
        if key in io_settings and type(io_settings[key]) is not bool:
            raise TypeError(f"long_range_io.{key} must be a boolean")
    if segment_dir is None and io_settings.get("segment_dir"):
        segment_dir = Path(io_settings["segment_dir"])
    if not resume:
        resume = bool(io_settings.get("resume", False))
    if not keep_intermediates:
        keep_intermediates = bool(io_settings.get("keep_intermediates", False))
    if segment_dir is None:
        segment_dir = output_path.parent / f".{output_path.name}.segments"
    else:
        segment_dir = segment_dir.expanduser().resolve()

    device_name = str(settings["runtime"]["device"])
    identity = _streaming_run_identity(
        settings=settings,
        anchor_video=anchor_video,
        output_path=output_path,
        input_frame_count=input_total_frames,
        input_fps=input_fps,
        input_width=input_metadata.width,
        input_height=input_metadata.height,
        requested_frames=requested_frames,
        checkpoint_path=checkpoint_path,
        initialized_adapters=initialized_adapters,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(validation_config.seed),
        inference_steps=int(validation_config.inference_steps),
        early_global_boost=early_global_boost,
        initial_prefix_artifact=initial_prefix_artifact,
        bootstrap_first_chunk_path=bootstrap_first_chunk_path,
    )
    planned_chunk_count = sum(
        1
        for _ in _iter_long_range_chunks(
            requested_frames, long_range, temporal_factor=sf.time
        )
    )
    store = SegmentedOutputStore(segment_dir)
    store.prepare(identity, resume=resume)
    completed_chunks, chunk_count = store.scan_completed(
        _iter_long_range_chunks(
            requested_frames, long_range, temporal_factor=sf.time
        ),
        output_height=height,
        output_width=width,
        base_seed=int(validation_config.seed),
    )
    if chunk_count != planned_chunk_count:
        raise RuntimeError("LongRange chunk plan changed during resume validation")
    # Missing new artifacts cannot be treated as a completed enhanced session.
    for completed in store.iter_records(completed_chunks):
        if recurrence["use_latent_carry"]:
            load_carry(store.root, completed,
                       expected_shape=(1, 128, 3, boundary.encoded_height // sf.height, boundary.encoded_width // sf.width),
                       expected_dtype=torch.bfloat16 if settings["runtime"]["dtype"] == "bfloat16" else torch.float32)
    emitted_frames = _online_lineage_end(store.iter_records(completed_chunks))
    resumed_chunks = completed_chunks

    lineart_prior = None
    lineart_sha = None
    lineart_directory = Path(certification_latent_dir).parent.parent / "lineart" if certification_latent_dir else store.root.parent / "lineart"

    def prepare_lineart(c0_record):
        from ltx_outpaint.lineart_reference import reference_from_segment
        from ltx_outpaint.lineart_global_prior import LineArtGlobalPrior
        from ltx_core.components.schedulers import LTX2Scheduler

        source_identity = {
            "input_sha256": identity["request"]["input_sha256"],
            "seed": int(validation_config.seed),
            "adapter_sha256": identity["request"]["adapter_artifact"]["content_sha256"],
            "c0_segment_pixel_sha256": c0_record["segment_pixel_sha256"],
        }
        strength, metadata, digest = reference_from_segment(
            lineart_directory, store.root / c0_record["segment_file"], source_identity)
        sigmas = tuple(float(x) for x in LTX2Scheduler().execute(steps=validation_config.inference_steps)[:-1])
        prior = LineArtGlobalPrior(strength, sigmas)
        print(json.dumps({"event": "lineart_reference_ready", "path": str(lineart_directory),
                          "metadata_sha256": digest, "start_chunk": 1,
                          "nonzero_source_cells": metadata["nonzero_source_cells"]}, sort_keys=True), flush=True)
        return prior, digest

    if recurrence["lineart_global_prior"] and completed_chunks:
        lineart_prior, lineart_sha = prepare_lineart(next(store.iter_records(completed_chunks)))
        for completed in store.iter_records(completed_chunks):
            if int(completed["chunk_index"]) > 0 and completed.get("lineart_metadata_sha256") != lineart_sha:
                raise RuntimeError("Committed chunk line-art identity mismatch")

    cuda_metrics = device_name.startswith("cuda") and torch.cuda.is_available()
    peak_allocated = peak_reserved = 0
    models_loaded = completed_chunks < chunk_count
    if models_loaded:
        if cuda_metrics:
            torch.cuda.reset_peak_memory_stats(torch.device(device_name))
        transformer, runner, vae_encoder, adapter_report = _load_models(
            settings,
            validation_config,
            boundary,
            checkpoint_path=checkpoint_path,
            initialized_adapters=initialized_adapters,
            early_global_boost=early_global_boost,
            memory_optimization=memory_optimization,
        )
        if cuda_metrics:
            metric_device = torch.device(device_name)
            peak_allocated = torch.cuda.max_memory_allocated(metric_device)
            peak_reserved = torch.cuda.max_memory_reserved(metric_device)
        store.record_model_ready(adapter_report)
    else:
        adapter_report = store.load_model_ready()
    print(
        json.dumps(
            {
                "event": "inference_model_ready",
                "adapter": adapter_report,
                "seed": int(validation_config.seed),
                "inference_steps": int(validation_config.inference_steps),
                "boundary_blend_width_pixels": boundary_blend_width_pixels,
                "early_global_boost": early_global_boost_metadata,
                "completed_chunks_resumed": completed_chunks,
                "model_load_skipped": not models_loaded,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    previous_prefix: torch.Tensor | None = (
        initial_prefix if models_loaded and not completed_chunks else None
    )
    del initial_prefix
    previous_carry = None
    if models_loaded and completed_chunks:
        resume_state = store.load_resume_state(
            completed_chunks - 1,
            prefix_frames=long_range.prefix_frames,
            output_height=height,
            output_width=width,
        )
        previous_prefix = resume_state.previous_prefix
        if recurrence["use_latent_carry"]:
            previous_carry = load_carry(store.root, list(store.iter_records(completed_chunks))[-1])
        _restore_post_chunk_rng(
            resume_state.cpu_rng_state,
            resume_state.cuda_rng_state,
            resume_state.cuda_device,
            expected_device=device_name,
        )
    elif models_loaded and bootstrap_first_chunk_path is not None:
        bootstrap_record = next(
            _iter_long_range_chunks(
                requested_frames, long_range, temporal_factor=sf.time
            )
        )
        if int(bootstrap_record["output_appended_count"]) != long_range.clip_frames:
            raise RuntimeError("Bootstrap chunk plan does not emit one complete clip")
        bootstrap_artifact = identity["request"]["bootstrap_first_chunk_artifact"]
        bootstrap_record["bootstrap_first_chunk"] = bootstrap_artifact
        bootstrap_record["runtime"] = {
            "source": "existing_first_chunk_mp4",
            "model_inference_skipped": True,
        }
        bootstrap_record["emitted_tensor_range"] = [0, long_range.clip_frames]
        cpu_rng_state, cuda_rng_state, cuda_device = _capture_post_chunk_rng(
            device_name
        )
        _committed, previous_prefix = store.commit_bootstrap_video(
            bootstrap_record,
            bootstrap_first_chunk_path,
            prefix_frames=long_range.prefix_frames,
            expected_fps=expected_fps,
            fps_tolerance=tolerance,
            output_height=height,
            output_width=width,
            chunk_seed=int(validation_config.seed),
            cpu_rng_state=cpu_rng_state,
            cuda_rng_state=cuda_rng_state,
            cuda_device=cuda_device,
        )
        emitted_frames = int(bootstrap_record["new_output_frame_range"][1])
        completed_chunks = 1
        print(
            json.dumps(
                {
                    "event": "inference_bootstrap_chunk_committed",
                    "bootstrap_path": str(bootstrap_first_chunk_path),
                    "chunk_index": 0,
                    "new_output_frame_range": bootstrap_record[
                        "new_output_frame_range"
                    ],
                    "next_chunk_absolute_start_frame": long_range.stride_frames,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if models_loaded:
        with StreamingAnchorReader(
            anchor_video, stride_frames=long_range.stride_frames
        ) as anchor_reader:
            for record in _iter_long_range_chunks(
                requested_frames, long_range, temporal_factor=sf.time
            ):
                if int(record["chunk_index"]) < completed_chunks:
                    continue
                start = int(record["absolute_start_frame"])
                working_count = int(record["working_frame_count"])
                chunk_anchor = anchor_reader.read_chunk(record)
                if chunk_anchor.shape[0] < working_count:
                    pad = chunk_anchor[-1:].expand(
                        working_count - chunk_anchor.shape[0], -1, -1, -1
                    )
                    chunk_anchor = torch.cat((chunk_anchor, pad), dim=0)
                    del pad

                chunk_metrics: dict[str, Any] = {}
                if cuda_metrics:
                    torch.cuda.reset_peak_memory_stats(torch.device(device_name))
                latent_observer = None
                if certification_latent_dir is not None:
                    latent_path = (
                        Path(certification_latent_dir).expanduser().resolve()
                        / f"chunk_{int(record['chunk_index']):06d}"
                        / "generated_latent.pt"
                    )
                    def latent_observer(
                        latent: torch.Tensor,
                        *,
                        _path: Path = latent_path,
                        _record: dict[str, Any] = record,
                        _start: int = start,
                    ) -> None:
                        _save_certification_latent(
                            _path,
                            latent,
                            record=_record,
                            seed=int(validation_config.seed) + _start,
                            checkpoint=Path(checkpoint_path or settings["model"]["adapter_path"]),
                        )
                enhanced = int(record["chunk_index"]) > 0
                carry_out = []
                if enhanced and recurrence["use_latent_carry"] and previous_carry is None:
                    raise RuntimeError("C1+ requires preceding causal latent carry")
                if enhanced and recurrence["lineart_global_prior"] and lineart_prior is None:
                    raise RuntimeError("C1+ requires C0 line-art reference")
                scope = lineart_prior.chunk_scope(int(record["chunk_index"]), prefix_latent_frames=3) if enhanced and lineart_prior else nullcontext()
                print(json.dumps({"event": "recurrence_chunk_start", "chunk_index": record["chunk_index"],
                                  "c0_legacy_path": not enhanced,
                                  "rebase": enhanced and recurrence["rebase_temporal_position"],
                                  "latent_carry": enhanced and recurrence["use_latent_carry"],
                                  "generated_prefix_strength": recurrence["generated_prefix_strength"] if enhanced else 1.0,
                                  "lineart": enhanced and lineart_prior is not None}, sort_keys=True), flush=True)
                with scope:
                    generated = sample_chunk(
                        transformer=transformer,
                        runner=runner,
                        vae_encoder=vae_encoder,
                        anchor_frames=chunk_anchor,
                        previous_prefix=previous_prefix,
                        boundary=boundary,
                        anchor_rect=anchor_rect,
                        absolute_start_frame=start,
                        runtime_metrics=chunk_metrics,
                        boundary_blend_width_pixels=boundary_blend_width_pixels,
                        allow_nonempty_text=True,
                        vae_decode_streaming=bool(memory_optimization["vae_decode_streaming"]),
                        progress_context={
                            "chunk_index": int(record["chunk_index"]),
                            "chunk_count": chunk_count,
                        },
                        final_latent_observer=latent_observer,
                        rebase_temporal_position=enhanced and recurrence["rebase_temporal_position"],
                        prefix_latent_override=previous_carry if enhanced and recurrence["use_latent_carry"] else None,
                        generated_prefix_strength=recurrence["generated_prefix_strength"] if enhanced else 1.0,
                        latent_carry_out=carry_out if recurrence["use_latent_carry"] else None,
                        latent_carry_frames=long_range.prefix_frames,

                    )
                if enhanced and lineart_prior is not None:
                    chunk_metrics["lineart_global_prior"] = lineart_prior.summary()
                    record["lineart_metadata_sha256"] = lineart_sha
                if recurrence["use_latent_carry"]:
                    if len(carry_out) != 1:
                        raise RuntimeError("Chunk did not return exactly one causal carry")
                    previous_carry = carry_out.pop()
                    record["causal_latent_carry"] = save_carry(store.root, int(record["chunk_index"]), previous_carry)
                record["recurrence"] = recurrence
                if cuda_metrics:
                    metric_device = torch.device(device_name)
                    chunk_allocated = torch.cuda.max_memory_allocated(metric_device)
                    chunk_reserved = torch.cuda.max_memory_reserved(metric_device)
                    chunk_metrics["peak_cuda_allocated_bytes"] = chunk_allocated
                    chunk_metrics["peak_cuda_reserved_bytes"] = chunk_reserved
                    peak_allocated = max(peak_allocated, chunk_allocated)
                    peak_reserved = max(peak_reserved, chunk_reserved)
                cpu_rng_state, cuda_rng_state, cuda_device = _capture_post_chunk_rng(
                    device_name
                )
                if int(record["chunk_index"]) == 0 and initial_prefix_artifact is not None:
                    if previous_prefix is None or not torch.equal(
                        generated[: long_range.prefix_frames], previous_prefix
                    ):
                        raise RuntimeError("Chunk 0 output does not exactly preserve initial FP32 prefix")
                    record["initial_prefix_bitwise_equal"] = True
                    record["initial_prefix_artifact"] = initial_prefix_artifact
                    record["initial_prefix_conditioned_frame_range"] = [
                        0, long_range.prefix_frames
                    ]
                # The old prefix is no longer reachable before cloning the next
                # one, avoiding a two-prefix peak while preserving exact FP32 RGB.
                del previous_prefix
                next_prefix = generated[-long_range.prefix_frames :].clone()
                append_offset = (
                    0 if int(record["chunk_index"]) == 0 else long_range.prefix_frames
                )
                append_count = int(record["output_appended_count"])
                finalized = generated[
                    append_offset : append_offset + append_count
                ]
                begin, end = map(int, record["new_output_frame_range"])
                if begin != emitted_frames or end - begin != finalized.shape[0]:
                    raise RuntimeError(
                        f"LongRange emission is not contiguous at chunk {record['chunk_index']}"
                    )
                record["runtime"] = chunk_metrics
                record["emitted_tensor_range"] = [
                    append_offset,
                    append_offset + append_count,
                ]
                if equivalence_observer is not None:
                    equivalence_observer(record, generated, next_prefix, finalized)
                    # Diagnostics must not perturb a later chunk even if an
                    # observer internally consumes PyTorch random numbers.
                    _restore_post_chunk_rng(
                        cpu_rng_state,
                        cuda_rng_state,
                        cuda_device,
                        expected_device=device_name,
                    )
                committed = store.commit_chunk(
                    record,
                    finalized,
                    next_prefix,
                    chunk_seed=int(validation_config.seed) + start,
                    cpu_rng_state=cpu_rng_state,
                    cuda_rng_state=cuda_rng_state,
                    cuda_device=cuda_device,
                )
                if chunk_committed_observer is not None:
                    chunk_committed_observer(
                        committed, store.root / committed["segment_file"]
                    )
                    _restore_post_chunk_rng(
                        cpu_rng_state, cuda_rng_state, cuda_device,
                        expected_device=device_name,
                    )
                if int(record["chunk_index"]) == 0 and recurrence["lineart_global_prior"]:
                    lineart_prior, lineart_sha = prepare_lineart(committed)
                    _restore_post_chunk_rng(cpu_rng_state, cuda_rng_state, cuda_device, expected_device=device_name)
                emitted_frames = end
                previous_prefix = next_prefix
                del finalized, generated, chunk_anchor
                print(
                    json.dumps(
                        {
                            "event": "inference_chunk_complete",
                            "chunk_index": record["chunk_index"],
                            "prefix_output_frame_range": record[
                                "prefix_output_frame_range"
                            ],
                            "new_output_frame_range": record["new_output_frame_range"],
                            "working_frame_count": working_count,
                            "repeat_last_padding_count": record[
                                "repeat_last_padding_count"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if emitted_frames != requested_frames:
        raise RuntimeError(
            f"LongRange assembly produced {emitted_frames} frames, expected {requested_frames}"
        )
    encoded_frames = encode_segments(
        store.iter_records(chunk_count),
        store.root,
        output_path,
        fps=expected_fps,
        width=width,
        height=height,
        expected_frames=requested_frames,
    )
    if encoded_frames != requested_frames:
        raise RuntimeError("Final encoder frame count changed during assembly")

    gap_count = duplicate_count = 0
    write_record_manifest(
        manifest_dir / "chunk_lineage.json",
        {
            "requested_frames": requested_frames,
            "clip_frames": long_range.clip_frames,
            "prefix_frames": long_range.prefix_frames,
            "stride_frames": long_range.stride_frames,
            "gap_count": gap_count,
            "duplicate_count": duplicate_count,
        },
        store.iter_records(chunk_count),
    )
    _write_json(
        manifest_dir / "inference_manifest.json",
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "input_path": str(anchor_video),
            "output_path": str(output_path),
            "input_rgb_shape": [
                requested_frames,
                3,
                input_metadata.height,
                input_metadata.width,
            ],
            "input_total_frames": input_total_frames,
            "input_fps": input_fps,
            "input_duration_seconds": input_total_frames / input_fps,
            "requested_frames": requested_frames,
            "output_rgb_shape": [requested_frames, 3, height, width],
            "output_finite": True,
            "seed": int(validation_config.seed),
            "inference_steps": int(validation_config.inference_steps),
            "boundary_blend_width_pixels": boundary_blend_width_pixels,
            "early_global_boost": early_global_boost_metadata,
            "positive_prompt": validation_config.samples[0].prompt,
            "negative_prompt": validation_config.negative_prompt,
            # Preserve legacy manifest keys for existing empty-prompt resume scripts.
            "fixed_positive_prompt": validation_config.samples[0].prompt,
            "fixed_negative_prompt": validation_config.negative_prompt,
            "architecture_version": int(settings["architecture"]["architecture_version"]),
            "memory_optimization": memory_optimization,
            "recurrence": recurrence,
            "lineart_metadata_sha256": lineart_sha,
            "adapter_load": adapter_report,
            "chunk_count": chunk_count,
            "prefix_gap_count": gap_count,
            "prefix_duplicate_count": duplicate_count,
            "segmented_deferred_generation": True,
            "segment_directory": str(store.root),
            "intermediates_retained": keep_intermediates,
            "completed_chunks_resumed": resumed_chunks,
            "bootstrap_first_chunk_artifact": (
                identity["request"].get("bootstrap_first_chunk_artifact")
            ),
            "manifest_directory": str(manifest_dir),
            "wall_time_seconds": time.monotonic() - started_at,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            **({"initial_prefix_artifact": initial_prefix_artifact,
                "initial_prefix_bitwise_equal": True}
               if initial_prefix_artifact is not None else {}),
        },
    )
    print(
        json.dumps(
            {
                "event": "inference_complete",
                "output_path": str(output_path),
                "chunk_count": chunk_count,
                "gap_count": gap_count,
                "duplicate_count": duplicate_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not keep_intermediates:
        store.cleanup()
    return output_path


def _load_settings(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    _install_official_sources(Path(settings["ltx_repo"]).expanduser().resolve())
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2.3 long-range video outpainting")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--anchor-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Final MP4 for LongRange modes")
    parser.add_argument(
        "--mode",
        choices=("normal_long_range", "prefix_seed_search", "long_range_from_prefix"),
        default="normal_long_range",
        help="Optional prefix search/continuation; default preserves normal LongRange",
    )
    parser.add_argument(
        "--prefix-output-dir", type=Path,
        help="Directory for prefix-search FP32 tensors, previews, and metadata",
    )
    parser.add_argument(
        "--seed-search-output-dir",
        type=Path,
        help="Run directory for step-10 predicted-clean seed-search PNGs",
    )
    parser.add_argument(
        "--preview-step",
        type=int,
        help="One-based model step to capture from the configured full schedule",
    )
    parser.add_argument(
        "--preview-frames",
        type=int,
        nargs="+",
        help="Output frame indices to retain from the predicted-clean preview",
    )
    parser.add_argument(
        "--prefix-seeds", type=int, nargs="+",
        help="Explicit candidate seeds for prefix_seed_search",
    )
    parser.add_argument(
        "--num-prefix-samples", type=int,
        help="Generate this many consecutive seeds starting at --seed/config seed",
    )
    parser.add_argument(
        "--initial-prefix", type=Path,
        help="Exact FP32 tensor saved by prefix_seed_search; continuation conditioning",
    )
    parser.add_argument(
        "--bootstrap-first-chunk",
        type=Path,
        help=(
            "Existing complete first-chunk MP4 to commit as chunk zero before "
            "continuing LongRange generation"
        ),
    )
    # Training stays empty-text only; inference may opt into native text conditioning.
    # Both omissions remain empty, never the native nonempty negative default.
    parser.add_argument(
        "--prompt", type=str, default="",
        help="Positive text prompt; omitted means the existing empty-text inference path",
    )
    parser.add_argument(
        "--negative-prompt", type=str, default="",
        help="Negative text prompt; omitted explicitly uses an empty string",
    )
    parser.add_argument("--num-frames", type=int)
    adapter = parser.add_mutually_exclusive_group()
    adapter.add_argument(
        "--checkpoint",
        type=Path,
        help="Training checkpoint directory containing adapter_model.safetensors, or that file",
    )
    adapter.add_argument(
        "--initialized-adapters",
        action="store_true",
        help="Explicit pre-training smoke mode using reproducibly initialized adapters",
    )
    parser.add_argument("--seed", type=int, help="Override the configured sampling/init seed")
    parser.add_argument(
        "--steps", type=int, help="Override the configured denoising step count"
    )
    parser.add_argument(
        "--early-global-boost",
        action="store_true",
        help="Boost learned Global contribution during the first 10 denoise steps",
    )
    parser.add_argument(
        "--segment-dir",
        type=Path,
        help="Directory for atomic lossless LongRange segments and resume state",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help=(
            "Directory for chunk_lineage.json and inference_manifest.json; "
            "defaults to the final MP4 directory"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the contiguous completed chunks in --segment-dir",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Retain lossless uint8 segments and FP32 prefix state after final encode",
    )
    parser.add_argument(
        "--video-only-gpu-weights", action=argparse.BooleanOptionalAction,
        default=None, help="Keep video-inactive Audio/AV weights on CPU after strict load",
    )
    parser.add_argument(
        "--vae-decode-streaming", action=argparse.BooleanOptionalAction,
        default=None, help="Transfer completed VAE temporal output to CPU without a full GPU cat",
    )
    parser.add_argument(
        "--ffn-token-chunking", action=argparse.BooleanOptionalAction,
        default=None, help="Experimental token-axis FFN chunks; BF16 output is not bitwise-equivalent",
    )
    parser.add_argument(
        "--ffn-chunk-tokens", type=int, default=None,
        help="Number of video tokens per FFN micro-chunk when enabled",
    )
    args = parser.parse_args()
    settings = _load_settings(args.config)
    overrides = {
        "video_only_gpu_weights": args.video_only_gpu_weights,
        "vae_decode_streaming": args.vae_decode_streaming,
        "ffn_token_chunking": args.ffn_token_chunking,
        "ffn_chunk_tokens": args.ffn_chunk_tokens,
    }
    if any(value is not None for value in overrides.values()):
        configured = settings.get("memory_optimization", {})
        if not isinstance(configured, dict):
            raise TypeError("memory_optimization must be a mapping")
        settings["memory_optimization"] = {
            **configured,
            **{key: value for key, value in overrides.items() if value is not None},
        }
    _memory_optimization_settings(settings)
    preview_options_present = any(
        value is not None
        for value in (
            args.seed_search_output_dir,
            args.preview_step,
            args.preview_frames,
        )
    )
    if preview_options_present:
        if (
            args.seed_search_output_dir is None
            or args.preview_step is None
            or args.preview_frames is None
        ):
            parser.error(
                "--seed-search-output-dir, --preview-step, and --preview-frames "
                "must be supplied together"
            )
        if args.mode != "normal_long_range":
            parser.error("Step preview seed search requires --mode normal_long_range")
        if (
            args.output is not None
            or args.segment_dir is not None
            or args.manifest_dir is not None
            or args.resume
        ):
            parser.error(
                "Step preview seed search writes PNGs only; MP4/segment/resume "
                "options are not accepted"
            )
        if (
            args.initial_prefix is not None
            or args.bootstrap_first_chunk is not None
            or args.prefix_output_dir is not None
        ):
            parser.error("Step preview seed search does not use prefix-mode options")
        if args.prefix_seeds is not None or args.num_prefix_samples is not None:
            parser.error("Step preview seed search accepts exactly one --seed")
        if args.seed is None:
            parser.error("Step preview seed search requires --seed")
        if args.num_frames is None:
            parser.error("Step preview seed search requires --num-frames 65")
        sample_seed_preview(
            settings,
            args.anchor_video,
            args.seed_search_output_dir,
            prompt=args.prompt,
            num_frames=args.num_frames,
            checkpoint_path=args.checkpoint,
            initialized_adapters=args.initialized_adapters,
            seed=args.seed,
            inference_steps=args.steps,
            preview_step=args.preview_step,
            preview_frames=tuple(args.preview_frames),
            early_global_boost=args.early_global_boost,
            negative_prompt=args.negative_prompt,
        )
        return
    if args.mode == "prefix_seed_search":
        if args.prefix_output_dir is None:
            parser.error("--prefix-output-dir is required for prefix_seed_search")
        if (
            args.output is not None
            or args.initial_prefix is not None
            or args.bootstrap_first_chunk is not None
            or args.manifest_dir is not None
        ):
            parser.error("prefix_seed_search uses --prefix-output-dir, not --output/--initial-prefix")
        if args.num_frames is not None or args.resume or args.segment_dir is not None:
            parser.error("prefix_seed_search always uses exactly the configured prefix frames")
        from ltx_outpaint.prefix_modes import sample_prefix_seed_search

        sample_prefix_seed_search(
            settings,
            args.anchor_video,
            args.prefix_output_dir,
            args.config,
            prompt=args.prompt,
            checkpoint_path=args.checkpoint,
            initialized_adapters=args.initialized_adapters,
            seed=args.seed,
            prefix_seeds=args.prefix_seeds,
            num_prefix_samples=args.num_prefix_samples,
            inference_steps=args.steps,
            early_global_boost=args.early_global_boost,
            negative_prompt=args.negative_prompt,
        )
        return
    if args.output is None:
        parser.error("--output is required for LongRange inference")
    if args.mode == "long_range_from_prefix" and args.initial_prefix is None:
        parser.error("--initial-prefix is required for long_range_from_prefix")
    if args.mode == "normal_long_range" and args.initial_prefix is not None:
        parser.error("--initial-prefix requires --mode long_range_from_prefix")
    if (
        args.mode != "normal_long_range"
        and args.bootstrap_first_chunk is not None
    ):
        parser.error("--bootstrap-first-chunk requires --mode normal_long_range")
    if args.prefix_seeds is not None or args.num_prefix_samples is not None or args.prefix_output_dir is not None:
        parser.error("Prefix search options require --mode prefix_seed_search")
    prefix_kwargs = (
        {"initial_prefix_path": args.initial_prefix}
        if args.mode == "long_range_from_prefix" else {}
    )
    optional_long_range_kwargs = {
        **prefix_kwargs,
        **(
            {"bootstrap_first_chunk_path": args.bootstrap_first_chunk}
            if args.bootstrap_first_chunk is not None
            else {}
        ),
        **(
            {"manifest_dir": args.manifest_dir}
            if args.manifest_dir is not None
            else {}
        ),
    }
    sample_long_range(
        settings,
        args.anchor_video,
        args.output,
        args.prompt,
        args.num_frames,
        args.checkpoint,
        args.initialized_adapters,
        args.seed,
        args.steps,
        args.early_global_boost,
        negative_prompt=args.negative_prompt,
        segment_dir=args.segment_dir,
        resume=args.resume,
        keep_intermediates=args.keep_intermediates,
        **optional_long_range_kwargs,
    )


if __name__ == "__main__":
    main()
