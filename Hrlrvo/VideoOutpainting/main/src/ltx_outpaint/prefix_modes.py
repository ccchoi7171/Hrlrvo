"""Optional first-prefix candidate generation using the production chunk sampler.

Only one VAE-aligned source prefix and one generated decoded RGB chunk are
resident at once; candidates share loaded inference weights but not output
tensor references. This module does not alter recurrent LongRange generation.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def resolve_prefix_seeds(
    *, base_seed: int, prefix_seeds: list[int] | None, num_prefix_samples: int | None
) -> list[int]:
    """Resolve explicit seeds or consecutive seeds without silent duplicates."""
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("Base seed must be a non-negative integer")
    if num_prefix_samples is not None and (
        type(num_prefix_samples) is not int or num_prefix_samples < 1
    ):
        raise ValueError("num_prefix_samples must be a positive integer")
    if prefix_seeds is not None:
        if not prefix_seeds:
            raise ValueError("prefix_seeds must not be empty")
        if any(type(value) is not int or value < 0 for value in prefix_seeds):
            raise ValueError("Every prefix seed must be a non-negative integer")
        if len(set(prefix_seeds)) != len(prefix_seeds):
            raise ValueError("Prefix seeds must be unique")
        if num_prefix_samples is not None and num_prefix_samples != len(prefix_seeds):
            raise ValueError("num_prefix_samples must equal the explicit prefix seed count")
        return list(prefix_seeds)
    count = 1 if num_prefix_samples is None else num_prefix_samples
    return list(range(base_seed, base_seed + count))


@torch.inference_mode()
def sample_prefix_seed_search(
    settings: dict[str, Any],
    anchor_video: Path,
    output_dir: Path,
    config_path: Path,
    prompt: str | None = "",
    checkpoint_path: Path | None = None,
    initialized_adapters: bool = False,
    seed: int | None = None,
    prefix_seeds: list[int] | None = None,
    num_prefix_samples: int | None = None,
    inference_steps: int | None = None,
    early_global_boost: bool = False,
    negative_prompt: str | None = "",
) -> list[dict[str, Any]]:
    """Generate independently seeded FP32 full-canvas prefix candidates.

    Each seed uses exactly source frames ``[0, prefix_frames)`` and the same
    ``sample_chunk`` path as normal inference, with no temporal repetition.
    Model weights and source RGB are loaded once even for hundreds of seeds.
    """
    from ltx_outpaint import inference
    from ltx_outpaint.prefix_io import write_prefix_candidate
    from ltx_outpaint.streaming_video import (
        StreamingAnchorReader,
        file_sha256,
        probe_video,
    )

    prompt = inference._normalize_inference_prompt(prompt, "prompt")
    negative_prompt = inference._normalize_inference_prompt(
        negative_prompt, "negative_prompt"
    )
    base_seed = int(settings["sampling"]["seed"]) if seed is None else seed
    candidates = resolve_prefix_seeds(
        base_seed=base_seed,
        prefix_seeds=prefix_seeds,
        num_prefix_samples=num_prefix_samples,
    )
    if inference_steps is not None and (
        type(inference_steps) is not int or inference_steps < 1
    ):
        raise ValueError("inference_steps must be positive")
    memory_optimization = inference._memory_optimization_settings(settings)
    blend_width = inference._inference_boundary_blend_width(settings)
    inference._early_global_boost_metadata(early_global_boost)
    inference._install_official_sources(Path(settings["ltx_repo"]).expanduser().resolve())
    from ltx_trainer.model_loader import read_video_scale_factors

    anchor_video = anchor_video.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_metadata = probe_video(anchor_video)
    expected_fps = float(settings["video"]["fps"])
    tolerance = float(settings["video"].get("fps_tolerance", 1e-3))
    if abs(source_metadata.fps - expected_fps) > tolerance:
        raise ValueError(
            f"Input FPS {source_metadata.fps:g} does not match configured FPS "
            f"{expected_fps:g}; resampling is disabled"
        )

    vae_path = Path(
        settings["model"].get("video_vae_path")
        or settings["model"]["base_model_path"]
    ).expanduser().resolve()
    scale_factors = read_video_scale_factors(vae_path)
    spec = inference.LongRangeSpec(**settings["long_range"])
    spec.validate(scale_factors.time)
    if source_metadata.frame_count < spec.prefix_frames:
        raise ValueError(
            f"Source has {source_metadata.frame_count} decodable frames, "
            f"but prefix search requires {spec.prefix_frames}"
        )
    width, height = map(int, settings["video"]["canvas"])
    boundary = inference._vae_boundary(
        height, width, scale_factors.height, scale_factors.width
    )
    anchor_rect = inference._center_anchor(
        boundary, source_metadata.height, source_metadata.width
    )
    expected_anchor = settings["video"].get("expected_anchor")
    if settings["video"].get("enforce_expected_anchor", False) and expected_anchor != [
        source_metadata.width, source_metadata.height
    ]:
        raise ValueError(
            f"Input anchor is {source_metadata.width}x{source_metadata.height}, "
            f"expected {expected_anchor}; resizing is disabled"
        )
    validation_config = inference._make_validation_config(
        settings, prompt, boundary, spec.prefix_frames, negative_prompt=negative_prompt
    )
    validation_config.seed = candidates[0]
    if inference_steps is not None:
        validation_config.inference_steps = inference_steps

    # Resolve checkpoint identity before model loading. Adapter bytes are
    # hashed once, never loaded/re-saved merely to construct metadata.
    if checkpoint_path is not None and initialized_adapters:
        raise ValueError("Choose either a checkpoint or initialized adapters, not both")
    selected_checkpoint = (
        checkpoint_path if checkpoint_path is not None else Path(settings["model"]["adapter_path"])
    )
    selected_checkpoint = selected_checkpoint.expanduser().resolve()
    adapter_file = (
        None if initialized_adapters else inference._resolve_adapter_path(selected_checkpoint)
    )
    source_hash = file_sha256(anchor_video)
    adapter_hash = None if adapter_file is None else file_sha256(adapter_file)

    # This reader owns only the first config-defined source prefix; no later
    # source frame is decoded into an RGB buffer.
    with StreamingAnchorReader(anchor_video, stride_frames=spec.stride_frames) as reader:
        source = reader.read_chunk(
            {"absolute_start_frame": 0, "real_input_frame_count": spec.prefix_frames}
        )
    if tuple(source.shape) != (
        spec.prefix_frames, 3, source_metadata.height, source_metadata.width
    ) or source.dtype != torch.float32:
        raise RuntimeError("Streaming input prefix shape or dtype changed")

    runtime_device = torch.device(settings["runtime"]["device"])
    cuda_metrics = runtime_device.type == "cuda" and torch.cuda.is_available()
    if cuda_metrics:
        # Candidate zero includes model placement, matching a fresh-process
        # memory peak. Later candidates report sampling with resident weights.
        torch.cuda.reset_peak_memory_stats(runtime_device)
    model_load_started = time.monotonic()
    transformer, runner, vae_encoder, _adapter_report = inference._load_models(
        settings,
        validation_config,
        boundary,
        checkpoint_path=checkpoint_path,
        initialized_adapters=initialized_adapters,
        early_global_boost=early_global_boost,
        memory_optimization=memory_optimization,
    )
    model_load_seconds = time.monotonic() - model_load_started
    results: list[dict[str, Any]] = []
    for candidate_index, candidate_seed in enumerate(candidates):
        if cuda_metrics and candidate_index:
            torch.cuda.reset_peak_memory_stats(runtime_device)
        # sample_chunk creates its own Gaussian generator from this configured
        # seed. Do not alter the production process-global RNG sequence.
        runner._config.seed = candidate_seed
        if cuda_metrics:
            torch.cuda.synchronize(runtime_device)
        sample_started = time.monotonic()
        chunk_metrics: dict[str, Any] = {}
        generated = inference.sample_chunk(
            transformer=transformer,
            runner=runner,
            vae_encoder=vae_encoder,
            anchor_frames=source,
            previous_prefix=None,
            boundary=boundary,
            anchor_rect=anchor_rect,
            absolute_start_frame=0,
            runtime_metrics=chunk_metrics,
            boundary_blend_width_pixels=blend_width,
            allow_nonempty_text=bool(prompt or negative_prompt),
            vae_decode_streaming=bool(memory_optimization["vae_decode_streaming"]),
        )
        if cuda_metrics:
            torch.cuda.synchronize(runtime_device)
        sample_seconds = time.monotonic() - sample_started
        vram_metrics: dict[str, Any] = {
            "sample_runtime_seconds": sample_seconds,
            "model_load_seconds": model_load_seconds if candidate_index == 0 else 0.0,
            "gpu_peak_scope": "model_load_and_sample" if candidate_index == 0 else "sample_with_model_loaded",
        }
        if cuda_metrics:
            gib = 1024**3
            vram_metrics.update(
                {
                    "torch_max_allocated_gib": torch.cuda.max_memory_allocated(runtime_device) / gib,
                    "torch_max_reserved_gib": torch.cuda.max_memory_reserved(runtime_device) / gib,
                }
            )
        if (
            generated.device.type != "cpu"
            or generated.dtype != torch.float32
            or tuple(generated.shape) != (spec.prefix_frames, 3, height, width)
        ):
            raise RuntimeError("Prefix sample must be decoded CPU FP32 full-canvas RGB")
        metadata = {
            "source_path": str(anchor_video),
            "source_sha256": source_hash,
            "checkpoint_path": str(selected_checkpoint),
            "adapter_sha256": adapter_hash,
            "config_path": str(config_path),
            "source_frame_range": [0, spec.prefix_frames],
            "source_frame_range_inclusive": [0, spec.prefix_frames - 1],
            "memory_optimization": dict(memory_optimization),
            "generation_timestamp": datetime.now(UTC).isoformat(),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "inference_steps": int(validation_config.inference_steps),
            "early_global_boost": early_global_boost,
            "boundary_blend_width_pixels": blend_width,
            "runtime_dtype": str(settings["runtime"]["dtype"]),
            "chunk_runtime": chunk_metrics,
            **vram_metrics,
        }
        prefix_path, preview_path, meta_path, artifact_metadata = write_prefix_candidate(
            generated,
            output_dir,
            seed=candidate_seed,
            fps=expected_fps,
            metadata=metadata,
        )
        record = {
            "seed": candidate_seed,
            "prefix_path": str(prefix_path),
            "preview_path": str(preview_path),
            "metadata_path": str(meta_path),
            "tensor_sha256": artifact_metadata["tensor_sha256"],
            **vram_metrics,
        }
        print(json.dumps({"event": "prefix_candidate_complete", **record}, sort_keys=True), flush=True)
        results.append(record)
        del generated
    return results
