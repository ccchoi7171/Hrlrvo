"""Official LTX-2.3 trainer integration for hierarchical video outpainting."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import math
import shutil
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import yaml
from accelerate import DistributedType
from accelerate.checkpointing import load_accelerator_state, save_accelerator_state
from accelerate.utils import gather_object, send_to_device, set_seed
from accelerate.utils.fsdp_utils import load_fsdp_optimizer, save_fsdp_optimizer
from ltx_trainer.config import (
    AccelerationConfig,
    CheckpointsConfig,
    DataConfig,
    FlowMatchingConfig,
    HubConfig,
    LoraConfig,
    LtxTrainerConfig,
    ModelConfig,
    OptimizationConfig,
    ValidationConfig,
    WandbConfig,
)
from ltx_trainer.model_loader import (
    embedding_weight_paths,
    load_embeddings_processor,
    load_text_encoder,
    read_video_scale_factors,
    resolve_video_vae_path,
)
from ltx_trainer.timestep_samplers import ShiftedLogitNormalTimestepSampler
from ltx_trainer.trainer import LtxvTrainer, TrainingStepOutput
from ltx_trainer.training_strategies.flexible import (
    FlexibleStrategy,
    FlexibleStrategyConfig,
)
from safetensors.torch import save_file
from torch import Tensor
from torch.utils.data import DataLoader

from .data import (
    DeterministicEpochSampler,
    OutpaintingCacheDataset,
    audit_ltx_cache,
    collate_outpainting,
)
from .geometry import CanvasGeometry, Rect
from .inference import _validate_empty_text_conditioning
from .model import (
    ArchitectureConfig,
    adapter_state_dict,
    attach_geometry,
    install_outpainting,
    load_adapter,
    parameter_counts,
)
from .outpainting_adapter import (
    checkpoint_provenance,
    reject_unsupported_conditioning,
    validate_provenance_config,
)
from .training_runtime import (
    config_fingerprint,
    prepare_run_artifacts,
    resolve_resume_directory,
    run_training,
    training_console,
)

logger = logging.getLogger(__name__)


class StableShiftedLogitNormalTimestepSampler(ShiftedLogitNormalTimestepSampler):
    """Pinned LTX shifted-logit sampler with stable float64 stretching."""

    def sample(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device | None = None,
    ) -> Tensor:
        mu = self._get_shift_for_sequence_length(seq_length)
        # Preserve native RNG dtype/order; promote only the unstable sigmoid and
        # percentile normalization at 4K sequence lengths.
        normal_samples = torch.randn((batch_size,), device=device) * self.std + mu
        logitnormal_samples = torch.sigmoid(normal_samples.to(torch.float64))
        percentile_999 = torch.sigmoid(
            torch.tensor(mu + self.normal_999_percentile, device=device, dtype=torch.float64)
        )
        percentile_005 = torch.sigmoid(
            torch.tensor(mu + self.normal_005_percentile, device=device, dtype=torch.float64)
        )
        denominator = percentile_999 - percentile_005
        if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
            raise FloatingPointError(
                f"Invalid shifted-logit-normal percentile range for sequence length {seq_length}: "
                f"lower={percentile_005.item()}, upper={percentile_999.item()}"
            )
        zero_terminal_raw = (logitnormal_samples - percentile_005) / denominator
        stretched_logit = torch.where(
            zero_terminal_raw >= self.eps,
            zero_terminal_raw,
            2 * self.eps - zero_terminal_raw,
        ).clamp_(0, 1).to(normal_samples.dtype)
        uniform = (1 - self.eps) * torch.rand((batch_size,), device=device) + self.eps
        prob = torch.rand((batch_size,), device=device)
        result = torch.where(prob > self.uniform_prob, stretched_logit, uniform)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError(f"Non-finite shifted-logit-normal sample at sequence length {seq_length}")
        return result


class OutpaintingStrategy(FlexibleStrategy):
    """Keep official flow targets/noise sampling, but source clean Anchor safely."""

    def __init__(self, config: FlexibleStrategyConfig, outpainting: dict[str, Any]) -> None:
        super().__init__(config)
        self.outpainting = outpainting

    def prepare_training_inputs(self, batch: dict[str, Any], timestep_sampler: Any):
        target = self._patchify_latent_data(batch["video_latents"], "video")
        anchor = self._patchify_latent_data(batch["anchor_latents"], "video")
        if target.latents.shape != anchor.latents.shape:
            raise ValueError("Target and isolated-anchor patchified tensors must have identical shapes")
        latents = target.latents
        batch_size, seq_len, _ = latents.shape
        noisy, targets, timesteps, loss_mask, sigmas = self._initialize_noisy_target(latents, timestep_sampler)

        provenance = batch.get("anchor_provenance")
        if not isinstance(provenance, list) or len(provenance) != batch_size:
            raise ValueError("Anchor provenance count must match the training batch size")
        configured_rect = tuple(map(int, self.outpainting["anchor_rect_pixels"]))
        rects = []
        for request in provenance:
            if (
                not isinstance(request, dict)
                or tuple(map(int, request.get("anchor_rect_canonical_yxyx", ()))) != configured_rect
            ):
                raise ValueError("Anchor provenance canonical rectangle differs from training config")
            rects.append(_latent_anchor_rect(request, target.height, target.width))

        spatial = torch.zeros(
            batch_size, target.height, target.width, dtype=torch.bool, device=latents.device,
        )
        for index, (y0, x0, y1, x1) in enumerate(rects):
            spatial[index, y0:y1, x0:x1] = True
        spatial = spatial.flatten(1).repeat(1, target.num_frames)

        prefix_frames = int(self.outpainting["prefix_frames"])
        temporal_scale = self.video_scale_factors.time
        if prefix_frames % temporal_scale != 1:
            raise ValueError("prefix_frames must satisfy frames % VAE temporal scale == 1")
        prefix_latents = 1 + (prefix_frames - 1) // temporal_scale
        prefix_probability = float(self.outpainting["prefix_probability"])
        use_prefix = torch.rand(batch_size, device=latents.device) < prefix_probability
        temporal = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=latents.device)
        temporal[:, : prefix_latents * target.height * target.width] = use_prefix[:, None]
        known = spatial | temporal

        # Prefix GT is causally observed and may be clean.  At all later times the
        # clean region comes exclusively from the independent lazy Anchor cache.
        prefix_values = temporal.unsqueeze(-1)
        condition_values = torch.where(prefix_values, latents, anchor.latents)
        noisy = torch.where(known.unsqueeze(-1), condition_values, noisy)
        timesteps = torch.where(known, torch.zeros_like(timesteps), timesteps)
        loss_mask = loss_mask & ~known

        positions = self._get_video_positions(
            target.num_frames, target.height, target.width, batch_size, target.fps, latents.device
        )
        conditions = batch["conditions"]
        from ltx_core.model.transformer.modality import Modality
        from ltx_trainer.training_strategies.base_strategy import ModelInputs

        video = Modality(
            enabled=True,
            latent=noisy,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=conditions["video_prompt_embeds"],
            context_mask=conditions["prompt_attention_mask"],
        )
        for name, value in (("video.sigma", video.sigma), ("video.timesteps", video.timesteps)):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Non-finite {name} immediately before transformer forward")
        geometry = CanvasGeometry(
            frames=target.num_frames,
            height=target.height,
            width=target.width,
            anchors=tuple(Rect(*rect) for rect in rects),
        )
        video = attach_geometry(video, geometry)
        return ModelInputs(
            video=video,
            audio=None,
            video_targets=targets,
            audio_targets=None,
            video_loss_mask=loss_mask,
            audio_loss_mask=None,
        )


def _latent_anchor_rect(
    provenance: dict[str, Any], height: int, width: int,
) -> tuple[int, int, int, int]:
    """Return the exact half-open rectangle used to place the cached Anchor."""
    rect = provenance.get("canonical_latent_rect_yxyx")
    anchor_hw = provenance.get("anchor_latent_hw")
    if not isinstance(rect, list) or len(rect) != 4:
        raise ValueError("Anchor provenance requires canonical_latent_rect_yxyx")
    if not isinstance(anchor_hw, list) or len(anchor_hw) != 2:
        raise ValueError("Anchor provenance requires anchor_latent_hw")
    y0, x0, declared_y1, declared_x1 = map(int, rect)
    anchor_height, anchor_width = map(int, anchor_hw)
    if anchor_height < 1 or anchor_width < 1:
        raise ValueError("Anchor latent shape must be positive")
    result = (y0, x0, y0 + anchor_height, x0 + anchor_width)
    if result != (y0, x0, declared_y1, declared_x1):
        raise ValueError("Anchor canonical rectangle extent differs from its cached latent shape")
    if y0 < 0 or x0 < 0 or result[2] > height or result[3] > width:
        raise ValueError(f"Anchor rectangle is outside the latent canvas: {result}")
    return result


class OutpaintingTrainer(LtxvTrainer):
    """LTX trainer subclass: native loop/optimizer/scheduler/Accelerate remain intact."""

    def __init__(
        self,
        trainer_config: LtxTrainerConfig,
        outpainting: dict[str, Any],
        *,
        cache_audit: Any | None = None,
        anchor_cache: Any | None = None,
    ) -> None:
        # Reject before the native constructor encodes validation text.
        _validate_empty_text_conditioning(trainer_config.validation)
        self._outpainting = outpainting
        reject_unsupported_conditioning(outpainting)
        ArchitectureConfig(**outpainting["architecture"])
        self._checkpoint_provenance = checkpoint_provenance(
            trainer_config.model.model_path, outpainting["provenance"]
        )
        self._cache_audit = cache_audit
        self._anchor_cache = anchor_cache
        self._constant_condition: dict[str, Tensor] | None = None
        self._accelerate_resume_dir: Path | None = None
        self._pending_adapter_state: dict[str, Tensor] | None = None
        self._resume_manifest: dict[str, Any] | None = None
        self._anchor_ordering: dict[str, Any] | None = None
        self._runtime_config_fingerprint = config_fingerprint(trainer_config, outpainting)
        # New branch/LoRA initialization occurs during the base constructor,
        # earlier than the official loop's set_seed call.
        set_seed(trainer_config.seed)
        super().__init__(trainer_config)
        strategy = OutpaintingStrategy(self._config.training_strategy, outpainting)
        strategy.video_scale_factors = self._training_strategy.video_scale_factors
        self._training_strategy = strategy
        self._runtime_state = {
            "run_id": Path(trainer_config.output_dir).name,
            "optimizer_step": self._resume_state[0],
            "batches_seen": 0,
            "samples_seen": 0,
            **(self._resume_manifest or {}).get("runtime_state", {}),
        }

    def _create_scheduler(self, optimizer):
        # Native LinearLR pops its kwargs; keep the resume/config fingerprint
        # immutable while retaining exactly the native scheduler and horizon.
        params = self._config.optimization.scheduler_params
        self._config.optimization.scheduler_params = copy.deepcopy(params)
        try:
            return super()._create_scheduler(optimizer)
        finally:
            self._config.optimization.scheduler_params = params

    def _load_models(self) -> None:
        super()._load_models()
        # Connectors remain in the differentiable hidden-state path, but their
        # pretrained parameters are fixed just like the text encoder and VAE.
        self._embeddings_processor.eval().requires_grad_(False)
        prompt = self._outpainting.get("constant_prompt")
        if self._outpainting.get("conditions_dir") is not None or prompt != "":
            raise ValueError("Hrlrvo v6 requires conditions_dir=null and constant_prompt='' (fixed empty prompt)")
        device = torch.device(f"cuda:{int(__import__('os').environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu")
        text_encoder = load_text_encoder(
            self._config.model.text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        ).eval().requires_grad_(False)
        processor = load_embeddings_processor(
            checkpoint_path=embedding_weight_paths(self._config.model.model_path, self._config.model.text_encoder_path),
            gemma_model_path=self._config.model.text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
        ).eval().requires_grad_(False)
        with torch.inference_mode():
            hidden_states, attention_mask = text_encoder.encode([str(prompt)])[0]
            video_features, audio_features = processor.feature_extractor(hidden_states, attention_mask, "left")
        if audio_features is None:
            raise RuntimeError("Pinned LTX-2.3 AV connector requires empty-prompt audio preconnector features")
        self._constant_condition = {
            "video_prompt_embeds": video_features[0].cpu().contiguous(),
            "audio_prompt_embeds": audio_features[0].cpu().contiguous(),
            "prompt_attention_mask": attention_mask[0].cpu().contiguous(),
        }
        del text_encoder, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _setup_lora(self) -> None:
        cfg = ArchitectureConfig(**self._outpainting["architecture"])
        self._architecture_config = cfg
        self._transformer = install_outpainting(self._transformer, cfg, self._checkpoint_provenance)
        self._parameter_counts = parameter_counts(self._transformer)
        self._adapter_parameter_names = {name for name, value in self._transformer.named_parameters() if value.requires_grad}
        if not self._adapter_parameter_names:
            raise RuntimeError("Outpainting installation exposed no trainable parameters")

    def _init_timestep_sampler(self) -> None:
        if self._config.flow_matching.timestep_sampling_mode == "shifted_logit_normal":
            self._timestep_sampler = StableShiftedLogitNormalTimestepSampler(
                **self._config.flow_matching.timestep_sampling_params
            )
            return
        super()._init_timestep_sampler()

    def _setup_accelerator(self) -> None:
        super()._setup_accelerator()

        def save_adapter_only(models: list[torch.nn.Module], weights: list[dict[str, Tensor]], output_dir: str) -> None:
            del output_dir
            if weights:
                unwrapped = self._accelerator.unwrap_model(models[0], keep_torch_compile=False)
                self._pending_adapter_state = adapter_state_dict(unwrapped, weights[0])
            # Accelerator still saves optimizer/scheduler/scaler and per-rank RNG,
            # but must not duplicate the 46 GB frozen pretrained transformer.
            weights.clear()

        def skip_model_reload(models: list[torch.nn.Module], input_dir: str) -> None:
            del input_dir
            models.clear()  # Adapter was loaded before Accelerator/FSDP wrapping.

        self._accelerator.register_save_state_pre_hook(save_adapter_only)
        self._accelerator.register_load_state_pre_hook(skip_model_reload)

    def _load_checkpoint(self) -> None:
        value = self._config.model.load_checkpoint
        if not value:
            self._resume_state = (0, None)
            return
        path = Path(value).expanduser().resolve()
        if path.is_file() and not self._config.checkpoints.no_resume:
            raise RuntimeError(
                "A standalone adapter has no optimizer/scheduler/RNG state; set checkpoints.no_resume=true "
                "for an intentional weights-only start"
            )
        if path.is_dir() and not (path / "adapter_model.safetensors").is_file():
            candidates = sorted(path.glob("checkpoint-*"))
            if not candidates:
                raise FileNotFoundError(f"No adapter checkpoint found under {path}")
            path = candidates[-1]
        weights_path = path / "adapter_model.safetensors" if path.is_dir() else path
        load_adapter(self._transformer, weights_path, self._config.model.model_path)
        manifest_path = path / "outpainting_manifest.json" if path.is_dir() else None
        if manifest_path is not None and manifest_path.is_file():
            step = int(json.loads(manifest_path.read_text(encoding="utf-8"))["global_step"])
        else:
            step = int(path.name.rsplit("-", 1)[-1]) if path.is_dir() else 0
        if path.is_dir() and not self._config.checkpoints.no_resume:
            required_patterns = ["optimizer*", f"random_states_{self._accelerator.process_index}.pkl"]
            if self._config.optimization.scheduler_type != "constant":
                required_patterns.append("scheduler*")
            if self._config.acceleration.mixed_precision_mode == "fp16":
                required_patterns.append("scaler.pt")
            missing = [pattern for pattern in required_patterns if not list(path.glob(pattern))]
            if manifest_path is None or not manifest_path.is_file():
                missing.append("outpainting_manifest.json")
            if missing:
                raise RuntimeError(f"Checkpoint cannot resume exactly; missing runtime state: {missing}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("provenance") != self._checkpoint_provenance:
                raise RuntimeError("Resume manifest base/source provenance differs from the validated runtime")
            if manifest.get("num_processes") != self._accelerator.num_processes:
                raise RuntimeError("Resume world size differs from the checkpoint world size")
            if manifest.get("distributed_type") != self._accelerator.distributed_type.value:
                raise RuntimeError("Resume distributed type differs from the checkpoint distributed type")
            expected_runtime = {
                "seed": self._config.seed,
                "per_device_batch_size": self._config.optimization.batch_size,
                "gradient_accumulation_steps": self._config.optimization.gradient_accumulation_steps,
                "temporal_sampling": self._outpainting.get("temporal_sampling", "first"),
            }
            mismatches = {
                key: (value, manifest.get(key))
                for key, value in expected_runtime.items()
                if manifest.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"Resume data-order settings differ from checkpoint: {mismatches}")
            if manifest.get("config_fingerprint") != self._runtime_config_fingerprint:
                raise RuntimeError("Resume config fingerprint differs or is missing; scheduler/data/run must match")
            runtime = manifest.get("runtime_state", {})
            if runtime.get("optimizer_step") != step:
                raise RuntimeError("Resume optimizer-step counter differs from checkpoint")
            expected_batches = step * self._config.optimization.gradient_accumulation_steps
            expected_samples = expected_batches * self._config.optimization.batch_size * self._accelerator.num_processes
            if runtime.get("batches_seen") != expected_batches or runtime.get("samples_seen") != expected_samples:
                raise RuntimeError("Resume sample/batch counters differ from completed optimizer steps")
            self._resume_manifest = manifest
            self._accelerate_resume_dir = path
        self._resume_state = (0 if self._config.checkpoints.no_resume else step, None)

    def _init_optimizer(self) -> None:
        super()._init_optimizer()
        if self._accelerate_resume_dir is not None and not self._config.checkpoints.no_resume:
            if self._accelerator.distributed_type == DistributedType.FSDP:
                load_fsdp_optimizer(
                    self._accelerator.state.fsdp_plugin,
                    self._accelerator,
                    self._optimizer,
                    self._transformer,
                    str(self._accelerate_resume_dir),
                    0,
                )
                override = load_accelerator_state(
                    str(self._accelerate_resume_dir),
                    [],
                    [],
                    self._accelerator._schedulers,
                    self._accelerator._dataloaders,
                    self._accelerator.state.process_index,
                    self._accelerator.scaler,
                )
                if "step" in override:
                    self._accelerator.step = override["step"]
            else:
                self._accelerator.load_state(str(self._accelerate_resume_dir))

    def _init_dataloader(self) -> None:
        if self._dataset is None:
            audit = self._cache_audit or audit_ltx_cache(
                self._outpainting["cache_root"],
                source_clip_frames=int(self._outpainting["clip_frames"]),
                scale_factors=self._training_strategy.video_scale_factors,
                requested_canvas=tuple(self._outpainting["canvas_pixels"]),
            )
            self._cache_audit = audit
            if self._anchor_cache is None:
                self._anchor_cache = _build_anchor_cache(audit, self._config, self._outpainting)
            self._dataset = OutpaintingCacheDataset(
                audit,
                self._anchor_cache,
                conditions_dir=self._outpainting.get("conditions_dir"),
                seed=self._config.seed,
                temporal_sampling=self._outpainting.get("temporal_sampling", "first"),
            )
            if self._resume_manifest is not None:
                expected_data = {
                    "dataset_length": len(self._dataset),
                    "order_fingerprint": audit.order_fingerprint,
                }
                mismatches = {
                    key: (value, self._resume_manifest.get(key))
                    for key, value in expected_data.items()
                    if self._resume_manifest.get(key) != value
                }
                if mismatches:
                    raise RuntimeError(f"Resume dataset differs from checkpoint: {mismatches}")
        current_ready, current_missing = self._anchor_cache.cache_inventory()
        current_ready = tuple(map(int, current_ready))
        current_missing = tuple(map(int, current_missing))
        _validate_anchor_partition(len(self._dataset), current_ready, current_missing)
        contract = _anchor_order_contract(self._dataset.audit, self._anchor_cache, self._outpainting)
        stored = _resume_anchor_snapshot(self._resume_manifest)
        ready_indices, missing_indices, self._anchor_ordering = _resolve_anchor_ordering(
            len(self._dataset), contract, current_ready, current_missing, stored
        )
        logger.info(
            "Anchor startup inventory: current READY=%d MISSING=%d; frozen READY=%d MISSING=%d",
            len(current_ready), len(current_missing), len(ready_indices), len(missing_indices),
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            fingerprints: list[str | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(fingerprints, self._anchor_ordering["fingerprint"])
            if len(set(fingerprints)) != 1:
                raise RuntimeError(f"Training ranks observed different Anchor startup inventories: {fingerprints}")
        workers = self._config.data.num_dataloader_workers
        local_batch = self._config.optimization.batch_size
        accumulation = self._config.optimization.gradient_accumulation_steps
        world = self._accelerator.num_processes
        global_accumulation_batch = local_batch * world * accumulation
        epoch_size = len(self._dataset) // global_accumulation_batch * global_accumulation_batch
        if epoch_size == 0:
            raise RuntimeError("Dataset is smaller than one global gradient-accumulation batch")
        completed_steps = self._resume_state[0]
        consumed_global_items = completed_steps * global_accumulation_batch
        start_epoch, start_offset = divmod(consumed_global_items, epoch_size)
        sampler = DeterministicEpochSampler(
            len(self._dataset),
            seed=self._config.seed,
            epoch_size=epoch_size,
            ready_indices=ready_indices,
            missing_indices=missing_indices,
            start_epoch=start_epoch,
            start_offset=start_offset,
        )
        worker_generator = torch.Generator().manual_seed(self._config.seed + 0x4D595DF4)
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=workers,
            pin_memory=workers > 0,
            persistent_workers=workers > 0,
            generator=worker_generator,
            collate_fn=collate_outpainting,
            multiprocessing_context="spawn" if workers > 0 else None,
        )
        self._dataloader = self._accelerator.prepare(dataloader, device_placement=[False])

    def _training_step(self, batch: dict[str, Any]) -> TrainingStepOutput:
        self._anchor_cache.complete_batch(batch)
        batch = send_to_device(batch, self._accelerator.device)
        if "conditions" in batch:
            raise RuntimeError("Hrlrvo v6 rejects dataset caption conditions; fixed empty prompt is mandatory")
        if self._constant_condition is None:
            raise RuntimeError("Fixed empty-prompt connector features were not initialized")
        required = {"video_prompt_embeds", "audio_prompt_embeds", "prompt_attention_mask"}
        missing = required - self._constant_condition.keys()
        if missing:
            raise RuntimeError(f"Incomplete fixed empty-prompt connector condition: missing {sorted(missing)}")
        batch_size = batch["video_latents"]["latents"].shape[0]
        device = batch["video_latents"]["latents"].device
        batch["conditions"] = {
            key: value.to(device).unsqueeze(0).expand(batch_size, *value.shape)
            for key, value in self._constant_condition.items()
        }
        output = super()._training_step(batch)
        # Checkpoint/resume counters, not diagnostics: native checkpoint saving
        # precedes the optional step callback, so count each successful microbatch here.
        self._runtime_state["batches_seen"] += 1
        self._runtime_state["samples_seen"] += batch_size * self._accelerator.num_processes
        return output

    def _run_validation(self, progress: Any) -> list[Path]:
        """Geometry-aware validation using the official runner's denoising implementation."""
        del progress
        anchor_paths = [Path(value) for value in self._outpainting.get("validation_anchor_videos", [])]
        samples = self._config.validation.samples
        if not samples:
            return []

        from ltx_trainer.model_loader import load_video_vae_encoder
        from ltx_trainer.validation_runner import ValidationRunner
        from ltx_trainer.video_utils import read_video, save_video

        from .inference import PixelBoundary, sample_chunk

        runner = self._validation_runner

        def geometry_modality(this: Any, state: Any, context: Tensor, sigma: Tensor) -> Any:
            modality = ValidationRunner._modality_from_latent_state(state, context, sigma)
            if getattr(this, "geometry", None) is None:
                raise RuntimeError("Validation geometry was not set")
            return attach_geometry(modality, this.geometry)

        runner._modality_from_latent_state = MethodType(geometry_modality, runner)
        runner.geometry = None
        vae_path = resolve_video_vae_path(self._config.model.model_path, self._config.model.video_vae_path)
        vae_encoder = load_video_vae_encoder(vae_path, device="cpu", dtype=torch.bfloat16).requires_grad_(False)
        requested_height, requested_width = map(int, self._outpainting["canvas_pixels"])
        encoded_width, encoded_height, frames = self._config.validation.video_dims
        boundary = PixelBoundary(
            requested_height=requested_height,
            requested_width=requested_width,
            encoded_height=encoded_height,
            encoded_width=encoded_width,
            top=(encoded_height - requested_height) // 2,
            left=(encoded_width - requested_width) // 2,
        )
        rank = self._accelerator.process_index
        world = self._accelerator.num_processes
        rank_indices = list(range(rank, len(samples), world))
        work: list[tuple[int, bool]] = [(index, True) for index in rank_indices]
        if self._accelerator.distributed_type == DistributedType.FSDP and world > 1:
            count = math.ceil(len(samples) / world)
            pad = rank_indices[-1] if rank_indices else 0
            work.extend((pad, False) for _ in range(count - len(work)))

        sampled: list[tuple[int, Path]] = []
        output = Path(self._config.output_dir) / "validation" / f"step_{self._global_step:06d}"
        cached_embeddings = runner._cached_embeddings
        was_training = self._transformer.training
        self._transformer.eval()
        try:
            for index, should_save in work:
                anchor_frames, fps = read_video(anchor_paths[index], max_frames=frames)
                if anchor_frames.shape[0] != frames or abs(fps - self._config.validation.frame_rate) > 1e-3:
                    raise ValueError(f"Validation anchor must contain {frames} frames at configured FPS: {anchor_paths[index]}")
                anchor_height, anchor_width = anchor_frames.shape[-2:]
                if anchor_height > requested_height or anchor_width > requested_width:
                    raise ValueError(f"Validation anchor exceeds canvas: {anchor_paths[index]}")
                y0 = boundary.top + (requested_height - anchor_height) // 2
                x0 = boundary.left + (requested_width - anchor_width) // 2
                rect = (y0, x0, y0 + anchor_height, x0 + anchor_width)
                # sample_chunk consumes slot zero; select the matching official
                # cached positive/negative prompt pair without re-encoding Gemma.
                runner._cached_embeddings = [cached_embeddings[index]]
                with self._accelerator.autocast():
                    video = sample_chunk(
                        transformer=self._transformer,
                        runner=runner,
                        vae_encoder=vae_encoder,
                        anchor_frames=anchor_frames,
                        previous_prefix=None,
                        boundary=boundary,
                        anchor_rect=rect,
                        absolute_start_frame=0,
                    )
                if should_save:
                    path = output / f"sample_{index:03d}.mp4"
                    save_video(video, path, fps=fps, video_format="FCHW")
                    sampled.append((index, path))
        finally:
            runner._cached_embeddings = cached_embeddings
            if was_training:
                self._transformer.train()
            del vae_encoder
        if world > 1:
            sampled = sorted(gather_object(sampled), key=lambda item: item[0])
        paths = [path for _, path in sampled]
        if self._accelerator.is_main_process and paths and self._wandb_run is not None and self._config.wandb.log_validation_videos:
            runner.log_to_wandb(self._wandb_run, paths, self._global_step)
        self._accelerator.wait_for_everyone()
        return paths

    def _save_checkpoint(self) -> Path | None:
        save_dir = Path(self._config.output_dir) / "checkpoints" / f"checkpoint-{self._global_step:06d}"
        self._accelerator.wait_for_everyone()
        if self._accelerator.distributed_type == DistributedType.FSDP:
            save_dir.mkdir(parents=True, exist_ok=True)
            # Accelerate.save_state writes the entire frozen FSDP model before its
            # save hook. Save only its FSDP optimizer plus standard scheduler,
            # scaler, dataloader and rank-local RNG state instead.
            save_fsdp_optimizer(
                self._accelerator.state.fsdp_plugin,
                self._accelerator,
                self._optimizer,
                self._transformer,
                str(save_dir),
                0,
            )
            save_accelerator_state(
                str(save_dir),
                [],
                [],
                self._accelerator._schedulers,
                self._accelerator._dataloaders,
                self._accelerator.state.process_index,
                self._accelerator.step,
                self._accelerator.scaler,
                save_on_each_node=self._accelerator.project_configuration.save_on_each_node,
                safe_serialization=True,
            )
            gathered = self._accelerator.get_state_dict(self._transformer)
            # Accelerate's full FSDP state dict is rank0_only; other ranks
            # participate in the gather but receive an empty dictionary.
            if self._accelerator.is_main_process:
                unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
                self._pending_adapter_state = adapter_state_dict(unwrapped, gathered)
        else:
            self._accelerator.save_state(str(save_dir), safe_serialization=True)
        if not self._accelerator.is_main_process:
            self._pending_adapter_state = None
            return None
        if self._pending_adapter_state is None:
            raise RuntimeError("Accelerator save hook did not collect adapter state")
        save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32
        adapter_path = save_dir / "adapter_model.safetensors"
        save_file(
            {
                key: value.detach().to(device="cpu", dtype=save_dtype).contiguous()
                for key, value in self._pending_adapter_state.items()
            },
            adapter_path,
            metadata={
                "base_model_path": str(Path(self._config.model.model_path).resolve()),
                "architecture": json.dumps(asdict(self._architecture_config)),
                "provenance": json.dumps(self._checkpoint_provenance, sort_keys=True),
            },
        )
        self._pending_adapter_state = None
        manifest = {
            "global_step": self._global_step,
            "architecture_version": self._architecture_config.architecture_version,
            "config_fingerprint": self._runtime_config_fingerprint,
            "runtime_state": {
                **self._runtime_state,
                "optimizer_step": self._global_step,
                "last_learning_rate": self._optimizer.param_groups[0]["lr"],
            },
            "base_model_path": str(self._config.model.model_path),
            "training_config": self._config.model_dump(mode="json"),
            "outpainting": self._outpainting,
            "provenance": self._checkpoint_provenance,
            "parameter_counts": self._parameter_counts,
            "num_processes": self._accelerator.num_processes,
            "distributed_type": self._accelerator.distributed_type.value,
            "seed": self._config.seed,
            "per_device_batch_size": self._config.optimization.batch_size,
            "gradient_accumulation_steps": self._config.optimization.gradient_accumulation_steps,
            "temporal_sampling": self._outpainting.get("temporal_sampling", "first"),
            "dataset_length": len(self._dataset),
            "order_fingerprint": self._dataset.audit.order_fingerprint,
            "anchor_ordering": self._anchor_ordering,
            "weights": "adapter-only; optimizer/scheduler/scaler/RNG are Accelerator state files",
        }
        (save_dir / "outpainting_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._checkpoint_paths.append(adapter_path)
        keep = self._config.checkpoints.keep_last_n
        if keep > 0 and len(self._checkpoint_paths) > keep:
            for old in self._checkpoint_paths[:-keep]:
                shutil.rmtree(old.parent)
            self._checkpoint_paths = self._checkpoint_paths[-keep:]
        return adapter_path


def load_training_config(path: str | Path) -> tuple[LtxTrainerConfig, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    reject_unsupported_conditioning(raw)
    outpainting = raw.pop("outpainting")
    optimization = dict(raw.get("optimization", {}))
    if "per_device_batch_size" in optimization:
        optimization["batch_size"] = optimization.pop("per_device_batch_size")
    raw["optimization"] = optimization
    data = dict(raw["data"])
    if "num_workers" in data:
        data["num_dataloader_workers"] = data.pop("num_workers")
    raw["data"] = data

    fields: dict[str, Any] = {
        "model": ModelConfig(**raw["model"]),
        "lora": LoraConfig(**raw["lora"]),
        "training_strategy": FlexibleStrategyConfig(**raw["training_strategy"]),
        "optimization": OptimizationConfig(**raw["optimization"]),
        "acceleration": AccelerationConfig(**raw.get("acceleration", {})),
        "data": DataConfig(**raw["data"]),
        "validation": ValidationConfig(**raw.get("validation", {})),
        "checkpoints": CheckpointsConfig(**raw.get("checkpoints", {})),
        "hub": HubConfig(**raw.get("hub", {})),
        "flow_matching": FlowMatchingConfig(**raw.get("flow_matching", {})),
        "wandb": WandbConfig(**raw.get("wandb", {})),
        "seed": int(raw.get("seed", 42)),
        "output_dir": str(Path(raw.get("output_dir", "outputs")).expanduser().resolve()),
    }
    # The official aggregate validator requires an on-disk conditions/ directory.
    # This integration intentionally supports a constant prompt encoded in memory,
    # so construct from individually validated official sub-configs.
    config = LtxTrainerConfig.model_construct(**fields)
    _validate_integration_config(config, outpainting)
    return config, outpainting


def _build_anchor_cache(audit: Any, config: LtxTrainerConfig, outpainting: dict[str, Any]) -> Any:
    from .anchor_cache import LazyAnchorCache

    vae_path = resolve_video_vae_path(config.model.model_path, config.model.video_vae_path)
    vae = outpainting["anchor_vae"]
    cache = LazyAnchorCache(
        audit,
        cache_root=outpainting["anchor_cache_root"],
        raw_root=outpainting["raw_source_root"],
        vae_path=vae_path,
        anchor_size=tuple(outpainting["anchor_size_pixels"]),
        device=str(vae["device"]),
        tile_size=int(vae["tile_size"]),
        tile_overlap=int(vae["tile_overlap"]),
        offload=bool(vae["offload"]),
    )
    if cache.geometry["anchor_rect_canonical_yxyx"] != list(outpainting["anchor_rect_pixels"]):
        raise ValueError("Configured Anchor condition rectangle differs from exact raw/GT pixel mapping")
    return cache


def _validate_integration_config(config: LtxTrainerConfig, outpainting: dict[str, Any]) -> None:
    """Enforce invariants omitted only from the official conditions-dir validator."""
    if config.model.training_mode != "lora" or config.lora is None:
        raise ValueError("Outpainting trains only the installed adapter/new modules on a frozen base")
    if config.acceleration.quantization is not None:
        raise ValueError("Quantized base training is not supported by the outpainting adapter checkpoint contract")
    video = config.training_strategy.video
    if video is None or not video.is_generated or config.training_strategy.audio is not None:
        raise ValueError("This entrypoint requires exactly one generated video modality")
    if video.conditions:
        raise ValueError("Spatial/prefix conditioning is constructed by OutpaintingStrategy, not duplicated here")
    architecture = outpainting["architecture"]
    ArchitectureConfig(**architecture)
    reject_unsupported_conditioning(outpainting)
    validate_provenance_config(outpainting["provenance"])
    if config.lora.rank != int(architecture["shared_outpainting_lora_rank"]):
        raise ValueError("Top-level LoRA rank must describe the shared Outpainting adapter")
    if float(config.lora.alpha) != float(architecture["shared_outpainting_lora_alpha"]):
        raise ValueError("Top-level LoRA alpha must describe the shared Outpainting adapter")
    if config.lora.dropout != float(architecture["lora_dropout"]) or config.lora.dropout != 0.0:
        raise ValueError("Official LoRA config and architecture must both use dropout=0")
    expected_targets = {
        "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
        "ff.net.0.proj", "ff.net.2",
    }
    if set(config.lora.target_modules) != expected_targets:
        raise ValueError("Top-level LoRA targets must describe v6 shared Q/K/V/O and FFN adapters")
    if not bool(architecture["shared_outpainting_lora_enabled"]):
        raise ValueError("Hrlrvo v6 requires the scratch shared Outpainting LoRA")
    if bool(architecture["native_lora_enabled"]):
        raise ValueError("Hrlrvo v6 has no Native-specific LoRA")
    if not bool(architecture["adaptive_lora_enabled"]) or not bool(architecture["global_lora_enabled"]):
        raise ValueError("Hrlrvo v6 requires independent Adaptive and Global LoRA adapters")
    if not bool(architecture["ffn_lora_enabled"]):
        raise ValueError("Hrlrvo v6 requires the shared FFN LoRA")
    if architecture["lora_type"] != "shared_outpainting_plus_branch_specialization":
        raise ValueError("Unsupported Hrlrvo v6 LoRA architecture identity")
    if architecture["lora_targets"] != "shared_qkvo_branch_qkv_ffn":
        raise ValueError("Unsupported Hrlrvo v6 LoRA target identity")
    if not 0.0 <= float(outpainting["prefix_probability"]) <= 1.0:
        raise ValueError("prefix_probability must be in [0, 1]")
    if outpainting.get("temporal_sampling", "first") != "first":
        raise ValueError("The FIRST-only Anchor cache contract requires temporal_sampling='first'")
    if int(outpainting["clip_frames"]) != 65:
        raise ValueError("The FIRST-only Anchor cache contract requires clip_frames=65")
    if outpainting.get("conditions_dir") is not None or outpainting.get("constant_prompt") != "":
        raise ValueError("Hrlrvo v6 main training is fixed prompt-free: conditions_dir=null, constant_prompt=''")
    _validate_empty_text_conditioning(config.validation)
    if not Path(outpainting["raw_source_root"]).expanduser().is_dir():
        raise ValueError(f"raw_source_root does not exist: {outpainting['raw_source_root']}")
    anchor_root = Path(outpainting["anchor_cache_root"]).expanduser()
    if anchor_root.exists() and not anchor_root.is_dir():
        raise ValueError(f"anchor_cache_root is not a directory: {anchor_root}")
    anchor_height, anchor_width = map(int, outpainting["anchor_size_pixels"])
    y0, x0, y1, x1 = map(int, outpainting["anchor_rect_pixels"])
    if (y1 - y0, x1 - x0) != (anchor_height, anchor_width):
        raise ValueError("anchor_size_pixels differs from anchor_rect_pixels")
    if config.validation.interval is not None:
        anchors = outpainting.get("validation_anchor_videos", [])
        if len(anchors) != len(config.validation.samples) or not anchors:
            raise ValueError("Enabled validation requires one isolated RGB validation_anchor_video per sample")
        if any(sample.conditions for sample in config.validation.samples):
            raise ValueError("Validation samples must not add native conditions; sample_chunk builds the safe union mask")
        if any(sample.video_dims is not None or sample.seed is not None for sample in config.validation.samples):
            raise ValueError("Per-sample validation dimension/seed overrides are not supported by this training runner")
        for path in anchors:
            if not Path(path).expanduser().is_file():
                raise ValueError(f"Validation anchor video does not exist: {path}")


def _validate_anchor_partition(
    dataset_length: int, ready_indices: tuple[int, ...], missing_indices: tuple[int, ...]
) -> None:
    combined = (*ready_indices, *missing_indices)
    if len(combined) != dataset_length or set(combined) != set(range(dataset_length)):
        raise RuntimeError("Anchor READY/MISSING inventory does not partition the GT dataset exactly once")


def _anchor_order_contract(audit: Any, anchor_cache: Any, outpainting: dict[str, Any]) -> dict[str, Any]:
    source_manifest = {
        record_id: {
            "relative_path": record.get("relative_path"),
            "source_fingerprint": record.get("source_fingerprint"),
            "fps": record.get("fps"),
            "original_num_frames": record.get("original_num_frames"),
        }
        for record_id, record in sorted(audit.records.items())
    }
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    first_path = audit.compatible_files[0]
    first_record_id = first_path.relative_to(audit.root / "latents").as_posix()
    representative_request = anchor_cache.make_request(first_record_id, 0)
    return {
        "policy": "ready_then_missing_startup_snapshot_v1",
        "dataset_order_fingerprint": audit.order_fingerprint,
        "gt_cache_contract_sha256": sha256(canonical(audit.contract).encode()).hexdigest(),
        "source_manifest_sha256": sha256(canonical(source_manifest).encode()).hexdigest(),
        "anchor_request_contract_sha256": sha256(
            canonical(
                {
                    "temporal_sampling": outpainting["temporal_sampling"],
                    "clip_frames": int(outpainting["clip_frames"]),
                    "anchor_size_pixels": list(map(int, outpainting["anchor_size_pixels"])),
                    "geometry": anchor_cache.geometry,
                    "representative_request": representative_request,
                    "vae_checkpoint_sha256": audit.checkpoint_sha256,
                    "temporal_scale": audit.temporal_scale,
                }
            ).encode()
        ).hexdigest(),
    }


def _make_anchor_ordering(
    contract: dict[str, Any], ready_indices: tuple[int, ...], missing_indices: tuple[int, ...]
) -> dict[str, Any]:
    ordering = {
        **contract,
        "ready_indices": list(ready_indices),
        "missing_indices": list(missing_indices),
    }
    payload = json.dumps(ordering, sort_keys=True, separators=(",", ":"), allow_nan=False)
    ordering["fingerprint"] = sha256(payload.encode()).hexdigest()
    return ordering


def _resolve_anchor_ordering(
    dataset_length: int,
    contract: dict[str, Any],
    current_ready: tuple[int, ...],
    current_missing: tuple[int, ...],
    stored: Any,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    _validate_anchor_partition(dataset_length, current_ready, current_missing)
    if stored is None:
        return current_ready, current_missing, _make_anchor_ordering(contract, current_ready, current_missing)
    if not isinstance(stored, dict):
        raise TypeError("Resume manifest lacks the frozen READY-before-MISSING ordering snapshot")
    if stored.get("policy") != "ready_then_missing_startup_snapshot_v1":
        raise RuntimeError(f"Unsupported resume Anchor ordering policy: {stored.get('policy')!r}")
    ready_indices = tuple(map(int, stored.get("ready_indices", ())))
    missing_indices = tuple(map(int, stored.get("missing_indices", ())))
    _validate_anchor_partition(dataset_length, ready_indices, missing_indices)
    if stored != _make_anchor_ordering(contract, ready_indices, missing_indices):
        raise RuntimeError("Resume Anchor ordering/config/source fingerprint differs from checkpoint")
    no_longer_valid = sorted(set(ready_indices) - set(current_ready))
    if no_longer_valid:
        raise RuntimeError(
            "Anchor entries that were READY at checkpoint startup are now missing or invalid: "
            f"{no_longer_valid[:16]}"
        )
    # Formerly missing entries may have been created by the prior run; retain
    # the checkpoint snapshot so item order does not silently change.
    return ready_indices, missing_indices, stored


def _resume_anchor_snapshot(resume_manifest: dict[str, Any] | None) -> Any:
    if resume_manifest is None:
        return None
    stored = resume_manifest.get("anchor_ordering")
    if stored is None:
        raise RuntimeError("Resume manifest lacks the frozen READY-before-MISSING ordering snapshot")
    return stored


def main() -> None:
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--disable-progress-bars", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="New training run directory (no overwrite)")
    parser.add_argument("--resume", type=Path, help="Full checkpoint directory; resumes its original run")
    parser.add_argument("--stop-after-step", type=int, help="Save and stop at this absolute optimizer step; scheduler horizon unchanged")
    parser.add_argument("--diagnostics", action="store_true", help="Opt in to costly health audits, Fusion observers and custom log files (default: off)")
    args = parser.parse_args()
    config, outpainting = load_training_config(args.config)
    resume = resolve_resume_directory(args.resume) if args.resume else None
    if resume is not None:
        manifest = json.loads((resume / "outpainting_manifest.json").read_text())
        original_output = Path(manifest["training_config"]["output_dir"]).resolve()
        if args.output_dir is not None and args.output_dir.resolve() != original_output:
            parser.error("--resume must retain the original output directory")
        config.output_dir = str(original_output)
        config.model.load_checkpoint = str(resume)
    elif args.output_dir is not None:
        config.output_dir = str(args.output_dir.resolve())
    if args.stop_after_step is not None and not 0 < args.stop_after_step <= config.optimization.steps:
        parser.error("--stop-after-step must be within the unchanged production scheduler horizon")
    prepare_run_artifacts(config, outpainting, resume)
    console = training_console(Path(config.output_dir) / "train_console.log") if args.diagnostics else contextlib.nullcontext()
    with console:
        scale_factors = read_video_scale_factors(
            resolve_video_vae_path(config.model.model_path, config.model.video_vae_path)
        )
        audit = audit_ltx_cache(
            outpainting["cache_root"],
            source_clip_frames=int(outpainting["clip_frames"]),
            scale_factors=scale_factors,
            requested_canvas=tuple(outpainting["canvas_pixels"]),
        )
        anchor_cache = _build_anchor_cache(audit, config, outpainting)
        trainer = OutpaintingTrainer(config, outpainting, cache_audit=audit, anchor_cache=anchor_cache)
        run_training(
            trainer, disable_progress_bars=args.disable_progress_bars, stop_after_step=args.stop_after_step,
            diagnostics_enabled=args.diagnostics,
        )


if __name__ == "__main__":
    main()
