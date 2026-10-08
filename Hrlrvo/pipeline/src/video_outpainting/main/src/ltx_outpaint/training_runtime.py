"""Runtime bookkeeping around the unchanged official LTX training loop."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)


def config_fingerprint(config, outpainting: dict[str, Any]) -> str:
    """Resume changes only its checkpoint path, not schedule/data/model semantics."""
    payload = copy.deepcopy(config.model_dump(mode="json"))
    payload["model"]["load_checkpoint"] = None
    return hashlib.sha256(json.dumps(
        {"training": payload, "outpainting": outpainting},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def resolve_resume_directory(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("--resume requires a full checkpoint directory, not an adapter-only file")
    if not (path / "adapter_model.safetensors").is_file():
        candidates = sorted(path.glob("checkpoint-*"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint directory under {path}")
        path = candidates[-1]
    if not (path / "outpainting_manifest.json").is_file():
        raise FileNotFoundError(f"Full resume manifest is missing: {path}")
    return path


class _Tee:
    """Fail-safe console duplication; never changes tensor computation."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file
        self.failed = False

    def write(self, text):
        result = self.stream.write(text)
        if not self.failed:
            try:
                self.file.write(text)
            except OSError:
                self.failed = True
                self.stream.write("\nWARNING: train_console.log write failed; continuing training.\n")
        return result

    def flush(self):
        self.stream.flush()
        if not self.failed:
            try:
                self.file.flush()
            except OSError:
                self.failed = True

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextlib.contextmanager
def training_console(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        file = path.open("a", encoding="utf-8", buffering=1)
    except OSError:
        logger.warning("Cannot open training console log %s; continuing", path)
        yield
        return
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, file)), contextlib.redirect_stderr(_Tee(sys.stderr, file)):
            yield
    finally:
        file.close()


def prepare_run_artifacts(config, outpainting, resume_dir: Path | None) -> dict[str, Any]:
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_manifest.json"
    fingerprint = config_fingerprint(config, outpainting)
    if path.exists():
        if resume_dir is None:
            raise FileExistsError(f"Run already exists; choose --output-dir or use --resume: {root}")
        manifest = json.loads(path.read_text())
        if manifest["config_fingerprint"] != fingerprint:
            raise ValueError("Resume run config fingerprint changed (including scheduler horizon/output directory)")
    else:
        if resume_dir is not None:
            raise FileNotFoundError(f"Resume must continue the original run_manifest.json: {root}")
        manifest = {
            "run_id": root.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": fingerprint,
            "architecture_version": outpainting["architecture"]["architecture_version"],
            "output_dir": str(root),
            "checkpoint_interval_unit": "optimizer_steps",
            "checkpoint_interval": config.checkpoints.interval,
            "optimizer": config.optimization.optimizer_type,
            "base_learning_rate": config.optimization.learning_rate,
            "scheduler_horizon_optimizer_steps": config.optimization.steps,
        }
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot = {**config.model_dump(mode="json"), "outpainting": outpainting}
    filename = "run_config.yaml" if resume_dir is None else f"resume_config_{resume_dir.name}.yaml"
    (root / filename).write_text(yaml.safe_dump(snapshot, sort_keys=False))
    return manifest


class _RequestedStop(Exception):
    def __init__(self, checkpoint: Path | None):
        self.checkpoint = checkpoint


def run_training(trainer, *, disable_progress_bars=False, stop_after_step=None, diagnostics_enabled=False):
    """Use the native production loop; expensive health observers are opt-in."""
    if diagnostics_enabled:
        return run_training_with_health(
            trainer, disable_progress_bars=disable_progress_bars, stop_after_step=stop_after_step,
        )

    accelerator = trainer._accelerator
    optimization = trainer._config.optimization
    trainer._runtime_state.update({
        "world_size": accelerator.num_processes,
        "effective_batch_size": optimization.batch_size * optimization.gradient_accumulation_steps * accelerator.num_processes,
    })
    if stop_after_step is None:
        # No hooks, wrappers, per-step callback, telemetry, or custom console output.
        return trainer.train(disable_progress_bars=disable_progress_bars)
    if not trainer._resume_state[0] < stop_after_step <= optimization.steps:
        raise ValueError("--stop-after-step must exceed resumed step and not exceed the configured scheduler horizon")

    def stop_at_requested_step(step, _total, _videos):
        if step >= stop_after_step:
            raise _RequestedStop(trainer._save_checkpoint())

    try:
        return trainer.train(disable_progress_bars=disable_progress_bars, step_callback=stop_at_requested_step)
    except _RequestedStop as stop:
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return stop.checkpoint, None


def _memory(device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    # A device-side assert can poison CUDA; never mask the original failure
    # with a secondary telemetry exception while writing its diagnostic.
    try:
        free, total = torch.cuda.mem_get_info(device)
    except RuntimeError:
        free, total = None, None
    result = {
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "physical_free_gib": None if free is None else free / 2**30,
        "physical_total_gib": None if total is None else total / 2**30,
    }
    return result


def _representative_indices(numel, device):
    # Integer arithmetic is essential: FP32 linspace rounds 2**26-1 up to
    # 2**26 for the real FFN projection, producing an out-of-range last index.
    return torch.arange(32, device=device, dtype=torch.int64) * (numel - 1) // 31


def run_training_with_health(trainer, *, disable_progress_bars=False, stop_after_step=None):
    """Keep native accumulate/backward/AdamW/scheduler/checkpoint operations intact.

    The native step callback provides a bounded stop AFTER a real optimizer
    update. Its private sentinel is caught here, without shrinking the scheduler
    horizon or changing architecture/config to make a smoke test easier.
    """
    from .health_logging import TrainingHealthLogger

    accelerator = trainer._accelerator
    model = accelerator.unwrap_model(trainer._transformer, keep_torch_compile=False)
    initial_step = trainer._resume_state[0]
    if stop_after_step is not None and not initial_step < stop_after_step <= trainer._config.optimization.steps:
        raise ValueError("--stop-after-step must exceed resumed step and not exceed the configured scheduler horizon")
    root = Path(trainer._config.output_dir)
    runtime = trainer._runtime_state
    accumulation = trainer._config.optimization.gradient_accumulation_steps
    world = accelerator.num_processes
    runtime.update({"world_size": world, "effective_batch_size": trainer._config.optimization.batch_size * accumulation * world})
    health = TrainingHealthLogger(model, root / "train_metrics.jsonl", runtime["run_id"])
    old_step, old_clip = trainer._training_step, accelerator.clip_grad_norm_
    old_init_optimizer = trainer._init_optimizer
    old_prepare = trainer._training_strategy.prepare_training_inputs
    handles, rows, losses, sigmas = [], [], [], []
    # Tiny, deterministic samples from frozen tensors; no full model snapshot.
    frozen_names = ("attn1.to_q.weight", "attn1.to_out.0.weight", "ff.net.0.proj.weight", "attn2.to_q.weight")
    frozen_samples = {}
    for suffix in frozen_names:
        name, parameter = next((n, p) for n, p in model.named_parameters() if n.endswith(suffix) and not p.requires_grad)
        indices = _representative_indices(parameter.numel(), parameter.device)
        frozen_samples[name] = (parameter, indices, parameter.detach().flatten()[indices].clone())
    latest, updates, snapshots = {}, {}, {}
    started = time.perf_counter()
    stage = "initialization"
    batch_started = started
    forward_seconds = 0.0
    forward_finished = started
    optimizer_started = started
    learning_rate_used = None

    def collect_inputs(batch, sampler):
        inputs = old_prepare(batch, sampler)
        # Scalars/metadata only; no alternate conditioning or loss path.
        video = inputs.video
        health.set_geometry(video.geometry, batch_size=batch["video_latents"]["latents"].shape[0])
        latest["training_geometry"] = {
            "latent_shape": list(batch["video_latents"]["latents"].shape),
            "anchor_canvas_shape": list(batch["anchor_latents"]["latents"].shape),
            "anchor_rectangles": [p["canonical_latent_rect_yxyx"] for p in batch["anchor_provenance"]],
            "anchor_latent_hw": [p["anchor_latent_hw"] for p in batch["anchor_provenance"]],
            "rgb_interval": [0, 65], "latent_interval": [0, 9],
            "empty_connector_keys": sorted(batch["conditions"]),
            "audio_generation": inputs.audio is not None,
        }
        latest["record_ids"] = [p["gt_record_id"] for p in batch["anchor_provenance"]]
        latest["_known_count"] = (~inputs.video_loss_mask).sum().detach()
        latest["_nonzero_known_timesteps"] = torch.count_nonzero(video.timesteps[~inputs.video_loss_mask]).detach()
        return inputs

    def training_step(batch):
        nonlocal stage, batch_started, forward_seconds, forward_finished
        if not losses:
            health.begin_optimizer_step()
            batch_started = time.perf_counter()
            forward_seconds = 0.0
        else:
            health.begin_forward()
        stage = "forward"
        begin = time.perf_counter()
        output = old_step(batch)
        health.end_forward()
        if not bool(torch.isfinite(output.loss).all()):
            raise FloatingPointError("Non-finite real training loss")
        forward_finished = time.perf_counter()
        forward_seconds += forward_finished - begin
        losses.append(output.loss.detach().mean())
        sigmas.append(output.sigma.detach().mean())
        stage = "backward"
        return output

    def clip(parameters, max_norm, *args, **kwargs):
        nonlocal stage
        stage = "gradient_audit_and_clip"
        latest.update(health.capture_pre_clip_gradients())
        if latest.get("nan_gradient_count", 0) or latest.get("inf_gradient_count", 0):
            raise FloatingPointError("Non-finite training gradients before clipping/optimizer update")
        if latest.get("frozen_parameter_grad_violation_count", 0):
            raise RuntimeError("Frozen model parameter received a training gradient")
        result = old_clip(parameters, max_norm, *args, **kwargs)
        post = health.capture_post_clip_global_norm()
        latest["grad_norm_after_clip"] = post
        latest["grad_norm_before_clip"] = latest.get("grad_norm_global")
        pre = latest.get("grad_norm_before_clip")
        latest["gradient_clipped"] = bool(pre is not None and pre > max_norm)
        return result

    def before_optimizer(optimizer, args, kwargs):
        nonlocal stage, optimizer_started, snapshots, learning_rate_used
        stage = "optimizer_step"
        optimizer_started = time.perf_counter()
        learning_rate_used = [group["lr"] for group in optimizer.param_groups]
        if "grad_norm_global" not in latest:
            latest.update(health.capture_pre_clip_gradients())
        snapshots = health.snapshot_representatives()

    def after_optimizer(optimizer, args, kwargs):
        nonlocal updates
        updates = health.representative_updates(snapshots)
        unchanged = torch.stack([
            torch.eq(parameter.detach().flatten()[indices], before).all()
            for parameter, indices, before in frozen_samples.values()
        ]).cpu().tolist()
        latest["frozen_representative_unchanged"] = dict(zip(frozen_samples, unchanged, strict=True))
        processor = getattr(trainer, "_embeddings_processor", None)
        latest["frozen_connector_violation_count"] = (
            sum(p.requires_grad or p.grad is not None for p in processor.parameters()) if processor is not None else 0
        )
        latest["optimizer_seconds"] = time.perf_counter() - optimizer_started
        latest["optimizer_state_parameter_count"] = len(optimizer.state)

    def init_optimizer():
        old_init_optimizer()
        optimizer = trainer._optimizer.optimizer
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise RuntimeError("Optimizer parameters do not equal unique trainables")
        if any(not p.requires_grad for group in optimizer.param_groups for p in group["params"]):
            raise RuntimeError("Frozen parameter entered optimizer")
        if any(group["initial_lr"] != trainer._config.optimization.learning_rate for group in optimizer.param_groups):
            raise RuntimeError("Unexpected trainable-group base LR multiplier")
        runtime["optimizer_groups"] = [{"parameters": sum(p.numel() for p in g["params"]),
                                         "initial_lr": g["initial_lr"], "restored_lr": g["lr"]}
                                        for g in optimizer.param_groups]
        runtime["restored_optimizer_state_parameters"] = len(optimizer.state)
        runtime["restored_scheduler_state"] = trainer._lr_scheduler.state_dict() if trainer._lr_scheduler else None
        runtime["restored_accelerator_step"] = accelerator.step
        handles.append(optimizer.register_step_pre_hook(before_optimizer))
        handles.append(optimizer.register_step_post_hook(after_optimizer))

    def callback(step, _total, _videos):
        nonlocal stage
        stage = "metrics"
        scalars = torch.stack([torch.stack(losses).mean(), torch.stack(sigmas).mean(),
                               latest.pop("_known_count").float(), latest.pop("_nonzero_known_timesteps").float()]).cpu().tolist()
        row = {**latest,
               "optimizer_step": step, "batches_seen": runtime["batches_seen"], "samples_seen": runtime["samples_seen"],
               "loss": scalars[0], "sigma": scalars[1], "known_token_count": int(scalars[2]),
               "nonzero_known_timestep_count": int(scalars[3]),
               "learning_rate": learning_rate_used[0], "learning_rate_by_group": learning_rate_used,
               "next_learning_rate": trainer._optimizer.param_groups[0]["lr"],
               "forward_seconds": forward_seconds, "backward_and_clip_seconds": optimizer_started - forward_finished,
               "total_step_seconds": time.perf_counter() - batch_started,
               "parameter_updates": updates, **health.fusion_metrics(), **_memory(accelerator.device)}
        if accelerator.is_main_process:
            health.log_step(row)
            print(f"OPTIMIZER_STEP {step}: loss={row['loss']:.8g} lr={row['learning_rate']:.8g} "
                  f"next_lr={row['next_learning_rate']:.8g} samples={runtime['samples_seen']} "
                  f"peak_allocated={row.get('peak_allocated_gib', 0):.3f} GiB", flush=True)
        rows.append(row)
        runtime["optimizer_step"] = step
        runtime["last_learning_rate"] = row["next_learning_rate"]
        latest.clear(); losses.clear(); sigmas.clear()
        if stop_after_step is not None and step >= stop_after_step:
            stage = "checkpoint_save"
            path = trainer._save_checkpoint()
            raise _RequestedStop(path)

    trainer._training_step = training_step
    trainer._init_optimizer = init_optimizer
    trainer._training_strategy.prepare_training_inputs = collect_inputs
    accelerator.clip_grad_norm_ = clip
    checkpoint = None
    error = None
    try:
        try:
            checkpoint, _ = trainer.train(disable_progress_bars=disable_progress_bars, step_callback=callback)
        except _RequestedStop as stop:
            checkpoint = stop.checkpoint
            accelerator.wait_for_everyone()
            accelerator.end_training()
        stage = "complete"
    except Exception as exc:
        error = {"stage": stage, "type": type(exc).__name__, "message": str(exc), **_memory(accelerator.device)}
        raise
    finally:
        trainer._training_step = old_step
        trainer._init_optimizer = old_init_optimizer
        trainer._training_strategy.prepare_training_inputs = old_prepare
        accelerator.clip_grad_norm_ = old_clip
        for handle in handles:
            handle.remove()
        health.close()
        result = {"status": "FAIL" if error else "COMPLETE", "initial_step": initial_step,
                  "final_step": trainer._global_step, "checkpoint": str(checkpoint) if checkpoint else None,
                  "elapsed_seconds": time.perf_counter() - started, "steps": rows, "error": error,
                  "runtime_state": runtime, "parameter_counts": trainer._parameter_counts,
                  "failure_health_metrics": {key: value for key, value in latest.items() if not key.startswith("_")} if error else None,
                  **_memory(accelerator.device)}
        if accelerator.is_main_process:
            (root / f"runtime_result_start_{initial_step:06d}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
