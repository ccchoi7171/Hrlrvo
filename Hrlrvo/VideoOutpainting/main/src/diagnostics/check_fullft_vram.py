"""Isolated real-data H200 diagnostic; never changes production configuration.

Run --mode audit before GPU work, --mode prepare-text once, then fullft and
baseline in separate processes sharing --output-dir. At most two optimizer
steps per process. The Native adapter is bypassed only on the diagnostic
instance. No resolution reduction, offload, quantization or checkpoint save.

LTX packs SA/FFN and CA modulation rows in the same Parameter. Diagnostic
gradient masks plus a no-decay parameter group freeze the CA-only rows exactly;
their dense gradient/state allocation is INCLUDED in the memory accounting.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import send_to_device, set_seed
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask
from ltx_trainer.model_loader import (
    embedding_weight_paths,
    load_embeddings_processor,
    load_text_encoder,
    load_transformer,
    read_video_scale_factors,
)
from ltx_trainer.timestep_samplers import SAMPLERS
from ltx_trainer.trainer import LtxvTrainer

from ltx_outpaint.data import (
    OutpaintingCacheDataset,
    audit_ltx_cache,
    collate_outpainting,
)
from ltx_outpaint.model import ArchitectureConfig, install_outpainting
from ltx_outpaint.outpainting_adapter import checkpoint_provenance
from ltx_outpaint.train import (
    OutpaintingStrategy,
    _build_anchor_cache,
    load_training_config,
)

GIB = 2**30
RECORD = "0002e965-b236-4ec5-846d-4d1b919cf3f0.pt"
TRAIN = {"shared_sa_qkv", "shared_sa_o", "shared_ffn", "shared_norm_modulation",
         "adaptive_lora", "global_lora", "compressors", "decompressors", "fusion"}
NEW = {"native_lora", "adaptive_lora", "global_lora", "compressors", "decompressors", "fusion"}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def category(name):
    if name.startswith("adaln_single."):
        return "shared_norm_modulation"
    if name.startswith("prompt_adaln_single."):
        return "cross_attention"
    if not name.startswith("transformer_blocks."):
        return "other_frozen"
    tail = name.split(".", 2)[2]
    if tail.startswith("attn2.") or tail == "prompt_scale_shift_table":
        return "cross_attention"
    if tail == "scale_shift_table":
        return "shared_norm_modulation"
    if tail.startswith("ff."):
        return "shared_ffn"
    if not tail.startswith("attn1."):
        return "other_frozen"
    component = tail.split(".")[1]
    if component in ("to_q", "to_k", "to_v"):
        return "shared_sa_qkv"
    if component == "to_out":
        return "shared_sa_o"
    if component in ("q_norm", "k_norm", "to_gate_logits"):
        return "shared_norm_modulation"
    if component in ("local_fusion", "global_fusion"):
        return "fusion"
    if component in NEW:
        return component
    raise ValueError(f"Unclassified video self-attention parameter: {name}")


def mixed_ca_start(name, parameter, dim):
    if name in ("adaln_single.linear.weight", "adaln_single.linear.bias"):
        assert parameter.shape[0] == 9 * dim
        return 6 * dim
    if name.startswith("transformer_blocks.") and name.split(".", 2)[2] == "scale_shift_table":
        assert parameter.shape == (9, dim)
        return 6
    return None


def configure_trainability(model, mode, snapshots=False):
    guards = []
    for name, parameter in model.named_parameters():
        enabled = category(name) in (TRAIN if mode == "fullft" else NEW)
        parameter.requires_grad_(enabled)
        start = mixed_ca_start(name, parameter, model.inner_dim) if mode == "fullft" else None
        if start is not None and snapshots:
            def mask_gradient(gradient, start=start):
                result = gradient.clone()
                result[start:] = 0
                return result
            parameter.register_hook(mask_gradient)
            guards.append((name, parameter, start, parameter[start:].detach().clone()))
    if mode == "fullft":
        for block in model.transformer_blocks:
            # Retain the registered, frozen Native parameters for honest model
            # memory accounting; native_local reads this runtime attribute as
            # None, so attend(..., lora=None) does not evaluate any Native LoRA.
            assert "native_lora" in block.attn1._modules
            object.__setattr__(block.attn1, "native_lora", None)
    return guards


def audit_parameters(model, mode):
    named = list(model.named_parameters(remove_duplicate=False))
    assert len(named) == len({id(p) for _, p in named}), "Duplicate Parameter registration"
    if not named[0][1].is_meta:
        storages = [p.untyped_storage().data_ptr() for _, p in named]
        assert len(storages) == len(set(storages)), "Aliased parameter storage"
    rows, counts = [], defaultdict(lambda: Counter())
    for name, p in named:
        cat = category(name)
        expected = cat in (TRAIN if mode == "fullft" else NEW)
        assert p.requires_grad == expected, name
        ca_start = mixed_ca_start(name, p, model.inner_dim) if mode == "fullft" else None
        inactive = 0 if ca_start is None else p[ca_start:].numel()
        row = {"name": name, "category": cat, "shape": list(p.shape), "numel": p.numel(),
               "dtype": str(p.dtype), "requires_grad": p.requires_grad,
               "frozen_ca_rows_inside_shared_parameter": inactive}
        rows.append(row)
        counts[cat]["total"] += p.numel()
        counts[cat]["trainable_allocated"] += p.numel() if expected else 0
        counts[cat]["trainable_active_elements"] += p.numel() - inactive if expected else 0
        counts[cat]["parameter_bytes"] += p.numel() * p.element_size()
        counts[cat]["gradient_bytes"] += p.numel() * p.element_size() if expected else 0
    trainable = sum(p.numel() for _, p in named if p.requires_grad)
    frozen = sum(p.numel() for _, p in named if not p.requires_grad)
    parameter_bytes = sum(p.numel() * p.element_size() for _, p in named)
    gradient_bytes = sum(p.numel() * p.element_size() for _, p in named if p.requires_grad)
    inactive_count = sum(r["frozen_ca_rows_inside_shared_parameter"] for r in rows)
    assert counts["cross_attention"]["trainable_allocated"] == 0
    if mode == "fullft":
        assert counts["native_lora"]["trainable_allocated"] == 0
        assert all(b.attn1.native_lora is None for b in model.transformer_blocks)
    for branch in ("adaptive_lora", "global_lora"):
        assert counts[branch]["trainable_allocated"] == 75_497_472
    return {
        "verdict": "TRAINABLE PARAMETER MAP: PASS", "mode": mode,
        "blocks": len(model.transformer_blocks), "total": trainable + frozen,
        "trainable_parameter_numel": trainable, "frozen_parameter_numel": frozen,
        "trainable_active_elements": trainable - inactive_count,
        "frozen_ca_elements_inside_shared_parameters": inactive_count,
        "trainable_percent": 100 * trainable / (trainable + frozen), "categories": dict(counts),
        "persistent_lower_bound_bytes": {
            "transformer_parameters": parameter_bytes, "gradients": gradient_bytes,
            "adam_first_moment": gradient_bytes, "adam_second_moment": gradient_bytes,
            "master_weights": 0, "transformer_total": parameter_bytes + 3 * gradient_bytes,
        },
        "dtype_policy": "Existing BF16 base + FP32 Hrlrvo trainables; requires_grad override only. "
                       "AdamW moments use zeros_like(parameter), no separate master weights. "
                       "Official training_mode=full would instead upcast the base to FP32; not silently applied.",
        "mixed_ca_policy": "CA-only rows of shared modulation Parameters have zero gradients; mixed Parameters "
                           "use weight_decay=0 to prevent frozen-row drift, with post-step equality assertions. "
                           "Dense gradient/moment storage is counted in full. Shared timestep embedding is trained "
                           "because it is not CA-only. "
                           "Input/output projection, output-head modulation and unused AV/audio paths remain frozen.",
        "parameters": rows,
    }


def meta_audit(config, outpainting, directory):
    metadata = SafetensorsModelStateDictLoader().metadata(str(config.model.model_path))
    with torch.device("meta"):
        model = LTXModelConfigurator.from_metadata(metadata).to(dtype=torch.bfloat16)
        install_outpainting(model, ArchitectureConfig(**outpainting["architecture"]), outpainting["provenance"])
    results = {}
    for mode in ("baseline", "fullft"):
        configure_trainability(model, mode)
        result = audit_parameters(model, mode)
        write_json(directory / f"{mode}_parameter_map.json", result)
        results[mode] = {k: v for k, v in result.items() if k != "parameters"}
        print(mode, result["verdict"], json.dumps(results[mode], default=str), flush=True)
    write_json(directory / "parameter_audit.json", results)


def hardware():
    def query(*args):
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=False).stdout.strip()
    return {"nvidia_smi": query(), "compute_processes": query("--query-compute-apps=pid,process_name,used_gpu_memory",
                                                            "--format=csv,noheader"),
            "mig": query("--query-gpu=mig.mode.current", "--format=csv,noheader"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "pid": os.getpid()}


def check_reservation(config_path):
    hold = config_path.resolve().parent.parent / "outputs/cache_anchor_2k/gpu_test_priority.hold"
    if not hold.is_file():
        raise RuntimeError("GPU-test reservation hold is missing; do not race the Anchor cache job")
    other = "\n".join(line for line in hardware()["compute_processes"].splitlines()
                      if line.split(",", 1)[0].strip() != str(os.getpid()))
    if other:
        raise RuntimeError(f"GPU already has compute users; no process will be killed: {other}")


def prepare_text(config, outpainting, directory):
    assert outpainting["constant_prompt"] == ""
    path = directory / "empty_prompt_features.pt"
    if path.exists():
        raise FileExistsError(path)
    set_seed(config.seed)
    encoder = load_text_encoder(config.model.text_encoder_path, device="cuda", dtype=torch.bfloat16,
                                load_in_8bit=config.acceleration.load_text_encoder_in_8bit).eval().requires_grad_(False)
    processor = load_embeddings_processor(
        embedding_weight_paths(config.model.model_path, config.model.text_encoder_path),
        config.model.text_encoder_path, device="cuda", dtype=torch.bfloat16,
    ).eval().requires_grad_(False)
    with torch.no_grad():
        hidden, mask = encoder.encode([""])[0]
        video, audio = processor.feature_extractor(hidden, mask, "left")
    payload = {"prompt": "", "base": str(config.model.model_path),
               "text_encoder": str(config.model.text_encoder_path),
               "conditions": {"video_prompt_embeds": video.detach().cpu(),
                              "audio_prompt_embeds": audio.detach().cpu(),
                              "prompt_attention_mask": mask.detach().cpu()}}
    torch.save(payload, path)
    print("REAL EMPTY PROMPT", {k: (list(v.shape), str(v.dtype)) for k, v in payload["conditions"].items()}, flush=True)
    del encoder, processor, hidden, video, audio, mask
    gc.collect()
    torch.cuda.empty_cache()


def load_batch(config, outpainting, record_id):
    factors = read_video_scale_factors(config.model.model_path)
    audit = audit_ltx_cache(outpainting["cache_root"], source_clip_frames=65, scale_factors=factors,
                            requested_canvas=tuple(outpainting["canvas_pixels"]))
    cache = _build_anchor_cache(audit, config, outpainting)
    dataset = OutpaintingCacheDataset(audit, cache, seed=config.seed, temporal_sampling="first")
    index = next(i for i, p in enumerate(dataset.files) if p.relative_to(audit.root / "latents").as_posix() == record_id)
    sample = dataset[(index, 0)]
    assert sample["_anchor_entries"] is not None, "Diagnostic refuses cache MISS; VAE encode is not allowed"
    batch = collate_outpainting([sample])
    cache.complete_batch(batch)
    assert cache.created_count == 0 and cache._encoder is None
    assert batch["video_latents"]["latents"].shape == (1, 128, 9, 68, 120)
    assert batch["anchor_latents"]["latents"].shape == (1, 128, 9, 68, 120)
    provenance = batch["anchor_provenance"][0]
    assert provenance["rgb_start_frame"] == provenance["latent_start_index"] == 0
    assert provenance["rgb_frame_count"] == 65 and provenance["latent_frame_count"] == 9
    return batch, factors, provenance


class Measurements:
    def __init__(self, path, initial):
        self.path = path
        self.result = {**initial, "stages": [], "status": "RUNNING", "last_block": None,
                       "lifecycle_peak_allocated_gib": 0, "lifecycle_peak_reserved_gib": 0}
        self.stage = "initial"
        self.started = time.monotonic()

    def snapshot(self):
        free, total = torch.cuda.mem_get_info()
        result = {"allocated_gib": torch.cuda.memory_allocated() / GIB,
                  "reserved_gib": torch.cuda.memory_reserved() / GIB,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
                  "free_gib": free / GIB, "total_gib": total / GIB}
        self.result["lifecycle_peak_allocated_gib"] = max(self.result["lifecycle_peak_allocated_gib"], result["peak_allocated_gib"])
        self.result["lifecycle_peak_reserved_gib"] = max(self.result["lifecycle_peak_reserved_gib"], result["peak_reserved_gib"])
        return result

    def save(self):
        self.result["elapsed_seconds"] = time.monotonic() - self.started
        write_json(self.path, self.result)

    def check_finite(self, name, tensor, continue_probe):
        if not bool(torch.isfinite(tensor).all()):
            self.result.setdefault("numerical_failures", []).append({"stage": self.stage, "tensor": name})
            print("NONFINITE", self.stage, name, flush=True)
            if not continue_probe:
                raise RuntimeError(f"Nonfinite {name}; cannot claim successful training")

    @contextmanager
    def measure(self, name):
        self.stage = name
        torch.cuda.synchronize()
        before = self.snapshot()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        print("BEGIN", name, json.dumps(before), flush=True)
        try:
            yield
            torch.cuda.synchronize()
        finally:
            row = {"stage": name, "before": before, **self.snapshot(),
                   "seconds": time.monotonic() - started}
            self.result["stages"].append(row)
            self.save()
            print("END", name, json.dumps(row), flush=True)


def run(config, outpainting, args):
    mode = args.mode
    expected = json.loads((args.output_dir / f"{mode}_parameter_map.json").read_text())
    assert expected["verdict"] == "TRAINABLE PARAMETER MAP: PASS"
    batch, factors, provenance = load_batch(config, outpainting, args.record_id)
    text = torch.load(args.output_dir / "empty_prompt_features.pt", map_location="cpu", weights_only=True)
    assert text["prompt"] == "" and text["base"] == str(config.model.model_path)
    batch["conditions"] = text["conditions"]
    set_seed(config.seed)
    accelerator = Accelerator(mixed_precision=config.acceleration.mixed_precision_mode,
                              gradient_accumulation_steps=config.optimization.gradient_accumulation_steps)
    assert accelerator.num_processes == 1
    tag = f"{mode}_nonfinite_memory_probe" if args.continue_nonfinite_memory_probe else mode
    if args.step_seed is not None:
        tag += f"_step_seed_{args.step_seed}"
    meter = Measurements(args.output_dir / f"{tag}_result.json", {
        "mode": mode, "hardware": hardware(), "sample": provenance,
        "gradient_checkpointing": config.optimization.enable_gradient_checkpointing,
        "production_config": str(args.config.resolve()), "steps_requested": args.steps,
        "nonfinite_memory_probe": args.continue_nonfinite_memory_probe,
        "model_initialization_seed": config.seed,
        "first_step_seed": args.step_seed if args.step_seed is not None else config.seed + 1,
        "production_entrypoint_gap": "Constant-prompt trainer omits audio features required by its AV connector. "
                                     "Diagnostic supplies actual empty-prompt audio features from the same official "
                                     "feature extractor; transformer audio modality remains None.",
        "parameter_audit": {k: v for k, v in expected.items() if k != "parameters"},
    })
    try:
        with meter.measure("A_model_load"):
            model = load_transformer(config.model.model_path, device="cpu", dtype=torch.bfloat16)
            model = install_outpainting(model, ArchitectureConfig(**outpainting["architecture"]), outpainting["provenance"])
            model.to(accelerator.device)
            processor = load_embeddings_processor(
                embedding_weight_paths(config.model.model_path, config.model.text_encoder_path),
                config.model.text_encoder_path, device=accelerator.device, dtype=torch.bfloat16,
            ).eval().requires_grad_(False)
            processor.feature_extractor = None  # Same persistent connectors as official trainer.
            meter.result["frozen_connector_parameters"] = sum(p.numel() for p in processor.parameters())
            meter.result["frozen_connector_bytes"] = sum(p.numel() * p.element_size() for p in processor.parameters())
            attention = model.transformer_blocks[0].attn1
            meter.result["attention_backend"] = str(attention.attention_function)
        with meter.measure("B_trainability"):
            guards = configure_trainability(model, mode, snapshots=True)
            actual = audit_parameters(model, mode)
            assert actual == expected, "Real loaded parameter map differs from meta audit"
            print(actual["verdict"], flush=True)
            meter.result["ca_frozen_snapshot_bytes"] = sum(saved.numel() * saved.element_size() for _, _, _, saved in guards)
            model.train()
            model.set_gradient_checkpointing(config.optimization.enable_gradient_checkpointing)
            for index, block in enumerate(model.transformer_blocks):
                def enter(_module, _args, _kwargs, index=index):
                    meter.result["last_block"] = index
                    print("BLOCK", meter.stage, index, flush=True)
                block.register_forward_pre_hook(enter, with_kwargs=True)
            wrapped = accelerator.prepare(model)
        with meter.measure("C_optimizer_init"):
            trainable = [p for p in model.parameters() if p.requires_grad]
            shim = object.__new__(LtxvTrainer)
            mixed_ids = {id(p) for _, p, _, _ in guards}
            groups = [{"params": [p for p in trainable if id(p) not in mixed_ids]}]
            if mixed_ids:
                groups.append({"params": [p for p in trainable if id(p) in mixed_ids], "weight_decay": 0.0})
            shim._config, shim._trainable_params, shim._accelerator = config, groups, accelerator
            LtxvTrainer._init_optimizer(shim)
            optimizer, scheduler = shim._optimizer, shim._lr_scheduler
            selected = [p for group in optimizer.param_groups for p in group["params"]]
            assert len(selected) == len({id(p) for p in selected})
            assert {id(p) for p in selected} == {id(p) for p in trainable}
            meter.result["optimizer"] = {"class": type(optimizer.optimizer).__name__,
                                         "defaults": optimizer.optimizer.defaults, "initial_states": len(optimizer.state),
                                         "scaler": str(accelerator.scaler),
                                         "groups": [{"numel": sum(p.numel() for p in group["params"]),
                                                     "weight_decay": group["weight_decay"]}
                                                    for group in optimizer.param_groups]}
        with meter.measure("D_batch_load"):
            batch = send_to_device(batch, accelerator.device)
            meter.result["batch_shape"] = list(batch["video_latents"]["latents"].shape)
            conditions = batch["conditions"]
            with torch.no_grad():
                mask = convert_to_additive_mask(conditions["prompt_attention_mask"], conditions["video_prompt_embeds"].dtype)
                video, audio, mask = processor.create_embeddings(conditions["video_prompt_embeds"],
                                                                 conditions["audio_prompt_embeds"], mask)
            batch["conditions"] = {"video_prompt_embeds": video, "audio_prompt_embeds": audio,
                                   "prompt_attention_mask": mask}
            meter.result["text_context_shape"] = list(video.shape)
            del conditions
        strategy = OutpaintingStrategy(config.training_strategy, outpainting)
        strategy.video_scale_factors = factors
        sampler = SAMPLERS[config.flow_matching.timestep_sampling_mode](**config.flow_matching.timestep_sampling_params)
        for step in range(1, args.steps + 1):
            step_seed = config.seed + step if args.step_seed is None else args.step_seed + step - 1
            set_seed(step_seed)
            with meter.measure(f"step{step}_inputs"):
                inputs = strategy.prepare_training_inputs(batch, sampler)
                meter.check_finite("sigma", inputs.video.sigma, args.continue_nonfinite_memory_probe)
                known = ~inputs.video_loss_mask
                assert torch.all(inputs.video.timesteps[known] == 0)
                meter.result.setdefault("samples", []).append({"step": step, "seed": step_seed,
                    "sigma": inputs.video.sigma.tolist(),
                    "known_tokens": int(known.sum()), "loss_tokens": int(inputs.video_loss_mask.sum()),
                    "latent_shape": list(inputs.video.latent.shape)})
            with accelerator.accumulate(wrapped):
                with meter.measure(f"step{step}_E_forward"):
                    video_prediction, audio_prediction = wrapped(video=inputs.video, audio=None, perturbations=None)
                    assert video_prediction.shape == inputs.video_targets.shape
                    meter.check_finite("prediction", video_prediction, args.continue_nonfinite_memory_probe)
                with meter.measure(f"step{step}_F_loss"):
                    loss = strategy.compute_loss(video_prediction, audio_prediction, inputs).mean()
                    meter.check_finite("loss", loss, args.continue_nonfinite_memory_probe)
                    meter.result.setdefault("losses", []).append(float(loss.detach()))
                with meter.measure(f"step{step}_G_backward"):
                    accelerator.backward(loss)
                with meter.measure(f"step{step}_gradient_audit_clip"):
                    gradients = Counter()
                    gradient_dtypes = Counter()
                    for name, p in model.named_parameters():
                        if not p.requires_grad:
                            assert p.grad is None, f"Frozen gradient: {name}"
                        elif p.grad is None:
                            raise RuntimeError(f"Expected trainable has no gradient: {name}")
                        else:
                            gradients[category(name)] += p.grad.numel()
                            gradient_dtypes[str(p.grad.dtype)] += p.grad.numel()
                    for name, p, start, _ in guards:
                        assert torch.count_nonzero(p.grad[start:]) == 0, name
                    gradnorm = accelerator.clip_grad_norm_(trainable, config.optimization.max_grad_norm)
                    meter.check_finite("gradient_norm", gradnorm, args.continue_nonfinite_memory_probe)
                    # clip_grad_norm multiplies by NaN when the norm is NaN.
                    # Even a memory-only nonfinite probe must keep CA rows frozen.
                    for _, p, start, _ in guards:
                        p.grad[start:] = 0
                    meter.result.setdefault("gradients", []).append({"step": step, "elements": dict(gradients),
                        "dtypes": dict(gradient_dtypes), "preclip_norm": float(gradnorm)})
                with meter.measure(f"step{step}_H_optimizer_step"):
                    optimizer.step()
                    with torch.no_grad():
                        for name, p, start, saved in guards:
                            assert torch.equal(p[start:], saved), name
                            assert torch.count_nonzero(optimizer.state[p]["exp_avg"][start:]) == 0, name
                            assert torch.count_nonzero(optimizer.state[p]["exp_avg_sq"][start:]) == 0, name
                    state_counts = Counter()
                    state_bytes = 0
                    for state in optimizer.state.values():
                        for value in state.values():
                            if isinstance(value, torch.Tensor):
                                state_counts[f"{value.device.type}/{value.dtype}"] += value.numel()
                                if value.is_cuda:
                                    state_bytes += value.numel() * value.element_size()
                    assert len(optimizer.state) == len(trainable), "Optimizer state did not materialize completely"
                    meter.result.setdefault("optimizer_states", []).append({"step": step,
                        "parameters": len(optimizer.state), "elements": dict(state_counts), "cuda_bytes": state_bytes})
                with meter.measure(f"step{step}_I_zero_grad"):
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    del loss, video_prediction, audio_prediction, inputs, known
        peak = meter.result["lifecycle_peak_reserved_gib"]
        total = meter.snapshot()["total_gib"]
        meter.result["memory_verdict"] = "PASS BUT UNSAFE HEADROOM" if peak / total >= 0.95 else "PASS"
        meter.result["status"] = ("MEMORY_ONLY_PASS_INVALID_NUMERICS" if meter.result.get("numerical_failures")
                                  else meter.result["memory_verdict"])
        meter.result["headroom_gib"] = total - peak
        meter.result["headroom_percent"] = 100 * (total - peak) / total
    except Exception as exc:  # noqa: BLE001 -- Preserve arbitrary diagnostic failures, never retry a smaller model.
        meter.result["status"] = "OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "BLOCKED_RUNTIME"
        meter.result["failure_stage"] = meter.stage
        meter.result["exception"] = str(exc)
        meter.result["traceback"] = traceback.format_exc()
        meter.result["failure_memory"] = meter.snapshot()
        print(meter.result["traceback"], flush=True)
    finally:
        meter.save()
        print(f"SINGLE H200 {mode.upper()}: {meter.result['status']}", flush=True)
    if meter.result["status"] not in ("PASS", "PASS BUT UNSAFE HEADROOM"):
        raise SystemExit(2 if meter.result["status"] == "OOM" else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_4k.yaml"))
    parser.add_argument("--mode", choices=("audit", "prepare-text", "fullft", "baseline"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-id", default=RECORD)
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--step-seed", type=int,
                        help="Explicit diagnostic input RNG seed; model initialization and native sampler stay unchanged")
    parser.add_argument("--continue-nonfinite-memory-probe", action="store_true",
                        help="Preserve invalid real sampler values and measure allocations; NEVER report training PASS")
    args = parser.parse_args()
    config, outpainting = load_training_config(args.config)
    if outpainting["architecture"]["architecture_version"] != 5:
        raise RuntimeError(
            "check_fullft_vram is a historical v5 diagnostic, incompatible with v6 LoRA ownership; "
            "do not reinterpret its results or run its trainability overrides on v6"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert config.optimization.batch_size == config.optimization.gradient_accumulation_steps == 1
    assert config.acceleration.mixed_precision_mode == "bf16" and config.acceleration.quantization is None
    assert config.optimization.optimizer_type == "adamw"
    assert outpainting["temporal_sampling"] == "first" and outpainting["clip_frames"] == 65
    if args.mode == "audit":
        sources = list(Path("src/ltx_outpaint").glob("*.py")) + list(Path("configs").glob("*.yaml"))
        write_json(args.output_dir / "production_source_hashes.json",
                   {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        meta_audit(config, outpainting, args.output_dir)
        return
    snapshot = json.loads((args.output_dir / "production_source_hashes.json").read_text())
    changed = [path for path, digest in snapshot.items()
               if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f"Production source/config changed since parameter audit: {changed}")
    verified = checkpoint_provenance(config.model.model_path, outpainting["provenance"])
    write_json(args.output_dir / f"{args.mode}_verified_provenance.json", verified)
    check_reservation(args.config)
    torch.set_num_threads(min(8, len(os.sched_getaffinity(0))))
    if args.mode == "prepare-text":
        assert (args.output_dir / "parameter_audit.json").is_file(), "Run parameter audit first"
        prepare_text(config, outpainting, args.output_dir)
    else:
        run(config, outpainting, args)


if __name__ == "__main__":
    main()
