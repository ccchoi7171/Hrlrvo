"""Read-only production components; all timing/instrumentation stays in this harness.

A single fixed real-cache optimizer step. No training checkpoint or cache write.
Production sources, model configuration and math are not patched on disk.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import ltx_outpaint.model as model_module
import ltx_outpaint.train as train_module
import torch
from accelerate.utils import set_seed
from ltx_outpaint.data import (
    OutpaintingCacheDataset,
    audit_ltx_cache,
    collate_outpainting,
)
from ltx_outpaint.health_logging import parameter_health_group
from ltx_outpaint.train import (
    OutpaintingTrainer,
    _build_anchor_cache,
    load_training_config,
)
from ltx_trainer.model_loader import read_video_scale_factors

ROOT = Path(__file__).resolve().parents[2]
RECORD = "0002e965-b236-4ec5-846d-4d1b919cf3f0.pt"
GIB = 2**30


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / GIB,
        "reserved_gib": torch.cuda.memory_reserved() / GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "free_gib": free / GIB,
        "total_gib": total / GIB,
    }


class Instrument:
    def __init__(self, directory, profile=False):
        self.directory = directory
        self.profile = profile
        self.phase = "setup"
        self.block = -1
        self.branch = "Shared"
        self.phases = []
        self.events = []
        self.counts = Counter()
        self.block_backward = defaultdict(dict)
        self.undo = []
        self.shapes = {}
        self.inputs_summary = {}

    @contextlib.contextmanager
    def stage(self, name, sync=True):
        prior = self.phase
        self.phase = name
        if sync:
            torch.cuda.synchronize()
        before = memory()
        start = time.perf_counter()
        print(f"BEGIN {name} {before}", flush=True)
        try:
            with torch.profiler.record_function("STAGE/" + name):
                yield
        finally:
            if sync:
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            row = {
                "stage": name,
                "seconds": seconds,
                "before": before,
                "after": memory(),
            }
            self.phases.append(row)
            save(self.directory / "timings_progress.json", self.phases)
            print(f"END {name}: {seconds:.6f}s {row['after']}", flush=True)
            self.phase = prior

    @contextlib.contextmanager
    def scope(self, name):
        label = f"HR/{self.phase}/b{self.block:02d}/{self.branch}/{name}"
        self.counts[label] += 1
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        begin = time.perf_counter()
        ctx = (
            torch.profiler.record_function(label)
            if self.profile
            else contextlib.nullcontext()
        )
        try:
            with ctx:
                yield
        finally:
            end.record()
            self.events.append((label, start, end, time.perf_counter() - begin))

    def patch(self, obj, name, function):
        old = getattr(obj, name)
        self.undo.append((obj, name, old))
        setattr(obj, name, function)
        return old

    def wrap_scope(self, obj, name, label, branch=None):
        old = getattr(obj, name)

        @functools.wraps(old)
        def wrapped(*args, **kwargs):
            previous = self.branch
            if branch:
                self.branch = branch
            try:
                with self.scope(label):
                    return old(*args, **kwargs)
            finally:
                self.branch = previous

        self.patch(obj, name, wrapped)

    def install(self, trainer):
        for name, branch in [
            ("native_local", "Native"),
            ("adaptive_local", "Adaptive"),
            ("global_attention", "Global"),
        ]:
            self.wrap_scope(model_module, name, "branch", branch)
        # Do not wrap nn.Module backward hooks: native block args are dataclasses.
        for i, block in enumerate(trainer._transformer.transformer_blocks):
            old = block.forward

            def wrapped(*args, _i=i, _old=old, **kwargs):
                previous = self.block
                self.block = _i
                video = args[0] if args else kwargs["video"]
                if self.phase == "forward" and video.x.requires_grad:

                    def end_gradient(grad, idx=_i):
                        e = torch.cuda.Event(enable_timing=True)
                        e.record()
                        self.block_backward[idx]["end"] = e
                        return grad

                    video.x.register_hook(end_gradient)
                try:
                    with self.scope("block"):
                        output = _old(*args, **kwargs)
                    if self.phase == "forward" and output[0].x.requires_grad:

                        def begin_gradient(grad, idx=_i):
                            e = torch.cuda.Event(enable_timing=True)
                            e.record()
                            self.block_backward[idx]["start"] = e
                            return grad

                        output[0].x.register_hook(begin_gradient)
                    return output
                finally:
                    self.block = previous

            self.patch(block, "forward", wrapped)
            a = block.attn1
            self.wrap_scope(a, "forward", "MultiView")
            self.wrap_scope(a.to_out, "forward", "O", "SharedO")
            self.wrap_scope(block.attn2, "forward", "TextCA", "TextCA")
            self.wrap_scope(block.ff, "forward", "FFN", "FFN")
            for scale in ("2", "4"):
                self.wrap_scope(a.compressors[scale], "forward", "C" + scale)
                self.wrap_scope(a.decompressors[scale], "forward", "R" + scale)
            self.wrap_scope(a.local_fusion, "forward", "LocalFusion", "LocalFusion")
            self.wrap_scope(a.global_fusion, "forward", "GlobalFusion", "GlobalFusion")
            # Fine provenance labels for subsequent isolated block replays.
            # The branch label is retained, exposing shared OP in all views.
            if self.profile:
                from ltx_outpaint.lora import LinearLoRA

                for name, module in block.named_modules():
                    if isinstance(module, LinearLoRA):
                        role = (
                            "F"
                            if "ffn_lora" in name
                            else "OP"
                            if "shared_outpainting_lora" in name
                            else "A"
                            if "adaptive_lora" in name
                            else "G"
                        )
                        self.wrap_scope(module, "forward", f"LoRA_{role}")
        for obj, name, stage in [
            (trainer._anchor_cache, "complete_batch", "anchor_cache_completion"),
            (train_module, "send_to_device", "H2D"),
            (trainer._embeddings_processor, "create_embeddings", "empty_connector"),
            (
                trainer._training_strategy,
                "prepare_training_inputs",
                "conditioning_noise_mask",
            ),
            (trainer._transformer, "forward", "forward"),
            (trainer._training_strategy, "compute_loss", "loss"),
        ]:
            old = getattr(obj, name)

            def staged(*args, _old=old, _stage=stage, **kwargs):
                with self.stage(_stage):
                    result = _old(*args, **kwargs)
                if _stage == "conditioning_noise_mask":
                    v = result.video
                    self.inputs_summary = {
                        "shape": list(v.latent.shape),
                        "sigma": v.sigma.detach().cpu().tolist(),
                        "geometry": str(v.geometry),
                        "known_token_count": int((v.timesteps == 0).sum()),
                        "latent_sha256": hashlib.sha256(
                            v.latent.detach()
                            .cpu()
                            .contiguous()
                            .view(torch.uint8)
                            .numpy()
                            .tobytes()
                        ).hexdigest(),
                    }
                return result

            self.patch(obj, name, staged)

    def finish(self):
        torch.cuda.synchronize()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        details = defaultdict(
            lambda: {"calls": 0, "gpu_seconds": 0.0, "cpu_seconds": 0.0}
        )
        for label, start, stop, cpu in self.events:
            row = details[label]
            row["calls"] += 1
            row["gpu_seconds"] += start.elapsed_time(stop) / 1000
            row["cpu_seconds"] += cpu
        backwards = {
            str(i): {
                "seconds": e["start"].elapsed_time(e.get("end", end)) / 1000,
                "end_is_backward_boundary": "end" not in e,
            }
            for i, e in self.block_backward.items()
            if "start" in e
        }
        save(
            self.directory / "scopes.json",
            {"details": details, "counts": self.counts, "block_backward": backwards},
        )
        save(self.directory / "sample_runtime.json", self.inputs_summary)
        for obj, name, old in reversed(self.undo):
            setattr(obj, name, old)
        self.undo.clear()


def graph_counts(tensor):
    counts = Counter()
    seen = set()
    todo = [tensor.grad_fn]
    while todo:
        node = todo.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        counts[type(node).__name__] += 1
        todo.extend(n for n, _ in node.next_functions)
    return {"total_nodes": len(seen), "node_types": counts}


def gradient_health(model):
    groups = defaultdict(list)
    param_count = Counter()
    missing = Counter()
    frozen_violations = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                frozen_violations.append(name)
            continue
        group = parameter_health_group(name)
        assert group, name
        param_count[group] += p.numel()
        if p.grad is None:
            missing[group] += 1
        else:
            groups[group].append(p.grad.detach())
    keys = sorted(groups)
    vals = []
    for key in keys:
        grads = groups[key]
        vals.append(
            torch.stack(
                [
                    sum(g.float().square().sum() for g in grads),
                    sum((~torch.isfinite(g)).sum() for g in grads).to(torch.float32),
                ]
            )
        )
    numbers = torch.stack(vals).cpu().tolist()
    return {
        "groups": {
            key: {
                "parameter_count": param_count[key],
                "tensor_count": len(groups[key]),
                "missing_grad_tensors": missing[key],
                "norm": numbers[i][0] ** 0.5,
                "nonfinite_elements": int(numbers[i][1]),
            }
            for i, key in enumerate(keys)
        },
        "frozen_gradient_violations": frozen_violations,
    }


def profiler_export(prof, directory):
    # Preserve raw Kineto evidence before expensive Python FunctionEvent trees.
    prof.export_chrome_trace(str(directory / "trace.json"))
    averages = []
    for e in prof.key_averages():
        averages.append(
            {
                "name": e.key,
                "device_type": str(e.device_type),
                "count": e.count,
                "self_cpu_us": e.self_cpu_time_total,
                "cpu_us": e.cpu_time_total,
                "self_cuda_us": e.self_device_time_total,
                "cuda_us": e.device_time_total,
                "cpu_memory": e.cpu_memory_usage,
                "cuda_memory": e.device_memory_usage,
            }
        )
    save(
        directory / "profiler_ops.json",
        sorted(averages, key=lambda r: r["self_cuda_us"], reverse=True),
    )
    # Correlate autograd sequence numbers to annotated forward operator parents.
    events = prof.events()
    forward = {}
    ambiguous = set()

    def scope(event):
        p = event.cpu_parent
        while p:
            if p.name.startswith("HR/"):
                return p.name
            p = p.cpu_parent
        return None

    for e in events:
        tag = scope(e)
        if (
            e.sequence_nr >= 0
            and tag
            and e.scope != 1
            and not e.name.startswith("autograd::")
        ):
            key = (e.thread, e.sequence_nr)
            if key in forward and forward[key] != tag:
                ambiguous.add(key)
            forward[key] = tag
    attribution = defaultdict(
        lambda: {"self_cuda_us": 0.0, "self_cpu_us": 0.0, "count": 0}
    )
    kernels = Counter()
    sample_backends = []
    seq_missing = Counter()
    for e in events:
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kernels[e.name] += (
                e.duration if hasattr(e, "duration") else e.time_range.elapsed_us()
            )
            continue
        tag = scope(e)
        p = e
        backward = False
        # Recompute aten ops are new forward work nested inside a backward
        # unpack. Keep their actual HR branch instead of charging the entire
        # recompute to the backward Function that triggered the unpack.
        recompute = bool(tag and "/backward_only/" in tag)
        if not recompute:
            while p:
                if p.scope == 1:  # backward Function, verified in torch 2.8
                    backward = True
                    key = (p.fwd_thread, p.sequence_nr)
                    mapped = forward.get(key) if key not in ambiguous else None
                    tag = mapped or tag
                    if mapped is None:
                        seq_missing[p.name] += 1
                    break
                p = p.cpu_parent
        tag = tag or "UNATTRIBUTED"
        key = (
            "recompute/" if recompute else "backward/" if backward else "forward/"
        ) + tag
        a = attribution[key]
        a["self_cuda_us"] += e.self_device_time_total
        a["self_cpu_us"] += e.self_cpu_time_total
        a["count"] += 1
        if any(
            s in e.name
            for s in (
                "scaled_dot_product",
                "efficient_attention",
                "flash_attention",
                "cudnn_attention",
            )
        ):
            sample_backends.append(
                {
                    "name": e.name,
                    "scope": tag,
                    "backward": backward,
                    "count": 1,
                    "self_cuda_us": e.self_device_time_total,
                    "cuda_us": e.device_time_total,
                    "sequence_nr": e.sequence_nr,
                }
            )
    save(
        directory / "profiler_attribution.json",
        {
            "scopes": attribution,
            "unmatched_backward": seq_missing,
            "ambiguous_sequence_keys": len(ambiguous),
            "backend_events": sample_backends,
        },
    )
    save(directory / "profiler_kernels.json", kernels.most_common())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--refine-ncdhw", action="store_true",
                        help="Isolated layout-only A/B: contiguous NCDHW at existing R2/R4 Conv3d input")
    parser.add_argument("--cudnn-benchmark", action="store_true",
                        help="Fresh-process convolution autotuning A/B, before any convolution")
    parser.add_argument("--repeat-warm-step", action="store_true",
                        help="One further same-sample optimizer step in the warmed diagnostic process")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ins = Instrument(out, args.profile)
    result = {
        "status": "RUNNING",
        "production_source_modified": False,
        "record_id": RECORD,
        "diagnostic_refine_ncdhw": args.refine_ncdhw,
        "diagnostic_cudnn_benchmark": args.cudnn_benchmark,
    }
    try:
        assert torch.cuda.device_count() == 1
        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        config, op = load_training_config(ROOT / "main/configs/train_4k.yaml")
        config.output_dir = str(
            out / "isolated_trainer"
        )  # output location only, no algorithm changes
        assert config.model.load_checkpoint is None
        assert (
            config.optimization.batch_size
            == config.optimization.gradient_accumulation_steps
            == 1
        )
        assert config.optimization.enable_gradient_checkpointing
        result["config"] = config.model_dump(mode="json")
        result["outpainting"] = op
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
        result["before"] = memory()
        save(out / "result.json", result)
        with ins.stage("cache_contract_setup"):
            factors = read_video_scale_factors(config.model.model_path)
            audit = audit_ltx_cache(
                op["cache_root"],
                scale_factors=factors,
                requested_canvas=tuple(op["canvas_pixels"]),
            )
            cache = _build_anchor_cache(audit, config, op)
            dataset = OutpaintingCacheDataset(audit, cache, seed=config.seed)
            index = next(
                i
                for i, p in enumerate(dataset.files)
                if p.relative_to(audit.root / "latents").as_posix() == RECORD
            )
        with ins.stage("real_CPU_data_load"):
            sample = dataset[(index, 0)]
            assert sample["_anchor_entries"] is not None, (
                "Refuse Anchor miss / VAE encode"
            )
            batch = collate_outpainting([sample])
        with ins.stage("model_load"):
            trainer = OutpaintingTrainer(
                config, op, cache_audit=audit, anchor_cache=cache
            )
        result["parameter_counts"] = trainer._parameter_counts
        with ins.stage("optimizer_setup"):
            trainer._init_optimizer()
            trainer._init_timestep_sampler()
        trainer._transformer.train()
        set_seed(config.seed)
        if args.refine_ncdhw:
            for block in trainer._transformer.transformer_blocks:
                for restore in block.attn1.decompressors.values():
                    old = restore.refine.forward

                    def contiguous_refine(volume, _old=old):
                        return _old(volume.contiguous())

                    ins.patch(restore.refine, "forward", contiguous_refine)
        ins.install(trainer)
        prof = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            )
            if args.profile
            else None
        )
        torch.cuda.reset_peak_memory_stats()
        with trainer._accelerator.accumulate(trainer._transformer):
            with prof if prof else contextlib.nullcontext():
                t0 = time.perf_counter()
                output = trainer._training_step(batch)
                result["training_step_forward_wall"] = time.perf_counter() - t0
                result["graph"] = graph_counts(output.loss)
                with ins.stage("backward_only"):
                    trainer._accelerator.backward(output.loss.mean())
                # Mark end before health audit or optimizer, no per-parameter sync inside backward.
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                end.synchronize()
                for i in ins.block_backward:
                    ins.block_backward[i].setdefault("end", end)
            result["loss"] = output.loss.detach().cpu().tolist()
            result["sigma"] = output.sigma.cpu().tolist()
            result["training_peak_before_diagnostics"] = memory()
            with ins.stage("gradient_audit_OUTSIDE_TIMING_AND_PROFILER"):
                result["gradient_health"] = gradient_health(trainer._transformer)
            with ins.stage("clip_grad_norm_only"):
                norm = trainer._accelerator.clip_grad_norm_(
                    trainer._trainable_params, config.optimization.max_grad_norm
                )
            with ins.stage("optimizer_step"):
                trainer._optimizer.step()
            with ins.stage("zero_grad"):
                trainer._optimizer.zero_grad()
            with ins.stage("scheduler_step"):
                trainer._lr_scheduler.step()
        result["global_grad_norm_before_clip"] = float(norm.detach().cpu())
        result["lr_after_step"] = [g["lr"] for g in trainer._optimizer.param_groups]
        result["optimizer_state_numel"] = sum(
            v.numel()
            for state in trainer._optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        )
        assert cache.created_count == 0 and cache._encoder is None
        result["status"] = "PASS"
        result["end"] = memory()
        ins.finish()
        result["phases"] = ins.phases
        save(out / "result.json", result)
        if prof:
            print("Exporting profiler...", flush=True)
            profiler_export(prof, out)
        print(
            "ONE REAL OPTIMIZER STEP: PASS",
            json.dumps(
                {
                    k: result[k]
                    for k in ("loss", "sigma", "global_grad_norm_before_clip", "end")
                }
            ),
            flush=True,
        )
        if args.repeat_warm_step:
            assert args.cudnn_benchmark and not args.profile and not args.refine_ncdhw
            # Distinct hypothesis: amortized autotuning with materialized Adam states.
            # Same real sample and random draw; weights now include the first update.
            warm_dir = out / "warm_step_2"
            warm_dir.mkdir()
            ins = Instrument(warm_dir)
            warm = {
                "status": "RUNNING",
                "diagnostic_optimizer_call": 2,
                "production_source_modified": False,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "parameter_counts": trainer._parameter_counts,
                "note": "Same real sample and seed/sigma; post-step-1 weights; same live Adam/scheduler and cuDNN cache",
            }
            with ins.stage("real_CPU_data_load"):
                batch = collate_outpainting([dataset[(index, 0)]])
            set_seed(config.seed)
            ins.install(trainer)
            torch.cuda.reset_peak_memory_stats()
            with trainer._accelerator.accumulate(trainer._transformer):
                output = trainer._training_step(batch)
                with ins.stage("backward_only"):
                    trainer._accelerator.backward(output.loss.mean())
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                end.synchronize()
                for i in ins.block_backward:
                    ins.block_backward[i].setdefault("end", end)
                warm["loss"] = output.loss.detach().cpu().tolist()
                warm["sigma"] = output.sigma.detach().cpu().tolist()
                with ins.stage("gradient_audit_OUTSIDE_TIMING_AND_PROFILER"):
                    warm["gradient_health"] = gradient_health(trainer._transformer)
                with ins.stage("clip_grad_norm_only"):
                    norm = trainer._accelerator.clip_grad_norm_(
                        trainer._trainable_params, config.optimization.max_grad_norm
                    )
                with ins.stage("optimizer_step"):
                    trainer._optimizer.step()
                with ins.stage("zero_grad"):
                    trainer._optimizer.zero_grad()
                with ins.stage("scheduler_step"):
                    trainer._lr_scheduler.step()
            warm["global_grad_norm_before_clip"] = float(norm.detach().cpu())
            warm["lr_after_step"] = [g["lr"] for g in trainer._optimizer.param_groups]
            warm["end"] = memory()
            warm["phases"] = ins.phases
            warm["status"] = "PASS"
            ins.finish()
            save(warm_dir / "result.json", warm)
            print("WARM SECOND OPTIMIZER STEP: PASS", json.dumps(warm["loss"]), flush=True)
    except BaseException as exc:
        result.update(
            status="OOM" if isinstance(exc, torch.OutOfMemoryError) else "FAIL",
            error=repr(exc),
            traceback=traceback.format_exc(),
            phases=ins.phases,
        )
        try:
            result["failure_memory"] = memory()
        except Exception as memory_error:  # noqa: BLE001 - CUDA can be poisoned by failure
            result["failure_memory_error"] = repr(memory_error)
        save(out / "result.json", result)
        raise


if __name__ == "__main__":
    main()
