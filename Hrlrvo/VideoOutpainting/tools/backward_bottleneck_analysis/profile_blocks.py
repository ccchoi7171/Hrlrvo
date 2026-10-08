"""Controlled real v8 block replay, not a substitute for the full-model baseline.

All experiments operate on isolated instances and current actual cache conditioning.
No production edits, checkpoints, cache generation, or long training.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
from dataclasses import replace
from pathlib import Path

import torch
from accelerate.utils import set_seed
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.attention import PytorchAttention
from ltx_outpaint.data import (
    OutpaintingCacheDataset,
    audit_ltx_cache,
    collate_outpainting,
)
from ltx_outpaint.train import (
    OutpaintingTrainer,
    _build_anchor_cache,
    load_training_config,
)
from ltx_trainer.model_loader import read_video_scale_factors
from profile_training import (
    RECORD,
    ROOT,
    Instrument,
    gradient_health,
    memory,
    profiler_export,
    save,
)
from torch.utils import checkpoint


class CapturedInputs(Exception):
    pass


def prepare(trainer, batch):
    holder = {}

    def capture(*args, **kwargs):
        assert not args
        holder.update(kwargs)
        raise CapturedInputs

    old = trainer._transformer.forward
    trainer._transformer.forward = capture
    try:
        trainer._training_step(batch)
    except CapturedInputs:
        pass
    finally:
        trainer._transformer.forward = old
    return holder["video"]


def cpu_grads(block):
    return {
        n: p.grad.detach().cpu()
        for n, p in block.named_parameters()
        if p.requires_grad and p.grad is not None
    }


def compare_grads(block, reference):
    errors = []
    for n, p in block.named_parameters():
        if n not in reference:
            continue
        g = p.grad.detach()
        r = reference[n].to(g.device)
        errors.append(
            torch.stack(
                [
                    (g - r).float().square().sum(),
                    r.float().square().sum(),
                    (g - r).abs().max().float(),
                ]
            )
        )
    stack = torch.stack(errors)
    nums = (
        torch.stack([stack[:, 0].sum(), stack[:, 1].sum(), stack[:, 2].max()])
        .cpu()
        .tolist()
    )
    return {
        "relative_l2": (nums[0] / max(nums[1], 1e-30)) ** 0.5,
        "max_absolute": nums[2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=0,
        help="Warm up once, then repeat without profiler/module hooks",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["k4_off", "k5_off", "k4_nonreentrant", "k4_reentrant"],
    )
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    cfg, op = load_training_config(ROOT / "main/configs/train_4k.yaml")
    cfg.output_dir = str(root / "isolated_trainer")
    factors = read_video_scale_factors(cfg.model.model_path)
    audit = audit_ltx_cache(
        op["cache_root"],
        scale_factors=factors,
        requested_canvas=tuple(op["canvas_pixels"]),
    )
    cache = _build_anchor_cache(audit, cfg, op)
    dataset = OutpaintingCacheDataset(audit, cache, seed=cfg.seed)
    ix = next(
        i
        for i, p in enumerate(dataset.files)
        if p.relative_to(audit.root / "latents").as_posix() == RECORD
    )
    sample = dataset[(ix, 0)]
    assert sample["_anchor_entries"] is not None
    batch = collate_outpainting([sample])
    trainer = OutpaintingTrainer(cfg, op, cache_audit=audit, anchor_cache=cache)
    trainer._init_timestep_sampler()
    set_seed(cfg.seed)
    trainer._transformer.train()
    video = prepare(trainer, batch)
    model = trainer._transformer
    with torch.no_grad(), trainer._accelerator.autocast():
        base = model.video_args_preprocessor.prepare(video, None)
    perturb = BatchedPerturbationConfig.empty(1, 48, base.x.device, base.x.dtype)
    references = {}
    results = []
    for experiment in args.experiments:
        index = 1 if experiment.startswith("k5") else 0
        mode = experiment.split("_", 1)[1]
        benchmark = mode.endswith("_benchmark")
        mode = mode.removesuffix("_benchmark")
        block = model.transformer_blocks[index]
        directory = root / experiment
        directory.mkdir(parents=True, exist_ok=True)
        ins = Instrument(directory, profile=not args.timing_repeats)
        if not args.timing_repeats:
            ins.install(trainer)
        # These wrappers are stage timers for full model only; replay invokes native block directly.
        tensor_shapes = {}
        old_attention = PytorchAttention.__call__
        forced = {
            "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            "math": torch.nn.attention.SDPBackend.MATH,
        }.get(mode)

        def attention_call(
            self,
            q,
            k,
            v,
            heads,
            mask=None,
            *,
            ins=ins,
            tensor_shapes=tensor_shapes,
            forced=forced,
            old_attention=old_attention,
        ):
            key = f"{ins.branch}/{tuple(q.shape)}/{tuple(k.shape)}"
            tensor_shapes[key] = {
                "q": list(q.shape),
                "k": list(k.shape),
                "v": list(v.shape),
                "q_stride": list(q.stride()),
                "k_stride": list(k.stride()),
                "v_stride": list(v.stride()),
                "heads": heads,
                "dtype": str(q.dtype),
                "mask": None
                if mask is None
                else {"shape": list(mask.shape), "dtype": str(mask.dtype)},
                "is_causal": False,
                "configured_priority": [str(x) for x in self._priority],
            }
            old = self._priority
            if forced:
                self._priority = [forced]
            try:
                with ins.scope("SDPA"):
                    return old_attention(self, q, k, v, heads, mask)
            finally:
                self._priority = old

        if not args.timing_repeats or forced:
            ins.patch(PytorchAttention, "__call__", attention_call)
        x = base.x.detach().clone().requires_grad_(True)
        block_args = model.block_input_processor(
            replace(base, x=x),
            perturb,
            index,
            self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
            cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
        )

        def fn(hidden, _block=block, _args=block_args):
            return _block(replace(_args, x=hidden), None)[0].x

        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = {
            "experiment": experiment,
            "index": index,
            "hidden_shape": list(x.shape),
            "note": "Single actual block, real first-block current hidden; not full-model latency. All48-block full-step measured separately.",
            "timing_repeats": args.timing_repeats,
            "cudnn_benchmark": benchmark,
        }
        try:
            profiler = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    profile_memory=True,
                    with_stack=False,
                )
                if not args.timing_repeats
                else contextlib.nullcontext()
            )
            with (
                torch.backends.cudnn.flags(
                    enabled=torch.backends.cudnn.enabled,
                    benchmark=benchmark,
                    deterministic=torch.backends.cudnn.deterministic,
                    allow_tf32=torch.backends.cudnn.allow_tf32,
                ),
                profiler as prof,
            ):
                for repetition in range(args.timing_repeats + 1):
                    model.zero_grad(set_to_none=True)
                    x.grad = None
                    with ins.stage("forward"), trainer._accelerator.autocast():
                        if mode == "nonreentrant":
                            y = checkpoint.checkpoint(
                                block, block_args, None, use_reentrant=False
                            )[0].x
                        elif mode == "reentrant":
                            y = checkpoint.checkpoint(fn, x, use_reentrant=True)
                        else:
                            y = fn(x)
                        # Fixed block-only VJP, not an alternate training loss.
                        objective = (y.float() - base.x.float()).square().mean()
                    with ins.stage("backward_only"):
                        objective.backward()
                    if repetition < args.timing_repeats:
                        del y, objective
            result["memory_before_audit"] = memory()
            result["objective"] = float(objective.detach().cpu())
            result["gradient_health"] = gradient_health(block)
            if index in references:
                result["grad_comparison_to_off"] = compare_grads(
                    block, references[index]
                )
            elif mode == "off":
                references[index] = cpu_grads(block)
            result["status"] = "PASS"
            ins.finish()
            result["phases"] = ins.phases
            save(directory / "result.json", result)
            save(directory / "attention_shapes.json", tensor_shapes)
            if prof is not None:
                profiler_export(prof, directory)
            del prof, y, objective
        except Exception as exc:  # noqa: BLE001 - preserve diagnostic failure evidence
            import traceback

            result.update(
                status="FAIL",
                error=repr(exc),
                traceback=traceback.format_exc(),
                memory=memory(),
                phases=ins.phases,
            )
            save(directory / "result.json", result)
            ins.finish()
            print(result["traceback"], flush=True)
            if isinstance(exc, torch.OutOfMemoryError):
                break
        results.append(result)
        save(root / "summary.json", results)
        print("EXPERIMENT", experiment, result["status"], flush=True)
        del x, block_args, fn
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    assert cache.created_count == 0


if __name__ == "__main__":
    main()
