"""Bounded real training through the production entrypoint; diagnostics only.

Only run_training is replaced in memory with a fixed-cache, measured loop.
The production main itself is responsible for the new cuDNN startup setting.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/backward_bottleneck_analysis"))
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from ltx_outpaint import train
from ltx_outpaint.data import OutpaintingCacheDataset, collate_outpainting
from ltx_outpaint.health_logging import parameter_health_group
from profile_training import RECORD, Instrument, gradient_health, memory, save


def digest(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--baseline-false", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "RUNNING",
        "steps": [],
        "baseline_false": args.baseline_false,
        "initial_flags": {
            "benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "first_convolutions": {},
        "method": "actual train.main; diagnostic-only bounded loop callback",
    }
    assert not torch.backends.cudnn.benchmark, (
        "Must start a genuinely fresh default process"
    )
    # Baseline is the old main with only the new assignment removed, in memory.
    if args.baseline_false:
        tree = ast.parse(Path(train.__file__).read_text())
        fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        assert ast.unparse(fn.body[0]) == "torch.backends.cudnn.benchmark = True"
        fn.body.pop(0)
        namespace = dict(train.__dict__)
        exec(  # noqa: S102 - trusted local AST; diagnostic-only pre-change entrypoint
            compile(
                ast.Module(body=[fn], type_ignores=[]), "<pre-change-main>", "exec"
            ),
            namespace,
        )
        entry = namespace["main"]
    else:
        namespace = train.__dict__
        entry = train.main
    expected_flag = not args.baseline_false
    for name in ("conv1d", "conv2d", "conv3d"):
        original = getattr(F, name)

        def first(*a, _name=name, _original=original, **kw):
            assert torch.backends.cudnn.benchmark == expected_flag
            result["first_convolutions"][_name] = {
                "benchmark": torch.backends.cudnn.benchmark,
                "input_shape": list(a[0].shape),
                "input_dtype": str(a[0].dtype),
                "deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            }
            save(out / "startup.json", result)
            setattr(F, _name, _original)
            return _original(*a, **kw)

        setattr(F, name, first)

    def measured_loop(trainer, **unused):
        assert torch.backends.cudnn.benchmark == expected_flag
        cfg = trainer._config
        assert (
            cfg.optimization.batch_size
            == cfg.optimization.gradient_accumulation_steps
            == 1
        )
        assert cfg.optimization.enable_gradient_checkpointing
        result["config"] = cfg.model_dump(mode="json")
        result["parameter_counts"] = trainer._parameter_counts
        result["startup_seconds"] = time.perf_counter() - started
        result["versions"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
        dataset = OutpaintingCacheDataset(
            trainer._cache_audit, trainer._anchor_cache, seed=cfg.seed
        )
        valid = []
        anchor_root = Path(trainer._outpainting["anchor_cache_root"]) / "latents"
        for i, p in enumerate(dataset.files):
            rid = p.relative_to(trainer._cache_audit.root / "latents").as_posix()
            if (anchor_root / rid).is_file():
                if rid == RECORD:
                    valid.insert(0, (i, rid))
                else:
                    valid.append((i, rid))
            if len(valid) >= 3 and valid[0][1] == RECORD:
                break
        assert len(valid) == 3 and valid[0][1] == RECORD
        result["records"] = valid
        trainer._init_optimizer()
        trainer._init_timestep_sampler()
        trainer._transformer.train()
        set_seed(cfg.seed)
        params = dict(trainer._transformer.named_parameters())
        reps = {}
        for name, p in params.items():
            group = parameter_health_group(name)
            if (
                p.requires_grad
                and group
                and group not in reps
                and ("lora" not in group or ".lora_B." in name)
            ):
                reps[group] = (name, p)
        assert len(reps) == 10
        result["representatives"] = {g: n for g, (n, p) in reps.items()}
        result["initial_parameter_hashes"] = {
            g: digest(p) for g, (n, p) in reps.items()
        }
        # CPU-only representative frozen values; entire frozen ownership audited after backward.
        frozen = [
            (n, p)
            for n, p in params.items()
            if not p.requires_grad
            and any(
                s in n
                for s in (
                    "transformer_blocks.0.attn1.to_q.weight",
                    "transformer_blocks.0.attn1.to_out.0.weight",
                    "transformer_blocks.0.attn2.to_q.weight",
                    "transformer_blocks.0.ff.net.2.weight",
                )
            )
        ]
        frozen_before = {
            n: p.detach().flatten()[:1024].cpu().clone() for n, p in frozen
        }
        route = [0, 0, 1, 2, 0]
        for step in range(1, args.steps + 1):
            stepdir = out / f"step_{step:02d}"
            stepdir.mkdir()
            ins = Instrument(stepdir)
            row = {
                "step": step,
                "record": valid[route[(step - 1) % 5]][1],
                "benchmark": torch.backends.cudnn.benchmark,
            }
            if step == 2:
                # Same random draw as prior analysis; genuine post-update weights/Adam.
                set_seed(cfg.seed)
            with ins.stage("real_CPU_data_load"):
                sample = dataset[(valid[route[(step - 1) % 5]][0], 0)]
                assert sample["_anchor_entries"] is not None, (
                    "Refuse missing Anchor / VAE encoding"
                )
                batch = collate_outpainting([sample])
                row["gt_shape"] = list(batch["video_latents"]["latents"].shape)
            ins.install(trainer)
            shape_counts = Counter()
            handles = []
            for bi, block in enumerate(trainer._transformer.transformer_blocks):
                for scale in ("2", "4"):

                    def observe(
                        module,
                        inputs,
                        _bi=bi,
                        _scale=scale,
                        _ins=ins,
                        _counts=shape_counts,
                    ):
                        v = inputs[0]
                        key = json.dumps(
                            {
                                "block": _bi,
                                "R": _scale,
                                "phase": _ins.phase,
                                "shape": list(v.shape),
                                "stride": list(v.stride()),
                                "dtype": str(v.dtype),
                            }
                        )
                        _counts[key] += 1

                    handles.append(
                        block.attn1.decompressors[
                            scale
                        ].refine.register_forward_pre_hook(observe)
                    )
            torch.cuda.reset_peak_memory_stats()
            wall = time.perf_counter()
            with trainer._accelerator.accumulate(trainer._transformer):
                output = trainer._training_step(batch)
                row["loss"] = float(output.loss.detach().cpu())
                row["sigma"] = output.sigma.detach().cpu().tolist()
                assert math.isfinite(row["loss"]) and all(
                    math.isfinite(v) for v in row["sigma"]
                )
                with ins.stage("backward_only"):
                    trainer._accelerator.backward(output.loss.mean())
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                end.synchronize()
                for b in ins.block_backward:
                    ins.block_backward[b].setdefault("end", end)
                row["memory_before_audit"] = memory()
                with ins.stage("audit_OUTSIDE_PHASE_SUM"):
                    row["gradient_health"] = gradient_health(trainer._transformer)
                    assert not row["gradient_health"]["frozen_gradient_violations"]
                    assert len(row["gradient_health"]["groups"]) == 10
                    for group, h in row["gradient_health"]["groups"].items():
                        assert (
                            h["missing_grad_tensors"] == 0
                            and h["nonfinite_elements"] == 0
                            and h["norm"] > 0
                        ), (group, h)
                    before = {g: p.detach().cpu().clone() for g, (n, p) in reps.items()}
                    if step == 1:
                        grad_values = {
                            n: p.grad.detach().cpu()
                            for n, p in params.items()
                            if p.grad is not None
                            and any(
                                f"transformer_blocks.{b}.attn1.decompressors." in n
                                for b in (0, 1, 47)
                            )
                        }
                        torch.save(
                            grad_values, out / "first_step_selected_R_gradients.pt"
                        )
                        del grad_values
                with ins.stage("clip_grad_norm_only"):
                    norm = trainer._accelerator.clip_grad_norm_(
                        trainer._trainable_params, cfg.optimization.max_grad_norm
                    )
                with ins.stage("optimizer_step"):
                    trainer._optimizer.step()
                with ins.stage("zero_grad"):
                    trainer._optimizer.zero_grad()
                with ins.stage("scheduler_step"):
                    trainer._lr_scheduler.step()
            row["global_grad_norm_before_clip"] = float(norm.detach().cpu())
            row["parameter_updates"] = {
                g: not torch.equal(before[g], p.detach().cpu())
                for g, (n, p) in reps.items()
            }
            assert all(row["parameter_updates"].values())
            row["frozen_representative_unchanged"] = {
                n: torch.equal(frozen_before[n], p.detach().flatten()[:1024].cpu())
                for n, p in frozen
            }
            assert all(row["frozen_representative_unchanged"].values())
            row["lr_after_step"] = [g["lr"] for g in trainer._optimizer.param_groups]
            row["memory"] = memory()
            row["wall_including_diagnostics"] = time.perf_counter() - wall
            ins.finish()
            for h in handles:
                h.remove()
            save(
                stepdir / "R_shapes.json",
                [
                    {"metadata": json.loads(k), "calls": v}
                    for k, v in shape_counts.items()
                ],
            )
            row["phases"] = ins.phases
            row["phase_sum_seconds"] = sum(
                p["seconds"]
                for p in ins.phases
                if p["stage"] != "audit_OUTSIDE_PHASE_SUM"
            )
            row["input"] = ins.inputs_summary
            row["status"] = "PASS"
            save(stepdir / "result.json", row)
            result["steps"].append(row)
            save(out / "result.json", result)
            print(
                "COMPLETED_STEP",
                step,
                "PHASE_SUM",
                row["phase_sum_seconds"],
                "LOSS",
                row["loss"],
                flush=True,
            )
            del output, batch, sample, before, ins, handles, shape_counts
            gc.collect()
        assert (
            trainer._anchor_cache.created_count == 0
            and trainer._anchor_cache._encoder is None
        )
        result["cache_created"] = 0
        result["optimizer_state_numel"] = sum(
            v.numel()
            for state in trainer._optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        )
        result["status"] = "PASS"
        save(out / "result.json", result)

    namespace["run_training"] = measured_loop
    sys.argv = [
        "ltx_outpaint.train",
        str(ROOT / "main/configs/train_4k.yaml"),
        "--output-dir",
        str(out / "isolated_run"),
        "--disable-progress-bars",
    ]
    started = time.perf_counter()
    try:
        entry()
    except BaseException as exc:
        result.update(
            status="OOM" if isinstance(exc, torch.OutOfMemoryError) else "FAIL",
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        try:
            result["failure_memory"] = memory()
        except Exception as memory_error:  # noqa: BLE001 - record CUDA diagnostic failure
            result["failure_memory_error"] = repr(memory_error)
        save(out / "result.json", result)
        raise


if __name__ == "__main__":
    main()
