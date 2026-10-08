"""Actual-block A/B probe for invocation-scoped constant caching.

This is deliberately a harness-only prototype.  It patches imported module
objects in memory, never production sources, and never writes model/cache
checkpoints.  Run it only when the training GPU is otherwise idle.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import sys
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import torch

# This must precede trainer construction, model loading, and every convolution.
torch.backends.cudnn.benchmark = True

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "backward_bottleneck_analysis"))
sys.path.insert(0, str(HERE.parents[2] / "main/src"))
sys.path.insert(0, str(HERE.parents[2] / "models/LTX2.3/packages/ltx-core/src"))
sys.path.insert(0, str(HERE.parents[2] / "models/LTX2.3/packages/ltx-trainer/src"))

import ltx_outpaint.adaptive_local_attention as adaptive_module
import ltx_outpaint.compression as compression_module
import ltx_outpaint.global_attention as global_module
import ltx_outpaint.model as model_module
import ltx_outpaint.native_local_attention as native_module
import ltx_outpaint.position as position_module
from accelerate.utils import set_seed
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    PerturbationType,
)
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
from profile_blocks import RECORD, ROOT, prepare
from profile_training import Instrument, memory, save
from torch.utils import checkpoint

EXPECTED_HIDDEN = (1, 73440, 4096)
PATCHED_TORCH_MODULES = (
    compression_module,
    adaptive_module,
    native_module,
    model_module,
)


def _literal(value: Any) -> Any:
    """Return a hashable exact literal representation, or raise TypeError."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_literal(item) for item in value))
    if isinstance(value, torch.device):
        return ("device", str(value))
    if isinstance(value, torch.dtype):
        return ("dtype", str(value))
    raise TypeError(type(value).__name__)


def _unique_tensor_bytes(values) -> int:
    storages = {}
    for value in values:
        for tensor in _tensors(value):
            storage = tensor.untyped_storage()
            storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return sum(storages.values())


def _tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _nested_tensors(value: Any, seen=None):
    """Walk block inputs without descending into arbitrary model objects."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nested_tensors(item, seen)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nested_tensors(item, seen)
    elif is_dataclass(value):
        for field in fields(value):
            yield from _nested_tensors(getattr(value, field.name), seen)


def _storage_keys(values):
    result = set()
    for value in values:
        for tensor in _nested_tensors(value):
            storage = tensor.untyped_storage()
            result.add((storage.data_ptr(), storage.nbytes(), tensor.device))
    return result


class ConstantTorchProxy:
    """Cache only no-grad results of exact literal tensor/arange requests."""

    def __init__(self):
        self.cache: dict[Any, torch.Tensor] = {}
        self.building = True
        self.hits = 0
        self.misses_after_freeze = 0
        self.build_seconds = 0.0

    def __getattr__(self, name):
        return getattr(torch, name)

    def _call(self, name, args, kwargs):
        original = getattr(torch, name)
        try:
            key = (name, _literal(args), _literal(tuple(sorted(kwargs.items()))))
        except TypeError:
            return original(*args, **kwargs)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        if not self.building:
            self.misses_after_freeze += 1
            return original(*args, **kwargs)
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Constant cache may only be populated under torch.no_grad()"
            )
        begin = time.perf_counter()
        value = original(*args, **kwargs)
        self.build_seconds += time.perf_counter() - begin
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError("Refusing to cache an autograd-connected tensor")
        self.cache[key] = value
        return value

    def tensor(self, *args, **kwargs):
        return self._call("tensor", args, kwargs)

    def arange(self, *args, **kwargs):
        return self._call("arange", args, kwargs)

    def summary(self):
        return {
            "entries": len(self.cache),
            "bytes": _unique_tensor_bytes(self.cache.values()),
            "build_seconds": self.build_seconds,
            "hits": self.hits,
            "misses_after_freeze": self.misses_after_freeze,
        }


class PositionCache:
    """Invocation-local Adaptive and Global final-RoPE cache."""

    def __init__(self):
        self.cache = {}
        self.building = True
        self.hits = 0
        self.misses_after_freeze = 0
        self.build_seconds = 0.0

    @staticmethod
    def key(kind, context, cells, scale, extra):
        return (
            kind,
            id(context.positions),
            id(cells),
            scale,
            extra,
            tuple(context.positions.shape),
            context.positions.device,
        )

    def get(self, kind, original, context, cells, scale, *extra):
        key = self.key(kind, context, cells, scale, extra)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if not self.building:
            self.misses_after_freeze += 1
            return original(context, cells, scale, *extra)
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Position cache may only be populated under torch.no_grad()"
            )
        begin = time.perf_counter()
        value = original(context, cells, scale, *extra)
        self.build_seconds += time.perf_counter() - begin
        if any(t.requires_grad or t.grad_fn is not None for t in _tensors(value)):
            raise RuntimeError("Refusing to cache an autograd-connected RoPE tensor")
        self.cache[key] = value
        return value

    def summary(self):
        return {
            "entries": len(self.cache),
            "bytes": _unique_tensor_bytes(self.cache.values()),
            "build_seconds": self.build_seconds,
            "hits": self.hits,
            "misses_after_freeze": self.misses_after_freeze,
        }


@contextlib.contextmanager
def cache_scope(metadata: bool, mapped_positions: bool, final_positions: bool):
    """Install and fully restore the narrow harness-only prototype."""
    undo = []
    tensor_cache = ConstantTorchProxy() if metadata else None
    mapped_cache = PositionCache() if mapped_positions else None
    position_cache = PositionCache() if final_positions else None
    validated_singleton_indices = set()
    try:
        if tensor_cache is not None:
            for module in PATCHED_TORCH_MODULES:
                undo.append((module, "torch", module.torch))
                module.torch = tensor_cache
        if mapped_cache is not None:
            old_mapped = position_module.target_query_positions

            def mapped(context, cells, scale):
                return mapped_cache.get(
                    "adaptive_mapped_positions", old_mapped, context, cells, scale
                )

            undo.append((position_module, "target_query_positions", old_mapped))
            position_module.target_query_positions = mapped
        if position_cache is not None:
            old_target = adaptive_module.target_query_rope
            old_phase = global_module.phase_rope

            def target(context, cells, scale, dtype):
                return position_cache.get(
                    "adaptive",
                    position_module.target_query_rope,
                    context,
                    cells,
                    scale,
                    dtype,
                )

            def phase(context, cells, scale, dtype):
                return position_cache.get(
                    "global",
                    position_module.phase_rope,
                    context,
                    cells,
                    scale,
                    dtype,
                )

            undo.extend(
                [
                    (adaptive_module, "target_query_rope", old_target),
                    (global_module, "phase_rope", old_phase),
                ]
            )
            adaptive_module.target_query_rope = target
            global_module.phase_rope = phase
        if metadata:
            old_subset = position_module.ViewContext.subset

            def subset(context, indices):
                # Fixed B=1 fixture: avoid a positions copy, but fail closed for
                # every other bucket shape or index.
                candidate = (
                    context.positions.shape[0] == 1
                    and context.sigma.numel() == 1
                    and isinstance(indices, torch.Tensor)
                    and indices.numel() == 1
                )
                if candidate:
                    # The first no-grad warm call validates the value.  Cached
                    # metadata then reuses this exact tensor, avoiding a GPU
                    # scalar synchronization inside measured/replayed calls.
                    identity = id(indices)
                    if identity in validated_singleton_indices:
                        return context
                    if not torch.is_grad_enabled() and int(indices.item()) == 0:
                        validated_singleton_indices.add(identity)
                        return context
                return old_subset(context, indices)

            undo.append((position_module.ViewContext, "subset", old_subset))
            position_module.ViewContext.subset = subset
        yield tensor_cache, mapped_cache, position_cache
    finally:
        for obj, name, value in reversed(undo):
            setattr(obj, name, value)


def compare_tensor(actual: torch.Tensor, reference: torch.Tensor):
    """Bound temporary CPU memory while computing exactness and relative L2."""
    a = actual.detach().cpu().reshape(-1)
    r = reference.reshape(-1)
    if a.shape != r.shape or a.dtype != r.dtype:
        return {"shape_dtype_equal": False, "exact_equal": False}
    numerator = denominator = 0.0
    exact = True
    finite = True
    chunk = 1_000_000
    for start in range(0, a.numel(), chunk):
        aa = a[start : start + chunk]
        rr = r[start : start + chunk]
        exact = exact and torch.equal(aa, rr)
        finite = finite and bool(torch.isfinite(aa).all())
        delta = aa.float() - rr.float()
        numerator += float(delta.square().sum())
        denominator += float(rr.float().square().sum())
    return {
        "shape_dtype_equal": True,
        "exact_equal": exact,
        "finite": finite,
        "relative_l2": (numerator / max(denominator, 1e-30)) ** 0.5,
    }


def snapshot(block, x, y):
    return {
        "output": y.detach().cpu(),
        "input_grad": x.grad.detach().cpu(),
        "parameter_grads": {
            name: None if parameter.grad is None else parameter.grad.detach().cpu()
            for name, parameter in block.named_parameters()
        },
    }


def compare_snapshot(actual, reference):
    actual_names = set(actual["parameter_grads"])
    reference_names = set(reference["parameter_grads"])
    parameters = {}
    for name in sorted(actual_names | reference_names):
        a = actual["parameter_grads"].get(name)
        r = reference["parameter_grads"].get(name)
        parameters[name] = (
            {"both_none": True, "exact_equal": True, "finite": True, "relative_l2": 0.0}
            if a is None and r is None
            else {"both_none": False, "exact_equal": False}
            if a is None or r is None
            else compare_tensor(a, r)
        )
    output = compare_tensor(actual["output"], reference["output"])
    input_grad = compare_tensor(actual["input_grad"], reference["input_grad"])
    return {
        "output": output,
        "input_grad": input_grad,
        "parameter_name_sets_equal": actual_names == reference_names,
        "parameter_grads": parameters,
        "all_exact": (
            output["exact_equal"]
            and input_grad["exact_equal"]
            and actual_names == reference_names
            and all(row.get("exact_equal", False) for row in parameters.values())
        ),
        "all_finite": (
            output.get("finite", False)
            and input_grad.get("finite", False)
            and all(row.get("finite", False) for row in parameters.values())
        ),
    }


def run_once(trainer, model, block, template, x_source, vjp, checkpointed, directory):
    model.zero_grad(set_to_none=True)
    set_seed(trainer._config.seed)
    x = x_source.detach().clone().requires_grad_(True)
    args = replace(template, x=x)
    ins = Instrument(directory, profile=False)
    torch.cuda.reset_peak_memory_stats()
    with trainer._accelerator.autocast(), ins.stage("forward"):
        if checkpointed:
            y = checkpoint.checkpoint(block, args, None, use_reentrant=False)[0].x
        else:
            y = block(args, None)[0].x
    with ins.stage("backward_only"):
        torch.autograd.backward(y, vjp)
    torch.cuda.synchronize()
    result = snapshot(block, x, y)
    audit = {
        "phases": ins.phases,
        "memory": memory(),
        "output_dtype": str(y.dtype),
        "input_dtype": str(x.dtype),
    }
    ins.finish()
    del y, args, x
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return result, audit


def warm_forward_backward(trainer, model, block, template, x_source, vjp, checkpointed):
    """Untimed dgrad/wgrad warmup; no CPU snapshot or graph survives it."""
    model.zero_grad(set_to_none=True)
    set_seed(trainer._config.seed)
    x = x_source.detach().clone().requires_grad_(True)
    block_args = replace(template, x=x)
    with trainer._accelerator.autocast():
        if checkpointed:
            y = checkpoint.checkpoint(block, block_args, None, use_reentrant=False)[0].x
        else:
            y = block(block_args, None)[0].x
    torch.autograd.backward(y, vjp)
    torch.cuda.synchronize()
    del y, block_args, x
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This probe requires CUDA; CPU execution is intentionally unsupported"
        )
    if not torch.backends.cudnn.benchmark:
        raise RuntimeError("cuDNN benchmark must be enabled before model construction")

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
    anchor_cache = _build_anchor_cache(audit, cfg, op)
    dataset = OutpaintingCacheDataset(audit, anchor_cache, seed=cfg.seed)
    index = next(
        i
        for i, path in enumerate(dataset.files)
        if path.relative_to(audit.root / "latents").as_posix() == RECORD
    )
    sample = dataset[(index, 0)]
    if sample["_anchor_entries"] is None:
        raise RuntimeError("Fixture must use the existing real cache")
    batch = collate_outpainting([sample])
    trainer = OutpaintingTrainer(cfg, op, cache_audit=audit, anchor_cache=anchor_cache)
    trainer._init_timestep_sampler()
    set_seed(cfg.seed)
    model = trainer._transformer
    model.train()
    if len(model.transformer_blocks) != 48:
        raise RuntimeError(
            f"Expected all 48 actual blocks resident, got {len(model.transformer_blocks)}"
        )
    video = prepare(trainer, batch)
    with torch.no_grad(), trainer._accelerator.autocast():
        base = model.video_args_preprocessor.prepare(video, None)
    if tuple(base.x.shape) != EXPECTED_HIDDEN:
        raise RuntimeError(
            f"Expected real first-block hidden {EXPECTED_HIDDEN}, got {tuple(base.x.shape)}"
        )
    if base.x.dtype != torch.bfloat16:
        raise RuntimeError(f"Expected BF16 hidden, got {base.x.dtype}")

    perturb = BatchedPerturbationConfig.empty(1, 48, base.x.device, base.x.dtype)
    references = {}
    results = []
    configurations = (
        ("baseline", False, False, False, True),
        ("metadata_cache", True, False, False, True),
        ("metadata_and_adaptive_mapped_positions", True, True, False, True),
        ("metadata_and_final_pe_cache", True, False, True, True),
        ("checkpoint_off", False, False, False, False),
        ("baseline_again", False, False, False, True),
    )

    for block_index in (0, 1):
        block = model.transformer_blocks[block_index]
        x_source = base.x.detach()
        # Reuse this immutable source as the fixed BF16 VJP too, avoiding two
        # extra 0.56-GiB fixture copies.  Every measured leaf is still fresh.
        vjp = x_source
        template = model.block_input_processor(
            replace(base, x=x_source),
            perturb,
            block_index,
            self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
            cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
        )
        for (
            name,
            metadata,
            mapped_positions,
            final_positions,
            checkpointed,
        ) in configurations:
            config_dir = (
                root / f"k{block.attn1.partition_count}_block{block_index}" / name
            )
            config_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "block_index": block_index,
                "partition_count": block.attn1.partition_count,
                "configuration": name,
                "checkpoint_nonreentrant": checkpointed,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "hidden_shape": list(x_source.shape),
                "repeats": [],
            }
            try:
                with cache_scope(metadata, mapped_positions, final_positions) as (
                    tensor_cache,
                    mapped_cache,
                    position_cache,
                ):
                    # Untimed no-grad execution both warms cuDNN and eagerly
                    # materializes every cache entry before checkpoint replay.
                    warm_args = replace(template, x=x_source)
                    torch.cuda.synchronize()
                    warm_begin = time.perf_counter()
                    with torch.no_grad(), trainer._accelerator.autocast():
                        warm_y = block(warm_args, None)[0].x
                    torch.cuda.synchronize()
                    row["warm_and_build_seconds"] = time.perf_counter() - warm_begin
                    del warm_y, warm_args
                    if tensor_cache is not None:
                        tensor_cache.building = False
                    if mapped_cache is not None:
                        mapped_cache.building = False
                    if position_cache is not None:
                        position_cache.building = False

                    backward_warm_begin = time.perf_counter()
                    warm_forward_backward(
                        trainer,
                        model,
                        block,
                        template,
                        x_source,
                        vjp,
                        checkpointed,
                    )
                    row["forward_backward_warm_seconds"] = (
                        time.perf_counter() - backward_warm_begin
                    )

                    config_reference = None
                    for repetition in range(args.repeats):
                        snap, measured = run_once(
                            trainer,
                            model,
                            block,
                            template,
                            x_source,
                            vjp,
                            checkpointed,
                            config_dir / f"repeat_{repetition + 1:02d}",
                        )
                        if name == "baseline" and repetition == 0:
                            references[block_index] = snap
                            comparison = compare_snapshot(snap, snap)
                        else:
                            comparison = compare_snapshot(snap, references[block_index])
                        measured["comparison_to_baseline"] = comparison
                        if config_reference is None:
                            config_reference = snap
                            repeatability = compare_snapshot(snap, snap)
                        else:
                            repeatability = compare_snapshot(snap, config_reference)
                        measured["comparison_to_first_same_configuration"] = (
                            repeatability
                        )
                        row["repeats"].append(measured)
                        if snap is not config_reference:
                            del snap
                    row["same_configuration_repeatability_pass"] = all(
                        repeat["comparison_to_first_same_configuration"]["all_exact"]
                        and repeat["comparison_to_first_same_configuration"][
                            "all_finite"
                        ]
                        for repeat in row["repeats"]
                    )
                    if config_reference is not references.get(block_index):
                        del config_reference
                    row["metadata_cache"] = (
                        None if tensor_cache is None else tensor_cache.summary()
                    )
                    row["adaptive_mapped_position_cache"] = (
                        None if mapped_cache is None else mapped_cache.summary()
                    )
                    row["position_cache"] = (
                        None if position_cache is None else position_cache.summary()
                    )
                    cached = list(tensor_cache.cache.values()) if tensor_cache else []
                    cached += list(mapped_cache.cache.values()) if mapped_cache else []
                    cached += (
                        list(position_cache.cache.values()) if position_cache else []
                    )
                    row["cached_tensors_autograd_safe"] = all(
                        not t.requires_grad and t.grad_fn is None
                        for value in cached
                        for t in _tensors(value)
                    )
                    protected = [template, x_source, vjp]
                    protected.extend(block.parameters())
                    row["cached_tensors_do_not_alias_inputs_or_parameters"] = not (
                        _storage_keys(cached) & _storage_keys(protected)
                    )
                    del cached, protected
                    for scoped_cache in (tensor_cache, mapped_cache, position_cache):
                        if scoped_cache is not None:
                            scoped_cache.cache.clear()
                row["equivalence_pass"] = all(
                    repeat["comparison_to_baseline"]["all_exact"]
                    and repeat["comparison_to_baseline"]["all_finite"]
                    for repeat in row["repeats"]
                )
                row["cache_safety_pass"] = (
                    row["cached_tensors_autograd_safe"]
                    and row["cached_tensors_do_not_alias_inputs_or_parameters"]
                )
                row["status"] = (
                    "PASS"
                    if row["equivalence_pass"] and row["cache_safety_pass"]
                    else "MISMATCH"
                )
            except Exception as exc:
                import traceback

                row.update(
                    status="FAIL", error=repr(exc), traceback=traceback.format_exc()
                )
                save(config_dir / "result.json", row)
                results.append(row)
                save(root / "summary.json", results)
                raise
            save(config_dir / "result.json", row)
            results.append(row)
            save(root / "summary.json", results)
            print("PROBE", block_index, name, row["status"], flush=True)
        del template, x_source, vjp, references[block_index]
        gc.collect()
        torch.cuda.empty_cache()

    if anchor_cache.created_count != 0:
        raise RuntimeError("Probe unexpectedly generated cache entries")
    save(
        root / "manifest.json",
        {
            "status": (
                "PASS"
                if all(row["status"] == "PASS" for row in results)
                else "MISMATCH"
            ),
            "record": RECORD,
            "hidden_shape": list(EXPECTED_HIDDEN),
            "actual_model_blocks_resident": len(model.transformer_blocks),
            "notes": [
                "Actual isolated blocks only; results do not claim model-wide speedup.",
                "Cache build and cuDNN warm time is separate from measured forward/backward.",
                "Position cache is scoped to one fixed model invocation and exact positions identity.",
                "Mapped-position and final-PE caches are nested alternatives; their timings must not be added.",
                "Native positional embeddings are intentionally not cached.",
            ],
        },
    )


if __name__ == "__main__":
    main()
