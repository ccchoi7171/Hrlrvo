"""Low-overhead optimizer-step health metrics for Hrlrvo training.

This module observes the existing model.  It neither changes the forward result
nor retains activation graphs.  Gradient scalar reductions are copied to the
host once per device, rather than calling ``item()`` for every parameter.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

GRADIENT_GROUPS = (
    "op_lora",
    "a_lora",
    "g_lora",
    "f_lora",
    "c2",
    "c4_shared",
    "r2",
    "r4",
    "local_fusion",
    "global_fusion",
)

_BLOCK_FUSION = re.compile(
    r"(?:^|\.)transformer_blocks\.(\d+)\.attn1\.(local_fusion|global_fusion)$"
)


def parameter_health_group(name: str) -> str | None:
    """Map an installed-v8 parameter name to its optimizer health group."""
    if ".ffn_lora." in name:
        return "f_lora"
    if ".shared_outpainting_lora." in name:
        return "op_lora"
    if ".adaptive_lora." in name:
        return "a_lora"
    if ".global_lora." in name:
        return "g_lora"
    if ".compressors.2." in name:
        return "c2"
    if ".compressors.4." in name:
        return "c4_shared"
    if ".decompressors.2." in name:
        return "r2"
    if ".decompressors.4." in name:
        return "r4"
    if ".local_fusion." in name:
        return "local_fusion"
    if ".global_fusion." in name:
        return "global_fusion"
    return None


class _JSONLSink:
    """Persistent buffered JSONL writer whose failures never stop training."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stream = None
        self._warned = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a", encoding="utf-8", buffering=256 * 1024)
        except Exception as error:  # noqa: BLE001 - logging must not abort training
            self._warn_once(error)

    def _warn_once(self, error: BaseException) -> None:
        if not self._warned:
            logger.warning("Training health logging disabled after an error: %s", error)
            self._warned = True

    def append(self, record: dict[str, Any]) -> bool:
        if self._stream is None:
            return False
        try:
            self._stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            # One flush per optimizer step keeps tail -f useful without reopening
            # the file or writing per-parameter records.
            self._stream.flush()
            return True
        except Exception as error:  # noqa: BLE001 - logging must not abort training
            self._warn_once(error)
            return False

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.close()
        except Exception as error:  # noqa: BLE001 - logging must not abort training
            self._warn_once(error)
        finally:
            self._stream = None


class _FusionGateTracker:
    """Aggregate actual sigmoid outputs without retaining their autograd graph."""

    def __init__(self, model: nn.Module, expected_blocks: int):
        self.expected_blocks = expected_blocks
        self._handles: list[Any] = []
        self._active = False
        self._warned = False
        self._errors: list[str] = []
        self._stats: dict[str, dict[int, dict[str, Any]]] = {}
        self._local_rectangles: dict[int, tuple[tuple[tuple[int, int, int, int, int], ...], ...]] = {}
        self._local_call_index: dict[int, int] = {}
        modules = dict(model.named_modules())
        self._attention_geometry = {}
        for name, module in modules.items():
            match = _BLOCK_FUSION.search(name)
            if match is None:
                continue
            block_index, kind = int(match.group(1)), match.group(2)
            attention = modules[name.rsplit(".", 1)[0]]
            self._attention_geometry[block_index] = (
                int(attention.partition_count), int(attention.scale)
            )
            handle = module.register_forward_hook(self._make_hook(kind, block_index))
            self._handles.append(handle)

    def _record_error(self, error: BaseException) -> None:
        self._errors.append(str(error))
        if not self._warned:
            logger.warning("Fusion health observer failed; model forward will continue: %s", error)
            self._warned = True

    @staticmethod
    def _rectangles(indices: tuple[int, ...], width: int) -> tuple[tuple[int, int, int, int], ...]:
        """Merge exact flat-mask indices into non-copying rectangular slices."""
        by_row: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            y, x = divmod(index, width)
            by_row[y].append(x)
        spans_by_row: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for y, xs in sorted(by_row.items()):
            start = previous = xs[0]
            for x in xs[1:]:
                if x != previous + 1:
                    spans_by_row[y].append((start, previous + 1))
                    start = x
                previous = x
            spans_by_row[y].append((start, previous + 1))
        rectangles: list[list[int]] = []
        active: dict[tuple[int, int], int] = {}
        for y, spans in sorted(spans_by_row.items()):
            next_active = {}
            for x0, x1 in spans:
                key = (x0, x1)
                position = active.get(key)
                if position is not None and rectangles[position][2] == y:
                    rectangles[position][2] = y + 1
                else:
                    rectangles.append([y, x0, y + 1, x1])
                    position = len(rectangles) - 1
                next_active[key] = position
            active = next_active
        return tuple(tuple(rectangle) for rectangle in rectangles)

    def set_geometry(self, geometry: Any, batch_size: int | None = None) -> None:
        """Precompute exact resolved-Target slices for Local gate statistics.

        The slices derive from the same cached ``adaptive_layout`` helper used by
        production forward.  They avoid expanding a mask across T and D or
        copying the very large fusion gate tensor.
        """
        from .geometry import adaptive_layout, local_windows

        size = len(geometry.anchors) if batch_size is None else int(batch_size)
        anchors = geometry.batch_anchors(size)
        prepared = {}
        for block, (partition_count, scale) in self._attention_geometry.items():
            calls = []
            for window in local_windows(geometry.height, geometry.width, partition_count):
                rectangles = []
                for batch, anchor in enumerate(anchors):
                    layout = adaptive_layout(anchor, window, geometry.height, geometry.width, scale)
                    for y0, x0, y1, x1 in self._rectangles(layout.write_target_indices, window.width):
                        rectangles.append((batch, y0, x0, y1, x1))
                calls.append(tuple(rectangles))
            prepared[block] = tuple(calls)
        self._local_rectangles = prepared

    @staticmethod
    def _reduce_views(views: list[Tensor]) -> dict[str, Any] | None:
        views = [view for view in views if view.numel()]
        if not views:
            return None
        sums, minima, maxima, nan_flags, inf_flags = [], [], [], [], []
        count = 0
        for view in views:
            # Gate output is FP32 in production. Reductions stream over its
            # existing storage; no output-sized float/safe/min/max copy exists.
            total = view.sum(dtype=torch.float32)
            minimum, maximum = torch.aminmax(view)
            sums.append(total)
            minima.append(minimum)
            maxima.append(maximum)
            # Scalar reductions propagate non-finite alpha, avoiding even a
            # temporary output-shaped bool tensor on the 4K Global gate.
            nan_flags.append(torch.isnan(total) | torch.isnan(minimum) | torch.isnan(maximum))
            inf_flags.append(torch.isinf(total) | torch.isinf(minimum) | torch.isinf(maximum))
            count += view.numel()
        return {
            "sum": torch.stack(sums).sum(),
            "count": count,
            "min": torch.stack(minima).amin(),
            "max": torch.stack(maxima).amax(),
            "nan": torch.stack(nan_flags).any(),
            "inf": torch.stack(inf_flags).any(),
            "calls": 1,
        }

    def _make_hook(self, kind: str, block_index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not self._active or not isinstance(output, Tensor):
                return
            try:
                alpha = output.detach()
                if kind == "local_fusion":
                    call_index = self._local_call_index.get(block_index, 0)
                    self._local_call_index[block_index] = call_index + 1
                    calls = self._local_rectangles.get(block_index)
                    if calls is None:
                        raise RuntimeError("Set CanvasGeometry before forward to measure Target-only Local fusion")
                    if call_index >= len(calls):
                        raise RuntimeError(f"Unexpected extra Local fusion call in block {block_index}")
                    views = [
                        alpha[batch, :, y0:y1, x0:x1, :]
                        for batch, y0, x0, y1, x1 in calls[call_index]
                    ]
                else:
                    views = [alpha]
                update = self._reduce_views(views)
                if update is None:
                    return
                stats = self._stats.setdefault(kind, {}).get(block_index)
                if stats is None:
                    self._stats[kind][block_index] = update
                    return
                stats["sum"] = stats["sum"] + update["sum"]
                stats["count"] += update["count"]
                stats["min"] = torch.minimum(stats["min"], update["min"])
                stats["max"] = torch.maximum(stats["max"], update["max"])
                stats["nan"] = stats["nan"] | update["nan"]
                stats["inf"] = stats["inf"] | update["inf"]
                stats["calls"] += 1
            except Exception as error:  # noqa: BLE001 - observer must not break forward
                self._record_error(error)

        return hook

    def begin_step(self) -> None:
        self._stats = {}
        self._errors = []
        self._local_call_index = {}
        self._active = True

    def begin_forward(self) -> None:
        """Resume collection for another microbatch in the same optimizer step."""
        self._local_call_index = {}
        self._active = True

    def end_forward(self) -> None:
        # Disable before backward so checkpoint recomputation cannot double-count
        # the same logical forward.  Call begin_forward for each accumulation batch.
        self._active = False

    @staticmethod
    def _copy_rows(rows: list[tuple[str, int, dict[str, Any]]]) -> list[list[float]]:
        if not rows:
            return []
        by_device: dict[torch.device, list[tuple[int, list[Tensor]]]] = defaultdict(list)
        for row_index, (_, _, values) in enumerate(rows):
            tensors = [values[key].reshape(()) for key in ("sum", "min", "max", "nan", "inf")]
            by_device[tensors[0].device].append((row_index, tensors))
        result: list[list[float] | None] = [None] * len(rows)
        for indexed in by_device.values():
            packed = torch.stack([torch.stack(tensors) for _, tensors in indexed]).detach().cpu().tolist()
            for (row_index, _), values in zip(indexed, packed, strict=True):
                result[row_index] = values
        return [value for value in result if value is not None]

    def metrics(self) -> dict[str, Any]:
        self._active = False
        rows = [
            (kind, block, values)
            for kind in ("local_fusion", "global_fusion")
            for block, values in sorted(self._stats.get(kind, {}).items())
        ]
        host_rows = self._copy_rows(rows)
        decoded: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
        for (kind, block, values), host in zip(rows, host_rows, strict=True):
            total, minimum, maximum, nan_count, inf_count = host
            count = values["count"]
            decoded[kind][block] = {
                "mean": total / count if count else math.nan,
                "min": minimum if count else math.nan,
                "max": maximum if count else math.nan,
                "nan": bool(nan_count),
                "inf": bool(inf_count),
                "calls": int(values["calls"]),
            }

        result: dict[str, Any] = {}
        for kind, primary, complement in (
            ("local_fusion", "local_native_ratio", "local_adaptive_ratio"),
            ("global_fusion", "global_local_ratio", "global_global_ratio"),
        ):
            block_stats = decoded.get(kind, {})
            by_block = [block_stats[index]["mean"] if index in block_stats else None
                        for index in range(self.expected_blocks)]
            observed = [value for value in by_block if value is not None]
            result[f"{primary}_by_block"] = by_block
            result[f"{complement}_by_block"] = [None if value is None else 1.0 - value for value in by_block]
            result[f"{primary}_mean"] = sum(observed) / len(observed) if observed else None
            result[f"{primary}_min"] = min((item["min"] for item in block_stats.values()), default=None)
            result[f"{primary}_max"] = max((item["max"] for item in block_stats.values()), default=None)
            complement_values = [1.0 - value for value in observed]
            result[f"{complement}_mean"] = (
                sum(complement_values) / len(complement_values) if complement_values else None
            )
            result[f"{complement}_min"] = (
                1.0 - result[f"{primary}_max"] if result[f"{primary}_max"] is not None else None
            )
            result[f"{complement}_max"] = (
                1.0 - result[f"{primary}_min"] if result[f"{primary}_min"] is not None else None
            )
            result[f"{kind}_observed_blocks"] = sorted(block_stats)
            result[f"{kind}_missing_blocks"] = sorted(set(range(self.expected_blocks)) - set(block_stats))
            result[f"{kind}_forward_calls_by_block"] = [
                block_stats[index]["calls"] if index in block_stats else 0
                for index in range(self.expected_blocks)
            ]
            result[f"{kind}_nonfinite_detected"] = any(
                item["nan"] or item["inf"] for item in block_stats.values()
            )
        result["local_fusion_target_mask_configured"] = bool(self._local_rectangles)
        result["fusion_observer_errors"] = list(self._errors)
        return result

    def close(self) -> None:
        self._active = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _gradient_values(gradient: Tensor) -> Tensor:
    if gradient.is_sparse:
        return gradient.coalesce().values()
    return gradient


class TrainingHealthLogger:
    """Collect optimizer-step gradients, fusion alpha, and append JSONL records."""

    def __init__(
        self,
        model: nn.Module,
        metrics_path: str | Path,
        run_id: str,
        *,
        expected_blocks: int = 48,
    ):
        self.model = model
        self.run_id = str(run_id)
        self._sink = _JSONLSink(metrics_path)
        self._warned = False
        self._fusion = _FusionGateTracker(model, expected_blocks)
        groups: dict[str, list[tuple[str, nn.Parameter]]] = {name: [] for name in GRADIENT_GROUPS}
        self.unclassified_trainable_parameters: list[str] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = parameter_health_group(name)
            if group is None:
                self.unclassified_trainable_parameters.append(name)
            else:
                groups[group].append((name, parameter))
        self.grouped_named_parameters = {key: tuple(value) for key, value in groups.items()}

    def _warn_once(self, error: BaseException) -> None:
        if not self._warned:
            logger.warning("Training health collection failed; training will continue: %s", error)
            self._warned = True

    def begin_optimizer_step(self) -> None:
        try:
            self._fusion.begin_step()
        except Exception as error:  # noqa: BLE001 - instrumentation is fail-safe
            self._warn_once(error)

    def set_geometry(self, geometry: Any, batch_size: int | None = None) -> bool:
        """Set the production CanvasGeometry used for exact Target-only ratios."""
        try:
            self._fusion.set_geometry(geometry, batch_size)
            return True
        except Exception as error:  # noqa: BLE001 - instrumentation is fail-safe
            self._warn_once(error)
            return False

    def begin_forward(self) -> None:
        try:
            self._fusion.begin_forward()
        except Exception as error:  # noqa: BLE001 - instrumentation is fail-safe
            self._warn_once(error)

    def end_forward(self) -> None:
        try:
            self._fusion.end_forward()
        except Exception as error:  # noqa: BLE001 - instrumentation is fail-safe
            self._warn_once(error)

    def _gradient_metrics(self, groups: Iterable[str] | None = None) -> dict[str, Any]:
        selected = tuple(groups or GRADIENT_GROUPS)
        rows: list[tuple[str, torch.device, Tensor, Tensor, Tensor]] = []
        details: dict[str, dict[str, Any]] = {}
        for group in selected:
            named = self.grouped_named_parameters[group]
            details[group] = {
                "parameter_count": sum(parameter.numel() for _, parameter in named),
                "parameter_tensors": len(named),
                "parameters_with_grad": 0,
                "parameters_without_grad": 0,
            }
            per_device: dict[torch.device, list[Tensor]] = defaultdict(list)
            for _, parameter in named:
                if parameter.grad is None:
                    details[group]["parameters_without_grad"] += 1
                    continue
                details[group]["parameters_with_grad"] += 1
                per_device[parameter.grad.device].append(_gradient_values(parameter.grad.detach()))
            for device, gradients in per_device.items():
                squared = []
                nan_counts = []
                inf_counts = []
                for gradient in gradients:
                    value = gradient.float()
                    finite = torch.isfinite(value)
                    squared.append(torch.where(finite, value, torch.zeros_like(value)).square().sum())
                    nan_counts.append(torch.isnan(value).sum())
                    inf_counts.append(torch.isinf(value).sum())
                rows.append((
                    group,
                    device,
                    torch.stack(squared).sum(),
                    torch.stack(nan_counts).sum(),
                    torch.stack(inf_counts).sum(),
                ))

        by_device: dict[torch.device, list[tuple[int, tuple[Tensor, Tensor, Tensor]]]] = defaultdict(list)
        for row_index, (_, device, squared, nan_count, inf_count) in enumerate(rows):
            by_device[device].append((row_index, (squared, nan_count, inf_count)))
        host_rows: list[list[float] | None] = [None] * len(rows)
        for indexed in by_device.values():
            packed = torch.stack([
                torch.stack(tensors) for _, tensors in indexed
            ]).detach().cpu().tolist()
            for (row_index, _), values in zip(indexed, packed, strict=True):
                host_rows[row_index] = values

        group_sq = {group: 0.0 for group in selected}
        group_nan = {group: 0 for group in selected}
        group_inf = {group: 0 for group in selected}
        for (group, _, _, _, _), host in zip(rows, host_rows, strict=True):
            assert host is not None
            group_sq[group] += float(host[0])
            group_nan[group] += int(host[1])
            group_inf[group] += int(host[2])

        for group in selected:
            norm = math.sqrt(group_sq[group])
            details[group].update({
                "grad_norm": norm,
                "nan_gradient_count": group_nan[group],
                "inf_gradient_count": group_inf[group],
                "finite": group_nan[group] == 0 and group_inf[group] == 0,
                "nonzero": norm > 0.0,
            })

        frozen_violations = sum(
            1 for parameter in self.model.parameters()
            if not parameter.requires_grad and parameter.grad is not None
        )
        total_sq = sum(group_sq.values())
        result: dict[str, Any] = {
            "grad_norm_global": math.sqrt(total_sq),
            "grad_norm_before_clip": math.sqrt(total_sq),
            "nan_gradient_count": sum(group_nan.values()),
            "inf_gradient_count": sum(group_inf.values()),
            "zero_gradient_trainable_groups": [
                group for group in selected if not details[group]["nonzero"]
            ],
            "finite_gradient_trainable_groups": [
                group for group in selected if details[group]["finite"]
            ],
            "nonzero_gradient_trainable_groups": [
                group for group in selected if details[group]["nonzero"]
            ],
            "frozen_parameter_grad_violation_count": frozen_violations,
            "gradient_group_details": details,
            "unclassified_trainable_parameter_count": len(self.unclassified_trainable_parameters),
        }
        for group in selected:
            result[f"grad_norm_{group}"] = details[group]["grad_norm"]
        return result

    def capture_pre_clip_gradients(self) -> dict[str, Any]:
        try:
            return self._gradient_metrics()
        except Exception as error:  # noqa: BLE001 - instrumentation is fail-safe
            self._warn_once(error)
            return {"health_logging_error": str(error)}

    def capture_post_clip_global_norm(self) -> float | None:
        try:
            rows: dict[torch.device, list[Tensor]] = defaultdict(list)
            for named in self.grouped_named_parameters.values():
                for _, parameter in named:
                    if parameter.grad is not None:
                        gradient = _gradient_values(parameter.grad.detach()).float()
                        rows[gradient.device].append(gradient.square().sum())
            total = 0.0
            for values in rows.values():
                total += float(torch.stack(values).sum().detach().cpu())
            return math.sqrt(total)
        except Exception as error:  # noqa: BLE001 - instrumentation is fail-safe
            self._warn_once(error)
            return None

    def fusion_metrics(self) -> dict[str, Any]:
        try:
            return self._fusion.metrics()
        except Exception as error:  # noqa: BLE001 - instrumentation is fail-safe
            self._warn_once(error)
            return {"fusion_health_logging_error": str(error)}

    def snapshot_representatives(self) -> dict[str, list[dict[str, Any]]]:
        """Keep one max-gradient scalar per parameter, never a full tensor copy."""
        snapshots = {}
        for group, named in self.grouped_named_parameters.items():
            if not named:
                continue
            selected = []
            for name, parameter in named:
                if parameter.grad is None:
                    continue
                index = _gradient_values(parameter.grad.detach()).abs().reshape(-1).argmax()
                selected.append({
                    "name": name,
                    "index": index,
                    "value": parameter.detach().reshape(-1).index_select(0, index.reshape(1)).squeeze(0).clone(),
                })
            if not selected:
                name, parameter = named[0]
                index = torch.zeros((), device=parameter.device, dtype=torch.long)
                selected.append({"name": name, "index": index,
                                 "value": parameter.detach().reshape(-1).index_select(0, index.reshape(1)).squeeze(0).clone()})
            snapshots[group] = selected
        return snapshots

    def representative_updates(self, before: dict[str, list[dict[str, Any]]]) -> dict[str, bool]:
        """Compare representative tensors with one synchronization per device."""
        rows: list[tuple[str, Tensor]] = []
        named_parameters = dict(self.model.named_parameters())
        for group, snapshots in before.items():
            changed = []
            for snapshot in snapshots:
                parameter = named_parameters.get(snapshot["name"])
                if parameter is None:
                    continue
                index = snapshot["index"].to(parameter.device)
                current = parameter.detach().reshape(-1).index_select(0, index.reshape(1)).squeeze(0)
                changed.append(torch.ne(current, snapshot["value"].to(current.device)))
            if changed:
                rows.append((group, torch.stack(changed).any()))
        by_device: dict[torch.device, list[tuple[str, Tensor]]] = defaultdict(list)
        for group, changed in rows:
            by_device[changed.device].append((group, changed))
        result = {}
        for values in by_device.values():
            host = torch.stack([changed for _, changed in values]).cpu().tolist()
            result.update({group: bool(changed) for (group, _), changed in zip(values, host, strict=True)})
        return result

    def log_step(self, record: dict[str, Any]) -> bool:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            **record,
        }
        return self._sink.append(payload)

    def close(self) -> None:
        self._fusion.close()
        self._sink.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
