"""Cached LTX latent input pipeline for 65-frame outpainting clips.

Target latents remain the official LTX ``process_videos`` products. Workers
only read target latents and already materialized lazy-Anchor entries; cache
miss encoding is deferred to the owning training rank outside DataLoader workers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from ltx_core.model.video_vae.video_vae import map_temporal_slice
from ltx_core.types import SpatioTemporalScaleFactors
from torch.utils.data import Dataset, Sampler, default_collate

EXPECTED_CACHE_FORMAT = "LTX-2.3 process_videos.compute_latents / PrecomputedDataset"


@dataclass(frozen=True)
class CacheAudit:
    root: Path
    latent_files: tuple[Path, ...]
    compatible_files: tuple[Path, ...]
    checkpoint_sha256: str
    latent_dtype: str
    fps: float
    latent_height: int
    latent_width: int
    clip_latent_frames: int
    source_clip_frames: int
    temporal_scale: int
    has_text_cache: bool
    order_fingerprint: str
    contract: dict[str, Any]
    records: dict[str, dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def audit_ltx_cache(
    root: str | Path,
    *,
    source_clip_frames: int = 65,
    scale_factors: SpatioTemporalScaleFactors | None = None,
    expected_fps: float = 24.0,
    requested_canvas: tuple[int, int] | None = None,
) -> CacheAudit:
    """Validate the existing cache without modifying or re-caching it."""
    root = Path(root).expanduser().resolve()
    scale_factors = scale_factors or SpatioTemporalScaleFactors.default()
    contract_path = root / "cache_contract.json"
    metadata_path = root / "cache_metadata.jsonl"
    latent_root = root / "latents"
    if not contract_path.is_file() or not metadata_path.is_file() or not latent_root.is_dir():
        raise ValueError(f"Not an audited LTX training cache: {root}")

    contract = _read_json(contract_path)
    if contract.get("format") != EXPECTED_CACHE_FORMAT:
        raise ValueError(f"Unsupported cache format: {contract.get('format')!r}")
    if contract.get("latent_dtype") != "torch.bfloat16":
        raise ValueError(f"Expected bfloat16 cached latents, got {contract.get('latent_dtype')!r}")
    cache_resolution = contract.get("cache_resolution")
    if not isinstance(cache_resolution, list) or len(cache_resolution) != 2:
        raise ValueError(f"Invalid cache resolution contract: {cache_resolution!r}")
    cache_width, cache_height = map(int, cache_resolution)
    if requested_canvas is not None:
        requested_height, requested_width = requested_canvas
        padding = contract.get("padding", {})
        if cache_width != requested_width or cache_height != requested_height + int(padding.get("top", -1)) + int(padding.get("bottom", -1)):
            raise ValueError("Cache canvas/padding does not match the requested training canvas")
        if padding.get("padding_value_normalized") != 0.0:
            raise ValueError("Cache padding must be normalized zero")
    temporal_scale = scale_factors.time
    if source_clip_frames % temporal_scale != 1:
        raise ValueError("Clip frames must satisfy frames % temporal_scale == 1")
    clip_latent_frames = 1 + (source_clip_frames - 1) // temporal_scale

    latent_files = tuple(sorted(latent_root.rglob("*.pt")))
    if len(latent_files) != int(contract.get("num_records", -1)):
        raise ValueError(
            f"Cache record mismatch: contract={contract.get('num_records')}, files={len(latent_files)}"
        )

    metadata_names: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    compatible_names: set[str] = set()
    fps_values: set[float] = set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            try:
                cached_frames = int(record["cached_num_frames"])
                latent_frames = int(record["latent_num_frames"])
                fps = float(record["fps"])
                relative_path = Path(record["relative_path"]).with_suffix(".pt")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid cache metadata at line {line_number}") from exc
            expected_latent_frames = 1 + (cached_frames - 1) // temporal_scale
            if latent_frames != expected_latent_frames:
                raise ValueError(
                    f"Temporal metadata mismatch for {relative_path}: "
                    f"{latent_frames} != {expected_latent_frames}"
                )
            fps_values.add(fps)
            relative_name = str(relative_path)
            if relative_name in metadata_names:
                raise ValueError(f"Duplicate cache metadata ID: {relative_name}")
            metadata_names.add(relative_name)
            records[relative_name] = record
            if int(record.get("original_num_frames", 0)) >= source_clip_frames and cached_frames >= source_clip_frames:
                compatible_names.add(relative_name)

    if fps_values != {expected_fps}:
        raise ValueError(f"Expected only {expected_fps} fps cache records, got {sorted(fps_values)}")
    file_names = {str(path.relative_to(latent_root)) for path in latent_files}
    if file_names != metadata_names:
        raise ValueError(
            f"Cache metadata/file IDs differ: missing_metadata={len(file_names - metadata_names)}, "
            f"missing_files={len(metadata_names - file_names)}"
        )
    compatible_files = tuple(path for path in latent_files if str(path.relative_to(latent_root)) in compatible_names)
    if not compatible_files:
        raise ValueError(f"No cached sequence can provide a {source_clip_frames}-frame aligned crop")
    order_fingerprint = sha256(
        "\n".join(str(path.relative_to(latent_root)) for path in compatible_files).encode()
    ).hexdigest()

    # Probe one official record.  Full scanning would read almost 1 TB and is not
    # needed because the creation contract and per-record metadata are exhaustive.
    probe = torch.load(compatible_files[0], map_location="cpu", weights_only=True)
    required = {"latents", "num_frames", "height", "width", "fps"}
    if not isinstance(probe, dict) or set(probe) != required:
        raise ValueError(f"Cache record does not have the official fields: {compatible_files[0]}")
    latent = probe["latents"]
    if latent.ndim != 4 or latent.dtype != torch.bfloat16:
        raise ValueError(f"Invalid cached latent tensor: shape={tuple(latent.shape)}, dtype={latent.dtype}")
    if int(probe["num_frames"]) != latent.shape[1] or int(probe["height"]) != latent.shape[2] or int(probe["width"]) != latent.shape[3]:
        raise ValueError("Probe tensor dimensions differ from its official metadata fields")
    if latent.shape[2] * scale_factors.height != cache_height or latent.shape[3] * scale_factors.width != cache_width:
        raise ValueError("Cached spatial latent shape is incompatible with checkpoint VAE factors")
    if float(probe["fps"]) != expected_fps:
        raise ValueError(f"Probe FPS mismatch: {probe['fps']}")

    return CacheAudit(
        root=root,
        latent_files=latent_files,
        compatible_files=compatible_files,
        checkpoint_sha256=str(contract["checkpoint_sha256"]),
        latent_dtype=str(contract["latent_dtype"]),
        fps=expected_fps,
        latent_height=int(latent.shape[-2]),
        latent_width=int(latent.shape[-1]),
        clip_latent_frames=clip_latent_frames,
        source_clip_frames=source_clip_frames,
        temporal_scale=temporal_scale,
        has_text_cache=(root / "conditions").is_dir(),
        order_fingerprint=order_fingerprint,
        contract=contract,
        records=records,
    )


class OutpaintingCacheDataset(Dataset):
    """Aligned target clips plus lazy isolated-Anchor requests."""

    def __init__(
        self,
        audit: CacheAudit,
        anchor_cache: Any,
        *,
        conditions_dir: str | Path | None = None,
        seed: int = 42,
        temporal_sampling: str = "first",
    ) -> None:
        if temporal_sampling != "first":
            raise ValueError("OutpaintingCacheDataset uses the FIRST-only Anchor cache contract")
        self.audit = audit
        self.target_root = audit.root / "latents"
        self.anchor_cache = anchor_cache
        self.conditions_root = Path(conditions_dir).expanduser().resolve() if conditions_dir else None
        self.files = audit.compatible_files
        self.seed = seed
        self.temporal_sampling = temporal_sampling

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _load_record(path: Path) -> dict[str, Any]:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict) or "latents" not in value:
            raise ValueError(f"Invalid GT latent record: {path}")
        return value

    def __getitem__(self, item: tuple[int, int]) -> dict[str, Any]:
        index, _epoch = item
        target_path = self.files[index]
        relative = target_path.relative_to(self.target_root)
        record_id = str(relative)
        target = self._load_record(target_path)
        target_latents = target["latents"]
        record = self.audit.records[record_id]
        if (
            target_latents.ndim != 4
            or int(target["num_frames"]) != target_latents.shape[1]
            or int(target["height"]) != target_latents.shape[2]
            or int(target["width"]) != target_latents.shape[3]
            or int(record["latent_num_frames"]) != target_latents.shape[1]
        ):
            raise ValueError(f"Target runtime tensor/cache metadata mismatch for {relative}")
        if float(target.get("fps", self.audit.fps)) != self.audit.fps or float(record["fps"]) != self.audit.fps:
            raise ValueError(f"Target FPS mismatch for {relative}")
        if target_latents.dtype != torch.bfloat16:
            raise ValueError(f"Target must retain the official bfloat16 cache dtype for {relative}")
        total_latent_frames = target_latents.shape[1]
        max_start = total_latent_frames - self.audit.clip_latent_frames
        if max_start < 0:
            raise ValueError(f"GT record is shorter than the requested aligned clip: {relative}")
        # Use the official causal mapping below rather than deriving RGB
        # indices independently from a duplicated temporal-scale constant.
        start = 0
        stop = start + self.audit.clip_latent_frames
        rgb_slice, _ = map_temporal_slice(start, stop, 0, 0, self.audit.temporal_scale)
        if self.temporal_sampling == "first" and (
            start != 0
            or stop != self.audit.clip_latent_frames
            or rgb_slice.start != 0
            or rgb_slice.stop != self.audit.source_clip_frames
        ):
            raise RuntimeError(
                "FIRST temporal contract requires RGB [0:65) and the corresponding latent [0:9)"
            )

        def crop(record: dict[str, Any], tensor: torch.Tensor) -> dict[str, Any]:
            return {
                "latents": tensor[:, start:stop].contiguous(),
                "num_frames": self.audit.clip_latent_frames,
                "height": int(record["height"]),
                "width": int(record["width"]),
                "fps": float(record.get("fps", self.audit.fps)),
                "rgb_start_frame": rgb_slice.start,
                "latent_start_index": start,
                "gt_record_id": record_id,
            }

        request = self.anchor_cache.make_request(record_id, start)
        result: dict[str, Any] = {
            "video_latents": crop(target, target_latents),
            "_anchor_requests": request,
            "_anchor_entries": self.anchor_cache.read_cached(request),
        }
        if self.conditions_root is not None:
            condition_path = self.conditions_root / relative
            if not condition_path.is_file():
                raise FileNotFoundError(f"Missing text condition cache: {condition_path}")
            result["conditions"] = torch.load(condition_path, map_location="cpu", weights_only=True)
        return result


def collate_outpainting(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensors normally while retaining heterogeneous lazy-cache state."""
    requests = [sample.pop("_anchor_requests") for sample in samples]
    entries = [sample.pop("_anchor_entries") for sample in samples]
    batch = default_collate(samples)
    batch["_anchor_requests"] = requests
    batch["_anchor_entries"] = entries
    return batch


class DeterministicEpochSampler(Sampler[tuple[int, int]]):
    """Stateless priority-group order with an explicit resume epoch and offset.

    Every rank constructs the same global order. ``Accelerator.prepare`` then
    shards its batches, so skipping ``global_offset`` here happens before rank
    sharding and exactly reproduces the next global item.  The priority groups
    are a startup snapshot: caches created lazily during training never change
    the order of an in-progress or resumed run.
    """

    def __init__(
        self,
        dataset_length: int,
        *,
        seed: int,
        epoch_size: int,
        ready_indices: tuple[int, ...] | None = None,
        missing_indices: tuple[int, ...] | None = None,
        start_epoch: int = 0,
        start_offset: int = 0,
    ) -> None:
        if epoch_size < 1 or epoch_size > dataset_length:
            raise ValueError("epoch_size must be within the dataset")
        if not 0 <= start_offset < epoch_size:
            raise ValueError("start_offset must be within the truncated epoch")
        self.dataset_length = dataset_length
        self.seed = seed
        self.epoch_size = epoch_size
        self.epoch = start_epoch
        self.offset = start_offset
        if (ready_indices is None) != (missing_indices is None):
            raise ValueError("ready_indices and missing_indices must be provided together")
        if ready_indices is None:
            ready_indices, missing_indices = tuple(range(dataset_length)), ()
        combined = (*ready_indices, *missing_indices)
        if len(combined) != dataset_length or set(combined) != set(range(dataset_length)):
            raise ValueError("READY/MISSING priority groups must partition the dataset exactly once")
        self.ready_indices = tuple(ready_indices)
        self.missing_indices = tuple(missing_indices)

    def __iter__(self):
        epoch, offset = self.epoch, self.offset
        generator = torch.Generator().manual_seed(self.seed + epoch)
        groups: list[torch.Tensor] = []
        for indices in (self.ready_indices, self.missing_indices):
            if indices:
                values = torch.tensor(indices, dtype=torch.int64)
                groups.append(values[torch.randperm(len(indices), generator=generator)])
        order = torch.cat(groups)[: self.epoch_size]
        self.epoch += 1
        self.offset = 0
        return iter((int(index), epoch) for index in order[offset:])

    def __len__(self) -> int:
        return self.epoch_size
