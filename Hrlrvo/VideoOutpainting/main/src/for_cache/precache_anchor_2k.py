"""Precache FIRST-65 Anchor records through the training LazyAnchorCache.

From main, with the Hrlrvo conda environment active:
    PYTHONPATH=src python -m for_cache.precache_anchor_2k --verify-one
    PYTHONPATH=src python -m for_cache.precache_anchor_2k --all

No transformer, training loop, content filtering, or second encoder path is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import signal
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import torch
from ltx_core.model.video_vae.video_vae import map_temporal_slice
from ltx_trainer.model_loader import read_video_scale_factors, resolve_video_vae_path
from torch.utils.data import DataLoader

from ltx_outpaint.anchor_cache import (
    CACHE_FORMAT,
    PREPROCESSING_VERSION,
    LazyAnchorCache,
)
from ltx_outpaint.data import (
    CacheAudit,
    OutpaintingCacheDataset,
    audit_ltx_cache,
    collate_outpainting,
)
from ltx_outpaint.train import _build_anchor_cache, load_training_config

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPU_PAUSED_EXIT_CODE = 75


class GPUPauseRequested(RuntimeError):
    """Request a clean batch-boundary exit so the CUDA context is released."""


def _atomic_json_replace(path: Path, payload: dict[str, Any]) -> None:
    """Publish operational progress atomically; cache serialization is unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Publish a completed benchmark without replacing an existing report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pause_reason(paths: tuple[Path, ...]) -> str | None:
    return next((str(path) for path in paths if path.exists()), None)


def _check_gpu_pause(paths: tuple[Path, ...]) -> None:
    reason = _pause_reason(paths)
    if reason is not None:
        raise GPUPauseRequested(f"GPU pause requested by {reason}")


def _release_encoder(cache: LazyAnchorCache | None) -> None:
    """Move the VAE off device and return cached CUDA allocations."""
    if cache is None or cache._encoder is None:
        return
    if cache.device.type == "cuda":
        torch.cuda.synchronize(cache.device)
    cache._encoder.to("cpu")
    if cache.device.type == "cuda":
        torch.cuda.empty_cache()


def verify_one(
    audit: CacheAudit,
    cache: LazyAnchorCache,
    record_id: str | None,
    seed: int,
    *,
    pause_files: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Validate one real artifact and the actual Dataset -> complete_batch path."""
    records = [path.relative_to(audit.root / "latents").as_posix() for path in audit.compatible_files]
    record_id = record_id or records[0]
    if record_id not in records:
        raise ValueError(f"Not a compatible GT record ID relative to latents/: {record_id}")
    index = records.index(record_id)
    request = cache.make_request(record_id, 0)
    gt_path = audit.compatible_files[index]
    anchor_path = cache.entry_path(request)
    source_record = audit.records[record_id]
    source_path = cache._verify_source(source_record)
    checks: dict[str, str] = {}

    def check(name: str, condition: bool) -> None:
        if not condition:
            raise ValueError(f"SINGLE SAMPLE assertion failed: {name}")
        checks[name] = "PASS"

    with av.open(str(source_path)) as container:
        stream = container.streams.video[0]
        source = {
            "absolute_path": str(source_path),
            "relative_path": source_record["relative_path"],
            "resolution_hw": [stream.height, stream.width],
            "fps": float(stream.average_rate),
            "total_frame_count": int(stream.frames),
            "fingerprint": request["source_fingerprint"],
        }
        check("source_fps", Fraction(stream.average_rate) == Fraction(str(audit.fps)))
        check("source_resolution", source["resolution_hw"] == request["source_resolution_hw"])
        check("source_frame_count", stream.frames == request["source_frame_count"] and stream.frames >= 65)

    gt = OutpaintingCacheDataset._load_record(gt_path)
    full = gt["latents"]
    check("GT_official_tensor_fields", set(gt) == {"latents", "num_frames", "height", "width", "fps"})
    check("GT_full_shape", tuple(full.shape) == (
        cache.latent_channels, source_record["latent_num_frames"], audit.latent_height, audit.latent_width,
    ) and (gt["num_frames"], gt["height"], gt["width"]) == tuple(full.shape[1:]))
    frames = 1 + (audit.source_clip_frames - 1) // cache.scale_factors.time
    rgb, _ = map_temporal_slice(0, frames, 0, 0, cache.scale_factors.time)
    check("official_FIRST_temporal_mapping", (rgb.start, rgb.stop, frames) == (0, 65, 9))
    check("rgb_start_frame", request["rgb_start_frame"] == 0)
    check("rgb_frame_count", request["rgb_frame_count"] == 65)
    check("latent_start_index", request["latent_start_index"] == 0)
    check("latent_frame_count", request["latent_frame_count"] == frames)
    check("GT_Anchor_same_record_id", anchor_path.relative_to(cache.root / "latents").as_posix() == record_id)
    check("raw_relative_identity", request["raw_source_relative_path"] == source_record["relative_path"]
          and Path(request["raw_source_relative_path"]).with_suffix(".pt").as_posix() == record_id)
    check("source_fingerprint", request["source_fingerprint"] == {
        key: source_record["source_fingerprint"][key] for key in ("size_bytes", "sample_sha256")
    })
    check("fps_correspondence", float(gt["fps"]) == request["source_fps"] == source["fps"])
    check("VAE_SHA_correspondence", request["vae_sha256"] == audit.checkpoint_sha256)
    cache._verify_checkpoint()  # Actual checkpoint bytes, not just matching declarations.
    y0, x0, y1, x1 = request["anchor_rect_source_yxyx"]
    ah, aw = cache.anchor_size
    sh, sw = source["resolution_hw"]
    check("exact_center_crop", [y0, x0, y1, x1] == [(sh - ah) // 2, (sw - aw) // 2,
                                                   (sh - ah) // 2 + ah, (sw - aw) // 2 + aw])
    check("requested_content_shape", [y1 - y0, x1 - x0] == request["requested_anchor_hw"])
    if (sh, sw, ah, aw) == (2160, 3840, 1080, 1920):
        check("2K_in_4K_exact_crop", [y0, x0, y1, x1] == [540, 960, 1620, 2880])
    padding = audit.contract["padding"]
    top, left = int(padding.get("top", 0)), int(padding.get("left", 0))
    check("GT_padding_canonical_rect", request["anchor_rect_canonical_yxyx"] ==
          [y0 + top, x0 + left, y1 + top, x1 + left])
    ly0, lx0, ly1, lx1 = request["canonical_latent_rect_yxyx"]
    check("canonical_latent_placement", 0 <= ly0 < ly1 <= full.shape[-2]
          and 0 <= lx0 < lx1 <= full.shape[-1]
          and [ly1 - ly0, lx1 - lx0] == request["anchor_latent_hw"])
    sy, sx = cache.scale_factors.height, cache.scale_factors.width
    check("canonical_pixel_to_latent_mapping", [ly0 * sy, lx0 * sx, ly1 * sy, lx1 * sx] ==
          request["encoded_rect_canonical_yxyx"])

    _check_gpu_pause(pause_files)
    before = cache.created_count
    entry = cache.get_or_create(request)  # The ONLY encoder/write implementation, also used by training.
    _check_gpu_pause(pause_files)
    created = cache.created_count - before
    reread = cache.read_cached(request)
    check("disk_provenance_checksum", reread is not None and reread["tensor_sha256"] == entry["tensor_sha256"])
    check("GT_Anchor_bfloat16", full.dtype == entry["latents"].dtype == torch.bfloat16)
    check("finite_GT_slice_Anchor", bool(torch.isfinite(full[:, :frames]).all())
          and bool(torch.isfinite(entry["latents"]).all()))
    check("Anchor_temporal_shape", entry["latents"].shape[1] == frames)

    before_hit = cache.created_count
    hit = cache.get_or_create(request)
    check("cache_hit_no_encode", cache.created_count == before_hit and hit["tensor_sha256"] == entry["tensor_sha256"])
    dataset = OutpaintingCacheDataset(audit, cache, seed=seed, temporal_sampling="first")
    loader = DataLoader(dataset, batch_size=1, sampler=[(index, 0)], num_workers=0, collate_fn=collate_outpainting)
    batch = next(iter(loader))
    check("training_loader_cache_HIT", batch["_anchor_entries"][0] is not None)
    check("training_GT_slice_0_9", torch.equal(batch["video_latents"]["latents"][0], full[:, :frames]))
    cache.complete_batch(batch)
    condition = batch["anchor_latents"]["latents"]
    check("training_canonical_Anchor_placement", torch.equal(
        condition[0, :, :, ly0:ly1, lx0:lx1], entry["latents"],
    ))
    check("training_temporal_source_provenance", batch["anchor_provenance"] == [request]
          and int(batch["video_latents"]["rgb_start_frame"][0]) == 0
          and int(batch["video_latents"]["latent_start_index"][0]) == 0)
    check("training_loader_no_second_encode", cache.created_count == before_hit)
    return {
        "verdict": "PASS",
        "source": source,
        "GT": {"path": str(gt_path), "record_id": record_id, "full_latent_shape": list(full.shape),
               "selected_latent_interval": [0, frames], "selected_latent_shape": list(full[:, :frames].shape),
               "dtype": str(full.dtype), "VAE_SHA256": audit.checkpoint_sha256, "contract": audit.contract},
        "Anchor": {"path": str(anchor_path), "record_id": record_id, "created_this_run": created,
                   "selected_RGB_interval": [0, 65], "selected_latent_interval": [0, frames],
                   "latent_shape": list(entry["latents"].shape), "dtype": str(entry["latents"].dtype),
                   "VAE_path": str(cache.vae_path), "tensor_sha256": entry["tensor_sha256"],
                   "provenance": request},
        "training_loader": {"GT_shape": list(batch["video_latents"]["latents"].shape),
                            "Anchor_canvas_shape": list(condition.shape), "cache_hit": True},
        "checks": checks,
        "note": "Independent Anchor and GT crop latent VALUES are intentionally not compared.",
    }


def precache_all(
    audit: CacheAudit,
    cache: LazyAnchorCache,
    *,
    gpu_batch_size: int,
    cpu_workers: int,
    prefetch: int,
    max_new_records: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    pause_files: tuple[Path, ...] = (),
    progress_path: Path | None = None,
    stage: str = "full",
    release_encoder: bool = True,
) -> dict[str, Any]:
    """Offline-only CPU producer / batched GPU consumer, with bounded RAM use.

    CPU threads decode and prepare inputs; only the main thread owns the VAE and
    commits cache entries. Training never invokes this offline prefetch queue.
    """
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_count must be positive and shard_index must be in [0, shard_count)")
    _check_gpu_pause(pause_files)
    started = time.monotonic()
    initial_created = cache.created_count
    inventory_started = time.monotonic()
    ready, missing = cache.cache_inventory()
    inventory_seconds = time.monotonic() - inventory_started
    _check_gpu_pause(pause_files)
    total = len(audit.compatible_files)
    available_missing = len(missing)
    missing = tuple(index for index in missing if index % shard_count == shard_index)
    shard_missing = len(missing)
    if max_new_records is not None:
        missing = missing[:max_new_records]
    selected_missing = len(missing)
    logger.info(
        "START stage=%s total_compatible=%d valid=%d missing=%d shard=%d/%d "
        "shard_missing=%d selected_missing=%d gpu_batch_size=%d cpu_workers=%d "
        "prefetch=%d inventory_seconds=%.3f",
        stage, total, len(ready), available_missing, shard_index, shard_count,
        shard_missing, selected_missing,
        gpu_batch_size, cpu_workers, prefetch, inventory_seconds,
    )
    skipped, processed = len(ready), len(ready)
    processed_work = 0
    batch_metrics: list[dict[str, Any]] = []

    def publish(last_batch: dict[str, Any] | None = None, *, complete: bool = False) -> None:
        if progress_path is None:
            return
        cached_estimate = min(total, len(ready) + processed_work)
        _atomic_json_replace(progress_path, {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "complete": complete,
            "total_compatible": total,
            "valid_at_stage_start": len(ready),
            "missing_at_stage_start": available_missing,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "shard_missing_at_stage_start": shard_missing,
            "selected_missing": selected_missing,
            "processed_selected": processed_work,
            "cached_count": cached_estimate,
            "missing_count": total - cached_estimate,
            "newly_created_stage": cache.created_count - initial_created,
            "gpu_batch_size": gpu_batch_size,
            "cpu_workers": cpu_workers,
            "prefetch": prefetch,
            "last_batch": last_batch,
        })

    publish()

    def prepare(index: int):
        start = time.monotonic()
        record_id = audit.compatible_files[index].relative_to(audit.root / "latents").as_posix()
        request = cache.make_request(record_id, 0)
        prepared = cache.prepare_anchor(request, decode_threads=1)
        return index, request, prepared, time.monotonic() - start

    indices = iter(missing)
    pending: deque[Any] = deque()
    old_offload = cache.offload
    # No transformer is resident in the offline process. Keep this same frozen
    # VAE on device between batches, without changing training's offload policy.
    cache.offload = False
    pool: ThreadPoolExecutor | None = None
    pipeline_started = time.monotonic()
    try:
        pool = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="anchor_rgb")

        def fill() -> None:
            while len(pending) < prefetch:
                _check_gpu_pause(pause_files)
                index = next(indices, None)
                if index is None:
                    break
                assert pool is not None
                pending.append(pool.submit(prepare, index))

        fill()
        while pending:
            _check_gpu_pause(pause_files)
            wait_start = time.monotonic()
            items = [pending.popleft().result() for _ in range(min(gpu_batch_size, len(pending)))]
            cpu_wait = time.monotonic() - wait_start
            _check_gpu_pause(pause_files)
            fill()  # CPU prepares the next batch while the GPU encodes this one.
            _check_gpu_pause(pause_files)
            requests = [item[1] for item in items]
            for index, request, _, _ in items:
                logger.info("RECORD index=%d/%d record_id=%s output=%s", index + 1, total,
                            request["gt_record_id"], cache.entry_path(request))
            if cache.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(cache.device)
            before = cache.created_count
            batch_start = time.monotonic()
            entries = cache.get_or_create_batch(requests, prepared=[item[2] for item in items])
            elapsed = time.monotonic() - batch_start
            created = cache.created_count - before
            skipped += len(items) - created
            processed += len(items)
            processed_work += len(items)
            peak = torch.cuda.max_memory_allocated(cache.device) / 2**30 if cache.device.type == "cuda" else 0.0
            batch_metric = {
                "count": len(items),
                "created": created,
                "gpu_encode_save_seconds": elapsed,
                "cpu_wait_seconds": cpu_wait,
                "cpu_prepare_mean_seconds": sum(item[3] for item in items) / len(items),
                "records_per_gpu_encode_save_second": len(items) / max(elapsed, 1e-9),
                "peak_cuda_allocated_gib": peak,
            }
            batch_metrics.append(batch_metric)
            logger.info("BATCH count=%d created=%d gpu_encode_save_seconds=%.3f cpu_wait_seconds=%.3f "
                        "cpu_prepare_mean_seconds=%.3f records_per_second=%.5f peak_cuda_GiB=%.3f",
                        len(items), created, elapsed, cpu_wait, sum(item[3] for item in items) / len(items),
                        len(items) / max(elapsed, 1e-9), peak)
            logger.info("PROGRESS processed=%d/%d skipped_valid=%d newly_created=%d remaining=%d elapsed_seconds=%.1f",
                        processed, total, skipped, cache.created_count - initial_created,
                        total - processed, time.monotonic() - started)
            publish(batch_metric)
            del entries, items, requests
            _check_gpu_pause(pause_files)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        pending.clear()
        cache.offload = old_offload
        if release_encoder:
            _release_encoder(cache)
    pipeline_seconds = time.monotonic() - pipeline_started
    stats = {
        "stage": stage,
        "total_compatible": total,
        "valid_at_stage_start": len(ready),
        "missing_at_stage_start": available_missing,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "shard_missing_at_stage_start": shard_missing,
        "selected_missing": selected_missing,
        "processed_selected": processed_work,
        "skipped_valid": skipped,
        "newly_created": cache.created_count - initial_created,
        "gpu_batch_size": gpu_batch_size,
        "cpu_workers": cpu_workers,
        "prefetch": prefetch,
        "inventory_seconds": inventory_seconds,
        "pipeline_seconds": pipeline_seconds,
        "records_per_pipeline_second": processed_work / max(pipeline_seconds, 1e-9),
        "peak_cuda_allocated_gib": max((item["peak_cuda_allocated_gib"] for item in batch_metrics), default=0.0),
        "process_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "batches": batch_metrics,
    }
    publish(batch_metrics[-1] if batch_metrics else None, complete=True)
    completion = "COMPLETE" if max_new_records is None else "BOUNDED COMPLETE"
    logger.info(
        "PRECACHE %s stage=%s total=%d skipped_valid=%d newly_created=%d "
        "pipeline_seconds=%.3f records_per_pipeline_second=%.5f",
        completion, stage, total, skipped, cache.created_count - initial_created,
        pipeline_seconds, stats["records_per_pipeline_second"],
    )
    return stats


def _benchmark_signature(audit: CacheAudit, cache: LazyAnchorCache, config_path: Path) -> dict[str, Any]:
    return {
        "config_path": str(config_path.expanduser().resolve()),
        "total_compatible": len(audit.compatible_files),
        "gt_root": str(audit.root),
        "anchor_root": str(cache.root),
        "raw_root": str(cache.raw_root),
        "vae_path": str(cache.vae_path),
        "vae_sha256": audit.checkpoint_sha256,
        "cache_format": CACHE_FORMAT,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_clip_frames": audit.source_clip_frames,
        "clip_latent_frames": audit.clip_latent_frames,
        "anchor_size_hw": list(cache.anchor_size),
        "vae_input_hw": cache.geometry["vae_input_hw"],
    }


def _load_completed_tuning(path: Path, signature: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("completed") is not True or report.get("signature") != signature:
        raise RuntimeError(f"Existing auto-tune report is incomplete or belongs to another contract: {path}")
    selected = report.get("selected_config")
    if not isinstance(selected, dict):
        raise TypeError(f"Existing auto-tune report has no selected_config: {path}")
    values = (selected.get("cpu_workers"), selected.get("gpu_batch_size"), selected.get("prefetch"))
    if any(not isinstance(value, int) or value < 1 for value in values):
        raise RuntimeError(f"Existing auto-tune report has invalid selected_config: {path}")
    if selected["prefetch"] < selected["gpu_batch_size"]:
        raise RuntimeError(f"Existing auto-tune report has prefetch smaller than batch: {path}")
    return report


def benchmark_batch_equivalence(
    audit: CacheAudit,
    cache: LazyAnchorCache,
    *,
    pause_files: tuple[Path, ...],
    progress_path: Path | None,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> dict[str, Any]:
    """Compare the same two prepared samples as individual and batched VAE calls."""
    _check_gpu_pause(pause_files)
    ready, _ = cache.cache_inventory()
    if len(ready) < 2:
        raise RuntimeError("Batch equivalence requires at least two validated production cache entries")
    indices = ready[:2]
    prepared: list[tuple[dict[str, Any], torch.Tensor]] = []
    cached_entries: list[dict[str, Any]] = []
    prepare_seconds: list[float] = []
    for index in indices:
        _check_gpu_pause(pause_files)
        record_id = audit.compatible_files[index].relative_to(audit.root / "latents").as_posix()
        request = cache.make_request(record_id, 0)
        cached = cache.read_cached(request)
        if cached is None:
            raise RuntimeError(f"Validated benchmark entry disappeared: {record_id}")
        started = time.monotonic()
        prepared.append(cache.prepare_anchor(request, decode_threads=1))
        prepare_seconds.append(time.monotonic() - started)
        cached_entries.append(cached)

    old_offload = cache.offload
    cache.offload = False
    singles: list[torch.Tensor] = []
    single_seconds: list[float] = []
    single_peak_gib = 0.0
    try:
        for item in prepared:
            _check_gpu_pause(pause_files)
            if cache.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(cache.device)
            started = time.monotonic()
            singles.append(cache.encode_prepared([item])[0].clone())
            single_seconds.append(time.monotonic() - started)
            if cache.device.type == "cuda":
                single_peak_gib = max(
                    single_peak_gib, torch.cuda.max_memory_allocated(cache.device) / 2**30,
                )
        _check_gpu_pause(pause_files)
        if cache.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cache.device)
        started = time.monotonic()
        batched = cache.encode_prepared(prepared)
        batch_seconds = time.monotonic() - started
        batch_peak_gib = (
            torch.cuda.max_memory_allocated(cache.device) / 2**30
            if cache.device.type == "cuda" else 0.0
        )
        _check_gpu_pause(pause_files)
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning("GPU batch-size 2 equivalence hit OOM; batch-size 1 remains eligible: %s", exc)
        _release_encoder(cache)
        return {
            "verdict": "OOM",
            "numeric_equivalence": False,
            "throughput_improved": False,
            "batch_size_2_eligible": False,
            "error": str(exc),
            "record_ids": [request["gt_record_id"] for request, _ in prepared],
        }
    finally:
        cache.offload = old_offload

    comparisons = []
    for index, ((request, _), single, cached) in enumerate(
        zip(prepared, singles, cached_entries, strict=True)
    ):
        batch_item = batched[index]
        cached_latent = cached["latents"]
        difference = (batch_item.float() - single.float()).abs()
        cached_difference = (single.float() - cached_latent.float()).abs()
        denominator = single.float().abs().clamp_min(1e-6)
        comparisons.append({
            "record_id": request["gt_record_id"],
            "batch_slot": index,
            "batch_vs_single_allclose": bool(torch.allclose(batch_item, single, atol=atol, rtol=rtol)),
            "single_vs_production_cache_allclose": bool(
                torch.allclose(single, cached_latent, atol=atol, rtol=rtol)
            ),
            "batch_vs_single_max_abs": difference.max().item(),
            "batch_vs_single_mean_abs": difference.mean().item(),
            "batch_vs_single_max_relative": (difference / denominator).max().item(),
            "single_vs_production_cache_max_abs": cached_difference.max().item(),
        })
    numeric_equivalence = all(
        item["batch_vs_single_allclose"] and item["single_vs_production_cache_allclose"]
        for item in comparisons
    )
    single_total = sum(single_seconds)
    single_throughput = len(prepared) / max(single_total, 1e-9)
    batch_throughput = len(prepared) / max(batch_seconds, 1e-9)
    throughput_improved = batch_throughput > single_throughput
    report = {
        "verdict": "PASS" if numeric_equivalence else "FAIL",
        "numeric_equivalence": numeric_equivalence,
        "throughput_improved": throughput_improved,
        "batch_size_2_eligible": numeric_equivalence and throughput_improved,
        "atol": atol,
        "rtol": rtol,
        "record_ids": [request["gt_record_id"] for request, _ in prepared],
        "same_prepared_inputs_used": True,
        "unique_record_ids": len({request["gt_record_id"] for request, _ in prepared}) == len(prepared),
        "prepare_seconds": prepare_seconds,
        "single_encode_seconds": single_seconds,
        "single_total_seconds": single_total,
        "batch_encode_seconds": batch_seconds,
        "single_records_per_second": single_throughput,
        "batch_records_per_second": batch_throughput,
        "batch_speedup_over_serial_single": batch_throughput / max(single_throughput, 1e-9),
        "single_peak_cuda_allocated_gib": single_peak_gib,
        "batch_peak_cuda_allocated_gib": batch_peak_gib,
        "comparisons": comparisons,
    }
    if progress_path is not None:
        _atomic_json_replace(progress_path, {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "benchmark_batch_equivalence",
            "complete": True,
            "batch_equivalence": report,
        })
    del batched, singles, cached_entries, prepared
    return report


def auto_tune(
    audit: CacheAudit,
    cache: LazyAnchorCache,
    *,
    config_path: Path,
    report_path: Path,
    pause_files: tuple[Path, ...],
    progress_path: Path | None,
) -> dict[str, Any]:
    """Run a bounded production-safe benchmark and select the long-run settings."""
    signature = _benchmark_signature(audit, cache, config_path)
    completed = _load_completed_tuning(report_path, signature)
    if completed is not None:
        logger.info("Reusing completed auto-tune report: %s", report_path)
        return completed

    partial_path = report_path.with_name(f"{report_path.stem}.partial{report_path.suffix}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed": False,
        "signature": signature,
        "plan": {
            "warmup_new_records": 1,
            "cpu_candidates": [1, 2, 4],
            "records_per_cpu_candidate": 3,
            "optional_gpu_batch_size": 2,
            "optional_batch_candidate_records": 4,
        },
        "results": {},
    }

    def checkpoint(phase: str) -> None:
        report["phase"] = phase
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json_replace(partial_path, report)

    _check_gpu_pause(pause_files)
    checkpoint("warmup")
    report["results"]["warmup"] = precache_all(
        audit, cache, gpu_batch_size=1, cpu_workers=1, prefetch=1,
        max_new_records=1, pause_files=pause_files, progress_path=progress_path,
        stage="benchmark_warmup", release_encoder=False,
    )
    checkpoint("cpu_candidates")

    affinity = len(os.sched_getaffinity(0))
    cpu_candidates: list[dict[str, Any]] = []
    for workers in (1, 2, 4):
        if workers > affinity:
            continue
        prefetch = max(workers, 2)
        stats = precache_all(
            audit, cache, gpu_batch_size=1, cpu_workers=workers, prefetch=prefetch,
            max_new_records=3, pause_files=pause_files, progress_path=progress_path,
            stage=f"benchmark_cpu_w{workers}_b1_q{prefetch}", release_encoder=False,
        )
        cpu_candidates.append(stats)
        report["results"]["cpu_candidates"] = cpu_candidates
        checkpoint(f"cpu_candidate_w{workers}_complete")
    if not cpu_candidates or not any(item["processed_selected"] for item in cpu_candidates):
        raise RuntimeError("No missing records were available for the bounded CPU benchmark")

    best_rate = max(item["records_per_pipeline_second"] for item in cpu_candidates)
    near_best = [
        item for item in cpu_candidates
        if item["records_per_pipeline_second"] >= best_rate * 0.97
    ]
    best_b1 = min(near_best, key=lambda item: item["cpu_workers"])

    checkpoint("batch_equivalence")
    equivalence = benchmark_batch_equivalence(
        audit, cache, pause_files=pause_files, progress_path=progress_path,
    )
    report["results"]["batch_equivalence"] = equivalence
    checkpoint("batch_equivalence_complete")

    batch_candidate = None
    if equivalence["batch_size_2_eligible"]:
        batch_workers = min(4, affinity)
        batch_prefetch = max(batch_workers, 4)
        batch_candidate = precache_all(
            audit, cache, gpu_batch_size=2, cpu_workers=batch_workers, prefetch=batch_prefetch,
            max_new_records=4, pause_files=pause_files, progress_path=progress_path,
            stage=f"benchmark_cpu_w{batch_workers}_b2_q{batch_prefetch}", release_encoder=False,
        )
        report["results"]["batch_candidate"] = batch_candidate
        checkpoint("batch_candidate_complete")

    selected = {
        "cpu_workers": best_b1["cpu_workers"],
        "gpu_batch_size": 1,
        "prefetch": best_b1["prefetch"],
    }
    reason = (
        f"batch1 CPU candidate within 3% of best, preferring fewer workers; "
        f"pipeline={best_b1['records_per_pipeline_second']:.6f} records/s"
    )
    if batch_candidate is not None:
        required = best_b1["records_per_pipeline_second"] * 1.02
        if batch_candidate["records_per_pipeline_second"] > required:
            selected = {
                "cpu_workers": batch_candidate["cpu_workers"],
                "gpu_batch_size": 2,
                "prefetch": batch_candidate["prefetch"],
            }
            reason = (
                "batch2 passed same-input numeric equivalence and exceeded the selected batch1 "
                f"pipeline throughput by >2% ({batch_candidate['records_per_pipeline_second']:.6f} "
                f"vs {best_b1['records_per_pipeline_second']:.6f} records/s)"
            )
        else:
            reason += "; batch2 did not exceed batch1 pipeline throughput by the required 2%"
    elif equivalence["verdict"] == "OOM":
        reason += "; batch2 equivalence OOM, so batch1 retained"
    elif not equivalence["numeric_equivalence"]:
        reason += "; batch2 numeric equivalence failed, so batch1 retained"
    else:
        reason += "; batch2 direct encode throughput did not improve, so batch1 retained"

    _check_gpu_pause(pause_files)
    report.update({
        "completed": True,
        "phase": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selected_config": selected,
        "selection_reason": reason,
        "bounded_new_records": sum(
            item.get("newly_created", 0)
            for item in [report["results"]["warmup"], *cpu_candidates]
        ) + (batch_candidate or {}).get("newly_created", 0),
        "max_process_rss_gib": max(
            item.get("process_max_rss_gib", 0.0)
            for item in [
                report["results"]["warmup"], *cpu_candidates,
                *([batch_candidate] if batch_candidate else []),
            ]
        ),
        "max_torch_cuda_allocated_gib": max(
            [item.get("peak_cuda_allocated_gib", 0.0)
             for item in [
                 report["results"]["warmup"], *cpu_candidates,
                 *([batch_candidate] if batch_candidate else []),
             ]]
            + [equivalence.get("single_peak_cuda_allocated_gib", 0.0),
               equivalence.get("batch_peak_cuda_allocated_gib", 0.0)]
        ),
    })
    _atomic_json_new(report_path, report)
    checkpoint("complete")
    logger.info("AUTO-TUNE COMPLETE selected=%s reason=%s report=%s", selected, reason, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify-one", action="store_true")
    modes.add_argument("--all", action="store_true")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train_4k.yaml")
    parser.add_argument("--record-id", help="GT .pt ID relative to latents/, only with --verify-one")
    parser.add_argument("--report", type=Path, help="New JSON verification report path; never overwritten")
    parser.add_argument("--gpu-batch-size", type=int, default=1, help="Offline VAE batch; training remains synchronous")
    parser.add_argument("--cpu-workers", type=int, default=len(os.sched_getaffinity(0)),
                        help="Independent CPU decode/crop workers; one FFmpeg thread each")
    parser.add_argument("--prefetch", type=int, help="Maximum queued CPU preparations (default max(workers, 2*batch))")
    parser.add_argument("--max-new-records", type=int, help="Bound --all to this many current misses")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Deterministically split GT-order records across independent runners")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based shard owned by this runner")
    parser.add_argument("--auto-tune", action="store_true",
                        help="Bounded workers/batch benchmark, then continue the full resumable --all run")
    parser.add_argument("--benchmark-report", type=Path,
                        help="Persistent auto-tune result (default outputs/cache_anchor_2k/benchmark.json)")
    parser.add_argument("--gpu-hold-file", type=Path,
                        help="Exit 75 at the next safe boundary when this leader-priority hold exists")
    parser.add_argument("--gpu-yield-file", type=Path,
                        help="Exit 75 at the next safe boundary when the supervisor requests GPU yield")
    parser.add_argument("--progress-report", type=Path, help="Atomically replaced operational progress JSON")
    args = parser.parse_args()
    if args.all and (args.record_id or args.report):
        parser.error("--record-id/--report require --verify-one")
    if not args.all and (args.max_new_records is not None or args.auto_tune or args.benchmark_report
                         or args.gpu_hold_file or args.gpu_yield_file or args.progress_report
                         or args.shard_count != 1 or args.shard_index != 0):
        parser.error("Operational/auto-tune options require --all")
    if args.auto_tune and args.max_new_records is not None:
        parser.error("--auto-tune continues the full run and cannot be combined with --max-new-records")
    if args.benchmark_report is not None and not args.auto_tune:
        parser.error("--benchmark-report requires --auto-tune")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-count must be positive and --shard-index must be in its range")
    if args.auto_tune and (args.shard_count != 1 or args.shard_index != 0):
        parser.error("--auto-tune is single-runner only; benchmark first, then launch explicit shards")
    if args.max_new_records is not None and args.max_new_records < 1:
        parser.error("--max-new-records must be positive")
    if args.gpu_batch_size < 1 or args.cpu_workers < 1 or (args.prefetch is not None and args.prefetch < args.gpu_batch_size):
        parser.error("GPU batch/CPU workers must be positive and prefetch must be >= GPU batch")
    prefetch = args.prefetch or max(args.cpu_workers, args.gpu_batch_size * 2)
    pause_files = tuple(
        path.expanduser().resolve()
        for path in (args.gpu_hold_file, args.gpu_yield_file)
        if path is not None
    )
    if args.all:
        torch.set_num_threads(1)  # One CPU thread per producer; avoid workers x intra-op oversubscription.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    single_validated = False
    cache: LazyAnchorCache | None = None
    try:
        _check_gpu_pause(pause_files)
        config, outpainting = load_training_config(args.config)
        if outpainting.get("temporal_sampling") != "first" or int(outpainting["clip_frames"]) != 65:
            raise ValueError("Precache and training must both use temporal_sampling=first and clip_frames=65")
        scale = read_video_scale_factors(resolve_video_vae_path(config.model.model_path, config.model.video_vae_path))
        audit = audit_ltx_cache(outpainting["cache_root"], source_clip_frames=int(outpainting["clip_frames"]),
                                scale_factors=scale, requested_canvas=tuple(outpainting["canvas_pixels"]))
        cache = _build_anchor_cache(audit, config, outpainting)
        if args.all:
            # Enforce the single-record gate even when --all is invoked directly.
            # A previous --verify-one makes this a disk HIT, not another encode.
            gate = verify_one(audit, cache, None, config.seed, pause_files=pause_files)
            single_validated = True
            logger.info("SINGLE SAMPLE ANCHOR CACHE VALIDATION: PASS (precache gate, record_id=%s)",
                        gate["GT"]["record_id"])
            del gate
            selected = {
                "gpu_batch_size": args.gpu_batch_size,
                "cpu_workers": args.cpu_workers,
                "prefetch": prefetch,
            }
            if args.auto_tune:
                benchmark_path = args.benchmark_report or (
                    PROJECT_ROOT / "outputs" / "cache_anchor_2k" / "anchor_cache_benchmark.json"
                )
                tuning = auto_tune(
                    audit, cache, config_path=args.config, report_path=benchmark_path,
                    pause_files=pause_files, progress_path=args.progress_report,
                )
                selected = tuning["selected_config"]
            precache_all(
                audit, cache,
                gpu_batch_size=selected["gpu_batch_size"],
                cpu_workers=selected["cpu_workers"],
                prefetch=selected["prefetch"],
                max_new_records=args.max_new_records,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                pause_files=pause_files,
                progress_path=args.progress_report,
                stage="full",
            )
        else:
            report = verify_one(audit, cache, args.record_id, config.seed)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            path = args.report or PROJECT_ROOT / "outputs" / "cache_anchor_2k" / f"verify_one_{stamp}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            print(json.dumps(report, indent=2), flush=True)
            print(f"Verification report: {path}", flush=True)
            print("SINGLE SAMPLE ANCHOR CACHE VALIDATION: PASS", flush=True)
            single_validated = True
    except GPUPauseRequested as exc:
        logger.info("%s; ending process so VAE/CUDA memory is fully returned", exc)
        raise SystemExit(GPU_PAUSED_EXIT_CODE) from None
    except KeyboardInterrupt:
        logger.info("Stopped; valid saved entries are reusable. Rerun --all to continue.")
        raise SystemExit(130) from None
    except Exception:
        logger.exception("Anchor precache failed; no existing cache is overwritten")
        if not single_validated:
            print("SINGLE SAMPLE ANCHOR CACHE VALIDATION: FAIL", flush=True)
        raise SystemExit(1) from None
    finally:
        _release_encoder(cache)


if __name__ == "__main__":
    main()
