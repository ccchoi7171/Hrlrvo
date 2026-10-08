"""Lazy, independent RGB Anchor encoding; the existing GT cache is read-only.

Workers only verify requests/read existing entries. The owning training rank
encodes cache misses before its forward, under a per-entry filesystem lock.
Only cropped Anchor RGB enters the frozen official VAE, never full-frame GT.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import uuid
from contextlib import ExitStack
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import av
import numpy as np
import torch
from ltx_core.model.video_vae.video_vae import (
    map_spatial_interval_to_latent,
    map_temporal_slice,
)
from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig
from ltx_pipelines.utils.media_io.resize import (
    ResizeMode,
    align_resolution,
    resize_and_reflect_pad,
)
from ltx_trainer.model_loader import load_video_vae_encoder, read_video_scale_factors
from safetensors import safe_open
from torch.utils.data import get_worker_info

if TYPE_CHECKING:
    from .data import CacheAudit

logger = logging.getLogger(__name__)
PREPROCESSING_VERSION = "ltx23-independent-anchor-cfr-canonical-reflect-v1"
CACHE_FORMAT = "ltx-independent-anchor-first65-v2"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=4)
def _checkpoint_digest(path: str, size: int, mtime_ns: int) -> str:
    """One full checksum per checkpoint version/process, only during actual use."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    after = Path(path).stat()
    if (after.st_size, after.st_mtime_ns) != (size, mtime_ns):
        raise RuntimeError("VAE checkpoint changed while its checksum was being verified")
    return digest.hexdigest()


def _tensor_digest(latent: torch.Tensor) -> str:
    return sha256(latent.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class LazyAnchorCache:
    """Small filesystem cache; no GPU/VAE work in the DataLoader workers."""

    def __init__(
        self,
        audit: CacheAudit,
        *,
        cache_root: str | Path,
        raw_root: str | Path,
        vae_path: str | Path,
        anchor_size: tuple[int, int],
        device: str = "cuda",
        tile_size: int = 512,
        tile_overlap: int = 128,
        offload: bool = True,
    ) -> None:
        self.audit = audit
        self.root = Path(cache_root).expanduser().resolve()
        self.raw_root = Path(raw_root).expanduser().resolve()
        self.vae_path = Path(vae_path).expanduser().resolve()
        if self.root == audit.root or self.root.is_relative_to(audit.root):
            raise ValueError("Anchor cache must not write inside the existing GT cache")
        if self.root == self.raw_root or self.root.is_relative_to(self.raw_root):
            raise ValueError("Anchor cache must be separate from raw RGB sources")
        if not self.raw_root.is_dir() or not self.vae_path.is_file():
            raise FileNotFoundError("Raw source root and official VAE checkpoint must exist")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("Anchor cache root is not a directory")
        self.scale_factors = read_video_scale_factors(str(self.vae_path))
        if self.scale_factors.time != audit.temporal_scale:
            raise ValueError("GT cache and selected VAE temporal compression differ")
        if audit.source_clip_frames != 65 or self.scale_factors.time != 8:
            raise ValueError("FIRST-only Anchor cache requires 65 RGB frames and the LTX temporal factor 8")
        expected_frames = 1 + (audit.source_clip_frames - 1) // self.scale_factors.time
        rgb, _ = map_temporal_slice(0, expected_frames, 0, 0, self.scale_factors.time)
        if audit.clip_latent_frames != expected_frames or (rgb.start, rgb.stop) != (0, audit.source_clip_frames):
            raise ValueError("FIRST-only Anchor cache temporal contract differs from official VAE mapping")
        with safe_open(str(self.vae_path), framework="pt", device="cpu") as handle:
            config = json.loads((handle.metadata() or {})["config"])["vae"]
        # Prefix safety depends on this actual encoder, not the decoder's flag.
        if config.get("_class_name") != "CausalVideoAutoencoder" or config.get("norm_layer") != "pixel_norm":
            raise ValueError("Clean cached prefixes require the audited causal, pixel-normalized LTX-2.3 encoder")
        causal_blocks = {"res_x", "res_x_y", "compress_time", "compress_space", "compress_all",
                         "compress_all_x_y", "compress_all_res", "compress_space_res", "compress_time_res"}
        if not config.get("encoder_blocks") or any(name not in causal_blocks for name, _ in config["encoder_blocks"]):
            raise ValueError("Selected encoder contains an unaudited temporal operation")
        self.latent_channels = int(config["latent_channels"])
        self.encoder_spatial_padding = config.get("spatial_padding_mode", config.get("encoder_spatial_padding_mode", "zeros"))
        self.anchor_size = tuple(map(int, anchor_size))
        if len(self.anchor_size) != 2 or min(self.anchor_size) < 1:
            raise ValueError("anchor_size must be positive (height, width)")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("Anchor VAE device must be cuda or cpu")
        self.tile_size, self.tile_overlap, self.offload = int(tile_size), int(tile_overlap), bool(offload)
        sy, sx = self.scale_factors.height, self.scale_factors.width
        if any(self.tile_size % s or self.tile_overlap % s for s in (sy, sx)):
            raise ValueError("VAE tile size/overlap must align to both checkpoint spatial factors")
        if self.tile_overlap < 64 or self.tile_size <= self.tile_overlap:
            raise ValueError("Native conv-VAE tiled encoding requires overlap >=64 and tile size > overlap")
        self._encoder = None
        self.created_count = 0
        self._verified_sources: dict[str, tuple[int, int, int, int]] = {}
        self._gt_contract_digest = sha256(_canonical_json(audit.contract).encode()).hexdigest()
        self.geometry = self._geometry()
        self._verify_checkpoint()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_encoder"] = None  # Never pickle a VAE into spawned workers.
        state["_verified_sources"] = {}
        return state

    def _geometry(self) -> dict[str, Any]:
        contract = self.audit.contract
        source_w, source_h = map(int, contract["source_resolution"])
        full_w, full_h = map(int, contract["cache_resolution"])
        full_pad = contract["padding"]
        gt_top, gt_left = int(full_pad.get("top", 0)), int(full_pad.get("left", 0))
        if (source_h + gt_top + int(full_pad.get("bottom", 0)),
                source_w + gt_left + int(full_pad.get("right", 0))) != (full_h, full_w):
            raise ValueError("GT source/padded canvas geometry does not agree")
        sy, sx = self.scale_factors.height, self.scale_factors.width
        if (self.audit.latent_height * sy, self.audit.latent_width * sx) != (full_h, full_w):
            raise ValueError("GT cache spatial mapping differs from selected VAE")
        ah, aw = self.anchor_size
        if ah > source_h or aw > source_w:
            raise ValueError("Requested Anchor exceeds source; no silent resize")
        y0, x0 = (source_h - ah) // 2, (source_w - aw) // 2
        cy0, cx0 = y0 + gt_top, x0 + gt_left
        cy1, cx1 = cy0 + ah, cx0 + aw
        top, left = cy0 % sy, cx0 % sx
        # Native round-up policy; choose padding sides from the canonical origin,
        # not from the Anchor tensor's shape. No interpolation or content crop.
        encoded_h = align_resolution(aw, ah + top, ResizeMode.REFLECT_PAD, sy)[1]
        encoded_w = align_resolution(aw + left, ah, ResizeMode.REFLECT_PAD, sx)[0]
        bottom, right = encoded_h - ah - top, encoded_w - aw - left
        if max(top, bottom) >= ah or max(left, right) >= aw:
            raise ValueError("Anchor is too small for the required native reflect padding")
        ey0, ex0 = cy0 - top, cx0 - left
        ey1, ex1 = ey0 + encoded_h, ex0 + encoded_w
        if ey1 > full_h or ex1 > full_w:
            raise ValueError("Aligned Anchor encoding extends beyond the canonical canvas")
        latent_y, _ = map_spatial_interval_to_latent(ey0, ey1, 0, 0, sy)
        latent_x, _ = map_spatial_interval_to_latent(ex0, ex1, 0, 0, sx)
        known_y, _ = map_spatial_interval_to_latent(cy0, cy1, 0, 0, sy)
        known_x, _ = map_spatial_interval_to_latent(cx0, cx1, 0, 0, sx)
        if known_y.stop <= known_y.start or known_x.stop <= known_x.start:
            raise ValueError("Anchor has no known tokens under the existing native mask rounding")
        return {
            "source_resolution_hw": [source_h, source_w],
            "full_encoded_resolution_hw": [full_h, full_w],
            "gt_pixel_padding": full_pad,
            "anchor_rect_source_yxyx": [y0, x0, y0 + ah, x0 + aw],
            "anchor_rect_canonical_yxyx": [cy0, cx0, cy1, cx1],
            "requested_anchor_hw": [ah, aw],
            "vae_input_hw": [encoded_h, encoded_w],
            "vae_padding": {"top": top, "bottom": bottom, "left": left, "right": right,
                            "mode": "reflect", "placement": "canonical_lattice_enclosure"},
            "vae_crop": [0, 0, 0, 0],
            "content_rect_in_vae_input_yxyx": [top, left, top + ah, left + aw],
            "encoded_rect_canonical_yxyx": [ey0, ex0, ey1, ex1],
            "anchor_latent_hw": [encoded_h // sy, encoded_w // sx],
            "canonical_latent_rect_yxyx": [latent_y.start, latent_x.start, latent_y.stop, latent_x.stop],
            # Preserve existing native spatial_crop mask semantics. Storage
            # support and the floor-rounded known mask are NOT the same box.
            "known_latent_rect_yxyx": [known_y.start, known_x.start, known_y.stop, known_x.stop],
            "known_mask_policy": "native_spatial_crop_floor_bounds",
        }

    def _source_path(self, relative: str) -> Path:
        path = (self.raw_root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(self.raw_root) or not path.is_file():
            raise FileNotFoundError(f"Unsafe or missing raw source match: {relative}")
        return path

    def _verify_source(self, record: dict[str, Any]) -> Path:
        path = self._source_path(record["relative_path"])
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if self._verified_sources.get(str(path)) == signature:
            return path
        # Exact algorithm used by this GT cache's SourceFingerprint producer:
        # first MiB + last MiB (if >1 MiB) + ASCII decimal file size.
        expected = record["source_fingerprint"]
        if stat.st_size != int(expected["size_bytes"]):
            raise ValueError(f"Raw/GT source size mismatch: {path}")
        digest = sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(1 << 20))
            if stat.st_size > 1 << 20:
                handle.seek(max(0, stat.st_size - (1 << 20)))
                digest.update(handle.read(1 << 20))
        digest.update(str(stat.st_size).encode())
        after = path.stat()
        if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Raw source changed during provenance verification: {path}")
        if digest.hexdigest() != expected["sample_sha256"]:
            raise ValueError(f"Raw/GT source fingerprint mismatch: {path}")
        # Relocated files may have different mtime; content digest/size govern identity.
        self._verified_sources[str(path)] = signature
        return path

    def make_request(self, record_id: str, latent_start: int) -> dict[str, Any]:
        if latent_start != 0:
            raise ValueError("FIRST-only Anchor cache rejects nonzero latent_start_index; use RGB [0:65)")
        self._record_path(record_id)
        record = self.audit.records[record_id]
        if Path(record["relative_path"]).with_suffix(".pt").as_posix() != record_id:
            raise ValueError("GT record ID and raw relative path do not agree")
        self._verify_source(record)
        if [int(record["original_height"]), int(record["original_width"])] != self.geometry["source_resolution_hw"]:
            raise ValueError("Raw record/source canvas geometry differs from cache contract")
        if float(record["fps"]) != self.audit.fps:
            raise ValueError("Raw record FPS differs from GT cache audit")
        stop = latent_start + self.audit.clip_latent_frames
        if latent_start < 0 or stop > int(record["latent_num_frames"]):
            raise ValueError("Requested latent crop is outside cached temporal support")
        rgb, _ = map_temporal_slice(latent_start, stop, 0, 0, self.scale_factors.time)
        if rgb.stop - rgb.start != self.audit.source_clip_frames or rgb.stop > int(record["cached_num_frames"]):
            raise ValueError("RGB interval and GT latent crop are not exactly aligned")
        fp = record["source_fingerprint"]
        return {
            "format": CACHE_FORMAT,
            "preprocessing_version": PREPROCESSING_VERSION,
            "temporal_sampling": "first",
            "record_path_policy": "gt_relative_record_id",
            "gt_record_id": record_id,
            "raw_source_relative_path": record["relative_path"],
            "source_fingerprint": {"size_bytes": int(fp["size_bytes"]), "sample_sha256": fp["sample_sha256"]},
            "rgb_start_frame": rgb.start,
            "rgb_stop_frame_exclusive": rgb.stop,
            "latent_start_index": latent_start,
            "latent_stop_index_exclusive": stop,
            "rgb_frame_count": self.audit.source_clip_frames,
            "latent_frame_count": self.audit.clip_latent_frames,
            "source_frame_count": int(record["original_num_frames"]),
            "cached_rgb_frame_count": int(record["cached_num_frames"]),
            "source_fps": float(record["fps"]),
            "anchor_temporal_encoding": "independent_selected_rgb_interval_with_causal_reset",
            "gt_temporal_encoding": "full_cached_prefix_then_aligned_latent_slice",
            "vae_identifier": self.vae_path.name,
            "vae_sha256": self.audit.checkpoint_sha256,
            "gt_cache_contract_sha256": self._gt_contract_digest,
            "gt_preprocessing_version": self.audit.contract["preprocessing_version"],
            "vae_scale_factors_thw": [self.scale_factors.time, self.scale_factors.height, self.scale_factors.width],
            "latent_channels": self.latent_channels,
            "dtype": "torch.bfloat16",
            "normalization": "official_rgb01_to_minus1_plus1; encoder_per_channel_statistics",
            "encoder_internal_padding": {"spatial": self.encoder_spatial_padding, "temporal": "causal_repeat_first"},
            "spatial_tiling": {"tile_size": self.tile_size, "tile_overlap": self.tile_overlap, "temporal_tiling": False},
            **self.geometry,
        }

    def _record_path(self, record_id: str) -> Path:
        record = Path(record_id)
        if (record.is_absolute() or ".." in record.parts or record.as_posix() != record_id
                or record.suffix != ".pt" or record_id not in self.audit.records):
            raise ValueError("Unsafe GT record ID")
        path = self.root / "latents" / record
        if path.resolve() != path:
            raise ValueError("Anchor cache symlink changes the canonical GT-relative record path")
        return path

    def entry_path(self, request: dict[str, Any]) -> Path:
        expected = {
            "format": CACHE_FORMAT,
            "temporal_sampling": "first",
            "record_path_policy": "gt_relative_record_id",
            "rgb_start_frame": 0,
            "rgb_stop_frame_exclusive": self.audit.source_clip_frames,
            "rgb_frame_count": self.audit.source_clip_frames,
            "latent_start_index": 0,
            "latent_stop_index_exclusive": self.audit.clip_latent_frames,
            "latent_frame_count": self.audit.clip_latent_frames,
        }
        if any(request.get(key) != value for key, value in expected.items()):
            raise ValueError("Incompatible Anchor request: only FIRST RGB [0:65) / latent [0:9) is supported")
        return self._record_path(request["gt_record_id"])

    def _validate_request(self, request: dict[str, Any]) -> Path:
        """Require the complete canonical request, not merely its path fields."""
        record_id = request.get("gt_record_id")
        if not isinstance(record_id, str) or request != self.make_request(record_id, 0):
            raise ValueError("Anchor request differs from canonical GT/source/VAE provenance")
        return self.entry_path(request)

    def cache_inventory(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Snapshot READY/MISSING in GT order, validating each hit, never encoding.

        Missing sources are verified synchronously when selected. Existing
        entries must pass the same provenance/checksum checks as training reads.
        No latent tensors are retained by this inventory.
        """
        ready, missing = [], []
        for index, target in enumerate(self.audit.compatible_files):
            record_id = target.relative_to(self.audit.root / "latents").as_posix()
            if self._record_path(record_id).exists():
                entry = self.read_cached(self.make_request(record_id, 0))
                if entry is None:
                    raise RuntimeError(f"Anchor cache disappeared during inventory: {record_id}")
                ready.append(index)
                del entry
            else:
                missing.append(index)
            if (index + 1) % 1000 == 0:
                logger.info("Anchor inventory %d/%d: valid=%d missing=%d", index + 1,
                            len(self.audit.compatible_files), len(ready), len(missing))
        return tuple(ready), tuple(missing)

    def read_cached(self, request: dict[str, Any]) -> dict[str, Any] | None:
        path = self._validate_request(request)
        if not path.exists():
            return None
        try:
            entry = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(entry, dict) or entry.get("provenance") != request:
                raise ValueError("provenance differs from requested GT/RGB interval")
            latent = entry["latents"]
            h, w = request["anchor_latent_hw"]
            expected = (request["latent_channels"], request["latent_frame_count"], h, w)
            if tuple(latent.shape) != expected or latent.dtype != torch.bfloat16:
                raise ValueError("latent shape/dtype mismatch")
            if (entry["num_frames"], entry["height"], entry["width"], entry["fps"]) != (*expected[1:], request["source_fps"]):
                raise ValueError("official latent record fields differ from tensor/provenance")
            if not torch.isfinite(latent).all().item() or _tensor_digest(latent) != entry["tensor_sha256"]:
                raise ValueError("latent is nonfinite or tensor checksum differs")
        except Exception as exc:
            raise RuntimeError(f"Corrupt/incompatible Anchor cache (not overwritten): {path}: {exc}") from exc
        return entry

    def _verify_checkpoint(self) -> None:
        stat = self.vae_path.stat()
        actual = _checkpoint_digest(str(self.vae_path), stat.st_size, stat.st_mtime_ns)
        if actual != self.audit.checkpoint_sha256:
            raise ValueError("Selected VAE checksum differs from the existing GT cache VAE")

    def _read_anchor_rgb(self, request: dict[str, Any], *, decode_threads: int = 0) -> torch.Tensor:
        """Keyframe seek, exact CFR PTS indices, crop each RGB frame before stacking."""
        if decode_threads < 0:
            raise ValueError("decode_threads must be nonnegative (zero keeps PyAV's default)")
        record = self.audit.records[request["gt_record_id"]]
        path = self._verify_source(record)
        start, count = request["rgb_start_frame"], request["rgb_frame_count"]
        y0, x0, y1, x1 = request["anchor_rect_source_yxyx"]
        frames = np.empty((count, y1 - y0, x1 - x0, 3), dtype=np.uint8)
        selected = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if decode_threads:
                stream.thread_count = decode_threads
            rate = stream.average_rate or stream.base_rate
            if rate is None or stream.time_base is None or stream.start_time is None:
                raise ValueError(f"Missing exact frame-index metadata: {path}")
            rate, time_base = Fraction(rate), Fraction(stream.time_base)
            if rate != Fraction(str(request["source_fps"])):
                raise ValueError(f"Source FPS differs from GT provenance: {path}")
            if [stream.height, stream.width] != request["source_resolution_hw"]:
                raise ValueError(f"Source resolution differs from GT provenance: {path}")
            if stream.frames > 0 and stream.frames != request["source_frame_count"]:
                raise ValueError(f"Source frame count differs from GT provenance: {path}")
            target_pts = Fraction(stream.start_time) + Fraction(start) / (rate * time_base)
            if target_pts.denominator != 1:
                raise ValueError("Requested RGB start is not an exact source timestamp")
            if start:
                container.seek(int(target_pts), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("Decoded frame has no PTS; nearest-frame fallback is forbidden")
                index = Fraction(frame.pts - stream.start_time) * time_base * rate
                if index.denominator != 1:
                    raise ValueError("Non-CFR frame timestamp; refusing approximate temporal alignment")
                index = int(index)
                if index < start:
                    continue  # Decode only the necessary preceding keyframe/GOP.
                if index != start + selected:
                    raise ValueError(f"Nonconsecutive RGB interval: expected {start + selected}, got {index}")
                if [frame.height, frame.width] != request["source_resolution_hw"]:
                    raise ValueError("Source resolution changed within the selected interval")
                rgb = frame.to_ndarray(format="rgb24")
                frames[selected] = rgb[y0:y1, x0:x1]
                selected += 1
                if selected == count:
                    break
        if selected != count:
            raise RuntimeError(f"Requested {count} exact RGB frames at {start}, decoded {selected}")
        self._verify_source(record)
        return torch.from_numpy(frames)

    def prepare_anchor(
        self, request: dict[str, Any], *, decode_threads: int = 0
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Decode and preprocess one exact Anchor entirely on CPU.

        The returned request-tagged tensor is safe to prepare concurrently, but
        contains no VAE result. It is shaped ``[1, 3, 65, H, W]`` in BF16 and
        uses the same native reflect-padding math as synchronous training misses.
        """
        self._validate_request(request)
        # Only 2K RGB is accumulated; full 4K frames are never VAE inputs.
        rgb = self._read_anchor_rgb(request, decode_threads=decode_threads).float().div_(255).clamp_(0, 1)
        padding = request["vae_padding"]
        top, left = padding["top"], padding["left"]
        with torch.inference_mode():
            # Native helper pads trailing sides only and skips interpolation.
            # Flipping first reuses its exact reflect operation on leading sides;
            # the offsets are determined by the existing canonical GT lattice.
            if top or left:
                leading = resize_and_reflect_pad(rgb.flip((1, 2)), rgb.shape[1] + top, rgb.shape[2] + left)
                rgb = leading[0].permute(1, 2, 3, 0).flip((1, 2))
            height, width = request["vae_input_hw"]
            video = resize_and_reflect_pad(rgb, height, width).mul_(2).sub_(1)
            if list(video.shape) != [1, 3, request["rgb_frame_count"], height, width]:
                raise RuntimeError("Official VAE input geometry differs from declared provenance")
        return request, video.to(dtype=torch.bfloat16).contiguous()

    def _validate_prepared(self, request: dict[str, Any], video: torch.Tensor) -> None:
        self._validate_request(request)
        height, width = request["vae_input_hw"]
        expected = (1, 3, request["rgb_frame_count"], height, width)
        if video.device.type != "cpu" or video.dtype != torch.bfloat16 or tuple(video.shape) != expected:
            raise ValueError(f"Prepared Anchor must be CPU BF16 {expected}, got {video.device}/{video.dtype}/{tuple(video.shape)}")
        minimum, maximum = torch.aminmax(video)
        if not torch.isfinite(torch.stack((minimum, maximum))).all().item() or minimum.item() < -1.0 or maximum.item() > 1.0:
            raise ValueError("Prepared Anchor contains nonfinite or out-of-range normalized pixels")

    def encode_prepared(
        self, prepared: list[tuple[dict[str, Any], torch.Tensor]]
    ) -> torch.Tensor:
        """Encode a request-tagged CPU batch through the one official VAE path."""
        if get_worker_info() is not None:
            raise RuntimeError("Anchor VAE encoding must not run inside a DataLoader worker")
        if not prepared:
            raise ValueError("encode_prepared requires at least one Anchor")
        record_ids = [request["gt_record_id"] for request, _ in prepared]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("A VAE batch cannot contain duplicate Anchor record IDs")
        for request, video in prepared:
            self._validate_prepared(request, video)
        shapes = {tuple(video.shape) for _, video in prepared}
        if len(shapes) != 1:
            raise ValueError("All prepared Anchors in a VAE batch must have identical shapes")
        video_batch = torch.cat([video for _, video in prepared], dim=0)
        with torch.inference_mode(), torch.random.fork_rng(
            devices=[self.device.index] if self.device.type == "cuda" else [],
        ):
            if self._encoder is None:
                self._encoder = load_video_vae_encoder(str(self.vae_path), device="cpu", dtype=torch.bfloat16)
                self._encoder.eval().requires_grad_(False)
            try:
                self._encoder.to(self.device)
                spatial = DimensionSizeConfig(tile_size=self.tile_size, overlap=self.tile_overlap)
                tiling = TileSizeConfig(height=spatial, width=spatial)
                latents = self._encoder.tiled_encode(video_batch, tiling).detach().cpu().contiguous()
            finally:
                if self.offload:
                    self._encoder.to("cpu")
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
        expected = (
            len(prepared), self.latent_channels, self.audit.clip_latent_frames,
            *prepared[0][0]["anchor_latent_hw"],
        )
        if tuple(latents.shape) != expected or latents.dtype != torch.bfloat16 or not torch.isfinite(latents).all().item():
            raise RuntimeError("Independent VAE batch differs from declared shape/dtype or is nonfinite")
        return latents

    def _encode(self, request: dict[str, Any]) -> torch.Tensor:
        """Compatibility single-item path used by synchronous training misses."""
        prepared = self.prepare_anchor(request)
        return self.encode_prepared([prepared])[0].clone()

    @staticmethod
    def _entry(request: dict[str, Any], latent: torch.Tensor) -> dict[str, Any]:
        return {
            "latents": latent,
            "num_frames": latent.shape[1],
            "height": latent.shape[2],
            "width": latent.shape[3],
            "fps": request["source_fps"],
            "provenance": request,
            "tensor_sha256": _tensor_digest(latent),
        }

    def _persist_locked(self, request: dict[str, Any], latent: torch.Tensor) -> tuple[dict[str, Any], bool]:
        path = self.entry_path(request)
        expected = (request["latent_channels"], request["latent_frame_count"], *request["anchor_latent_hw"])
        if tuple(latent.shape) != expected or latent.dtype != torch.bfloat16 or not torch.isfinite(latent).all().item():
            raise RuntimeError("Independent VAE result differs from declared shape/dtype or is nonfinite")
        entry = self._entry(request, latent)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                torch.save(entry, handle)
                handle.flush()
                os.fsync(handle.fileno())
            # Cooperating writers hold this same lock. Never replace an already
            # present entry, even if an external writer appeared.
            if path.exists():
                cached = self.read_cached(request)
                if cached is None:
                    raise RuntimeError(f"Anchor cache disappeared during competing write: {path}")
                return cached, False
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        persisted = self.read_cached(request)
        if persisted is None:
            raise RuntimeError(f"Anchor cache disappeared after atomic save: {path}")
        return persisted, True

    def get_or_create_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        prepared: list[tuple[dict[str, Any], torch.Tensor]] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve one offline VAE batch with ordered locks and atomic writes.

        This API enables offline batched precaching only. Training continues to
        call :meth:`get_or_create`, which invokes this method with exactly one
        request and therefore remains synchronous sample-by-sample.
        """
        if get_worker_info() is not None:
            raise RuntimeError("Anchor cache misses must be encoded by the owning process")
        if not requests:
            return []
        self._verify_checkpoint()
        paths = [self._validate_request(request) for request in requests]
        record_ids = [request["gt_record_id"] for request in requests]
        if len(paths) != len(set(paths)) or len(record_ids) != len(set(record_ids)):
            raise ValueError("get_or_create_batch requires unique Anchor record IDs")

        prepared_by_id: dict[str, tuple[dict[str, Any], torch.Tensor]] | None = None
        if prepared is not None:
            prepared_by_id = {}
            for prepared_request, video in prepared:
                record_id = prepared_request["gt_record_id"]
                if record_id in prepared_by_id:
                    raise ValueError("Prepared Anchor batch contains duplicate record IDs")
                prepared_by_id[record_id] = (prepared_request, video)
            if set(prepared_by_id) != set(record_ids):
                raise ValueError("Prepared Anchor identities differ from requested batch identities")
            for request in requests:
                prepared_request, video = prepared_by_id[request["gt_record_id"]]
                if prepared_request != request:
                    raise ValueError("Prepared Anchor provenance differs from its requested identity")
                self._validate_prepared(request, video)

        resolved: list[dict[str, Any] | None] = [self.read_cached(request) for request in requests]
        missing_indices = [index for index, entry in enumerate(resolved) if entry is None]
        if not missing_indices:
            return [entry for entry in resolved if entry is not None]
        for index in missing_indices:
            paths[index].parent.mkdir(parents=True, exist_ok=True)

        with ExitStack() as stack:
            for index in sorted(missing_indices, key=lambda value: str(paths[value])):
                lock = stack.enter_context(paths[index].with_suffix(".lock").open("a+b"))
                fcntl.flock(lock, fcntl.LOCK_EX)
                stack.callback(fcntl.flock, lock, fcntl.LOCK_UN)
            # A concurrent process may have completed entries while locks were acquired.
            missing_indices = []
            for index, request in enumerate(requests):
                if resolved[index] is None:
                    resolved[index] = self.read_cached(request)
                if resolved[index] is None:
                    missing_indices.append(index)
            if missing_indices:
                if prepared_by_id is None:
                    to_encode = [self.prepare_anchor(requests[index]) for index in missing_indices]
                else:
                    to_encode = [prepared_by_id[requests[index]["gt_record_id"]] for index in missing_indices]
                logger.info("Encoding %d independent Anchor cache misses as one VAE batch", len(to_encode))
                latents = self.encode_prepared(to_encode)
                for batch_index, request_index in enumerate(missing_indices):
                    entry, created = self._persist_locked(requests[request_index], latents[batch_index].clone())
                    resolved[request_index] = entry
                    self.created_count += int(created)
        if any(entry is None for entry in resolved):
            raise RuntimeError("Anchor batch resolution completed with an unresolved entry")
        return [entry for entry in resolved if entry is not None]

    def get_or_create(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.get_or_create_batch([request])[0]

    def complete_batch(self, batch: dict[str, Any]) -> None:
        """Place independent Anchor cells by canonical pixel mapping, never GT crop."""
        if get_worker_info() is not None:
            raise RuntimeError("Anchor batch completion belongs to the training rank")
        self._verify_checkpoint()
        requests, entries = batch.pop("_anchor_requests"), batch.pop("_anchor_entries")
        target = batch["video_latents"]["latents"]
        if len(requests) != target.shape[0] or len(entries) != target.shape[0]:
            raise ValueError("Anchor request count and target batch size differ")
        condition = torch.zeros_like(target)
        for index, (request, cached) in enumerate(zip(requests, entries, strict=True)):
            self._validate_request(request)  # Fail closed even for externally collated cache hits.
            target_meta = batch["video_latents"]
            if (target_meta["gt_record_id"][index] != request["gt_record_id"]
                    or int(target_meta["rgb_start_frame"][index]) != request["rgb_start_frame"]
                    or int(target_meta["latent_start_index"][index]) != request["latent_start_index"]):
                raise ValueError("Target/Anchor source or temporal interval mismatch")
            entry = cached if cached is not None else self.get_or_create(request)
            if entry["provenance"] != request:
                raise ValueError("Collated Anchor and target provenance differ")
            latent = entry["latents"]
            y0, x0, y1, x1 = request["canonical_latent_rect_yxyx"]
            if tuple(latent.shape) != (target.shape[1], target.shape[2], y1 - y0, x1 - x0):
                raise ValueError("Independent Anchor cannot map exactly into this target canvas")
            if (target.shape[-2], target.shape[-1]) != (self.audit.latent_height, self.audit.latent_width):
                raise ValueError("Target canvas differs from audited canonical geometry")
            condition[index, :, :, y0:y1, x0:x1] = latent.to(device=condition.device, dtype=condition.dtype)
        batch["anchor_latents"] = {**batch["video_latents"], "latents": condition}
        batch["anchor_provenance"] = requests
