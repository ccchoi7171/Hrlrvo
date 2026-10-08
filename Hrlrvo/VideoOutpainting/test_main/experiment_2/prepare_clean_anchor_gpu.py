"""GPU-batched dense foreground cleanup for Experiment 2's full 1025f anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn

HERE = Path(__file__).resolve().parent
SOURCE = Path("/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/autonomous_013000_20261006/inputs/longrange_960x540_1025f.mkv")
WEIGHTS = Path("/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/canonical_background_013000_20261007/semantic_masks/weights/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth")
O_TO_C = Path("/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/canonical_background_013000_20261007/foreground_preparation/registered_semantic_v2/original_to_canonical_1025x3x3_float32.npy")
C_TO_O = Path("/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/canonical_background_013000_20261007/foreground_preparation/registered_semantic_v2/canonical_to_original_1025x3x3_float32.npy")
UPPER_EQUIPMENT_MASKS = Path("/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/canonical_background_013000_20261007/input_controls/full_upper_equipment_removed_v1/upper_equipment_masks_1025x540x960_uint8.npy")
VALIDATED_DONOR_PLATE = Path("/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/canonical_background_013000_20261007/foreground_preparation/registered_semantic_v2/semantic_donor_plate.npy")
H200_UUID = "GPU-a01d425d-ffe8-26e3-c637-6c87a4cf0f38"
FRAMES, HEIGHT, WIDTH = 1025, 540, 960


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def verify_gpu() -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != H200_UUID:
        raise RuntimeError("Experiment 2 cleanup requires the authorized H200 UUID")
    line = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader", "--id=0"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    uuid, name = [item.strip() for item in line.split(",", 1)]
    if uuid != H200_UUID or name != "NVIDIA H200":
        raise RuntimeError(f"Unexpected GPU identity: {line}")
    return {"uuid": uuid, "name": name}


def decode_source(path: Path, destination: Path) -> np.memmap:
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.uint8, shape=(FRAMES, HEIGHT, WIDTH, 3)
    )
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (WIDTH, HEIGHT) or abs(float(stream.average_rate) - 24) > 1e-3:
            raise ValueError("Source must be 960x540@24")
        for frame in container.decode(stream):
            if count >= FRAMES:
                raise ValueError("Source contains extra frames")
            output[count] = frame.to_ndarray(format="rgb24")
            count += 1
    output.flush()
    if count != FRAMES:
        raise ValueError(f"Source frame count {count} != {FRAMES}")
    return output


def encode_ffv1(path: Path, frames) -> int:
    temporary = path.with_suffix(".tmp.mkv")
    count = 0
    with av.open(str(temporary), "w") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "bgr0"
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    os.replace(temporary, path)
    return count


def infer_dense_masks(model, source: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    masks = np.lib.format.open_memmap(
        HERE / "person_masks_dense_uint8.npy", mode="w+", dtype=np.uint8,
        shape=(FRAMES, HEIGHT, WIDTH),
    )
    kernel = np.ones((21, 21), np.uint8)
    with torch.inference_mode():
        for begin in range(0, FRAMES, batch_size):
            end = min(FRAMES, begin + batch_size)
            tensors = [
                torch.from_numpy(np.asarray(source[index])).permute(2, 0, 1).to(
                    device=device, dtype=torch.float32
                ).div_(255)
                for index in range(begin, end)
            ]
            predictions = model(tensors)
            for index, prediction in enumerate(predictions, begin):
                selected = (prediction["labels"] == 1) & (prediction["scores"] >= 0.60)
                if bool(selected.any()):
                    soft = prediction["masks"][selected, 0].amax(0)
                    mask = (soft >= 0.30).to(torch.uint8).cpu().numpy()
                    masks[index] = cv2.dilate(mask, kernel)
                else:
                    masks[index] = 0
            del tensors, predictions
            if begin % (batch_size * 16) == 0:
                masks.flush()
                print(json.dumps({"event": "dense_masks", "through": end - 1}), flush=True)
    masks.flush()
    return masks


def registered_arrays(source: np.ndarray, masks: np.ndarray, matrices: np.ndarray):
    registered = np.lib.format.open_memmap(
        HERE / "registered_source_uint8.npy", mode="w+", dtype=np.uint8,
        shape=(FRAMES, HEIGHT, WIDTH, 3),
    )
    registered_masks = np.lib.format.open_memmap(
        HERE / "registered_masks_uint8.npy", mode="w+", dtype=np.uint8,
        shape=(FRAMES, HEIGHT, WIDTH),
    )
    for index in range(FRAMES):
        registered[index] = cv2.warpPerspective(
            source[index], matrices[index], (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        registered_masks[index] = cv2.warpPerspective(
            masks[index], matrices[index], (WIDTH, HEIGHT),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        )
    registered.flush(); registered_masks.flush()
    return registered, registered_masks


def masked_normalized_donor(
    samples: torch.Tensor,
    invalid: torch.Tensor,
    zero_support_fallback: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lighting-normalized temporal median using only observed background pixels."""
    if samples.ndim != 4 or samples.shape[-1] != 3 or invalid.shape != samples.shape[:3]:
        raise ValueError("Donor samples/masks have incompatible shapes")
    if invalid.dtype != torch.bool or not torch.isfinite(samples).all():
        raise ValueError("Donor samples must be finite with boolean invalid masks")
    centers = []
    for frame, masked in zip(samples, invalid, strict=True):
        if bool((~masked).sum() == 0):
            raise ValueError("Every donor frame requires observed background")
        centers.append(
            torch.stack([frame[..., channel][~masked].median() for channel in range(3)])
        )
    centers = torch.stack(centers)
    target = centers.median(dim=0).values
    normalized = samples - centers[:, None, None, :] + target
    valid = ~invalid
    donors = normalized.masked_fill(~valid[..., None], torch.nan)
    donor = torch.nanmedian(donors, dim=0).values
    counts = valid.sum(dim=0)
    zero = counts == 0
    if bool(zero.any()):
        if (
            zero_support_fallback is None
            or zero_support_fallback.shape != donor.shape
            or not torch.isfinite(zero_support_fallback).all()
        ):
            raise ValueError("Zero-support donor pixels require a validated background fallback")
        donor = torch.where(zero[..., None], zero_support_fallback, donor)
    donor = donor.clamp(0, 255)
    deviations = (normalized - donor.unsqueeze(0)).abs().mean(-1)
    mad = torch.nanmedian(deviations.masked_fill(~valid, torch.nan), dim=0).values.nan_to_num(0)
    return donor, counts, mad


def gpu_clean_registered(
    registered: np.ndarray,
    masks: np.ndarray,
    upper_equipment: np.ndarray,
    validated_donor_plate: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    sample_indices = list(range(0, FRAMES, 4))
    samples = torch.from_numpy(np.asarray(registered[sample_indices])).to(
        device=device, dtype=torch.float32
    )
    sampled_masks = (
        torch.from_numpy(np.asarray(masks[sample_indices])).to(device).bool()
        | torch.from_numpy(np.asarray(upper_equipment[sample_indices])).to(device).bool()
    )
    # Normalize each observed background frame before the masked temporal donor.
    # This preserves per-frame lighting in the later offset fit without baking
    # performers/camera into the immutable donor plate.
    fallback = torch.from_numpy(np.asarray(validated_donor_plate)).to(
        device=device, dtype=torch.float32
    )
    donor, donor_count, mad = masked_normalized_donor(
        samples, sampled_masks, fallback
    )
    del samples, sampled_masks
    cleaned = np.lib.format.open_memmap(
        HERE / "clean_registered_uint8.npy", mode="w+", dtype=np.uint8,
        shape=(FRAMES, HEIGHT, WIDTH, 3),
    )
    combined = np.lib.format.open_memmap(
        HERE / "combined_masks_uint8.npy", mode="w+", dtype=np.uint8,
        shape=(FRAMES, HEIGHT, WIDTH),
    )
    offsets = []
    # Data-driven stable pixels; no fixed semantic ROI is erased.
    stable = mad < 10
    for begin in range(0, FRAMES, batch_size):
        end = min(FRAMES, begin + batch_size)
        frames = torch.from_numpy(np.asarray(registered[begin:end])).to(device=device, dtype=torch.float32)
        person = torch.from_numpy(np.asarray(masks[begin:end])).to(device=device).bool()
        known_equipment = torch.from_numpy(
            np.asarray(upper_equipment[begin:end])
        ).to(device=device).bool()
        valid = stable.unsqueeze(0) & ~person & ~known_equipment
        delta = frames - donor.unsqueeze(0)
        # Robust per-frame/channel fit on a deterministic sparse stable lattice.
        fit = valid[:, ::4, ::4]
        sampled = delta[:, ::4, ::4]
        offset_rows = []
        for row, keep in zip(sampled, fit, strict=True):
            offset_rows.append(torch.stack([row[..., c][keep].median() for c in range(3)]))
        offset = torch.stack(offset_rows).clamp(-32, 32)
        fitted = (donor.unsqueeze(0) + offset[:, None, None]).clamp(0, 255)
        source_luma = frames.mean(-1)
        donor_luma = fitted.mean(-1)
        residual = (frames - fitted).abs().mean(-1)
        # All-frame temporal residual proxy catches thin/dark moving equipment;
        # it is not semantic camera/staff ground truth.
        residual_equipment = (
            (mad.unsqueeze(0) >= 10)
            & (donor_luma - source_luma > 25)
            & (residual > 30)
            & ~person
        )
        # Previously observed, source-connected upper camera/staff evidence is
        # admitted independently of temporal MAD, which can be zero for brief passes.
        equipment = known_equipment | residual_equipment
        mask = person | equipment
        mask = F.max_pool2d(mask.float().unsqueeze(1), 13, stride=1, padding=6)
        alpha = F.avg_pool2d(mask,  nine := 9, stride=1, padding=nine // 2).squeeze(1).clamp(0, 1)
        result = frames * (1 - alpha[..., None]) + fitted * alpha[..., None]
        cleaned[begin:end] = result.round().to(torch.uint8).cpu().numpy()
        combined[begin:end] = (mask.squeeze(1) > 0).to(torch.uint8).cpu().numpy()
        offsets.extend(offset.cpu().tolist())
        del frames, person, known_equipment, delta, fitted, result, mask, alpha
    cleaned.flush(); combined.flush()
    return cleaned, {
        "donor_indices": sample_indices,
        "masked_color_normalized_temporal_donor": True,
        "donor_valid_count_min": int(donor_count.min().item()),
        "donor_valid_count_max": int(donor_count.max().item()),
        "donor_zero_count_pixels": int((donor_count == 0).sum().item()),
        "zero_support_fallback": {
            "path": str(VALIDATED_DONOR_PLATE),
            "sha256": digest(VALIDATED_DONOR_PLATE),
            "used_only_where_observed_donor_count_is_zero": True,
            "never_unmasked_foreground_temporal_median": True,
        },
        "gpu_temporal_median": True,
        "gpu_framewise_RGB_offset": True,
        "gpu_donor_fill": True,
        "RGB_offsets": offsets,
        "known_upper_equipment_masks": {
            "path": str(UPPER_EQUIPMENT_MASKS),
            "sha256": digest(UPPER_EQUIPMENT_MASKS),
            "coordinate": "registered canonical camera",
            "admission": "observed source-connected mask, independent of temporal MAD",
        },
        "equipment_proxy": "known upper observed mask OR all-frame dark temporal residual outside dense person masks; diagnostic, not semantic camera/staff GT",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 2 or args.batch_size > 8:
        raise ValueError("GPU mask batch size must be 2..8")
    targets = [HERE / name for name in ("source_rgb_uint8.npy", "person_masks_dense_uint8.npy", "clean_anchor_1025f.mkv", "preparation_manifest.json", "job.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Fresh Experiment 2 preparation outputs required")
    started = time.monotonic()
    gpu = verify_gpu()
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.17, 0)
    source = decode_source(SOURCE, HERE / "source_rgb_uint8.npy")
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    model.load_state_dict(state, strict=True); del state
    model.eval().to("cuda:0")
    masks = infer_dense_masks(model, source, args.batch_size, torch.device("cuda:0"))
    del model; torch.cuda.empty_cache()
    o_to_c = np.load(O_TO_C, mmap_mode="r", allow_pickle=False)
    c_to_o = np.load(C_TO_O, mmap_mode="r", allow_pickle=False)
    upper_equipment = np.load(UPPER_EQUIPMENT_MASKS, mmap_mode="r", allow_pickle=False)
    validated_donor = np.load(VALIDATED_DONOR_PLATE, mmap_mode="r", allow_pickle=False)
    if o_to_c.shape != (FRAMES, 3, 3) or c_to_o.shape != o_to_c.shape:
        raise ValueError("Camera registration lineage shape changed")
    if upper_equipment.shape != (FRAMES, HEIGHT, WIDTH) or upper_equipment.dtype != np.uint8:
        raise ValueError("Observed upper-equipment mask contract changed")
    if validated_donor.shape != (HEIGHT, WIDTH, 3) or validated_donor.dtype != np.uint8:
        raise ValueError("Validated semantic donor plate contract changed")
    registered, registered_masks = registered_arrays(source, masks, o_to_c)
    clean_registered, method = gpu_clean_registered(
        registered,
        registered_masks,
        upper_equipment,
        validated_donor,
        torch.device("cuda:0"),
        args.batch_size,
    )
    clean_path = HERE / "clean_anchor_1025f.mkv"
    def original_camera_frames():
        for index in range(FRAMES):
            yield cv2.warpPerspective(
                clean_registered[index], c_to_o[index], (WIDTH, HEIGHT),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )
    if encode_ffv1(clean_path, original_camera_frames()) != FRAMES:
        raise RuntimeError("Clean FFV1 frame count changed")
    manifest = {
        "status": "PASS_GPU_CLEAN_ANCHOR_READY",
        "source": {"path": str(SOURCE), "sha256": digest(SOURCE), "frames": FRAMES, "fps": 24},
        "clean_anchor": {"path": str(clean_path), "sha256": digest(clean_path), "frames": FRAMES, "width": WIDTH, "height": HEIGHT, "fps": 24, "codec": "FFV1"},
        "person_segmentation": {"model": "torchvision maskrcnn_resnet50_fpn", "weights": str(WEIGHTS), "weights_sha256": digest(WEIGHTS), "dense_all_frames": True, "batch_size": args.batch_size, "score_threshold": 0.60, "mask_threshold": 0.30},
        "cleanup": method,
        "registration": {"working_coordinate": "registered canonical camera", "return_coordinate": "original per-frame camera pose", "original_to_canonical": {"path": str(O_TO_C), "sha256": digest(O_TO_C)}, "canonical_to_original": {"path": str(C_TO_O), "sha256": digest(C_TO_O)}},
        "clean_anchor_is_model_observation": True,
        "original_RAW_anchor_post_reinsertion": False,
        "limitations": ["Equipment proxy is temporal/color evidence, not semantic camera or staff ground truth.", "Reflections, wires and small color residuals are diagnostic and do not abort the full run.", "Registration round-trip interpolation can soften cleaned pixels."],
        "gpu": {**gpu, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()},
        "runtime_seconds": time.monotonic() - started,
        "preparer_sha256": digest(Path(__file__)),
    }
    atomic_json(HERE / "preparation_manifest.json", manifest)
    job = {
        "name": "experiment_2_phase_rebase_full1025", "experiment": 2,
        "variant": "gpu_clean_anchor_phase_aligned_last2_temporal_rebase",
        "backend": "native", "config": str(HERE / "native_65f_resume.yaml"),
        "input": str(clean_path), "input_sha256": manifest["clean_anchor"]["sha256"],
        "output": str(HERE / "output/generated.mp4"),
        "checkpoint": "/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/hrlrvo_main_train_v1/checkpoints/checkpoint-013000",
        "seed": 33333333, "steps": 30, "frames": 1025, "memory_fraction": 0.49,
        "recurrence": {"reduced_geometry": True, "temporal_rebase": True},
        "attention": {"collect_branch_stats": False}, "prompt": "",
        "source_frame_start": 0, "source_frame_end_inclusive": 1024,
        "local_absolute_start_frame": 0, "restore_original_anchor_after_inference": False,
        "requires_inverse_registration": False, "latent_carry_mode": "exterior_last2_keep_native_row0",
        "producer_script": "run_full_phase_rebase.py",
        "clean_preparation_manifest": str(HERE / "preparation_manifest.json"),
        "clean_preparation_manifest_sha256": digest(HERE / "preparation_manifest.json"),
    }
    atomic_json(HERE / "job.json", job)
    print(json.dumps({"event": "experiment2_preparation_complete", "job": str(HERE / "job.json"), "clean_anchor": str(clean_path)}), flush=True)


if __name__ == "__main__":
    main()
