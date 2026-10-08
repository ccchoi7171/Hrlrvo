"""Generate all fourteen independent reference-conditioned 9+64+8 bodies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from body_conditioning import EndpointConditioner, endpoint_mask_contract
from PIL import Image

ROOT = Path(__file__).resolve().parent
TEST_ROOT = ROOT.parent
COMMON = TEST_ROOT / "common"
COMMON_SOURCE = COMMON / "source"
EXPECTED_SOURCE_SHAPE = (1025, 540, 960, 3)
REFERENCE_MANIFEST = ROOT / "reference_bank/reference_manifest.json"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_h200(expected_uuid: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError("Body worker must be bound to the authorized H200 UUID")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if expected_uuid not in result or "NVIDIA H200" not in result:
        raise RuntimeError("Physical H200 binding was not confirmed")


def bridge_paths(bridge_id: str, plan: dict) -> tuple[Path, int, int]:
    row = next(item for item in plan["bridges"] if item["bridge_id"] == bridge_id)
    if bridge_id == "B00":
        full = Path(row["reuse_full_context"])
    else:
        full = ROOT / "bridge_bank" / bridge_id / "context_65f_fp32.npy"
    return full, int(row["extract_start"]), int(row["extract_stop"])


def load_bridge(bridge_id: str, plan: dict) -> tuple[torch.Tensor, torch.Tensor]:
    full_path, begin, end = bridge_paths(bridge_id, plan)
    if bridge_id == "B00":
        manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
        if full_path.resolve() != Path(manifest["source"]).resolve() or digest(full_path) != manifest["source_sha256"]:
            raise RuntimeError("B00 bridge differs from the fixed reference-bank source")
        expected_tensor_sha256 = None
    else:
        status_path = ROOT / "bridge_bank" / bridge_id / "status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"Bridge status is missing: {status_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "PASS" or status.get("bridge", {}).get("bridge_id") != bridge_id:
            raise RuntimeError(f"Bridge has not completed with matching PASS provenance: {bridge_id}")
        expected_tensor_sha256 = status.get("context_sha256")
        if not isinstance(expected_tensor_sha256, str) or len(expected_tensor_sha256) != 64:
            raise RuntimeError(f"Bridge PASS status lacks a context hash: {bridge_id}")
    full_array = np.load(full_path, mmap_mode="r")
    if full_array.shape != (65, 3, 1080, 1920) or full_array.dtype != np.float32:
        raise RuntimeError(f"Invalid full bridge context {bridge_id}: {full_array.shape}, {full_array.dtype}")
    full = torch.from_numpy(np.array(full_array, copy=True))
    if expected_tensor_sha256 is not None and tensor_digest(full) != expected_tensor_sha256:
        raise RuntimeError(f"Bridge tensor bytes differ from PASS status: {bridge_id}")
    bridge = full[begin:end].clone()
    if tuple(bridge.shape) != (9, 3, 1080, 1920):
        raise RuntimeError(f"Invalid bridge extraction for {bridge_id}")
    return full, bridge


def source_interval(source: np.ndarray, start: int, count: int) -> torch.Tensor:
    stop = start + count
    real_stop = min(len(source), stop)
    pieces = [np.asarray(source[start:real_stop])]
    if stop > len(source):
        pieces.append(np.repeat(source[-1:], stop - len(source), axis=0))
    joined = np.concatenate(pieces, axis=0)
    if joined.shape != (count, 540, 960, 3) or joined.dtype != np.uint8:
        raise RuntimeError(f"Invalid source interval at {start}: {joined.shape}, {joined.dtype}")
    return torch.from_numpy(np.ascontiguousarray(joined)).permute(0, 3, 1, 2).float().div_(255.0)


def load_reference_records():
    from reference_memory import ReferenceArtifact, validate_reference_artifacts

    payload = json.loads(REFERENCE_MANIFEST.read_text())
    if payload.get("schema") != "hrlrvo_four_image_reference_memory_v1" or payload.get("reference_count") != 4:
        raise ValueError("Invalid four-image reference manifest")
    fields = ("path", "sha256", "source_frame", "vae_encoded_separately")
    records = validate_reference_artifacts(tuple(
        ReferenceArtifact(**{field: row[field] for field in fields})
        for row in payload["artifacts"]
    ))
    return payload, records


def build_reference_cache(*, inference, transformer, runner, encoder, boundary, scale_factors, handle, records):
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape
    from ltx_outpaint.geometry import CanvasGeometry, Rect
    from ltx_outpaint.model import attach_geometry

    device = next(transformer.parameters()).device
    dtype = next(transformer.parameters()).dtype
    encoded = []
    for record in records:
        image = np.asarray(Image.open(record.path).convert("RGB"))
        if image.shape != (1080, 1920, 3) or image.dtype != np.uint8:
            raise RuntimeError(f"Reference must be full-canvas uint8 RGB: {record.path}")
        frame = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        padded = inference._reflect_pad_rgb(
            frame,
            top=boundary.top,
            bottom=boundary.encoded_height - boundary.bottom,
            left=boundary.left,
            right=boundary.encoded_width - boundary.right,
        )
        latent = inference._encode_condition_rgb(padded, encoder, runner, device)
        if tuple(latent.shape) != (1, 128, 1, 36, 60):
            raise RuntimeError(f"Reference did not encode as one full-canvas latent: {tuple(latent.shape)}")
        encoded.append(latent.to(device=device, dtype=dtype))
    references = torch.cat(encoded, dim=0)
    shape = VideoLatentShape.from_torch_shape(references.shape)
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=shape,
        fps=runner._config.frame_rate,
        scale_factors=scale_factors,
        causal_fix=True,
    )
    state = tools.create_initial_state(device=device, dtype=dtype, initial_latent=references)
    embeddings = runner._cached_embeddings[0]
    v_pos, _a_pos, _v_neg, _a_neg = runner._get_prompt_embeddings(embeddings, device)
    context = v_pos.expand(4, *v_pos.shape[1:]).contiguous()
    sigma = torch.zeros(4, device=device, dtype=torch.float32)
    modality = Modality(
        enabled=True,
        latent=state.latent,
        sigma=sigma,
        timesteps=torch.zeros_like(state.denoise_mask[..., 0]),
        positions=state.positions.expand(4, -1, -1, -1),
        context=context,
        context_mask=None,
        keyframes_mask=state.keyframes_mask,
    )
    modality = attach_geometry(
        modality,
        CanvasGeometry(frames=1, height=36, width=60, anchors=(Rect(0, 0, 36, 60),)),
    )

    calls = {"count": 0}

    def clean_stream():
        calls["count"] += 1
        if calls["count"] != 1:
            raise RuntimeError("Reference stream callback must run exactly once")
        return transformer(video=modality, audio=None, perturbations=None)

    report = handle.build_cache(clean_stream, expected_reference_count=4)
    if calls["count"] != 1:
        raise RuntimeError("Reference cache did not use exactly one clean frozen forward")
    report["encoded_reference_latent_sha256"] = tensor_digest(references)
    del references, encoded, state
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args()
    worker = json.loads(args.worker.read_text())
    jobs_raw = worker.get("jobs", [])
    if worker.get("schema") != "hrlrvo_body_worker_v1" or not 1 <= len(jobs_raw) <= 14:
        raise ValueError("Body worker must contain between one and fourteen independent intervals")
    worker_id = worker.get("worker_id")
    if type(worker_id) is not int or worker_id not in (0, 1):
        raise ValueError("Body worker_id must be parity lane 0 or 1")
    memory_fraction = float(worker.get("torch_cuda_memory_fraction", -1))
    if memory_fraction != 0.245 or int(worker.get("memory_reservation_mib", -1)) != 35840:
        raise ValueError("Each body worker is capped to the authorized 35 GiB reservation")
    recurrence_spec = worker.get("recurrence")
    if recurrence_spec != {
        "reduced_geometry": True,
        "reassert_anchor_latents": True,
        "temporal_rebase": True,
    }:
        raise ValueError("Body recurrence must retain temporal_rebase and Anchor reassertion")
    if worker.get("load_model_once") is not True:
        raise ValueError("Body worker must reuse one loaded model across its assigned jobs")
    manifest_path = Path(worker.get("reference_manifest", "")).expanduser().resolve()
    if manifest_path != REFERENCE_MANIFEST.resolve():
        raise ValueError("Body worker must use the fixed Experiment 1 reference manifest")
    reference_alpha = float(worker.get("reference_alpha", -1))
    if reference_alpha != 0.1:
        raise ValueError("Experiment 1 reference alpha is fixed at 0.1")
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    jobs = [json.loads(Path(path).read_text()) for path in worker["jobs"]]
    validate_h200(jobs[0]["h200_uuid"])
    plan = json.loads((ROOT / "plan.json").read_text())
    source_path = Path(jobs[0]["source_rgb"])
    if digest(source_path) != jobs[0]["source_rgb_sha256"]:
        raise RuntimeError("Body source cache changed after preparation")
    source = np.load(source_path, mmap_mode="r")
    if source.shape != EXPECTED_SOURCE_SHAPE or source.dtype != np.uint8:
        raise RuntimeError(f"Unexpected source cache: {source.shape}, {source.dtype}")
    expected_checkpoint = Path(jobs[0]["checkpoint"]).resolve()
    expected_adapter_sha256 = digest(expected_checkpoint / "adapter_model.safetensors")
    for job in jobs:
        if job.get("schema") != "hrlrvo_body_job_v1" or job["reference_conditioning"] is not True:
            raise ValueError("Body job does not match the reference-conditioned experiment")
        if job["seed"] != 33333333 or job["steps"] != 30:
            raise ValueError("Body sampling settings changed")
        body_index = int(job["body"]["index"])
        if body_index % 2 != worker_id:
            raise ValueError("Body job is assigned to the wrong parity worker")
        if Path(job["source_rgb"]).resolve() != source_path.resolve() or job["source_rgb_sha256"] != jobs[0]["source_rgb_sha256"]:
            raise ValueError("Body worker mixes source artifacts")
        checkpoint = Path(job["checkpoint"])
        if checkpoint.resolve() != expected_checkpoint or job["adapter_sha256"] != expected_adapter_sha256:
            raise RuntimeError("Body job checkpoint/adapter provenance mismatch")
        for bridge_id in (job["body"]["left_bridge"], job["body"]["right_bridge"]):
            path, _, _ = bridge_paths(bridge_id, plan)
            if not path.exists():
                raise FileNotFoundError(f"Immutable bridge dependency is missing: {path}")

    sys.path.insert(0, str(COMMON_SOURCE))
    sys.path.insert(0, str(COMMON))
    from ltx_outpaint import inference
    if Path(inference.__file__).resolve() != COMMON_SOURCE / "ltx_outpaint/inference.py":
        raise RuntimeError("Original inference source imported instead of test_main copy")
    config_path = TEST_ROOT / "common/native_65f.yaml"
    settings = yaml.safe_load(config_path.read_text())
    inference._install_official_sources(Path(settings["ltx_repo"]))
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_trainer.model_loader import read_video_scale_factors
    from recurrence import install as install_recurrence
    from reference_memory import install_reference_memory

    recurrence_context = install_recurrence(inference, recurrence_spec)
    vae_path = settings["model"].get("video_vae_path") or settings["model"]["base_model_path"]
    scale_factors = read_video_scale_factors(Path(vae_path))
    boundary = inference._vae_boundary(1080, 1920, scale_factors.height, scale_factors.width)
    anchor_rect = inference._center_anchor(boundary, 540, 960)
    validation = inference._make_validation_config(settings, "", boundary, 81)
    validation.seed = 33333333
    validation.inference_steps = 30
    memory = inference._memory_optimization_settings(settings)
    checkpoint = Path(jobs[0]["checkpoint"])
    started = time.monotonic()
    transformer, runner, encoder, adapter_report = inference._load_models(
        settings, validation, boundary, checkpoint_path=checkpoint, memory_optimization=memory
    )
    model_load_seconds = time.monotonic() - started
    reference_manifest, reference_records = load_reference_records()
    reference_handle = install_reference_memory(transformer, alpha=reference_alpha)
    cache_report = build_reference_cache(
        inference=inference,
        transformer=transformer,
        runner=runner,
        encoder=encoder,
        boundary=boundary,
        scale_factors=scale_factors,
        handle=reference_handle,
        records=reference_records,
    )
    endpoint = EndpointConditioner(inference, runner, reference_handle)
    endpoint.install()
    native_noise = GaussianNoiser._sample_noise
    active: dict[str, object] = {}

    def observed_noise(self, state):
        noise = native_noise(self, state)
        active["noise"] = {
            "generator_seed": self.generator.initial_seed(),
            "shape": list(noise.shape),
            "dtype": str(noise.dtype),
            "sha256": tensor_digest(noise),
        }
        print(json.dumps({"event": "body_noise_captured", "body_id": active.get("body_id"), **active["noise"]}), flush=True)
        return noise

    GaussianNoiser._sample_noise = observed_noise

    class Progress:
        def __init__(self):
            self.step = 0
            self.started = time.monotonic()

        def advance_step(self):
            self.step += 1
            if self.step % 5 == 0:
                print(json.dumps({"event": "body_denoising_progress", "body_id": active.get("body_id"), "step": self.step, "elapsed_seconds": time.monotonic() - self.started}), flush=True)

    inference._SamplingProgress = Progress
    status_path = ROOT / "status" / f"body_worker_{worker_id}.json"
    state = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "worker_id": worker_id,
        "job_count": len(jobs),
        "memory_fraction": memory_fraction,
        "memory_reservation_mib": 35840,
        "recurrence": recurrence_spec,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model_load_seconds": model_load_seconds,
        "reference_manifest_sha256": digest(REFERENCE_MANIFEST),
        "reference_manifest": reference_manifest,
        "reference_cache": cache_report,
        "adapter_report": adapter_report,
        "completed": [],
    }
    atomic_json(status_path, state)
    try:
        for ordinal, job in enumerate(jobs):
            body = job["body"]
            body_id = body["body_id"]
            output_dir = Path(job["output_dir"])
            body_status_path = output_dir / "status.json"
            if body_status_path.exists() and json.loads(body_status_path.read_text()).get("status") == "PASS":
                state["completed"].append({"body_id": body_id, "status": "ALREADY_PASS"})
                atomic_json(status_path, state)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            body_state = {
                "status": "RUNNING",
                "pid": os.getpid(),
                "ordinal": ordinal,
                "worker_id": worker_id,
                "body": body,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(body_status_path, body_state)
            active.clear()
            active["body_id"] = body_id
            left_full, left = load_bridge(body["left_bridge"], plan)
            right_full, right = load_bridge(body["right_bridge"], plan)
            right_row = next(item for item in plan["bridges"] if item["bridge_id"] == body["right_bridge"])
            suffix_stop = int(right_row["extract_start"]) + 8
            suffix_context = right_full[:suffix_stop]
            if tuple(suffix_context.shape) != (33, 3, 1080, 1920):
                raise RuntimeError("Right bridge suffix context lost its causal VAE phase")
            anchor = source_interval(source, int(body["global_start"]), 81)
            torch.cuda.reset_peak_memory_stats()
            sample_started = time.monotonic()
            with endpoint.body(suffix_context) as capture:
                decoded = inference.sample_chunk(
                    transformer=transformer,
                    runner=runner,
                    vae_encoder=encoder,
                    anchor_frames=anchor,
                    previous_prefix=left,
                    boundary=boundary,
                    anchor_rect=anchor_rect,
                    absolute_start_frame=int(body["global_start"]),
                    boundary_blend_width_pixels=0,
                    allow_nonempty_text=True,
                    vae_decode_streaming=bool(memory["vae_decode_streaming"]),
                )
                if capture.known_mask is None or capture.anchor_rect is None:
                    raise RuntimeError("Endpoint capture was incomplete")
                endpoint_mask_contract(capture.known_mask, capture.anchor_rect)
            if tuple(decoded.shape) != (81, 3, 1080, 1920) or decoded.dtype != torch.float32:
                raise RuntimeError(f"Unexpected body output: {tuple(decoded.shape)}, {decoded.dtype}")
            if not torch.equal(decoded[:9], left):
                raise RuntimeError("Decoded body did not preserve the immutable left bridge")
            # Native sampling returns an inference tensor. Assemble immutable
            # RGB endpoints without mutating it outside torch.inference_mode.
            decoded = torch.cat((decoded[:-8], right[:8]), dim=0)
            # Stored bridge center pixels must be the same observed source RGB.
            y0, x0, _y1, _x1 = anchor_rect
            visible_y0, visible_x0 = y0 - boundary.top, x0 - boundary.left
            source_right = anchor[-8:]
            right_anchor = decoded[-8:, :, visible_y0:visible_y0 + 540, visible_x0:visible_x0 + 960]
            if not torch.equal(right_anchor, source_right):
                raise RuntimeError("Right bridge RGB conflicts with the current original Anchor")
            full = decoded.cpu().contiguous().numpy()
            np.save(output_dir / "generated_81f_fp32.npy", full, allow_pickle=False)
            body_state.update(
                status="PASS",
                ended_utc=datetime.now(timezone.utc).isoformat(),
                sample_seconds=time.monotonic() - sample_started,
                noise=active.get("noise"),
                output_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
                suffix_latent_sha256=capture.suffix_latent_sha256,
                known_mask_sha256=tensor_digest(capture.known_mask),
                reference_cache_sha256=cache_report["cache_sha256"],
                recurrence_events=list(recurrence_context["events"]),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            atomic_json(body_status_path, body_state)
            state["completed"].append({"body_id": body_id, "status": "PASS", "sample_seconds": body_state["sample_seconds"]})
            atomic_json(status_path, state)
            del decoded, full, anchor, left, right, left_full, right_full, suffix_context
            torch.cuda.empty_cache()
        state.update(status="PASS", ended_utc=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.monotonic() - started)
    except BaseException as error:
        state.update(status="FAIL", error=str(error), traceback=traceback.format_exc(), ended_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        GaussianNoiser._sample_noise = native_noise
        endpoint.remove()
        reference_handle.remove()
        atomic_json(status_path, state)


if __name__ == "__main__":
    main()
