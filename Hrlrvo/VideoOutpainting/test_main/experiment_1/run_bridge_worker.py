"""Run a sequence of independent 65f bridge contexts with one model load."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent
TEST_ROOT = ROOT.parent
COMMON = TEST_ROOT / "common"
COMMON_SOURCE = COMMON / "source"
PY_SOURCE = COMMON_SOURCE / "ltx_outpaint/inference.py"
EXPECTED_SHAPE = (1025, 540, 960, 3)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_h200(expected_uuid: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError("Bridge worker must be bound to the authorized H200 UUID")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if expected_uuid not in result or "NVIDIA H200" not in result:
        raise RuntimeError("Physical H200 binding was not confirmed")


def source_context(source: np.ndarray, bridge: dict) -> torch.Tensor:
    start, stop = int(bridge["context_start"]), int(bridge["context_stop"])
    real_start, real_stop = max(0, start), min(len(source), stop)
    pieces = []
    if start < 0:
        pieces.append(np.repeat(source[:1], -start, axis=0))
    pieces.append(np.asarray(source[real_start:real_stop]))
    if stop > len(source):
        pieces.append(np.repeat(source[-1:], stop - len(source), axis=0))
    joined = np.concatenate(pieces, axis=0)
    if joined.shape != (65, 540, 960, 3) or joined.dtype != np.uint8:
        raise RuntimeError(f"Invalid bridge source context: {joined.shape}, {joined.dtype}")
    return torch.from_numpy(np.ascontiguousarray(joined)).permute(0, 3, 1, 2).float().div_(255.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args()
    worker = json.loads(args.worker.read_text())
    if worker.get("schema") != "hrlrvo_bridge_worker_v1" or not worker.get("jobs"):
        raise ValueError("Invalid or empty bridge worker file")
    jobs = [json.loads(Path(path).read_text()) for path in worker["jobs"]]
    memory_fraction = float(worker.get("torch_cuda_memory_fraction", 0.245))
    if memory_fraction != 0.245:
        raise ValueError("Bridge worker memory fraction must remain at the scheduled 0.245 cap")
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    expected_uuid = jobs[0]["h200_uuid"]
    validate_h200(expected_uuid)
    if any(job["h200_uuid"] != expected_uuid for job in jobs):
        raise ValueError("Bridge worker mixes GPU bindings")
    config_path = Path(jobs[0]["config"])
    checkpoint = Path(jobs[0]["checkpoint"])
    source_path = Path(jobs[0]["source_rgb"])
    for job in jobs:
        if (
            job.get("schema") != "hrlrvo_bridge_job_v1"
            or job["seed"] != 33333333
            or job["steps"] != 30
            or job["local_absolute_start_frame"] != 0
            or job["reference_conditioning"] is not False
        ):
            raise ValueError("Bridge jobs must use the frozen independent65 no-reference contract")
        for field, path in (("config_sha256", config_path), ("adapter_sha256", checkpoint / "adapter_model.safetensors"), ("source_rgb_sha256", source_path)):
            if job[field] != digest(path):
                raise RuntimeError(f"Bridge job provenance mismatch: {field}")

    source = np.load(source_path, mmap_mode="r")
    if source.shape != EXPECTED_SHAPE or source.dtype != np.uint8:
        raise RuntimeError(f"Unexpected source cache: {source.shape}, {source.dtype}")

    sys.path.insert(0, str(COMMON_SOURCE))
    sys.path.insert(0, str(COMMON))
    from ltx_outpaint import inference
    if Path(inference.__file__).resolve() != PY_SOURCE:
        raise RuntimeError("Original inference source imported instead of test_main copy")
    settings = yaml.safe_load(config_path.read_text())
    inference._install_official_sources(Path(settings["ltx_repo"]))
    from ltx_trainer.model_loader import read_video_scale_factors
    from ltx_core.components.noisers import GaussianNoiser
    from recurrence import install as install_recurrence

    install_recurrence(inference, {"reduced_geometry": True})
    vae_path = settings["model"].get("video_vae_path") or settings["model"]["base_model_path"]
    scale_factors = read_video_scale_factors(Path(vae_path))
    boundary = inference._vae_boundary(1080, 1920, scale_factors.height, scale_factors.width)
    anchor_rect = inference._center_anchor(boundary, 540, 960)
    validation = inference._make_validation_config(settings, "", boundary, 65)
    validation.seed = 33333333
    validation.inference_steps = 30
    memory = inference._memory_optimization_settings(settings)
    started = time.monotonic()
    transformer, runner, encoder, adapter_report = inference._load_models(
        settings,
        validation,
        boundary,
        checkpoint_path=checkpoint,
        memory_optimization=memory,
    )
    model_load_seconds = time.monotonic() - started
    print(json.dumps({"event": "bridge_worker_model_ready", "jobs": len(jobs), "model_load_seconds": model_load_seconds}), flush=True)

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
        print(json.dumps({"event": "bridge_noise_captured", "bridge_id": active.get("bridge_id"), **active["noise"]}), flush=True)
        return noise

    GaussianNoiser._sample_noise = observed_noise
    worker_status = ROOT / "status" / f"bridge_worker_{worker['worker']}.json"
    state = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "worker": worker["worker"],
        "jobs": len(jobs),
        "model_load_seconds": model_load_seconds,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed": [],
    }
    atomic_json(worker_status, state)
    try:
        for ordinal, job in enumerate(jobs):
            bridge = job["bridge"]
            bridge_id = bridge["bridge_id"]
            output_dir = Path(job["output_dir"])
            status_path = output_dir / "status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("status") == "PASS":
                state["completed"].append({"bridge_id": bridge_id, "status": "ALREADY_PASS"})
                atomic_json(worker_status, state)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            bridge_state = {
                "status": "RUNNING",
                "pid": os.getpid(),
                "bridge": bridge,
                "job": job,
                "worker": worker["worker"],
                "worker_job_ordinal": ordinal,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(status_path, bridge_state)
            active.clear()
            active["bridge_id"] = bridge_id
            anchor = source_context(source, bridge)
            torch.cuda.reset_peak_memory_stats()
            sample_started = time.monotonic()
            decoded = inference.sample_chunk(
                transformer=transformer,
                runner=runner,
                vae_encoder=encoder,
                anchor_frames=anchor,
                previous_prefix=None,
                boundary=boundary,
                anchor_rect=anchor_rect,
                absolute_start_frame=0,
                boundary_blend_width_pixels=0,
                allow_nonempty_text=True,
                vae_decode_streaming=bool(memory["vae_decode_streaming"]),
            )
            if tuple(decoded.shape) != (65, 3, 1080, 1920) or decoded.dtype != torch.float32:
                raise RuntimeError(f"Unexpected decoded bridge context: {tuple(decoded.shape)}, {decoded.dtype}")
            full = decoded.cpu().contiguous().numpy()
            extract_start, extract_stop = int(bridge["extract_start"]), int(bridge["extract_stop"])
            bridge_rgb = np.ascontiguousarray(full[extract_start:extract_stop])
            np.save(output_dir / "context_65f_fp32.npy", full, allow_pickle=False)
            np.save(output_dir / "bridge_9f_fp32.npy", bridge_rgb, allow_pickle=False)
            bridge_state.update(
                status="PASS",
                ended_utc=datetime.now(timezone.utc).isoformat(),
                sample_seconds=time.monotonic() - sample_started,
                noise=active.get("noise"),
                context_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
                bridge_sha256=hashlib.sha256(bridge_rgb.tobytes()).hexdigest(),
                source_padding={"left": bridge["pad_left"], "right": bridge["pad_right"]},
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                adapter_report=adapter_report,
            )
            atomic_json(status_path, bridge_state)
            state["completed"].append({"bridge_id": bridge_id, "status": "PASS", "sample_seconds": bridge_state["sample_seconds"]})
            atomic_json(worker_status, state)
            del decoded, anchor, full, bridge_rgb
            torch.cuda.empty_cache()
        state.update(status="PASS", ended_utc=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.monotonic() - started)
    except BaseException as error:
        state.update(status="FAIL", error=str(error), traceback=traceback.format_exc(), ended_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        GaussianNoiser._sample_noise = native_noise
        atomic_json(worker_status, state)


if __name__ == "__main__":
    main()
