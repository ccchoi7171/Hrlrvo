"""Prepare immutable bridge/body job records without launching a GPU process."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from layout import body_plans, bridge_plans, validate_contract

ROOT = Path(__file__).resolve().parent
TEST_ROOT = ROOT.parent
SOURCE_RGB = Path(
    "/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/"
    "canonical_background_013000_20261007/foreground_preparation/source_rgb.npy"
)
B0_REUSE = Path(
    "/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/diagnostics/"
    "longrange_bridge_013000_20261007/outputs/independent_start_0000/generated_65f_fp32.npy"
)
CHECKPOINT = Path(
    "/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/"
    "hrlrvo_main_train_v1/checkpoints/checkpoint-013000"
)
CONFIG_65 = TEST_ROOT / "common/native_65f.yaml"
H200_UUID = "GPU-a01d425d-ffe8-26e3-c637-6c87a4cf0f38"
SEED = 33333333


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_fresh_json(path: Path, value: object) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != value:
            raise FileExistsError(f"Refusing to replace a different prepared artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-workers", type=int, default=2)
    args = parser.parse_args()
    if args.bridge_workers != 2:
        raise ValueError("The authorized H200 memory budget is prepared for exactly two bridge workers")
    validate_contract()
    for required in (SOURCE_RGB, B0_REUSE, CHECKPOINT / "adapter_model.safetensors", CONFIG_65):
        if not required.exists():
            raise FileNotFoundError(required)

    bridges = bridge_plans(reuse_b0=B0_REUSE)
    bodies = body_plans()
    plan_path = ROOT / "plan.json"
    prepared_utc = (
        json.loads(plan_path.read_text()).get("prepared_utc")
        if plan_path.exists()
        else datetime.now(timezone.utc).isoformat()
    )
    plan = {
        "schema": "hrlrvo_immutable_bridge_v1",
        "prepared_utc": prepared_utc,
        "output_frames": 1025,
        "fps": 24,
        "seed": SEED,
        "steps": 30,
        "bridge_context_frames": 65,
        "body_operator": {"left": 9, "generated_middle": 64, "right": 8, "total": 81},
        "source_rgb": str(SOURCE_RGB),
        "source_rgb_sha256": digest(SOURCE_RGB),
        "checkpoint": str(CHECKPOINT),
        "adapter_sha256": digest(CHECKPOINT / "adapter_model.safetensors"),
        "config_65": str(CONFIG_65),
        "config_65_sha256": digest(CONFIG_65),
        "h200_uuid": H200_UUID,
        "bridge_reference_conditioning": False,
        "body_reference_conditioning": True,
        "bridge_workers": args.bridge_workers,
        "estimated_peak_gib_per_bridge_worker": 33.5,
        "estimated_combined_peak_gib": 67.0,
        "bridges": [row.to_dict() for row in bridges],
        "bodies": [row.to_dict() for row in bodies],
    }
    write_fresh_json(plan_path, plan)

    pending = []
    for row in bridges:
        if row.index == 0:
            continue
        output_dir = ROOT / "bridge_bank" / row.bridge_id
        job = {
            "schema": "hrlrvo_bridge_job_v1",
            "bridge": row.to_dict(),
            "source_rgb": str(SOURCE_RGB),
            "source_rgb_sha256": plan["source_rgb_sha256"],
            "config": str(CONFIG_65),
            "config_sha256": plan["config_65_sha256"],
            "checkpoint": str(CHECKPOINT),
            "adapter_sha256": plan["adapter_sha256"],
            "output_dir": str(output_dir),
            "seed": SEED,
            "steps": 30,
            "local_absolute_start_frame": 0,
            "reference_conditioning": False,
            "h200_uuid": H200_UUID,
        }
        job_path = ROOT / "jobs/bridges" / f"{row.bridge_id}.json"
        write_fresh_json(job_path, job)
        pending.append(str(job_path))

    batches = [pending[index::args.bridge_workers] for index in range(args.bridge_workers)]
    for worker, jobs in enumerate(batches):
        write_fresh_json(
            ROOT / "jobs/bridge_workers" / f"worker_{worker}.json",
            {
                "schema": "hrlrvo_bridge_worker_v1",
                "worker": worker,
                "jobs": jobs,
                "expected_peak_gib": 33.5,
                "torch_cuda_memory_fraction": 0.245,
                "load_model_once": True,
            },
        )

    for row in bodies:
        write_fresh_json(
            ROOT / "jobs/bodies" / f"{row.body_id}.json",
            {
                "schema": "hrlrvo_body_job_v1",
                "body": row.to_dict(),
                "source_rgb": str(SOURCE_RGB),
                "source_rgb_sha256": plan["source_rgb_sha256"],
                "checkpoint": str(CHECKPOINT),
                "adapter_sha256": plan["adapter_sha256"],
                "seed": SEED,
                "steps": 30,
                "reference_conditioning": True,
                "output_dir": str(ROOT / "bodies" / row.body_id),
                "h200_uuid": H200_UUID,
            },
        )
    from prepare_body_workers import payload as body_worker_payload

    for worker_id in (0, 1):
        write_fresh_json(
            ROOT / "jobs/body_workers" / f"worker_{worker_id}.json",
            body_worker_payload(worker_id),
        )
    print(json.dumps({"event": "experiment_1_jobs_prepared", "bridge_jobs": len(pending), "body_jobs": len(bodies), "plan": str(ROOT / "plan.json")}))


if __name__ == "__main__":
    main()
