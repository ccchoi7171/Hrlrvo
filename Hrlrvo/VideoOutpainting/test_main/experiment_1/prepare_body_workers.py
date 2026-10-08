"""Prepare two parity body-worker payloads within Experiment 1's 70 GiB budget."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "jobs"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def payload(worker_id: int) -> dict[str, object]:
    paths = [JOBS / "bodies" / f"I{index:02d}.json" for index in range(worker_id, 14, 2)]
    if len(paths) != 7 or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"Parity worker {worker_id} does not have seven prepared body jobs")
    return {
        "schema": "hrlrvo_body_worker_v1",
        "worker_id": worker_id,
        "jobs": [str(path) for path in paths],
        "load_model_once": True,
        "torch_cuda_memory_fraction": 0.245,
        "memory_reservation_mib": 35840,
        "expected_peak_gib": 35.0,
        "aggregate_experiment_1_budget_gib": 70.0,
        "reference_manifest": str(ROOT / "reference_bank/reference_manifest.json"),
        "reference_alpha": 0.1,
        "recurrence": {
            "reduced_geometry": True,
            "reassert_anchor_latents": True,
            "temporal_rebase": True
        },
        "seed_policy": "33333333 + absolute body global_start; unaffected by temporal rebase",
    }


def main() -> None:
    destinations = []
    for worker_id in (0, 1):
        destination = JOBS / "body_workers" / f"worker_{worker_id}.json"
        atomic_json(destination, payload(worker_id))
        destinations.append(str(destination))
    profile = payload(0)
    profile["jobs"] = profile["jobs"][:1]
    profile["profile_first_body_only"] = True
    profile_destination = JOBS / "body_workers" / "profile_worker_0_I00.json"
    atomic_json(profile_destination, profile)
    print(json.dumps({"event": "body_workers_prepared", "workers": destinations}))


if __name__ == "__main__":
    main()
