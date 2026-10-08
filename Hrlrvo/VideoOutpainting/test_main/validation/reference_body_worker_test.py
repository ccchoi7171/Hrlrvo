from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1] / "experiment_1"
sys.path.insert(0, str(ROOT))

from body_conditioning import endpoint_mask_contract
from prepare_body_workers import payload
from run_body_worker import load_reference_records


def test_actual_reference_manifest_parses_only_contract_fields():
    manifest, records = load_reference_records()
    assert manifest["reference_count"] == 4
    assert [record.source_frame for record in records] == [9, 17, 45, 53]
    assert all(Path(record.path).is_file() for record in records)


def test_two_body_worker_payloads_cover_each_interval_once():
    workers = [payload(0), payload(1)]
    job_paths = [path for worker in workers for path in worker["jobs"]]
    assert len(job_paths) == len(set(job_paths)) == 14
    assert {Path(path).stem for path in job_paths} == {f"I{index:02d}" for index in range(14)}
    for worker_id, worker in enumerate(workers):
        assert worker["worker_id"] == worker_id
        assert len(worker["jobs"]) == 7
        assert worker["torch_cuda_memory_fraction"] == 0.245
        assert worker["memory_reservation_mib"] == 35840
        assert worker["recurrence"]["temporal_rebase"] is True
        assert [int(Path(path).stem[1:]) % 2 for path in worker["jobs"]] == [worker_id] * 7


def test_materialized_worker_payloads_match_builder():
    for worker_id in (0, 1):
        path = ROOT / "jobs" / "body_workers" / f"worker_{worker_id}.json"
        assert json.loads(path.read_text(encoding="utf-8")) == payload(worker_id)
    profile = json.loads(
        (ROOT / "jobs/body_workers/profile_worker_0_I00.json").read_text(encoding="utf-8")
    )
    assert len(profile["jobs"]) == 1
    assert Path(profile["jobs"][0]).stem == "I00"
    assert profile["profile_first_body_only"] is True


def test_endpoint_mask_is_exact_union_of_bridges_and_anchor():
    anchor = type("Rect", (), {"y0": 1, "x0": 2, "y1": 3, "x1": 5})()
    mask = torch.zeros(1, 11, 4, 8, 1, dtype=torch.bool)
    mask[:, :2] = True
    mask[:, -1:] = True
    mask[:, 2:-1, anchor.y0:anchor.y1, anchor.x0:anchor.x1] = True
    endpoint_mask_contract(mask, anchor)
    mask[:, 5, 0, 0] = True
    with pytest.raises(ValueError, match="Middle exterior"):
        endpoint_mask_contract(mask, anchor)
