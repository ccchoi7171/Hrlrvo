"""Resume-safe Experiment 2 full1025 phase carry + temporal-rebase producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
SOURCE = COMMON / "source"
SNAPSHOT = HERE.parent / "validation/common_snapshot.json"
H200_UUID = "GPU-a01d425d-ffe8-26e3-c637-6c87a4cf0f38"
MODE = "exterior_last2_keep_native_row0"

sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(SOURCE))
from latent_prefix_carry_phase_aligned import (
    PhaseAlignedLatentPrefixCarry,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def validate_common_snapshot() -> dict:
    receipt = json.loads(SNAPSHOT.read_text())
    observed = {}
    for relative, expected in receipt["files"].items():
        path = COMMON / relative
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Common frozen source changed: {relative}")
        observed[relative] = actual
    return observed


def validate_job(job: dict, job_path: Path) -> dict:
    required = {
        "name": "experiment_2_phase_rebase_full1025",
        "experiment": 2,
        "backend": "native",
        "seed": 33333333,
        "steps": 30,
        "frames": 1025,
        "recurrence": {"reduced_geometry": True, "temporal_rebase": True},
        "attention": {"collect_branch_stats": False},
        "prompt": "",
        "restore_original_anchor_after_inference": False,
        "requires_inverse_registration": False,
        "latent_carry_mode": MODE,
        "producer_script": Path(__file__).name,
    }
    for key, value in required.items():
        if job.get(key) != value:
            raise ValueError(f"Experiment 2 job field changed: {key}")
    if job.get("source_frame_start") != 0 or job.get("source_frame_end_inclusive") != 1024:
        raise ValueError("Experiment 2 requires exact source0:1025")
    if not (0 < float(job.get("memory_fraction", 0)) <= 0.49):
        raise ValueError("Experiment 2 process cap must remain <=49% H200")
    for key in ("input", "config", "checkpoint", "clean_preparation_manifest"):
        if not Path(job[key]).is_absolute():
            raise ValueError(f"Absolute path required: {key}")
    if digest(Path(job["input"])) != job.get("input_sha256"):
        raise ValueError("Clean anchor SHA changed")
    manifest = Path(job["clean_preparation_manifest"])
    if digest(manifest) != job.get("clean_preparation_manifest_sha256"):
        raise ValueError("Clean preparation manifest SHA changed")
    preparation = json.loads(manifest.read_text())
    if (
        preparation.get("status") != "PASS_GPU_CLEAN_ANCHOR_READY"
        or preparation.get("clean_anchor", {}).get("sha256") != job["input_sha256"]
        or preparation.get("person_segmentation", {}).get("dense_all_frames") is not True
        or preparation.get("cleanup", {}).get("gpu_donor_fill") is not True
        or preparation.get("original_RAW_anchor_post_reinsertion") is not False
    ):
        raise ValueError("GPU clean-anchor preparation contract incomplete")
    config = yaml.safe_load(Path(job["config"]).read_text())
    sampling = config["sampling"]
    if (
        config["video"]["canvas"] != [1920, 1080]
        or config["video"]["expected_anchor"] != [960, 540]
        or config["video"]["boundary_blend_width_pixels"] != 0
        or config["runtime"]["dtype"] != "bfloat16"
        or sampling != {
            "negative_prompt": "", "inference_steps": 30, "cfg_scale": 3.0,
            "stg_scale": 1.0, "stg_blocks": [28], "guidance_rescale": 0.7,
            "seed": 33333333,
        }
        or config["long_range"] != {
            "clip_frames": 65, "prefix_frames": 17, "stride_frames": 48,
            "final_chunk_policy": "repeat_last",
        }
        or config["long_range_io"] != {
            "segment_dir": None, "resume": False, "keep_intermediates": True,
        }
    ):
        raise ValueError("Native65/BF16/guidance/resume config changed")
    return {"job_path": str(job_path), "job_sha256": digest(job_path), "config": config}


def committed_count(segment_dir: Path) -> int:
    count = 0
    while (segment_dir / f"chunk_{count:06d}.json").is_file():
        count += 1
    return count


def resume_enabled(segment_dir: Path, completed: int) -> bool:
    metadata = segment_dir / "run.json"
    if completed and not metadata.is_file():
        raise FileNotFoundError("Committed chunks require native resume metadata")
    return metadata.is_file()


def reconcile_resume(folder: Path, completed: int) -> dict:
    lineage = folder / "phase_aligned_latent_carry_lineage.jsonl"
    emissions = folder / "phase_emission_lineage.jsonl"
    rows = [] if not lineage.exists() else [json.loads(line) for line in lineage.read_text().splitlines() if line.strip()]
    emission_rows = [] if not emissions.exists() else [json.loads(line) for line in emissions.read_text().splitlines() if line.strip()]
    tails = sorted(folder.glob("phase_tail_chunk_*_bf16.pt"))
    orphan_first65 = completed == 0 and (folder / "first_native_chunk_fp32.npy").exists()
    orphan_rows = rows[completed:]
    orphan_tails = [path for path in tails if int(path.stem.split("_")[3]) >= completed]
    orphan_emissions = emission_rows[completed:]
    recovery = {"completed_chunks": completed, "orphan_lineage_rows": len(orphan_rows), "orphan_emission_rows": len(orphan_emissions), "orphan_tail_files": []}
    if orphan_rows or orphan_emissions or orphan_tails or orphan_first65:
        archive = folder / "resume_orphans" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        if orphan_rows:
            (archive / "lineage_orphans.jsonl").write_text("".join(json.dumps(row) + "\n" for row in orphan_rows))
        if orphan_emissions:
            (archive / "emission_orphans.jsonl").write_text("".join(json.dumps(row) + "\n" for row in orphan_emissions))
        lineage.write_text("".join(json.dumps(row) + "\n" for row in rows[:completed]))
        emissions.write_text("".join(json.dumps(row) + "\n" for row in emission_rows[:completed]))
        for path in orphan_tails:
            destination = archive / path.name
            shutil.move(path, destination)
            recovery["orphan_tail_files"].append(str(destination))
        if orphan_first65:
            destination = archive / "first_native_chunk_fp32.npy"
            shutil.move(folder / "first_native_chunk_fp32.npy", destination)
            recovery["orphan_first65"] = str(destination)
        recovery["archive"] = str(archive)
    if len(rows[:completed]) != completed:
        raise RuntimeError("Committed chunks and phase lineage are not contiguous")
    if len(emission_rows[:completed]) != completed:
        raise RuntimeError("Committed chunks and emission lineage are not contiguous")
    return recovery


def make_observer(
    folder: Path,
    plan: list[dict],
    adapter,
    completed: int,
    recurrence_holder: dict,
):
    state = {"chunks": completed, "emitted": 0 if completed == 0 else int(plan[completed - 1]["new_output_frame_range"][1])}

    def observer(record, generated, next_prefix, finalized):
        index = state["chunks"]
        expected = plan[index]
        if any(record[key] != expected[key] for key in ("chunk_index", "absolute_start_frame", "new_output_frame_range", "output_appended_count")):
            raise RuntimeError("Native chunk planning changed")
        if not torch.equal(next_prefix, generated[-17:]):
            raise RuntimeError("Chronological RGB17 prefix changed")
        offset = 0 if index == 0 else 17
        if not torch.equal(finalized, generated[offset : offset + record["output_appended_count"]]):
            raise RuntimeError("Native 65+48 emission changed")
        row = adapter.consume()
        context = recurrence_holder.get("context")
        if not isinstance(context, dict):
            raise TypeError("Recurrence context was not captured")
        rebase = [
            event
            for event in context.get("events", [])
            if event.get("chunk") == index and event.get("action") == "temporal_rebase"
        ]
        expected_seconds = record["absolute_start_frame"] / 24
        if len(rebase) != 1 or abs(float(rebase[0].get("seconds_removed", -1)) - expected_seconds) > 1e-9:
            raise RuntimeError("Exactly one actual temporal-rebase event per chunk required")
        row.update(
            new_output_frame_range=record["new_output_frame_range"],
            emitted_tensor_range=[offset, offset + len(finalized)],
            temporal_rebase_event=rebase[0],
            chunk_seed=33333333 + int(record["absolute_start_frame"]),
            chronological_prefix_verified=True,
        )
        # consume() already writes the base row; append the emission binding separately.
        with (folder / "phase_emission_lineage.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if index == 0:
            first_path = folder / "first_native_chunk_fp32.npy"
            first = np.lib.format.open_memmap(first_path, mode="w+", dtype=np.float32, shape=tuple(generated.shape))
            first[:] = generated.detach().cpu().numpy(); first.flush(); del first
        state["chunks"] += 1
        state["emitted"] = int(record["new_output_frame_range"][1])
        print(json.dumps({"event": "experiment2_chunk_verified", "chunk": index, "emitted": state["emitted"]}), flush=True)

    return observer, state


def validate_committed_evidence(
    plan: list[dict], emissions: list[dict], segment_dir: Path
) -> int:
    """Aggregate durable per-chunk proof across any number of resumed invocations."""
    if len(emissions) != len(plan):
        raise RuntimeError("Committed phase emission lineage length differs")
    for index, (record, emission) in enumerate(zip(plan, emissions, strict=True)):
        start = index * 48
        marker = segment_dir / f"chunk_{index:06d}.json"
        committed = json.loads(marker.read_text())
        expected_prefix = None if index == 0 else [start, start + 17]
        if (
            emission.get("chunk_index") != index
            or emission.get("absolute_start_frame") != start
            or emission.get("chunk_seed") != 33333333 + start
            or emission.get("chronological_prefix_verified") is not True
            or emission.get("temporal_rebase_event")
            != {
                "chunk": index,
                "action": "temporal_rebase",
                "seconds_removed": start / 24,
            }
            or committed.get("chunk_seed") != 33333333 + start
            or committed.get("absolute_start_frame") != start
            or committed.get("new_output_frame_range")
            != record["new_output_frame_range"]
            or committed.get("prefix_output_frame_range") != expected_prefix
        ):
            raise RuntimeError(
                f"Resume-aggregated seed/prefix/rebase proof differs at chunk {index}"
            )
    return len(emissions)


def run(job_path: Path) -> None:
    job = json.loads(job_path.read_text())
    validation = validate_job(job, job_path)
    common = validate_common_snapshot()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != H200_UUID:
        raise RuntimeError("Experiment 2 requires the authorized H200 UUID")
    physical = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader", "--id=0"],
        check=True, capture_output=True, text=True,
    ).stdout
    if H200_UUID not in physical or "NVIDIA H200" not in physical:
        raise RuntimeError("Physical H200 binding failed")
    from ltx_outpaint import inference

    if Path(inference.__file__).resolve() != SOURCE / "ltx_outpaint/inference.py":
        raise RuntimeError("Frozen common inference import required")
    settings = validation["config"]
    plan = list(inference._iter_long_range_chunks(1025, inference.LongRangeSpec(**settings["long_range"]), temporal_factor=8))
    if len(plan) != 21 or [row["absolute_start_frame"] for row in plan] != list(range(0, 961, 48)):
        raise RuntimeError("Expected exact 21-chunk source0..960 plan")
    folder = Path(job["output"]).parent
    segment_dir = folder / "segments"
    folder.mkdir(parents=True, exist_ok=True)
    completed = committed_count(segment_dir)
    if completed > 21:
        raise RuntimeError("Too many committed chunks")
    recovery = reconcile_resume(folder, completed)
    should_resume = resume_enabled(segment_dir, completed)
    native_sample = inference.sample_chunk
    native_condition = inference._condition_latents
    native_long = inference.sample_long_range
    adapter = PhaseAlignedLatentPrefixCarry(native_sample, native_condition, MODE, folder=folder)
    adapter.calls = completed
    if completed:
        bank_path = folder / f"phase_tail_chunk_{completed - 1:03d}_bf16.pt"
        if not bank_path.is_file():
            raise RuntimeError("Resume requires the last committed phase bank")
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        if tuple(bank.shape) != (1, 128, 2, 36, 60) or bank.dtype != torch.bfloat16:
            raise RuntimeError("Resume phase bank shape/dtype changed")
        adapter._bank = bank.detach().contiguous().cpu().clone()
        adapter._bank_from_start = (completed - 1) * 48
    recurrence_holder: dict = {}
    observer, observed = make_observer(
        folder, plan, adapter, completed, recurrence_holder
    )

    def long_with_observer(*args, **kwargs):
        if kwargs.get("equivalence_observer") is not None:
            raise RuntimeError("Experiment 2 owns the only observer")
        return native_long(
            *args,
            **{
                **kwargs,
                "equivalence_observer": observer,
                "resume": should_resume,
            },
        )

    adapter.install(inference)
    inference.sample_long_range = long_with_observer
    argv = sys.argv[:]
    import recurrence

    native_recurrence_install = recurrence.install

    def observed_recurrence_install(module, spec):
        context = native_recurrence_install(module, spec)
        recurrence_holder["context"] = context
        return context

    recurrence.install = observed_recurrence_install
    try:
        import run_job

        sys.argv = [str(COMMON / "run_job.py"), "--job", str(job_path)]
        run_job.main()
        if observed != {"chunks": 21, "emitted": 1025} or adapter.pending is not None:
            raise RuntimeError("Experiment 2 did not complete all chunks")
        status_path = folder / "job_status.json"
        status = json.loads(status_path.read_text())
        events = status.get("recurrence_events", [])
        rebases = [row for row in events if row.get("action") == "temporal_rebase"]
        if status.get("status") != "PASS" or (
            completed < 21 and len(rebases) != 21 - completed
        ):
            raise RuntimeError("Underlying PASS or current-invocation temporal-rebase events missing")
        lineage = folder / "phase_aligned_latent_carry_lineage.jsonl"
        rows = [json.loads(line) for line in lineage.read_text().splitlines()]
        if len(rows) != 21 or rows[0].get("carry_applied") is not False or any(row.get("carry_applied") is not True for row in rows[1:]):
            raise RuntimeError("Phase-aligned lineage incomplete")
        emission_path = folder / "phase_emission_lineage.jsonl"
        emissions = [json.loads(line) for line in emission_path.read_text().splitlines()]
        if validate_committed_evidence(plan, emissions, segment_dir) != 21:
            raise RuntimeError("Committed phase evidence must contain 21 rows")
        status["experiment_2_contract"] = {
            "status": "PASS_FULL1025_PHASE_ALIGNED_TEMPORAL_REBASE",
            "common_snapshot": common,
            "runner_sha256": digest(Path(__file__)),
            "job_sha256": validation["job_sha256"],
            "clean_manifest_sha256": job["clean_preparation_manifest_sha256"],
            "chunks": 21,
            "chunk_starts": list(range(0, 961, 48)),
            "phase_lineage_path": str(lineage),
            "phase_lineage_sha256": digest(lineage),
            "phase_emission_lineage_path": str(emission_path),
            "phase_emission_lineage_sha256": digest(emission_path),
            "temporal_rebase_events": 21,
            "temporal_rebase_proof": "per-committed-chunk durable observer evidence",
            "seed_policy": "33333333 + absolute_start_frame",
            "original_RAW_anchor_post_reinsertion": False,
            "clean_anchor_is_native_observation": True,
            "resume_recovery": recovery,
        }
        status_path.write_text(json.dumps(status, indent=2))
    finally:
        sys.argv = argv
        recurrence.install = native_recurrence_install
        inference.sample_chunk = native_sample
        inference._condition_latents = native_condition
        inference.sample_long_range = native_long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    run(args.job.expanduser().resolve())


if __name__ == "__main__":
    main()
