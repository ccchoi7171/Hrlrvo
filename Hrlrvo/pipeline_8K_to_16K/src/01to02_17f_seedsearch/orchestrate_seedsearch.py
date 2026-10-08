"""Run 33 rounds: three independent B=1 processes per video, videos sequentially."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    MAIN,
    OUTPUT_ROOT,
    VIDEO_IDS,
    atomic_json,
    inference_identity,
    output_for,
    preflight_inputs,
    runtime_identity,
    utc_now,
    validate_candidate,
)
from seed_schedule import ensure_schedule

SOURCE = Path(__file__).resolve().parent
LOGGER = logging.getLogger("seedsearch")


def args_from_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke-single", "smoke-triple", "run", "bootstrap", "status"),
                        default="bootstrap")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=30)
    return parser.parse_args()


def configure_logging(root: Path) -> None:
    folder = root / "_logs"
    folder.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s UTC %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(folder / "orchestrator.log")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def completed_counts(root: Path, schedule: dict[str, Any]) -> dict[str, int]:
    counts = {}
    for video_id in VIDEO_IDS:
        count = 0
        for seed in schedule["seeds_in_generation_order"]:
            meta = output_for(video_id, seed, root)[2]
            if meta.is_file():
                try:
                    with meta.open("r", encoding="utf-8") as handle:
                        record = json.load(handle)
                    if record.get("status") == "success" and record.get("seed") == seed:
                        count += 1
                except (OSError, ValueError):
                    pass
        counts[video_id] = count
    return counts


def failures(root: Path) -> list[dict[str, Any]]:
    path = root / "_manifests/failed_runs.jsonl"
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"error": "Malformed failed_runs.jsonl line", "line": line})
    return records


def write_progress(root: Path, schedule: dict[str, Any], *, phase: str,
                   round_index: int | None = None, video_id: str | None = None,
                   seeds: list[int] | None = None,
                   active: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    counts = completed_counts(root, schedule)
    bad = failures(root)
    summary = {
        "updated_at": utc_now(), "phase": phase, "current_round": round_index,
        "current_video": video_id, "current_seeds": seeds,
        "active_processes": active or [], "per_video_success": counts,
        "success_count": sum(counts.values()), "failure_count": len(bad),
        "total_expected": 495, "per_video_expected": 99,
    }
    atomic_json(root / "_manifests/progress.json", summary)
    LOGGER.info("progress phase=%s round=%s video=%s seeds=%s active=%s success=%s/495 failures=%s per_video=%s",
                phase, round_index, video_id, seeds, len(active or []), summary["success_count"],
                summary["failure_count"], counts)
    return summary


def run_group(args: argparse.Namespace, *, root: Path, schedule: dict[str, Any],
              schedule_sha: str, inputs: dict[str, Any], identity: dict[str, Any],
              video_id: str, round_index: int, seeds: list[int],
              slot_indices: list[int] | None = None,
              verify_tensor_seed: int | None = None,
              launch_record_path: Path | None = None) -> dict[str, Any]:
    if len(seeds) > 3:
        raise ValueError("At most three independent processes are allowed")
    if slot_indices is None:
        slot_indices = list(range(1, len(seeds) + 1))
    if len(slot_indices) != len(seeds) or any(slot not in (1, 2, 3) for slot in slot_indices):
        raise ValueError("slot_indices must correspond to seeds and be 1, 2, or 3")
    active: list[tuple[subprocess.Popen[str], Path, int, int]] = []
    skipped: list[int] = []
    log_dir = root / "_logs" / video_id
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        for slot_index, seed in zip(slot_indices, seeds, strict=True):
            try:
                validate_candidate(
                    video_id, seed, root,
                    expected_input_sha=inputs[video_id]["sha256"],
                    expected_config_sha=identity["config_sha256"],
                    expected_adapter_sha=identity["adapter_sha256"],
                    expected_checkpoint=args.checkpoint, expected_steps=args.steps,
                    expected_schedule_sha=schedule_sha,
                    expected_round=round_index, expected_slot=slot_index,
                )
            except Exception as error:  # noqa: BLE001 - invalid/partial output is retried, never skipped
                LOGGER.info("candidate not complete; retry video=%s seed=%s reason=%s",
                            video_id, seed, error)
            else:
                skipped.append(seed)
                LOGGER.info("resume verified/skipped video=%s seed=%s round=%s slot=%s",
                            video_id, seed, round_index, slot_index)
                continue
            cmd = [
                sys.executable, str(SOURCE / "run_single_seedsearch.py"),
                "--video-id", video_id, "--seed", str(seed),
                "--output-root", str(root), "--config", str(args.config.resolve()),
                "--checkpoint", str(args.checkpoint.resolve()), "--steps", str(args.steps),
                "--round-index", str(round_index), "--slot-index", str(slot_index),
                "--schedule-sha256", schedule_sha,
                "--expected-input-sha256", inputs[video_id]["sha256"], "--resume",
            ]
            if seed == verify_tensor_seed:
                cmd.append("--verify-tensor-load")
            log_path = log_dir / f"{video_id}_{seed}_round{round_index:02d}_slot{slot_index}.wrapper.log"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n=== launch {utc_now()} command: {cmd!r} ===\n")
                log.flush()
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            active.append((process, log_path, seed, slot_index))
        if launch_record_path is not None and len(active) == 3:
            atomic_json(launch_record_path, {
                "launched_at": utc_now(), "video_id": video_id,
                "round_index": round_index, "seeds": seeds,
                "processes": [{"pid": p.pid, "seed": seed, "slot_index": slot,
                               "log": str(log)} for p, log, seed, slot in active],
            })
        write_progress(root, schedule, phase="running_group", round_index=round_index,
                       video_id=video_id, seeds=seeds,
                       active=[{"pid": p.pid, "seed": seed, "slot": slot, "log": str(log)}
                               for p, log, seed, slot in active])
        LOGGER.info("launched independent B=1 OS processes: %s",
                    [(p.pid, seed) for p, _, seed, _ in active])
        problems = []
        for process, log_path, seed, slot_index in active:
            code = process.wait()
            if code:
                problems.append({"seed": seed, "pid": process.pid, "exit_code": code,
                                 "log": str(log_path)})
            else:
                try:
                    validate_candidate(
                        video_id, seed, root,
                        expected_input_sha=inputs[video_id]["sha256"],
                        expected_config_sha=identity["config_sha256"],
                        expected_adapter_sha=identity["adapter_sha256"],
                        expected_checkpoint=args.checkpoint, expected_steps=args.steps,
                        expected_schedule_sha=schedule_sha,
                        expected_round=round_index, expected_slot=slot_index,
                    )
                except Exception as error:  # noqa: BLE001 - persist every subprocess validation failure
                    problems.append({"seed": seed, "pid": process.pid,
                                     "error": str(error), "log": str(log_path)})
        if problems:
            atomic_json(root / "_manifests/failed_runs.json",
                        {"updated_at": utc_now(), "failed_attempts": failures(root),
                         "current_group_problems": problems})
            LOGGER.error("group failed video=%s round=%s problems=%s", video_id, round_index, problems)
            raise RuntimeError(f"Group failed; inspect logs and failed_runs.json: {problems}")
        write_progress(root, schedule, phase="group_complete", round_index=round_index,
                       video_id=video_id, seeds=seeds)
        return {"video_id": video_id, "round_index": round_index, "seeds": seeds,
                "launched": [{"pid": p.pid, "seed": seed, "slot_index": slot, "log": str(log)}
                             for p, log, seed, slot in active], "skipped_valid": skipped,
                "status": "pass"}
    except BaseException:
        # Keep group ordering strict even if launch/validation fails.
        for process, _, _, _ in active:
            if process.poll() is None:
                process.wait()
        raise


def prepare(args: argparse.Namespace, root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    LOGGER.info("preflight: five exact 24fps inputs, full streaming decode, SHA-256")
    preflight = preflight_inputs(root)
    identity = inference_identity(args.config, args.checkpoint, args.steps)
    schedule, schedule_path, schedule_sha = ensure_schedule(root)
    context = {
        "prepared_at": utc_now(), "preflight_status": preflight["status"],
        "input_preflight_path": str(root / "_manifests/input_preflight.json"),
        "schedule_path": str(schedule_path), "schedule_sha256": schedule_sha,
        "input_sha256": {video_id: preflight["videos"][video_id]["sha256"] for video_id in VIDEO_IDS},
        "inference": identity,
        "runtime": runtime_identity(),
    }
    context_path = root / "_manifests/run_context.json"
    if context_path.is_file():
        previous = json.loads(context_path.read_text(encoding="utf-8"))
        for key in ("schedule_sha256", "input_sha256", "inference", "runtime"):
            if previous.get(key) != context[key]:
                raise RuntimeError(f"Existing run context changed at {key}; refuse to mix candidates. "
                                   f"Use a separate output root or restore the original condition: {context_path}")
    else:
        atomic_json(context_path, context)
    failure_summary = root / "_manifests/failed_runs.json"
    if not failure_summary.is_file():
        atomic_json(failure_summary, {"updated_at": utc_now(), "failed_attempts": []})
    LOGGER.info("preflight PASS, config=%s sha=%s checkpoint=%s adapter_sha=%s schedule=%s sha=%s",
                identity["config_path"], identity["config_sha256"],
                identity["checkpoint_path"], identity["adapter_sha256"], schedule_path, schedule_sha)
    return preflight["videos"], identity, schedule, schedule_sha


def smoke_single(args: argparse.Namespace, root: Path, schedule: dict[str, Any],
                 schedule_sha: str, inputs: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    # Use the last scheduled seed. Smoke B then launches all three first-round
    # seeds concurrently, without wasting a fourth, out-of-schedule candidate.
    seed = schedule["rounds"][-1]["seeds"][-1]
    report_path = root / "_manifests/smoke_single.json"
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("status") == "pass":
            try:
                validate_candidate("01", seed, root, expected_input_sha=inputs["01"]["sha256"],
                                   expected_config_sha=identity["config_sha256"],
                                   expected_adapter_sha=identity["adapter_sha256"],
                                   expected_checkpoint=args.checkpoint, expected_steps=args.steps,
                                   expected_schedule_sha=schedule_sha, expected_round=33,
                                   expected_slot=3)
            except Exception as error:  # noqa: BLE001 - an invalid smoke artifact must be rerun
                LOGGER.warning("Prior Smoke A invalid; rerun: %s", error)
            else:
                LOGGER.info("Smoke A already PASS; verified candidate reused")
                return previous
    LOGGER.info("Smoke A: video 01, seed=%s (round 33 / slot 3)", seed)
    result = run_group(args, root=root, schedule=schedule, schedule_sha=schedule_sha,
                       inputs=inputs, identity=identity, video_id="01",
                       round_index=33, seeds=[seed], slot_indices=[3], verify_tensor_seed=seed)
    if result["skipped_valid"]:
        sys.path.insert(0, str(MAIN / "src"))
        from ltx_outpaint.prefix_io import load_prefix_artifact

        loaded, _ = load_prefix_artifact(output_for("01", seed, root)[1],
                                         expected_frames=17, height=2160, width=3840)
        del loaded
    result["tensor_load_verified"] = True
    result["completed_at"] = utc_now()
    atomic_json(report_path, result)
    LOGGER.info("Smoke A PASS: %s", result)
    return result


def smoke_triple(args: argparse.Namespace, root: Path, schedule: dict[str, Any],
                 schedule_sha: str, inputs: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    report = root / "_manifests/smoke_single.json"
    if not report.is_file() or json.loads(report.read_text(encoding="utf-8")).get("status") != "pass":
        raise RuntimeError("Smoke A must pass before Smoke B")
    seeds = schedule["rounds"][0]["seeds"]
    report_path = root / "_manifests/smoke_triple.json"
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("status") == "pass":
            try:
                for slot, seed in enumerate(seeds, start=1):
                    validate_candidate("01", seed, root,
                                       expected_input_sha=inputs["01"]["sha256"],
                                       expected_config_sha=identity["config_sha256"],
                                       expected_adapter_sha=identity["adapter_sha256"],
                                       expected_checkpoint=args.checkpoint, expected_steps=args.steps,
                                       expected_schedule_sha=schedule_sha, expected_round=1,
                                       expected_slot=slot)
            except Exception as error:  # noqa: BLE001 - an invalid smoke artifact must be rerun
                LOGGER.warning("Prior Smoke B invalid; rerun: %s", error)
            else:
                LOGGER.info("Smoke B already PASS; three candidates verified/reused")
                return previous
    LOGGER.info("Smoke B: video 01, first-round seeds=%s, three independent B=1 processes", seeds)
    result = run_group(args, root=root, schedule=schedule, schedule_sha=schedule_sha,
                       inputs=inputs, identity=identity, video_id="01",
                       round_index=1, seeds=seeds,
                       launch_record_path=root / "_manifests/smoke_triple_launch.json")
    launch_record = root / "_manifests/smoke_triple_launch.json"
    evidence = json.loads(launch_record.read_text(encoding="utf-8")) if launch_record.is_file() else {}
    if (evidence.get("video_id") != "01" or evidence.get("round_index") != 1
            or evidence.get("seeds") != seeds or len(evidence.get("processes", [])) != 3):
        raise RuntimeError("Smoke B lacks proof that three independent OS processes were launched")
    result["independent_os_processes_launched"] = 3
    result["concurrency_evidence_path"] = str(launch_record)
    result["completed_at"] = utc_now()
    atomic_json(report_path, result)
    LOGGER.info("Smoke B PASS: %s", result)
    return result


def long_run(args: argparse.Namespace, root: Path, schedule: dict[str, Any],
             schedule_sha: str, inputs: dict[str, Any], identity: dict[str, Any]) -> None:
    for name in ("smoke_single.json", "smoke_triple.json"):
        path = root / "_manifests" / name
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("status") != "pass":
            raise RuntimeError(f"Long run blocked: {path} must show Smoke PASS")
    LOGGER.info("Long run START: 33 rounds x 5 sequential videos x 3 independent processes = 495 results")
    for group in schedule["rounds"]:
        round_index = group["round_index"]
        seeds = group["seeds"]
        for video_id in VIDEO_IDS:
            run_group(args, root=root, schedule=schedule, schedule_sha=schedule_sha,
                      inputs=inputs, identity=identity, video_id=video_id,
                      round_index=round_index, seeds=seeds)
    progress = write_progress(root, schedule, phase="complete", round_index=33,
                              video_id="05", seeds=schedule["rounds"][-1]["seeds"])
    if progress["success_count"] != 495:
        raise RuntimeError(f"Expected 495 completed candidates, got {progress['success_count']}")
    LOGGER.info("Long run COMPLETE: all 495 scheduled candidates")


def main() -> None:
    args = args_from_cli()
    root = args.output_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    configure_logging(root)
    if args.phase == "status":
        progress = root / "_manifests/progress.json"
        print(progress.read_text(encoding="utf-8") if progress.is_file() else "No progress manifest yet")
        return
    inputs, identity, schedule, schedule_sha = prepare(args, root)
    write_progress(root, schedule, phase="preflight_pass")
    if args.phase == "preflight":
        return
    if args.phase in ("smoke-single", "bootstrap"):
        smoke_single(args, root, schedule, schedule_sha, inputs, identity)
    if args.phase in ("smoke-triple", "bootstrap"):
        smoke_triple(args, root, schedule, schedule_sha, inputs, identity)
    if args.phase in ("run", "bootstrap"):
        long_run(args, root, schedule, schedule_sha, inputs, identity)


if __name__ == "__main__":
    main()
