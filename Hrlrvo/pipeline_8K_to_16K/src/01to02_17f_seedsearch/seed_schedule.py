"""Persist the global 99-seed schedule before any GPU work starts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import random
from pathlib import Path
from typing import Any

from common import OUTPUT_ROOT, atomic_json, sha256_file, utc_now

MASTER_SEED = 33333333
ROUND_COUNT = 33
SLOTS_PER_ROUND = 3
SEED_COUNT = ROUND_COUNT * SLOTS_PER_ROUND
GENERATOR = "python.random.Random(33333333).sample(range(1, 2**31), 99)"


def _make_schedule() -> dict[str, Any]:
    seeds = random.Random(MASTER_SEED).sample(range(1, 2**31), SEED_COUNT)
    rounds = [
        {"round_index": index + 1, "seeds": seeds[index * SLOTS_PER_ROUND:(index + 1) * SLOTS_PER_ROUND]}
        for index in range(ROUND_COUNT)
    ]
    return {
        "schema_version": 1, "generator": GENERATOR, "master_seed": MASTER_SEED,
        "python_version_at_creation": platform.python_version(),
        "created_at": utc_now(), "seed_count": SEED_COUNT,
        "round_count": ROUND_COUNT, "slots_per_round": SLOTS_PER_ROUND,
        "seeds_in_generation_order": seeds, "rounds": rounds,
    }


def _validate(schedule: dict[str, Any]) -> None:
    seeds = schedule.get("seeds_in_generation_order")
    if (schedule.get("schema_version") != 1 or schedule.get("master_seed") != MASTER_SEED
            or schedule.get("generator") != GENERATOR or schedule.get("seed_count") != SEED_COUNT
            or schedule.get("round_count") != ROUND_COUNT
            or schedule.get("slots_per_round") != SLOTS_PER_ROUND
            or not isinstance(seeds, list) or len(seeds) != SEED_COUNT
            or any(type(seed) is not int or not 1 <= seed < 2**31 for seed in seeds)
            or len(set(seeds)) != SEED_COUNT):
        raise ValueError("Existing seed schedule has invalid parameters or duplicate seeds")
    expected_rounds = [
        {"round_index": index + 1, "seeds": seeds[index * SLOTS_PER_ROUND:(index + 1) * SLOTS_PER_ROUND]}
        for index in range(ROUND_COUNT)
    ]
    if schedule.get("rounds") != expected_rounds:
        raise ValueError("Existing seed schedule round grouping is inconsistent")


def ensure_schedule(root: Path = OUTPUT_ROOT) -> tuple[dict[str, Any], Path, str]:
    folder = Path(root) / "_schedule"
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "seed_schedule.json"
    csv_path = folder / "seed_schedule.csv"
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as handle:
            schedule = json.load(handle)
        _validate(schedule)
    else:
        schedule = _make_schedule()
        _validate(schedule)
        atomic_json(json_path, schedule)
    csv_data = io.StringIO()
    writer = csv.writer(csv_data)
    writer.writerow(("round_index", "slot_index", "seed"))
    for group in schedule["rounds"]:
        for slot_index, seed in enumerate(group["seeds"], start=1):
            writer.writerow((group["round_index"], slot_index, seed))
    if csv_path.is_file():
        if csv_path.read_bytes().decode("utf-8") != csv_data.getvalue():
            raise ValueError(f"Existing CSV disagrees with immutable JSON schedule: {csv_path}")
    else:
        csv_path.write_text(csv_data.getvalue(), encoding="utf-8")
    return schedule, json_path, sha256_file(json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    schedule, path, digest = ensure_schedule(args.output_root)
    print(json.dumps({"status": "pass", "path": str(path), "sha256": digest,
                      "seeds": len(schedule["seeds_in_generation_order"]),
                      "rounds": len(schedule["rounds"])}))


if __name__ == "__main__":
    main()
