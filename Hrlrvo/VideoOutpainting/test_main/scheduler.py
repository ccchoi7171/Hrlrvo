"""Run the two authorized H200 experiments within separate GPU budgets."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GPU = "GPU-a01d425d-ffe8-26e3-c637-6c87a4cf0f38"
BUDGET_MIB = 70 * 1024
PHYSICAL_RESERVE_MIB = 2048


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", f"--id={GPU}", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    used, free = map(int, result.stdout.strip().split(","))
    apps = subprocess.run(
        ["nvidia-smi", f"--id={GPU}", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    by_pid = {}
    for line in apps.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and all(part.isdigit() for part in fields):
            by_pid[int(fields[0])] = int(fields[1])
    return used, free, by_pid


def main():
    queue = ROOT / "queue"
    queue.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    state_path = ROOT / "scheduler_status.json"
    entries = {}
    running = {}
    started = utc()
    while True:
        for name, (process, log) in list(running.items()):
            code = process.poll()
            if code is not None:
                entries[name].update(status="COMPLETED" if code == 0 else "FAILED", exit_code=code, ended_utc=utc())
                log.close()
                del running[name]
        used, free, by_pid = gpu_memory()
        reservations = {1: 0, 2: 0}
        counts = {1: 0, 2: 0}
        for name in running:
            row = entries[name]
            measured = by_pid.get(row["pid"], 0)
            row["measured_mib"] = measured
            row["peak_measured_mib"] = max(row.get("peak_measured_mib", 0), measured)
            reservations[row["experiment"]] += max(row["reservation_mib"], measured + 512)
            counts[row["experiment"]] += 1
        pending = []
        if not (ROOT / "PAUSE").exists():
            tasks = []
            for path in queue.glob("*.json"):
                task = json.loads(path.read_text())
                task["task_path"] = str(path)
                tasks.append(task)
            tasks.sort(key=lambda task: (task.get("priority", 0), task["name"]))
            for task in tasks:
                name = task["name"]
                if name in entries:
                    continue
                experiment = task["experiment"]
                reserve = int(task["reservation_mib"])
                if experiment not in (1, 2) or not 0 < reserve <= BUDGET_MIB:
                    raise ValueError(f"Invalid experiment GPU reservation: {name}")
                if not all(Path(path).exists() for path in task.get("requires", [])):
                    pending.append({"name": name, "reason": "prerequisites"})
                    continue
                if reservations[experiment] + reserve > BUDGET_MIB or counts[experiment] >= task.get("max_group_processes", 2):
                    pending.append({"name": name, "reason": "experiment_budget"})
                    continue
                # Count reservations of processes still loading, whose CUDA
                # allocation is not yet visible, against physical free memory.
                unallocated = sum(max(entries[n]["reservation_mib"] - by_pid.get(entries[n]["pid"], 0), 0) for n in running)
                if free - unallocated < reserve + PHYSICAL_RESERVE_MIB:
                    pending.append({"name": name, "reason": "physical_headroom"})
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
                env.update(task.get("env", {}))
                if env["CUDA_VISIBLE_DEVICES"] != GPU:
                    raise ValueError("Only the authorized H200 is allowed")
                log_path = ROOT / "logs" / f"{name}.log"
                log = log_path.open("ab", buffering=0)
                process = subprocess.Popen(task["argv"], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                entries[name] = dict(task, status="RUNNING", pid=process.pid, log=str(log_path), started_utc=utc(), measured_mib=0, peak_measured_mib=0)
                running[name] = (process, log)
                reservations[experiment] += reserve
                counts[experiment] += 1
        atomic_json(state_path, {"pid": os.getpid(), "started_utc": started, "checked_utc": utc(), "state": "PAUSED" if (ROOT / "PAUSE").exists() else "RUNNING", "GPU": GPU, "budget_per_experiment_mib": BUDGET_MIB, "physical_reserve_mib": PHYSICAL_RESERVE_MIB, "gpu_used_mib": used, "gpu_free_mib": free, "group_reserved_mib": reservations, "tasks": entries, "pending": pending})
        time.sleep(10)


if __name__ == "__main__":
    main()
