"""Preserve live jobs while supplementary workers finish their late intervals."""
from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "validation/parallel_expansion_guard.json"


def atomic(data: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, STATE)


def latest_step(log: Path, body: str) -> int:
    if not log.exists():
        return 0
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 4096))
        lines = stream.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "body_denoising_progress" and event.get("body_id") == body:
            return int(event["step"])
    return 0


def complete(body: str) -> bool:
    path = ROOT / "experiment_1/bodies" / body / "status.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "PASS"


def main() -> None:
    paused: dict[int, int] = {}
    events: list[dict] = []
    while True:
        scheduler = json.loads((ROOT / "scheduler_status.json").read_text())
        tasks = scheduler["tasks"]
        additional_ended = True
        for parity in (0, 1):
            base = tasks.get(f"exp1_body_worker_{parity}_retry1", {})
            extra = tasks.get(f"exp1_body_additional_{parity + 2}")
            tail = [f"I{10 + parity:02d}", f"I{12 + parity:02d}"]
            extra_done = all(complete(body) for body in tail)
            extra_exited = extra is not None and extra.get("status") in ("COMPLETED", "FAILED")
            additional_ended &= extra_done or extra_exited
            if parity in paused and (extra_done or extra_exited):
                pid = paused.pop(parity)
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                events.append({"event": "resumed_base", "pid": pid, "utc": datetime.now(timezone.utc).isoformat()})
            if base.get("status") != "RUNNING" or extra_done or extra_exited or parity in paused:
                continue
            # The original worker approaches the reserved tail only after I08/I09.
            # A five-step cushion protects its result without altering sampling.
            before_tail = f"I{8 + parity:02d}"
            status_path = ROOT / "experiment_1/bodies" / before_tail / "status.json"
            if not status_path.exists():
                continue
            active = json.loads(status_path.read_text())
            if active.get("status") == "RUNNING" and active.get("pid") == base["pid"] and latest_step(Path(base["log"]), before_tail) >= 25:
                os.kill(base["pid"], signal.SIGSTOP)
                paused[parity] = base["pid"]
                events.append({"event": "paused_before_owned_tail", "pid": base["pid"], "body": before_tail, "utc": datetime.now(timezone.utc).isoformat()})
        atomic({"status": "PASS_EXTRA_FINISHED" if additional_ended else "MONITORING", "pid": os.getpid(), "paused": paused, "events": events})
        if additional_ended and not paused:
            return
        time.sleep(3)


if __name__ == "__main__":
    main()
