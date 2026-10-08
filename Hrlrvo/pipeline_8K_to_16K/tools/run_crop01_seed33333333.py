#!/usr/bin/env python3
"""Run the approved cropped-01 preprocess and outpaint stages only.

The certification CLI fixes the historical seed for video 01.  This isolated
runner validates that template, then changes only the in-memory request seed
and run identity before calling the same stage functions.  It never dispatches
the upscale stage.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_cert import (  # noqa: E402
    artifact_path,
    atomic_json,
    run_outpaint_stage,
    run_preprocess_stage,
    sha256_file,
    validate_config,
)

RUN = ROOT / "state/runs/01_crop_seed33333333"
SOURCE = ROOT / "data/00_Input_8K_7680x4320/01_Input_8K.mp4"
SOURCE_SHA256 = "469853124f36e1b954bc312e63f28546da6365dbee0b1439d3ef10a6086e9124"
TEMPLATE = ROOT / "configs/certification/01.json"
REQUEST = ROOT / "configs/experiments/01_crop_seed33333333.json"


def status(stage: str, **details: object) -> None:
    atomic_json(
        RUN / "status.json",
        {
            "stage": stage,
            "pid": os.getpid(),
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            **details,
        },
    )
    print(f"[RUN_STATUS] stage={stage} details={details}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "runner.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.chdir(ROOT)
        config, *_ = validate_config(TEMPLATE)
        expected = copy.deepcopy(config)
        expected["seed"] = 33333333
        expected["job_name"] = "01_crop_seed33333333"
        expected["output_root"] = "state/runs/01_crop_seed33333333"
        config = json.loads(REQUEST.read_text(encoding="utf-8"))
        if config != expected:
            raise RuntimeError("Experimental request differs from the validated template beyond the approved seed/run identity")

        if sha256_file(SOURCE) != SOURCE_SHA256:
            raise RuntimeError("Cropped 8K source SHA changed")

        request_path = RUN / "requested_config.json"
        if args.resume:
            if not request_path.is_file() or json.loads(request_path.read_text()) != config:
                raise RuntimeError("Resume request does not match the original run")
        else:
            if request_path.exists():
                raise RuntimeError("Run already initialized; use --resume")
            if artifact_path(config, "outpaint").exists():
                raise RuntimeError("Outpainting destination already exists")
            atomic_json(request_path, config)

        print(
            f"[RUN_START] resume={args.resume} source={SOURCE} "
            "frames=1952 seed=33333333 stages=preprocess,outpaint",
            flush=True,
        )
        try:
            status("PREPROCESS_RUNNING")
            preprocessed = run_preprocess_stage(config, resume=args.resume)
            status("PREPROCESS_DONE", output=str(preprocessed))

            if (RUN / "stop.requested").exists():
                status("STOPPED_AFTER_PREPROCESS", output=str(preprocessed))
                return 0

            status("OUTPAINT_RUNNING", input=str(preprocessed))
            output = run_outpaint_stage(config, resume=args.resume)
            if sha256_file(SOURCE) != SOURCE_SHA256:
                raise RuntimeError("Cropped 8K source changed during the run")
            status("COMPLETE", input=str(preprocessed), output=str(output))
            print(f"[RUN_DONE] output={output}", flush=True)
            return 0
        except KeyboardInterrupt:
            status("INTERRUPTED")
            raise
        except BaseException as error:
            status("FAILED", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    raise SystemExit(main())
