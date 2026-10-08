"""Collect already-completed artifacts only; no model or CUDA imports."""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/cudnn_benchmark_validation/20260912_0637"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def phases(row):
    return {p["stage"]: p["seconds"] for p in row["phases"]}


def scopes(path):
    d = read(path)
    parents = defaultdict(lambda: defaultdict(float))
    children = defaultdict(lambda: defaultdict(float))
    replay_count = 0
    for key, value in d["details"].items():
        _, phase, _block, branch, label = key.split("/")
        if phase not in ("forward", "backward_only"):
            continue
        if label == "branch":
            parents[phase][branch] += value["gpu_seconds"]
        if label in (
            "block",
            "O",
            "TextCA",
            "FFN",
            "C2",
            "C4",
            "R2",
            "R4",
            "LocalFusion",
            "GlobalFusion",
        ):
            children[phase][branch + "/" + label] += value["gpu_seconds"]
        if phase == "backward_only" and label == "block":
            replay_count += value["calls"]
    return {
        "parent_seconds": dict(parents),
        "child_seconds": dict(children),
        "replay_calls": replay_count,
        "block_backward": d["block_backward"],
    }


def main():
    result = {}
    training = read(BASE / "production_benchmark/result.json")
    if training:
        warm = training["steps"][1:]
        result["production_status"] = training["status"]
        result["cold"] = phases(training["steps"][0])
        result["warm_mean"] = {
            k: st.mean(phases(s)[k] for s in warm) for k in phases(warm[0])
        }
        result["warm_phase_sum"] = {
            "mean": st.mean(s["phase_sum_seconds"] for s in warm),
            "median": st.median(s["phase_sum_seconds"] for s in warm),
            "min": min(s["phase_sum_seconds"] for s in warm),
            "max": max(s["phase_sum_seconds"] for s in warm),
        }
        result["warm_wall_including_diagnostics"] = [
            s["wall_including_diagnostics"] for s in warm
        ]
        result["peaks"] = {
            "allocated": max(
                s["memory"]["peak_allocated_gib"] for s in training["steps"]
            ),
            "reserved": max(
                s["memory"]["peak_reserved_gib"] for s in training["steps"]
            ),
        }
        result["warm_scopes"] = {}
        result["shapes"] = {}
        for s in training["steps"]:
            directory = BASE / "production_benchmark" / f"step_{s['step']:02d}"
            ss = read(directory / "R_shapes.json")
            shape = defaultdict(lambda: defaultdict(int))
            for v in ss:
                m = v["metadata"]
                shape[m["R"]][
                    json.dumps(
                        {
                            "shape": m["shape"],
                            "stride": m["stride"],
                            "dtype": m["dtype"],
                            "phase": m["phase"],
                        }
                    )
                ] += v["calls"]
            result["shapes"][str(s["step"])] = dict(shape)
            if s["step"] > 1:
                result["warm_scopes"][str(s["step"])] = scopes(
                    directory / "scopes.json"
                )
    control = read(BASE / "fresh_false_control/result.json")
    if control and control.get("steps"):
        false = control["steps"][0]
        true = training["steps"][0]
        result["numeric_comparison"] = {
            "loss_false": false["loss"],
            "loss_true": true["loss"],
            "loss_absolute_difference": abs(false["loss"] - true["loss"]),
            "input_sha_equal": false["input"]["latent_sha256"]
            == true["input"]["latent_sha256"],
            "representative_initial_hashes_equal": control["initial_parameter_hashes"]
            == training["initial_parameter_hashes"],
            "norms": {
                g: {
                    "false": f["norm"],
                    "true": true["gradient_health"]["groups"][g]["norm"],
                    "relative_difference": abs(
                        true["gradient_health"]["groups"][g]["norm"] - f["norm"]
                    )
                    / f["norm"],
                }
                for g, f in false["gradient_health"]["groups"].items()
            },
            "global_norm_false": false["global_grad_norm_before_clip"],
            "global_norm_true": true["global_grad_norm_before_clip"],
        }
        result["control_phases"] = phases(false)
        result["control_phase_sum"] = false["phase_sum_seconds"]
    cp = read(BASE / "constant_cache_probe/summary.json")
    if cp:
        result["constant_cache"] = []
        for row in cp:
            r = {
                "block": row["block_index"],
                "mode": row["configuration"],
                "status": row["status"],
            }
            if row.get("repeats"):
                r["forward"] = st.mean(phases(x)["forward"] for x in row["repeats"])
                r["backward"] = st.mean(
                    phases(x)["backward_only"] for x in row["repeats"]
                )
                r["total"] = r["forward"] + r["backward"]
                r["peak_allocated"] = max(
                    x["memory"]["peak_allocated_gib"] for x in row["repeats"]
                )
                r["post_forward_allocated"] = st.mean(
                    next(
                        p["after"]["allocated_gib"]
                        for p in x["phases"]
                        if p["stage"] == "forward"
                    )
                    for x in row["repeats"]
                )
                r["max_output_relative_l2"] = max(
                    x["comparison_to_baseline"]["output"]["relative_l2"]
                    for x in row["repeats"]
                )
                r["max_input_grad_relative_l2"] = max(
                    x["comparison_to_baseline"]["input_grad"]["relative_l2"]
                    for x in row["repeats"]
                )
                r["max_parameter_grad_relative_l2"] = max(
                    y.get("relative_l2", 0)
                    for x in row["repeats"]
                    for y in x["comparison_to_baseline"]["parameter_grads"].values()
                )
                r["all_exact"] = all(
                    x["comparison_to_baseline"]["all_exact"] for x in row["repeats"]
                )
                r["all_finite"] = all(
                    x["comparison_to_baseline"]["all_finite"] for x in row["repeats"]
                )
            result["constant_cache"].append(r)
    (BASE / "collected_summary.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("warm_scopes", "shapes")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
