"""Linear/linearithmic raw Kineto summary without FunctionEvent-tree creation.

Kernel sums may overlap across streams. Report union separately. Backward scope
attribution is accepted only for unambiguous original-forward sequence numbers.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def union_us(intervals):
    end = -float("inf")
    total = 0.0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    args = ap.parse_args()
    events = json.loads(args.trace.read_text())["traceEvents"]
    cpu = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation")
    ]
    threads = defaultdict(list)
    byid = {}
    for i, e in enumerate(cpu):
        threads[(e["pid"], e["tid"])].append(i)
        eid = e.get("args", {}).get("External id")
        if eid is not None:
            byid[eid] = i
    parent = [None] * len(cpu)
    selfus = [e["dur"] for e in cpu]
    for ids in threads.values():
        stack = []
        for i in sorted(ids, key=lambda j: (cpu[j]["ts"], -cpu[j]["dur"])):
            e = cpu[i]
            end = e["ts"] + e["dur"]
            while stack and end > cpu[stack[-1]]["ts"] + cpu[stack[-1]]["dur"] + 0.01:
                stack.pop()
            if stack:
                parent[i] = stack[-1]
                selfus[stack[-1]] -= e["dur"]
            stack.append(i)
    tags = [None] * len(cpu)
    backseq = [None] * len(cpu)
    for i, e in enumerate(cpu):
        j = i
        while j is not None:
            name = cpu[j]["name"]
            if tags[i] is None and name.startswith("HR/"):
                tags[i] = name
            if backseq[i] is None and name.startswith(
                "autograd::engine::evaluate_function:"
            ):
                backseq[i] = cpu[j].get("args", {}).get("Sequence number")
            if tags[i] is not None and backseq[i] is not None:
                break
            j = parent[j]
    sequence_tags = defaultdict(set)
    for i, e in enumerate(cpu):
        seq = e.get("args", {}).get("Sequence number")
        if seq is not None and tags[i] and tags[i].startswith("HR/forward/"):
            sequence_tags[seq].add(tags[i].replace("HR/forward/", ""))
    cpuops = defaultdict(
        lambda: {
            "calls": 0,
            "self_cpu_us": 0.0,
            "inclusive_cpu_us": 0.0,
            "kernel_sum_us": 0.0,
            "kernel_count": 0,
        }
    )
    for i, e in enumerate(cpu):
        if e.get("cat") == "cpu_op":
            row = cpuops[e["name"]]
            row["calls"] += 1
            row["self_cpu_us"] += max(0, selfus[i])
            row["inclusive_cpu_us"] += e["dur"]
    kernel_names = defaultdict(lambda: {"calls": 0, "sum_us": 0.0})
    scopes = defaultdict(lambda: {"kernel_count": 0, "kernel_sum_us": 0.0})
    intervals = defaultdict(list)
    backends = defaultdict(lambda: {"calls": 0, "kernel_sum_us": 0.0})
    fwd_stage = next(e for e in cpu if e["name"] == "STAGE/forward")
    bwd_stage = next(e for e in cpu if e["name"] == "STAGE/backward_only")
    unattributed = Counter()
    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e["name"]
        dur = e["dur"]
        span = (e["ts"], e["ts"] + dur)
        row = kernel_names[name]
        row["calls"] += 1
        row["sum_us"] += dur
        phase = "forward" if e["ts"] < bwd_stage["ts"] else "backward"
        intervals[phase].append(span)
        i = byid.get(e.get("args", {}).get("External id"))
        if i is None:
            tag = "UNATTRIBUTED"
            op = "UNATTRIBUTED"
        else:
            j = i
            while j is not None and cpu[j].get("cat") != "cpu_op":
                j = parent[j]
            op = cpu[j]["name"] if j is not None else "UNATTRIBUTED"
            if op in cpuops:
                cpuops[op]["kernel_sum_us"] += dur
                cpuops[op]["kernel_count"] += 1
            tag = tags[i]
            if tag and tag.startswith("HR/backward_only/"):
                tag = "recompute/" + tag.removeprefix("HR/backward_only/")
            elif phase == "backward":
                match = sequence_tags.get(backseq[i], set())
                tag = (
                    "gradient/" + next(iter(match))
                    if len(match) == 1
                    else "gradient/UNATTRIBUTED"
                )
                if len(match) != 1:
                    unattributed[op] += 1
            elif tag:
                tag = "forward/" + tag.removeprefix("HR/forward/")
            else:
                tag = "forward/UNATTRIBUTED"
            j = i
            while j is not None:
                if "scaled_dot_product" in cpu[j]["name"]:
                    key = phase + "/" + cpu[j]["name"] + "/" + str(tag)
                    backends[key]["calls"] += 1
                    backends[key]["kernel_sum_us"] += dur
                    break
                j = parent[j]
        scopes[tag]["kernel_count"] += 1
        scopes[tag]["kernel_sum_us"] += dur
        if "wgrad" in name.lower():
            intervals[phase + "_wgrad"].append(span)
        if "convolveNd_wgrad_engine" in name:
            intervals[phase + "_pathological_wgrad"].append(span)
    result = {
        "source": str(args.trace),
        "event_count": len(events),
        "cpu_events": len(cpu),
        "categories": Counter(e.get("cat", "none") for e in events),
        "cpu_stage_annotation_seconds": {
            "forward": fwd_stage["dur"] / 1e6,
            "backward": bwd_stage["dur"] / 1e6,
        },
        "kernel_union_seconds": {k: union_us(v) / 1e6 for k, v in intervals.items()},
        "top_operators_by_kernel_sum": sorted(
            [{"name": k, **v} for k, v in cpuops.items()],
            key=lambda v: -v["kernel_sum_us"],
        )[:60],
        "top_operators_by_self_cpu": sorted(
            [{"name": k, **v} for k, v in cpuops.items()],
            key=lambda v: -v["self_cpu_us"],
        )[:60],
        "top_kernels": sorted(
            [{"name": k, **v} for k, v in kernel_names.items()],
            key=lambda v: -v["sum_us"],
        )[:80],
        "scopes": dict(scopes),
        "backend_kernel_events": dict(backends),
        "ambiguous_forward_sequence_keys": sum(
            len(v) > 1 for v in sequence_tags.values()
        ),
        "unattributed_backward_kernel_ops": unattributed,
        "limitations": [
            "CUDA operator/kernel sums may overlap; only interval union is elapsed occupancy.",
            "Backward scope mapping requires an unambiguous original forward sequence; unmapped kernels stay UNATTRIBUTED.",
            "Raw nearest-CPU external-id attribution, not exclusive branch wall time; no profiler memory tracing.",
        ],
    }
    path = args.trace.with_name("raw_summary.json")
    path.write_text(json.dumps(result, indent=2))
    print(
        path, result["cpu_stage_annotation_seconds"], result["kernel_union_seconds"], flush=True
    )


if __name__ == "__main__":
    main()
