"""Exact production R2/R4 shapes, original layout, fresh cuDNN flag A/B.

Operator-level random current-hidden/VJP probes, not substitutes for real-cache
full-model steps. The production SpatialDecompressor class is imported intact.
"""

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from ltx_outpaint.compression import SpatialDecompressor


def union_us(intervals):
    end = -float("inf")
    total = 0.0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--benchmark", type=int, choices=[0, 1], required=True)
    args = ap.parse_args()
    assert not torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = bool(args.benchmark)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    selected = {}
    for count in (0, 135, 21, 24, 39, 84):
        case = "R4" if count == 0 else f"R2_q{count}"
        compressed, target, batch = (
            ((17, 30), (68, 120), 1) if count == 0 else ((1, 1), (2, 2), count)
        )
        torch.manual_seed(33333333)
        token = torch.randn(
            batch,
            9,
            compressed[0] * compressed[1],
            4096,
            device="cuda",
            dtype=torch.bfloat16,
        ).requires_grad_()
        dy = torch.randn(batch, 9, *target, 4096, device="cuda", dtype=torch.bfloat16)
        module = SpatialDecompressor(4096).cuda()
        holder = {}

        def capture(_m, inputs, holder=holder):
            v = inputs[0]
            holder["shape"] = list(v.shape)
            holder["stride"] = list(v.stride())
            holder["input"] = v[:, :8].detach().cpu().double()

        handle = module.refine.register_forward_pre_hook(capture)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = module(token, compressed, target)
        y.backward(dy)
        torch.cuda.synchronize()
        cold = time.perf_counter() - start
        handle.remove()
        del y
        pad = F.pad(holder["input"], (1, 1, 1, 1, 1, 1))
        cpu_dy = dy[..., :8].permute(0, 4, 1, 2, 3).cpu().double()
        reference = torch.empty(8, 1, 3, 3, 3, dtype=torch.float64)
        for dt in range(3):
            for dh in range(3):
                for dx in range(3):
                    reference[:, 0, dt, dh, dx] = (
                        pad[:, :, dt : dt + 9, dh : dh + target[0], dx : dx + target[1]]
                        * cpu_dy
                    ).sum((0, 2, 3, 4))
        dw = module.refine.weight.grad[:8].cpu().double()
        selected[case] = dw
        error = float((dw - reference).norm() / reference.norm().clamp_min(1e-30))
        times = []
        for _ in range(3):
            module.zero_grad(set_to_none=True)
            token.grad = None
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = module(token, compressed, target)
            torch.cuda.synchronize()
            fwd = time.perf_counter() - start
            start = time.perf_counter()
            y.backward(dy)
            torch.cuda.synchronize()
            bwd = time.perf_counter() - start
            times.append({"forward": fwd, "backward": bwd})
            del y
        module.zero_grad(set_to_none=True)
        token.grad = None
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = module(token, compressed, target)
            y.backward(dy)
            torch.cuda.synchronize()
        path = out / f"{case}_trace.json"
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        names = Counter(e["name"] for e in kernels)
        wgrad = [
            e
            for e in kernels
            if "wgrad" in e["name"].lower() or "backward_weight" in e["name"].lower()
        ]
        row = {
            "case": case,
            "benchmark": torch.backends.cudnn.benchmark,
            "input_shape": holder["shape"],
            "input_stride": holder["stride"],
            "cold_forward_backward_seconds": cold,
            "warm_times": times,
            "channels_fp64_checked": 8,
            "dW_relative_l2_vs_fp64": error,
            "finite_output": bool(torch.isfinite(y).all()),
            "finite_dw": bool(torch.isfinite(module.refine.weight.grad).all()),
            "groupwise_convolveNd_wgrad_count": sum(
                v for k, v in names.items() if "convolveNd_wgrad_engine" in k
            ),
            "wgrad_kernels": [
                {"name": k, "count": v}
                for k, v in names.items()
                if "wgrad" in k.lower() or "backward_weight" in k.lower()
            ],
            "wgrad_kernel_union_ms": union_us(
                [(e["ts"], e["ts"] + e["dur"]) for e in wgrad]
            )
            / 1000,
            "wgrad_kernel_sum_ms": sum(e["dur"] for e in wgrad) / 1000,
        }
        rows.append(row)
        (out / "results.json").write_text(json.dumps(rows, indent=2))
        torch.save(selected, out / "selected_dW.pt")
        print(json.dumps(row), flush=True)
        del (
            module,
            token,
            dy,
            y,
            prof,
            events,
            kernels,
            names,
            wgrad,
            holder,
            pad,
            cpu_dy,
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
