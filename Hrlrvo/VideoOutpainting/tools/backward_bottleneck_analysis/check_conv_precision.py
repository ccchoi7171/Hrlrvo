"""FP64 CPU reference for selected-channel dW of exact full-size R2/R4 probes."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
from pathlib import Path

import torch
from ltx_outpaint.compression import SpatialDecompressor
from torch.nn import functional as F


def error(x, reference):
    x, reference = x.double(), reference.double()
    return {
        "relative_l2": float(
            (x - reference).norm() / reference.norm().clamp_min(1e-30)
        ),
        "max_abs": float((x - reference).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--modes", nargs="+", default=["current", "current_repeat", "ncdhw"])
    parser.add_argument("--trace-warm", action="store_true")
    args = parser.parse_args()
    # Set before the first convolution: toggling after cache warmup is not an A/B.
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in ("R4", "R2"):
        torch.manual_seed(33333333)
        batch, compressed, target = (
            (1, (17, 30), (68, 120)) if case == "R4" else (135, (1, 1), (2, 2))
        )
        token = torch.randn(
            batch,
            9,
            compressed[0] * compressed[1],
            4096,
            device="cuda",
            dtype=torch.bfloat16,
        ).requires_grad_()
        dy = torch.randn(batch, 9, *target, 4096, device="cuda", dtype=torch.bfloat16)
        cpu_dy = dy[..., :8].permute(0, 4, 1, 2, 3).cpu().double()
        reference = None
        for mode in args.modes:
            module = SpatialDecompressor(4096).cuda()
            holder = {}

            def hook(_module, inputs, *, mode=mode, holder=holder):
                volume = inputs[0]
                holder["input"] = volume[:, :8].detach().cpu()
                return (volume.contiguous() if mode == "ncdhw" else volume,)

            handle = module.refine.register_forward_pre_hook(hook)
            token.grad = None
            profiler = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                )
                if mode == "ncdhw"
                else contextlib.nullcontext()
            )
            with profiler as prof:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = module(token, compressed, target)
                output.backward(dy)
                torch.cuda.synchronize()
            if prof is not None:
                prof.export_chrome_trace(
                    str(args.output_dir / f"{case}_ncdhw_trace.json")
                )
                ops = [
                    {
                        "name": e.key,
                        "count": e.count,
                        "device": str(e.device_type),
                        "self_cuda_us": e.self_device_time_total,
                    }
                    for e in prof.key_averages()
                ]
                (args.output_dir / f"{case}_ncdhw_ops.json").write_text(
                    json.dumps(ops, indent=2)
                )
            dw = module.refine.weight.grad[:8].cpu().double()
            if reference is None:
                pad = F.pad(holder["input"].double(), (1, 1, 1, 1, 1, 1))
                reference = torch.empty(8, 1, 3, 3, 3, dtype=torch.float64)
                for dt in range(3):
                    for dh in range(3):
                        for dx in range(3):
                            shifted = pad[
                                :,
                                :,
                                dt : dt + 9,
                                dh : dh + target[0],
                                dx : dx + target[1],
                            ]
                            reference[:, 0, dt, dh, dx] = (shifted * cpu_dy).sum(
                                (0, 2, 3, 4)
                            )
            row = {
                "case": case,
                "mode": mode,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "channels_checked": 8,
                "reference": "FP64 CPU direct full-support convolution dW, actual BF16 input and dy",
                "error_vs_fp64": error(dw, reference),
                "error_vs_fp64_rounded_bf16": error(dw, reference.bfloat16().double()),
            }
            results.append(row)
            print(row, flush=True)
            (args.output_dir / "results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            if args.trace_warm:
                module.zero_grad(set_to_none=True)
                token.grad = None
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as warm_prof:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        warm_output = module(token, compressed, target)
                    warm_output.backward(dy)
                    torch.cuda.synchronize()
                warm_prof.export_chrome_trace(
                    str(args.output_dir / f"{case}_{mode}_warm_trace.json")
                )
                del warm_output
            handle.remove()
            del module, output, dw, holder
            gc.collect()
            torch.cuda.empty_cache()
        del token, dy, cpu_dy, reference
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
