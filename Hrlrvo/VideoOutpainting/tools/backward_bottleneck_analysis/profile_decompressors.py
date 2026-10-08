"""Isolated current R2/R4 kernel A/B; never a replacement production benchmark.

Exact production module, dimensions, FP32 master weights and BF16 autocast.
Fixed seeded probe tokens/cotangents isolate the operator; real-cache full-model
measurements are in profile_training.py. No persistent source/backend changes.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from ltx_outpaint.compression import SpatialDecompressor


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def compare(x, ref):
    delta = x.detach().float() - ref.to(x.device).float()
    values = (
        torch.stack(
            [
                delta.square().sum(),
                ref.to(x.device).float().square().sum(),
                delta.abs().max(),
                (~torch.isfinite(x)).sum().float(),
            ]
        )
        .cpu()
        .tolist()
    )
    return {
        "relative_l2": (values[0] / max(values[1], 1e-30)) ** 0.5,
        "max_abs": values[2],
        "nonfinite": int(values[3]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["R4", "R2"])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "current",
            "benchmark",
            "weight_cl3d",
            "input_ncdhw",
            "input_ncdhw_benchmark",
            "no_cudnn",
        ],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Torch", torch.__version__, "cuDNN", torch.backends.cudnn.version(), flush=True
    )
    print(
        "Original flags",
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.benchmark_limit,
        torch.backends.cudnn.deterministic,
        flush=True,
    )
    results = []
    for case in args.cases:
        torch.manual_seed(33333333)
        batch, compressed, target = (
            (1, (17, 30), (68, 120)) if case == "R4" else (135, (1, 1), (2, 2))
        )
        source = torch.randn(
            batch,
            9,
            compressed[0] * compressed[1],
            4096,
            device="cuda",
            dtype=torch.bfloat16,
        )
        cotangent = torch.randn(
            batch, 9, *target, 4096, device="cuda", dtype=torch.bfloat16
        )
        reference = None
        for mode in args.modes:
            module = SpatialDecompressor(4096).cuda()
            if mode == "weight_cl3d":
                torch.nn.utils.convert_conv3d_weight_memory_format(
                    module, torch.channels_last_3d
                )
            layouts = {}

            def layout_hook(_module, inputs, *, mode=mode, layouts=layouts):
                volume = inputs[0]
                if mode.startswith("input_ncdhw"):
                    volume = volume.contiguous()
                layouts.update(
                    input_shape=list(volume.shape),
                    input_stride=list(volume.stride()),
                    input_ncdhw=volume.is_contiguous(),
                    input_cl3d=volume.is_contiguous(
                        memory_format=torch.channels_last_3d
                    ),
                    weight_stride=list(_module.weight.stride()),
                )
                return (volume,)

            handle = module.refine.register_forward_pre_hook(layout_hook)
            token = source.detach().requires_grad_(True)
            row = {
                "case": case,
                "mode": mode,
                "token_shape": list(source.shape),
                "cotangent_shape": list(cotangent.shape),
                "iterations": [],
            }
            try:
                with torch.backends.cudnn.flags(
                    enabled=mode != "no_cudnn",
                    benchmark="benchmark" in mode,
                    deterministic=False,
                ):
                    for index in range(4):
                        module.zero_grad(set_to_none=True)
                        token.grad = None
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output = module(token, compressed, target)
                        torch.cuda.synchronize()
                        middle = time.perf_counter()
                        output.backward(cotangent)
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                        row["iterations"].append(
                            {
                                "index": index,
                                "warmup": index == 0,
                                "forward_s": middle - start,
                                "backward_s": end - middle,
                                "peak_allocated_gib": torch.cuda.max_memory_allocated()
                                / 2**30,
                                "peak_reserved_gib": torch.cuda.max_memory_reserved()
                                / 2**30,
                            }
                        )
                        print(case, mode, row["iterations"][-1], flush=True)
                        if index < 3:
                            del output
                    tensors = {
                        "output": output,
                        "dx": token.grad,
                        "dw": module.refine.weight.grad,
                        "db": module.refine.bias.grad,
                    }
                    if reference is None:
                        reference = {
                            key: value.detach().cpu() for key, value in tensors.items()
                        }
                    row["numerical_comparison_to_current"] = {
                        key: compare(value, reference[key])
                        for key, value in tensors.items()
                    }
                    row.update(status="PASS", layouts=layouts)
                    del tensors, output
            except Exception as error:  # noqa: BLE001 - diagnostic records unsupported backend
                row.update(status="FAIL", error=repr(error), layouts=layouts)
                print(case, mode, "FAIL", repr(error), flush=True)
            finally:
                handle.remove()
            results.append(row)
            dump(args.output_dir / "results.json", results)
            del module, token
            gc.collect()
            torch.cuda.empty_cache()
        del source, cotangent, reference
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
