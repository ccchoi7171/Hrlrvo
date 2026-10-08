"""CPU-only evidence writer; compares existing outputs, never runs inference."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_latents(before, after):
    a = torch.load(before, map_location="cpu", weights_only=True)
    b = torch.load(after, map_location="cpu", weights_only=True)
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("Expected actual generated-latent tensor artifacts")
    same_shape = a.shape == b.shape
    same_dtype = a.dtype == b.dtype
    equal = same_shape and same_dtype and torch.equal(a, b)
    result = {
        "before": str(before),
        "after": str(after),
        "before_sha256": sha256(before),
        "after_sha256": sha256(after),
        "before_shape": list(a.shape),
        "after_shape": list(b.shape),
        "before_dtype": str(a.dtype),
        "after_dtype": str(b.dtype),
        "tensor_bitwise_equal": equal,
        "container_sha_required": False,
    }
    if same_shape and not equal:
        difference = (a.float() - b.float()).abs()
        result.update(
            max_abs_diff=float(difference.max()),
            mean_abs_diff=float(difference.mean()),
            allclose_diagnostic_only=bool(torch.allclose(a.float(), b.float())),
        )
    return result


def compare_native(before, after, frames):
    a = np.load(before, mmap_mode="r", allow_pickle=False)
    b = np.load(after, mmap_mode="r", allow_pickle=False)
    if a.dtype != np.uint8 or b.dtype != np.uint8 or a.shape[1:] != b.shape[1:]:
        raise ValueError("Native segment shape/dtype mismatch")
    if len(a) != frames or len(b) != frames:
        raise ValueError("C0 segment does not contain exactly the requested frames")
    left, right, matched, mismatches = hashlib.sha256(), hashlib.sha256(), 0, []
    for index in range(frames):
        left.update(a[index].tobytes())
        right.update(b[index].tobytes())
        if np.array_equal(a[index], b[index]):
            matched += 1
        else:
            mismatches.append(index)
    return {
        "before": str(before),
        "after": str(after),
        "dtype": "uint8",
        "shape": list(a.shape),
        "matched_frames": matched,
        "frames": frames,
        "before_pixel_sha256": left.hexdigest(),
        "after_pixel_sha256": right.hexdigest(),
        "mismatch_frame_indices": mismatches,
        "exact": matched == frames,
    }


def decoded_hashes(path, frames):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-hwaccel",
        "none",
        "-threads",
        "8",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frames),
        "-an",
        "-pix_fmt",
        "rgb24",
        "-fps_mode",
        "passthrough",
        "-f",
        "framemd5",
        "-",
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    hashes = [
        line.split(",")[-1].strip()
        for line in process.stdout.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(hashes) != frames:
        raise ValueError(f"Decoded {len(hashes)} frames instead of {frames}: {path}")
    return hashes, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU comparison must not initialize CUDA")
    torch.set_num_threads(2)
    a, b = args.baseline.resolve(), args.candidate.resolve()
    latent = Path("chunks/latents/chunk_000000/generated_latent.pt")
    native = Path("chunks/native/segment_000000.npy")
    final = Path("chunks/native/05_Outpaint_4K_24fps.mp4")
    result = {
        "video": "05",
        "seed": 71634910,
        "frames": 65,
        "generated_latent": compare_latents(a / latent, b / latent),
        "native_rgb": compare_native(a / native, b / native, 65),
    }
    left, left_command = decoded_hashes(a / final, 65)
    right, right_command = decoded_hashes(b / final, 65)
    mismatches = [
        index for index, (x, y) in enumerate(zip(left, right, strict=True)) if x != y
    ]
    result["decoded_native_rgb"] = {
        "before": str(a / final),
        "after": str(b / final),
        "before_command": left_command,
        "after_command": right_command,
        "before_frame_md5": left,
        "after_frame_md5": right,
        "matched_frames": 65 - len(mismatches),
        "frames": 65,
        "mismatch_frame_indices": mismatches,
        "exact": not mismatches,
        "comparison": "native24fps sequential decoded rgb24, index k against index k; no retime used",
    }
    passed = (
        result["generated_latent"]["tensor_bitwise_equal"]
        and result["native_rgb"]["exact"]
        and result["decoded_native_rgb"]["exact"]
    )
    result["verdict"] = "C0 EXACT PRESERVED" if passed else "C0 NOT EXACT"
    result["cuda_initialized"] = torch.cuda.is_initialized()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {"verdict": result["verdict"], "evidence": str(args.output)}, indent=2
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
