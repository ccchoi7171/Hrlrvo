"""CPU-only, real monocular depth candidates for one fixed C0 RGB frame."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference/candidates/depth"
IMAGE = ROOT / "reference/c0_frame000.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--kind", choices=("relative", "metric_indoor"), default="relative")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This script requires CUDA_VISIBLE_DEVICES='' and never uses GPU")
    if not args.name.replace("_", "").isalnum():
        raise ValueError("Candidate name must be alphanumeric/underscore")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=False)
    image = Image.open(IMAGE).convert("RGB")
    print(json.dumps({"event": "loading", "model": args.model, "device": "cpu", "size": args.size}), flush=True)
    load_started = time.monotonic()
    processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, trust_remote_code=False, use_safetensors=True)
    model.to(device="cpu", dtype=torch.float32).eval()
    load_seconds = time.monotonic() - load_started
    inputs = processor(images=image, size={"height": args.size, "width": args.size}, return_tensors="pt")
    if any(value.device.type != "cpu" for value in inputs.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("Depth inputs must stay on CPU")
    started = time.monotonic()
    with torch.inference_mode():
        outputs = model(**inputs)
        # Official processor restores the relative-depth prediction to original H,W.
        result = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])[0]
        depth = result["predicted_depth"].float().cpu().numpy()
    seconds = time.monotonic() - started
    if depth.shape != (1080, 1920) or not np.isfinite(depth).all() or np.ptp(depth) <= 0:
        raise RuntimeError("Invalid native-resolution depth prediction")
    np.save(out / "depth_raw.npy", depth.astype(np.float32), allow_pickle=False)
    lo, hi = np.percentile(depth, [0.5, 99.5]).tolist()
    normal = np.clip((depth - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)
    np.save(out / "prediction_display_normalized.npy", normal, allow_pickle=False)
    # Relative Depth Anything predicts inverse depth; metric model predicts
    # distance. Normalize their DISPLAY orientation to white=near, never raw data.
    near = normal if args.kind == "relative" else 1.0 - normal
    gray = np.rint(near * 255).astype(np.uint8)
    Image.fromarray(gray).save(out / "depth_gray.png")
    Image.fromarray(255 - gray).save(out / "depth_white_far.png")
    Image.fromarray(np.rint(near * 65535).astype(np.uint16)).save(out / "depth_gray_16bit.png")
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out / "depth_color.png"), color)
    overlay = cv2.addWeighted(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), 0.55, color, 0.45, 0)
    cv2.imwrite(str(out / "depth_overlay.png"), overlay)
    processor.save_pretrained(out / "processor_config")
    model.config.to_json_file(out / "model_config.json")
    report = {
        "name": args.name, "model_id": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
        "source": str(IMAGE), "source_sha256": hashlib.sha256(IMAGE.read_bytes()).hexdigest(),
        "device": "cpu", "dtype": "float32", "model_class": type(model).__name__,
        "processor_class": type(processor).__name__, "weights": "pretrained safetensors; no training or diffusion",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "requested_processor_size": args.size, "actual_input_shape": list(inputs["pixel_values"].shape),
        "prediction_kind": args.kind,
        "raw_network_output_shape": list(outputs.predicted_depth.shape), "saved_depth_shape": list(depth.shape),
        "load_seconds": load_seconds, "inference_and_postprocess_seconds": seconds,
        "raw_min_max": [float(depth.min()), float(depth.max())],
        "display_percentile_clip": [0.5, 99.5], "display_clip_values": [lo, hi],
        "display_note": "Normalized independently per model; grayscale white=near, black=far. Relative models have no meters; metric-indoor model values are unvalidated model distance estimates, not measured ground truth. Color/inverted versions are the SAME prediction, not additional models.",
        "artifacts": [p.name for p in sorted(out.iterdir())],
    }
    (out / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "complete", **report}), flush=True)
    del model, inputs, outputs
    gc.collect()


if __name__ == "__main__":
    main()
