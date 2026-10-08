# DOVE 4K → 16K benchmark specification

## 1. Scope

This lane evaluates the official pretrained **DOVE Stage-2** one-step video
diffusion model without training or an image-SR fallback. It adds DOVE to the
same launcher, output validation, timing/VRAM report, source-preservation
metrics, temporal sanity checks, and contact-sheet comparison used by
FlashVSR v1.1, SeedVR2, Real-ESRGAN, and HAT.

The bounded run is deliberately DOVE-only:

```bash
cd /home/work/ai-graphics_33/Hrlrvo/Upscaler/architect
./run_dove_benchmarks.sh
```

It executes `micro` and then `mini` serially on one GPU. It does **not** rerun
the four completed models and does not start a 92-frame full benchmark.

## 2. Official sources and immutable revisions

- Code: <https://github.com/zhengchen1999/DOVE>
  at `0cd4240442cb5d122839c279977142cb6d648987`
- Paper: <https://arxiv.org/abs/2505.16239> (NeurIPS 2025)
- Official Stage-2 archive: Google Drive file
  `1Nl3XoJndMtpu6KPFcskUTkI0qWBiSXF2`, expected 30,169,469,563 bytes
- Author Hugging Face model revision:
  `ZhengChen1999/DOVE@55ea86b7b2fbad77019a300e785b7fe1be065924`
- Official base VAE revision:
  `zai-org/CogVideoX1.5-5B@fdc5267c90b5c06492985b966e43aae984e189e0`

The Google Drive download was quota-blocked on 2026-09-07. The reproducible
fallback assembly therefore uses the DOVE author's published transformer,
text encoder, tokenizer, and scheduler plus the VAE declared by the official
CogVideoX1.5-5B base. It is not silently represented as the unavailable zip;
component revisions and local hashes are recorded in
`models/DOVE/checkpoint/provenance.json`.

## 3. Fixed benchmark units

| Mode | Real source frames | Reason |
|---|---:|---|
| micro | 9 | Smallest unpadded DOVE video unit satisfying `F = 8n + 1`; validates temporal attention instead of a degenerate single image. |
| mini | 17 | Adds one native eight-frame unit while remaining one temporal chunk; allows an incremental cost fit. |

For a long run the locked planning assumption is temporal window 17, overlap
8, stride 9. The micro/mini measurements fit the first nine-frame cost and the
incremental eight-frame cost. Long-video estimates explicitly charge the
warm-up again for every overlapping temporal chunk.

## 4. Exact neural and tiling path

The official computation order is retained:

```text
selected 3840×2160 RGB frames
→ bilinear x4 conditioning in the 15360×8640 output domain
→ output-domain spatial tiles
→ CogVideoX VAE encode
→ DOVE transformer once at sr_noise_step=399
→ scheduler velocity reconstruction
→ CogVideoX VAE decode
→ discard half-overlaps and place each valid center region exactly once
→ stream 15360×8640 frames into the selected lossless/lossy encoder
```

Fixed initial spatial plan:

```text
tile height 768, tile width 1360, overlap 32 output pixels
```

These dimensions are in the **16K output domain**, matching official DOVE
semantics. This is not a 1360×768 source tile. If CUDA OOM occurs, at most two
documented reductions are allowed. Resolution, frame count, neural model, and
native x4 path may never be changed. There is no resize-only success path.

The upstream script materializes three full-resolution FP32 video tensors.
For 9 frames alone that is about 40.05 GiB, and for 17 frames about 75.65 GiB,
before inference activations and encoding copies. The wrapper may replace only
that assembly mechanism with disk-backed staging/stream encoding; it must not
change DOVE's VAE/transformer/scheduler/decode computation.

## 5. Required measurements and validation

Each DOVE report must include:

```text
model_load_seconds
neural_inference_seconds
decode_seconds
merge_seconds
encode_seconds
total_wall_seconds

peak_torch_allocated_mib
peak_torch_reserved_mib
peak_process_gpu_mib
peak_total_gpu_used_mib

effective_input_frames_processed
effective_output_frames_written
spatial_tile_count
temporal_chunk_count
native_temporal_unit_count

pure_inference_seconds_per_input_frame
end_to_end_seconds_per_input_frame
pure_inference_seconds_per_output_frame
end_to_end_seconds_per_output_frame
```

PASS additionally requires exact `15360×8640`, 9 or 17 output frames, source
24 fps, official checkpoint provenance, and completion of the same automatic
quality analysis as every other runnable model. The shared comparison uses the
identical leading five frames for `micro` and nine for `mini`, because SeedVR2
has the shortest valid clips. Qualitative contact sheets use matched source
indices and native-detail center crops.

The automatic SSIM/PSNR/MAD/detail and temporal metrics are diagnostics after
downsampling to source resolution. They do not substitute for human review or
ground-truth 16K evaluation.

## 6. Failure and stop policy

- Do not retry the unavailable Google Drive artifact indefinitely.
- Do not start while less than 100,000 MiB GPU memory is free.
- A CUDA OOM can reduce only the spatial tile plan, at most twice.
- A missing/mismatched checkpoint component is a fail-closed error.
- No lower-resolution output, classical resize, or frame-image fallback is
  accepted as DOVE success.
- After both runs, regenerate the common quantitative and qualitative reports
  from all available PASS model results.

Expected artifacts:

```text
output/benchmark/DOVE/*_DOVE_x4_micro_f000000_n9.*
output/benchmark/DOVE/*_DOVE_x4_mini_f000000_n17.*
result/benchmark/DOVE/benchmark_micro.{json,md}
result/benchmark/DOVE/benchmark_mini.{json,md}
result/benchmark/DOVE/run_{micro,mini}.log
result/benchmark/DOVE/final_{comparison.md,completion.json}
result/benchmark/run_status_dove.json
result/benchmark/{performance,quality}_comparison.*
result/benchmark/qualitative/*contact_sheet.png
```

`run_dove_benchmarks.sh` invokes the final reporter even on failure. On success
it also replaces the marked DOVE addendum in `result/benchmark/summary.md`, so
the tmux job leaves a numeric completion report without requiring an attached
Codex process.
