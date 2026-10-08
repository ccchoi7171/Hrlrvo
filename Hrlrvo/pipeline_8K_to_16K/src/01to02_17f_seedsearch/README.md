# Reproducible 17-frame 4K prefix seed search

The Python files in this directory **only wrap** `VideoOutpainting/main/src/ltx_outpaint/inference.py` with `--mode prefix_seed_search`. No model, denoising, conditioning or VAE math is duplicated.

## Fixed inputs and inference identity

- Input mapping: `common.py` maps `01`–`05` only to `pipeline/data/01-1_FPS_2K_1920_1080_24fps/XX_Input_2K.mp4`.
- Inference config: `VideoOutpainting/main/configs/inference_4k.yaml` (24 fps, canvas 3840×2160, prefix 17 frames). **Not** `train_4k.yaml`.
- Adapter checkpoint: `VideoOutpainting/runs/hrlrvo_main_train_v1/checkpoints/checkpoint-007500`.
- CLI overrides: 30 steps; `--video-only-gpu-weights --vae-decode-streaming --no-ffn-token-chunking`.
- The native prefix mode only loads source frames `[0,17)`; preflight separately full-decodes each input with streaming FFmpeg, without retaining all frames in RAM.

## Execution

Use the `Hrlrvo` conda environment. One video/seed:

```bash
pipeline/config/01/01_seedsearch_4K.sh 33333333
```

The command refuses to overwrite an existing valid candidate. To *actually regenerate* the same seed for comparison, use a different output root while keeping every other argument identical:

```bash
SEEDSEARCH_OUTPUT_ROOT=/path/to/reproduction pipeline/config/01/01_seedsearch_4K.sh 33333333
```

For a safe resume, a valid candidate is checked against its MP4/tensor hashes, geometry, fps, 17-frame full decode, input/config/adapter hashes, steps and schedule identity before it is skipped. Partial or corrupt results are never counted as complete. The FP32 tensor plus adjacent metadata retains the native `load_prefix_artifact()` contract, despite the requested `XX_seed_4K_prefix.pt` name.

Orchestrator (run from the same environment):

```bash
python pipeline/src/01to02_17f_seedsearch/orchestrate_seedsearch.py --phase bootstrap
```

`bootstrap` performs all-five preflight, Smoke A (video 01 / round 33 slot 3), Smoke B (video 01 / round 1, three independent OS processes), then the 33-round run. It does **not** launch the long run if either smoke fails. To resume after an interruption:

```bash
python pipeline/src/01to02_17f_seedsearch/orchestrate_seedsearch.py --phase run
```

The global seed schedule is created once under output-root `_schedule/`, from `random.Random(33333333).sample(range(1, 2**31), 99)`. The file and its SHA-256 are immutable for this output root. `run_context.json` blocks resuming with changed input/config/checkpoint/runtime identity. Each round processes videos `01` through `05` sequentially, with at most three independent B=1 processes for one video at a time.

Progress: `data/02_seedSearch_4K_17f_3840_2160/_manifests/progress.json` and `_logs/orchestrator.log`. Candidate records: `XX/XX_seed_4K_meta.json`, with `_manifests/run_manifest.jsonl` and per-run logs under `_logs/XX/`.
