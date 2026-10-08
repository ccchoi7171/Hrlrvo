# Final Hrlrvo v8 Training / Resume / Inference CLI

## Environment and tmux

These are real `argparse` options in `main/src/ltx_outpaint/train.py` and `inference.py`.
The training config is a **positional argument**; inference uses `--config`.
Verified on 2026-09-11: real 4K training steps1–5, full-state resume5→6,
two initialized-adapter videos (two recurrent chunks each, 30 steps/chunk),
and actual checkpoint5-loaded 65-frame/30-step inference all **PASS**.
Runtime evidence: [final report](../outputs/final_runtime_validation/FINAL_RUNTIME_VALIDATION_REPORT.md).
The third input was excluded at the user's request. No long production training was started.

Subsequent user-requested schedule update (2026-09-11): the production cache audit
counts **30,024 compatible GT records and 30,024 matching Anchor records**, no missing IDs.
The user's final production budget is **30,000 optimizer steps** at batch1/accumulation1/world1:
30,000 of30,024 records (99.92% of one dataset pass); the remaining24 are not reached in this run.
Each record contributes its FIRST 65 RGB frames / first 9 latent frames, not every
temporal window of its full-length GT cache. Count evidence:
`outputs/training_schedule_one_epoch/cache_count_audit.json`.
Historical 5+1-step smoke artifacts retain their original 2,000-step scheduler horizon;
they have not been rewritten to claim a new runtime test.

Start a new shell session:

```bash
tmux new -s hrlrvo_train
cd /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting
source /home/work/ai-graphics_33/miniconda3/etc/profile.d/conda.sh
conda activate Hrlrvo
export PYTHONPATH="$PWD/main/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
nvidia-smi
```

Use the H200 only when no conflicting workload is present. These commands do not kill/pause cache jobs or other processes. `Ctrl-b d` detaches (does not stop training); reattach with `tmux attach -t hrlrvo_train`. To intentionally stop the foreground training process use Ctrl-C in its session; resume from the last complete checkpoint, not a partially written checkpoint.

## 1. New production training

```bash
python -u -m ltx_outpaint.train \
  main/configs/train_4k.yaml \
  --output-dir /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/hrlrvo_v8 \
  --disable-progress-bars
```

The run refuses to silently reuse an existing `run_manifest.json`; use Resume instead. A deliberately new run needs a different `--output-dir`.

Current policy, not a tuning recommendation:

| Setting | Value |
|---|---|
| Micro batch / accumulation / single-device world size | 1 / 1 / 1 |
| Effective batch | 1 sample/optimizer step |
| Dataset | Production 4K GT + independently encoded 2K Anchor, FIRST-only |
| RGB / latent temporal slice | `[0:65)` / `[0:9)` |
| Spatial latent / Anchor extent | 68×120 / y17:51, x30:90 |
| Temporal prefix | 17 RGB = 3 latent, probability 0.5 |
| Precision / gradient checkpointing | BF16 / ON |
| Optimizer | Native torch AdamW, one group, LR 1e-4 |
| Training length | 30,000 optimizer steps = 30,000 records (of30,024 available) |
| Scheduler | Native LinearLR, start factor1, end factor0.1, 30,000 optimizer-step horizon |
| Max grad norm | 1.0 |
| Flow loss / timestep sampler | Known-excluded active-element MSE / shifted_logit_normal |
| Checkpoint policy | Every **500 optimizer steps**; 500×1×1×1 = **500 samples** |
| Validation during training | Disabled (`samples=[]`, interval null) |

No module-specific LR multipliers. After optimizer step `k`, next LR is
`1e-4*(1-0.9*k/30000)`; it starts at1e-4 and reaches1e-5 at completion.
Checkpoint cadence remains500 optimizer steps (500 samples): steps500,1000,...,30000;
the final checkpoint is **checkpoint-030000**.
This is an explicit step budget, not automatic dataset-size discovery.

The measured training peak was 129.318 GiB allocated / 137.951 GiB reserved.
The allocator retains substantial reusable memory; do not run another large GPU job concurrently.
The full READY/provenance inventory may take roughly 20–25 minutes before the first
batch on this filesystem. This is startup validation, not a stalled optimizer step.

Outputs under `runs/hrlrvo_v8/`:
- `run_config.yaml`, `run_manifest.json`
- `checkpoints/checkpoint-000500/`, `checkpoint-001000/`, ..., final `checkpoint-030000/`

Production custom diagnostics are **off by default**. No health logger, Fusion observation
hooks, gradient audits, update probes, custom timers/memory metrics, `OPTIMIZER_STEP`
summary, `train_metrics.jsonl`, or custom `train_console.log` tee is installed. The
native LTX console remains active; with `--disable-progress-bars`, its loss/LR/Time
status is printed every **20 optimizer steps**, not every step. Checkpoint-required
sample/batch counters remain ordinary Python integer bookkeeping.

The costly historical health collection is available only with explicit `--diagnostics`
for debugging; do not add that flag to normal production commands. `--stop-after-step`
alone does not enable diagnostics. Historical validation artifacts are unchanged.

```bash
tmux capture-pane -p -J -t hrlrvo_train -S -80
```

## 2. Resume production training

Change only the `CHECKPOINT` value to an existing **complete checkpoint directory**:

```bash
CHECKPOINT=/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/hrlrvo_v8/checkpoints/checkpoint-000500
python -u -m ltx_outpaint.train \
  main/configs/train_4k.yaml \
  --resume "$CHECKPOINT" \
  --disable-progress-bars
```

Resume restores the original run directory, FP32 adapter/new-module weights, Adam state, scheduler, Accelerator/RNG state and step/sample counters. It retains the saved deterministic READY-before-MISSING order even if additional Anchor cache files now exist. Changed architecture, provenance, config fingerprint (including LR/horizon), dataset/order or incompatible world size is rejected. An adapter-only safetensors file is **not** a full training resume checkpoint. Do not pass a different output directory when resuming.

Do not resume the historical 2,000-step smoke run using the new30,000-step production
config: the scheduler/config mismatch is intentionally rejected. Its weights remain
valid for inference, whose architecture is unchanged. To resume that old run for a
bounded diagnostic, use its original `run_config.yaml` snapshot instead.

The official loop's generic startup wording is not authoritative for resume: `runtime_result_start_*.json`, loaded scheduler/optimizer state and optimizer-step metrics are the actual evidence.

## 3. Inference environment

In a separate terminal (not concurrently with GPU training):

```bash
tmux new -s hrlrvo_infer
cd /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting
source /home/work/ai-graphics_33/miniconda3/etc/profile.d/conda.sh
conda activate Hrlrvo
export PYTHONPATH="$PWD/main/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

All examples retain 4K geometry, 48 blocks, 30 denoising steps, fixed-empty positive/negative text, CFG/STG and recurrent17f prefix. With no `--num-frames`, all input frames are processed. A partial chunk uses `repeat_last` only up to the smallest VAE-aligned working length (`1 + 8k` frames, at most 65), and appends only real new frames. Thus `--num-frames 17` and `--num-frames 1` run 17- and 1-frame denoising rather than a padded 65-frame denoising pass. `--output` is an **MP4 filename**, not a directory.

LongRange input/output buffering is bounded by the configured chunk and prefix,
not total video length. Finalized frames are atomically stored as lossless uint8
RGB segments, then streamed through the existing PyAV/x264 settings. The exact
decoded FP32 recurrent prefix and post-chunk RNG state are stored separately and
are never reconstructed from the segments. Intermediates are removed only after
successful final encoding and manifests. Add `--keep-intermediates` to retain
them; after an interruption, rerun the identical command with `--resume`.
`--segment-dir PATH` overrides the default hidden directory beside the output.

## 4. Pre-training inference (explicit initialized adapters)

```bash
mkdir -p outputs/pretraining_baseline/video_01
set -o pipefail
python -u -m ltx_outpaint.inference \
  --config main/configs/inference_4k.yaml \
  --anchor-video input/anchor/ef334007-aa0b-432d-a7e8-76a3eeeec252.mp4 \
  --output outputs/pretraining_baseline/video_01/generated.mp4 \
  --initialized-adapters --seed 42 --steps 30 \
  2>&1 | tee outputs/pretraining_baseline/video_01/console.log
```

`--initialized-adapters` is mutually exclusive with `--checkpoint`; missing checkpoints never silently fall back to fresh adapters. Initialization is seeded before model installation. This is not a trained-quality inference claim.

## 5. Training-checkpoint inference

Change `CHECKPOINT` once. A checkpoint directory or its exact `adapter_model.safetensors` file is accepted.
The production `checkpoint-000500` example becomes available only after training reaches
that step. For the already validated checkpoint, use:
`/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/outputs/final_runtime_validation/training_smoke/validated_run/checkpoints/checkpoint-000005`.

```bash
CHECKPOINT=/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/hrlrvo_v8/checkpoints/checkpoint-000500
mkdir -p outputs/checkpoint_inference/video_01
set -o pipefail
python -u -m ltx_outpaint.inference \
  --config main/configs/inference_4k.yaml \
  --anchor-video input/anchor/ef334007-aa0b-432d-a7e8-76a3eeeec252.mp4 \
  --output outputs/checkpoint_inference/video_01/generated.mp4 \
  --checkpoint "$CHECKPOINT" --seed 42 --steps 30 \
  2>&1 | tee outputs/checkpoint_inference/video_01/console.log
```

For the second validated input, replace `--anchor-video` and use a separate output directory:
- `input/anchor/0f2313f5-5a61-4ccc-811c-f3bbf1d41151.mp4` → video_02

The third input is excluded from this validation at the user's request.

Each directory receives `generated.mp4`, `inference_manifest.json`, `chunk_lineage.json`, and the console log. No nonempty `--prompt` is allowed. Text CA and the required video/audio text-connector streams remain structurally active; audio generation stays disabled.

## Historical bounded runtime validation commands (2,000-step horizon)

The original smoke ran before the production horizon was extended. Its original
config is preserved in `validated_run/run_config.yaml`. Do not repeat a new-run
command into this existing directory; it intentionally refuses overwrite. To resume
an existing complete historical checkpoint while retaining its original horizon:

```bash
python -u -m ltx_outpaint.train \
  "$PWD/outputs/final_runtime_validation/training_smoke/validated_run/run_config.yaml" \
  --resume "$PWD/outputs/final_runtime_validation/training_smoke/validated_run/checkpoints/checkpoint-000005" \
  --stop-after-step 6 --disable-progress-bars
```

The above5→6 resume was already completed; it is shown for CLI semantics, not an instruction
to replay into existing output artifacts. `--stop-after-step` is an absolute optimizer-step
stop and does not change the horizon of the supplied config. Five-step weights are
runtime compatibility evidence, not a production-quality model.
