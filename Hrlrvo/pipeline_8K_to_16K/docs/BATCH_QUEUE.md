# Batch Outpainting / FlashVSR Queue

## Purpose

`run_batch.sh` overlaps one Outpainting job with one FlashVSR job while preserving the single certified stage implementation. The batch runner does not import model code and does not implement either model. Each stage is a separate OS process launched through:

```text
run_batch.py
  -> run_cert.sh --config <job.json> --stage outpaint [--resume]
  -> run_cert.sh --config <job.json> --stage upscale [--resume]
```

This process separation isolates Python, NumPy, PyTorch, Outpainting, and FlashVSR RNG state between concurrently running stages.

## Invocation

```bash
cd /home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K
./run_batch.sh --config configs/batch/certification.json
```

Resume both queue state and child stage manifests with:

```bash
./run_batch.sh --config configs/batch/certification.json --resume
```

Only one batch process can own the workspace. A non-blocking exclusive lock is held at `state/batch_runner.lock`; a second runner exits without starting children.

## Configuration contract

```json
{
  "jobs": [
    "configs/certification/05.json",
    "configs/certification/04.json",
    "configs/certification/03.json",
    "configs/certification/02.json",
    "configs/certification/01.json"
  ],
  "outpaint_workers": 1,
  "upscale_workers": 1,
  "queue_policy": "fifo"
}
```

The runner rejects any worker count other than exactly one and any policy other than `fifo`. Each job config must contain `job_name` and `input.video_id`.

The current user-selected order is **05 → 04 → 03 → 02 → 01**, so video05 is the first final-output candidate. Only the batch job list was reordered; per-video seeds, Full frame limits, model settings and the queue implementation are unchanged. In a fresh successful run both Outpainting and Upscale follow this order. Existing audit evidence containing the former01→05 order remains historical.

## Scheduling semantics

There are two independent worker threads, but every model stage runs in its own child process:

1. The producer invokes Outpainting for the next video.
2. On success, it atomically records `OUTPAINT_DONE`, then `UPSCALE_WAITING`, and enqueues that video.
3. The producer immediately starts the next Outpainting job without waiting for FlashVSR.
4. The consumer takes successful Outpainting jobs in FIFO completion order and invokes FlashVSR.
5. A failed Outpainting is marked `FAILED_OUTPAINT` and is never enqueued.
6. A failed FlashVSR is marked `FAILED_UPSCALE`; the consumer continues with the next waiting item.

Therefore the maximum model concurrency is one Outpainting plus one FlashVSR. The implementation cannot start two simultaneous Outpainting children or two simultaneous FlashVSR children.

Batch operation intentionally starts from already copied and validated canonical PreProc artifacts. The `outpaint` child is responsible for validating that canonical input before model load. The batch runner does not invent a `PREPROCESS_READY` state without successful stage validation.

## State and resume

`state/batch_request.json` binds resume to the exact batch JSON SHA256, ordered job list, and each job JSON SHA256. A fresh invocation refuses to overwrite an existing request or queue state; use `--resume` for recorded work. A resume is rejected if any bound configuration bytes changed.

`state/batch_state.json` is rewritten through a temporary file, `fsync`, and atomic rename at every transition. Its values are:

- `PENDING`
- `PREPROCESS_READY`
- `OUTPAINT_RUNNING`
- `OUTPAINT_DONE`
- `UPSCALE_WAITING`
- `UPSCALE_RUNNING`
- `DONE`
- `FAILED_OUTPAINT`
- `FAILED_UPSCALE`

On `--resume`, a previously completed Outpainting (`OUTPAINT_DONE`, `UPSCALE_WAITING`, or queue-level `DONE`) is queued for the certified upscale stage without rerunning Outpainting. Even queue-level `DONE` is re-entered through `--stage upscale --resume`; queue JSON alone never bypasses the stage's real artifact/manifest validation. Interrupted or failed work is retried through the corresponding certified stage with `--resume`, allowing that stage's own manifests to decide artifact reuse.

Without `--resume`, queue state starts fresh. Child stages do not receive `--resume` and therefore follow their normal fresh-run policies.

## Logging and interruption

Queue transitions are printed and appended to `logs/batch_pipeline.log`. Child stdout and stderr are merged and streamed line by line with video and stage prefixes; output is not accumulated in memory.

Typical transitions are:

```text
[QUEUE] start_outpaint video=05
[QUEUE] outpaint_done video=05
[QUEUE] enqueue_upscale video=05
[QUEUE] start_outpaint video=04
[QUEUE] start_upscale video=05
[QUEUE] upscale_done video=05
```

On keyboard interruption, no unrelated process is signalled. The runner requests that no new jobs start, waits for already-running child processes to finish normally, persists their final transition, and leaves queued work as `UPSCALE_WAITING` for a later `--resume`.

The batch runner never kills GPU processes, resets a GPU, modifies FlashVSR tile selection, or changes model parameters.
