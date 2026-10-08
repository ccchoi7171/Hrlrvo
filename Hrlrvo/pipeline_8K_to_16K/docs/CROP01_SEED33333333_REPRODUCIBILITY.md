# Cropped 01: PreProc + VideoOutpainting, seed 33333333

This is an isolated **stage-only** request. It does not dispatch FlashVSR or
modify `src/cert_pipeline/preprocess.py`, `run_cert.py`, `run_batch.py`, or the
certification configuration. The active run is in the `hrlrvo_train` tmux
session, window `crop01_33333333`.

## Request identity

- Current cropped input: `data/00_Input_8K_7680x4320/01_Input_8K.mp4`, SHA-256
  `469853124f36e1b954bc312e63f28546da6365dbee0b1439d3ef10a6086e9124`.
- Preserved original: `data_backUP01/00_Input_8K_7680x4320/01_Input_8K.mp4`,
  SHA-256 `2d73992d09e37442fa7f0821711be0bb69584c6c5683ead99a8ad72ffe218d0c`.
- The current input was made from that original by cropping 768 pixels on each
  side and 864 pixels at the top (`crop=6144:3456:768:864`), retaining the
  bottom edge, then Lanczos scaling to 7680×4320. It has 1,952 frames at
  16 fps; the original AAC audio was copied. The input SHA above, not the old
  PreProc JSON's SHA, is the binding for this request.
- Fixed request: `configs/experiments/01_crop_seed33333333.json`. It equals
  `configs/certification/01.json` except for `seed=33333333`,
  `job_name=01_crop_seed33333333`, and the matching `output_root`.
- `state/runs/01_crop_seed33333333/requested_config.json` stores the executed
  request. Resume refuses a different request or input SHA.

## Stage sequence and outputs

1. `run_preprocess_stage()` calls the **unchanged**
   `src/cert_pipeline/preprocess.py` historical CRF 18 contract. It preserves
   all 1,952 frame indices, resizes to 1920×1080, retimes 16→24 fps without
   interpolation, and speeds up AAC audio by 1.5×. Its full-decode and
   producer/toolchain checks run before publication. The new result is at
   `data/01_PreProc_2K_1920x1080/metadata/01/generated/01_PreProc_2K.mp4`;
   the old canonical result remains under `data_backUP01`.
2. `run_outpaint_stage()` consumes that verified generated PreProc. The
   unchanged model/checkpoint and 30-step 65/17/48 LongRange/recurrence
   contract process all 1,952 frames with base seed 33333333. Native chunks
   and merged master remain 3840×2160/24 fps; only the complete master is
   retimed to the 3840×2160/16 fps final deliverable at
   `data/02_Outpaint_4K_3840x2160/01/final/01_Outpaint_4K.mp4`.
3. The run stops after Outpainting. There is no call to `run_upscale_stage()`.

The original `run_batch.py`/`run_batch.sh` queue is not used for this request:
its producer automatically enqueues Upscale, requires one Upscale worker, and
its child `run_cert.sh` rejects an override of the fixed certification-01
seed. This run uses the **same shared PreProc and Outpainting stage functions**
through `tools/run_crop01_seed33333333.py`, with the three request differences
explicitly checked against the validated certification template. No broad
seed-validation or queue-policy change is made.

## Run, inspect, resume

```bash
cd /home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K
./run_crop01_seed33333333.sh                 # fresh only when target/state are absent
./run_crop01_seed33333333.sh --resume        # same input/config, after a stopped run
cat state/runs/01_crop_seed33333333/status.json
tail -f state/runs/01_crop_seed33333333/execution.log
tmux attach -t hrlrvo_train
```

Do not start a second worker while the tmux run is active. Do not use
`--force`, edit native chunk state, change the input/config/code/checkpoint,
or delete partial files to bypass a failed resume. If asked to stop, inspect
the live process and the latest `OUTPAINT_CHUNK_DONE`/native commit before
signaling only this run. Resume validates the contiguous completed chunks and
restores prefix/RNG/carry; an in-flight chunk may have to be recalculated.
There is no denoising-step-level resume or guaranteed zero-work-loss stop.

The PreProc provenance JSON and Outpainting `stage_request.json`, `run.json`,
`inference_manifest.json`, `seed.json`, and `stage_complete.json` provide the
execution evidence. Repeating the same request is reproducible at the
configuration/lineage level; bitwise GPU output identity should be measured
in an actual repeat rather than assumed.
