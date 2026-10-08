# 4K → 16K video-SR + frame-image-SR minimal benchmark 명세

## 0. 실행 계약과 성공 조건

최종 실행 파일은 `run_benchmark_minimal.sh` 하나이다. 기본 실행은 다음
순서를 **한 GPU에서 직렬로** 수행하며 92-frame full run은 호출하지 않는다.

```text
FlashVSR micro → SeedVR2 micro → DOVE micro → Upscale-A-Video availability
→ Real-ESRGAN micro → HAT micro
→ FlashVSR mini → SeedVR2 mini → DOVE mini → Upscale-A-Video availability
→ Real-ESRGAN mini → HAT mini
```

각 실제 모델 실행은 별도 process group으로 격리한다. INT/TERM/HUP 또는
wrapper 예외가 발생하면 해당 process group을 종료하고 `run_status.json`에
실패를 남긴다. SeedVR2는 기존 측정 peak 142318 MiB + guard 512 MiB, 즉
free VRAM **142830 MiB 이상**일 때만 시작한다.

FlashVSR/SeedVR2/DOVE/Real-ESRGAN/HAT run의 PASS 조건은 모두 충족해야 한다.

1. official pretrained checkpoint와 official upstream repository identity 확인
2. neural wrapper exit code 0
3. 출력이 정확히 15360×8640
4. 선택 frame 수, 24 fps가 입력 sample과 일치
5. timing/VRAM/frame/tile/chunk 필수 필드 기록
6. downsampled SSIM/PSNR/detail 보존 분석과 temporal sanity 분석 완료

Upscale-A-Video는 official code의 4K guard 때문에 4K→16K를 만들지 못하므로
GPU run을 가장하지 않고 `BLOCKED_REFERENCE_ONLY`로 기록하는 것이 정상 성공
동작이다. 학습, 비공식 weight, video-SR wrapper 안의 image-SR fallback, 출력
해상도 축소는 금지한다. Real-ESRGAN/HAT는 fallback이 아니라 명시적으로
분리된 `FRAME_INDEPENDENT_BASELINE`이다.

## 1. Objective and non-goals

This benchmark measures the smallest temporally meaningful inference units for
official pretrained video super-resolution models and compares two explicitly
labelled frame-independent image-SR baselines over identical 21/29-frame ranges.
It is designed to estimate long-video cost before a full run.

- Input: `../input/0a8556b7-296c-4d33-90fe-b596ce5eba74.mp4`
- Source: 3840×2160, 92 frames, 24 fps
- Required output: 15360×8640, same frame count and fps for a full run
- No training, fine-tuning, hidden image-SR fallback, or reduced-resolution output
- The default launcher runs only `micro` and `mini`; it never starts the
  92-frame full benchmark.
- Models run sequentially on one GPU. SeedVR2 is admitted only when at least
  142830 MiB is free, based on the retained 142318 MiB process peak plus a
  512 MiB guard.

## 2. Model benchmark units

| Model | micro | mini | Why |
|---|---:|---:|---|
| FlashVSR v1.1 Tiny Long | 21 real frames | 29 real frames | Tiny Long pads to an `8n+1` internal timeline. 21 real outputs exercise the first `F=25` causal emission; 29 add one steady-state 8-frame emission (`F=33`). |
| SeedVR2 3B | 5 frames | 9 frames | The temporal VAE requires `4n+1`; 5 is one native unit. 9 frames with window 5/overlap 1 creates two chunks and exercises temporal blending. |
| DOVE Stage-2 | 9 frames | 17 frames | DOVE pads to `8n+1`; 9 is the smallest non-degenerate unpadded video unit and 17 adds one native eight-frame unit. |
| Upscale-A-Video | 8 frames | 14 frames | Official denoising uses an 8-frame window with 2-frame overlap/stride 6; 14 creates two windows. Execution is availability-only because the untouched official 4K path does not output 16K. |
| RealESRGAN_x4plus | 21 frames | 29 frames | Native temporal unit is one frame. The sample lengths intentionally match FlashVSR exactly and expose frame-to-frame instability on a non-trivial sequence. |
| Real_HAT_GAN_SRx4 | 21 frames | 29 frames | Native temporal unit is one frame. The fidelity checkpoint and the same exact ranges isolate model quality from clip selection. |

`--start-frame K` selects an exact zero-based source position. `--max-frames N`
can cap a wrapper run, but micro/mini reject values shorter than their meaningful
temporal minimum.

## 3. Fixed 4K spatial execution plans

### FlashVSR v1.1

```text
3840×2160 source
→ 320×320 source tiles, 32 px overlap
→ grid 14×8 = 112 tiles
→ each full tile produces 1280×1280 output, 128 px overlap
→ causal Tiny Long DiT + LQ projection + TCDecoder per tile
→ feather-weighted spatial accumulation
→ 15360×8640 stream encode
```

Each spatial tile keeps its DiT KV, causal LQ and TCDecoder state for its whole
timeline. Frames are not independently upscaled and there are no independent
temporal clips to stitch.

### SeedVR2

```text
temporal windows: 5 frames, overlap 1, stride 4
each window
→ 488×488 source tiles, 60 px overlap
→ grid 9×5 = 45 tiles
→ bicubic conditioning to 1952×1952
→ official VAE encode → one-step DiT → VAE decode
→ cosine spatial merge + cosine temporal overlap merge
→ 15360×8640 stream encode
```

This is the retained H200-safe plan. The already disproved automatic sequence
`13/5,768/96 → 9/5,768/96 → 5/1,768/96 → 5/1,612/76 → 5/1,488/60`
is documented rather than repeated in minimal runs.

### Upscale-A-Video

The public official inference contains this high-resolution guard:

```text
3840×2160 → area downsample 960×540 → official ×4 → 3840×2160
```

It therefore cannot be reported as an untouched official 4K→16K run. If that
guard were custom-bypassed, its 256 px tile cores would form a 15×9=135 tile
grid, with 64 px context halos and core-crop assignment. The benchmark records
this as `BLOCKED_REFERENCE_ONLY` and performs no misleading GPU run.

### DOVE Stage-2

```text
3840×2160 source → bilinear native x4 conditioning
→ 15360×8640 output-domain tiles, initially 1360×768 with 32 px overlap
→ CogVideoX VAE encode → one DOVE transformer evaluation at timestep 399
→ scheduler velocity → CogVideoX VAE decode
→ half-overlap center crop/direct placement → stream encode
```

Micro/mini are 9/17 real frames. Long-video planning uses temporal window 17,
overlap 8 and stride 9. The wrapper uses bounded disk-backed assembly rather
than the upstream script's three complete FP32 16K tensors; this changes the
storage strategy, not the official neural path. Details and provenance are in
`DOVE_BENCHMARK_SPEC.md`.

### RealESRGAN_x4plus frame baseline

```text
each decoded 3840×2160 frame independently
→ 512×512 core tiles, 10 px context halo
→ grid 8×5 = 40 tiles/frame
→ official 23-block RRDBNet, FP16, native ×4 only
→ halo crop + direct core placement (no weighted blend)
→ 15360×8640 stream encode
```

`outscale` resampling and GFPGAN face enhancement are disabled. Every frame is
processed deterministically without temporal state; this makes it fast and easy
to parallelize, but texture shimmer/flicker is a primary quality risk.

### Real_HAT_GAN_SRx4 frame baseline

```text
each decoded 3840×2160 frame independently
→ 512×512 core tiles, 32 px context halo
→ grid 8×5 = 40 tiles/frame
→ official HAT architecture + params_ema fidelity checkpoint, FP32, native ×4
→ halo crop + direct core placement (no weighted blend)
→ 15360×8640 stream encode
```

Tile core and halo are multiples of HAT's 16-pixel attention window. The
`Real_HAT_GAN_SRx4` fidelity checkpoint is used rather than the sharper variant
to reduce hallucination and independent-frame texture variation.

## 4. Required report contract

Every runnable model report records:

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
tile_count
temporal_chunk_count or streaming_step_count

pure_inference_seconds_per_input_frame
end_to_end_seconds_per_input_frame
pure_inference_seconds_per_output_frame
end_to_end_seconds_per_output_frame
```

Runner validation additionally records downsampled structure SSIM/PSNR,
absolute difference, Laplacian/edge-energy ratios, suspected duplicate ratio
and temporal discontinuity warnings. These metrics compare source preservation;
they do not claim ground-truth 16K perceptual fidelity.

SeedVR2 additionally separates conditioning VAE encode, one-step diffusion
sampling, and VAE decode. FlashVSR separates DiT/LQ neural work from
TCDecoder/color correction. `pure_inference_seconds_per_*` uses neural work
plus decode; it excludes spatial merge, codec work and one-time model loading.
The two image baselines report `frame_independent=true`, `chunk_boundaries=[]`,
40 tiles/frame, and their exact native ×4 tile inference/placement cost.

## 5. Estimation models

### FlashVSR

```text
streaming_steps(N) = 1                              , N ≤ 21
                   = 1 + ceil((N - 21) / 8)         , N > 21

T(N) ≈ model_load_once
     + warmup_full_SR_cost
     + (streaming_steps(N)-1) × steady_full_SR_step_cost
     + merge_encode_cost_per_frame × N
```

Micro measures the causal warm-up. Mini supplies the first steady-state step;
both are required for a credible long-video estimate.

### SeedVR2

```text
chunks(N) = 1                                       , N ≤ 5
          = 1 + ceil((N - 5) / (5 - 1))             , N > 5

T(N) ≈ model_load_once
     + full_SR_cost_per_tile_chunk × 45 × chunks(N)
     + merge_encode_cost_per_frame × N
```

The long-run temporal compute overhead approaches `5/4 = 1.25×` because one
of every five window frames is overlapped.

The fitted `full_SR` coefficients use the wrapper's complete inference loop,
including device offload/transfers and conditioning overhead. The separately
reported `pure_inference_seconds_per_*` remains neural encode/DiT/decode only;
using that smaller value for wall-clock prediction would under-estimate time.

### Upscale-A-Video reference formula

```text
chunks(N) = 1                                       , N ≤ 8
          = 1 + ceil((N - 8) / 6)                   , N > 8
```

No timing coefficients are fitted while the official 4K path fails the 16K
output contract.

### Frame-independent image-SR baselines

```text
T(N) ≈ model_load_once + fixed_orchestration
     + warm_first_frame_cost
     + max(0, N-1) × steady_frame_cost
```

Micro and mini deltas fit steady per-frame cost without assuming the first CUDA
frame is representative. There is no temporal-overlap surcharge; tile count is
exactly `40×N` at the locked 512-pixel plan.

All runnable reports estimate 10, 100, 1000 and 1920 frames. Estimates are
engineering projections from micro/mini measurements, not substitute output
quality evidence.

The summarizer also emits 2880-frame estimates, corresponding to two minutes
at the source rate of 24 fps.

## 6. Launcher and examples

Default bounded comparison:

```bash
cd /home/work/ai-graphics_33/Hrlrvo/Upscaler/architect
./run_benchmark_minimal.sh
```

One model or one mode:

```bash
./run_benchmark_minimal.sh --models FlashVSR_v1.1 --modes micro
./run_benchmark_minimal.sh --models SeedVR2 --modes micro mini
./run_dove_benchmarks.sh
./run_benchmark_minimal.sh --models Upscale-A-Video --modes micro mini
./run_benchmark_minimal.sh --models Real-ESRGAN HAT --modes micro mini
./run_frame_image_benchmarks.sh
```

Different source position without changing the source file:

```bash
./run_benchmark_minimal.sh --start-frame 24 --models FlashVSR_v1.1 SeedVR2 --modes micro
```

Print the complete command plan without inference:

```bash
./run_benchmark_minimal.sh --dry-run
```

The legacy 92-frame runner remains `run_test_all.sh`; it is intentionally not
called by the minimal launcher.

## 7. Output contract

```text
../output/benchmark/<model>/...micro...{mp4|mkv}
../output/benchmark/<model>/...mini...{mp4|mkv}

../result/benchmark/<model>/benchmark_micro.{json,md}
../result/benchmark/<model>/benchmark_mini.{json,md}
../result/benchmark/<model>/run_micro.log
../result/benchmark/<model>/run_mini.log
../result/benchmark/run_status.json
../result/benchmark/benchmark_estimates.json
../result/benchmark/performance_comparison.md
../result/benchmark/quality_comparison.{json,md}
../result/benchmark/qualitative/<same-frame native-detail crops and contact sheets>
```

Root-cause, research, ranking and final synthesis live in the same immutable
benchmark namespace instead of overwriting `result/test` incident evidence.
When micro and mini reports pass validation, the launcher fits the appropriate
causal/window or independent-frame timing model and writes cross-model reports.
Previously valid reports are reused when only the two image baselines are run;
the runner never reruns FlashVSR/SeedVR2 merely to regenerate comparison files.
Arbitrary projection targets can also be generated with
`python summarize_benchmarks.py --frames 300 2400`.

The explicit summarizer command also recomputes an apples-to-apples quality
comparison on identical source ranges: first 5 frames for micro and first 9
frames for mini. This avoids comparing FlashVSR's 21/29-frame native units
directly against SeedVR2's 5/9-frame units.

## 8. Failure policy

- Never reduce output resolution.
- Never substitute a per-frame/image model inside a video-SR run. The two image
  models are separately named baselines with distinct outputs and reports.
- A caught CUDA OOM may only reduce the documented temporal/spatial plan.
- Seed minimal runs allow at most three reductions because they start from the
  retained safe plan.
- Unsupported 16K encoders are recorded; no silent resize is allowed.
- Each child is a separate process group and is terminated on launcher
  cancellation.
- A model failure is written to `run_status.json`; later independent model
  availability checks may still run.
- Full execution requires a separate explicit command and is not part of this
  specification's default stop condition.
