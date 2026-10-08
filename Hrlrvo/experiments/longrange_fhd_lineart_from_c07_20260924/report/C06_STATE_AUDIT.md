# C06 state recovery audit

## Confirmed facts
- Exact original C6 FP32 prefix is not recoverable from retained artifacts: previous_prefix slots contain C22/C23. Only last3 latent slices were retained, not the full9-latent context needed for exact decode.
- C6 saved RGB segment corresponds to working output frames17:65. Its last17 frames equal original next_prefix frames48:65 after RGB uint8 truncation.
- Writer uses `frame.mul(255).to(torch.uint8)`. Reconstructed q/255 differs by less than1/255 plus FP32 arithmetic tolerance. This bounds bootstrap pixels only, NOT downstream diffusion differences.
- Independent empirical C22 check (available original FP32 tail vs stored uint8 tail): max0.0039215609, mean0.00160592, RMSE0.00215292,33.515% exact. This is illustrative, not a measurement of missing C6 FP32 data.
- C6 BF16 carry is copied byte-for-byte. Causal-safe recurrence re-encodes first1-frame prefix keyframe; only that generated-side keyframe is affected by RGB precision recovery. Remaining2 direct latent slices unchanged; current anchor encoding overrides anchor bbox.
- CPU/CUDA default RNG states are byte-identical in original archivedC0 and retainedC22,C23. CPU SHA91c8f242eb8ff100b04d9c455ae6cad0ab171fa6e10449428fade765ecaef29f; CUDA Philox seed71634910,offset0.
- This checkpoint's ConvVAE encoder returns normalized means, not sampled posterior. Decoder timestep_conditioning=false, so its guarded random noise path is inactive. Denoising uses a separate explicit generator seeded71634910+absoluteStart; restoring original archivedC0 default states is valid here.

## Source evidence
- copied `inference.py:850-905`: causal-safe carry, first-frame encode, anchor overwrite.
- copied `inference.py:1160-1163`: isolated per-chunk noise generator.
- `streaming_video.py:187-233`: RGB24 truncation; `:541-591`: committed record validation; `:621-674`: latest prefix/RNG validation.
- upstream read-only `video_vae.py:266-420`: deterministic mean encode.
- upstream read-only `conv_video_decoder.py:149-175,266-360`: decoder context and optional random noise guard.

## Decision
New isolated experiment only. Preserve C0-C6 emitted RGB/carry exactly, reconstruct only missing C6 RGB prefix from lossless uint8 .npy (never MP4), record new state digest honestly, use exact original RNG, generate C7-C23. This is a one-time quantized RGB bootstrap, NOT bit-exact original FP32 recurrence. No replay or training.
