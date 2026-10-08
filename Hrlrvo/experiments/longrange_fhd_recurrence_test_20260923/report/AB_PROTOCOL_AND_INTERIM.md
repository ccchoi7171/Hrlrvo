# Concurrent strength A/B protocol (interim, not final verdict)

## Isolation
- Original `pipeline` and `VideoOutpainting` remain read-only. All copied source, derived input, outputs, logs, patches, and comparisons stay under this workspace.
- The 0.8 branch is `modified`; the 1.0 branch is `strength_1p0`. The two have separate chunks, RGB outputs, BF16 latent resume states, logs, and tmux sessions.
- Source import file is the copied `source/VideoOutpainting_test/main/src/ltx_outpaint/inference.py`; both run manifests contain its same SHA256 `69542830e48229650f8afce87e2240403ab6f78049fd1868252158b9ff97a426`.

## Controlled comparison
- Same 960x540@24fps lossless anchor, same 1920x1080 visible/1920x1152 encoded geometry, same base seed `71634910`, same checkpoint `checkpoint-007500`, same prompt/negative prompt, same 30 steps, same chunk-local positions, same causal-safe hybrid latent handoff.
- `chunks/modified/run.json` and `chunks/strength_1p0/run.json` differ only in run mode/output namespace and generated-prefix strength (0.8 versus 1.0).
- Strength 1.0 branch starts at C1 from a byte-identical copy of C0's RGB segment, FP32 17-frame prefix, BF16 final-clean latent carry, and RNG state. C0 has no incoming prefix, so strength has no effect on C0. SHA checks and provenance: `chunks/strength_1p0/bootstrap_provenance.json`.
- Both target C0-C23 inclusive = 24 chunks = 1169 frames = 48.708 seconds. This is the prepared-input maximum and interprets the user's "24 chunks" request.
- 0.8 C0-C4 completed as 257f/24fps MP4, then a tmux waiter resumed C5-C23 from saved state without rerunning C0-C4. 1.0 C1-C23 runs in a separate tmux session. Chunk preview MP4s are emitted under each branch's `outputs/<mode>/chunk_previews`.

## Preliminary C0-C1 boundary (not geometry verdict)
- Using lossless RGB segments and 4x spatial sampling, side boundary frame-to-frame MAE at F64->F65:
  - strength 0.8: left 8.77, right 8.64; center 1.39.
  - strength 1.0: left 2.20, right 1.89; center 1.37.
- Visual: `comparisons/C00_C01_strength_AB_left.jpg` and `_right.jpg`; the 0.8 sides visibly change tonality more at the boundary. This supports a prefix-conditioning discontinuity caused by the 0.8 amplitude attenuation approximation; it does **not** prove which branch has better long-range structural stability.
- 0.8 C1-C4 side boundary MAE remains ~7.5-8.5 per boundary; see `report/boundary_modified_C0_C4.tsv`.
- No direct PSNR/LPIPS comparison to the 4K baseline is used because resolution and newly generated C0 differ.

## Runtime
- Solo FHD 0.8 denoising C0-C2: 561.2, 556.6, 560.1 s/chunk. First overlapping 0.8 C3: 834.1 s. Overlapping 1.0 C1: 988.0 s. Parallel GPU use ~68-70GB, utilization ~98-100%; aggregate throughput improves modestly, not 2x.
- This document is interim. Final geometry/onset judgment requires C19/C23 and timestamp comparisons.

## 2026-09-23 14:19 UTC update: inferiority stop
- At shared C0-C7, 0.8 mean left/right boundary MAE was 8.34/8.15 versus 1.0 at 2.34/1.66. The center remained similar (1.43 versus 1.36). See `report/boundary_modified_C0_C7.tsv` and `report/boundary_strength_1p0_C0_C7.tsv`.
- Lossless generated-side crop at F400 (16.67s) showed 0.8 with washed-out/window-frame degradation and 1.0 with clearer geometry: `comparisons/strength_AB_lossless/F000400_left.png` and `_right.png`. By F544 (22.67s), the separation remains visible in corresponding lossless crops.
- User asked whether one branch can be omitted; the inferior 0.8 branch was terminated **after C10 committed**. C0-C10 = 545 frames = 22.708s was stitched and ffprobe-verified at `outputs/modified/C0_C10_545f_1920x1080.mp4`. All chunk segments and BF16 latent resume states through C10 are retained. The 1.0 branch continues unmodified to C23.
- The old dual-branch finalizer was retired and replaced by `commands/finalize_strength1_C23.sh` in a detached tmux session. The 1.0 branch runs alone again at about 9.3min denoising/chunk.
