# A6000 GPU0: fixed C0 D5 Depth Global prior

Remote root: `/home/oem/hrlrvo/experiments/a6000_depth_global_prior_20260924`
Mode: `strength_1p0_global_depth_prior`

## Controlled intervention
Every inference module is an exact hash-pinned snapshot of the H200 lineart experiment. The internal `LineArtGlobalPrior` class name is deliberately retained to avoid any model-code delta. Only the input map is D5 relative inverse depth rather than line density. The additive Global attention bias remains `B_ij=lambda(step)*g_i*g_j`; this is near-depth salience, **not** pairwise same-depth similarity or a trained depth controller. Masks, all-keys read support, dual decompression and known-prefix/anchor restoration are unchanged. Lambda=0.5 for steps0..9,0.3 for10..19,0.1 for20..29. No Global Boost or Native Halo.

## Reference preparation
The source is A6000 arm01 baseline lossless `chunks/segment_000000.npy[0]`, not an H200 PNG or MP4 decode. On the A6000 server, reproduce `D5_v2_base_518`: Depth-Anything-V2-Base-hf revision `b1958afc87fb45a9e3746cb387596094de553ed8`, CPU float32 (matching the original D5 extraction), processor518 actual input518x924; official postprocess1080x1920. Relative inverse depth white=near, percentile0.5/99.5 normalization to float32[0,1]. Literal zero padding36px top/bottom then32x32 and4x4 area averaging gives9x15. Both means accumulate float64 and castfloat32. No gamma, no inversion, no line-density divisor0.05. NPY supplies the prior, PNG is visualization only. Exact hashes and numerical policy: `reference/depth_reference.json`.

## Exact C0 continuation
**Recovery finding:** the existing two-slot prefix journal overwrote C0 FP32 prefix/RNG with C2. The original C0 RGB24 segment and latent carry survive. A GPU0 replay in `recovery/` uses unchanged original A6000 baseline modules and a runner differing only in end_chunk0, and MUST match original C0 pixel/prefix-RNG/carry digests exactly before the recovered prefix can be archived. Original baseline files are never overwritten. This requires one extra C0 inference; no RGB24-to-FP32 approximation is accepted.

`bootstrap/c0` copies only A6000 baseline C0 segment, FP32 prefix plus CPU/CUDA RNG state, clean BF16 carry, record and preview. It does not import baseline C1+ or baseline run identity. SHA checks in `report/c0_recovery_verification.json`; copied C0 record committed last. After loading the model, restore and tensor-compare the exact saved RNG; continue C1..C23. All other conditions remain seed71634910 + absolute_start_frame, checkpoint7500 same tensor data, strength1.0, latent carry ON, time rebasing ON,65/17/48 frames,30 steps,CFG3,STG1,rescale0.7,block28,empty positive and original negative prompt,input960x540@24,visible1920x1080/encoded1920x1152.

## Files
- `reference/c0_frame000.png`: source first frame from A6000 C0.
- `reference/depth_gray.png`, `depth_color.png`, `depth_gray_16bit.png`: depth visualization.
- `reference/depth_raw.npy`, `depth_inverse_normalized.npy`, `global_depth_strength.npy`: full raw/normalized and9x15 prior.
- `outputs/strength_1p0_global_depth_prior/chunk_previews/C00_full65.mp4`: copied exact baseline C0.
- Same directory `C01_new48.mp4` through `C23_new48.mp4`: depth-conditioned new chunks.
- `outputs/strength_1p0_global_depth_prior/C0_C23_1169f_1920x1080.mp4`: stitched final48.708s video only when complete.
- `chunks/strength_1p0_global_depth_prior/`: lossless emitted segments and RGB-prefix/RNG journals.
- `latent_states/strength_1p0_global_depth_prior/`: clean carry checkpoints.
- `logs/latest.log`, `logs/previews_*.log`, `status.json`: runtime evidence.

## Safety and operation
Run only via remote tmux `hrlrvo` window `exp_depth_gpu0`, `bash start_depth.sh`. Launcher pins physical GPU0, uses flock and refuses any existing GPU0 compute process. No automatic retry, no other-job termination. CPU prep/preflight must use CUDA_VISIBLE_DEVICES empty. Resume is identity/hash gated. GPU1/2/3 and H200 experiment/source/results remain untouched. No dependency installation was needed. New feature quality is not yet proven; launch verification is distinct from completed rendering.
