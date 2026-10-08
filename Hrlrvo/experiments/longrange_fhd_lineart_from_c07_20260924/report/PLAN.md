# Baseline through C06, detailed Global LineArt from C07

User-requested replacement of C1-start experiment. Production, baseline, and preceding LineArt source/results remain unchanged. C1-start run stopped by SIGINT during C2; C0/C1 committed artifacts retained.

1. Copy previous experiment source into this new workspace; change runner bootstrap only.
2. Preserve baseline C0-C6 lossless RGB segments and all seven BF16 carry tensors byte-for-byte.
3. C6 FP32 prefix was overwritten by two-slot rotating storage. Reconstruct once from saved lossless uint8 .npy tail /255, not MP4. Precision difference is less than 1/255 apart from float32 roundoff; this is NOT a bit-exact FP32 bootstrap. Original C6 carry remains exact. Future chunks retain native FP32 recurrence.
4. Archived original C0 and surviving C22/C23 CPU and CUDA default RNG states are all byte-identical. Restore these unchanged states; denoising uses the original separate generator seed 71634910+48*chunk.
5. C0-C6 yield exact baseline output frames [0,353). First working chunk C7 starts336, fixed prefix[336,353), emits[353,401).
6. Generate C7-C23 with the same detailed.png reference, lambda .5/.3/.1, strength1.0, causal-safe hybrid latent carry, local time, checkpoint007500 and unchanged sampling.
7. CPU per-chunk preview and cumulative MP4 watchers. C4 milestone is reused baseline-only; C9/C14/C19/C23 contain LineArt generated outputs.

Expected final:1169 frames,1920x1080,24fps. This intervention-onset experiment differs from a C1-start prevention experiment, and also has one-time bootstrap RGB quantization. Do not label it as bit-exact baseline state.
