# LongRange three-toggle selective experiment plan

- Scope: C21-C23 first; C14-C16 only if resources/time allow. 05 baseline and all of pipeline are read-only.
- Source isolation: copy `VideoOutpainting/main/src/ltx_outpaint` into this workspace, patch only copied `inference.py`; original SHA256 in `patches/original_sha256.txt`.
- Toggle OFF path must preserve existing `sample_chunk` behavior.
- Toggles: chunk-local positions (skip absolute offset); latent carry (take last 3 slices of post-Euler sigma=0 state before VAE decode, hybrid preserve independently encoded center anchor); side-prefix latent amplitude factor 0.8 while retaining mask=1, denoise_mask=0, timestep=0, no known-token noise. This is an influence approximation, not upstream `strength=0.8`.
- Window seed: prior baseline pre-encoder RGB24 segment last 17 frames, reconstructed FP32 /255; exact pre-quantization FP32 state was overwritten by baseline rolling prefix slots. Window first chunk therefore uses RGB seed; subsequent chunks use direct clean latent carry.
- Keep absolute global `start` for anchor decode, output index and seed rule `base_seed + absolute_start`; only model temporal positions rebased.
- Save per-chunk RGB24 segments, FP32 output prefix, carry latent, metadata and logs only here.
- CPU tests: patch OFF equivalence, local position flag, hybrid anchor/side mask, side attenuation and fixed denoise mask, output global coverage. Then start GPU.
