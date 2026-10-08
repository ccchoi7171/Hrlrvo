# Static preflight before GPU

- Original `pipeline` and `VideoOutpainting` trees are read-only; copied module import path, cwd and PYTHONPATH proven by `logs/preflight_after_hybrid.log`.
- Original 8K 05 source is 7680×4320, 16fps, 1952f. Independent lossless FFV1 anchor is 960×540, 24fps, 1169f; see `input/anchor_ffprobe.json`.
- Visible canvas 1920×1080; encoded 1920×1152; latent 60×36; pixel anchor bbox y306:846,x480:1440; latent bbox y9:27,x15:45. Global C4, Local K4/K5, Adaptive geometry and visible crop pass.
- Prefix known mask has denoise mask 0 and timestep 0; GaussianNoiser preserves it. Center anchor factor 1, generated left/right factor 0.8. C4/C19/C23 stitch counts 257/977/1169 pass.
- Official scheduler's final sigma=0. Copied sampler captures last 3 slices of final clean latent after Euler denoising and before VAE decode. C0 captures them without an old RGB prefix.
- **Causal-safe handoff:** previous clip's last three latent slices correspond to old RGB frames 41..48, 49..56, 57..64. Next clip prefix is frames 48..64: its first latent is a one-frame causal keyframe, so that one slice is VAE-encoded from previous output RGB frame48; the remaining two latent slices are carried directly from previous final clean latent. The carried three-slice tensor is saved for resume, and the first slice is replaced only when building next clip's conditioning. This is a documented hybrid, not a strict all-three-slices direct carry.
- Copied-source CPU tests: `logs/cpu_tests_hybrid.log` 56 passed. Preflight: `logs/preflight_after_hybrid.log` status PASS. Static gate passed; GPU smoke C0-C4 may start.
