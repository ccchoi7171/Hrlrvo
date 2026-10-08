# Certification package: reference-runtime trace

This document records the execution path used for the package snapshot. It is
not a replacement algorithm specification. The original research repositories
are left unchanged.

## VideoOutpainting

Production driver: `VideoOutpainting/main/src/ltx_outpaint/inference.py`, reached
by the existing `pipeline/config/XX/XX_VideoOutPaint_4K.sh` scripts via
`python -m ltx_outpaint.inference --mode normal_long_range`.

Actual call path:

1. `_load_settings` loads `main/configs/inference_4k.yaml`; CLI/API overrides
   choose checkpoint, seed, steps, memory switches and requested frame count.
2. `sample_long_range` probes the 24 fps FHD anchor and resolves `num_frames=None`
   to its decoded source frame count. It obtains VAE temporal/spatial factors
   from the base checkpoint and validates the native `LongRangeSpec`.
3. `_load_models` loads the LTX-2.3 22B DEV base, Gemma-3-12B text encoder,
   causal video VAE from the base checkpoint, and the selected outpainting
   adapter. The evaluation-selected adapter is checkpoint-007500; the
   reference YAML's adapter default is checkpoint-030000 and is overridden.
4. `_iter_long_range_chunks` uses **65 pixel frames / 17 prefix / 48 stride**,
   and repeat-last padding for a short final chunk. The first chunk emits all
   its requested frames; later chunks retain the 17-frame generated prefix and
   append only new frames. Absolute source frame positions are maintained.
5. `sample_chunk` applies the FHD anchor, causal VAE encoding and
   `VideoLatentTools(causal_fix=True)`, known-mask conditioning and the native
   `ValidationRunner._run_denoising` scheduler/CFG/STG loop. The native runner
   invokes `sampling_ctx.advance_step()` after each actual Euler step.
6. Final latent state is decoded by the native VAE path. The existing
   `_composite_anchor_rgb` applies a single 16 px inward alpha feather. The
   generated RGB tail is cloned as the next FP32 prefix. No wrapper recomputes
   latent alignment, first-latent semantics or chunk merge.
7. `SegmentedOutputStore.commit_chunk` atomically writes emitted RGB as a
   lossless `.npy`, the next FP32 prefix plus CPU/CUDA RNG state as `.pt`, and a
   JSON completion record last. `encode_segments` assembles the final MP4.
   `chunk_lineage.json` and `inference_manifest.json` capture frame ranges and
   run provenance. Resume validates these native records and restores prefix
   and RNG state. A bootstrap 65f MP4 is optional; normal 65f Quick runs the
   first chunk through the model directly.

The reference does **not** natively save the post-denoise video latent or a
per-chunk MP4. Certification instrumentation must observe the existing final
latent immediately before VAE decode, and turn the already-committed RGB
segment into an inspection MP4 without feeding it back into inference.

## FlashVSR v1.1

Production driver: `Upscaler/architect/FlashVSR_v1.1/upscale.py`, called by
the existing `pipeline/src/03to04/upscale_blend_16k.py`. It loads official
`FlashVSRTinyLongPipeline`, `Causal_LQ4x_Proj`, `TCDecoder`, the fixed
cross-attention context, and the three v1.1 checkpoint files. The model runs
in bfloat16 with GPU residency management.

The input is cropped into overlapping **spatial tiles**. Each tile is an
independent, continuous causal temporal stream—not independent temporal clips
and not per-frame upscaling. `build_tile_store` decodes source frames into a
disk-backed, padded conditioning memmap. `_stream_official_tile_impl` keeps
LQ projection, DiT KV and TCDecoder caches across every temporal emission in
the tile; it uses an initial 21-frame emission and then 8-frame emissions.
Model input is padded to the smallest `8n+1` length of at least 25. Each tile
uses the reference `base_seed + tile.index` generator derivation. Its output
callback stages each real frame as a lossless tile PNG. A complete stage
manifest names the tiles and all their frame artifacts. The native reference
resumes only after the complete stage; the certification snapshot adds an
atomic, SHA-verified **spatial-tile** commit journal. A completed tile may be
reused, while an interrupted tile is restarted from frame zero with its own
continuous causal state. There is no mid-tile resume or persisted causal KV state.

The reference's `encode_tiles` performs weighted spatial tile merging and
MP4 encoding. The existing pipeline wrapper's
`composite_original_center_rows` replaces the 16K center with the same-index
original 8K frame and applies a 64 px inward smoothstep feather. This
composite runs *after* FlashVSR, not inside its temporal algorithm.

## Package boundary

The certification entrypoint only validates JSON, resolves source frame
counts, calls these snapshotted production paths, preserves their native
intermediates, performs frame-preserving 24→16 timestamp retiming, and records
inspection artifacts. It does not reconstruct either model's temporal logic.

## Code-level trace and invariants

### Outpainting configuration and loading

The existing `pipeline/config/XX/XX_VideoOutPaint_4K.sh` commands select
`--mode normal_long_range`, `--checkpoint .../checkpoint-007500`, `--steps
30`, video-specific seeds, and video-only GPU weights plus streaming VAE
without FFN token chunking. The old scripts had experiment-specific fixed
frame counts and a bootstrap MP4. The certification runner deliberately
supplies the **runtime-probed** frame limit and does not bootstrap from an
older experimental video; this is still the same `normal_long_range` model
path. It never invokes the seed-search or prefix-search modes.

`inference._load_settings` reads YAML; `_install_official_sources` inserts
`ltx-core`, `ltx-pipelines`, and `ltx-trainer` from the packaged LTX source
snapshot. `_load_models` validates adapter provenance, loads the LTX DEV
transformer/video VAE/Gemma encoder and installs the existing outpainting
attention + LoRA adapter. The YAML's base and adapter paths were changed only
to package-relative equivalents. The requested adapter is checkpoint-007500.
The copied LTX package has no `.git` directory, while the immutable adapter
header records original absolute training paths. The packaged provenance
guard validates all 260 copied official source files against
`src/manifests/ltx_source_snapshot.json`, confirms the pinned revision and
base-weight SHA, then compares the adapter's historical provenance without
trying to access the original checkout. The adapter's base path is accepted
after relocation only when its checkpoint name and bytes match the validated
base; the copied safetensors file itself is unchanged.
`GeometryValidationRunner` delegates the actual sampler to upstream
`ValidationRunner._run_denoising`; it is not replaced by this package.

### Outpainting temporal geometry

`_iter_long_range_chunks` is authoritative. It plans 65 pixel frames, a
17-pixel-frame recurrent prefix, and 48-pixel-frame stride. Its record has the
actual `absolute_start_frame`, `anchor_source_frame_range`,
`prefix_output_frame_range`, and `new_output_frame_range`. A shortened final
chunk uses the shortest `(F-1) % temporal_factor == 0` working length and
repeat-last padding; only its real emitted frames are committed. No wrapper
calculates alternate ranges.

`_condition_latents` uses the official
`map_temporal_interval_to_latent(0,F,0,0,temporal_factor)` mapping. The
observed FHD anchor is independently VAE encoded with training-aligned
spatial 512/128 tiling and placed into the actual encoded latent footprint.
The previous **decoded RGB FP32** prefix (not a presumed latent cache) is
independently encoded with the native VAE path; `_known_mask` marks its mapped
initial latent interval and the anchor latent rectangle clean/known. This
preserves causal first-latent handling through
`VideoLatentTools(causal_fix=True)` and `VideoConditionByMask`. The wrapper
never packs/unpacks, slices, or propagates these latent positions itself.

`sample_chunk` shifts state positions by `absolute_start_frame / 24` and
constructs the native Gaussian noiser generator with `seed +
absolute_start_frame`. It obtains positive/negative embeddings through the
native runner and runs the existing 30-step scheduler, CFG/STG/geometry
branches. The progress hook observes only native `advance_step()` calls.
The actual post-denoise latent is copied **after** `_run_denoising` and before
native VAE decode. This is a retained artifact, never fed back. The native
streaming VAE decode returns RGB; `_composite_anchor_rgb` applies the one
16-pixel inward feather. The old decoded prefix is then restored exactly at
chunk head, and the generated tail's last 17 FP32 RGB frames become the next
prefix. `SegmentedOutputStore` atomically writes each finalized RGB `.npy`,
an alternating `.pt` containing the real prefix and CPU/CUDA RNG state, and
a commit JSON. The certification copy of each `.pt` is an audit record only;
native alternating slots remain the actual resume state. Native
`chunk_lineage.json` is the merge/provenance authority.

### FlashVSR TinyLong temporal behavior

The production wrapper probes the 4K video, chooses spatial source tiles
(default 32-pixel overlap, auto VRAM-dependent tile size with OOM retry),
then `build_tile_store` frame-index decodes each tile into padded conditioning
memmap. `initialize_pipeline` loads the bfloat16 pretrained DiT through
`ModelManager`, installs `Causal_LQ4x_Proj` and `TCDecoder` checkpoints,
initializes fixed prompt cross-KV, and enables VRAM management. Each tile is
one uninterrupted TinyLong stream; an OOM retry discards that incomplete
attempt and restarts with a smaller spatial tile. It does **not** treat
frames independently.

`padded_model_frames` chooses the smallest `8n+1` model length that can emit
the real frames, with minimum 25. `_stream_official_tile_impl` clears LQ and
TCDecoder caches only once per tile, initializes one `torch.Generator` from
`base_seed + tile.index`, and carries DiT `pre_cache_k` / `pre_cache_v` plus
LQ and decoder causal state throughout the tile. The initial process uses
seven LQ slices / six latent time steps and emits 21 real frames. Later
processes use two slices / two latent time steps and emit up to eight frames.
The model's `model_fn_wan_video(... is_stream=True, ...)`, TCDecoder, color
correction, dtype, and spatial overlap/merge are unchanged. No independent
temporal overlap or cross-segment stitch exists in the reference. The
certification hook records metadata after each **actual** causal emission and
keeps the already-staged lossless tile PNGs. No KV/cache `.pt` is invented.

The complete stage manifest is the native FlashVSR resume boundary. The
certification snapshot additionally commits each independently processed
spatial tile after verifying every PNG and actual TinyLong emission metadata
by SHA. A rerun skips only fully committed tiles; incomplete tiles are
archived and regenerated through the unchanged continuous causal stream. Its
separate CPU encode path merges the tile PNGs into raw 16K. The final center
composite is a snapshot of the existing `pipeline/src/03to04/` implementation:
it reconstructs weighted spatial tile rows, reads original 8K frames by
index, and inserts x[3840:11520), y[2160:6480) using the existing 64px
inward smoothstep alpha. Quick input is a lossless 21-decoded-frame clip of
the first 21 final outpaint frames; the 8K source reader likewise consumes
its first 21 frames without time-based seeking. The final mux uses the
original 8K audio, never the 1.5x processing audio.

### Change classification

- **Reused as-is:** LTX model/LongRange/scheduler/causal VAE logic, official
  LTX package source, FlashVSR TinyLong model/causal stream, FlashVSR common
  media utilities, model weights.
- **Copied with path adaptation:** packaged LTX YAML and FlashVSR `MODEL_ROOT`;
  packaged-only LTX provenance relocation guarded by source/base SHA checks.
- **Instrumented only:** actual outpainting denoise-step, post-denoise latent
  and native chunk commit; actual FlashVSR temporal emission metadata/PNG
  reference, per-spatial-tile commit/restart and global RNG seed
  initialization. No model tensors/caches are modified by the observers.
- **New orchestration:** JSON preflight, frame-limit resolution, source-index
  preprocessing, chunk audit MP4/carry copies, 24→16 timestamp retime,
  manifest-based resume checks, raw 16K retention, final audio mux, logging
  and output directories.
