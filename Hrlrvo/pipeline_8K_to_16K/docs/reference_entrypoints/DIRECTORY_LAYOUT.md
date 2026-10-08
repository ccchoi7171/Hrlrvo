# Certification inference package layout

Run either profile with the **same** entrypoint. The only model-workload difference
between the JSON profiles is `frames.outpaint` / `frames.upscale` (`all/all` or
`65/21`). A source frame count is resolved with ffprobe at run time, never
hard-coded.

```bash
./run_cert.sh --config configs/certification/03.json
./run_cert.sh --config configs/quick/03.json
```

`run_cert.sh` uses the named Conda environments `Hrlrvo` and
`Hrlrvo-FlashVSR` if present. Otherwise it uses the active Python for both.
For another environment layout set `CERT_OUTPAINT_PYTHON` and
`CERT_FLASH_PYTHON` to the respective interpreter paths. The runtime source,
configuration, weights, and model repositories are under `src/`; the original
research repositories are not runtime dependencies.

- `configs/certification/01..05.json` and `configs/quick/01..05.json`: immutable
  requested inference conditions and output destinations.
- `src/video_outpainting/`: packaged production LTX-2.3 LongRange outpainting
  code and YAML. `src/video_outpainting/models/LTX2.3/` contains the official
  LTX package sources used by that code.
- `src/flashvsr/`: packaged production FlashVSR v1.1 TinyLong implementation.
- `src/models/`: copied LTX DEV, Gemma, LoRA checkpoint-007500 and FlashVSR
  runtime/weights. No link points to the original repositories.
- `src/manifests/`: per-file model/source/package hashes, the pinned 260-file
  LTX source snapshot certificate, and the inspected Python environment inventory.
- `data/00_Input_8K_7680x4320/`: source videos (read-only to the runner).
- `data/{certification,quick}/video_XX_seedNNN/`: separate run records per
  video/seed/profile, with `config/requested_config.json`, runtime-resolved
  `config/resolved_config.json`, and `logs/pipeline.log`.

Each of the ten run directories already exists, including empty artifact
subdirectories:

- `0001_Preprocess_FHD24/output/`: 8K/16fps → FHD/24fps **processing input**.
  The same source-indexed N frames are retained; Lanczos resize, timebase
  compression, optional 1.5× processing audio, 10-bit HEVC CRF1. `metadata/`
  contains input/output ffprobe JSON and `frame_alignment.json`;
  `logs/preprocess.log` has stage progress.
- `0002_Outpaint_4K16/chunks/native/`: the native LongRange lossless RGB segment,
  alternating recurrent-prefix/RNG state, and per-chunk commit records.
  `chunks/latents/` retains each actual final generated latent and the
  corresponding actual carried prefix state. `chunks/mp4/` holds an inspection
  MP4 per **native** chunk emission range. `chunks/manifests/` holds the native
  chunk lineage/plan/results. `final/` has the identical sequence retimed
  **without frame resampling** to 4K/16fps. `metadata/` and `logs/` hold
  inference configuration, provenance, ffprobe and real denoise progress.
- `0003_Upscale_FlashVSR_16K/segments/`: retained source clip (only if the
  selected frame count is shorter), per-tile lossless PNGs, per-actual-TinyLong
  emission metadata, SHA-verified spatial-tile commit journal and the complete
  native stage manifest. The model runs a
  continuous causal/KV timeline per spatial tile: there are no independent
  temporal chunks to merge. `raw_flashvsr/` is the uncomposited 16K/16fps
  result. `final/` is the 16K/16fps result with the same-index original 8K
  center inserted at x[3840:11520), y[2160:6480), 64px feather, and original
  audio (or video-only when absent). `metadata/` and `logs/` hold stage
  verification and progress. There is no `0004` stage.

The packaged reference trace is in `REFERENCE_IMPLEMENTATION_ANALYSIS.md`.
Existing older research outputs under `data/` are left untouched.

Resume: `--resume` validates native LongRange chunk records and FlashVSR
stage/tile artifacts before reusing them. An incomplete FlashVSR spatial tile
restarts its full continuous causal timeline; no temporal KV cache is guessed
or restored. Uncommitted MP4/clip outputs are archived and regenerated.
`--force` archives any previous run
records under the profile's `.archive/` rather than silently deleting them.
A requested config mismatch aborts resume. A final file alone is never the
proof of completion.

Optional full-decode verification can be enabled by adding
`"verification": {"full_decode": true}` to a JSON config; no CLI inference
conditions are accepted.
