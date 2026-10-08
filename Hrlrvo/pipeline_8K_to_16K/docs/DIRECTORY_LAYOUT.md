# Stage0001 layout correction — historical reproduction

Previous certification draft: CRF1. Corrected historical reproduction contract: CRF18.

- `data/01_PreProc_2K_1920x1080/<ID>_PreProc_2K.mp4` and `.json`: immutable imported canonical reference/cache.
- `data/01_PreProc_2K_1920x1080/metadata/<ID>/runs/<timestamp>/`: actual corrected Stage0001 work, producer command/identity, probes, strict full decode, alignment and output.
- `.../metadata/<ID>/generated/<ID>_PreProc_2K.mp4` and `.json`: normal fresh Full/Quick/stage-only published output. It is selected downstream when present; invalid output fails closed. Canonical is not overwritten.
- `.../metadata/<ID>/preserved/<timestamp>/`: previous **generated** results, not a substitute for protecting canonical originals.
- `data/validation/preprocess_reproduction/<ID>/`: isolated regeneration plus full RGB24 frame-hash comparison for all5.
- `configs/validation/preprocess_reproduction/<ID>.json`: same fixed producer, isolated artifact paths; only `--stage preprocess` used for acceptance.
- `state/preprocess_reproduction/`: pre-change snapshots, diffs, CPU regressions, static/scope evidence.

Outpaint/Flash stage directories and queue code are unchanged. See [PREPROCESS_REPRODUCTION_01_05.md](PREPROCESS_REPRODUCTION_01_05.md) for actual acceptance, not merely layout.

---
## Original migration layout — archived pre-correction description

The below original text, including CRF1 and prior canonical publication policy, is retained as history and is superseded for Stage0001 by the correction above. Original bytes are also in `../state/preprocess_reproduction/before/docs/DIRECTORY_LAYOUT.md`.

# New stage-centered directory layout

`pipeline_8K_to_16K/` is independent at runtime. Protected references remain read-only.

- `run_cert.sh` → `run_cert.py`: single Full/Quick/stage entrypoint.
- `run_batch.sh` → `run_batch.py`: independent Outpaint1 + Upscale1 FIFO queue; same stage entrypoint.
- `configs/certification/01..05.json`: all/all.
- `configs/quick/01..05.json`: 65/21. Only frame limits and artifact identities differ.
- `configs/validation/05.json`: 65/21 parameters with validation-only artifact paths; no Flash was authorized for this task.
- `configs/schemas/`: job/batch schemas; runtime validation also enforces fixed model settings and canonical paths.
- `data/00_Input_8K_7680x4320/`: five SHA-verified original copies.
- `data/01_PreProc_2K_1920x1080/`: five exact historical PreProc copies and original JSON provenance. They are NOT recreated during migration. Historical encoder CRF is18.
- `data/01_PreProc_2K_1920x1080/metadata/<ID>/`: fresh/resume Stage0001 work, probes, logs and `preserved/<stamp>/` copies of prior canonical bytes before replacement. Fresh historical-reproduction processing is CRF18, starts from8K and never silently skips due to a copied file.
- `data/02_Outpaint_4K_3840x2160/<ID>/final/<ID>_Outpaint_4K.mp4`: Certification N-frame final4K16.
- `.../<ID>/chunks/rgb/`: per-native-commit MP4, named by native new-output frame range.
- `.../<ID>/chunks/latents/chunk_000000/`: final denoised latent plus exact native decodedRGB17f/RNG carry state copy; latent itself is not carried to next chunk.
- `.../<ID>/chunks/native/`: native uint8 `.npy` segments, alternating prefix/RNG files, per-chunk commits and retained24fps assembly.
- `.../<ID>/chunks/manifests/`: chunk_plan/results/lineage/inference manifests.
- `.../<ID>/metadata/`: stage request/input identity/completion SHA plus native metadata; `logs/`: stage log.
- `data/03_Upscale_16K_15360x8640/<ID>/`: final center-composite16K, temporal emission JSONs, tilePNG journals and stage logs. `raw_flashvsr/` is a layout directory; its MP4 is created only in opt-in `legacy_raw` mode. No independent temporal segments/algorithm rewrite. See `docs/UPSCALE_OUTPUT_MODES.md`.
- `data/03_Upscale_16K_15360x8640/<ID>/legacy_raw/`: isolated optional legacy encode artifacts, using the canonical committed 4K Outpainting input.
- `.../<ID>/quick/`: separate Quick artifacts; Full scans/archives only its owned subdirectories, not this nested Quick directory.
- `data/validation/`:05 regeneratedPreProc, first65 RGB framemd5, reproduction gate and baseline evidence. Never overwrites copied historical05PreProc.
- `state/runs/<job>/`: requested/resolved config snapshots.
- `state/locks/`: cross-process oneOutpaint/oneUpscale + same-video stage protection.
- `state/migration/`: full source/destination SHA copy inventory, media headers, decode-readability result, static/diff evidence.
- `state/batch_state.json`, `state/batch_request.json`: atomic state and byte-bound batch/job identity after a real queue launch.
- `logs/`: migration, CPU validation, static tests, per-job/batch terminal mirrors.
- `src/`: copied source/model package. `src/video_outpainting`, `src/flashvsr`, `src/models`, `src/manifests` retained byte-for-byte.
- `docs/reference_configs`, `docs/reference_entrypoints`: unmodified old JSON/entrypoint copies for diff evidence; not executable runtime imports.

All standard01–05 Certification/Quick final/chunk/segment/metadata/log directories exist before inference with `.gitkeep`. Empty folders are not evidence of successful generation.
