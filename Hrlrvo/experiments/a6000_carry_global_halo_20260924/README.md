# A6000 four-arm FHD experiment (2026-09-24)

Remote root: `/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924`
Local source/protocol mirror: `/home/work/ai-graphics_33/Hrlrvo/experiments/a6000_carry_global_halo_20260924`

| GPU | Arm directory | Intervention |
|---|---|---|
| 0 | `arms/01_baseline` | Exact archived strength_1p0 inference modules; fresh C0 reproducibility run |
| 1 | `arms/02_no_carry` | RGB/VAE prefix instead of clean latent carry; baseline current-anchor authority retained |
| 2 | `arms/03_global_boost` | Global contribution gate scale 1.4 for steps 1–10, legacy for 11–30 in each chunk |
| 3 | `arms/04_native_halo` | Core Q/output unchanged; clamped KV halo radius **1 Native spatial token** (32px grid stride); 2-token Halo is explicitly forbidden |

All runs: checkpoint 007500 tensor weights, base seed **71634910**, chunk seed = base + absolute_start_frame, strength 1.0, temporal rebase ON, 65 frames / prefix 17 / stride 48, 30 steps, CFG 3, STG 1, guidance rescale 0.7, original empty positive + recorded negative prompt, identical 960×540@24 input, 1920×1080 visible / 1920×1152 encoded canvas. C0–C23 = 1169 frames, 48.708s. No strength 0.8 experiment. All four start from fresh C0; no existing generated prefix/carry copied into these runs.

## Per-arm saved files
- `source/main/src/ltx_outpaint/`: isolated inference source.
- `config.json`: intervention and actual base seed. `settings.json`: relocated baseline config; historical sampling.seed is overridden exactly as archived runner did, actual sampling saved in `effective_sampling.json` and `chunks/run.json`.
- `source_manifest.json`: SHA256 of every inference module; `effective_settings.json`: execution settings.
- `logs/latest.log`: symlink to current timestamped inference log; `logs/previews_*.log`: CPU preview encoder log.
- `outputs/chunk_previews/C00_full65.mp4`, `C01_new48.mp4` … `C23_new48.mp4`: each committed chunk preview.
- `outputs/C0_C23_1169f_1920x1080.mp4`: final combined video, created after all chunks finish.
- `outputs/C0_C23_manifest.json`: final output metadata.
- `chunks/segment_XXXXXX.npy`: lossless RGB24 emitted frames; `chunk_XXXXXX.json`: commit/runtime record; `previous_prefix_*.pt`: FP32 RGB prefix + RNG resume state.
- `latent_states/carry_XXXXXX.pt`: BF16 clean latent handoff; empty for no-carry arm.
- `producer.pid`, `status.json`: process and completion state. GPU allocation is fixed by `start_arm.sh` and isolated by CUDA_VISIBLE_DEVICES.

## Integrity and portability
`verification/source_isolation.json` records exact baseline source equivalence and per-arm one-file whitelist. `verification/asset_verification.json` hashes all required base/adapter/text encoder/tokenizer/input/runtime source files against original server. Genuine Git bundle checkout pins LTX revision `598ab41247a77dbfe29b5186e915bcf4f9040ec7` under `shared/LTX2.3`.

The original adapter embeds absolute base/source paths. `shared/checkpoint-007500/adapter_model.safetensors` is an isolated **metadata-only relocation**, leaving all tensor names/shapes/offsets and data bytes identical. `relocation_receipt.json` records original/runtime metadata, payload SHA and full relocated-file SHA. Original portable model/input files are not modified. This allows the unchanged strict baseline loader to validate on the other server without bypasses.

## Reproducibility interpretation
The historical strength_1p0 reused committed C0 from the modified branch and resumed C1 with recorded RNG. These runs regenerate C0, so compare C0 first; do not claim bitwise cross-GPU equality merely from equal seed. Lossless pixels/latent values are the comparison target, not lossy MP4 bytes. Source/weights/settings equivalence is verified before inference; generated-output reproducibility remains pending until actual outputs exist. Full 1952-frame 4K late-horizon behavior is not established by this 1169-frame FHD proxy.

Halo adds read support only; never writes into its extra KV strip and never wraps at screen boundaries. Existing Native core may overlap anchor and its baseline hidden-feature evolution is not newly frozen. Existing known-mask and 16px inward anchor composite policy are retained; this is not a separate hard-composite experiment.

## Monitoring / resume
All runs are in tmux session `hrlrvo`, windows `exp01_baseline`, `exp02_no_carry`, `exp03_global_boost`, `exp04_native_halo`.
`bash status.sh` prints GPU processes and last committed chunk/status for each arm.
For a failed/stopped run, `bash start_arm.sh ARM --resume` resumes only compatible committed source/settings/RNG state. Do not invoke while the arm or its assigned GPU is busy. No automatic baseline-setting changes or silent OOM fallback.
