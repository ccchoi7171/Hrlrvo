# 05 reproduction evidence — historical attempt retained

The CRF1 attempt below is preserved unchanged as historical failure evidence. The user subsequently corrected Stage0001 to the complete historical CRF18 contract. Current **preprocessing-only** results are in [PREPROCESS_REPRODUCTION_01_05.md](PREPROCESS_REPRODUCTION_01_05.md), using distinct `data/validation/preprocess_reproduction/05/` artifacts.

This correction task does **not** execute 05 Outpainting/FlashVSR. No old GPU reproduction status is promoted to PASS by a preprocessing result. Original document bytes are preserved at `../state/preprocess_reproduction/before/docs/REPRODUCIBILITY_05.md`.

---
## Archived CRF1 attempt

# 05 Reproducibility Validation

## Verdict and stop decision

**VERDICT: TEST FAILED — preprocessing exact-input gate.**

**Overall requested migration acceptance: INCOMPLETE.**

The actual new Stage 0001 completed successfully as a video transformation, but its first 65 decoded frames **do not equal** the historical seed-search PreProc. The explicit identity gate failed; therefore the permitted downstream GPU reproduction test was **not started**. This is not a measured Outpainting tensor mismatch and is not a CUDA/model failure.

```text
COPY_SOURCE_INTEGRITY                  PASS
MODEL_WEIGHT_HASH                      PASS
CHECKPOINT_HASH                        PASS
CONFIG_COPY                            PASS (model conditions; layout separately adapted)
05_PREPROC_FIRST65_DECODED_IDENTITY     FAIL — 0/65 matched, first mismatch frame 0
05_OUTPAINT_REPRODUCTION                NOT RUN — prerequisite STOP condition

05 reference Outpainting GPU runs       0
05 new-pipeline Outpainting GPU runs    0
FlashVSR GPU runs                       0
Other GPU inference runs               0
Resident GPU processes killed/reset    NO
```

Machine-readable decision: `state/migration/acceptance_gate.json`.

## 1. Input identities — independently copied and SHA-verified

| Artifact | Path | SHA256 |
|---|---|---|
| Existing 05 source 8K | `../pipeline/data/00_Input_8K_7680x4320/05_Input_8K.mp4` | `684b400a0057478295e22ddf7aa5caf5b5ad1a4fb77fafdbf9a5c74baa2cb454` |
| New copied 05 source 8K | `data/00_Input_8K_7680x4320/05_Input_8K.mp4` | `684b400a0057478295e22ddf7aa5caf5b5ad1a4fb77fafdbf9a5c74baa2cb454` |
| Existing exact 05 PreProc | `../pipeline/data/01_PreProc_2K_1920x1080/05_PreProc_2K.mp4` | `751c3e53c0df078e7c5b4ba821d55b9c4b78fb12423bf2b2acff9db5447f8400` |
| New copied exact 05 PreProc | `data/01_PreProc_2K_1920x1080/05_PreProc_2K.mp4` | `751c3e53c0df078e7c5b4ba821d55b9c4b78fb12423bf2b2acff9db5447f8400` |
| Actual newly regenerated Stage 0001 | `data/validation/05_preprocess_from_8k/05_PreProc_2K.mp4` | `46df0bee9b675809986e24a572268500775d5a04356c2d28285ce6c693e286dc` |

The existing exact PreProc and its new canonical copy were **never overwritten or re-encoded** by this validation. Regeneration used a separate validation destination. A different MP4 SHA by itself was **not** used to declare failure; the decoded-frame test below was decisive.

## 2. Actual CPU execution

Executed the same `run_cert.run_preprocess_stage(config)` used by Full and `--stage preprocess`, with `configs/validation/05.json`. It calls the unchanged `cert_pipeline.preprocess.run_preprocess()`. This was a direct call of the real stage function, not a hand-built replacement FFmpeg preprocessing path. No model import/load was needed.

Validation output:

```text
data/validation/05_preprocess_from_8k/
├── 05_PreProc_2K.mp4
├── 05_PreProc_2K.json
└── metadata/
    ├── logs/preprocess.log
    └── runs/20260923T191436171280/
        ├── output/05_Input_FHD_24fps.mp4
        └── metadata/
            ├── input_ffprobe.json
            ├── output_ffprobe.json
            └── frame_alignment.json
```

The exact requested config was also retained as
`data/validation/05_preprocess_requested_config.json`.
Detailed terminal output: `logs/05_preprocess_validation.log`.

The inherited Stage 0001 used CPU FFmpeg `libx265`, preset `fast`, **CRF 1**, `yuv420p10le`, Lanczos/full-range scaling, `setpts=PTS*2/3,fps=24`, audio `atempo=1.5`, and unchanged count/probe/publish logic. Source count is resolved at runtime, not hard-coded in the new runner. The actual transformed output was verified as **1952 frames**, 1920×1080, 24fps before atomic publication.

## 3. Metadata gate

| Field | Exact historical PreProc | Regenerated audited Stage 0001 | Result |
|---|---|---|---|
| Width × height | 1920×1080 | 1920×1080 | PASS |
| FPS | 24/1 | 24/1 | PASS |
| Frames | 1952 | 1952 | PASS |
| Pixel format | yuv420p10le | yuv420p10le | PASS |
| Codec | HEVC | HEVC | PASS |

All five copied historical PreProc files were also fully CPU-decoded for readability during migration. That is a readability check, not an old/new equality check.

## 4. First65 decoded equality — actual result

Both MP4s were sequentially decoded by the same FFmpeg command shape, with no timestamp seeking, to the same `rgb24` pixel format:

```bash
ffmpeg -hide_banner -nostdin -v error -xerror -threads 8 \
  -i <old-or-regenerated-PreProc.mp4> \
  -map 0:v:0 -an -frames:v 65 -pix_fmt rgb24 -f framemd5 -
```

The frame payload MD5 column was compared by decoded frame index, not container metadata or timestamps.

```text
Old decoded frames:                     65
New decoded frames:                     65
Matched frame payloads:                 0/65
First mismatched frame (zero based):    0
Mismatched indices:                     0 through 64 inclusive
PREPROCESS_05_FIRST65_DECODED_IDENTITY: FAIL
```

Evidence:

- `data/validation/05_old_first65_rgb24.framemd5`
- `data/validation/05_new_first65_rgb24.framemd5`
- `data/validation/reproducibility_05.json`
- `logs/05_preprocess_comparison.log`
- `tools/compare_preprocess_05.py`

Full old/new 1952-frame equality was not additionally computed: the mandatory first65 gate had already definitively failed. No visual-similarity or tolerance metric overrode this exact-equality result.

## 5. Confirmed preprocessing mismatch

**CONFIRMED BY FILE/CODE:** historical `05_PreProc_2K.json` records `crf: 18` and the actual command `-crf 18`. The audited source `src/cert_pipeline/preprocess.py:38` defines `VIDEO_CRF = 1`; its actual execution log records `-crf 1`. This file was copied byte-identically.

The filter graph, dimensions, FPS and frame count match, but these are different lossy encode conditions. The differing CRF is an established configuration difference consistent with the observed pixel differences; it is not necessary to speculate about model behavior to stop the test. The historical command also specifies input decoder thread count explicitly while the audited builder does not. This validation did not run extra experiments to isolate every encoder/decoder option's contribution.

**No code/config/encoder parameter was changed to force equality.** In particular, the audited implementation was not changed to CRF18, and the historical CRF18 input was not replaced by CRF1 bytes.

This means two individually legitimate paths must not be conflated:

- **Fresh Full/Quick:** audited CRF1 preprocessing from 8K, followed by the shared Outpaint stage.
- **Stage-only/batch using the imported canonical cache:** historical exact CRF18 input after provenance/hash validation, followed by that same Outpaint stage.

The shared model stage is identical; its input pixels are not necessarily identical across these two input-producer histories. Exact old-seed reproduction from the requested fresh 8K start is therefore not established.

## 6. Model, checkpoint and requested conditions

```text
video                    05
seed                     71634910
steps                    30
positive prompt          empty string
negative prompt          people, person, human, staff, audience, crowd, backstage, seats
anchor                   1920×1080
canvas                   3840×2160
requested Outpaint       first65
LongRange                65 / RGB-prefix17 / stride48
checkpoint               checkpoint-007500
```

| Asset / config | SHA256 |
|---|---|
| `src/models/outpaint_checkpoint_007500/adapter_model.safetensors` | `1acabcbf9c6b81282639d9e5f745487191f5854452d75fc93350ba78a70ae951` |
| `src/models/ltx_2_3/ltx-2.3-22b-dev.safetensors` | `7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f` |
| `configs/validation/05.json` | `62d19eb7cb4f15dcef59df367616a4caeb8a039492579813ec953dce74314fee` |
| `configs/quick/05.json` | `af88e9c1bf059fe577e0681fd8a11e713861836ae1137d94bb00d1088168dee8` |

The complete config and asset hash inventory is in `state/migration/`. The validation config's `upscale=21` is retained from Quick solely to preserve the normal config contract; **no Upscale stage was invoked**.

## 7. Baseline search and eligibility

Existing candidate:

```text
/home/work/ai-graphics_33/Hrlrvo/pipeline/data/03_fullCheck_65f_30step/05/05_71634910_65f_30step.mp4
```

Actual candidate container SHA verified during this task:
`85cd586b3bbafeeedbd3f5be537cac1a9401e64fa0d53207cf3203c25607b23d`.

Its stored metadata records video05, seed71634910, checkpoint007500, steps30, first65 frames, the requested negative prompt, 3840×2160, and the video-only/streaming-VAE memory flags. It is an H264/yuv420p **24fps** historical rendered result, not the new stage's HEVC/16fps final encode.

However:

- A retained original `generated_latent.pt` was not established for this candidate.
- The candidate's original native generation segment was not retained.
- The later `02_VideoOutPaint.../chunk/05/segment_000000.npy` must **not** be promoted into a pristine native baseline: its `chunk_000000.json` explicitly records `bootstrap_first_chunk`, `model_inference_skipped=true`, `source=existing_first_chunk_mp4`. It was derived from the already encoded candidate MP4.
- The historical executable-byte identity to today's audited package is not fully established by those scalar config values alone.

Small evidence files were copied into `data/validation/baseline05_evidence/`:
`metadata.json`, `inference_manifest.json`, `chunk_lineage.json`, `resolved_config.yaml`,
`command.txt`, `later_full_bootstrap_chunk_000000.json`, `later_full_run.json`,
`candidate_file_identity.json`.

The candidate is documented as **CANDIDATE ONLY**, not falsely marked an eligible exact latent/native baseline. Had the prerequisite preprocessing identity passed, an eligible baseline or the authorized sequential reference/new A/B would have been needed. Since it failed, **neither A nor B was run**.

## 8. Outpainting comparison fields

| Requested comparison | Baseline | New result | Outcome |
|---|---|---|---|
| Final generated-latent SHA / shape / dtype / bitwise equality | UNAVAILABLE as a verified eligible baseline | NOT GENERATED | NOT RUN |
| Native generation segment SHA / array equality | UNAVAILABLE; later bootstrap excluded | NOT GENERATED | NOT RUN |
| Decoded final 4K RGB | Historical encoded candidate exists | NOT GENERATED | NOT RUN — not `0/65` |
| GPU numerical equivalence | Not established | Not established | INCOMPLETE |

The **0/65** result in this report concerns **FHD preprocessing**, not an Outpainting output comparison. No `allclose` or visual similarity was used to claim seed-exact reproduction.

## 9. GPU and process protection

Before and after read-only GPU snapshots:
`state/migration/gpu_before.txt`, `state/migration/gpu_after.txt`.
Both list the same resident H200 process **PID574892**, using 35,356 MiB process memory (device total usage 35,371 MiB at those snapshots).

No kill, reset, suspend, priority change, clock/MIG/persistence operation was used.
No idle wait or model-memory parameter alteration was attempted. There was no CUDA OOM because no task-owned GPU model was loaded.

The actual CPU validation parent exited normally; model child count was zero.
See `state/migration/process_cleanup.json`. The resident process was left untouched.

## 10. Acceptance boundary

The new package's copy integrity, common stage interfaces, queue state machine and static checks are complete. The requested exact 05 reproduction gate is **not** complete and is not waived.

The remaining decision is the conflict between the historical CRF18 seed-search input and the audited CRF1 fresh-preprocess implementation. This task records the conflict without changing model or preprocess computation to make the test pass. Until the intended input-production policy is resolved and the prerequisite identity gate plus permitted GPU comparison are passed, do not label this workspace `EXACT REPRODUCED` or production-validated end-to-end.
