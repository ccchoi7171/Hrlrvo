# Migration Diff Audit — 8K → 16K

## Current correction — historical reproduction contract

**Previous certification draft: CRF1.**  
**Corrected historical reproduction contract: CRF18.**  
**Reason: historical PreProc reproduction requirement.** The user explicitly corrected the earlier CRF1 specification; this is an intentional Stage0001 correction, not a hidden change to copied model computation.

The complete historical command is now enforced, not just CRF: input `-threads 8`, `-loglevel error`, one video + one audio with `-map 0:v:0 -map 0:a:0`, Lanczos/pc→pc, `setsar=1,setpts=PTS*2/3,fps=24`, `-fps_mode:v passthrough`, libx265/fast/CRF18/yuv420p10le, `-x265-params log-level=error:pools=8`, pc/bt709/bt709 tags, no color_transfer override, hvc1, atempo1.5/AAC192k, map_metadata0, timescale24576, `-n`, and historical `_PreProc_2K[.partial].mp4` naming. Full/Quick/validation JSON and schema carry the same fixed contract.

Current source: `src/cert_pipeline/preprocess.py:302-376` (`build_ffmpeg_command`), `:200-291` (input/output validation), `:422-473` (strict full video/audio decode), `:522+` (`run_preprocess`). Shared runner: `run_cert.py:918-989` (`run_preprocess_stage`). These anchors describe the corrected workspace, not the archived old package.

### Current Stage0001 control flow

```text
run_cert.sh -> run_cert.py:main()
  -> validate_config() [one PREPROCESS_CONTRACT / JSON/schema]
  -> execute_stages(stage=all or preprocess)
       -> run_preprocess_stage()
            -> preflight_source(): header nb_frames, historical source validation
            -> source SHA + producer_identity(contract, implementation SHA, tool versions)
            -> preprocess.run_preprocess()
                 -> header probe (no source -count_frames)
                 -> historical build_ffmpeg_command() and CPU libx265
                 -> output header/resolution/fps/count/pixel/color/timebase/audio/duration checks
                 -> strict full FFmpeg video/audio decode + count
                 -> atomic output + producer.json/full_decode.json/frame_alignment.json
            -> publish mutable generated artifact + SHA-bound receipt
       -> run_outpaint_stage() [only for all; model code unchanged]
            -> validate_preprocessed() -> selected generated artifact
```

Quick still preprocesses **all source frames**; only model frame limits differ. Startup Stage0001 no longer does the old repeated 8K `-count_frames` scans: it follows historical `nb_frames` handling and performs strict output full-decode verification. Full input SHA and full source downscale/encode still occur. FlashVSR's separate original-source checks are unchanged.

### Canonical reference preservation and effective artifact path

- Immutable imported reference: `data/01_PreProc_2K_1920x1080/<ID>_PreProc_2K.mp4` and its original JSON. Neither this validation nor future fresh Full publishes over that path.
- Normal fresh Full/Quick/stage-only preprocess output: `data/01_PreProc_2K_1920x1080/metadata/<ID>/generated/<ID>_PreProc_2K.mp4` plus receipt.
- `artifacts.preprocess` retains the declared canonical/cache path. `preprocess_output_path()` derives the mutable destination from `artifacts.preprocess_metadata`; `preprocessed_input_path()` prefers that generated artifact, else the validated canonical cache. Both Full and stage-only use this same selector.
- `resolved_config.json.resolved_preprocess_artifact` records the actual selected path. Invalid generated artifacts fail closed, not silently fall back.
- Validation configs explicitly route outputs to `data/validation/preprocess_reproduction/<ID>/`; they never publish normal/canonical outputs.
- Resume requires source identity, full corrected contract, implementation SHA and FFmpeg/FFprobe version identity. CRF1 receipts do not qualify. Stage-only validates historical exact command provenance or corrected generated producer identity.
- Batch queue code remains unchanged: it uses the existing stage-only Outpaint input validation. It has no independent preprocessing implementation and does not automatically create missing preprocessing artifacts; when needed, the shared `--stage preprocess` is the preparation path.

### Verification and historical evidence boundary

**Actual CPU acceptance: PREPROCESS REPRODUCED.** All five videos passed **65/65 first frames and 1952/1952 full RGB24 frames** against their current canonical inputs. All five regenerated MP4 SHA256 values also matched canonical. Evidence is recorded in [PREPROCESS_REPRODUCTION_01_05.md](PREPROCESS_REPRODUCTION_01_05.md) and `data/validation/preprocess_reproduction/summary.json`; this is an actual CPU experiment, not a contract-only inference. GPU/Outpainting/FlashVSR were not executed by this correction task. Outpainting's own CRF1 chunk/retime encoding and Flash/final composite settings are **unchanged**; only Stage0001 changed.

Old migration/static manifests describe the historical snapshot and are preserved, not rewritten to pretend the new producer is byte-identical. Current diffs, static checks and protected-scope evidence are under `state/preprocess_reproduction/`. The old `tools/final_static_validation.py` main encodes the previous byte-identity assumptions and is retained as historical verification code, not used as the corrected Stage0001 acceptance check.

### Updated change classification

| File / group | Current classification |
|---|---|
| `src/cert_pipeline/preprocess.py` | **HISTORICAL CONTRACT CORRECTION**, no longer byte-identical to the CRF1 audited draft |
| `run_cert.py` | **STAGE0001 CONTRACT / PUBLICATION ADAPTATION**; model stage functions are AST-identical |
| Full/Quick/validation JSON + job schema | **PREPROCESS CONTRACT CORRECTION**; seed/model/prompt/frame/final parameters unchanged |
| `src/video_outpainting`, `src/flashvsr`, `src/models`, Outpaint/Flash adapters | **UNCHANGED** by this correction |
| `run_batch.py`, `run_batch.sh` | **BYTE IDENTICAL** to pre-correction snapshot |
| Canonical input/PreProc MP4 + original sidecars | **UNCHANGED**, exact references preserved |
| New validation harness / CPU regressions / docs | **PREPROCESS VERIFICATION ONLY**, no duplicate producer |

---

## Archived pre-correction report — historical facts, NOT current Stage0001 behavior

The original document bytes are preserved at `../state/preprocess_reproduction/before/docs/MIGRATION_DIFF_AUDIT.md`. Statements below such as “current CRF1”, old source line numbers, copy identity, NO execution, or prior05 gate FAIL describe that earlier audit/migration. They must not override the current correction above. No past evidence has been silently rewritten.

### 1. Scope and status

Current audited pipeline (READ ONLY):
`/home/work/ai-graphics_33/Hrlrvo/pipeline`

New workspace:
`/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K`

Additional READ ONLY references:
`/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting` and
`/home/work/ai-graphics_33/Hrlrvo/Upscaler`.

**The new workspace was initially empty.** All reference transfers were COPY ONLY. No reference file was moved, removed, rewritten, checked out or cleaned. No external symlink substitutes a model/source asset. The new source tree contains zero symbolic links.

**Copy and static integration verification: PASS. Numerical/GPU acceptance is separate.** Read [REPRODUCIBILITY_05.md](REPRODUCIBILITY_05.md) and `state/migration/acceptance_gate.json` for the actual preprocessing/reproduction gate, rather than interpreting a source-copy PASS as a numerical reproduction PASS.

`docs/PIPELINE_EXECUTION_AUDIT.md` is a byte-preserved copy of the **old** audit. Its line numbers, old folder names and original `NO execution` statements describe that historical audit, not the new migration's CPU preprocessing experiment. This document provides the new orchestration map.

### 2. Copy evidence and verification method

- `state/migration/capacity.json`: capacity checked before copying; a 10 GiB reserve was required in addition to the planned bytes.
- `state/migration/copy_integrity.jsonl`: one record per copied file; original absolute source, relative destination, size, source SHA256, destination SHA256 and source mtime.
- `state/migration/copy_summary.json`: **6,332 files / 83,429,041,182 bytes**, all COPY/SHA PASS.
- `state/migration/final_integrity.json`: no source size/mtime changes since the copy; no destination changes outside the specifically permitted Outpaint layout adapter. Small files were rehashed during final review. Large weights were fully hashed at source and destination during copy, then checked for unchanged size/mtime; they were not pointlessly reread a third time.
- `state/migration/media_headers.json`: actual FFprobe headers for all ten input/PreProc MP4 files.
- `state/migration/preproc_decode.json`: all five copied PreProc videos fully decoded with CPU FFmpeg, return code 0 and empty error output.

The inventories retain the audited package's broad runtime dependency snapshot, including its LTX/Gemma/FlashVSR repositories and custom dependencies. No model cache directory outside that audited snapshot was guessed or indiscriminately imported.

### 3. File classification

| Files / subtree | Classification | Evidence / change |
|---|---|---|
| `src/video_outpainting/**` (286 copied files) | **BYTE IDENTICAL** | Complete SHA-verified audited snapshot; no migration edits |
| `src/flashvsr/**` (6 copied files) | **BYTE IDENTICAL** | Flash wrapper/runtime unchanged |
| `src/models/**` (5,975 copied files) | **BYTE IDENTICAL** | All weights, model definitions, text/tokenizer assets, repositories copied and SHA-verified |
| `src/manifests/**` (6 files) | **BYTE IDENTICAL** | Historical provenance is preserved; not rewritten to claim the new runner is unchanged |
| `src/cert_pipeline/preprocess.py` | **BYTE IDENTICAL** | Audited CRF1 Stage 0001 is unchanged |
| `src/cert_pipeline/flash_composite.py` | **BYTE IDENTICAL** | Tile reconstruction, 64px center feather, index reader, audio and encoding unchanged |
| `src/cert_pipeline/outpaint.py` | **DATA-LAYOUT CHANGE** | Exactly two path/name substitutions, below; no model operations altered |
| `run_cert.sh` | **BYTE IDENTICAL** | Existing environment selection and single entrypoint preserved |
| `run_cert.py` | **NEW ORCHESTRATION / DATA-LAYOUT CHANGE** | Shared stage selection, cache/provenance validation, snapshots, paths and locks |
| `_run_flash_native()` in `run_cert.py` | **PATH-ONLY CHANGE** | Former `run_flash()` body differs only in output stage lookup; AST comparison PASS after this one substitution |
| `configs/certification/*.json`, `configs/quick/*.json` | **DATA-LAYOUT CHANGE** | Only `output_root` changed and `artifacts` added; model/frame/seed/prompt/encoder settings identical to each corresponding original JSON |
| `docs/reference_configs/**` | **BYTE IDENTICAL** | Original JSON bytes retained separately |
| `docs/reference_entrypoints/**` | **BYTE IDENTICAL** | Original runner/scripts/layout retained as reference text; not imported by the new runner |
| `run_batch.py`, `run_batch.sh`, `configs/batch/**`, `configs/schemas/**` | **NEW ORCHESTRATION** | FIFO scheduler and schema, no model implementation |
| `configs/validation/05.json` | **NEW ORCHESTRATION** | Same 05 Quick computational values, dedicated validation artifact destinations |
| Input and canonical PreProc MP4/JSON | **BYTE IDENTICAL** | Exact copies; canonical historical PreProc was not re-encoded during migration |
| `tests/**`, `tools/**`, new docs, `state/migration/**` | **NEW ORCHESTRATION / VERIFICATION** | CPU evidence, no production model computation |

#### Exact Outpaint adapter diff

At `src/cert_pipeline/outpaint.py:639,644`:

```diff
-    mp4_dir = chunks_dir / "mp4"
+    mp4_dir = chunks_dir / "rgb"
-    final_path = stage_dir / "final" / f"{video_id}_Outpaint_4K_16fps.mp4"
+    final_path = stage_dir / "final" / f"{video_id}_Outpaint_4K.mp4"
```

Full machine-readable diffs:
`state/migration/outpaint_layout.diff` and `state/migration/runner_orchestration.diff`.
No migration hunk exists in LongRange planning, prefix construction, VAE packing, denoising, architecture, scheduler, alpha mask computation, Flash causal stream, automatic tiling or per-tile seed logic.

The static verifier also AST-compares these inherited runner helpers to the old runner:
`preflight_source`, `verify_video`, `seed_all`, `run_logged`,
`first_frame_checksums`, `upscale_input` — all identical.

### 4. Preserved computation contracts

The actual YAML is `src/video_outpainting/main/configs/inference_4k.yaml`:
`clip_frames=65`, `prefix_frames=17`, `stride_frames=48`,
`final_chunk_policy=repeat_last`, 30 inference steps. The packaged
`main/src/ltx_outpaint/inference.py` remains byte-identical to the audited version.

- The actual carry remains **17 decoded RGB frames plus the saved RNG state**, not the generated-latent recording.
- `generated_latent.pt` is the final denoised latent audit artifact. It was not promoted into a new recurrent state.
- `base_seed + absolute_start_frame`, causal VAE handling, CFG/STG, LoRA/IC choices, merge policy and 16px anchor feather remain unchanged.
- Outpaint's native 24fps sequence is still retimed to 16fps without an `fps=16` resampler; the inherited HEVC re-encode is unchanged.
- FlashVSR remains a **continuous causal temporal stream for each spatial tile**, not an independent per-frame upscaler or newly invented temporal segment planner.
- FlashVSR's automatic tile decisions, seed isolation per tile, emission PNG persistence and causal merge are unchanged.
- Final original-8K center insertion and its **64px** feather remain unchanged.

**This proves copied computation code identity, not GPU numerical equivalence.** No numerical equivalence is inferred from hashes, syntax or mock-based queue tests.

### 5. Exact new call graph

```text
run_cert.sh (unchanged shell)
  -> run_cert.py:main()
       -> validate_config() [paths/config/YAML/assets; no model load]
       -> fresh downstream-artifact precheck
       -> requested_config.json
       -> execute_stages(config, selected_stage)
            all / preprocess -> run_preprocess_stage()
                                -> preflight_source()
                                -> cert_pipeline.preprocess.run_preprocess()
                                -> preserve old canonical copy, atomic publish + provenance
            all / outpaint   -> run_outpaint_stage()
                                -> validate_preprocessed()
                                -> snapshot_resolved_frames()
                                -> prepare_stage()
                                -> seed_all()
                                -> cert_pipeline.outpaint.run_outpaint()
                                -> unchanged native sample_long_range()
                                -> final SHA + stage_complete.json
            all / upscale    -> run_upscale_stage()
                                -> validate_outpaint_artifact()
                                -> prepare_stage()
                                -> _run_flash_native()
                                -> inherited upscale_input()
                                -> packaged cert_pipeline/flash_composite.py child
                                -> packaged FlashVSR child / causal tile inference
                                -> original-center composite + audio + final verification
       -> resolved_config.json, PIPELINE_DONE

run_batch.py:_production_child_runner()
  -> a separate run_cert.sh --stage outpaint process
       -> SAME run_outpaint_stage()
  -> a separate run_cert.sh --stage upscale process
       -> SAME run_upscale_stage()
```

Source anchors: `run_cert.py:801` (`validate_preprocessed`), `run_cert.py:880` (`snapshot_resolved_frames`), `run_cert.py:901` (`run_preprocess_stage`), `run_cert.py:969` (`run_outpaint_stage`), `run_cert.py:1025` (`validate_outpaint_artifact`), `run_cert.py:1054` (`run_upscale_stage`), `run_cert.py:1078` (`execute_stages`), `run_cert.py:1094` (`main`).
Queue dispatch: `run_batch.py:232,278`.

There is no alternate Quick/model implementation. The only computation difference between certification and Quick is the frame limits. Output identifiers/layout differ as expected.

### 6. Stage/cache contract and non-destructive behavior

#### Fresh Full / Quick

Both start at the configured 8K source and actually call the audited Stage 0001. An existing canonical `01_PreProc.../<ID>_PreProc_2K.mp4` is **not** an implicit cache hit. When publishing fresh Stage 0001, the new workspace's prior canonical MP4 and JSON are first copied to `metadata/<ID>/preserved/<timestamp>/`.

All output locations are validated against canonical video/job paths before execution. A dirty downstream stage is rejected before fresh Full preprocessing unless `--resume` or `--force` is explicit. `--force` archives only the selected new-workspace stage's direct owned subdirectories; it cannot archive input, another video, the reference roots, or nested Quick results.

#### Explicit Stage 0001 resume

Reuse requires a matching `run_preprocess_stage` producer, CRF1 preprocessing settings, original source identity and valid output metadata/SHA. A historical imported CRF18 PreProc is not silently accepted as the fresh Full CRF1 output.

#### Stage-only Outpaint / batch cache input

No Stage 0001 is executed. `validate_preprocessed()` checks:
file existence; decoded FHD frame count; 1920×1080/24/yuv420p10le;
canonical SHA against sidecar; original source SHA and video basename;
manifest source/output frame counts; original 8K dimensions/fps/count;
and output audio presence against the original source's audio presence.
The original header count is used when available, otherwise a count-decode is performed.

An explicitly validated historical CRF18 cache remains valid as an operational model input. Its producer CRF is logged. This is intentionally distinguished from fresh Full's audited CRF1 preprocessing.

#### Stage-only Upscale

Requires a committed Outpaint stage receipt with matching computational request, final MP4 SHA, all four chunk/manifests files and verified 4K/16 frame count. It cannot silently accept an unrelated MP4 placed in `final/`.

#### Locks

Global locks separately cap Outpaint and Upscale at one process per workspace. A per-video lock prevents same-video upstream mutation while another stage uses that video's data. Different-video Outpaint and Upscale may overlap. All locks are under the new workspace `state/locks/`; no resident process or GPU state is modified.

#### Empty native-store compatibility

The precreated output layout uses `.gitkeep` files. Native `SegmentedOutputStore.prepare()` rejects any nonempty fresh journal directory, and the audited Outpaint wrapper also rejects a nonempty native folder without `run.json` on resume. New `prepare_stage()` therefore removes **only zero-byte, non-symlink `.gitkeep` placeholders within the selected stage's owned subdirectories** before invoking the unchanged runtime. Nonempty/symlink markers fail visibly. A CPU regression covers both fresh and initial `--resume`. This is layout orchestration, not a change to native journal or temporal semantics.

### 7. Static / CPU integration verification

See `state/migration/static_validation.json`, `logs/static_validation.log`,
`logs/cpu_tests.log`, `logs/ruff.log`.

- **641 Python files** compiled from bytes without importing model code or writing bytecode: PASS.
- Shell `bash -n run_cert.sh run_batch.sh`: PASS.
- Eleven job JSONs: parse, all local-schema keywords and the real `validate_config()` function: PASS.
- Batch schema/JSON: PASS. No new JSON-schema dependency was installed. The CPU schema checker explicitly handles every keyword present in these files; it is not presented as a general-purpose JSON Schema implementation.
- Corresponding ten original/new job configs compared after removing only layout keys: no computational differences.
- Ruff over new runners, certificate adapters, CPU tests and migration tools: PASS. Unmodified third-party/vendor packages were syntax-checked, not reformatted or subjected to style fixes.
- **20 CPU-only unit tests**: PASS, including shared dispatch, fresh preprocess/non-silent cache, source SHA binding, audio provenance, path escape, force/Quick isolation, locks, pre-model snapshots, FIFO overlap, failures, resume races and request binding.
- Active runtime/config scan: no direct dependency on the old pipeline, VideoOutpainting or Upscaler absolute roots. Their names remain legitimately in provenance, this document and migration utilities.
- Required certification/Quick stage directories: present. `src` external symlinks: zero.
- Runner/config import audit: `torch` was not imported by static validation.

The queue was tested with deterministic fake child runners and temporary state, not with GPU models. Full, Quick and FlashVSR end-to-end inference were **not** run as part of these tests.

### 8. Remaining boundaries

1. Preprocessing identity and 05 GPU reproduction are hard gates; see the actual result in `REPRODUCIBILITY_05.md`.
2. The copied historical `src/manifests/*` are **provenance snapshots**, not current-runner manifests. New migration/config identities are under `state/migration/`.
3. The existing audited startup count-decode cost, tile×frame PNG storage, 16K assembly cost, optional `full_decode=false` verification scope and audio exact-sample-length gap remain documented in the copied audit. They were not disguised as migration fixes.
4. `--force` archives subdirectories sequentially. A filesystem failure during archival stops visibly and can leave a partial archive; it never rolls back or edits a protected reference.
5. Current compatible Conda/Python environments are assumed; no environment was created, upgraded or otherwise changed.
