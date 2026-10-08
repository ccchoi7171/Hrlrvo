# Recurrence / LineArt Integration

## Status and scope

**READY FOR FULL QUEUE — code/config/CPU checks plus preserved C0 GPU evidence.**

C0 GPU exact preservation is **PASS**. At the user's request, the 113-frame smoke
was stopped during its C0 (last completed denoising step 23/30); C1 was not reached
and no smoke chunk was committed. It is **CANCELLED BY USER**, not a C1 PASS or a
numerical failure. The revised acceptance explicitly uses code/config/CPU tests
for C1+ wiring and prohibits additional inference. No implementation or config
was changed during that final review. Only CPU/static tests and audit evidence
were added/updated. The earlier independent CPU final-retime regression is PASS.

Only `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K` is changed. The sibling
`pipeline`, `VideoOutpainting`, `Upscaler`, and experiment snapshots are read-only.
No FlashVSR inference or long Full run is part of this validation.

This change deliberately improves **C1 and later**. C0 means the first complete
65-frame generation, zero-based chunk index 0. The seed-search C0 must remain
exact. The implementation is not a new experimental runner.

## Source of truth / provenance

All experiment paths below are under
`/home/work/ai-graphics_33/Hrlrvo/experiments/`:

| Purpose | Read-only source |
|---|---|
| Temporal rebasing, hybrid carry, generated-prefix strength | `longrange_fhd_recurrence_test_20260923/source/VideoOutpainting_test/main/src/ltx_outpaint/inference.py` |
| Recurrence runner/options | Same experiment's `source/VideoOutpainting_test/main/scripts/run_strength1_chain.py` and `patches/inference_fhd_proxy.patch` |
| Final prior implementation | `longrange_fhd_lineart_from_c07_20260924/source/VideoOutpainting_test/main/src/ltx_outpaint/{inference.py,lineart_global_prior.py,global_attention.py}` |
| Final experimental runner | Same snapshot's `scripts/run_lineart_chain.py` |
| Applied experimental options | `longrange_fhd_lineart_from_c07_20260924/commands/run_lineart_from_c07.sh` and `configs/effective_strength_1p0_global_lineart_from_c07_lineart.json` |
| Extraction algorithm | `longrange_fhd_lineart_global_prior_20260923/scripts/make_lineart_candidates.py` |
| Selected extraction provenance | `longrange_fhd_lineart_global_prior_20260923/reference/candidates/lineart/detailed.png` |

The selected experimental PNG is **not** reused for unrelated videos. Each video
creates its own prior from its own committed C0. Experiment-specific FHD model
geometry, external-prefix runners, C07 start, and proxy-only settings were not
copied into production. Source SHA/diff evidence is stored with this integration.

## Files and shared call path

Existing files changed:

- `src/video_outpainting/main/src/ltx_outpaint/inference.py`: selective changes in
  `_condition_latents`, `sample_chunk`, `sample_long_range`; no wholesale FHD copy.
- `src/video_outpainting/main/src/ltx_outpaint/global_attention.py`: scoped prior,
  exact legacy fallback, eligible-query selection and known-position restoration.
- `src/cert_pipeline/outpaint.py`: passes the explicit recurrence contract,
  records producer hashes, archives/validates actual causal carry artifacts.
- `run_cert.py`: validates recurrence and isolated GPU-validation layouts;
  includes `lineart/` in stage artifact receipts and archive/reuse handling.
- Existing job JSON files: only `outpainting.recurrence` added. The schema records
  the operational fixed contract. Model/inference parameters are unchanged.

New runtime modules:

- `src/cert_pipeline/recurrence.py`: dependency-light contract and validation.
- `src/video_outpainting/main/src/ltx_outpaint/lineart_global_prior.py`.
- `src/video_outpainting/main/src/ltx_outpaint/lineart_reference.py`.
- `src/video_outpainting/main/src/ltx_outpaint/recurrence_artifacts.py`.

New verification surfaces (not inference implementations):

- `tests/recurrence_cpu.py`: standalone CPU tensor/regression tests.
- `tests/recurrence_compare.py`: CPU-only comparison of existing artifacts.
- `configs/validation/recurrence_lineart/c0.json`: isolated 05/65-frame request.
- `configs/validation/recurrence_lineart/c1.json`: isolated 05/113-frame request.

```text
run_cert.sh -> run_cert.py.main -> execute_stages
  Full / Quick / --stage outpaint
    -> run_outpaint_stage
      -> cert_pipeline.outpaint.run_outpaint
        -> ltx_outpaint.inference.sample_long_range
          -> same sample_chunk for C0 and C1+, with scoped optional recurrence

run_batch.sh -> run_batch.py
  independent Outpaint child process
    -> run_cert.sh --config <job.json> --stage outpaint
      -> the identical run_outpaint_stage above
```

The validation JSONs may set `validation.native_outpaint_only=true`; this selects
only the output boundary of the same stage, not a second sampler. Such configs
are restricted to recurrence C0/C1 audit jobs and `--stage outpaint`; they cannot
enter FlashVSR. Normal Full and Quick finish the complete requested sequence and
then create the 16fps deliverable.

`run_batch.py` and `run_batch.sh` are unchanged. There remains one Outpaint worker
and one Upscale worker, with separate OS process RNG states. Preprocessing and
FlashVSR computation are unchanged.

## Explicit operational contract

Every Full and Quick JSON records:

```json
{
  "rebase_temporal_position": true,
  "use_latent_carry": true,
  "generated_prefix_strength": 1.0,
  "lineart_global_prior": true,
  "lineart_bias_schedule": [
    {"start_step": 0, "end_step": 9, "strength": 0.5},
    {"start_step": 10, "end_step": 19, "strength": 0.3},
    {"start_step": 20, "end_step": 29, "strength": 0.1}
  ],
  "lineart_source": "chunk00_first_final_rgb_frame",
  "lineart_start_chunk": 1,
  "lineart_reference_resolution": [3840, 2160],
  "lineart_global_map": [30, 17],
  "lineart_density_normalizer": 0.05
}
```

The full JSON additionally records extraction parameters and direct 4K
Global footprint density method. `generated_prefix_strength` is a separate latent amplitude
interface, **not** `VideoConditionByMask.strength`. Operational value 1.0 takes
no attenuation operation. The validator supports finite values in [0,1] for
future explicit experimental control; released operational schema/configs fix
1.0 and enabled features. No inference-condition CLI override was added.

## C0 unchanged design

`enhanced = chunk_index > 0` is established in the existing native chunk loop:

| Property | C0 | C1+ |
|---|---|---|
| Temporal rebase | False; retain original `positions += 0` | True; omit absolute-time addition |
| Incoming latent override | `None` | Previous final-clean CPU carry |
| Prefix amplitude | 1.0 and no generated prefix | Recorded setting, operational 1.0 |
| LineArt context | No bound prior (`nullcontext`) | Scope starting chunk index 1 |
| Global attention | Original `attend` then one decompressor | Unmasked + biased path, selective restoration |
| Output/global frame numbering | Original | Original |
| Seed | `base_seed + 0` | `base_seed + absolute_start_frame` |

The original anchor-only `_condition_latents` fast route is retained. Scheduler,
CFG/STG, initial noiser/generator, checkpoint, Local/text branches, VAE decode and
16px RGB feather are not substituted. Existing final-latent observation still
runs at the same point. New carry extraction occurs **after denoising**, and is
nonmutating/nonrandom. C0 native RGB commit and artifact observer finish before
CPU LineArt extraction. Captured RNG state is restored after observers/extraction.

The actual 05 GPU comparison, not just source review, confirms identical C0
denoised latent, 65 native RGB frames, and 65 decoded native24fps RGB frames.

## C1+ temporal and causal semantics

Planner remains `clip=65`, `prefix=17`, `stride=48`; end policy remains repeat-last.
113-frame validation uses exactly:

| Chunk | Source input (inclusive) | Reused RGB prefix | New emitted frames | Seed |
|---|---|---|---|---|
| C0 | 0–64 | None | 0–64 (65) | 71634910 |
| C1 | 48–112 | 48–64 (17) | 65–112 (48) | 71634958 |

Rebasing changes only RoPE temporal positions in C1+. Source reading, output
ordering, absolute range manifests and seed calculation remain absolute.

For 65 RGB frames the causal VAE yields 9 temporal latent slices. Slice 0 is a
single-frame keyframe; later slices cover groups of 8 frames. At the end of C0,
the last three clean slices represent 41–48, 49–56, 57–64. Directly treating all
three as a new 17-frame clip is causally wrong: its first slice must mean frame48
alone, not the preceding eight-frame span.

The implemented hybrid sequence is:

1. `tools.clear_conditioning(state)` and `tools.unpatchify(...)` expose final clean
   latent `[1,128,9,68,120]` for production4K. Copy the last 3 slices to CPU,
   preserving bfloat16, as `[1,128,3,68,120]`.
2. Keep the original **17 decoded FP32 RGB frames + RNG state** carry unchanged.
3. For C1 conditioning, VAE-encode `previous_prefix[:1]` (global frame48) as the
   correct single-frame keyframe and replace carried slice0.
4. Keep carried slices1–2 directly (global49–56 and57–64).
5. Overwrite the entire current anchor box using the current chunk's independent
   anchor encode. Current observed input wins over carried anchor values.
6. Optional strength affects only left/right generated prefix side context,
   matching experiment semantics. At1.0 the multiplication is bypassed.

`generated_latent.pt` remains the existing recorded final denoised **patched**
state. It is not relabeled as recurrence carry. `carry_prefix_state.pt` remains
native RGB/RNG state. The new causal latent is stored separately.

## Direct 4K LineArt extraction and native Global footprints

The latest user correction supersedes the earlier FHD adaptation proposal.
The runtime contains **no downscale, FHD intermediate prior, or interpolation**:

```text
C0 final 4K RGB frame0 (3840×2160)
  → direct 4K LineArt (3840×2160)
  → direct density on existing30×17 Global footprints
  → g=clip(density/0.05,0,1)
  → C1+ Global self-attention prior
```

Input is C0's first **final composited native RGB** frame from the SHA-validated
uint8 segment, after the existing16px feather and C0 commit, before lossy MP4
re-encoding. This is neither an anchor-only frame nor an intermediate VAE decode.

1. Preserve the exact3840×2160 final frame as `source_frame_4k.png`.
2. Direct RGB→gray at3840×2160.
3. `GaussianBlur((3,3),0.8,BORDER_REFLECT_101)`.
4. `Canny(35,100,apertureSize=3,L2gradient=True)`.
5. Keep8-connected components only when pixels≥12 **and** max(width,height)≥12.
6. Save `lineart_4k.png`: black background/white thin lines. No CLAHE, dilation,
   morphology or thinning. No numerical extraction parameter is scaled with4K.
7. PNG readback must equal the original source array before atomic commit.

### Direct density calculation

Production geometry stays2160→2176 (8px top/bottom padding), VAE factor32,
latent68×120, GlobalC4→17×30. Each actual Global cell has a128×128 encoded-pixel
footprint. Add8 rows of **zero line evidence** above/below the visible4K line map,
then compute:

```python
binary = (lineart_4k > 0).astype(np.float32)       # [2160,3840]
padded = np.pad(binary, ((8, 8), (0, 0)))         # [2176,3840]
density = padded.reshape(17,128,30,128).mean((1,3), dtype=np.float64).astype(np.float32)
g = np.clip(density / 0.05, 0, 1).astype(np.float32)  # [17,30]
```

This is direct pooling on the existing model's exact footprints, not spatial
interpolation and not a change to model geometry. Grid width×height is30×17;
saved NumPy shape is `[height,width]=[17,30]`. Row-major flattening matches native
`global_cells` ordering and temporal token flattening. Outside the visible frame,
synthetic VAE-padding rows contribute zero observed line evidence; the nominal
128×128 denominator is unchanged. Actual VAE RGB reflect-padding is unchanged.

Metadata records source/LineArt3840×2160, grid30×17, extraction values, padding,
density normalizer, lambda schedule, and SHA256 of both PNGs and both numerical
maps. CPU tests check exact footprint boundary pixels and edge denominators.

### Global-only injection

The prior is bound through a `ContextVar` only during C1+ `sample_chunk`. It has
no parameters and never modifies the checkpoint or Local/text attention code.

For each real scheduler sigma, resolve its actual zero-based step (not a block
call counter), then form `B(i,j)=lambda*g_i*g_j`. Eligible coarse queries require:

- temporal latent index≥3 (after the17-frame prefix);
- a coarse cell containing generated, non-anchor positions;
- query line strength>0.

All key tokens remain available. Ineligible queries take the original unmasked
attention result. Learned spatial/temporal decompression can mix neighboring
cells; after decompression, `torch.where` restores the **original Global output**
at every prefix or anchor fine position. Only newly generated fine positions
receive the selected result. Device bias/mask caches are discarded per chunk.

## Persistence, resume and failure boundaries

Normal production artifact root:
`data/02_Outpaint_4K_3840x2160/<ID>/` (Quick uses its existing `quick/` subdirectory).
Validation runs are isolated under `data/validation/recurrence_lineart/<job>/outpaint/`.

```text
<ID>/
  chunks/
    native/
      05_Outpaint_4K_24fps.mp4            # retained complete native24fps merged master
      segment_000000.npy                 # original emitted uint8 RGB
      previous_prefix_0.pt               # original alternating FP32RGB/RNG state
      causal_latent_carry_000000.pt       # NEW final-clean last3 CPU slices
      chunk_000000.json                  # native atomic commit, includes carry SHA
      run.json                          # settings + producer identity
    latents/chunk_000000/
      generated_latent.pt                # unchanged actual final denoised state
      carry_prefix_state.pt              # unchanged RGB/RNG audit copy
      causal_latent_carry.pt              # NEW verified copy of actual latent carry
      metadata.json
      certification_artifacts.json      # includes new carry's shape/dtype/SHA
    rgb/chunk_0001_frames_000000_000064.mp4
    manifests/...                       # original plan/result/lineage manifests
  lineart/
    source_frame_4k.png
    lineart_4k.png
    global_line_density.npy
    global_line_strength.npy             # direct native17×30
    lineart_metadata.json                # extraction/source/file SHA, committed last
  metadata/inference_config.json          # recurrence contract + runtime file SHA
  final/<ID>_Outpaint_4K.mp4               # final-only lossless24→16 retime
  final/<ID>_Outpaint_4K.retime.json        # exact RGB sequence/frame-clock proof
  logs/outpaint.log
```

Causal carry is fsync+atomic rename before native chunk commit. Every completed
carry is checked for SHA, shape, dtype and finiteness before resumed model load.
The last completed carry is reloaded along with the original RGB/RNG state.
Wrapper copy checks protect both native and archived versions. A committed
record missing a carry is an error, not a fallback to guessed state.

C0 LineArt metadata is committed last. On resume, PNGs/maps are re-derived from
the independently validated C0 native segment and compared exactly, in addition
to source/contract/file-set/SHA checks. An attacker or accidental edit cannot
replace both a map and its own sidecar hash and pass this check. A crash before
the sidecar commit permits reconstruction of derived LineArt files; committed
corrupt metadata/artifacts fail. C1+ records bind the LineArt metadata SHA.

The wrapper passes hashes of all five recurrence runtime files into the native
run identity. Old RGB-only sessions or a changed producer cannot silently resume
as enhanced sessions. Existing pre-feature outputs remain preserved; use distinct
jobs or the existing explicit archive mechanism rather than overwriting them.

## Expected new logs

In addition to existing model/chunk/30-step/artifact logs:

```text
[OUTPAINT_RECURRENCE] <complete recorded JSON contract>
{"event":"recurrence_chunk_start","chunk_index":0,"c0_legacy_path":true,
 "rebase":false,"latent_carry":false,"generated_prefix_strength":1.0,"lineart":false}
... actual original C0 denoising and commit ...
{"event":"lineart_reference_ready","path":"<stage>/lineart",
 "metadata_sha256":"<sha>","start_chunk":1,"nonzero_source_cells":<count>}
{"event":"recurrence_chunk_start","chunk_index":1,"c0_legacy_path":false,
 "rebase":true,"latent_carry":true,"generated_prefix_strength":1.0,"lineart":true}
... actual C1 denoising and commit ...
```

Prior active-call counts, actual per-step calls, bias range, source/runtime map
sizes and configured attention backend identities are in the C1 chunk record.
These are measured call observations, not a synthetic progress timer. Configured
backend identity does not prove which fused kernel ran; no profiler claim is made.

## Validation protocol and results

Pre-edit byte snapshot: `state/recurrence_lineart/before/` and `before_index.json`.
An exact-configuration historical05 65-frame MP4 exists, but its located
`segment_000000.npy` is a **decoded-MP4 bootstrap** (`bootstrap_first_chunk_artifact`
in that run's request), not an original lossless generation segment. No original
final denoised latent was found. One original pre-edit stage-only C0 run therefore
supplies the missing strong latent/native/final comparison baseline. It was started before model runtime edits and had imported
its original model/attention modules before modifications.

| Check | Result / evidence |
|---|---|
| CPU recurrence regressions | PASS:13; `state/recurrence_lineart/cpu_recurrence.log` |
| Stage/queue/preprocess/retime + final AST/mock regressions | PASS:45; `state/recurrence_lineart/final_cpu_mock.log` |
| Fresh CPU recurrence regressions | PASS:13; `state/recurrence_lineart/final_cpu_recurrence.log` |
| Syntax/Ruff/shell/config/static scope | PASS:39 syntax files,18 operational job configs plus batch schema, shell, Ruff; `static_checks.json`, `final_schema_scope.json`, `final_ruff.log` |
| 05 generated latent exact | PASS: identical bfloat16 tensor `[1,73440,128]`; `c0_exact_comparison.json` |
| 05 native RGB65/65 exact | PASS: identical uint8 `[65,2160,3840,3]`; `c0_exact_comparison.json` |
| 05 native24fps decodedRGB65/65 exact | PASS: ordered rgb24 framemd5 equality; no retime in this gate |
| Native24fps MP4 vs historical seed search | PASS: baseline and candidate both have historical SHA `85cd586b3bbafeeedbd3f5be537cac1a9401e64fa0d53207cf3203c25607b23d` |
| 05 CPU causal carry / direct4K reference | PASS: `[1,128,3,68,120]` bfloat16 carry, 4K PNGs and `[17,30]` maps; `c0_recurrence_artifact_checks.json` |
| Final-retime common production helper | PASS: 65/65 exact decoded RGB, nominal/average16/1; `final_retime_contract.json` |
| C1 two-chunk113-frame smoke | CANCELLED BY USER before C1; `c1_cancelled_by_user.json`, `c1_terminal.log`; no output/quality PASS claimed |
| C1+ wiring / carry / query-only prior / queue | PASS by source/AST/CPU/mock checks; `final_acceptance.json` |
| Final-review implementation/config freeze | PASS:56 file SHA checks unchanged, four existing C0/retime evidence files unchanged; `final_schema_scope.json` |
| Current native runtime vs executed exact-C0 producer | PASS: all five recurrence runtime file SHA256 values still match executed C0 `chunks/native/run.json` |

Comparison is tensor dtype/shape/bitwise equality, per-frame uint8 native equality,
and identical CPUFFmpeg rgb24 decoded-frame hashes from the **native24fps master**.
The C0 gate never consumes a16fps deliverable. Container SHA alone is not
an acceptance criterion. Allclose, if reported for failure diagnosis, never passes
an exact-reproduction gate.

Exact C0 evidence (all paths are inside the new workspace):

- Before: `data/validation/recurrence_lineart/05_recurrence_baseline/outpaint/`.
- After: `data/validation/recurrence_lineart/05_recurrence_c0/outpaint/`.
- Latent file SHA before/after:
  `647a1b1c120288c245e2d1a17b5631cc79210b756dff941f22990ff8e178f9f7`.
- Native RGB pixel SHA before/after:
  `5bc0a94c42ecb6586ed483bee24d388f2c3e5bb3d367a3378773deeee375f139`.
- Comparison receipt: `state/recurrence_lineart/c0_exact_comparison.json`.
- The actual C0 request predates the latest output-boundary correction and is
  preserved verbatim at `state/recurrence_lineart/c0_executed_request.json`.
  The current `c0.json` now selects native-only validation for future use; this
  config update did not regenerate or replace either accepted native C0 result.
- `baseline.json` is a frozen pre-feature request, not an operational config
  accepted by the new recurrence schema. It is deliberately preserved unchanged.

Historical validation commands (same stage implementation). These are provenance,
**not instructions to rerun**: the final review executed neither command below.

```bash
./run_cert.sh --config configs/validation/recurrence_lineart/c0.json --stage outpaint
# Started only after C0 exact comparison passed; subsequently cancelled by user:
./run_cert.sh --config configs/validation/recurrence_lineart/c1.json --stage outpaint
```

GPU inventory is read-only and saved before/after. No resident process is killed,
reset, suspended or reprioritized. Each validation child exits before the next GPU
run starts. CPU tests force `CUDA_VISIBLE_DEVICES=''` and load no model weights.

## Unchanged and limitations

- Historical CRF18 Stage0001 reproduction code/canonical files are unchanged.
- Seed policy, steps30, checkpoint007500, prompts,16px feather, causal VAE core,
  native scheduler, pinned LTX/LoRA architecture and output merge remain unchanged.
- FlashVSR, final64px center composite and queue concurrency are unchanged.
- No claim is made for other videos' GPU numerical equality, C1 numerical output,
  long-run perceptual quality, or Full throughput. Two complete 05C0 runs supplied
  the original/new exact comparison; the subsequent gated113f run was interrupted
  during its C0. There was no further GPU inference after the stop instruction.
- Per-video LineArt is generated lazily after that video's C0; precreated01–05
  directories do not mean all five references have been generated.
- The extraction numerical values are unchanged from the successful experiment,
  but now operate directly at4K, as requested. Full-run visual quality remains
  outside this minimal functional smoke test; the model grid is unchanged.
- Dense additive bias and the additional masked attention/decompression increase
  C1+ work. No claimed speed or peak-memory target is inferred from CPU tests.
- The low-level packaged sampler imports the pure `cert_pipeline.recurrence`
  contract. All supported pipeline entrypoints add `pipeline_8K_to_16K/src` to
  `sys.path`; standalone native-CLI use without this package path is not covered.
- External prefix/MP4 bootstrap is rejected when recurrence is enabled: it lacks
  the original C0 clean carry/reference provenance. Fully disabled recurrence
  retains the older standalone route; released configs enable the new behavior.
- Standalone `keep_intermediates=False` cleanup does not remove the new carry
  files. Supported certification stages always preserve intermediates, as required.

## Runtime discovery: existing16fps output defect (not LineArt)

The pre-edit C0 run completed all30 denoising steps and committed65 native frames,
RGB/RNG state, latent, chunk MP4 and processing24fps MP4. That processing MP4 is
**byte-identical** to the historical05 seed-search MP4:
`85cd586b3bbafeeedbd3f5be537cac1a9401e64fa0d53207cf3203c25607b23d`.

The original `_retime_to_16fps` then encoded without an explicit output rate/time
base. FFmpeg retained the24fps encoder clock despite `settb=1/16,setpts=N` and
`fps_mode=passthrough`; verification reported avg FPS `780/49`, not16. The original
run exited1; its generated baseline artifacts are preserved. This is an existing
CPU encoding defect, not a denoising/C0 failure. The final temporary invalid MP4
was removed by the existing function's `finally`; latent/native/24fps remain.

See `state/recurrence_lineart/baseline_result.json` and `baseline_terminal.log`.
The subsequent user instruction explicitly authorizes correcting the **final merged
output postprocess** and separates this defect from the C0/C1 native24fps gates.
No baseline GPU rerun is required. The in-flight candidate had already loaded the
old postprocess code and also exited1 at that obsolete final FPS check. Its native
results have nevertheless passed the exact C0 gate above. No successful final
stage receipt is fabricated for either exited failure.

## Final-retime contract (latest timing correction)

Invariant:

```text
LongRange internal domain = always24fps
Native merged Outpainting master =24fps, retained
Final Outpainting deliverable = same decoded RGB sequence, retimed once to16fps
FlashVSR input = only the final16fps deliverable
```

`sample_long_range` completes every chunk and the65/17/48 contiguous emission
merge before `outpaint.run_outpaint` can call `_retime_to_16fps`. No chunk callback,
prefix/carry/resume operation, LineArt extraction, or attention code calls retime.
The corrected chunk MP4 writer uses24fps, and these audit MP4s are never
substituted for native RGB/RNG/latent carry. Frozen pre-correction baseline/C0
audit chunk MP4s retain their former16fps timing as historical evidence only;
neither those files nor an old final-retime output are used by the exact gate,
recurrence, or the new C1 smoke. No GPU regeneration was performed to replace
those audit-only files.
An AST regression test verifies retime is outside loops/chunk callbacks and follows
the sole shared `sample_long_range` call.

Operational Full and Quick perform final retime once after their **whole requested
Outpaint sequence** is merged. Quick still needs a16fps deliverable for its21-frame
FlashVSR input. C0/C1 native-only validation requests stop at the24fps master;
C1's113-frame smoke does not need any16fps file.

The common production helper now matches the read-only historical
`pipeline/config/01..05/<ID>_fps_4K.sh` timing/lossless encoding contract:

```text
-vf settb=expr=1/16,setpts=N
-fps_mode:v passthrough
-enc_time_base:v 1:16
-c:v libx264 -preset medium -crf 0 -pix_fmt yuv420p
-movflags +faststart
```

There is no `fps=16` filter and no output `-r`. Previous HEVC CRF1 was lossy and
cannot guarantee the newly explicit **exact decoded RGB sequence** requirement.
The historical lossless H264 option replaces that final encoding only; the existing
native24fps master encoder and all model/chunk computations remain unchanged.

Before atomic final-file commit, the helper requires:

- input master3840×2160,24/1 nominal and average FPS, expectedN decoded frames;
- output3840×2160,16/1 nominal and average FPS, the sameN decoded frames;
- every sequential decoded rgb24 frame hash matches at the same index.

It rejects pixel differences and keeps the master intact. Resume revalidates pixel
identity as well as metadata. The `.retime.json` receipt records both file roles,
SHA256, codec/timebase/filter and the complete ordered frame-hash sequence.

Independent CPU regression using the preserved baseline master (not a C0 gate):
`state/recurrence_lineart/final_retime_contract.json` is PASS,65/65 decoded RGB
exact, nominal/average16/1,65 frames,4.0625seconds. This executes the actual common
production helper, not a validation-only alternate encoder. It performs no CUDA
API/model work. The earlier isolated HEVC clock probe proved metadata only and is
not treated as pixel-identity evidence.

Full output pixel validation performs two complete sequential RGB decodes (native
master and final deliverable), in addition to metadata/frame-count probes. This
CPU/I/O cost and the larger lossless H264 deliverable are deliberate consequences
of the requested exact frame-sequence contract; no Full throughput benchmark is
claimed. The native master itself keeps the unchanged original encoder, so the
retime proof concerns its decoded sequence, not equality between an encoded
master and the pre-encoding uint8 `.npy` pixels.

## Final code/CPU acceptance — no further inference

The user explicitly revised the final gate from completing113f generation to
verifying current wiring. `state/recurrence_lineart/final_acceptance.json` records
the verdict and links the evidence. The owned smoke PID696408 received SIGINT and
exited; no broad kill, GPU reset, or action against another job was performed.
The isolated interrupted run and logs are preserved. The already completed
baseline and exact C0 candidate were neither regenerated nor overwritten.

| Required invariant | Evidence / source | Result |
|---|---|---|
| C0 has no incoming recurrence or bias | `inference.py:1143-1150,1173-1181,2216-2252`; existing exact-C0 latent/native/decoded proof | PASS |
| Every C1/C2/... uses the recurrence flags | `enhanced = chunk_index > 0`, no upper bound; `inference.py:2216-2252`; AST tests evaluate actual argument expressions at C0/C1/C2/C3/C40/C1000 | PASS |
| Hybrid carry, not pure last-three paste | `inference.py:898-930,1233-1245`; CPU stub VAE test checks first-RGB reencode, direct later slices, current anchor override, input tensor unmodified | PASS |
| Strength1.0 is an amplitude no-op | `inference.py:926-930`; released configs use1.0, separate from `VideoConditionByMask.strength` | PASS |
| Carry save/load/resume | `recurrence_artifacts.py:14-66`, `inference.py:2025-2030,2108-2123,2258-2263,2316-2324`; roundtrip, corruption, shape/dtype rejection CPU tests | PASS |
| Own C0 reference per video | `inference.py:2034-2062,2316-2335`, `lineart_reference.py:68-151`; input/adapter/seed/C0 pixel SHA binding and per-stage paths | PASS |
| Direct4K→30×17 density | `lineart_reference.py:39-65`; CPU128px-footprint boundary tests and existing05C0 reference artifacts | PASS |
| Global self-attention only, new queries only | Sole helper caller `global_attention.py:17-63`; eligibility/fine restore `lineart_global_prior.py:188-251`; CPU mock tests preserve prefix/anchor values | PASS |
| LongRange and chunk audit remain24fps | `outpaint.py:245-330,410-424,729-742`, `inference.py:1872-1880,2362-2372`; AST/config checks | PASS |
| Final retime occurs once, after complete merge | `outpaint.py:835-899,531-623`; AST ordering/absence from loops and chunk callback; existing65-frame exact CPU retime receipt | PASS |
| Flash accepts only final16fps deliverable | `run_cert.py:1074-1126`; stage receipt, output SHA, manifests and16fps checks; native-only rejection test | PASS |
| Full/stage/batch share one implementation | `run_cert.py:1015-1049,1129-1144`, `run_batch.py:232-255`; dispatcher/child-command mocks | PASS |

All references above are to files in the new workspace; bare `inference.py`
means `src/video_outpainting/main/src/ltx_outpaint/inference.py`, and `outpaint.py`
means `src/cert_pipeline/outpaint.py`. Final-review tests add no production branch.

### Actual Full batch configuration and FIFO order

Read directly from `configs/batch/certification.json:1-12` and its five referenced
job JSONs; all jobs request `outpaint=all`, `upscale=all`:

Subsequent user-requested ordering correction: **05 first, then04/03/02/01**.
Only the batch job list changed. The earlier `final_acceptance.json` preserves its
then-current01→05 order as historical evidence; the new configuration and CPU-only
mock regression result are recorded under `state/batch_order_0504030201/`.

| Order | Video | Base seed |
|---|---|---:|
| 1 | 05 | 71634910 |
| 2 | 04 | 71634910 |
| 3 | 03 | 33333333 |
| 4 | 02 | 25621683 |
| 5 | 01 | 50374151 |

```text
Outpainting actual order: 05 -> 04 -> 03 -> 02 -> 01
Upscale actual order (fresh successful run): 05 -> 04 -> 03 -> 02 -> 01
max simultaneous Outpaint workers: 1
max simultaneous Upscale workers: 1
Outpaint + Upscale concurrent: YES (up to one of each)
next Outpaint waits for previous Upscale: NO
```

`run_batch.py:154-188` validates exactly1/1 workers and FIFO and preserves JSON
order. The producer (`320-352`) waits for only its current Outpaint child, commits
OUTPAINT_DONE→UPSCALE_WAITING, queues that job, and advances to the next Outpaint.
The separate consumer (`354-383`) dequeues ready jobs FIFO and waits for its own
Upscale child. Both run concurrently (`388-394`) in separate model OS processes.
For example, Upscale05 may run while Outpaint04 and then Outpaint03 finish; those
completed jobs wait in FIFO rather than blocking the producer.

Outpaint failure excludes that video from the ready queue and continues with the
next job. Upscale failure also permits subsequent ready jobs to proceed. Therefore
the stated five-video Upscale order is the normal fresh, successful case, not a
claim that failed Outpaint jobs are upscaled.

Resume is not a blind skip: config byte identity is bound; interrupted/failed
Outpaint resets to pending, and prior completed/failed Upscale states are queued
for the common `--stage upscale --resume` validation/reuse path (`260-308`).
Existing DONE can therefore be revalidated through the stage; that does not imply
unconditional new Flash inference. The normal01–05 output areas and real batch
state were still empty at final review; CPU queue tests used temporary roots.

The operational command is:

```bash
./run_batch.sh --config configs/batch/certification.json
```

**It was not executed in this audit.** Batch uses validated canonical/generated
PreProc artifacts through `--stage outpaint`; it does not redo preprocessing for
every job. A fresh individual Full `run_cert.sh --config ...` still invokes the
shared preprocessing stage before Outpaint.

### Evidence boundary

- **CONFIRMED BY CODE / CPU TESTS:** all final wiring and queue invariants above.
- **CONFIRMED BY EXISTING GPU EVIDENCE:** video05 C0 exact preservation against
  the pre-edit run and historical native24fps MP4.
- **NOT RUNTIME-VERIFIED:** C1 numeric/visual output, long Full recurrence quality,
  Full queue throughput, and concurrent Outpaint+Flash performance in this task.
  These are disclosed limitations, not hidden PASS claims or a reason to launch
  the additional inference that the user prohibited.
- No remaining code blocker was found under the explicitly accepted scope.
