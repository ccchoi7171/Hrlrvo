# Certification Pipeline Execution Audit

Audit type:  
READ-ONLY static execution trace

Code modified:  
NO

GPU inference executed:  
NO

FFmpeg preprocessing executed:  
NO

Smoke test executed:  
NO

Static execution trace verdict:  
**READY WITH WARNINGS** — the call path and artifact contracts are present in code, but this audit does not establish runtime success, pixel identity, available disk/VRAM, or numerical equivalence.

Certification 03: `input metadata frames = 1952` (previous metadata observation, **not** a decoded count remeasured in this audit); JSON requests `outpaint = all`, `upscale = all`. The running program resolves both `all` values from its own full-source frame count, not from 1952 hard-coded in source.

Quick 03: JSON requests `outpaint = 65`, `upscale = 21`; final output is 21 frames. **Quick still counts and preprocesses the whole source.**

Same processing implementation: **YES** (the Quick 21f input-clip branch is a frame-limit consequence, not a different model algorithm). Full-source frame-count traversal before Quick inference: **YES**.

**Evidence convention.** **CONFIRMED BY CODE** means the cited current source specifies that behavior. **STATIC DERIVATION** means arithmetic applied to the cited planner with the explicitly stated 1952-frame example. **RUNTIME OBSERVATION (PRIOR)** is the earlier report that an 8K `-count_frames` check took over five minutes; no timing check was repeated for this document. **NOT RUNTIME-VERIFIED** means no inference, FFmpeg processing, GPU/model loading, smoke test, or output decode was performed during this audit. All `Lx-Ly` references below are relative to this package root unless an external reference path is stated. A successful-run terminal sequence is conditional, not evidence that a run succeeded.

## 1. Whole-pipeline call flow

```text
run_cert.sh [run_cert.sh:L1-L14]
  └─ exec <Outpaint Python> run_cert.py "$@"
       └─ main() [run_cert.py:L682-L795]
          ├─ validate_config() [L111-L277]
          ├─ preflight_source() [L280-L288]
          │    └─ preprocess.probe_media(): ffprobe -count_frames [preprocess.py:L134-L163]
          ├─ resolve_limit() ×2; prepare_run(); requested/resolved snapshots [L104-L108,L346-L367,L694-L728]
          ├─ seed_all() [L370-L387]
          ├─ preprocess.run_preprocess() [preprocess.py:L374-L512]
          │    ├─ probe_media(source), SHA source
          │    ├─ build_ffmpeg_command() [L243-L314] → whole-source FHD/24 encode
          │    └─ probe/validate/publish FHD output
          ├─ outpaint.run_outpaint() [outpaint.py:L581-L849]
          │    └─ ltx_outpaint.inference.sample_long_range() [inference.py:L1733-L2300]
          │         ├─ StreamingAnchorReader; _iter_long_range_chunks()
          │         ├─ _load_models(); sample_chunk() → native ValidationRunner._run_denoising()
          │         ├─ generated-latent observer; native segment/prefix commit
          │         └─ encode_segments() → 4K/24 MP4
          │    └─ _retime_to_16fps() → final 4K/16 MP4
          ├─ run_flash() [run_cert.py:L558-L679]
          │    ├─ upscale_input(): Full passes 4K MP4; Quick creates verified 21f clip [L456-L555]
          │    └─ flash_composite.main()/execute() [flash_composite.py:L1029-L1293,L1345-L1379]
          │         ├─ 4K/original-8K probe, timestamp validation, request identity
          │         ├─ packaged upscale.py main()/execute() [upscale.py:L986-L1439,L1486-L1514]
          │         │    ├─ spatial tiles; continuous Tiny Long causal stream per tile
          │         │    ├─ PNG + emission JSON + tile commit + stage manifest
          │         │    └─ execute_encode_stage()/encode_tiles() → raw 16K MP4
          │         ├─ encode_staged_tiles() + OriginalFrameReader.read(k)
          │         │    └─ composite_original_center_rows() → final video-only 16K MP4
          │         └─ original-8K audio mux if present → final MP4 + final_manifest.json
          ├─ verify_video(raw), verify_video(final); optional full decode [L640-L668]
          └─ [UPSCALE_DONE], [PIPELINE_DONE] [L669-L679,L786-L795]
```

The model computation is neither in `run_cert.py` nor an independent per-frame upscaler. The orchestrator calls packaged implementations. Flash `upscale.py` is launched as a child process, whereas Outpainting's `sample_long_range()` is imported and called in the runner process. The final center composite belongs to stage 0003; there is no stage 0004.

## 2. Entrypoint-to-final step contract

### Step 1 — Select Python and enter the single runner

- **Source:** `run_cert.sh:L1-L14`; **Function:** Bash entrypoint.
- **Input:** `--config configs/certification/03.json` or `--config configs/quick/03.json` (also optional `--resume` or `--force`).
- **Operation:** `set -Eeuo pipefail`; resolve script directory; disable Python bytecode and enable unbuffered output. If conda is on PATH, select `Hrlrvo/bin/python` for the runner and export `Hrlrvo-FlashVSR/bin/python` for the Flash child unless environment overrides exist. Otherwise use `CERT_OUTPAINT_PYTHON`/`python3` and later `CERT_FLASH_PYTHON`/current interpreter. No Full/Quick shell branch.
- **Output:** no video; process replacement by `run_cert.py`.
- **Next:** `main()`.

### Step 2 — Parse config and validate the package contract

- **Source:** `run_cert.py:L25-L29,L83-L108,L111-L277,L682-L694`; **Function:** `main()` → `validate_config(path)`.
- **Input:** JSON bytes, package root, source/model/config paths.
- **Operation:** parse JSON, check schema/video ID/fixed seed, resolve paths inside package root, validate frame-limit types and all fixed preprocess/outpaint/upscale/final settings, check required files and inspect YAML via `yaml.safe_load`. CLI cannot override model, seed, steps, prompt or frame count.
- **Output:** `(config, source_path, output_root, outpaint_yaml_path, flash_model_root)`.
- **Next:** `preflight_source()`.

### Step 3 — Block on authoritative source frame count

- **Source:** `run_cert.py:L280-L288,L694-L703`; `preprocess.py:L117-L194`; **Function:** `preflight_source()` → `probe_media()` → `_frame_count()`/`validate_input_probe()`.
- **Input:** the same 8K/16fps source for both configs.
- **Operation:** `ffprobe -count_frames` traverses the uncapped video stream, uses `nb_read_frames` before `nb_frames`, verifies one video stream, 7680×4320, average 16fps, positive count. `resolve_limit()` sets Full to N/N, Quick to 65/21, rejects requests beyond available frames and upscale below 21.
- **Output:** integer source N and resolved outpaint/upscale limits.
- **Next:** `prepare_run()`; no `[PREFLIGHT]` has yet appeared.

### Step 4 — Establish run identity and logging

- **Source:** `run_cert.py:L291-L367,L704-L741`; **Function:** `prepare_run()`, `atomic_copy()`, `atomic_json()`.
- **Input:** output root and `--resume`/`--force` state.
- **Operation:** existing non-`.gitkeep` files require `--resume` or are archived by `--force`; create stage layout; on resume compare requested JSON **bytes** with the previous snapshot; write `requested_config.json` and `resolved_config.json`; open run-level `pipeline.log` Tee. Then print `[PREFLIGHT]`.
- **Output:** run directories and two config snapshots; terminal/file log mirror.
- **Next:** `seed_all()` then stage 0001. `seed_all()` seeds Python/NumPy/PyTorch and checks CUDA availability **during a real run**, before FFmpeg preprocessing (`run_cert.py:L370-L387`). It was not called in this audit.

### Step 5 — Make the full-length FHD/24 processing input

- **Source:** `run_cert.py:L742-L752`; `preprocess.py:L374-L512`; **Function:** `run_preprocess()`.
- **Input:** original 8K MP4 and source N, **not** Quick's 65f.
- **Operation:** second full-source frame count, full-source SHA, one FFmpeg command for spatial/temporal/audio conversion, output full frame count/spec validation, atomic publication.
- **Output:** `0001_Preprocess_FHD24/output/03_Input_FHD_24fps.mp4` plus probe/alignment metadata.
- **Next:** `run_outpaint()`.

### Step 6 — Generate and persist LongRange 4K frames

- **Source:** `run_cert.py:L753-L774`; `outpaint.py:L581-L849`; **Function:** `run_outpaint()` → `sample_long_range()`.
- **Input:** full FHD/24 MP4; limit N or 65; seed, checkpoint, 30 steps, positive/negative prompt, 16px feather.
- **Operation:** resolve packaged YAML/weights, invoke native planner/model/causal VAE/denoising/prefix chain; persist final latent and committed per-chunk frames/prefix state; assemble 24fps MP4, then retime and encode exactly the requested frame count at 16fps.
- **Output:** `0002_Outpaint_4K16/final/03_Outpaint_4K_16fps.mp4` plus native/audit artifacts.
- **Next:** `run_flash()`.

### Step 7 — Select the Flash input range

- **Source:** `run_cert.py:L456-L555,L558-L575`; **Function:** `upscale_input()`.
- **Input:** final 4K/16 MP4, `upscale_frames`, `outpaint_frames`.
- **Operation:** if equal, return the 4K MP4 unchanged (Full). If unequal, FFmpeg `trim=end_frame=21,setpts=N/(16*TB)` creates a 21f lossless-HEVC video-only clip, verifies 4K/16/21 and compares decoded RGB `framemd5` against the first 21 outpaint frames (Quick). No timestamp seek.
- **Output:** Full: same 4K path; Quick: `0003.../segments/input/03_Outpaint_4K_16fps_21f.mp4` and `metadata/upscale_input.json`.
- **Next:** `flash_composite.main()` via `run_logged()`.

### Step 8 — Validate Flash inputs and run packaged Tiny Long

- **Source:** `run_cert.py:L576-L639`; `flash_composite.py:L1029-L1135`; `upscale.py:L986-L1439`.
- **Input:** selected 4K MP4, original 8K MP4, packaged model/runtime, expected N or 21, spatial/temporal options.
- **Operation:** Flash wrapper recounts 4K and whole original 8K, checks CFR timestamps and SHA request identity; launches packaged `upscale.py --stage-mode inference --benchmark-mode full`. Native runtime loads official FlashVSR v1.1 Tiny Long components, chooses spatial tiles from GPU free memory, runs one continuous causal time stream per tile, persists PNGs/emission records/tile commits, then a complete stage manifest.
- **Output:** `segments/flashvsr_stage_manifest.json` and reconstructable spatial-tile PNGs.
- **Next:** raw 16K assembly.

### Step 9 — Encode raw 16K and original-center final

- **Source:** `flash_composite.py:L1137-L1293`; `upscale.py:L828-L913`; `flash_composite.py:L758-L882`.
- **Input:** committed PNG tile set, first N or 21 original 8K frames, original audio if any.
- **Operation:** native `execute_encode_stage()` blends tile PNGs and encodes raw 16K; wrapper independently assembles bounded row bands, reads `OriginalFrameReader.read(k)`, feathers original 8K into fixed center, encodes a 16K video-only partial, optionally copies original audio with `-t frames/16`, atomically publishes final, commits `final_manifest.json`.
- **Output:** retained raw MP4 and `0003_Upscale_FlashVSR_16K/final/<config filename>`.
- **Next:** runner verification.

### Step 10 — Verify and report

- **Source:** `run_cert.py:L390-L402,L640-L679,L775-L795`; **Function:** `verify_video()` and `run_flash()` tail.
- **Input:** raw and final 16K paths, expected frame count.
- **Operation:** full-stream `ffprobe -count_frames` checks 15360×8640, average 16fps and expected count; write final/raw probe, Flash config and segment-plan metadata. Only if `verification.full_decode=true` does runner additionally call `ffmpeg -v error -i FINAL -f null -`; both 03 JSONs set it `false`. Then print `[UPSCALE_DONE]` and `[PIPELINE_DONE]`.
- **Output:** run returns exit code 0 on the success path. **NOT RUNTIME-VERIFIED.**

## 3. Certification/Quick 03 configuration comparison

Actual files: `configs/certification/03.json` and `configs/quick/03.json`. Validation and dispatch: `run_cert.py:L111-L277,L682-L785`.

| Field | Certification 03 | Quick 03 | Same/Different |
|---|---|---|---|
| `job_name` / output identifier | `03_certification` | `03_quick` | Different |
| `output_root` | `data/certification/video_03_seed33333333` | `data/quick/video_03_seed33333333` | Different |
| video / source | `03` / `data/00_Input_8K_7680x4320/03_Input_8K.mp4` | same | Same |
| seed | `33333333` | `33333333` | Same |
| outpaint / upscale frame limit | `all` / `all` | `65` / `21` | Different |
| source→processing | 7680×4320/16 → 1920×1080/24, Lanczos | same | Same |
| Outpainting checkpoint | `src/models/outpaint_checkpoint_007500` | same | Same |
| Outpainting runtime YAML / steps | `src/video_outpainting/main/configs/inference_4k.yaml` / 30 | same | Same |
| positive / negative prompt | `""` / `people, person, human, staff, audience, crowd, backstage, seats` | same | Same |
| Outpaint input/output, feather | 1920×1080 → 3840×2160, 16px | same | Same |
| Flash model/runtime | `src/models/flashvsr_v1_1` / `src/flashvsr/architect/FlashVSR_v1.1/upscale.py` | same | Same |
| Flash scale / input/output / FPS | x4, 3840×2160 → 15360×8640, 16 | same | Same |
| Flash options | auto tile/overlap, min 128, sparse 2.0, local 11, GPU 0, 32 FFmpeg threads, libx265 | same | Same |
| center composite / feather / final FPS | original 8K center, 64px, 16 | same | Same |
| final filename | `03_Certification_Final_16K_16fps.mp4` | `03_Quick_Final_16K_16fps_21f.mp4` | Different |
| final audio / full decode | `original_8k` / `false` | same | Same |

**Certification and Quick computation implementation: SAME.** The same `main()` invokes the same three stage functions and native model functions. **Literal branch sequence: DIFFERENT by frame-limit consequence**: `upscale_input()` returns the existing 4K video for Full and executes the verified first-21f clipping branch for Quick (`run_cert.py:L465-L555`). Model/LongRange/Flash algorithm settings are not mode-switched. The output path and name also differ by JSON.

## 4. Preflight in exact execution order

The word *preflight* here includes everything from shell entry to the stage-0001 call; it does **not** imply that all checks print a log. Except for any shell selection error, steps 1–5 are silent until the `[PREFLIGHT]` line.

| Order | Check / action | Source and command/function | Cost characteristic; result used by |
|---:|---|---|---|
| 1 | Parse only `--config`, `--resume`, `--force`; root-relative resolution | `run_cert.py:L682-L693` | Small; establishes path and mode flags. |
| 2 | JSON existence/parse, schema, video ID and fixed seed | `validate_config():L111-L125` | Small; rejects wrong/missing JSON before models. |
| 3 | Input package-relative path and file existence | `absolute_package_path():L83-L91`; `validate_config():L126-L133` | `stat`/path resolution; forbids escaping package root. |
| 4 | Frame limit syntax; exact preprocess/outpaint/Flash/final constants | `validate_config():L134-L246` | Small; permits `all` or positive integer only. |
| 5 | Outpaint YAML, adapter, LTX/Gemma paths and required Flash asset **existence**; parse YAML and compare key values | `validate_config():L177-L276` | Small path/YAML reads; does **not** read/hash full model weights or model manifest. |
| 6 | FFmpeg binaries, source resolution/FPS and frame count | `preflight_source():L280-L288` → `preprocess.probe_media():L134-L163` (`ffprobe -v error -count_frames -show_streams -show_format -of json <8K>`) → `validate_input_probe():L166-L194` | **Full original 8K video-stream traversal; blocking before stage 0001 and before `[PREFLIGHT]`.** N resolves both `all` values. |
| 7 | Limits within N and Flash minimum 21 real input frames | `resolve_limit():L104-L108`; `main():L696-L703` | Small; Full N/N, Quick 65/21 if N≥65. |
| 8 | Existing-output policy and directory layout | `prepare_run():L346-L367`; `make_layout():L320-L344` | Scans existing files; `--force` archives, no `--resume`/`--force` with artifacts errors. No free-space check. |
| 9 | Requested config byte identity on resume; copy and resolved snapshot | `main():L705-L723` | Small JSON/file operations; records N/limits. |
| 10 | Open run-level Tee, print `[PREFLIGHT]`, seed Python/NumPy/PyTorch | `main():L724-L741`; `seed_all():L370-L387` | First visible success log. Real execution calls `torch.cuda.is_available()` before stage 0001; **not performed in this audit**. |

No source packet-count path is used. No PyAV source decode occurs in this preflight: PyAV's `StreamingAnchorReader` and its full FHD `probe_video()` occur in stage 0002 (`streaming_video.py:L86-L106,L109-L184`). No model manifest SHA or output-disk-capacity gate exists in this startup path. The principal core files were observed to exist during read-only listing; that is not a content-integrity or load test.

## 5. Source Frame Counting and Startup Cost

**CONFIRMED BY CODE:** `preprocess.probe_media()` unconditionally passes `-count_frames`, with no `-read_intervals`, `-frames`, or quick limit (`preprocess.py:L134-L163`). `_frame_count()` prefers `nb_read_frames`, then `nb_frames` (`L117-L131`). Hence the whole original stream is traversed to obtain the startup N. `nb_frames=1952` is a previous metadata value, not a numeric constant in the runner and not a decoded count reverified here.

### Certification 03

1. Before stage 0001: one whole-8K `-count_frames` in `preflight_source()`; blocking before `[PREFLIGHT]` (`run_cert.py:L694-L695`).
2. At the start of stage 0001: a **second** whole-8K `-count_frames` and **first** full-file source SHA256 pass (`preprocess.py:L414-L421,L65-L79`).
3. Stage 0001 FFmpeg then decodes/resizes/encodes the **entire N**, with no frame cap (`preprocess.py:L243-L314,L468-L480`).
4. At stage 0003 entry: a **third** whole-8K `-count_frames`, a whole-8K `-show_frames` timestamp traversal, and a **second** full-file source SHA256 pass for the Flash request identity (`flash_composite.py:L1047-L1072,L214-L269,L376-L419`).
5. Final original-center insertion sequentially decodes the first N original frames; for Full N=all (`flash_composite.py:L517-L610,L786-L807`).

Thus, on a fresh successful path, the original 8K input alone is the subject of **three `-count_frames` calls, one separate full timestamp-frame traversal, two full-file SHA passes, one full preprocessing decode/encode, and one final center-reader decode of N frames**. This count excludes any FFmpeg internal reads for audio mux or other validation. No runtime duration is inferred.

### Quick 03

The same startup full-source count occurs **before** the 65/21 limits can be applied (`run_cert.py:L694-L703`). The second count, full SHA and entire-source FHD preprocessing likewise happen before Outpainting reads only its first 65 FHD frames (`preprocess.py:L419-L421,L468-L480`; `streaming_video.py:L86-L106,L109-L184`). Stage 0003 again counts and timestamp-traverses all original 8K frames, even though final `OriginalFrameReader` limits its sequential decode to 21 (`flash_composite.py:L1047-L1068,L517-L610`). Quick is a short **model** run, not a short source-preparation run.

**RUNTIME OBSERVATION (PRIOR, not repeated):** an 8K full `-count_frames` check was reported to exceed five minutes. The code explains why an uncapped scan can precede any visible `[PREFLIGHT]`, but this document makes no new runtime measurement. **NOT RUNTIME-VERIFIED:** actual decoded count and duration in a later run.

## 6. Stage 0001: exact preprocess command and outputs

**Source:** `src/cert_pipeline/preprocess.py:L27-L41,L232-L314,L374-L512`. Input is 7680×4320/16fps/N; output is 1920×1080/24fps/N on the intended success path. The command builder conditionally includes audio arguments only if source audio exists:

```text
ffmpeg -hide_banner -nostdin -loglevel info -xerror -err_detect explode
  -i <data/00_Input_8K_7680x4320/03_Input_8K.mp4>
  -map 0:v:0 [-map 0:a:0]
  -vf 'scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24'
  -fps_mode:v passthrough
  -c:v libx265 -preset fast -crf 1 -pix_fmt yuv420p10le
  -x265-params log-level=error:pools=8
  -color_range pc -colorspace bt709 -color_primaries bt709 -tag:v hvc1
  [-af atempo=1.5 -c:a aac -b:a 192k]
  -map_metadata 0 -video_track_timescale 24576
  -y <0001_Preprocess_FHD24/output/.03_Input_FHD_24fps.<pid>.partial.mp4>
```

`scale` is spatial Lanczos with full-range (`pc`) input/output and square sample aspect ratio. `setpts` changes playback timestamps; `fps=24` is a real FFmpeg frame-rate filter. Audio, if present, is sped 1.5× and AAC re-encoded **only for the processing intermediate**. The output command has no `-frames:v 65` or analogous cap. After FFmpeg finishes, output `ffprobe -count_frames` must report exact FHD/24/N, `yuv420p10le`, and expected audio presence. The temporary MP4 is atomically renamed to `output/03_Input_FHD_24fps.mp4` only after metadata validation (`preprocess.py:L197-L229,L475-L510`). The stage writes `metadata/input_ffprobe.json`, `output_ffprobe.json`, `frame_alignment.json` and `logs/preprocess.log`.

## 7. 16fps → 24fps temporal analysis and alignment claim

For ideal CFR input with zero-based frame `k` at PTS `k/16` seconds, the filter's `setpts=PTS*2/3` maps it to `k/24`. If FFmpeg's `fps=24` sampling grid starts on the matching phase, the expected frame sequence is the same N frames at 24fps; no motion interpolation algorithm is configured. This is **STATIC DERIVATION**, not a result observed on source pixels.

`fps=24` can nevertheless duplicate/drop when source timestamps or rounding do not occupy its expected grid, including at the end boundary. `-fps_mode:v passthrough` comes **after** that filter and cannot undo it. The runtime output count check rejects N±1; it does **not** distinguish a net-zero drop-plus-duplication or prove each `source[k]` maps to `FHD[k]`. `frame_alignment.json` writes `frame_order_preserved`, `frame_drop=false`, `frame_duplication=false` and a `source[k] -> output[k]` policy statement, but these fields are constructed from the expected N rather than independent frame hashes (`preprocess.py:L339-L371`). This is warning W2. The later Flash timestamp check validates CFR of each video independently, not visual cross-video identity (`flash_composite.py:L214-L269`). **NOT RUNTIME-VERIFIED:** actual filter output and off-by-one outcome for source 03.

## 8. Stage 0001 terminal sequence

The stage logger uses UTC prefix + `[PREPROCESS]` and mirrors child FFmpeg output to terminal and `preprocess.log`; run-level Tee also captures the terminal stream in `pipeline.log` (`preprocess.py:L82-L87,L317-L336`; `run_cert.py:L724-L728`). Actual source strings, with dynamic values replaced:

```text
<UTC> [PREPROCESS] status=START input=<8K path> output=<FHD path> expected_frames=<N> resume=False
  Source: preprocess.py:L414-L418. Meaning: stage entered; second count and SHA still pending.
<UTC> [PREPROCESS] command=ffmpeg <full command>
  Source: preprocess.py:L468-L475. Meaning: FFmpeg will be launched.
<FFmpeg native log/progress lines>
  Source: preprocess.py:L317-L336. Text depends on FFmpeg and is not fabricated here.
<UTC> [PREPROCESS] status=PASS frames=<N> resolution=1920x1080 fps=24 output=<FHD path>
  Source: preprocess.py:L507-L512. Meaning: metadata/count checks passed and output was published.
```

On an eligible resume the final line is instead `status=RESUME_PASS ...` (`preprocess.py:L423-L465`). For an error within the FFmpeg/verify `try`, `status=FAIL error=<...>` appears (`L475-L505`); earlier validation exceptions do not pass through that catch.

## 9. Stage 0002: packaged VideoOutpainting call graph

**CONFIRMED BY CODE:** run_cert.py:L753-L774 enters cert_pipeline.outpaint.run_outpaint(). This wrapper reads src/video_outpainting/main/configs/inference_4k.yaml, converts packaged LTX/base/Gemma/adapter paths to package-root absolute paths, sets JSON seed/steps/24fps/4K/16px, then calls ltx_outpaint.inference.sample_long_range() without implementing a temporal planner (outpaint.py:L557-L579,L617-L768). YAML supplies clip65/prefix17/stride48, CUDA bfloat16, CFG3/STG1. JSON supplies video-only GPU weights=true, streaming VAE decode=true, FFN token chunking=false (inference_4k.yaml:L65-L99; both 03 JSONs).

Actual order, including the placement of feather (**before** chunk artifact writing, not after final merge):

1. sample_long_range() installs packaged official LTX sources, reads FHD/24fps metadata with PyAV probe_video() (whole-FHD decode/count even in Quick), checks N/65 and actual VAE factors (inference.py:L1770-L1827; streaming_video.py:L86-L106).
2. It prepares SegmentedOutputStore identity and native chunk records; loads LTX base, Gemma text encoder, VAE and checkpoint adapter if chunks remain (inference.py:L1907-L2004,L604-L712).
3. StreamingAnchorReader.read_chunk(record) takes frames by sequential decoded index, retaining one overlap, and returns FP32 [F,3,1080,1920] in [0,1] (streaming_video.py:L109-L184).
4. Native _iter_long_range_chunks() supplies source/prefix/new-output ranges. A short last input is repeat-last padded to a 1+8k working count (inference.py:L220-L267,L2054-L2071).
5. sample_chunk() makes pixel/latent shapes, independently VAE-encodes anchor and prior decoded-RGB prefix, marks the known anchor/prefix, sets absolute time, seeds noise with base_seed+absolute_start_frame and calls official ValidationRunner._run_denoising() with CFG/STG and 30 steps (inference.py:L771-L914,L1038-L1157; validation_runner.py:L855-L982).
6. A final-latent observer saves the denoised latent **before** VAE decode. Decoded output is cropped from the VAE boundary, receives one 16px original-anchor feather, and previous prefix RGB is exactly restored (inference.py:L1176-L1229,L1698-L1730).
7. Native generated[-17:].clone() becomes the next FP32 RGB prefix. Chunk 0 emits all frames; later chunks emit only their non-prefix new range. Native segment/prefix/RNG/completion record are committed, after which the wrapper saves the carry copy and chunk MP4 (inference.py:L2124-L2205; streaming_video.py:L420-L539; outpaint.py:L346-L427).
8. encode_segments() reads committed .npy new-frame segments in range order and encodes 24fps MP4. The wrapper validates 4K/24/count, writes lineage/audit manifests, retimes to 16fps and writes metadata (inference.py:L2207-L2300; outpaint.py:L770-L849).

The packaged inference.py is not byte-identical to its reference: source_manifest.json marks it instrumented_only. A read-only diff shows that step/latent/commit observer points were added while the native denoising call, planner and prefix computation remain the same. streaming_video.py is recorded as copied_as_is. GPU numerical equivalence is **NOT RUNTIME-VERIFIED**.

## 10. LongRange planner: every 03 Full example chunk

**CONFIRMED BY CODE:** LongRangeSpec(clip_frames=65,prefix_frames=17,stride_frames=48,final_chunk_policy=repeat_last) and _iter_long_range_chunks() (inference.py:L57-L76,L220-L267; inference_4k.yaml:L65-L69). First chunk starts at 0. Later source starts advance by 48; 17 prefix frames are context already emitted, and at most 48 new frames are appended. Ranges are zero-based, end-exclusive. Carry is decoded RGB generated[-17:], **not latent KV**. The last padded carry is saved but not consumed. **STATIC DERIVATION** below assumes the prior metadata example N=1952; runtime resolves N again.

| Chunk index (0-based) | Input/global [start,end) | Existing prefix | Newly appended [start,end) | Carry for next | Completed frames |
|---:|---|---|---|---|---:|
| 0 | [0,65) | — | [0,65) | [48,65) | 65 |
| 1 | [48,113) | [48,65) | [65,113) | [96,113) | 113 |
| 2 | [96,161) | [96,113) | [113,161) | [144,161) | 161 |
| 3 | [144,209) | [144,161) | [161,209) | [192,209) | 209 |
| 4 | [192,257) | [192,209) | [209,257) | [240,257) | 257 |
| 5 | [240,305) | [240,257) | [257,305) | [288,305) | 305 |
| 6 | [288,353) | [288,305) | [305,353) | [336,353) | 353 |
| 7 | [336,401) | [336,353) | [353,401) | [384,401) | 401 |
| 8 | [384,449) | [384,401) | [401,449) | [432,449) | 449 |
| 9 | [432,497) | [432,449) | [449,497) | [480,497) | 497 |
| 10 | [480,545) | [480,497) | [497,545) | [528,545) | 545 |
| 11 | [528,593) | [528,545) | [545,593) | [576,593) | 593 |
| 12 | [576,641) | [576,593) | [593,641) | [624,641) | 641 |
| 13 | [624,689) | [624,641) | [641,689) | [672,689) | 689 |
| 14 | [672,737) | [672,689) | [689,737) | [720,737) | 737 |
| 15 | [720,785) | [720,737) | [737,785) | [768,785) | 785 |
| 16 | [768,833) | [768,785) | [785,833) | [816,833) | 833 |
| 17 | [816,881) | [816,833) | [833,881) | [864,881) | 881 |
| 18 | [864,929) | [864,881) | [881,929) | [912,929) | 929 |
| 19 | [912,977) | [912,929) | [929,977) | [960,977) | 977 |
| 20 | [960,1025) | [960,977) | [977,1025) | [1008,1025) | 1025 |
| 21 | [1008,1073) | [1008,1025) | [1025,1073) | [1056,1073) | 1073 |
| 22 | [1056,1121) | [1056,1073) | [1073,1121) | [1104,1121) | 1121 |
| 23 | [1104,1169) | [1104,1121) | [1121,1169) | [1152,1169) | 1169 |
| 24 | [1152,1217) | [1152,1169) | [1169,1217) | [1200,1217) | 1217 |
| 25 | [1200,1265) | [1200,1217) | [1217,1265) | [1248,1265) | 1265 |
| 26 | [1248,1313) | [1248,1265) | [1265,1313) | [1296,1313) | 1313 |
| 27 | [1296,1361) | [1296,1313) | [1313,1361) | [1344,1361) | 1361 |
| 28 | [1344,1409) | [1344,1361) | [1361,1409) | [1392,1409) | 1409 |
| 29 | [1392,1457) | [1392,1409) | [1409,1457) | [1440,1457) | 1457 |
| 30 | [1440,1505) | [1440,1457) | [1457,1505) | [1488,1505) | 1505 |
| 31 | [1488,1553) | [1488,1505) | [1505,1553) | [1536,1553) | 1553 |
| 32 | [1536,1601) | [1536,1553) | [1553,1601) | [1584,1601) | 1601 |
| 33 | [1584,1649) | [1584,1601) | [1601,1649) | [1632,1649) | 1649 |
| 34 | [1632,1697) | [1632,1649) | [1649,1697) | [1680,1697) | 1697 |
| 35 | [1680,1745) | [1680,1697) | [1697,1745) | [1728,1745) | 1745 |
| 36 | [1728,1793) | [1728,1745) | [1745,1793) | [1776,1793) | 1793 |
| 37 | [1776,1841) | [1776,1793) | [1793,1841) | [1824,1841) | 1841 |
| 38 | [1824,1889) | [1824,1841) | [1841,1889) | [1872,1889) | 1889 |
| 39 | [1872,1937) | [1872,1889) | [1889,1937) | [1920,1937) | 1937 |
| 40 | [1920,1952) | [1920,1937) | [1937,1952) | 17f saved, unused; padded tail | 1952 |

The last record has real_input_frame_count=32, working_frame_count=33, repeat_last_padding_count=1 and output_appended_count=15. The repeated working frame is not emitted in the 1952-frame result. Total: **41 chunks, 1952 appended frames**. sample_long_range() uses the actual VAE temporal factor, scans native resume records and verifies emission continuity (inference.py:L1927-L1944,L2141-L2156,L2207-L2219).

**Quick 65f**, same planner and no quick algorithm:

| Chunk index | Input/global | Existing prefix | Newly appended | Saved carry | Completed |
|---:|---|---|---|---|---:|
| 0 | [0,65) | — | [0,65) | [48,65), saved but unused | 65 |

## 11. Causal VAE temporal mapping and conditioning

The packaged LTX video VAE documents F'=1+(F-1)/8, spatial compression /32 and valid F=1+8k (src/video_outpainting/models/LTX2.3/packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py:L156-L170,L266-L275). Official temporal interval mapping computes slice(begin//8,(end-1)//8+1) (video_vae.py:L611-L636). The keyframe code describes temporal cell 0 as the **causal first pixel frame alone** and subsequent cells as eight pixel-frame positions (keyframes.py:L156-L171). Nominal local 65f indexing:

| Latent time index | Nominal local RGB frame span | Meaning |
|---:|---|---|
| 0 | frame 0 | Causal first-frame cell. |
| 1 | frames 1–8 | Subsequent eight-frame cell. |
| 2 | frames 9–16 | A 17f prefix occupies indices 0–2. |
| 3…8 | frames 17–64 in eight-frame cells | 65f → nine latent-time positions. |

These are **indexing cells**, not a statement that causal latent contents have no historical receptive field. With temporal8/spatial32 and VAE-padded canvas 3840×2176, a 65f chunk has nominal latent shape [1,128,9,68,120], a 17f prefix maps to three time cells, and 33 working frames map to five. Actual factors are read from the model at runtime and validated; these numeric shapes are **STATIC DERIVATION, NOT RUNTIME-VERIFIED** (inference.py:L492-L517,L1077-L1108,L1807-L1827; ltx_core/types.py:L107-L123).

previous_prefix is **17 decoded FP32 4K RGB frames**, not three carried latent slices. _condition_latents() reflect-pads and **re-encodes** these RGB frames, inserts their prefix latent into condition[:,:,0:3], and _known_mask() marks those three latent-time positions clean across the entire canvas. Anchor VAE encoding fills the observed center across time; prefix conditioning overrides full-canvas first three indices. VideoLatentTools(causal_fix=True) builds the state, and absolute source time is added before native denoising (inference.py:L771-L914,L1101-L1129). No pipeline wrapper re-implements causal packing.

## 12. Actual Outpainting carry, RNG and non-carry state

| State | Produced / source variable | Shape and dtype | Saved / consumed |
|---|---|---|---|
| **Actual next-chunk carry** decoded RGB prefix | next_prefix = generated[-17:].clone(), after decode/feather/old-prefix restoration (inference.py:L1222-L1229,L2127-L2140) | [17,3,2160,3840] FP32; native resume validator enforces this | Native alternating previous_prefix_0.pt / _1.pt; audit copy chunks/latents/chunk_XXXXXX/carry_prefix_state.pt. Next sample_chunk(previous_prefix=...) or load_resume_state() (streaming_video.py:L415-L418,L496-L539,L621-L674). |
| CPU/CUDA RNG resume state | _capture_post_chunk_rng() (inference.py:L2124-L2126) | CPU uint8 RNG tensors; CUDA optional | Same prefix-state .pt and audit copy. Restored on resume and after observers; not a separate neural cache (inference.py:L1991-L2004,L2158-L2184). |
| **Not carry:** final denoised latent | state.latent after 30 steps and before decode | Actual shape/dtype recorded; detach+CPU copy, no forced dtype cast | chunks/latents/chunk_XXXXXX/generated_latent.pt and metadata. Not an input to next chunk (inference.py:L1176-L1180,L1698-L1730). |
| **Not carry:** condition/mask/noise | condition_latent, mask, GaussianNoiser in sample_chunk() | Per-chunk runtime tensors | Created/consumed within one chunk, not persisted as next-chunk state (inference.py:L1089-L1129). |
| Cross-chunk Outpainting KV/cache | No such argument in this execution path | Not applicable | None claimed. FlashVSR has separate per-tile KV caches. |

Base seed is fixed in JSON, but native chunk noise uses base_seed+absolute_start_frame (inference.py:L1127-L1129). seed.json records this policy (outpaint.py:L843-L849).

## 13. Thirty-step denoising and real progress

_make_validation_config() loads JSON positive/negative strings and YAML CFG/STG/rescale values into ValidationConfig (inference.py:L547-L579). sample_chunk() prepares clean/noisy video states and cached positive/negative embeddings, then invokes runner._run_denoising(...,sampling_ctx=_SamplingProgress(...)) (L1101-L1157). The normal call delegates to official ValidationRunner._run_denoising(), not a wrapper scheduler (inference.py:L394-L407).

Each of 30 iterations uses LTX2Scheduler.execute(steps=30), EulerDiffusionStep, positive model call, a conditional negative CFG call, a conditional STG perturbation call, MultiModalGuider, reapplication of clean conditioning mask, and Euler latent update (src/video_outpainting/models/LTX2.3/packages/ltx-trainer/src/ltx_trainer/validation_runner.py:L855-L982). sampling_ctx.advance_step() follows the update at line 980. Packaged _SamplingProgress.advance_step() prints JSON with event=inference_denoise_step, zero-based chunk index, total chunk count, one-based step, total_steps=30, elapsed and ETA (inference.py:L310-L336). There is **no tqdm progress bar** or literal text “chunk 08/24 | denoise 17/30”. Only the final denoised latent is saved per chunk, not 30 step latents.

## 14. First-chunk artifact life cycle and resume use

The first native planner record is chunk_index=0, source/new-output [0,65). Precise file lifetime:

1. SegmentedOutputStore.prepare() creates chunks/native/run.json session identity; record_model_ready() writes the model report (streaming_video.py:L372-L407).
2. StreamingAnchorReader yields frames 0–64. The **pre-decode** observer writes a temporary generated_latent.pt, computes SHA256, atomically renames it, then writes chunks/latents/chunk_000000/metadata.json with shape/dtype/seed/checkpoint (inference.py:L2076-L2115,L1698-L1730).
3. After decode and feather, native commit_chunk() writes chunks/native/segment_000000.npy as uint8 RGB; previous_prefix_0.pt with decoded FP32 prefix/RNG; finally chunk_000000.json as completion marker. Native resume validates these including segment-pixel and prefix-state SHA (streaming_video.py:L420-L447,L496-L591,L621-L674).
4. **After native commit**, wrapper _persist_committed_chunk() validates latent SHA, copies native prefix state with SHA to chunks/latents/chunk_000000/carry_prefix_state.pt, reads native segment by mmap, atomically encodes chunks/mp4/chunk_0001_frames_000000_000064.mp4 at 16fps HEVC CRF1, and writes certification_artifacts.json (outpaint.py:L241-L427). This audit copy protects old alternating prefix slots; it is not a new model input.
5. The wrapper's [OUTPAINT_CHUNK_DONE] is printed inside the commit observer **after** MP4/carry work, before the native inference_chunk_complete JSON (inference.py:L2168-L2205; outpaint.py:L725-L750).
6. At end, wrapper builds chunk_plan.json and chunk_results.jsonl **from native committed records**, not guessed ranges. Native chunk_lineage.json and inference_manifest.json record contiguous output lineage (outpaint.py:L440-L491; inference.py:L2223-L2284).

Resume is not MP4-existence-only. Native run.json identity must match, chunk records must form a contiguous prefix, and segment size/shape/dtype/pixel SHA plus latest prefix/RNG state must pass. Wrapper reconciles committed chunks before model load and requires the generated latent; it can repair a missing audit carry copy/MP4 from available native state/segment. A missing older overwritten prefix or missing pre-commit latent may be fatal rather than silently accepted (outpaint.py:L669-L699; streaming_video.py:L541-L674). Native generation resume is anchored in segment/record/prefix state, not chunk MP4.

## 15. Expected artifact tree for 03 (file roles)

Directories are precreated; these are files a successful run **would** create, not files produced during this audit. N is runtime-resolved; dynamic attempt IDs and repeated chunks/tiles are abbreviated.

~~~text
data/certification/video_03_seed33333333/
├── config/
│   ├── requested_config.json                # byte-identical selected JSON
│   └── resolved_config.json                 # source/outpaint/upscale/final integer counts
├── logs/pipeline.log                        # terminal mirror after run Tee opens
├── 0001_Preprocess_FHD24/
│   ├── output/03_Input_FHD_24fps.mp4         # full-source FHD/24 processing video
│   ├── metadata/input_ffprobe.json          # full-stream source probe
│   ├── metadata/output_ffprobe.json         # full-stream FHD output probe
│   ├── metadata/frame_alignment.json        # declared mapping and source identity
│   └── logs/preprocess.log
├── 0002_Outpaint_4K16/
│   ├── final/03_Outpaint_4K_16fps.mp4       # requested N or 65 output frames
│   ├── chunks/
│   │   ├── native/
│   │   │   ├── run.json                     # native session identity
│   │   │   ├── model_ready.json             # committed model/adapter report
│   │   │   ├── segment_000000.npy           # new-output uint8 RGB; native resume
│   │   │   ├── previous_prefix_0.pt         # alternating FP32 prefix + RNG
│   │   │   ├── previous_prefix_1.pt         # other alternating slot
│   │   │   ├── chunk_000000.json            # native completion marker
│   │   │   ├── ...                          # subsequent native records/segments
│   │   │   └── 03_Outpaint_4K_24fps.mp4     # native sequential assembly
│   │   ├── latents/chunk_000000/
│   │   │   ├── generated_latent.pt           # pre-decode final latent; audit
│   │   │   ├── metadata.json                 # latent SHA/shape/dtype/seed/range
│   │   │   ├── carry_prefix_state.pt         # verified native prefix/RNG copy
│   │   │   └── certification_artifacts.json
│   │   ├── mp4/chunk_0001_frames_000000_000064.mp4
│   │   └── manifests/
│   │       ├── chunk_plan.json              # exact committed planner records
│   │       ├── chunk_results.jsonl           # native + audit artifact per chunk
│   │       ├── chunk_lineage.json            # native contiguous ranges
│   │       └── inference_manifest.json       # native inference provenance
│   ├── metadata/
│   │   ├── ffprobe_processing_24fps.json
│   │   ├── ffprobe_final.json
│   │   ├── inference_config.json
│   │   ├── checkpoint_info.json
│   │   └── seed.json
│   └── logs/outpaint.log
└── 0003_Upscale_FlashVSR_16K/
    ├── segments/
    │   ├── flashvsr_stage_request.json       # input/options request identity
    │   ├── flashvsr_stage_manifest.json      # complete tile/emission handoff
    │   ├── flashvsr_stage_report.json        # staged/raw report
    │   ├── input/                            # Quick: 03_Outpaint_4K_16fps_21f.mp4
    │   ├── metadata/attempt_001_<time>/
    │   │   └── tile_0000_emission_0000_frames_000000000_000000020.json
    │   └── tiles/certification_stage/
    │       ├── certification_identity.json
    │       ├── tiles/tile_0000/
    │       │   ├── 000000000.png             # one x4 spatial tile crop, frame 0
    │       │   ├── 000000001.png
    │       │   └── ...
    │       ├── tile_commits/tile_0000.json   # PNG/metadata SHA list
    │       └── ...                           # more tiles; transient conditioning_*.rgb
    ├── raw_flashvsr/03_FlashVSR_raw_16K_16fps.mp4
    ├── final/03_Certification_Final_16K_16fps.mp4
    ├── metadata/
    │   ├── final_manifest.json              # PASS request/output provenance
    │   ├── ffprobe_raw.json
    │   ├── ffprobe_final.json
    │   ├── flashvsr_config.json
    │   └── segment_plan.json                 # causal-emission references
    └── logs/flashvsr.log
~~~

Quick uses data/quick/video_03_seed33333333/ with the same shape. Outpaint final has 65f; segments/input/03_Outpaint_4K_16fps_21f.mp4 and metadata/upscale_input.json appear; raw/final have 21f; final filename is 03_Quick_Final_16K_16fps_21f.mp4 (run_cert.py:L456-L679). The tree is derived from actual path constructors, **not a claim that results already exist** (run_cert.py:L320-L344; outpaint.py:L637-L655; flash_composite.py:L317-L324; upscale.py:L1131-L1433).

## 16. Outpaint merge, 16px feather and 24→16 restore

Prefix is context, **not newly appended output**. Each record's new_output_frame_range must begin at the prior emitted end. Native stores generated[append_offset:append_offset+append_count], with append_offset=0 for chunk 0 and 17 afterward. encode_segments() iterates those .npy emitted segments in order, so prefix frames are not duplicated in final timeline (inference.py:L2141-L2156,L2207-L2235; streaming_video.py:L700-L748). Wrapper independently checks contiguous ranges before audit manifests (outpaint.py:L440-L491).

The **16px feather is applied per chunk in sample_chunk()**. _inward_anchor_boundary_alpha() computes the minimum distance to active anchor edges; alpha=smoothstep(clamp(d/16,0,1))=t²(3−2t). _composite_anchor_rgb() uses torch.lerp(decoded,source,alpha) in the narrow band, then copies source RGB exactly in the interior. At the edge alpha0 retains generated pixels; ≥16px inward alpha1 gives source. Corners use one nearest-edge field (inference.py:L934-L1034,L1210-L1229). Subsequent chunks restore their old-prefix output exactly.

Native merged MP4 is 3840×2160/24fps with requested count. Wrapper _retime_to_16fps() applies -vf 'settb=expr=1/16,setpts=N' -fps_mode:v passthrough -frames:v <count>, **without fps=16 filter**. Ordered input frame k gets output PTS k/16 on the intended path; libx265/CRF1/yuv420p10le re-encodes, then ffprobe checks 3840×2160/16/count (outpaint.py:L31-L34,L494-L554,L801-L849). This is frame-count/order/timestamp preserving but **not pixel bit-exact** across HEVC re-encode. It is distinct from the stage-0001 fps=24 resampler.

## 17. Stage 0002 actual terminal order

Logger strings below are current code strings, not invented events. Native/library stdout and FFmpeg logs may appear between them. Full <chunks>=41 only if runtime N=1952 with the stated VAE factors; otherwise native planner supplies its count.

~~~text
<UTC> [OUTPAINT] video=03 seed=33333333 frames=<N> steps=30 status=START
  Source: run_cert.py:L755-L764
<model loader/library output>
{"event":"video_only_gpu_weights",...}
  Source: inference.py:L700-L705; configured optimization branch
{"event":"inference_model_ready",...,"inference_steps":30,...}
  Source: inference.py:L1950-L1985
{"event":"inference_denoise_step","chunk_index":0,"chunk_count":<chunks>,"step":1,"total_steps":30,...}
...
{"event":"inference_denoise_step","chunk_index":0,"chunk_count":<chunks>,"step":30,"total_steps":30,...}
  Source: inference.py:L310-L336; validation_runner.py:L897-L980
[OUTPAINT_CHUNK_FFMPEG] command=ffmpeg <chunk MP4 command>
<FFmpeg native output>
[OUTPAINT_CHUNK_DONE] chunk=1/<chunks> completed_frames=65/<N> mp4=<path> latent=<path> carry=<path> elapsed=<s>s eta_total=<s>s
{"event":"inference_chunk_complete","chunk_index":0,...}
  Source: outpaint.py:L291-L330,L725-L750; inference.py:L2188-L2205
<repeat denoise/commit block for each remaining native chunk>
{"event":"inference_complete","chunk_count":<chunks>,"gap_count":0,"duplicate_count":0,...}
  Source: inference.py:L2285-L2297
[OUTPAINT_FFMPEG] command=ffmpeg <24→16 retime command>
<FFmpeg native output>
<UTC> [OUTPAINT_DONE] video=03 frames=<N>/<N> status=PASS output=<4K/16 path>
  Source: outpaint.py:L220-L238,L494-L554; run_cert.py:L768-L774
~~~

There is **no** separate literal [OUTPAINT_MODEL_LOAD], [CHUNK_PLAN], [OUTPAINT_VERIFY] or tqdm bar. Quick substitutes frames=65, native chunk_count=1 and exactly one 30-step block, ending completed_frames=65/65. On complete-chunk resume inference_model_ready may report model_load_skipped=true, but native segment assembly still runs (inference.py:L1950-L1969,L2207-L2219).

## 18. Stage 0003: packaged FlashVSR call graph and reference boundary

**CONFIRMED BY CODE:** run_cert.run_flash() first calls upscale_input(), constructs arguments and launches cert_pipeline/flash_composite.py under CERT_FLASH_PYTHON or the runner interpreter (run_cert.py:L456-L639). Full passes the 4K/16 result unchanged. Quick executes trim=end_frame=21,setpts=N/(16*TB) with lossless HEVC, verifies 4K/16/21, and compares decoded RGB frame MD5 against source frames 0–20 before giving the clip to Flash (run_cert.py:L465-L555). That extra clip is a limit consequence, not a different Flash model.

Then:

~~~text
flash_composite.main() → resolve_paths() → execute() [flash_composite.py:L1029-L1379]
  → required_assets(); read_outpainting_feather_width()
  → probe_video(4K); probe_video(original8K); validate_frame_index_alignment()
  → stage_request_payload() [includes SHA of 4K, original8K, runtime and YAML]
  → load_reference_module(packaged upscale.py)
  → build_stage_command(): packaged upscale.py --stage-mode inference --benchmark-mode full
      → upscale.main() → execute() [upscale.py:L986-L1439,L1486-L1514]
         → ffprobe(4K), padded_model_frames(), choose_auto_tile()
         → initialize_pipeline(): official FlashVSRTinyLongPipeline + weights
         → make_tiles() in y-major/x-minor order
         → for each spatial tile:
            build_tile_store(): 4K sequential FFmpeg decode/crop, x4 bicubic conditioning memmap
            stream_official_tile() → _stream_official_tile_impl()
              → continuous DiT/LQ/TCDecoder causal emissions, keep caches across emissions
              → save_tile_frame() PNG per output frame
              → record_emission() JSON after each causal emission
            _commit_tile(): SHA-committed tile journal
         → write complete stage manifest and STAGED report
  → validate_stage_manifest()
  → packaged upscale.execute_encode_stage() → encode_tiles() → raw 16K MP4
  → encode_staged_tiles() [flash_composite.py:L758-L882]
     → OriginalFrameReader.read(k), per-band tile merge
     → composite_original_center_rows() with original 8K and 64px alpha
     → RawVideoEncoder: final video-only 16K partial MP4
  → optional original-8K audio mux; atomic final publication; final_manifest.json
  → parent run_flash.verify_video(raw/final), write stage metadata, optional full decode
~~~

Native model loading: upscale.py:L318-L354 loads the official pretrained streaming DMD safetensors through ModelManager, Causal_LQ4x_Proj from LQ_proj_in.ckpt, TCDecoder.ckpt, positive prompt tensor, bfloat16 and VRAM management, then reports [MODEL_LOAD]. The x4 path is a bicubic-conditioned **neural causal video** path, not a frame-wise generic scaler. Packaged upstream location is src/models/flashvsr_v1_1/repository; model weights are under src/models/flashvsr_v1_1/checkpoint (upscale.py:L35-L53,L116-L128). The original reference path appears in source_manifest.json as provenance only, not as a runtime import path.

## 19. FlashVSR's real processing unit

A **spatial tile** is the independently resumable unit. Each tile's entire selected time sequence is one uninterrupted Tiny Long causal stream. Its temporal outputs are **emissions**, not separately inferred segments with overlap or cross-segment merge. In _stream_official_tile_impl(), first emission uses seven LQ conditioning slices, latent_t=6 and emits 21 real frames; subsequent emissions use two slices, latent_t=2 and normally emit eight frames. pre_cache_k/pre_cache_v, LQ_proj_in streaming cache and TCDecoder memory persist across emissions within one tile; they reset for a new tile. Per-tile generator seed is JSON base seed + tile index. These facts are in upscale.py:L390-L516,L1128-L1272. Config local_range=11 is forwarded into model_fn_wan_video(); this is not an independent segment-overlap length. Runtime reports temporal_mode=official_continuous_causal_kv_no_independent_chunks and chunk_boundaries=[] (upscale.py:L1083-L1100).

Spatial tile geometry: make_tiles() calls tile_starts() on x and y and iterates y outer, x inner (row-major). Auto overlap is 32 **source-space pixels**; source stride is tile_size−32 except the last tile is end-anchored and can have a larger effective overlap. Output tile dimensions are source dimensions ×4. Tile size auto is 320 source pixels if GPU free memory ≥100,000 MB, 256 if ≥60,000 MB, else 192; OOM decreases in 32px steps down to the configured minimum 128. Tile count is therefore **NOT statically fixed** (upscale.py:L141-L169,L916-L926,L1024-L1031,L1274-L1300).

| Selected source tile | Output PNG for a full interior tile | Spatial tiles on 3840×2160 with 32px overlap | Quick PNGs, tiles×21 | Full example PNGs, tiles×1952 |
|---:|---:|---:|---:|---:|
| 320×320 | 1280×1280 | 112 | 2,352 | 218,624 |
| 256×256 | 1024×1024 | 170 | 3,570 | 331,840 |
| 192×192 | 768×768 | 336 | 7,056 | 655,872 |
| 128×128 (minimum OOM size) | 512×512 | 920 | 19,320 | 1,795,840 |

Counts are **STATIC DERIVATION** from tile_starts()/make_tiles(), not a prediction of which GPU branch will run. No temporal tile/segment overlap parameter exists in this execution path. Spatial overlap is blended at raw/final assembly.

## 20. Full 1952f and Quick 21f temporal plans

padded_model_frames(real_frames) returns the smallest 8n+1 model length of at least 25 that can emit the real count: max(25,ceil_multiple(real_frames+3,8)+1) (upscale.py:L135-L138). _stream_official_tile_impl() computes process_total=(padded−1)/8−2 and truncates callbacks when written reaches real_frames (L412-L516).

| Example per spatial tile | Model input after last-frame padding | Causal emissions | Real output |
|---|---:|---|---:|
| Full input 1952 | 1961 | 243: first [0,21), then [21,29), [29,37), …, [1941,1949), final [1949,1952) | 1952 |
| Quick input 21 | 25 | 1: [0,21) | 21 |

The final Full emission emits only three real frames; the model may internally produce further padded-time frames, but callback stops at 1952. There is no temporal restart at emission boundaries. Total temporal-emission metadata records on the Full example are 243×spatial_tiles; Quick has exactly one per tile. **NOT RUNTIME-VERIFIED:** chosen tile size, OOM retries, model result and actual N.

## 21. Flash intermediate artifact and exact persistence point

build_tile_store() creates a temporary uint8 conditioning memmap at segments/tiles/certification_stage/conditioning_XXXX.rgb with shape [padded_frames,pad_h,pad_w,3]. It runs FFmpeg on the **selected 4K input once per spatial tile**, selects sequential frame indices, crops, x4 bicubic-resizes, pads dimensions to 128 and repeats last frame for model padding (upscale.py:L258-L315). That file is removed after the tile commits successfully (L1260-L1272). It is not a retained certification artifact.

For each emitted output frame, stage_output() calls save_tile_frame() immediately. It crops model padding, writes one PNG at segments/tiles/certification_stage/tiles/tile_XXXX/000000000.png, 000000001.png, …; each PNG is **a spatial crop at x4 size**, not a complete 15360×8640 frame (upscale.py:L519-L525,L1168-L1188). Once the native causal emission finishes, record_emission() writes a JSON under segments/metadata/attempt_###_<ns>/tile_XXXX_emission_####_frames_<start>_<last>.json containing tile, seed, exact range, PNG list, timing and COMPLETE state. It then logs [UPSCALE_SEGMENT_DONE] (upscale.py:L1190-L1242). Despite the log token “SEGMENT”, this is one emission of a continuous stream.

Only after **all real frames for that tile** are present does _commit_tile() hash every PNG and emission JSON and atomically write tile_commits/tile_XXXX.json. Stage manifest is written after all tiles finish (upscale.py:L786-L807,L1397-L1433). No per-emission MP4 is produced. Reconstruction uses the retained PNG set, tile geometry and emission metadata; total PNG formula is **tile_count × selected_frames**.

## 22. Disk, RAM and I/O structural cost — no benchmark claim

**Original 8K source, fresh successful run:** three full-stream -count_frames calls, one separate -show_frames timestamp traversal, two full-file SHA passes, full-source stage-0001 decode/reencode; final center reader decodes N frames for Full or 21 for Quick (§5). Additional original audio input read occurs if muxing. These counts follow run_cert.py:L694-L752, preprocess.py:L65-L79,L419-L480 and flash_composite.py:L1047-L1072,L376-L419,L1200-L1216. No runtime duration is inferred.

**Selected 4K Flash input:** native build_tile_store() opens FFmpeg independently **per tile** (upscale.py:L258-L315,L1138-L1165), so 112/170/336 initial tile choices imply at least that many selected-4K decode passes. For Full each pass requests N frames; for Quick each pass requests 21. Spatially cropped output does not mean FFmpeg can skip all earlier compressed frames. Stage wrapper/native input metadata counts and Quick clip frame-MD5 checks add traversals.

**Conditioning scratch:** one tile at a time is held as uint8 memmap. For Full example padded_frames=1961, a 320px source tile has 1280×1280×3×1961 ≈ **9.639 GB decimal** on disk while that tile runs; 256px ≈6.169 GB, 192px ≈3.470 GB, 128px ≈1.542 GB. These are formula-derived **temporary** file sizes, not observed free space. PNGs and manifests are retained after success; the exact compressed PNG bytes cannot be inferred from code.

**Raw 16K merge:** upscale.encode_tiles() allocates for each frame a full 15360×8640×3 float32 accumulator (≈1.593 GB decimal) and 15360×8640 float32 normalization plane (≈0.531 GB), or **≈2.123 GB decimal / 1.978 GiB simultaneously before tile images/weights/encoded frame and FFmpeg buffers** (upscale.py:L565-L653). The arrays are per-frame, not held for the entire video. The raw HEVC stream is then encoded once.

**Final center encode:** flash_composite.encode_staged_tiles() instead uses bounded vertical-band float32 accumulators and normalization, one original RGB8 frame (7680×4320×3 ≈99.5 MB decimal), plus active tile PNGs/cache and encoder buffers (flash_composite.py:L758-L882). It does **not** reuse the raw 16K MP4 pixels; it reassembles from PNGs and encodes the final video again. Exact peak memory is **NOT RUNTIME-VERIFIED**.

**16K counting/probes:** the fresh audio-present path counts raw once in the Flash wrapper, final video-only partial once, audio-mux partial once, then parent counts published raw and final once each: **five full-stream 16K -count_frames traversals** in that branch. Without audio it is four. The code also performs a one-frame 16K encoder capability probe for raw and another for final (§28). These are structural counts for the current path, not a speed estimate.

## 23. Flash resume and stage validation

There is **tile-level resume, not emission-level resume**. The child accepts a stable scratch path and request SHA; certification_identity.json binds input path, frame count, tile size/overlap and request fingerprint (upscale.py:L1040-L1072). _validate_tile_commit() requires COMPLETE, identical geometry/count/seed/request SHA and checks **every committed tile PNG size+SHA** plus emission metadata SHA before skipping that tile, logging [UPSCALE_TILE_RESUME] (upscale.py:L751-L783,L1138-L1159). An uncommitted partial tile is moved to incomplete_tiles and regenerated from frame 0; model temporal caches cannot resume from an emission JSON alone. _commit_tile() writes the tile marker only after all N/21 PNGs hash successfully (L786-L807,L1260-L1269).

On CUDA OOM the native runtime clears GPU cache, deletes the attempt's staged tiles, tile commits and emission metadata, reduces source tile size by 32 down to min128 and retries the **complete tile plan**; it logs “CUDA OOM at tile …; retrying complete run …” (upscale.py:L1274-L1300). This is a documented exception to retaining *failed-attempt* intermediates; committed successful-run artifacts remain intended output.

If flashvsr_stage_manifest.json already exists, outer flash_composite.execute() checks matching stage-request JSON, then validate_stage_manifest() checks COMPLETE, input/output identity, model temporal mode/options, expected tile geometry/count, PNG **count/endpoint/existence**, emission metadata and contiguous emission coverage (flash_composite.py:L422-L514,L1107-L1135). **That stage-level shortcut does not rehash every PNG's bytes**; native tile-resume SHA validation runs when the native inference path is re-entered. If raw MP4 validates at expected 16K/16/count, it is reused; otherwise it is retained as an orphan and regenerated. Final output reuse additionally requires PASS final_manifest, matching request and file size+mtime identity plus raw/stage presence and ffprobe metadata (flash_composite.py:L897-L934,L1137-L1163). Neither completion check is a full pixel/content SHA of final MP4.

## 24. Raw 16K assembly

After a COMPLETE stage manifest, flash_composite calls the packaged reference.execute_encode_stage(), which validates stage paths/PNG counts and calls encode_tiles() (flash_composite.py:L1129-L1163; upscale.py:L828-L913). encode_tiles() selects a 16K-capable encoder using a **one-black-frame 15360×8640 test**; with configured preferred libx265, it tests that candidate (media_utils.py:L394-L491). For each frame k it creates full-size float32 accumulator and normalization arrays, reads every tile's frame-k PNG, converts BGR→RGB, computes overlap weights using feather_axis()/tile_weight(), performs weighted addition, divides/rounds to uint8 RGB and streams that frame to FFmpeg (upscale.py:L528-L548,L565-L653). This is **spatial** overlap blending; there is no temporal segment merge.

FFmpeg raw output arguments derive from selected libx265 candidate: preset fast, CRF14, yuv420p10le, hvc1, x265 parameters with bounded pools/frame-threads chosen by select_encoder() (media_utils.py:L394-L451). It writes a video-only temporary MP4 and publishes raw_flashvsr/03_FlashVSR_raw_16K_16fps.mp4. The 4K Outpainting input is video-only, so native finalize_video_with_audio() takes the video-only publish branch (media_utils.py:L577-L635; outpaint.py:L514-L540). Wrapper then re-counts/validates the raw MP4. Raw is retained as certification intermediate, not replaced by final composite.

## 25. Original 8K center reinsertion and 64px mask

**CONFIRMED BY CODE:** 16K expected=(15360,8640), original=(7680,4320), CENTER_BBOX=(2160,3840,6480,11520) in (y0,x0,y1,x1) order (flash_composite.py:L36-L46). Thus exclusive Python slices are **x[3840:11520), y[2160:6480)** exactly. The source YAML's 16px Outpainting width is read and scaled ×4; anything other than final64 fails (flash_composite.py:L270-L281,L1039-L1045).

encode_staged_tiles() constructs 16K output row bands, reads original frame k with OriginalFrameReader.read(k), merges tile PNG rows, and calls composite_original_center_rows() before RawVideoEncoder.write() (flash_composite.py:L758-L882). For only rows intersecting the fixed center bbox, inward_smoothstep_alpha() builds an alpha mask of [intersected-row-count,7680] from the **minimum** distance to top, bottom, left and right center edges. At horizontal/vertical borders and corners distance0→alpha0, so generated pixels remain; from distance64 inward alpha1, so exact original RGB is copied; the intervening 64px band uses alpha=t²(3−2t) and round-to-uint8 blend. A single nearest-edge mask prevents double-blending at corners (flash_composite.py:L637-L710). This is separate from Outpainting's 16px anchor feather.

## 26. Frame-index alignment: guarantee and limit

| Link | Code's selection/order mechanism | What is actually checked |
|---|---|---|
| Original[k] → preprocess[k] | FFmpeg source decode then scale, setpts, fps=24; no explicit frame cap (preprocess.py:L243-L314) | Output count N, size/FPS, **not** per-frame RGB identity. fps filter is a resampler; W2. |
| FHD[k] → Outpaint[k] | StreamingAnchorReader sequential PyAV index and native planner absolute frame ranges (streaming_video.py:L109-L184; inference.py:L220-L267) | Contiguous new-output ranges and N emitted frames; generated visual content is necessarily changed outside anchor. |
| Outpaint[k] → Flash[k] | Full passes same 4K MP4; Quick lossless first-21 frame trim + decoded RGB framemd5 comparison (run_cert.py:L456-L555) | Quick pixel identity of selected decoded RGB clip to first 21 outpaint frames; Full same file. |
| Flash[k] → center Original[k] | Tile callbacks name PNG by k; final loop calls OriginalFrameReader.read(k), a sequential first-N rawvideo decoder using -frames:v N (upscale.py:L498-L515; flash_composite.py:L517-L610,L804-L865) | Reader rejects requested index different from next sequential index. **No timestamp-nearest seek.** |
| 4K/8K timelines | ffprobe -show_frames obtains full timestamp sequences for both and tests first requested positions against 16fps CFR grid (flash_composite.py:L214-L269) | Relative timing regularity, not visual or perceptual identity. |

Therefore the **downstream index policy** is explicit, but a mathematical end-to-end proof of Original[k]=Preprocessed[k] for every k is unavailable from current checks. This remains **NOT RUNTIME-VERIFIED**, especially due to stage-0001 fps filter.

## 27. Audio: processing copy versus final original

Stage 0001 maps only the first source audio stream if one exists, applies atempo=1.5, encodes AAC 192k and preserves audio presence in its metadata validation (preprocess.py:L243-L314,L197-L229). Native Outpainting generates video only; its 24fps assembly uses a video stream and its 16fps retime explicitly sets -an (streaming_video.py:L709-L748; outpaint.py:L514-L540). Quick's 21f selected clip also explicitly uses -an (run_cert.py:L505-L555); Flash raw is video-only (§24).

The **final** mux does not use sped-up processing audio. flash_composite calls media_utils.probe_media(original8k,count_frames=False) for audio presence. If present, it runs FFmpeg with final video partial as input 0 and original 8K as input 1, maps 0:v:0 and **1:a:0**, uses -c copy, -t expected_frames/16, -movflags +faststart, then atomically publishes the muxed MP4 (flash_composite.py:L1200-L1223). For the metadata example N=1952, Full trim argument is 122 seconds; Quick 21/16 is **1.3125 seconds**. If source audio is absent, final video-only partial is published. The audio codec is copied, not re-encoded at final mux. MP4 audio packet/sample boundaries may differ from the decimal -t target; **no separate final audio stream-presence, decoded-sample count or exact audio-duration validation exists** (W6). The earlier source-03 AAC observation is prior metadata, **not re-probed here**.

## 28. Raw and final 16K encoding count and arguments

There are **two distinct full-video 16K encodes on the fresh path**: native tile-PNG weighted raw encode and wrapper tile-PNG row-band + center composite final video-only encode. Audio mux, if present, then **copies** the already encoded final video stream rather than video-reencoding it (upscale.py:L565-L653; flash_composite.py:L1137-L1223). Each 16K encoder selection invokes select_encoder(), which actually encodes **one 15360×8640 black test frame** (raw once, final once), so there are **two 16K codec capability probes** on that path (media_utils.py:L424-L491; flash_composite.py:L1165-L1178). This audit did not run them.

Configured preferred encoder is libx265. Its selected arguments are -c:v libx265 -preset fast -crf 14 -pix_fmt yuv420p10le -tag:v hvc1 and x265 log-level/error plus bounded pools/frame-threads; no bitrate target is specified (media_utils.py:L394-L451). Both raw and final use raw RGB pipe input at 15360×8640/16fps; final RawVideoEncoder adds -an, -frames:v expected_count and propagates available color flags from the 4K input (media_utils.py:L494-L568). The final optional audio-copy mux adds -movflags +faststart; the direct video-only final branch does not add that flag here. Final manifest records encoder choice, probe attempts, frame mapping and audio-source path if present (flash_composite.py:L1224-L1276).

## 29. Final verification contract

The Flash wrapper validate_video_spec() checks resolution, avg and nominal r frame rates, decoded-frame count from ffprobe -count_frames, and approximate duration within max(0.05s,1/fps) for its raw/final checks (flash_composite.py:L120-L212,L885-L934). Parent run_cert.verify_video() again checks raw and final resolution 15360×8640, avg16fps and exact frame count; it writes ffprobe_raw.json/ffprobe_final.json plus Flash config/segment plan and prints [UPSCALE_DONE] (run_cert.py:L390-L402,L640-L679). Final reuse also checks final_manifest PASS, request equality and output size+mtime identity, not final content SHA (flash_composite.py:L897-L934).

Both 03 JSONs set verification.full_decode=false. Thus the optional ffmpeg -v error -i FINAL -f null - in run_cert.py:L665-L668 **does not run by default**. **PASS guarantees only the code-checked metadata/count/provenance conditions and successful encoder exits**. It does **not** prove pixel-by-pixel alignment, every 16K pixel's decode quality, numerical equivalence to reference GPU inference, output hash, or exact final audio sample length. ffprobe -count_frames still traverses all raw/final 16K frames and can itself be expensive; it is not the separately configured full null-decode check.

## 30. Resume state machine

The CLI accepts --config plus mutually exclusive --resume/--force, not a separate Quick program (run_cert.py:L682-L690). **CONFIRMED BY CODE:** --resume still validates JSON, counts the whole original 8K source and resolves limits before stage reuse (run_cert.py:L693-L704). The decisions are:

~~~text
validate_config → preflight_source (full 8K -count_frames)
  ├─ existing run artifacts without --resume/--force → fail
  ├─ --force → move prior run to sibling .archive → fresh execution
  └─ --resume → requested_config.json must be byte-identical
       ↓
0001: probe original again + whole-file SHA
  ├─ FHD + frame_alignment.json + matching source identity
  │  + output probe/count → RESUME_PASS
  ├─ FHD without alignment → archive orphan → regenerate
  └─ changed source or invalid output → fail
       ↓
0002: reconcile native LongRange committed records
  ├─ native files but no run.json → fail
  ├─ native committed chunks → verify contiguous lineage and hashes,
  │  restore RGB prefix and CPU/CUDA RNG; missing chunks are generated
  └─ all chunks committed → skip model load; assemble/verify 24fps N frames
       ↓
wrapper repairs or verifies chunk MP4/latent/carry audit artifacts;
writes manifests; reuses 4K16 only after video-spec/count validation
       ↓
0003 input: Full passes same 4K; Quick validates or creates 21f clip
  ├─ existing Quick clip → source identity + probe/count + decoded-RGB MD5
  └─ missing manifest → preserve orphan clip and regenerate
       ↓
Flash wrapper probes source/4K, counts source again, checks timestamps
and constructs request hash (including whole original+4K SHA)
  ├─ final + PASS final manifest/request + size/mtime/spec → [SKIP]
  ├─ final without PASS manifest → preserve orphan; rebuild
  ├─ complete stage manifest + matching request → reuse stage
  └─ incomplete stage → child reuses individually SHA-verified tiles;
     an uncommitted tile reprocesses its whole causal temporal stream
       ↓
validate/reuse or regenerate raw 16K; reconstruct final center/audio
MP4; PASS manifest; parent ffprobe raw and final
~~~

Run/config branch: run_cert.py:L346-L367,L704-L728. Stage 0001: preprocess.py:L419-L512. Native session/record reconciliation: outpaint.py:L669-L699, streaming_video.py:L541-L674 and inference.py:L1933-L2004; the wrapper always calls sample_long_range() then validates the 24fps assembly and 4K16 final (outpaint.py:L752-L849). Quick clip: run_cert.py:L456-L555. Flash final/stage/tile reuse: flash_composite.py:L1070-L1163 and upscale.py:L751-L807,L1138-L1159. **Emission JSON is not an independent resume unit**. Complete-stage Flash reuse checks manifest coverage/files but does not recompute every PNG SHA (§23, W4).

On a successful --resume branch, the additional literal log forms can include <code>[PREPROCESS] status=RESUME_PASS ...</code> (preprocess.py:L460-L465), JSON <code>{"event":"inference_model_ready",...,"model_load_skipped":true}</code> if every native chunk is committed (inference.py:L1950-L1985), <code>[UPSCALE_TILE_RESUME] tile=... frames=N/N status=PASS</code> for each independently SHA-validated tile (upscale.py:L1140-L1151), <code>[RESUME] reusing complete FlashVSR stage: ...</code>, <code>[RESUME] reusing verified raw FlashVSR video: ...</code> and <code>[SKIP] valid final output already exists: ...</code> for their respective wrapper decisions (flash_composite.py:L1073-L1180). An interrupted Quick clip without manifest emits <code>[UPSCALE_INPUT_ORPHAN_ARCHIVED] path=...</code> (run_cert.py:L481-L489). --force can emit <code>[ARCHIVE] previous run retained at ...</code> **before** run-level Tee opens (run_cert.py:L354-L367).

## 31. Failure paths and exits

| Failure | Code reaction / visible text | Retained or discarded artifact |
|---|---|---|
| Invalid config, unsupported limit, missing model/input path, wrong source spec/count | Top-level <code>[PIPELINE_FAIL] Type: message</code>, exit 1 (run_cert.py:L111-L288,L693-L704,L798-L807). | Before Tee; no new pipeline.log guaranteed. |
| Existing run without --resume/--force, or changed requested-config bytes on resume | <code>[PIPELINE_FAIL]</code> (run_cert.py:L354-L367,L705-L713). | --force archives previous run; --resume rejects mismatched bytes. |
| Stage-0001 FFmpeg or post-encode validation failure inside try | <code>[PREPROCESS] status=FAIL error=...</code>, then <code>[PIPELINE_FAIL]</code> (preprocess.py:L475-L505). | Partial FHD MP4 removed; named output only after verified publication. An error before the try may have only top-level failure. |
| LTX/model/adapter/LongRange/chunk failure | Exception reaches <code>[PIPELINE_FAIL]</code>; no <code>[OUTPAINT_FAILED]</code> event (outpaint.py:L581-L849; run_cert.py:L798-L807). | Previously committed native records/audit artifacts persist; incomplete records do not become PASS. |
| Outpaint chunk or final retime FFmpeg failure | <code>[OUTPAINT_CHUNK_FFMPEG]</code> or <code>[OUTPAINT_FFMPEG]</code> and child diagnostics precede top-level failure (outpaint.py:L220-L239,L291-L333,L494-L554). | Temporary MP4 removed; earlier commits retained. |
| Flash request/model/native child failure | Child prints <code>ERROR: Type: message</code> and attempts a FAIL report; wrapper prints <code>[ERROR] Type: message</code>; parent prints <code>[PIPELINE_FAIL]</code> (upscale.py:L1486-L1510; flash_composite.py:L1345-L1379; run_cert.py:L405-L420,L798-L807). | Request/scratch/committed tiles remain as code permits; no PASS stage manifest for failure. |
| Flash CUDA OOM | Child clears this attempt's tile PNGs, commits and emission metadata, reduces tile size, prints <code>CUDA OOM at tile ...; retrying complete run ...</code>; minimum-size OOM raises (upscale.py:L1274-L1303). | Invalid attempt is not reused. |
| Raw/final 16K encode, audio mux or verify failure | Wrapper <code>[ERROR]</code>, then parent <code>[PIPELINE_FAIL]</code> (flash_composite.py:L1137-L1293). | Invalid raw is archived; partial final removed; final published without PASS manifest is archived on recovery. |
| Optional full pixel decode error | Parent <code>[PIPELINE_FAIL]</code> after raw/final metadata checks (run_cert.py:L640-L668,L798-L807). | This branch is **not entered** by current 03 JSONs (full_decode=false). |

There are no dedicated <code>[PREFLIGHT_FAILED]</code>, <code>[OUTPAINT_FAILED]</code>, <code>[UPSCALE_FAILED]</code> or <code>[VERIFY_FAILED]</code> event strings in this call path. The top-level catches Exception, prints one line and exits 1 rather than re-raising a Python traceback (run_cert.py:L798-L807); subprocesses may already have printed diagnostics. **NOT RUNTIME-VERIFIED:** machine-specific FFmpeg/CUDA error text.

## 32. EXPECTED TERMINAL WALKTHROUGH — Certification 03

Command: <code>./run_cert.sh --config configs/certification/03.json</code>. This table is a **conditional successful fresh-run trace**, not a captured execution. Dynamic paths, N, C, T, elapsed, GPU tile choice and FFmpeg's own lines are placeholders. Each row is a real code-owned log string or an explicitly identified silent operation; source and meaning follow each visible shape.

| Order | Visible line shape or silence | Source | Meaning |
|---:|---|---|---|
| 0 | **No pipeline progress line yet** | run_cert.py:L691-L728; preprocess.py:L134-L163 | JSON validation, full 8K <code>ffprobe -count_frames</code>, frame-limit resolution, run layout and config snapshots precede the run-level Tee and <code>[PREFLIGHT]</code>. |
| 1 | <code>&lt;UTC&gt; [PREFLIGHT] job=03_certification video=03 seed=33333333 input=03_Input_8K.mp4 resolution=7680x4320 fps=16 frames=&lt;N&gt; config=&lt;path&gt; status=PASS</code> | run_cert.py:L729-L740 | Source preflight passed. |
| 2 | <code>&lt;UTC&gt; [SEED] seed=33333333 python=global numpy=global pytorch=global flashvsr=per_tile_generator_plus_global_rng</code> | run_cert.py:L370-L387,L741 | Global RNG initialization. |
| 3 | <code>&lt;UTC&gt; [PREPROCESS] status=START input=&lt;8K&gt; output=&lt;FHD&gt; expected_frames=&lt;N&gt; resume=False</code> | preprocess.py:L414-L418 | Stage 0001 begins; another full source count and SHA are performed before its command. |
| 4 | <code>&lt;UTC&gt; [PREPROCESS] command=ffmpeg ... scale=1920:1080:flags=lanczos:in_range=pc:out_range=pc,setsar=1,setpts=PTS*2/3,fps=24 ...</code> | preprocess.py:L243-L314,L467-L475 | Full-source FFmpeg command; native FFmpeg lines are mirrored after it. |
| 5 | <code>&lt;FFmpeg output/progress&gt;</code>, then <code>&lt;UTC&gt; [PREPROCESS] status=PASS frames=&lt;N&gt; resolution=1920x1080 fps=24 output=&lt;FHD&gt;</code> | preprocess.py:L475-L512 | Verified FHD output published. |
| 6 | <code>&lt;UTC&gt; [OUTPAINT] video=03 seed=33333333 frames=&lt;N&gt; steps=30 status=START</code> | run_cert.py:L753-L767 | Same packaged Outpainting implementation as Quick. |
| 7 | Possibly JSON <code>{"event":"video_only_gpu_weights",...}</code>; then JSON <code>{"event":"inference_model_ready","seed":33333333,"inference_steps":30,"model_load_skipped":false,...}</code> | inference.py:L700-L704,L1950-L1985 | Optional memory-optimization report, then native model/adapter readiness. Actual JSON is sorted by key; shape here is abbreviated. |
| 8 | Chunk 1: 30 JSON <code>{"event":"inference_denoise_step","chunk_index":0,"chunk_count":&lt;C&gt;,"step":1..30,"total_steps":30,"elapsed_seconds":...,"eta_seconds":...}</code> lines | inference.py:L310-L336; packaged validation_runner.py:L855-L982 | Actual Euler-step-completion callback, not a fake timer. |
| 9 | <code>[OUTPAINT_CHUNK_FFMPEG] command=ffmpeg ...</code>, FFmpeg lines; <code>[OUTPAINT_CHUNK_DONE] chunk=1/&lt;C&gt; completed_frames=65/&lt;N&gt; mp4=&lt;path&gt; latent=&lt;path&gt; carry=&lt;path&gt; elapsed=&lt;s&gt;s eta_total=&lt;s&gt;s</code>; then JSON <code>{"event":"inference_chunk_complete",...}</code> | outpaint.py:L241-L333,L725-L750; inference.py:L2175-L2202 | Native commit has completed; wrapper persists MP4/carry before native chunk-complete print. |
| 10 | Repeat step JSON, chunk MP4, chunk-done and native chunk-complete for chunks 2…C | inference.py:L2054-L2202 | For N=1952, C=41 **by static planner derivation** (§10), not a hard-coded runtime assumption. Final completed_frames=N/N. |
| 11 | JSON <code>{"event":"inference_complete","chunk_count":&lt;C&gt;,"gap_count":0,"duplicate_count":0,...}</code> | inference.py:L2285-L2297 | Native 24fps assembly complete with no gaps/duplicates in a successful run. |
| 12 | <code>[OUTPAINT_FFMPEG] command=ffmpeg ...</code>, FFmpeg lines; <code>&lt;UTC&gt; [OUTPAINT_DONE] video=03 frames=&lt;N&gt;/&lt;N&gt; status=PASS output=&lt;4K16&gt;</code> | outpaint.py:L494-L554,L794-L849; run_cert.py:L768-L774 | 24→16 retime and wrapper metadata checks finished. |
| 13 | <code>&lt;UTC&gt; [UPSCALE] video=03 seed=33333333 frames=&lt;N&gt; source=&lt;4K16&gt; model=FlashVSR_v1.1 scale=4 seed_mode=per_tile_generator_plus_global_rng</code>; <code>&lt;UTC&gt; [COMMAND] argv=&lt;Flash wrapper command&gt;</code> | run_cert.py:L558-L639,L405-L420 | Full passes unchanged 4K16 path to child wrapper. |
| 14 | **Possible long gap**, then <code>[FLASHVSR] &lt;packaged upscale.py command&gt;</code> | flash_composite.py:L1047-L1125 | Before that line wrapper counts full original 8K again, enumerates first N timestamps and hashes whole original+4K for its request identity. |
| 15 | <code>[SEED] model=FlashVSR_v1.1 base_seed=33333333 seed_mode=per_tile_generator_plus_global_rng</code>; labels <code>Input path / Resolution / FPS / Frame count / Duration / Pixel format / Audio presence</code>; <code>[MODEL_LOAD] model=&lt;name&gt; gpu=&lt;id&gt; seed=33333333 status=READY elapsed=&lt;s&gt;s</code> | upscale.py:L1001-L1006,L107-L113,L1115-L1123 | Child runtime metadata and actual model load. |
| 16 | Per tile, <code>[UPSCALE] tile=&lt;t&gt;/&lt;T&gt; frame=1/&lt;N&gt; progress=&lt;p&gt;% elapsed=&lt;s&gt;s eta=&lt;s&gt;s</code>, then frames 10,20,…,N; interleaved <code>[UPSCALE_SEGMENT_DONE] tile=&lt;t&gt;/&lt;T&gt; emission=&lt;e&gt; frames=&lt;a&gt;-&lt;b&gt; artifact=&lt;JSON&gt; elapsed=&lt;s&gt;s</code>; then <code>[UPSCALE_TILE_DONE] tile=&lt;t&gt;/&lt;T&gt; frames=&lt;N&gt;/&lt;N&gt; status=PASS</code> | upscale.py:L1168-L1269 | Percentage and ETA are **tile-local**. Emission JSON names tile PNGs; only complete tiles are resume units. |
| 17 | Child JSON <code>{"status":"STAGED","output_path":...,"report_json":...,"peak_torch_allocated_mb":...}</code>; <code>[FLASHVSR_RAW_ENCODE] output=&lt;raw16K&gt;</code> | upscale.py:L1496-L1503; flash_composite.py:L1153-L1163 | All tile streams staged, then raw 16K encode. No per-frame raw-encoder progress log is printed here. |
| 18 | <code>[ENCODE] video_id=03 frame=1/&lt;N&gt;</code>, then 10,20,…,N; <code>[DONE] video_id=03 output=&lt;final16K&gt;</code> | flash_composite.py:L758-L882,L1290-L1293 | Final tile row-band assembly, center feather and encode; optional original-audio mux occurs before DONE. |
| 19 | <code>&lt;UTC&gt; [UPSCALE_DONE] temporal_units=&lt;E&gt; tiles=&lt;T&gt; frames=&lt;N&gt;/&lt;N&gt; resolution=15360x8640 fps=16 status=PASS output=&lt;final&gt;</code>; <code>&lt;UTC&gt; [PIPELINE_DONE] job=03_certification video=03 seed=33333333 frames=&lt;N&gt; output=&lt;final&gt; status=PASS</code> | run_cert.py:L640-L679,L786-L794 | Parent raw/final ffprobe validation and metadata publication finished. |

FFmpeg/library diagnostics may add lines; the table lists only code-owned log shapes. GPU tile choice and OOM fallback can change T. **NOT RUNTIME-VERIFIED:** an actual successful terminal transcript.

## 33. EXPECTED TERMINAL WALKTHROUGH — Quick 03

Command: <code>./run_cert.sh --config configs/quick/03.json</code>. The **same functions run in the same order**; limits/job/output identifiers differ (§3). Fresh successful sequence:

| Order | Actual visible line shape or silence | Source | Meaning |
|---:|---|---|---|
| 0 | **No pipeline progress line yet** | run_cert.py:L693-L728 | Quick still counts the full 8K original before PREFLIGHT logging. |
| 1 | <code>&lt;UTC&gt; [PREFLIGHT] job=03_quick video=03 seed=33333333 ... frames=&lt;source N&gt; ... status=PASS</code>; <code>&lt;UTC&gt; [SEED] seed=33333333 ...</code> | run_cert.py:L729-L741 | Preflight frames is full N, not 65. |
| 2 | <code>[PREPROCESS] status=START ... expected_frames=&lt;source N&gt; resume=False</code>; <code>[PREPROCESS] command=ffmpeg ...</code>; FFmpeg lines; <code>[PREPROCESS] status=PASS frames=&lt;source N&gt; ...</code> | preprocess.py:L414-L512 | Second 8K count, whole-file SHA, **whole-source** FHD24 encode. |
| 3 | <code>&lt;UTC&gt; [OUTPAINT] video=03 seed=33333333 frames=65 steps=30 status=START</code>; optional memory JSON; <code>{"event":"inference_model_ready",...}</code> | run_cert.py:L753-L767; inference.py:L1950-L1985 | First 65 sequential FHD frames enter the same native planner; one chunk (§10). |
| 4 | 30 <code>{"event":"inference_denoise_step",...,"step":1..30,"total_steps":30}</code> lines; <code>[OUTPAINT_CHUNK_FFMPEG] ...</code>; <code>[OUTPAINT_CHUNK_DONE] chunk=1/1 completed_frames=65/65 ...</code>; <code>{"event":"inference_chunk_complete",...}</code> | inference.py:L310-L336,L2175-L2202; outpaint.py:L725-L750 | No second LongRange chunk. |
| 5 | <code>{"event":"inference_complete","chunk_count":1,...}</code>; <code>[OUTPAINT_FFMPEG] ...</code>; <code>&lt;UTC&gt; [OUTPAINT_DONE] video=03 frames=65/65 status=PASS output=&lt;4K16&gt;</code> | inference.py:L2285-L2297; outpaint.py:L494-L554; run_cert.py:L768-L774 | 65-frame 4K16 output retained. |
| 6 | <code>&lt;UTC&gt; [COMMAND] argv=ffmpeg ... trim=end_frame=21,setpts=N/(16*TB) ...</code>, then FFmpeg lines | run_cert.py:L456-L555 | **Before [UPSCALE]**, creates a lossless-HEVC first-21-frame 4K clip and compares decoded-RGB checksums. On valid resume, command is absent. |
| 7 | <code>&lt;UTC&gt; [UPSCALE] video=03 seed=33333333 frames=21 source=&lt;21f clip&gt; model=FlashVSR_v1.1 scale=4 ...</code>; <code>&lt;UTC&gt; [COMMAND] argv=&lt;Flash wrapper&gt;</code> | run_cert.py:L573-L639 | Wrapper still counts/hashes original full 8K; native model input is only the 21f clip. |
| 8 | <code>[FLASHVSR] &lt;native command&gt;</code>; <code>[SEED] model=FlashVSR_v1.1 base_seed=33333333 ...</code>; native metadata <code>Frame count   : 21</code>; <code>[MODEL_LOAD] ...</code> | flash_composite.py:L1047-L1125; upscale.py:L1001-L1123 | Same native model; 21 real frames padded to 25 model frames (§20). |
| 9 | Per tile: <code>[UPSCALE] ... frame=1/21 progress=4.8%</code>, then <code>frame=10/21 progress=47.6%</code>, <code>frame=20/21 progress=95.2%</code>, <code>frame=21/21 progress=100.0%</code>; <code>[UPSCALE_SEGMENT_DONE] ... emission=1 frames=0-20 artifact=&lt;JSON&gt; ...</code>; <code>[UPSCALE_TILE_DONE] ... frames=21/21 status=PASS</code> | upscale.py:L1168-L1269 | One causal emission metadata record per tile. Percentages are source format <code>:.1f</code>. |
| 10 | Child JSON <code>{"status":"STAGED",...}</code>; <code>[FLASHVSR_RAW_ENCODE] output=&lt;raw16K&gt;</code>; <code>[ENCODE] video_id=03 frame=1/21</code>, 10/21, 20/21, 21/21; <code>[DONE] video_id=03 output=&lt;03_Quick_Final_16K_16fps_21f.mp4&gt;</code> | upscale.py:L1496-L1503; flash_composite.py:L1153-L1293 | PNGs → raw 16K → final center composite. Original audio trimmed to 21/16 s if present. |
| 11 | <code>&lt;UTC&gt; [UPSCALE_DONE] ... frames=21/21 ... status=PASS</code>; <code>&lt;UTC&gt; [PIPELINE_DONE] job=03_quick video=03 seed=33333333 frames=21 ... status=PASS</code> | run_cert.py:L640-L679,L786-L794 | Parent final probe and run completion. |

**CONFIRMED BY CODE:** Quick's full-source operations are not canceled by its later 65/21 limits. **NOT RUNTIME-VERIFIED:** timings, chosen tile size and terminal transcript.

## 34. Artifact timeline (fresh 03 run)

Layout directories and placeholder files already exist. This table concerns *new runtime content*, relative to the selected <code>data/{certification|quick}/video_03_seed33333333/</code>. Atomic writes may briefly create a temporary sibling before the final name.

| Time | Certification 03 | Quick 03 | Publication source |
|---|---|---|---|
| T0 after preflight | <code>config/requested_config.json</code>, <code>resolved_config.json</code>, <code>logs/pipeline.log</code> | Same in quick root; resolved source=N, outpaint=65, upscale=21 | run_cert.py:L704-L728 |
| T1 preprocess start | <code>0001_Preprocess_FHD24/logs/preprocess.log</code>, <code>metadata/input_ffprobe.json</code> after SHA/probe | Same whole-source work | preprocess.py:L414-L474 |
| T2 preprocess commit | Temporary FHD MP4 then <code>output/03_Input_FHD_24fps.mp4</code>, <code>metadata/output_ffprobe.json</code>, <code>frame_alignment.json</code> | Same filename in separate quick root, with full N frames | preprocess.py:L475-L512 |
| T3 Outpaint initialize | <code>0002_Outpaint_4K16/chunks/native/run.json</code>, native journal/session files, stage log | Same, but frame limit65 | inference.py:L1905-L1945; outpaint.py:L637-L768 |
| T4 first chunk | Generated-latent audit, native segment/record, wrapper <code>chunks/mp4/chunk_0001_frames_000000_000064.mp4</code> and <code>chunks/latents/chunk_000000/{metadata.json,generated_latent.pt,carry_prefix_state.pt,certification_artifacts.json}</code> | Same first-chunk names; only one chunk | inference.py:L1698-L1730,L2054-L2202; outpaint.py:L241-L427 |
| T5 later chunks | Repeat T4 through N; 41 total only if N=1952 | None | inference.py:L2203-L2297 |
| T6 Outpaint complete | <code>chunks/native/03_Outpaint_4K_24fps.mp4</code>, <code>chunks/manifests/{chunk_plan.json,chunk_results.jsonl}</code>, <code>final/03_Outpaint_4K_16fps.mp4</code>, metadata config/checkpoint/seed/probes | Same names in quick root, 65f | outpaint.py:L770-L849 |
| T7 Flash input | Full passes existing final 4K16; no selected clip | <code>0003_Upscale_FlashVSR_16K/segments/input/03_Outpaint_4K_16fps_21f.mp4</code>, <code>metadata/upscale_input.json</code> after RGB-MD5 equality | run_cert.py:L456-L555 |
| T8 Flash request | <code>segments/flashvsr_stage_request.json</code>; <code>segments/tiles/certification_stage/certification_identity.json</code> | Same under quick root; request identifies 21f clip | flash_composite.py:L317-L373,L1070-L1125; upscale.py:L1043-L1070 |
| T9 tile processing | <code>segments/tiles/certification_stage/tiles/tile_XXXX/000000000.png...</code>, <code>segments/metadata/attempt_.../tile_..._emission_...json</code>, <code>segments/tiles/certification_stage/tile_commits/...</code> | 21 tile-crop PNGs per tile; one emission metadata per tile | upscale.py:L1131-L1269 |
| T10 staged/raw | <code>segments/flashvsr_stage_manifest.json</code>, stage report, <code>raw_flashvsr/03_FlashVSR_raw_16K_16fps.mp4</code> | Same shape | upscale.py:L1397-L1503; flash_composite.py:L1129-L1163 |
| T11 final | Temporary final video-only MP4; optional temporary audio mux; <code>final/03_Certification_Final_16K_16fps.mp4</code>, <code>metadata/final_manifest.json</code> | <code>final/03_Quick_Final_16K_16fps_21f.mp4</code> | flash_composite.py:L1180-L1293 |
| T12 parent verify | <code>metadata/{ffprobe_raw.json,ffprobe_final.json,flashvsr_config.json,segment_plan.json}</code>, final run/stage log lines | Same in quick root | run_cert.py:L640-L679 |

The stage manifest records the actual absolute scratch path on a real run. **NOT RUNTIME-VERIFIED:** no T0–T12 runtime artifact was created during this documentation task.

## 35. Known warnings (code-grounded, no code changes)

### W1 — Quick pays full-source startup and preprocess cost

- **Evidence:** Quick resolves 65/21 only *after* preflight_source(); stage 0001 receives source_frames, not 65. Its source identity hashes the entire file. Flash wrapper probes/counts the original again and stage_request_payload() SHA-hashes the whole original and selected 4K input. The first user-facing preflight line is after the initial count.
- **Source:** run_cert.py:L693-L752,L573-L639; preprocess.py:L65-L79,L134-L163,L374-L512; flash_composite.py:L1047-L1072,L376-L419.
- **Potential impact:** A nominal 65f/21f Quick run can wait for **three complete original 8K -count_frames traversals**, two full-original SHA traversals, one timestamp frame-list traversal, and whole-source FHD preprocessing before model work. The child additionally counts the selected 4K input. The earlier report of one 8K -count_frames taking **more than five minutes** is a **RUNTIME OBSERVATION (PRIOR)**, not a new measurement.
- **Currently mitigated by:** FHD stage is reusable under --resume when source identity and output metadata match; the Quick model itself receives only 65/21 frames.
- **Not verified because:** No startup timing, full decode or preprocess was run in this document task.

### W2 — Stage-0001 fps filter is a resampler, not a formal one-to-one mapping proof

- **Evidence:** Builder uses <code>scale=...,setsar=1,setpts=PTS*2/3,fps=24</code>. For ideal 16fps CFR timestamps, scaled PTS are spaced 1/24s and the fps filter is expected to emit N frames, but it makes its own boundary decisions; input irregularity/end rounding could alter emitted frames. frame_alignment.json records counts/spec/source identity, not a per-frame visual or timestamp-index equality proof.
- **Source:** preprocess.py:L31-L41,L243-L314,L166-L229,L319-L372.
- **Potential impact:** A mismatch in emitted N is caught by validate_output_probe(), but a count-preserving replacement/duplication at a boundary would not be detected by frame_alignment.json alone.
- **Currently mitigated by:** Input 16fps and output exact N/24fps checks. Quick's later decoded-RGB comparison protects only the Outpaint→Flash clip boundary, **not** Original→FHD identity.
- **Not verified because:** No real FFmpeg preprocessing or decoded frame-by-frame comparison was executed.

### W3 — Full Flash tile PNGs, memmaps and 16K assembly are large

- **Evidence:** One PNG per tile per output frame; for N=1952, 112/170/336/920 spatial tiles imply 218,624/331,840/655,872/1,795,840 tile-crop PNGs depending on auto size/OOM fallback (§19). build_tile_store creates a tile conditioning memmap; raw encode_tiles() allocates a full 16K float32 RGB accumulator and float32 normalization array, approximately 2.123 GB decimal together before tile weights/frame/FFmpeg buffers; final composite uses row-band accumulation. Fresh run performs raw and final full-video 16K encodes.
- **Source:** upscale.py:L258-L315,L519-L525,L565-L653,L916-L926,L1274-L1303; flash_composite.py:L758-L882,L1137-L1223.
- **Potential impact:** Output disk, inode count, memory and I/O pressure; tile commit SHA and stage coverage checks also traverse many files.
- **Currently mitigated by:** Per-tile causal processing, on-disk PNG/memmap storage, OOM tile-size fallback and final row-band assembly. These lower peak memory in some phases but do not remove artifact cost.
- **Not verified because:** No benchmark, actual PNG compression ratio, free-space check or VRAM execution was performed.

### W4 — Complete Flash-stage reuse does not rehash every tile PNG

- **Evidence:** _validate_tile_commit() checks each PNG's path/size/SHA and each emission JSON SHA when resuming an *incomplete native stage*. However validate_stage_manifest() for a completed stage checks tile geometry, PNG count/endpoint/presence and emission coverage, not every PNG's content hash. Final reuse checks final file size+mtime/request/spec, not final SHA.
- **Source:** upscale.py:L751-L807,L1138-L1159; flash_composite.py:L422-L514,L897-L934,L1107-L1135.
- **Potential impact:** Same-size altered PNG in a completed stage, or replaced final MP4 with preserved size/mtime and plausible probe metadata, may evade complete-stage reuse validation.
- **Currently mitigated by:** Initial tile commit SHA, complete tile count/coverage/spec checks, request hashing of original/4K/runtime/YAML, final manifest and FFprobe checks.
- **Not verified because:** No mutation or corruption-injection test was run; this is static validation-scope analysis.

### W5 — Default PASS is not full-pixel-decode validation

- **Evidence:** Both 03 JSONs set verification.full_decode=false. The sole optional <code>ffmpeg -v error -i FINAL -f null -</code> branch is guarded by that value. Normal raw/final probes use ffprobe -count_frames; no explicit full pixel null-decode validation follows.
- **Source:** configs/certification/03.json, configs/quick/03.json; run_cert.py:L640-L668; flash_composite.py:L885-L934.
- **Potential impact:** A metadata/count PASS does not certify every pixel decodes correctly, nor GPU numerical equivalence or center-image fidelity.
- **Currently mitigated by:** Encoder exit checks, ffprobe stream/count/resolution/FPS checks and atomic publication.
- **Not verified because:** This audit intentionally did not decode an output or generate any video.

### W6 — Final audio exact sample length is not validated

- **Evidence:** Final mux maps original audio stream 1:a:0, copies it and applies <code>-t N/16</code>. validate_final_output() and parent verify_video() validate the video stream's geometry/FPS/frame count; no post-mux audio sample count or exact duration comparison exists.
- **Source:** flash_composite.py:L1200-L1223,L885-L934; run_cert.py:L390-L402,L640-L679.
- **Potential impact:** Packet/sample boundaries may yield audio duration slightly different from the nominal N/16 video duration. Audio absence still yields a valid video-only output.
- **Currently mitigated by:** Original audio source is used rather than sped-up processing audio; FFmpeg mux exit is checked and final video is probed.
- **Not verified because:** No final output exists from this task and no audio decoding/sample count was run.

### W7 — Model-manifest hash and output-free-space are not preflight gates

- **Evidence:** validate_config() checks required package paths/asset files and expected JSON/YAML values but does not read the packaged model manifest or recompute model-weight SHA, nor call disk_usage/statvfs for prospective output space. Flash stage request records model asset path/size/mtime identities, not model weight SHA; source/config/runtime hashes are stronger.
- **Source:** run_cert.py:L111-L277,L280-L288; flash_composite.py:L376-L419. The static package manifest is <code>src/manifests/model_manifest.json</code>.
- **Potential impact:** Existing-but-corrupt weights or insufficient space can fail after expensive earlier work; copied model manifest alone is provenance, not a runtime integrity gate.
- **Currently mitigated by:** Existence checks, source/config request hashes, model loader failure if corrupt and atomic artifact commits. Operators can check disk and package hashes before launch.
- **Not verified because:** Neither model hashes nor free disk/weights were revalidated here; code audit does not equal runtime integrity validation.

These are **WARNING**, not proof of failure on a particular machine. No new code-proven **BLOCKER** was found in this static documentation pass; **READY WITH WARNINGS** remains conditional on an actual pre-run environment and inference test.

## 36. Notes and evidence boundaries

1. **CONFIRMED BY CODE:** Flash temporal processing is a continuous causal/KV stream *inside each spatial tile*, with emission records. An emission is not an independent overlapping temporal segment or resume checkpoint (upscale.py:L390-L516,L751-L807).
2. **CONFIRMED BY CODE:** Outpaint 24→16 retime uses <code>settb=expr=1/16,setpts=N</code>, passthrough FPS mode, N-frame verification and HEVC CRF1 re-encode. Frame sequence/count is retained by code contract; decoded pixels need not be bit-exact through lossy encoding (outpaint.py:L494-L554). Stage-0001's fps=24 is a distinct resampler (W2).
3. **CONFIRMED BY CODE:** Center Original[k] is decoded sequentially with an index assertion; it is not selected by a timestamp-nearest seek (flash_composite.py:L517-L610,L758-L882). This does not prove upstream Preprocess[k] pixel identity (§26).
4. **CONFIRMED BY CODE:** The active runner/configs resolve source/model paths beneath this package; targeted grep over run_cert.sh, run_cert.py, cert_pipeline, packaged YAML and packaged Flash entrypoint found no runtime dependency on the two original absolute repo paths. The original absolute paths remain valid *provenance data* in manifests. External Python environments, FFmpeg and system libraries are still required (run_cert.sh:L1-L14; run_cert.py:L83-L90; src/manifests/source_manifest.json).
5. **CONFIRMED BY CODE:** Run-level Tee mirrors Python stdout/stderr and child output to pipeline.log; preprocess writes its own stage file, and stage_log mirrors outpaint/Flash output to stage files (run_cert.py:L52-L80,L724-L785; preprocess.py:L82-L87).
6. **NOT RUNTIME-VERIFIED:** No inference result, startup timing, checkpoint numerical equivalence, tile count for this GPU, video/audio bitstream integrity or pixel alignment was established by this audit.

## 37. Reference implementation preservation and local changes

The package manifest records source/destination, hashes and relation categories; the two external repositories were compared **read-only** with their packaged counterparts for the principal runtime files (src/manifests/source_manifest.json). The relationship is **not** that the orchestration rewrites a similar model: it calls packaged native implementations. File-level categories:

| File / responsibility | Classification | Code-level evidence and boundary |
|---|---|---|
| <code>src/video_outpainting/main/src/ltx_outpaint/streaming_video.py</code> | **REUSED AS-IS** | Read-only <code>diff -q</code> against VideoOutpainting/main/src/ltx_outpaint/streaming_video.py found no difference. This owns native segment persistence, contiguous lineage, RGB prefix/RNG resume and 24fps assembly. |
| <code>src/video_outpainting/models/LTX2.3/packages/ltx-trainer/src/ltx_trainer/validation_runner.py</code> and packaged official LTX model implementation | **REUSED AS-IS** where recorded in source manifest | Euler/scheduler/CFG/STG loop remains the packaged official runner, called by sample_chunk() (validation_runner.py:L855-L982; source_manifest.json). |
| <code>src/video_outpainting/main/src/ltx_outpaint/inference.py</code> | **INSTRUMENTED ONLY** | Read-only diff shows additions to _SamplingProgress.advance_step(), optional final_latent_observer before native VAE decode, _save_certification_latent(), optional committed observer after store.commit_chunk(), and RNG restoration after observer. The original _iter_long_range_chunks(), temporal prefix logic and sample_chunk denoising invocation remain on the native call path. The diff hunk around commit changes the call to assign its return for the observer, not the native commit arguments (inference.py:L310-L336,L1176-L1178,L1695-L1730,L2073-L2185). |
| <code>src/video_outpainting/main/src/ltx_outpaint/model.py</code>, <code>outpainting_adapter.py</code> | **INSTRUMENTED + PROVENANCE/PATH ADAPTATION** | Manifest marks instrumented_with_provenance_path_adaptation. The package-relative YAML/weights are used; model architecture remains the packaged native implementation (source_manifest.json; inference.py:L604-L712). |
| <code>src/video_outpainting/main/configs/inference_4k.yaml</code> | **PATH ADAPTATION** | Manifest marks copied_with_path_adaptation; run_cert.validate_config() requires its model/text/checkpoint paths to resolve inside package (run_cert.py:L247-L276). |
| <code>src/flashvsr/architect/FlashVSR_v1.1/upscale.py</code> | **INSTRUMENTED + PATH ADAPTATION** | Read-only diff changes MODEL_ROOT to src/models/flashvsr_v1_1; adds emission callback/metadata, global seed, staged tile PNG commit/SHA/resume and progress logs. The diff around _stream_official_tile_impl() retains the preexisting causal model/decoder/caches and adds callback after frame emission, not a new temporal inference loop (upscale.py:L35-L53,L390-L516,L751-L807). |
| <code>src/flashvsr/architect/common/media_utils.py</code> | **REUSED AS-IS** | Read-only <code>diff -q</code> against Upscaler/architect/common/media_utils.py found no difference; encoder selection/probing and RawVideoEncoder are reused. |
| <code>src/cert_pipeline/flash_composite.py</code> | **COPIED WITH STAGE/PATH ADAPTATION** | Manifest source is this pipeline's earlier src/03to04/upscale_blend_16k.py, not a newly written Flash temporal algorithm. It orchestrates staged native Flash, raw encode, center original and audio (source_manifest.json; flash_composite.py:L1029-L1293). |
| <code>run_cert.sh</code>, <code>run_cert.py</code>, <code>src/cert_pipeline/preprocess.py</code>, <code>src/cert_pipeline/outpaint.py</code> | **NEW ORCHESTRATION / ARTIFACT WRAPPER** | JSON enforcement, preflight, FFmpeg preprocessing, native function call, audit persistence, retime, logs, resume and final checks. These are not replacements for LongRange or Tiny Long computation. |
| <code>src/models/...</code> weights and model assets | **PACKAGED ASSETS** | src/manifests/model_manifest.json documents copied asset size/SHA and provenance; runtime config resolves package-relative paths. Manifest presence is distinct from runtime hash checking (W7). |

**Read-only diff scope:** the two directly compared native source pairs streaming_video.py and media_utils.py are identical; inference.py and upscale.py differ in the instrumentation/path hunks described above. This is **CONFIRMED BY CODE**, not a numerical GPU equivalence proof. Preserved LongRange chunk planner, RGB prefix/carry semantics, causal VAE conversion and Flash model_fn_wan_video()/pre_cache_k/pre_cache_v/TCDecoder computation are cited in §§9–13 and §§18–21. **NOT RUNTIME-VERIFIED:** identical GPU outputs between reference and packaged builds; instrumentation I/O and environment differences can affect runtime even when computation code is retained.

## Pre-Run Certification Checklist

The following are operator checks **before** a real run; this audit did not perform them:

- [ ] Select the intended JSON: <code>configs/certification/03.json</code> for all/all or <code>configs/quick/03.json</code> for 65/21; inspect job/output filename.
- [ ] Confirm <code>03_Input_8K.mp4</code> exists at the package-relative input path and is the intended 7680×4320/16fps source; compare trusted external provenance if required.
- [ ] Confirm seed 33333333, checkpoint-007500, steps30, prompts, feather16/64, 4× Flash model and final16fps in the selected JSON/YAML.
- [ ] Verify the packaged model files and checkpoint against <code>src/manifests/model_manifest.json</code> by a trusted hash check if certification policy requires weight integrity; current runtime preflight checks existence, not every asset hash (W7).
- [ ] Check that the designated Outpainting/Flash Python environments and required Python packages are installed; <code>ffmpeg</code>, <code>ffprobe</code> and libx265/selected 16K encoder are available.
- [ ] Confirm the configured GPU index, available VRAM and system RAM; automatic Flash tile size depends on actual free GPU memory.
- [ ] **Check sufficient output disk free space and inode capacity**, especially for Full Flash tile×frame PNG retention, conditioning memmaps, two 16K videos, per-chunk latent/MP4 artifacts and archive/resume data (W3).
- [ ] Decide fresh/--resume/--force intentionally. Inspect the selected run directory and existing requested_config.json, LongRange native commits and Flash tile/stage/final manifests; --force archives the previous run rather than deleting it.
- [ ] Allow for the silent initial full-8K frame count and Quick's full-source SHA/FHD preprocessing; absence of early <code>[PREFLIGHT]</code> does not by itself prove a hang (W1).
- [ ] Review source/audio stream presence and whether exact final audio sample-length validation is a certification requirement (W6).
- [ ] Decide whether the current metadata/count validation is sufficient; both 03 JSONs currently disable optional full null-decode. Any change to this verification condition is a separate configuration decision outside this documentation task (W5).
- [ ] Retain requested/resolved config snapshots, stage logs, native chunk/tile journals, raw 16K and final MP4 as assessment evidence.

**Audit stop condition:** document completed by reading current code and manifests only. Pipeline source/config unchanged. No GPU inference, model load, CUDA test, FFmpeg preprocessing, smoke test, generated result or output validation was executed.
