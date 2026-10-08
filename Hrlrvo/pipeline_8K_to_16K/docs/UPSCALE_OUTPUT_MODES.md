# Upscale output modes

The shared `run_cert.sh` / `run_batch.sh` FlashVSR stage now defaults to
`upscale.output_mode = "final_only"` when the JSON key is absent. The existing
certification and quick JSON files are deliberately byte-unchanged. Their
future **fresh** runs therefore use final-only, while the batch queue still
calls the same `run_upscale_stage()` implementation.

| Mode | FlashVSR temporal inference | Retained tile PNGs | Raw 16K MP4 | Final 16K MP4 |
| --- | --- | --- | --- | --- |
| `final_only` (default) | Packaged continuous causal stage | Yes | **Not encoded** | Tile reconstruction + original 8K center/64px feather + original audio |
| `legacy_raw` (opt-in) | Same packaged continuous causal stage | Yes | Encoded and retained | Same direct tile reconstruction/composite, then original audio |

The old raw MP4 was **not** an input to the final composite: both modes use
`flash_composite.encode_staged_tiles()` for the final image. Final-only omits
only `reference.execute_encode_stage()` and its raw 16K verification. It does
not skip FlashVSR inference, change auto tiling/seed/temporal behavior, or
change center feather geometry.

## Commands

Default Full and Quick remain unchanged:

```bash
./run_cert.sh --config configs/certification/05.json
./run_cert.sh --config configs/quick/05.json
./run_batch.sh --config configs/batch/certification.json
```

For a separate, opt-in legacy raw run **after** the corresponding canonical
Outpainting stage has a PASS receipt:

```bash
./run_cert.sh --config configs/legacy_raw/05.json --stage upscale
```

The same `configs/legacy_raw/<ID>.json` exists for 01–05. It uses the
canonical committed Outpainting input but writes Upscale artifacts beneath
`data/03_Upscale_16K_15360x8640/<ID>/legacy_raw/`, leaving the canonical
final output untouched. This separate profile performs its own FlashVSR
stage; it does not alias the canonical tile journal. The legacy profile is
stage-only by design, so it cannot accidentally rerun canonical preprocessing
or Outpainting.

## Verification and resume

- Final-only checks the producer's exact `frames_written` counter and
  `ffprobe` container/header `nb_frames`, 16fps, 15360×8640, and duration.
  No raw 16K MP4 or `ffprobe_raw.json` is created. `ffprobe_final.json` records
  that decoded-frame counting was **not** performed. The PASS manifest does
  **not** assert full pixel decode or visual-quality verification.
- Legacy raw preserves the previous raw and final `-count_frames` checks.
  `verification.full_decode=true` additionally requests a full final decode
  in either mode; it is not the default.
- A final-only PASS is reusable without raw MP4. An already-completed legacy
  PASS can still be reused under the unchanged original JSON with `--resume`;
  it is verified as **legacy**, not mislabeled as newly generated final-only.
  Explicit mode changes on an occupied stage are rejected by stage-request
  validation rather than silently mixing outputs. Existing files are not
  deleted or converted.

## Evidence boundary

The final compositor was already a direct tile-to-final path in the audited
implementation. Isolated H200/A6000 final-only runs and bounded representative
pixel equivalence provided prior operational evidence. This shared-path change
was checked with Python/shell syntax, config validation, Ruff and CPU-only
mock tests; **no new GPU inference or 16K encode was executed for this edit**.
The complete new shared Full/Quick/Batch path remains runtime-unverified until
a future actual run.

The dated `src/manifests/source_manifest.json` and `package_manifest.json`
describe the original packaged snapshot and are intentionally not rewritten
as if this newer orchestration change had been part of that historical copy.
