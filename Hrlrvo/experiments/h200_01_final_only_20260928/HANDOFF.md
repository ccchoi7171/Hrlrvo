# Video 01 H200 final-only handoff

This isolated overlay is the user-authorized replacement for the **redundant raw
16K encoding phase only**. The completed FlashVSR inference stage remains in the
production `data/03_Upscale_16K_15360x8640/01/segments/` directory. No tile,
temporal-state, seed, model, or reference 03/04 result was changed.

## Evidence and transition

- Stage manifest: `state=COMPLETE`, 112 auto-planned spatial tiles, each with
  1,952 committed frames; resolved tile size 320 and overlap 32.
- Input and 8K anchor SHA-256 are verified against the original stage request
  by `run_final_only.sh --preflight-only`.
- `verification/final_only_equivalence.json`: real video01 frame 0, production
  serial tile reconstruction/composite vs copied 03/04 threaded renderer,
  `max_abs_error=0`, including the 64px center-feather boundary.
- The original 01 process group was deliberately sent SIGTERM **after** these
  checks. The detached original watcher consequently recorded `UPSCALE_FAILED`
  (`code=-15`); this is a handoff marker, not an inference/tile failure.
- The incomplete former raw-encode scratch MP4 was preserved at
  `verification/abandoned_partial_raw_encode_01.mp4`. It is **not** a valid raw
  intermediate or the final result.

## Active final-only job

- tmux: `h200_01_final_only_20260928:final`
- status: `status/final_only.json`
- log: `logs/final_only.log`
- output: `../../pipeline_8K_to_16K/data/03_Upscale_16K_15360x8640/01/final/01_Certification_Final_16K_16fps.mp4`
- final manifest: sibling `metadata/final_manifest.json`

The job revalidates the complete stage, reconstructs frames from committed
tile PNGs with the copied 03/04 threaded renderer, inserts the matching
original 8K frame in the center with the unchanged 64px smoothstep helper,
encodes the final 16K MP4 once, and copies original audio. It does **not** load
the GPU model, generate tiles again, or create/probe a raw 16K intermediate.
Its PASS means validated producer receipts, encoder frame counter, and header
metadata; it does not claim a full pixel-decode test.

The production `run_cert.py` still describes the older raw-then-final route;
do not interpret its old watcher status as the authority for this isolated
handoff. The authoritative current status is `status/final_only.json` plus the
final manifest. Do not run the old stage concurrently with this job.
