# A6000-only final-path acceleration — 2026-09-26

## Authority and unchanged contracts
The user explicitly authorizes removing unnecessary raw/final duplication and beneficial parallelism, and A6000-only restarts if justified by time saved. Preserve H200 entirely. Preserve the four currently running inference workers, completed tile PNGs, seeds, bf16, model/code/checkpoint, full temporal processing, original global 170-tile layout and weights, correct anchors, final 15360x8640/16fps/1952 frames, production center blending, audio, and final encoder quality settings.

## Evidence and chosen direction
- Running GPU inference already uses all four A6000s, two per video. Restarting the same computation loses partial-tile progress and cannot remove the dominant future CPU work.
- Frozen final composition reads tile PNGs directly plus the original 8K anchor; it never reads raw 16K video pixels. The raw merge/encode is an independently retained artifact, not a dependency.
- Existing 05 raw merge/encode took 58184 seconds; final center composition/encode 27898 seconds. Remove the former under the user's new instruction; record SKIPPED, never fake a raw PASS.
- Existing two postprocess coordinators share one CPU lock. Their inputs/outputs are disjoint, so permit bounded parallel final encoding if representative CPU measurements support it.

## Options
1. Keep inference alive, replace only waiting postprocess coordinators with an isolated versioned final-only path. Chosen: large guaranteed work removal, no GPU progress loss.
2. Restart all inference or change tile/temporal/precision. Rejected: no measured benefit, violates preserved quality conditions or loses work.
3. Accelerate frame preparation with bounded CPU threads while reusing exact pixel formulas/order and original encode/audio utilities. Accept only after regression tests show identical pre-encode pixels and a bounded representative measurement shows benefit. Otherwise use the original compositor in the final-only path.

## Steps and ownership
1. Leader captures actual process identities/current progress; read-only architecture review checks takeover risks.
2. Before implementation, add regression tests for raw exclusion, same final helper/codec/anchor/geometry, failures and fresh-output contracts. Implement only under this new versioned directory; frozen package/tools remain unchanged.
3. Build a bounded threaded frame renderer only if needed; lock pixel math with regression tests first. Measure original and candidate on at most a few representative 16K frames, never a full video. No new dependencies.
4. Run targeted/full tests, lint, compile; independent reviewer checks lifecycle, manifest truthfulness, pixel equality and resource bounds.
5. Deploy immutable overlay plus caller-pinned checksums; validate original source/environment/compatibility gates still PASS. Identity-check and retire only the two WAITING_FOR_TWO_SHARDS coordinator PIDs (no children, no encode started), recording takeover intent/completion. Leave all four supervisors/wrappers/inference workers untouched. Start replacement coordinators with per-video ownership locks/status/logs; fail closed, preserve old output targets and no-overwrite rules.
6. Verify original inference PIDs and frame progress continue; both new coordinators wait for valid complete original shard receipts; final-only parallel policy is armed. Report measured savings, updated conditional ETA and tmux paths. Long-running production jobs continue beyond this turn.

## Tests / stop rules
Wrong anchor, incomplete/duplicate/global-renumbered tile sets, changed pinned source, failed worker, encoder failure, existing output, missing frame or unordered frame must fail, not silently fallback. New manifest explicitly states raw omitted by policy and truthful final counter/header/exit evidence. Compare exact RGB output against the production compositor for overlap, borders, feather and center. Verify no full-decode command is introduced. No modification or signals to H200. No casual rewrites of frozen manifests/assets: current workers and future gates depend on their identities.
