# 03/04 A6000 isolated spatial-tile execution

## Scope and invariants
- Preserve production code/configs and every existing H200 artifact.
- Never signal active 02 Outpainting or its descendants. Continue 01 after its PASS commit.
- 05 must retain real completion evidence; do not synthesize DONE or alter the batch journal.
- Remote scope is only 04 and 03. GPU 0/1 = 04 tiles 0–84/85–169; GPU 2/3 = 03 same ranges.
- Source tile 256, overlap 32, global geometry and global seeds unchanged; full 1952-frame causal timeline per tile.
- Use an isolated package, upstream model repository/checkpoints unchanged, no new algorithm.
- Minimum representative compatibility test must PASS before full workers start.

## Live findings 2026-09-26 14:11 UTC
- Parent 1228715 uses Python 3.14.6 and has a parent-only KeyboardInterrupt drain path.
- 05 completed normally at 14:10:04 UTC (DONE, real final/raw metadata present).
- 04 auto-started input-only verification at 14:10:04; only its runner PID 1807436 was SIGSTOPped after identity validation. Its output directory still has only placeholders. No signal to 02.
- 02 PID 2979965 continues; 01 is a future producer-loop job, not a separate running process.
- Remote existing SSH key authenticated; four A6000s idle, 503 GiB host RAM, ~1.9 TiB disk free.

## Execution sequence and acceptance
1. Independently review parent-only drain and interpreter join semantics; write/test durable 02→01 outpaint-only continuation before committing the handoff.
2. Request documented drain only on identity-verified parent; verify drain log and unchanged 02 PID/progress. Retire paused 04 input-only stage only after no-new-job latch is proven. Preserve truthful old journal plus separate migration record.
3. Isolated A6000 package/env/assets/input transfer with file hashes; exact 03↔03 and 04↔04 anchor assertion. Retain original code snapshot and explicit environment/provenance.
4. Add launcher-only global-tile sharding and aggregate 170-tile completion checks; regression tests precede production launch. Preserve neural functions and weighting code.
5. Reuse original spatial composition and audio code; narrow validation to successful tile/encoder counters, nonempty outputs and cheap metadata. Do not repeatedly decode 16K. Fail closed on inconsistency.
6. Representative 29-frame tile smoke across devices; check precision/kernel, range, frames, geometry, no OOM, output ownership. No full-length duplicated validation generation.
7. Start four full-length workers in distinct remote tmux windows; durable coordinator waits for both shards per video then uses global original-layout postprocessing, bounded CPU concurrency.
8. Verify actual PIDs, tmux windows, first outputs, initial VRAM/speed; report launch state, not completion. Record H200 continuation readiness and independent failure domains.

## Ownership
- Leader: live H200 handoff, continuation, remote environment/assets and integration/launch.
- Architect: read-only H200 safety review.
- Tile executor: tools/tile_worker.py, tools/merge_shards.py, tile tests (original upscale.py read-only).
- Composite executor: isolated flash_composite.py and bounded validation tests/tools only; preserve pixel math.

## Risks / stop conditions
- Never assume stale PIDs; validate cmdline, PPID, start ticks immediately before a signal.
- Interrupted producer join must not close the parent while 02 still writes stdout.
- No shared shard manifests or overlapping outputs. A partial shard is not full-stage COMPLETE.
- Disk/RAM/driver/kernel failure blocks remote full launch, never triggers H200 recovery/stop.
- Same A6000 smoke may pass without promising bit-identical H200/A6000 floating-point results.
