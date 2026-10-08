# Final-only parallel postprocessing — deployed

**Takeover PASS: 2026-09-26 17:52:47 UTC / September 27 02:52:47 KST.**
This supersedes the serial raw+final postprocessing and final-output ETA in the earlier launch report. Final production MP4s are not complete yet; GPU tile generation continues.

## What changed / what did not

- Eliminated the **raw 16K intermediate merge/encode**. The production final compositor reads PNG tiles and the 8K anchor directly and never consumes that raw MP4. Manifest records `SKIPPED`, never a fabricated raw PASS.
- Both videos may now run final composition/encoding **concurrently**, using separate per-video state/ownership/output paths. One continuous libx265 encoder per video; no segment encodes or concatenation.
- Frame preparation uses **8 CPU threads per video**, disjoint original vertical bands, identical float32 tile accumulation order and rounding, original center/64px feather helper. Frame-invariant weights/normalization are computed once. PNG reads are bounded to one frame; at most one next composed frame is prefetched while the encoder consumes the current frame. OpenCV inner threading is capped at1 to avoid nested oversubscription.
- Same final codec settings: libx265, fast, CRF14, yuv420p10le, hvc1, pools32. Same audio stream-copy. No NVENC substitution or quality reduction.
- **All four GPU inference processes were left alive, with identical PID/start-time/cmdline.** No tile generation restarted or discarded. Full170 global layout, full1952-frame causal timeline, frame order, tile256/overlap32, bf16, seed71634910+original global index, checkpoints and model code remain unchanged.
- H200 production processes, results, scheduler drain and01-only continuation were not modified.

## Active sessions / PIDs

GPU inference stays in **`flashvsr_0304_20260926`**:

| GPU | Video / global tiles | Inference PID |
|---|---|---|
|0|04 / 0–84|1870485|
|1|04 / 85–169|1870504|
|2|03 / 0–84|1870324|
|3|03 / 85–169|1870332|

New downstream session **`flashvsr_finalonly_0304_20260927`**:

| Window | Pane | Coordinator PID | State after deployment |
|---|---|---|---|
|`0:final_v04`|%79|1985689|WAITING_FOR_TWO_SHARDS|
|`1:final_v03`|%80|1985690|WAITING_FOR_TWO_SHARDS|

Each starts its final-only production path automatically when its two original 85-tile workers finish successfully. Waiting is expected: an entire16K frame needs the full spatial layout. There is no cross-video serial postprocess lock under the selected policy.

Old waiting coordinator PIDs1870245/1870250 were inspected for exact command/PPID/start ticks, WAITING state, no children and no postprocessing artifacts, then retired using PID-safe signals only after the two replacement windows were ready behind a start barrier. No signal was sent to a GPU worker. Original `status/video03.json` and `video04.json` now truthfully route to the new status files as `REPLACED_BY_FINAL_ONLY_V2`.

## Fresh evidence

- **118 regression tests PASS**, Ruff PASS, compile PASS. Independent code reviewer: APPROVE,0 issues.
- Remote baseline source/environment/4-GPU compatibility proof rechecks:03 and04 preflightPASS. Frozen package/tools remain byte/identity unchanged.
- Full15360x8640 representative RGB frame matches immutable production compositor SHA exactly for both correct anchors. Threaded scheduling preserves all four sampled RGB hashes between serial and parallel runs.
- Actual short16K x265 encodes,4frames/video, all exit0,16fps,4 producer/header frames, correct resolution. No full1952-frame duplicate generation or full16K decode was used.
- Sequential optimized short encodes:03 **28.033s**,04 **26.634s**; combined **54.667s**.
- Concurrent group: **31.758s**, **41.9066% less combined wall time**. Individual slowdowns only1.044x/1.068x. This passed the >=10% makespan benefit gate.
- Single full16K CPU fixture: original13.629s, threaded8 8.028s; identical RGB. Initial anchor decoder startup is included, so this is not a steady-state FPS measurement.
- Benchmark limitation: the full global170 geometry repeats available, genuine committed PNG tiles; all170 production tiles do not yet exist. Thus cache/I/O and short codec/GOP behavior do not establish an exact full-video ETA. Recorded explicitly in the signed decision.
- Post-takeover health: all4 GPU workers RUNNING, **6/85 tiles committed each**, seventh tile advancing, no current traceback/OOM/runtime error. GPU process memory14,564MiB each (~14.22GiB). Free disk~1.8TiB.

## Provisional ETA, KST

Recent per-tile medians at takeover were597s/627s/592s/645s on GPUs0/1/2/3. Linear estimates put the slow shard of04 nearSeptember27 **16:36** and03 near**17:00**; allow **16:00–19:00** for remaining tile variability.

The new final path removes the05 baseline's **16h10m raw work per video** altogether. Short dual-encode extrapolation is ~4.3h for1952frames, but startup/content/cache differences limit precision; reserve roughly **4–6h after the tile stage** as a provisional range.

**Both final videos: provisionally September27 20:00 to September28 01:00 KST**, not the previous September29–30 serial/raw estimate. This is an estimate, not a completed-output claim; actual full production postprocessing logs will refine it.

## Remote paths

Root: `/home/oem/hrlrvo/experiments/a6000_flashvsr_0304_20260926`

- New immutable overlay: `optimizations/final_only_v2/`
- New states: `status/final_only_v2/video03.json`, `video04.json`
- New logs: `logs/final_only_v2/video03.log`, `video04.log`
- Unchanged tile logs: `runs/04/gpu{0,1}/worker.log`, `runs/03/gpu{2,3}/worker.log`
- Final outputs remain `outputs/{03,04}/final/{03,04}_Certification_Final_16K_16fps.mp4`
- Final manifests remain `outputs/{03,04}/metadata/final_manifest.json`
- Takeover/health evidence: `verification/final_only_v2_takeover.json`, `final_only_v2_health.json`
- Benchmark decision: `verification/final_only_v2_bench/benchmark_decision.json`

## Frozen v2 pins

- Overlay manifest: `5d5eda9c4076b90341011f827c94744758f67b0b19428cf6cd49c5a8c5224507`
- final_only.py: `ed0bdc01242c7199fc44ac1c3dd078f219bea4911a55daf53958989561ea7d00`
- parallel_renderer.py: `515e1a7a27de1ab320057ba1748cb5ad57819c932093a2f83fc7b5844477e03a`
- coordinate_v2.py: `6edaad4ef1dca3c43adfbe7250f147c553dcad722fb0290665715f708d2ac53f`
- Benchmark decision: `88153b450b9e18f442b14e66b6539c9576bd289155fc5450cdf38c08b337e41f`

Do not casually edit these running overlay scripts or the old frozen assets: both sets are checked at downstream gates. Do not stop either production session when this optimization task is reported complete.
