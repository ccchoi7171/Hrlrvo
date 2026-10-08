# A6000 03/04 launch and health report

> Superseded downstream policy/ETA: at2026-09-26 17:52:47UTC the user-authorized [final-only parallel overlay](../optimizations/final_only_v2/DEPLOYMENT.md) replaced the two waiting postprocessing coordinators without restarting any GPU inference. Raw16K generation is now intentionally skipped; both final encodes can overlap. Read that deployment report for current post tmux/status and ETA. Historical launch evidence below remains preserved.

Observed: **2026-09-26 16:57 UTC / 2026-09-27 01:57 KST**.
This is a launch/health report, not a claim that final videos are complete.

## H200 preserved

- 05 completed naturally: real batch `DONE`, final manifest `PASS`, 2026-09-26 14:10:04 UTC. No validation process was killed to manufacture success. Existing raw/final results are retained.
- Original batch PID **1228715** is draining with the documented `finish_running_children_no_new_jobs` interrupt latch. No later H200 upscale can be scheduled by this batch.
- Original 02 Outpainting PID **2979965** remains alive, never signaled. At 16:57:01 UTC it completed **19/41 chunks, 929/1952 frames**, then began chunk index 19 (20th chunk).
- 01-only continuation PID **1939579**, tmux **`hrlrvo_train:outpaint_only_01_20260926`**, remains armed. It waits for the original process identities/locks to clear and 02 PASS evidence, then runs only 01 Outpainting with `--resume`. No upscale path is present.
- Historical batch `04: FAILED_UPSCALE` records intentional pre-inference migration cancellation; it is not an Outpainting failure. This journal was not rewritten. Historical `01: PENDING` and `03: UPSCALE_WAITING` are not the independent continuation/A6000 status sources.
- Production code, configuration, original 03/04 Outpainting outputs, and 05 outputs remain untouched by the isolated A6000 implementation.

## Real A6000 launch

Root: `/home/oem/hrlrvo/experiments/a6000_flashvsr_0304_20260926`

Tmux session: **`flashvsr_0304_20260926`**

Launch: **2026-09-26 16:44:31 UTC / 2026-09-27 01:44:31 KST**.

| GPU | Video | Original global tile range | Window | Supervisor PID | Wrapper PID | Actual GPU inference PID |
|---|---|---|---|---|---|---|
| 0 | 04 | 0–84 | `0:gpu0_v04` | 1870225 | 1870282 | 1870485 |
| 1 | 04 | 85–169 | `1:gpu1_v04` | 1870230 | 1870281 | 1870504 |
| 2 | 03 | 0–84 | `2:gpu2_v03` | 1870235 | 1870283 | 1870324 |
| 3 | 03 | 85–169 | `3:gpu3_v03` | 1870240 | 1870284 | 1870332 |

Postprocessing coordinators: `4:post_v04` PID 1870245 and `5:post_v03` PID 1870250. Both correctly wait for their two complete shards. CPU raw merge/encode and final center compositing are **serialized between videos** by the independent postprocessing lock.

All four workers have committed their first **full 1952-frame tile with PASS**, not merely a smoke-test output. Each has completed 1/85 assigned tiles, approximately 1.18% of tile count. No traceback, CUDA OOM, or runtime exception was found in current worker logs. GPU inference process memory was **13,286 MiB each (~12.97 GiB)**; this is observed initial memory, not a future peak guarantee. Free disk was ~1.9 TiB.

First full tile wall times from launch to committed PASS: 11m22s–11m49s. Neural execution plus PNG staging inside those tiles took 451.6–488.7 seconds (about 4.0–4.3 tile-output frames/s); the remaining time includes initial model/input preparation. These rates are **not full 16K video FPS**. A transient 0% GPU-utilization snapshot is normal during CPU decode or staging; committed frames and continuing logs are the health evidence.

## Identity and compatibility

- Source transfer: **6202 files SHA-verified PASS**.
- FlashVSR source commit: `cf910c61a60733e610e9c6e8b607f80c3a6c202b` (MODEL_INFO provenance plus exact file hashes; the enclosing Hrlrvo Git revision is not misrepresented as the upstream commit).
- Checkpoint revision: `27561b186ded3402d7c975f4fd722e2885b6135f`; weights verified against H200 hashes.
- Exact Python 3.11.13 / torch 2.6.0+cu124 / 102 package versions; same FFmpeg 6.1.1 binary bundle.
- Same Block-Sparse-Attention source `49d6c39e4dc0303442cda3bb758b3925d4399c49`, rebuilt only for Ampere-compatible kernels. Different GPU architectures do not imply bitwise-identical floating-point output.
- Fixed **tile 256 / overlap 32**, bf16, **base seed 71634910 + original global tile index**, sparse ratio 2.0, local range 11. No tile-192 fallback.
- Each tile retains the original uninterrupted causal temporal processing and frame ordering across all 1952 frames. Original full 170-tile geometry and weights are used in the final merge, not per-shard weights.
- 03 Outpainting is paired only with `03_Input_8K.mp4`; 04 only with `04_Input_8K.mp4`. Basename/video-ID checks and input/anchor hashes are enforced.
- All four GPUs passed 29-frame representative tile compatibility tests (global tile 0 or 85), with normal frame ordering, separate directories, correct effective seeds, and no OOM. Gate PASS at 16:44:21 UTC, before full launch.
- 62 local regression tests passed; lint/compile/shell syntax checks passed before code freeze.
- Original 8K-center geometry and 64-pixel inward feather compositing code are reused. Encoder counts, exit codes, manifests and header metadata are checked without repeated full-16K decode validation.

## Logs and outputs

All paths below are relative to the remote root above.

- Detailed workers: `runs/04/gpu0/worker.log`, `runs/04/gpu1/worker.log`, `runs/03/gpu2/worker.log`, `runs/03/gpu3/worker.log`.
- Independent statuses: `status/gpu0.json` through `gpu3.json`, `status/video03.json`, `status/video04.json`, `status/launch.json`.
- Postprocessing logs: `logs/post_v03.log`, `logs/post_v04.log`.
- Raw targets: `runs/03/raw_flashvsr/03_FlashVSR_raw_16K_16fps.mp4` and corresponding `runs/04/.../04_FlashVSR_raw_16K_16fps.mp4`.
- Final targets: `outputs/03/final/03_Certification_Final_16K_16fps.mp4` and corresponding `outputs/04/.../04_Certification_Final_16K_16fps.mp4`.
- Final PASS receipts: `outputs/{03,04}/metadata/final_manifest.json`.
- Compatibility/identity evidence: `verification/{source_transfer_verification,environment_preflight,compatibility_test}.json`.
- Actual execution evidence: `verification/launch_observation.json` and `verification/health_observation_20260926T1657.json`.

## Provisional ETA — explicitly separate phases

All following dates are **KST**; no new benchmark or video full decode was run for this estimate.

- H200 02: recent chunks average 25.25 minutes; 19/41 completed. Accounting for remaining chunks and previous 03's ~26-minute finishing phase gives **September 27 around 11:30 (roughly 11:20–12:00)**.
- H200 01: previous 03 took ~17h34m from start to Outpainting PASS. If 01 is comparable and continuation gates pass, **September 28 around 05:00–06:00**.
- A6000 GPU tile stage, both videos in parallel: extrapolating the first tile across 85 tiles/worker gives ~16–17 hours from launch. Early-data range: **September 27 around 17:00–21:00**. Later tiles and I/O contention can change this.
- **Final videos are not done when GPU tiles finish.** H200 05's measured raw global merge+encode was **58,184.25 s (16h10m)**; final original-center composite+encode was **27,898.20 s (7h45m)**. These are real processing times, not the duplicated validation being removed.
- With the current serial CPU postprocessing and *the same per-video CPU speed as H200 05*, the first final video would finish around **September 28 late afternoon/evening**, and both around **September 29 late afternoon/evening**. The video finishing order is whichever shard pair first acquires the postprocessing lock.
- A6000 host CPU is Xeon Gold 5218R, not the H200 host's Xeon Platinum 8480+. Its full merge/composite rate has not yet been measured. Consequently the final-output ETA is **low-confidence: provisionally September 29–30, possibly later if CPU/I/O is slower**. Do not present the GPU-phase ETA as final completion or promise an exact final hour.

## Frozen deployment

Do not modify active tools/assets casually: future coordinator gates verify pinned identities.

- code manifest SHA256: `5f2a5784c6347db524d5665ada9447cd9dc4f809d7a31d5be46aafe67350fd60`
- runtime FFmpeg manifest: `5ef96a13c4168319933098cbb0e2e5b8ddff0e357c52e09271cf8b69f4d2f17a`
- remote_control.py: `36d55fb5045c274b241f5aad5a3d6f98d0d70d20923f5d251dd2a57ee4683298`
- code bundle: `f2eb4ce8e1fc568de82eac7a191d27938e8e90b1ea2aa79473ef62a877cafa37`

No production job should be stopped merely because this migration/launch task has been reported complete. Both hosts' long-running jobs intentionally continue.
