# CPU cumulative previews (A6000)

The independent watcher adds cumulative previews without changing or restarting inference.

| Completed chunks | Included chunks | Frames | Filename |
| --- | --- | --- | --- |
| 5 | C0–C4 | 257 | `C00_C04_257f.mp4` |
| 10 | C0–C9 | 497 | `C00_C09_497f.mp4` |
| 15 | C0–C14 | 737 | `C00_C14_737f.mp4` |
| 20 | C0–C19 | 977 | `C00_C19_977f.mp4` |
| 24 | C0–C23 | 1169 | `C00_C23_1169f.mp4` |

C0 contributes 65 frames; subsequent chunks contribute only their 48 new frames, not their 17-frame prefix. Every MP4 is 1920×1080 at 24 fps. The existing per-chunk previews and final video are retained.

## Remote directories

- GPU1: `/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/arms/02_no_carry/outputs/progress/`
- GPU2: `/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/arms/03_global_boost/outputs/progress/`
- GPU3: `/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/arms/04_native_halo/outputs/progress/`
- Depth GPU0: `/home/oem/hrlrvo/experiments/a6000_depth_global_prior_20260924/outputs/strength_1p0_global_depth_prior/progress/`

Each directory has `latest.json` and a self-refreshing `index.html`. Output is encoded from committed lossless NPY segments using CPU libx264 (two encoding threads, nice 10), published by atomic rename, then fully decoded to verify frame count, dimensions and rate before a ready event is published. No GPU allocation, inference signals or model-code edits are performed.

## Operation and evidence

Watchers run in `tmux hrlrvo` windows `progress_02_no_carry`, `progress_03_global_boost`, `progress_04_native_halo` and `progress_depth`. The depth watcher waits for its authenticated producer and committed bootstrap C0 before creating output files. Each arm has a nonblocking lock against duplicate watchers. Producer exits before a required milestone cause an explicit watcher error, not partial output presented as complete.

Launcher inventory: `verification/progress_watchers_started.json`.
Fresh runtime/decode verification: `verification/progress_health.json`.
Logs: `logs/progress_<arm>.log`; an adjacent `.exit` file means the watcher exited.
Tests: `tests/test_progress_videos.py`, `tests/test_progress_arm_wrapper.py` (9 passed locally and remotely).

Implementation: `scripts/watch_progress_videos.py` is the unchanged H200 watcher; `scripts/run_progress_for_arm.py` only pins paths, authenticates the producer, enforces CPU isolation and owns the lock.
