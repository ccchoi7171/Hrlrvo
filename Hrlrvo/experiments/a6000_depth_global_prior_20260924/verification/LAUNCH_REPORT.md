# Verified launch — A6000 GPU0 Depth prior

Verified UTC: 2026-09-23T18:27:12.191632+00:00
State: **running**, NOT a completed output/quality verdict.
Physical GPU0 PID: 1299913. Tmux: `hrlrvo:exp_depth_gpu0` (pane %70).

- All 23 inference modules are the frozen, byte-identical H200 lineart snapshot; the existing Global logits bias/masks/schedule are retained. The only prior map change is normalized D5 relative inverse depth (salience outer product, not same-depth affinity).
- D5_v2_base_518 was extracted on A6000 from its baseline C0 lossless first frame. No H200 reference PNG was reused.
- The original two-slot journal had overwritten the C0 prefix. Isolated original-source C0 replay matched the original segment pixels/file, FP32 prefix+CPU/CUDA RNG, and BF16 carry hashes exactly. Original C0 is preserved; only its missing exact recurrence state was recovered.
- Model load completed; exact saved CPU/CUDA RNG tensor equality passed; first generated chunk is C1 with the depth prior scope active. GPU0 activity remained healthy for the 90-second startup check. Seed71634910, checkpoint7500, strength1.0, carry ON and temporal rebasing ON are pinned.
- GPU1/2/3 original producer PIDs1203845/1206052/1206190 remain running, with no inference source or process modifications.
- Four independent CPU cumulative watchers are active. Counts5/10/15/20/24 produce cumulative C0-based MP4s. GPU1/2/3 first5-chunk previews were fully decoded and validated. Depth waits for C4 to commit.

Changed/new implementation surfaces: `run_depth_chain.py`, `start_depth.sh`, `scripts/prepare_depth_reference.py`, `scripts/bootstrap_and_configure.py`, `scripts/preflight.py`, CPU preview/progress scripts and tests. Existing H200 and baseline inference sources were not edited. Reused the existing LineArtGlobalPrior and progress watcher core rather than adding a second model implementation. No new dependencies.

Validation: 28 depth targeted tests passed locally/remotely; 280 baseline regression tests passed with1 CUDA-only skip against the frozen source;9 cumulative tests passed locally/remotely. Syntax/static checks passed. Runtime evidence: `verification/startup_health.json`, `verification/progress_health.json`, `verification/preflight.json`, `report/c0_recovery_verification.json`.

Remaining risks: Depth C1 has started but is not yet committed; depth quality and per-chunk runtime are not yet measured. Similarity of future results is not promised; this is an experimental attention prior.

Output root: `/home/oem/hrlrvo/experiments/a6000_depth_global_prior_20260924/outputs/strength_1p0_global_depth_prior/`.
- `chunk_previews/C00_full65.mp4` and then `C01_new48.mp4` etc.
- `progress/C00_C04_257f.mp4`, then C09/C14/C19/C23 cumulative milestones.
- `C0_C23_1169f_1920x1080.mp4` after all24 chunks finish.

See `PROGRESS.md` for GPU1/2/3 cumulative paths. Log: `logs/latest.log`.
