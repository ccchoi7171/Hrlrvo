# Concurrent writes observed outside this experiment

This experiment's commands only wrote under `/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_recurrence_test_20260923`. `report/original_git_status_before.txt` was empty at start. At 2026-09-23 14:22 UTC, read-only `git status` showed new untracked files under `pipeline/` (for example `run_cert.py`, `DIRECTORY_LAYOUT.md`, `src/cert_pipeline/`). Their filesystem mtimes are during the experiment. They were not produced by this experiment's commands and are not altered or removed here. Other Codex/tmux sessions were active concurrently; exact writer not established.

The three individually protected files captured in `report/protected_stat_before.txt` still matched their original size, mtime, and mode at 14:22 UTC:
- `pipeline/data/00_Input_8K_7680x4320/05_Input_8K.mp4`
- `pipeline/data/02_VideoOutPaint_4K_3840x2160/05_VideoOutPaint_4K.mp4`
- `VideoOutpainting/main/src/ltx_outpaint/inference.py`

Consequently, report `production files changed by this experiment: NO`, but do not report the entire pipeline tree as globally unchanged. Do not clean up external untracked files.
