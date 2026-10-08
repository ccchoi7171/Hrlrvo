# Experiment 2 — clean full anchor + phase-aligned recurrence

This directory owns the second authorized 1,025-frame H200 experiment. It does not modify production code or `test_main/common`.

## Method

1. `prepare_clean_anchor_gpu.py` decodes the actual 960×540×1025 source and performs dense all-frame person segmentation in H200 batches.
2. It registers observations to the common camera coordinate, builds a GPU masked and color-normalized temporal donor, and excludes both dense person masks and the previously observed source-connected upper camera/staff masks from donor statistics.
3. It combines dense people, known upper equipment, and conservative all-frame dark temporal residual equipment evidence. Donor fill and per-frame RGB offset correction run on GPU.
4. The clean registered result is returned to the original per-frame camera pose and encoded losslessly as FFV1. This preserves moving camera/lighting as input evidence. Registration round-trip interpolation is recorded.
5. `run_full_phase_rebase.py` runs the native 65+48 chronology for 1,025 frames. It preserves stock row0, all nine current clean-Anchor rows, and future rows3:9; only prior final BF16 rows7:9 are copied to next exterior rows1:3.
6. The recurrence wrapper rebases transformer time positions locally per chunk, while source reads and Gaussian seeds remain absolute (`33333333 + absolute_start_frame`).

The model's native current Anchor composite uses the **clean input**. There is no post-generation original-RAW center restoration.

## Scheduler tasks

GPU preparation:

```bash
/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python \
  test_main/scheduler.py --task test_main/experiment_2/prepare_task.json
```

The preparation writes `preparation_manifest.json`, `clean_anchor_1025f.mkv`, and a fully SHA-bound `job.json`. After those exist, schedule the full producer:

```bash
/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python \
  test_main/scheduler.py --task test_main/experiment_2/run_task.json
```

The prep requests 24,000 MiB and batch size 8. The long producer requests 50,000 MiB and caps its process at 49% of H200 physical memory. Root remains responsible for aggregate scheduling.

## Resume behavior

The copied native segmented store retains every committed segment and its RNG state. On restart, the runner loads the immediately preceding committed BF16 phase bank. A tail/lineage written before an interrupted segment commit is moved to a timestamped `resume_orphans` folder and recomputed; committed evidence is never overwritten silently.

## Limits

- Mask R-CNN person support and the equipment residual proxy are not foreground ground truth.
- Known upper camera/staff masks are observed temporal evidence, not proof that every wire or reflection was removed.
- Minor wire, reflection, color, or mask residuals are diagnostic and do not stop the authorized full run.
- The phase-aligned carry and temporal rebase are experimental interventions. Completion does not itself establish quality improvement.
