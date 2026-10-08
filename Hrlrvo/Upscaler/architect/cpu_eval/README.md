# Existing-output CPU evaluation

This directory contains a read-only evaluator for the ten completed benchmark
videos. It does not import PyTorch, load checkpoints, or invoke an upscaler.

Run from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=void
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python \
  architect/cpu_eval/evaluate_existing.py
```

After an `--inventory-only` direct probe has completed, `--reuse-probes` may be
used to avoid decoding all ten 16K videos a second time. It verifies that each
recorded path still exists and reuses the saved direct-probe metadata.

`--reuse-probes --report-only` regenerates the timing tables and report from the
saved metrics/crop artifacts without decoding the videos again.

The evaluator uses sequential, low-priority CPU `ffprobe`/`ffmpeg` processes and
writes only beneath `result/cpu_quantitative_eval/`. Native 16K videos are
decoded as streams and are never accumulated as a sequence in RAM.
