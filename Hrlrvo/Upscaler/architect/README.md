# Hrlrvo 4K → 16K Pretrained Video + Frame Super-Resolution Benchmark

This benchmark runs **official pretrained inference only** for FlashVSR v1.1,
SeedVR2-3B, RealESRGAN_x4plus, and Real_HAT_GAN_SRx4 on the same input. The
last two are explicitly labelled frame-independent baselines rather than silent
fallbacks. It never trains or fine-tunes any model.

## Recommended bounded run

For performance/root-cause work, use the bounded multi-model launcher:

```bash
cd /home/work/ai-graphics_33/Hrlrvo/Upscaler/architect
./run_benchmark_minimal.sh
```

It runs only the meaningful micro/mini temporal units, validates exact 16K
metadata, calculates source-preservation/temporal metrics, and fits 10/100/
1000/1920-frame estimates. FlashVSR and SeedVR2 run serially on the GPU;
Upscale-A-Video is recorded as official-path `BLOCKED_REFERENCE_ONLY` because
its released inference downsamples a 4K input before ×4 and returns 4K, not
16K. Full methodology and PASS criteria are in [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).

```bash
./run_benchmark_minimal.sh --dry-run
./run_benchmark_minimal.sh --models FlashVSR_v1.1 --modes micro
./run_benchmark_minimal.sh --models SeedVR2 --modes micro mini
./run_frame_image_benchmarks.sh
```

## Explicit full-video run (not the default recommendation)

```bash
cd /home/work/ai-graphics_33/Hrlrvo/Upscaler/architect
bash run_test_all.sh
```

`run_test_all.sh` is the explicit legacy/full 92-frame entrypoint. It is not
called by the minimal launcher and should be used only after the estimates and
short output quality have been reviewed.

The default is `PARALLEL_MODE=auto`. Supported overrides are:

```bash
PARALLEL_MODE=parallel bash run_test_all.sh
PARALLEL_MODE=sequential bash run_test_all.sh
```

`PARALLEL_MODE=parallel` requests two concurrent processes but remains subject
to the same measured VRAM/headroom gate; an unsafe request is recorded and
downgraded to sequential execution rather than risking an avoidable OOM.

## Prepared official sources

| Model | Official code | Pinned commit | Official checkpoint | Pinned revision |
|---|---|---|---|---|
| FlashVSR v1.1 | [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR) | `cf910c61a60733e610e9c6e8b607f80c3a6c202b` | [JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1) | `27561b186ded3402d7c975f4fd722e2885b6135f` |
| SeedVR2-3B | [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) | `e4de8c24441a67e1b7df56abea10645059bb1185` | [ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) | `37255ff8cccfb01071b87f635a5948ca8d53117c` |
| RealESRGAN_x4plus | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | `a4abfb2979a7bbff3f69f58f58ae324608821e27` | [official GitHub release](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) | SHA256 `4fa0d389...682f1` |
| Real_HAT_GAN_SRx4 | [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT) | `1638a9a822581657811867bf670717f8371fc3e5` | [official Google Drive folder](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0?usp=sharing) | SHA256 `f5b1e3bb...ffcc7` |

Prepared environments:

- Orchestration/analysis: `/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo`
- FlashVSR: `/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-FlashVSR`
- SeedVR2: `/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-SeedVR2`
- Frame-image SR: `/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-ImageSR`

`run_test_all.sh` invokes the interpreters directly; no manual activation is required.

The frame-image environment, pinned repositories, official weights and hashes
can be reproduced with `./prepare_frame_image_models.sh`.

## Frame-independent baseline policy

- `RealESRGAN_x4plus`: official RRDBNet x4, FP16, tile 512 / halo 10.
- `Real_HAT_GAN_SRx4`: official HAT `params_ema` fidelity checkpoint, FP32,
  tile 512 / halo 32 / attention window 16.
- Both run micro 21 frames and mini 29 frames from source frame 0, matching
  FlashVSR's exact samples. Their mathematical minimum is one frame.
- Both use native neural x4 only. Real-ESRGAN post-Lanczos `outscale`, GFPGAN,
  and any reduced-resolution path are disabled.
- Since there is no temporal state, flicker/texture crawling is evaluated in
  addition to SSIM, PSNR, detail energy, runtime and VRAM.

## Runtime workflow

1. Probe the input with `ffprobe` and record resolution, exact FPS and frame count.
2. Validate pinned repositories, checkpoint hashes, interpreters, GPU and imports.
3. Execute each requested micro/mini model **sequentially on one GPU**. SeedVR2 is
   skipped unless its conservative free-VRAM gate passes; disproved OOM plans are
   not retried.
4. Stream native x4 output directly to the exact-resolution x265 encoder. No
   reduced-resolution or resize fallback is allowed.
5. Verify 15360×8640, requested frame count, 24 fps, codec and pixel format.
6. Calculate common-frame source-preservation and temporal diagnostics, generate
   labelled contact sheets, fit long-video time models, and persist all logs/JSON/MD.
7. Stop after mini. The 92-frame full path is a separate explicit user decision.

The older GPU/CPU overlap scheduler described by `run_test_all.sh` applies only
to the explicit full FlashVSR/SeedVR2 run; it is not used by the bounded minimal
benchmark.

If the installed NVIDIA driver exposes `utilization.gpu`, average and peak
utilization are sampled every 0.5 seconds. On systems where `nvidia-smi`
returns `[Not Found]` for that counter, reports retain all memory peaks and
record utilization as unavailable rather than fabricating a zero value.

## Memory-bounded model adapters

### FlashVSR v1.1

`FlashVSR_v1.1/upscale.py` uses the official `FlashVSRTinyLongPipeline`, official streaming diffusion weight, official causal LQ projector, and official temporal conditional decoder. Each spatial tile uses one uninterrupted causal/KV/LQ/decoder timeline; it is **not** split into independent temporal clips and it never falls back to per-frame image SR.

The stock upstream preprocessor crops output dimensions to a multiple of 128 and may truncate arbitrary frame counts. The adapter instead pads each tile to a supported canvas, runs the official pretrained temporal path, then crops the padding and trims only replicated tail frames. This preserves exact `input_width × 4`, `input_height × 4`, and source frame count.

FlashVSR upstream does not expose DiT spatial tiling. Source-space spatial tiling and feather aggregation are therefore an explicitly benchmark-owned memory adapter. Tile size and overlap are CLI-adjustable; a caught CUDA OOM reduces the tile and retries rather than silently lowering output resolution.

### SeedVR2

`SeedVR2/upscale.py` uses the official `VideoDiffusionInfer`, official 3B EMA DiT, official VAE, official text embeddings, BF16 autocast, and one diffusion step. SeedVR2 accepts a target resolution rather than an inherent fixed scale, so every tile target is derived as exactly four times its source extent.

Upstream SeedVR2 currently loads a whole video and explicitly lists Tile-VAE/progressive aggregation as future work. The benchmark therefore adds clearly reported overlapping `4n+1` temporal windows, cosine temporal blending, source-space spatial tiles, and cosine spatial blending. Every inference unit remains a multi-frame SeedVR2 video inference; no image model or per-frame SR substitution is used. OOM recovery reduces the temporal window first, then the spatial tile.

## Encoder policy

Output is never resized below the expected 4× dimensions. At runtime, candidate encoders are tested with a real one-frame encode at the exact target resolution. The order is NVENC HEVC, NVENC AV1, x265, SVT-AV1, then lossless FFV1/Matroska. Unsupported candidates are recorded. If no installed codec can encode the exact dimensions, the run fails explicitly instead of producing an 8K fallback.

The actual extension may therefore be `.mp4` or `.mkv`. The report contains the selected codec, container, arguments, and probe failures. Audio is stream-copied where the selected container supports it; Matroska is attempted when MP4 cannot carry the original audio codec.

## Entry points and tuning

```bash
/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-FlashVSR/bin/python FlashVSR_v1.1/upscale.py --help
/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-SeedVR2/bin/python SeedVR2/upscale.py --help
```

Both expose `--tile-size`, `--tile-overlap`, `--ffmpeg-threads`, `--encoder`, and probe options. SeedVR2 additionally exposes `--temporal-window` and `--temporal-overlap`. Defaults use live free VRAM and are validated by the pre-run probes; values are not derived from a fixed input frame count or FPS.

## Outputs

```text
../output/benchmark/<model>/*_{micro|mini}_*.mp4
../result/benchmark/<model>/benchmark_{micro|mini}.{json,md}
../result/benchmark/{performance_comparison,quality_comparison}.{json,md}
../result/benchmark/qualitative/*.png
../result/benchmark/{summary,model_ranking,frame_image_comparison}.md

# Explicit legacy/full run only:
../output/test/FlashVSR_v1.1/<input>_FlashVSR_v1.1_x4.{mp4|mkv}
../output/test/SeedVR2/<input>_SeedVR2_x4.{mp4|mkv}

../result/test/FlashVSR_v1.1/{report.md,report.json,run.log}
../result/test/SeedVR2/{report.md,report.json,run.log}
../result/test/{comparison.md,comparison.json,run_all.log}
```

The structure/quality section compares uniformly sampled source frames with Lanczos-downsampled outputs using SSIM, PSNR, mean absolute difference, luminance/color drift, Laplacian variance, and edge energy. These are preservation diagnostics, not claims of subjective 16K quality without ground truth.
