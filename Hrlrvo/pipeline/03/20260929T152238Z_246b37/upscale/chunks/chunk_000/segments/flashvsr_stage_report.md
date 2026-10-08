# FlashVSR v1.1 Benchmark Report

- Result: **STAGED**
- Input: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/KOCCA_Verification/Runs/03/20260929T152238Z_246b37/outpaint/final/03_Outpaint_4K.mp4`
- Output: `None`
- Resolution: 3840x2160 -> 15360x8640 (4x)
- Frames / FPS: 21 / 16
- Official continuous causal/KV path: yes
- Source tiling: 320 px, overlap 32 px

## Timing

| Metric | Value |
|---|---:|
| Model initialization / checkpoint loading | 17.159 s |
| Neural inference (DiT) | 79.885 s |
| TCDecoder + color correction | 32.225 s |
| Spatial weighted merge | 0.000 s |
| Video encode / mux | 0.000 s |
| Total wall-clock | 581.045 s |
| Pure inference sec/input frame | 5.338609 |
| End-to-end sec/input frame | 27.668823 |
| Pure inference sec/output frame | 5.338609 |
| End-to-end sec/output frame | 27.668823 |
| Effective inference FPS | 0.151455 |
| Effective end-to-end FPS | 0.036142 |

## GPU

| Metric | Value |
|---|---:|
| GPU | NVIDIA H200 |
| Total VRAM | 143771 MiB |
| Peak torch allocated | 14528 MiB |
| Peak torch reserved | 17278 MiB |
| Peak process memory | 17966 MiB |
| Peak total GPU used | 17975 MiB |
| Average / peak utilization | 22.7% / 95.0% |
