# FlashVSR v1.1 Benchmark Report

- Result: **STAGED**
- Input: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/KOCCA_Verification/Runs/03/20260929T165136Z_65d2d7/outpaint/final/03_Outpaint_4K.mp4`
- Output: `None`
- Resolution: 3840x2160 -> 15360x8640 (4x)
- Frames / FPS: 21 / 16
- Official continuous causal/KV path: yes
- Source tiling: 320 px, overlap 32 px

## Timing

| Metric | Value |
|---|---:|
| Model initialization / checkpoint loading | 9.384 s |
| Neural inference (DiT) | 78.887 s |
| TCDecoder + color correction | 32.170 s |
| Spatial weighted merge | 0.000 s |
| Video encode / mux | 0.000 s |
| Total wall-clock | 593.778 s |
| Pure inference sec/input frame | 5.288427 |
| End-to-end sec/input frame | 28.275140 |
| Pure inference sec/output frame | 5.288427 |
| End-to-end sec/output frame | 28.275140 |
| Effective inference FPS | 0.148423 |
| Effective end-to-end FPS | 0.035367 |

## GPU

| Metric | Value |
|---|---:|
| GPU | NVIDIA H200 |
| Total VRAM | 143771 MiB |
| Peak torch allocated | 14528 MiB |
| Peak torch reserved | 17278 MiB |
| Peak process memory | 17966 MiB |
| Peak total GPU used | 17975 MiB |
| Average / peak utilization | 21.9% / 95.0% |
