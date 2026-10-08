# FlashVSR v1.1 Benchmark Report

- Result: **STAGED**
- Input: `/home/work/ai-graphics_33/Hrlrvo/pipeline_8K_to_16K/KOCCA_Verification/Runs/05/20260929T170529Z_9837d5/outpaint/final/05_Outpaint_4K.mp4`
- Output: `None`
- Resolution: 3840x2160 -> 15360x8640 (4x)
- Frames / FPS: 21 / 16
- Official continuous causal/KV path: yes
- Source tiling: 320 px, overlap 32 px

## Timing

| Metric | Value |
|---|---:|
| Model initialization / checkpoint loading | 9.112 s |
| Neural inference (DiT) | 78.831 s |
| TCDecoder + color correction | 32.194 s |
| Spatial weighted merge | 0.000 s |
| Video encode / mux | 0.000 s |
| Total wall-clock | 549.873 s |
| Pure inference sec/input frame | 5.286909 |
| End-to-end sec/input frame | 26.184432 |
| Pure inference sec/output frame | 5.286909 |
| End-to-end sec/output frame | 26.184432 |
| Effective inference FPS | 0.151238 |
| Effective end-to-end FPS | 0.038191 |

## GPU

| Metric | Value |
|---|---:|
| GPU | NVIDIA H200 |
| Total VRAM | 143771 MiB |
| Peak torch allocated | 14528 MiB |
| Peak torch reserved | 17278 MiB |
| Peak process memory | 17966 MiB |
| Peak total GPU used | 17975 MiB |
| Average / peak utilization | 23.5% / 96.0% |
