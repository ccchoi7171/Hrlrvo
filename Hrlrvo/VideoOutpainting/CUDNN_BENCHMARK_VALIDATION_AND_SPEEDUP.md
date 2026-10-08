# cuDNN Benchmark Validation and Further Speed Optimization

Date: 2026-09-12 (UTC). Project: `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting`.
Environment: `Hrlrvo`; GPU: `CUDA_VISIBLE_DEVICES=0`, NVIDIA H200.
Evidence root: `outputs/cudnn_benchmark_validation/20260912_0637/`.
**Production has NOT been restarted. Existing run, checkpoints, cache/data and `hrlrvo_train` tmux are preserved.**

## 1. Executive Summary

**cuDNN startup-autotuning validation: PASS.** The only production edit is one assignment at the beginning of `ltx_outpaint.train.main()`. Five actual full-48-block, 4K65f optimizer steps completed. Cold step: **249.818 s**; warm steps 2–5: **54.609 s mean / 54.650 s median**, with **38.782 s backward-only**. The previous 350–380 s backward pathology is not present in warmed measurements.

Peak allocated/reserved: **129.253/135.963 GiB**. No OOM, but physical free after warm steps is only about 3.07 GiB: this is a validated isolated workload, not generous multi-workload headroom. All ten trainable groups have finite, nonzero gradients; frozen gradient violations are zero; all ten representative trainable tensors update.

R4 changed from 4,096 `convolveNd_wgrad_engine` launches to two cuDNN xmma wgrad kernels. All five observed R2 shape families have zero pathological group-wise launches after autotuning. Some R2 sizes select one efficient grouped-direct kernel rather than xmma; the evidence does **not** claim every autotuned shape uses the same engine.

The new cost is distributed: checkpoint replay is **15.521 s** (about 40.0% of backward), with Adaptive preparation/restoration, repeated elementwise/copy work, native projections and shared FFN still material. There is no observed MATH-SDPA fallback. Additional cache/checkpoint experiments below are diagnostic prototypes, **not production changes**.

## 2. Production Change

File/symbol: `main/src/ltx_outpaint/train.py:967–970`, `main()`. Exact diff:

```diff
--- train.py.before
+++ train.py
@@ -965,6 +965,7 @@
 
 
 def main() -> None:
+    torch.backends.cudnn.benchmark = True
     parser = argparse.ArgumentParser(description=__doc__)
     parser.add_argument("config", type=Path)
     parser.add_argument("--disable-progress-bars", action="store_true")
```

This runs before argument parsing, `load_training_config`, `prepare_run_artifacts`, cache audit, `OutpaintingTrainer` construction, model loading and every model forward. `torch` was already imported. Importing this module for inference does not execute `main()`. No pinned LTX file or config changed; no observer, per-step logging, CUDA synchronization, layout conversion or profiler was added to production.

The diagnostic invokes the actual `train.main()` and intercepts the first `torch.nn.functional.conv2d/conv3d` calls. Both recorded `benchmark=True`, `deterministic=False`, and deterministic-algorithms disabled. First Conv2D input was `[2844,4096,2,2]`; first Conv3D input was `[135,4096,9,2,2]`, both BF16. See `production_benchmark/startup.json` and final `result.json` (the startup snapshot itself intentionally says RUNNING).

Why fresh process matters: PyTorch's cuDNN-v8 convolution implementation looks up a cached plan before choosing its heuristic/benchmark branch; the cache key does not distinguish the benchmark flag. A late flag switch can therefore reuse an old plan. [PyTorch 2.8 cache key](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cudnn/Conv_v8.cpp#L168-L218), [lookup and execution](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cudnn/Conv_v8.cpp#L974-L1068).

## 3. Static Correctness Verification

| Check | Result / evidence |
| --- | --- |
| Production source/config integrity | PASS: 335-file pre-change SHA256 snapshot; only train.py differs, by the one line above |
| Python compile/import and YAML parse | PASS: static_verification.json; imports do not initialize CUDA |
| Actual startup ordering | PASS: first main AST statement; CPU --help executes it; actual first-convolution hook confirms it |
| Regression | 13 CPU tests passed: main/tests/test_training_runtime.py; cpu_regression.log |
| Ruff / final integrity | See final_integrity.json and final_checks.log; no production semantic edits besides the flag |
| Backend flag overwrite | No later False assignment in active Hrlrvo/pinned trainer path |
| Determinism | No active deterministic=True/use_deterministic_algorithms(True)/CUBLAS_WORKSPACE_CONFIG policy; none changed |
| Accelerate seed | Installed set_seed defaults deterministic=False; only explicit True calls torch.use_deterministic_algorithms |
| Workers | CPU data loading does not mutate parent-process cuDNN settings; no worker configuration change |
| Custom monitoring | --diagnostics remains false; no train_metrics.jsonl or observer reinstated |
| Checkpoint semantics | Save/resume implementation and checkpoint configuration byte-identical; not a new save/resume execution test |
| Protected resources | Production run, pinned source and configuration hashes retained; probes create no Anchor entries |

Runtime library versions: PyTorch `2.8.0+cu128`, CUDA build `12.8`, runtime cuDNN integer `91002`, NVIDIA driver `580.126.20`. The shell's `CUDNN_VERSION=9.8.0.87` is not the cuDNN version PyTorch actually reports. `benchmark_limit=10`, cuDNN enabled, allow_tf32=True were left unchanged. Existing `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, `CUDA_CACHE_DISABLE=1`, and `CUDA_MODULE_LOADING=LAZY` were not modified. Probes use the user's production OMP/MKL thread count 8.

`benchmark=True` chooses among measured convolution algorithms; it does not change the Conv3D definition or disable deterministic policy. Algorithm selection/reduction order can vary across process, hardware and shape; bitwise reproducibility is not promised. [Backend flag documentation](https://docs.pytorch.org/docs/2.8/backends.html#torch.backends.cudnn.benchmark), [reproducibility notes](https://docs.pytorch.org/docs/2.8/notes/randomness.html#cuda-convolution-benchmarking).

| Production property | Verified value (unchanged) |
| --- | --- |
| Architecture | v8; 48 actual LTX blocks; existing Native/Adaptive/Global and hierarchical fusion |
| Geometry | 3840×2176 internal / 3840×2160 visible; GT [1,128,9,68,120]; Anchor [1,128,9,34,60] |
| Temporal | FIRST 65 RGB frames; 9 latent frames; prefix 17 RGB / 3 latent, probability 0.5 |
| Batch | microbatch=1, accumulation=1, world_size=1 |
| Precision / checkpoint | BF16; all 48 blocks checkpointed, use_reentrant=False |
| Optimizer / scheduler | Actual AdamW; all groups base LR 1e-4; LinearLR 1.0→0.1 over 30,000 steps |
| Clipping | Actual Accelerator.clip_grad_norm_, max_grad_norm=1.0 |
| Sampling / loss | Unchanged shifted-logit-normal; masked final flow-matching MSE; known Anchor OR prefix excluded |
| Seed / checkpoint interval | 33333333; 500 optimizer steps |
| Counts | Trainable 280,363,008; frozen 18,988,838,144; installed 19,269,201,152 |

## 4. Cold vs Warm Training Timing

Method: fresh process; actual production entrypoint/model/`OutpaintingTrainer._training_step`, actual AdamW/LinearLR/Accelerator backward and clipping. Only the outer `run_training` callback is replaced in memory with a bounded fixed-record diagnostic loop. Real `OutpaintingCacheDataset` and collator are used; parallel DataLoader scheduling and the native console/progress loop are not benchmarked. Stage boundaries synchronize CUDA **only in the diagnostic harness**. Gradient/snapshot audits occur outside reported phase sums. These are real updates, not forward-only or a toy model; no production checkpoint is written. The measured phases still include bounded CUDA-event scopes and R-shape hooks; they are conservative diagnostic timings, not zero-overhead production throughput.

| Step | Kind | Forward s | Backward only s | Clip s | Adam s | Phase sum s | Loss | Sigma | Grad norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cold/autotune | 31.678260 | 216.443203 | 0.043497 | 0.290649 | 249.818253 | 0.344488770 | 0.924650311 | 0.196922064 |
| 2 | warm | 15.742476 | 38.787661 | 0.016115 | 0.029688 | 54.651771 | 0.335164279 | 0.924650311 | 0.189505950 |
| 3 | warm | 15.604614 | 38.816693 | 0.016022 | 0.028427 | 54.651027 | 0.258476973 | 0.954690933 | 0.052009679 |
| 4 | warm | 15.722846 | 38.743969 | 0.016311 | 0.027978 | 54.649891 | 0.417675972 | 0.975600958 | 0.145877019 |
| 5 | warm | 15.581465 | 38.779000 | 0.016437 | 0.028401 | 54.483194 | 0.287171245 | 0.966838479 | 0.110223606 |

Warm mean/median/min/max: **54.608971 / 54.650459 / 54.483194 / 54.651771 s**. Mean forward **15.662850**, mean backward **38.781831 s**. Backward/forward is now **2.48×**, not 12–15×.

| Other measured phase | Warm mean seconds |
| --- | --- |
| real_CPU_data_load | 0.082892 |
| anchor_cache_completion | 0.004195 |
| H2D | 0.004967 |
| empty_connector | 0.020496 |
| conditioning_noise_mask | 0.002596 |
| loss | 0.000547 |
| clip_grad_norm_only | 0.016221 |
| optimizer_step | 0.028624 |
| zero_grad | 0.003586 |
| scheduler_step | 0.000166 |

Cold phase sum exceeds warm mean by **195.209 s**. This is a combined cold/algorithm-search/first-Adam-allocation cost, not a pure measurement of search alone. Initial model/config/cache setup was additionally 108.079 s. Warm diagnostic wall times including health work were 54.978, 54.895, 55.019, 54.811 s. Four synchronized warm samples are encouraging, not a 30,000-step statistical throughput guarantee.

Fresh pre-change control: F=24.645 s, B=376.793 s, phase sum=403.093 s. This agrees with the earlier production 350–380 s backward. Do not interpret cold-False versus warm-True as a rigorously identical warm-state A/B; first-step state and cold costs differ. The dispatch microtraces independently establish the pathological-kernel replacement.

## 5. Multi-Sample / Multi-Shape Validation

| Step | Actual matching relative record ID | Prefix mode | Known tokens | Total s |
| --- | --- | --- | --- | --- |
| 1 | 0002e965-b236-4ec5-846d-4d1b919cf3f0.pt | 17f prefix | 36720 | 249.818 |
| 2 | 0002e965-b236-4ec5-846d-4d1b919cf3f0.pt | 17f prefix | 36720 | 54.652 |
| 3 | 0003bf1d-b0ad-4b86-a902-3afc92dd7408.pt | prefix-free | 18360 | 54.651 |
| 4 | 00070a93-e947-4b87-8248-d5321fa4f89a.pt | prefix-free | 18360 | 54.650 |
| 5 | 0002e965-b236-4ec5-846d-4d1b919cf3f0.pt | 17f prefix | 36720 | 54.483 |

All three records use the production-fixed Anchor rectangle y=[17:51), x=[30:90) and the same 68×120 canonical canvas. There are **not three different spatial Anchor geometries** in this test: inventing them would change the requested production contract. Instead every real sample exercises both K4/K5 partitions and all five R2 query-count shapes, while the three records and two prefix states test reuse across content/noise/conditioning. Step 2 resets seed to match the earlier comparison draw but retains step-1 updated parameters and live Adam state; steps 3–5 continue the RNG. Thus this is controlled sample selection, not a shuffled production-epoch benchmark.

| Refinement input N,C,T,H,W | Occurrences per original forward | Occurrences during replay | Observed reuse |
| --- | --- | --- | --- |
| R2 [135,4096,9,2,2] | 288 | 288 | K4; same in all 5 steps |
| R2 [84,4096,9,2,2] | 384 | 384 | K5; same in all 5 steps |
| R2 [39,4096,9,2,2] | 96 | 96 | K5 |
| R2 [24,4096,9,2,2] | 48 | 48 | K5 |
| R2 [21,4096,9,2,2] | 48 | 48 | K5 |
| R4 [1,4096,9,68,120] | 48 | 48 | All blocks |

All shape/stride/dtype observations are in each `step_*/R_shapes.json`. No warmed-step cold spike appears when record or prefix mode changes. This demonstrates process-local plan reuse for the tested production family, not direct introspection of cuDNN cache-hit counters. Unseen batch/shape/layout/device/dtype would need its own search; none is silently padded or reduced here.

## 6. R2/R4 Kernel Dispatch Verification

Bounded fresh-False/fresh-True operator probes use the actual unchanged `SpatialDecompressor(4096)`, original strides, BF16, full temporal extent 9, and spatial R2/R4 shapes. These synthetic operator-level VJPs are separate from the real-cache full training test. Each has a cold run, three unprofiled warm runs, then a bounded trace. Reported wgrad time is a CUDA interval **union**, not a sum of concurrent group streams.

| Case | False storm kernels | True storm kernels | False dW union ms | True dW union ms | True selected dW engine | Warm True full R backward ms |
| --- | --- | --- | --- | --- | --- | --- |
| R4 | 4096 | 0 | 5861.570 | 11.151 | 2 × xmma (main + split-K) | 59.738 |
| R2_q135 | 4096 | 0 | 110.563 | 1.083 | 2 × xmma (main + split-K) | 4.571 |
| R2_q21 | 4096 | 0 | 62.931 | 0.320 | 1 × grouped-direct | 0.919 |
| R2_q24 | 4096 | 0 | 63.155 | 0.344 | 1 × grouped-direct | 1.098 |
| R2_q39 | 0 | 0 | 0.654 | 0.564 | 1 × grouped-direct | 1.524 |
| R2_q84 | 0 | 0 | 0.848 | 1.177 | 1 × grouped-direct | 3.027 |

Exact names include:
- Old: `cudnn::cnn::convolveNd_wgrad_engine<__nv_bfloat16,3,...>` ×4096.
- R4/R2-q135: `sm80_xmma_wgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_..._execute_kernel__5x_cudnn` + split-K.
- Some smaller R2: `cudnn::cnn::wgrad2d_grouped_direct_kernel<true,true,int,__nv_bfloat16,float,float>` ×1.

The grouped-direct name does not imply one kernel per group; it handles the operation in a single launch. R2-q84 already had a fast default xmma plan; its true dW-only probe is not faster in isolation, while total R backward still is slightly faster. Autotuning need not improve every subkernel to eliminate the pathological family.

The independent **warmed full48 trace** contains no `convolveNd_wgrad_engine` kernel at all. All wgrad-name kernel interval union is **1.847 s**, including compressors, not just R2/R4. Raw evidence: `kernels_false/`, `kernels_true/`, `warmed_profile/step_02/trace.json` and `raw_summary.json`.

## 7. Gradient / Numerical Validation

| Group | Parameters | Grad norm min over 5 | Grad norm max over 5 | Missing / nonfinite | Representative updated |
| --- | --- | --- | --- | --- | --- |
| op_lora | 50,331,648 | 0.00727929 | 0.05215190 | 0 / 0 | 5/5 |
| a_lora | 75,497,472 | 0.00243377 | 0.01507261 | 0 / 0 | 5/5 |
| g_lora | 75,497,472 | 0.00484880 | 0.03058438 | 0 / 0 | 5/5 |
| f_lora | 62,914,560 | 0.00818478 | 0.04978944 | 0 / 0 | 5/5 |
| c2 | 983,040 | 0.02036683 | 0.05478522 | 0 / 0 | 5/5 |
| c4_shared | 3,342,336 | 0.04457313 | 0.14247592 | 0 / 0 | 5/5 |
| r2 | 5,505,024 | 0.00405279 | 0.02869511 | 0 / 0 | 5/5 |
| r4 | 5,505,024 | 0.01170999 | 0.09911616 | 0 / 0 | 5/5 |
| local_fusion | 393,216 | 0.00041094 | 0.00259830 | 0 / 0 | 5/5 |
| global_fusion | 393,216 | 0.00055233 | 0.00474678 | 0 / 0 | 5/5 |

All 1,728 trainable tensors had gradients; zero-B LoRA A gradients may individually be zero, which is not treated as a disconnect. All five losses/sigmas finite; frozen Transformer gradient violations zero. Four representative frozen SA-Q/O/Text-CA/FFN slices were unchanged after every step. This is **not a bytewise comparison of every 19B frozen weight**; full frozen-gradient enumeration and unchanged optimizer membership complement representative value comparisons. The empty connector remains frozen; no text/VAE training or cache encoding is performed.

Actual clipping executes with max_grad_norm=1.0. All measured preclip norms are below 1, so these steps exercise the clipping path but not a norm>1 rescale. Adam states materialized; optimizer state tensor elements after five steps: 560,727,744 = 2 × 280,363,008 + 1,728 scalar step counters. LR after step1→5: 9.9997e-05, 9.9994e-05, 9.9991e-05, 9.9988e-05, 9.9985e-05. The production save/resume format is unchanged and covered by CPU regression/static integrity; this task did not add another actual save/resume round trip.

Fresh False/True step1 same input SHA256: `81a74ca4c1df1ed2470c2a2e9a630283f1e143e996f3912c095fb16bfd3504ac`; same record, seed, sigma and ten representative initial weight hashes. The baseline main is compiled in memory with only the new assignment omitted, not toggled after convolutions. Loss is **exactly 0.344488769770** in both processes; global norm False/True: **0.196897879243/0.196922063828**. The largest ten-group **norm** relative difference is 0.03363%.

| Selected full dW | False vs True relative L2 | Max absolute difference | Finite |
| --- | --- | --- | --- |
| b0.R2.weight | 1.462973% | 7.6293945e-06 | True |
| b0.R4.weight | 1.441420% | 9.5367432e-06 | True |
| b1.R2.weight | 0.680249% | 7.6293945e-06 | True |
| b1.R4.weight | 0.370411% | 7.6293945e-06 | True |
| b47.R2.weight | 0.123230% | 4.7683716e-07 | True |
| b47.R4.weight | 0.025587% | 9.5367432e-07 | True |

**Do not confuse similar gradient norms with identical vectors.** Early-block selected R2/R4 dW differ by up to about 1.46% relative L2, while late-block differences are smaller. Full trainable tensors/activations were not all snapshotted. Different BF16 convolution reduction order and propagated upstream differences are permitted here; no universal BF16 tolerance or long-run quality equivalence is asserted.

| Direct operator dW vs CPU FP64 (8 selected channels) | False relative L2 | True relative L2 |
| --- | --- | --- |
| R4 | 0.00173752 | 0.00173797 |
| R2_q135 | 0.01245502 | 0.00168876 |
| R2_q21 | 0.00644785 | 0.00151480 |
| R2_q24 | 0.00577022 | 0.00155202 |
| R2_q39 | 0.00171022 | 0.00171022 |
| R2_q84 | 0.00170869 | 0.00170869 |

These channel-wise FP64 references check the exact operator on the same rounded BF16 input/upstream gradient, not end-to-end training quality. True errors are approximately 0.15–0.17%; R4 is essentially unchanged versus False and pathological R2 cases improve. Together with five-step finite health, this supports numerical sanity, **not bitwise reproducibility**.

## 8. Memory Usage

| Step | Peak allocated GiB | Peak reserved GiB | End physical free GiB |
| --- | --- | --- | --- |
| 1 | 127.303927 | 135.947266 | 3.087036 |
| 2 | 129.253167 | 135.960938 | 3.073364 |
| 3 | 129.253167 | 135.960938 | 3.073303 |
| 4 | 129.253167 | 135.960938 | 3.073303 |
| 5 | 129.253167 | 135.962891 | 3.071350 |

NVML total 143,771 MiB (140.401 GiB); `torch.cuda.mem_get_info` total 139.801 GiB. These APIs expose slightly different totals; the report does not confuse the marketed 141 GB class with GiB. GPU initially had no compute process. Warm live allocated peak leaves ~10.55 GiB relative to PyTorch total; reserved peak leaves ~3.84 GiB in that same accounting, and actual free is ~3.07 GiB after non-allocator overhead. Reserved includes reusable cache, not all live tensors. No OOM occurred, but concurrent jobs and additional long-lived activation/PE caches require explicit revalidation.

## 9. New Warmed Bottleneck Breakdown

Unprofiled warm timing is the baseline (54.609 s), not the slower Kineto run. The separately profiled warm step measured F=18.623, B=42.114, total=60.872 s. Its 8,577,761 trace events / 726,130 kernel launches create observable overhead. Record-shapes/stack/memory profiling were disabled to bound that overhead; peak memory comes from allocator counters instead.

### Unprofiled CUDA-event scopes (mean of warm steps 2–5)

| Scope | Original forward s | Backward replay s |
| --- | --- | --- |
| Native (inclusive) | 2.432 | 2.392 |
| Adaptive (inclusive) | 6.733 | 6.741 |
| Global (inclusive) | 1.814 | 1.804 |
| Adaptive/C2 | 1.132 | 0.757 |
| Adaptive/C4 | 0.192 | 0.192 |
| Adaptive/R2 | 1.884 | 1.849 |
| Global/C4 | 0.129 | 0.128 |
| Global/R4 | 1.475 | 1.467 |
| SharedO/O | 0.250 | 0.250 |
| TextCA/TextCA | 0.700 | 0.699 |
| FFN/FFN | 1.759 | 1.761 |
| LocalFusion/LocalFusion | 0.163 | 0.161 |
| GlobalFusion/GlobalFusion | 0.118 | 0.117 |
| Shared/block | 15.620 | 15.521 |

All 48 blocks replay **once** in each warm step; no nested/window-level checkpoint or accidental multiple replay was observed. Replay 15.521 s is about 40.0% of backward. B wall minus block replay interval sum is ~23.261 s; that subtraction is a useful bound, not an exclusive kernel attribution. Parent/child event scopes overlap and must NOT be added together.

### Warm trace: nearest-scope CUDA kernel time

| Scope | Forward kernel sum s | Replay kernel sum s | Derivative kernel sum s |
| --- | --- | --- | --- |
| Native | 2.208 | 2.178 | 4.419 |
| Adaptive | 3.423 | 3.370 | 6.037 |
| Global | 1.722 | 1.713 | 3.507 |
| SharedO | 0.251 | 0.250 | 0.268 |
| TextCA | 0.704 | 0.702 | 1.257 |
| FFN | 1.770 | 1.765 | 1.865 |
| LocalFusion | 0.140 | 0.138 | 0.043 |
| GlobalFusion | 0.120 | 0.120 | 0.133 |
| Shared | 1.616 | 1.540 | 1.673 |
| UNATTRIBUTED | 0.042 | 0.000 | 1.148 |

Attribution uses CUDA external IDs → nearest CPU operator/scope, plus only unambiguous original-forward autograd sequence numbers for derivatives. There are 3,527 ambiguous forward sequence keys; ~1.148 s of derivative kernels remain explicitly unattributed. Kernel sums can overlap across streams and are not exclusive branch wall times. Whole-trace kernel interval unions: F=11.996, B=32.126 s. The difference from wall time also includes launch gaps, CPU work, profiler overhead and memcpy/memset categories omitted from this kernel-only union; it is not all removable Python overhead.

Adaptive is the largest measured branch grouping after the fix. R2/R4 restoration still has interpolation/layout/dgrad work, with derivative kernel sums R2=2.574 s and R4=2.819 s, but it is no longer a hundreds-of-seconds wgrad storm. Copies/casts and elementwise work are now major operator families. Only geometry/position-independent-of-hidden work is a low-risk cache target; the many hidden-dependent copies and BF16/FP32 casts cannot simply be deleted.

### Actual attention backend

All observed Native (984 calls/pass), Adaptive (864), Global (48), and Text CA (48) use cuDNN SDPA: 1,944 original forward calls, 1,944 replay calls, 1,944 backward calls. Profiler operators `_cudnn_attention_forward` and `_scaled_dot_product_cudnn_attention_backward`, with `cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_...` kernels, establish dispatch. The internal kernel name containing `flash`/`f16` does not imply the external flash-attn package or a changed FP16 policy. No observed MATH/EFFICIENT fallback; installing FlashAttention is not evidence-based treatment for this measured remaining bottleneck.

### Top 35 operators in the warmed trace

| Operator | Calls | Direct CUDA kernel sum s | CPU self s |
| --- | --- | --- | --- |
| aten::copy_ | 210,502 | 5.682955 | 18.427981 |
| aten::mul | 144,405 | 4.706558 | 4.624255 |
| aten::addmm | 12,247 | 4.608860 | 0.766214 |
| aten::convolution_backward | 2,352 | 4.493497 | 0.364492 |
| aten::cudnn_convolution | 4,704 | 4.198768 | 0.214549 |
| aten::mm | 74,473 | 3.228340 | 5.385416 |
| aten::_upsample_bilinear2d_aa | 5,472 | 2.263444 | 0.055983 |
| aten::add_ | 69,608 | 2.113689 | 2.795335 |
| aten::add | 36,194 | 1.767218 | 0.612323 |
| aten::fill_ | 45,186 | 1.763616 | 1.762273 |
| aten::_scaled_dot_product_cudnn_attention_backward | 1,944 | 1.742591 | 0.355773 |
| aten::_upsample_bilinear2d_aa_backward | 912 | 1.683123 | 0.039295 |
| aten::_cudnn_attention_forward | 3,888 | 1.264809 | 0.314335 |
| aten::index | 2,880 | 0.506520 | 0.127940 |
| aten::addcmul_ | 15,168 | 0.450999 | 0.352863 |
| aten::pow | 24,096 | 0.387138 | 0.777748 |
| aten::div | 15,420 | 0.377931 | 0.838763 |
| aten::index_add_ | 4,800 | 0.377888 | 0.080211 |
| aten::gather | 13,896 | 0.350882 | 0.247888 |
| aten::sum | 8,494 | 0.330738 | 0.499864 |
| aten::mean | 8,068 | 0.258104 | 0.182024 |
| aten::index_copy | 6,240 | 0.207626 | 0.204018 |
| aten::where | 5,904 | 0.183978 | 0.256135 |
| aten::addcmul | 4,128 | 0.183855 | 0.078839 |
| aten::gelu | 96 | 0.131458 | 0.003521 |
| aten::_index_put_impl_ | 1,423 | 0.120372 | 0.141276 |
| aten::sigmoid_ | 2,064 | 0.111079 | 0.049283 |
| aten::lerp | 2,064 | 0.109257 | 0.074505 |
| aten::gelu_backward | 48 | 0.090425 | 0.001393 |
| aten::sigmoid_backward | 1,127 | 0.082937 | 0.058602 |
| aten::neg | 7,584 | 0.073483 | 0.124446 |
| aten::index_select | 18,504 | 0.057254 | 0.267885 |
| aten::cat | 5,479 | 0.053095 | 0.101861 |
| aten::sub | 8,235 | 0.049041 | 0.134049 |
| aten::sub_ | 912 | 0.036341 | 0.055134 |

CPU self and CUDA columns overlap asynchronously and are not additive. For example copy_ CPU time can include synchronization/waiting; 5.68 s of copy CUDA kernels is not a promise that removing a metadata allocation saves 5.68 s. `raw_summary.json` also contains top kernels, CPU operators, backend attribution and exact call counts.

## 10. Remaining Optimization Candidates

The current code already LRU-caches Python Local windows, Adaptive layouts and Global cells (`geometry.py::local_windows/adaptive_layout/global_cells`). Caching those functions again is not a new optimization. Candidate work is device tensor materialization/validation and constant positional preparation inside repeated block/view calls.

| Candidate | Current code / cost targeted | Bounded experiment / disposition |
| --- | --- | --- |
| Immutable device metadata plan | compression.py::SpatialCompressor.forward cell validation/index tensors; Adaptive qi/locations/write indices; Native and assembly indices; ViewContext.subset B=1 copies | metadata_cache A/B below; no hidden or trainable tensor cached |
| Mapped Adaptive PMA positions | position.py::target_query_positions; coordinates independent of current hidden and weights | metadata+mapped-position A/B; call-scoped, exact context positions key |
| Final Adaptive/Global PE cache | target_query_rope/phase_rope repeated precompute_freqs_cis in forward/replay | metadata+final-PE A/B; alternative to mapped-position-only; higher memory |
| Selective checkpoint | 15.52 s replay over 48 blocks | Exact spatial/temporal single-block CP ON/OFF below; whole-model 4/8 OFF not blindly attempted |
| Hidden layout/copy changes | copy/cast, interpolation, convolution tensor transforms | Not changed; constants are the safe subset. R2/R4 NCDHW explicitly excluded from production |
| Window batching/vectorization | 984 Native +864 Adaptive attention calls/pass, variable bucket shapes | Static feasibility only; no measured full-step speedup; requires VRAM/reduction-order audit |
| torch.compile | Potential Python/launch work | Current model.py::_forward_with_native_gate explicitly rejects torch.compiler.is_compiling(); not applied or advertised as ready |
| CUDA Graph | Stable finite shape family, many small launches | Not established: host ContextVar/control, checkpoint capture and memory-pool lifetime need work. Dynamic record alone is not a sufficient reason to reject |
| Optimizer/data | Adam 0.029 s; read/H2D small | Not a worthwhile target; unchanged |

Cache safety boundary: cache only detached immutable indices/masks/coordinates/PE, never current hidden, packed tokens, QKV, compressor/restored outputs, gates, loss or gradients. A correct key includes actual position/geometry, device, dtype, scale and RoPE settings; prefix/time/fps or batch changes must invalidate it. Build eagerly before checkpoint entry, then keep the same cache-hit path in original forward and replay. Do not mutate a cache for the first time during replay and violate non-reentrant saved-tensor equivalence.

## 11. Selective Checkpoint Investigation

Production remains all48 checkpoint ON, official non-reentrant block scope `models/LTX2.3/packages/ltx-core/src/ltx_core/model/transformer/model.py::_process_transformer_blocks`. The following uses actual block0/K4 and block1/K5 with real first-block hidden `[1,73440,4096]`, all48 weights resident. Block1 is a controlled isolated fixture, **not the actual block0 output**. No reduced frame/resolution/model substitute is used, but fixed VJP is the captured hidden rather than the final diffusion loss, and there is no optimizer in these isolated tests. The isolated input is forced `requires_grad_(True)` to compare input gradients, so block0's input-gradient boundary is not identical to the frozen-preprocessor boundary in full-model training.

| Block | Mode | F s | B s | F+B s | Peak allocated GiB | Post-F allocated GiB | Strict bitwise status (all repeats) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b0/K4 | baseline | 0.321710 | 0.746917 | 1.068628 | 99.777 | 48.695 | MISMATCH |
| b0/K4 | checkpoint_off | 0.406812 | 0.441663 | 0.848475 | 99.252 | 91.407 | MISMATCH |
| b1/K5 | baseline | 0.340059 | 0.902634 | 1.242693 | 101.453 | 48.694 | MISMATCH |
| b1/K5 | checkpoint_off | 0.433901 | 0.526679 | 0.960580 | 101.037 | 93.192 | MISMATCH |

MISMATCH in this column includes unchanged baseline repetitions: outputs are identical and gradients finite, but backward is not bitwise repeatable even without a candidate change. It is not a claim that checkpoint ON/OFF changed the architecture; see section12's numerical controls.

Measured checkpoint-OFF post-forward retention increases: K4: **+42.713 GiB** for **0.220 s** isolated F+B saving; K5: **+44.498 GiB** for **0.282 s** isolated F+B saving. The isolated peak can be almost unchanged because only one block is differentiated, but retained-forward state is what overlaps with other blocks' backward peaks in the real 48-block graph. This distinction is why the isolated peak is not a safe full-model memory forecast.

A checkpoint-free block retains tens of GiB more activation state, whereas full warm training has a ~129.25 GiB allocated peak. Four/eight checkpoint-free blocks therefore cannot be justified by multiplying a single-block timing speedup while ignoring live activation overlap. An early block's retained state survives most of the reverse pass; a late block has a different lifetime and could behave differently. We did **not** claim every possible one-block tail selection is impossible, and did not repeatedly OOM the full model to test arbitrary subsets. No selective checkpoint policy is recommended for immediate production; a tail-lifetime memory proof would be a separate experiment, for a subsecond-per-block ceiling.

## 12. Additional A/B Experiments

`probe_constant_cache.py` is an invocation-local prototype, never imported by production. For each actual block K4/K5 it runs baseline, metadata, metadata+mapped Adaptive positions, metadata+final Adaptive/Global PE, checkpoint OFF, baseline again. Caches are eagerly built in an untimed no-grad pass; an additional untimed forward/backward warms wgrad. Each row then has two synchronized measured repetitions. Forward autocast and backward-outside-autocast match production ordering. CPU snapshots of output/input gradient/all block parameter gradients are compared **outside timings**. Allocator cache is cleared between isolated runs; thus the block timings are not the exact allocation lifetime of full48 training.

| Block | Mode | F mean s | B mean s | F+B s | Peak alloc GiB | Output relL2 | Input-grad relL2 | Max parameter-grad relL2 | Exact/finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0/K4 | baseline | 0.321710 | 0.746917 | 1.068628 | 99.777 | 0 | 1.86e-05 | 8.78e-05 | False/True |
| b0/K4 | metadata_cache | 0.268109 | 0.714460 | 0.982569 | 99.776 | 0 | 2.32e-05 | 6.3e-05 | False/True |
| b0/K4 | metadata_and_adaptive_mapped_positions | 0.258642 | 0.708000 | 0.966642 | 99.777 | 0 | 2.39e-05 | 8.74e-05 | False/True |
| b0/K4 | metadata_and_final_pe_cache | 0.285366 | 0.689335 | 0.974701 | 99.775 | 0 | 2.53e-05 | 8.8e-05 | False/True |
| b0/K4 | checkpoint_off | 0.406812 | 0.441663 | 0.848475 | 99.252 | 0 | 2.04e-05 | 8.76e-05 | False/True |
| b0/K4 | baseline_again | 0.331945 | 0.748261 | 1.080205 | 99.777 | 0 | 1.93e-05 | 7.02e-05 | False/True |
| b1/K5 | baseline | 0.340059 | 0.902634 | 1.242693 | 101.453 | 0 | 9.71e-05 | 0.000135 | False/True |
| b1/K5 | metadata_cache | 0.301901 | 0.857841 | 1.159742 | 101.444 | 0 | 9.4e-05 | 0.000162 | False/True |
| b1/K5 | metadata_and_adaptive_mapped_positions | 0.290138 | 0.843512 | 1.133650 | 101.445 | 0 | 9.63e-05 | 0.000167 | False/True |
| b1/K5 | metadata_and_final_pe_cache | 0.272705 | 0.815548 | 1.088253 | 101.442 | 0 | 9.34e-05 | 0.000132 | False/True |
| b1/K5 | checkpoint_off | 0.433901 | 0.526679 | 0.960580 | 101.037 | 0 | 9.9e-05 | 0.000135 | False/True |
| b1/K5 | baseline_again | 0.343887 | 0.900168 | 1.244056 | 101.453 | 0 | 0.000103 | 0.00015 | False/True |

The strict probe status requires bitwise equality, which is stronger than the benchmark algorithm-change acceptance criterion. Baseline-again and within-mode repeatability distinguish intrinsic nondeterminism from a candidate-specific problem; nonzero differences are disclosed rather than renamed PASS. Aggregate probe status: **MISMATCH**. Details: `constant_cache_probe/summary.json` and each mode's result.

In fact **every baseline and candidate output is bitwise equal**, while even unchanged baseline repetitions have non-bitwise gradients. Across candidates, maximum reported parameter-gradient relative L2 is about 1.67e-4, similar in order to the unchanged K5 baseline repeat (1.35e-4) and baseline-again (1.50e-4); all are finite. This is evidence of small nondeterministic gradient variation, not proof of exact candidate equivalence. Additional optimizations therefore remain **INCONCLUSIVE for strict gradient identity**, not production-approved PASS. Their timing signal is useful for prioritization; a future implementation needs a declared numerical tolerance/repeatability control and full-model validation.

| Block | Mode | Measured cache metadata | No grad/alias | Within-mode exact repeat |
| --- | --- | --- | --- | --- |
| 0 | metadata_cache | metadata_cache={"entries":44,"bytes":253416,"build_seconds":0.06341862119734287,"hits":1167,"misses_after_freeze":0} | True | False |
| 0 | metadata_and_adaptive_mapped_positions | metadata_cache={"entries":44,"bytes":253416,"build_seconds":0.06466291286051273,"hits":1167,"misses_after_freeze":0}<br>adaptive_mapped_position_cache={"entries":12,"bytes":829440,"build_seconds":0.009509777650237083,"hits":72,"misses_after_freeze":0} | True | False |
| 0 | metadata_and_final_pe_cache | metadata_cache={"entries":44,"bytes":253416,"build_seconds":0.0639702882617712,"hits":1167,"misses_after_freeze":0}<br>position_cache={"entries":13,"bytes":320716800,"build_seconds":0.01728968694806099,"hits":78,"misses_after_freeze":0} | True | False |
| 1 | metadata_cache | metadata_cache={"entries":115,"bytes":313104,"build_seconds":0.04511665180325508,"hits":2195,"misses_after_freeze":0} | True | False |
| 1 | metadata_and_adaptive_mapped_positions | metadata_cache={"entries":115,"bytes":313104,"build_seconds":0.04705784656107426,"hits":2195,"misses_after_freeze":0}<br>adaptive_mapped_position_cache={"entries":24,"bytes":1040688,"build_seconds":0.01463242806494236,"hits":144,"misses_after_freeze":0} | True | False |
| 1 | metadata_and_final_pe_cache | metadata_cache={"entries":115,"bytes":313104,"build_seconds":0.04625432565808296,"hits":2195,"misses_after_freeze":0}<br>position_cache={"entries":25,"bytes":392822784,"build_seconds":0.026649001985788345,"hits":150,"misses_after_freeze":0} | True | False |

`build_seconds` inside each cache summary is CPU enqueue time, **not synchronized GPU build time**. The synchronized `warm_and_build_seconds` in each raw mode result covers a whole-block warm/build, so it is also not a pure cache-build cost. This prototype only intercepts literal `torch.tensor/arange` requests and the verified all-batch subset identity; it does not implement the complete proposed geometry validation/index/count plan.

Measured per-probe storage, **calculated K4+K5 concurrent sum** (the probes actually run sequentially): metadata 566,520 bytes = 0.5403 MiB; mapped Adaptive coordinates extra 1,870,128 bytes = 1.7835 MiB; final Adaptive/Global PE 713,539,584 bytes = 0.6645 GiB. A full-call implementation would have to share these constants across repeated same-geometry blocks rather than multiply storage by48. Native per-window PE was intentionally not cached. These arithmetic capacity estimates are not measured full48 peak deltas; per-call lifetimes and allocator fragmentation matter. Mapped-position and final-PE timings are **nested alternatives**; their improvements must not be added. Cache first-build cost is excluded from the two-repeat table and must be paid once per new invocation if using the safest per-call ownership.

## 13. Recommended Next Optimizations

Only the benchmark flag is approved/applied. The next-three assessment is based on the new warmed baseline, not 370 s backward. Full-model estimates below are **conditional extrapolations**, not measured complete training results. They use 24 K4 +24 K5 isolated-block deltas relative to bracketing baseline and are capped by actual remaining cost; two repeats do not establish confidence intervals. First-build cost, changed activation lifetimes and allocator overhead can erase the estimate.

| Rank | Optimization | Measured isolated F+B | 24+24-block arithmetic (NOT full-step measurement) | Conditional sec/step | Risk | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Immutable index/mask plan + B=1 subset identity | K4: 0.983s / K5: 1.160s | +4.056…+4.367s projected saving before build/lifetime cost | 50.24…50.55s conditional | Low math risk; validate indices/alias/geometry keys; no hidden cache | Candidate only; no production application |
| 2 | Metadata + mapped Adaptive PMA positions | K4: 0.967s / K5: 1.134s | +5.065…+5.375s projected saving before build/lifetime cost | 49.23…49.54s conditional | Low–medium implementation risk; exact time/fps/scale invalidation | Candidate only; no production application |
| 3 | Metadata + final Adaptive/Global PE | K4: 0.975s / K5: 1.088s | +5.961…+6.271s projected saving before build/lifetime cost | 48.34…48.65s conditional | Medium memory risk; no Native PE; call-scoped prebuild required | Candidate only; no production application |

If a candidate's delta is within baseline drift, it is inconclusive rather than a confirmed speedup. A safe next implementation would share only immutable per-call constants across equal-geometry blocks, then validate checkpoint replay and every gradient again. Do not globally cache by shape alone; equal shape can still mean different canonical positions or frame-rate/time coordinates. For operational planning, keep **54.61s/step** until a full-model candidate run proves otherwise; the conditional values above are investigation targets.

## 14. Expected 30K Training Time

| Scenario | Seconds/step | 30,000-step hours | Days | Status |
| --- | --- | --- | --- | --- |
| Previous observed production | 372–403 | 3100–3358.3 | 129.17–139.93 | Historical measurement range |
| Autotune measured warm mean | 54.608971 | 455.075 | 18.961 | 5 real steps; warm n=4 |
| 55–70s planning range | 55–70 | 458.33–583.33 | 19.10–24.31 | Planning, not measured 30K run |

Add model/cache startup, cold autotune (~195 s above warm for this first step), checkpoint I/O every 500 optimizer steps, restarts, validation if later enabled and hardware contention. The estimate does not imply production was started. No additive speed claim is made for earlier NCDHW/cuDNN-off results; those addressed the same removed pathology and are alternatives, not extra gains.

## 15. Risks / Rejected Optimizations

- **Memory:** current workload passes alone but has tight physical headroom; do not run competing GPU jobs or retain many geometry/PE caches.
- **Numerics:** benchmark can select different reductions/plans across fresh processes. Loss/finite health and FP64 probes pass; bitwise and long-training quality equality are not promised.
- **Cold search:** first step and genuinely new convolution signatures may be slow. Restarting frequently discards process-local plans. Do not classify the first slow step as a regression.
- **Determinism:** no deterministic-policy change was made. If a launcher later enables deterministic algorithms or changes backend defaults, re-audit actual dispatch.
- **No production monitoring resurrection:** all events, scans, clones, .cpu comparisons and Kineto instrumentation live only under tools and diagnostic output roots.
- **No R layout/Conv change:** NCDHW-contiguous, disabling cuDNN, channel/group/kernel changes, alternate R layers, precision changes and interpolation changes were not applied.
- **Selective CP:** no full48 arbitrary 4/8-block OFF OOM loop; large retained activations make a naive policy unsafe.
- **Compile:** current eager native-gate contract explicitly rejects compilation; removing its guard without a separate semantic design/validation is not a safe flag tweak.
- **CUDA Graph / batching:** unmeasured with actual capture/lifetime semantics; not given invented speedup figures.
- **Dataset coverage:** 3 actual records and the 5 present R2 shape families, not every possible future geometry. Cache-created count=0.
- **Checkpoint:** existing files/format preserved; this task's save/load claim is static+CPU regression, not another GPU resume test.

## 16. Final Verdict

**CUDNN BENCHMARK PRODUCTION CHANGE: PASS.**

The A–G acceptance conditions are met: onlyflagdiff; before-first-conv activation; actual pathological-kernel elimination; warm backward below60 s and total below75 s on all four warm steps; finite loss/gradients; intended group updates; noOOM. Memory remains tight, and additional optimizations remain unapproved experiments.

| Question | Answer |
| --- | --- |
| Q1: Before first convolution? | Yes: first statement in actual train.main plus actual first Conv2D/3D flag assertions. |
| Q2: Other production semantic changes? | None: exactly one added assignment; protected source/config/run integrity checked. |
| Q3: Cold step? | 249.818253s phase sum; setup 108.079s separate. |
| Q4: Warm steps2–5 mean/median? | 54.608971/54.650459s. |
| Q5: Warm backward-only mean? | 38.781831s. |
| Q6: Multiple production R2 shapes? | Yes: q135/84/39/24/21, in 3 real records with prefix-free/17f prefix. All fixed spatial geometry. |
| Q7: Group-wise pathology gone? | Yes in all representative True probes and full warmed trace. R4/q135 xmma; smaller R2 often grouped-direct. |
| Q8: Loss/grad health? | 5/5 finite; all10 groups nonzero; no missing trainable gradients; frozen violations0; representative updates5/5. |
| Q9: VRAM safe? | NoOOM at 129.253 allocated/135.963 reserved GiB; ~3.07GiB actual free is tight, not broad headroom. |
| Q10: New #1 bottleneck? | 15.521s block replay plus distributed Adaptive/data-movement/elementwise work; not a lone 370s Conv kernel. |
| Q11: Next3 safest investigations? | Immutable metadata; metadata+mapped PMA coordinates; metadata+final Adaptive/Global PE (nested alternatives). See measured caveats. |
| Q12: Expected time/risk? | See section13 conditional block extrapolations, not full-model speedup claims. No extra optimization applied. |

### Reproducibility / artifacts

- Production diff/snapshots: `production.diff`, `train.py.before`, `protected_before.json`, `final_integrity.json`.
- Static: `static_verification.json`, `backend_defaults_cpu.json`, `cpu_regression.log`, `final_checks.log`.
- Real5steps: `production_benchmark/result.json`, `step_*/{result,scopes,R_shapes,sample_runtime}.json`.
- False control: `fresh_false_control/result.json`, `fresh_full_gradient_comparison.json`, selected dW `.pt` files.
- Dispatch: `kernels_false/results.json`, `kernels_true/results.json`, per-case traces.
- Warm full profile: `warmed_profile/step_02/trace.json` (~2.2GB), `raw_summary.json`.
- Extra A/B: `constant_cache_probe/{summary,manifest}.json` and per-mode result files.
- CPU result aggregation: `collected_summary.json`; diagnostic scripts are all under `tools/cudnn_benchmark_validation/`.

All paths above are relative to the evidence root unless stated otherwise. No production process was launched or resumed; `hrlrvo_train` remains available for the user's later explicit instruction.
