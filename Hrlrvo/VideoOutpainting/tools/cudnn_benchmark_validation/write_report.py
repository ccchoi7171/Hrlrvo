"""Render the final report from completed measurements; CPU/JSON only."""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/cudnn_benchmark_validation/20260912_0637"


def read(name):
    return json.loads((BASE / name).read_text())


def phases(row):
    return {p["stage"]: p["seconds"] for p in row["phases"]}


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in rows]
    )


def main():
    training = read("production_benchmark/result.json")
    control = read("fresh_false_control/result.json")
    profile_run = read("warmed_profile/result.json")
    profile = read("warmed_profile/step_02/raw_summary.json")
    collected = read("collected_summary.json")
    ab = read("constant_cache_probe/summary.json")
    ab_manifest = read("constant_cache_probe/manifest.json")
    final_static = read("final_integrity.json")
    assert training["status"] == control["status"] == profile_run["status"] == "PASS"
    assert len(training["steps"]) == 5 and len(ab) == 12
    assert final_static["changed_protected_files"] == ["main/src/ltx_outpaint/train.py"]
    steps = training["steps"]
    warm = steps[1:]
    wm = collected["warm_mean"]
    total = collected["warm_phase_sum"]["mean"]
    median = collected["warm_phase_sum"]["median"]
    alloc = collected["peaks"]["allocated"]
    reserved = collected["peaks"]["reserved"]
    kernels = {v: {r["case"]: r for r in read(f"kernels_{v}/results.json")} for v in ("false", "true")}
    gradients = read("fresh_full_gradient_comparison.json")
    replay = st.mean(
        s["child_seconds"]["backward_only"]["Shared/block"]
        for s in collected["warm_scopes"].values()
    )
    cpu_gpu = defaultdict(lambda: defaultdict(float))
    labels = defaultdict(float)
    for name, row in profile["scopes"].items():
        parts = name.split("/")
        phase = parts[0]
        branch = parts[2] if len(parts) > 2 else "UNATTRIBUTED"
        cpu_gpu[phase][branch] += row["kernel_sum_us"] / 1e6
        labels[f"{phase}/{branch}/{parts[-1]}"] += row["kernel_sum_us"] / 1e6
    ab_summary = {(r["block"], r["mode"]): r for r in collected["constant_cache"]}
    text = []
    add = text.append
    add("# cuDNN Benchmark Validation and Further Speed Optimization\n")
    add("Date: 2026-09-12 (UTC). Project: `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting`.\n"
        "Environment: `Hrlrvo`; GPU: `CUDA_VISIBLE_DEVICES=0`, NVIDIA H200.\n"
        "Evidence root: `outputs/cudnn_benchmark_validation/20260912_0637/`.\n"
        "**Production has NOT been restarted. Existing run, checkpoints, cache/data and `hrlrvo_train` tmux are preserved.**\n")
    add("## 1. Executive Summary\n")
    add(f"**cuDNN startup-autotuning validation: PASS.** The only production edit is one assignment at the beginning of `ltx_outpaint.train.main()`. "
        f"Five actual full-48-block, 4K65f optimizer steps completed. Cold step: **{steps[0]['phase_sum_seconds']:.3f} s**; "
        f"warm steps 2–5: **{total:.3f} s mean / {median:.3f} s median**, with **{wm['backward_only']:.3f} s backward-only**. "
        "The previous 350–380 s backward pathology is not present in warmed measurements.\n")
    add(f"Peak allocated/reserved: **{alloc:.3f}/{reserved:.3f} GiB**. No OOM, but physical free after warm steps is only about 3.07 GiB: "
        "this is a validated isolated workload, not generous multi-workload headroom. All ten trainable groups have finite, nonzero gradients; "
        "frozen gradient violations are zero; all ten representative trainable tensors update.\n")
    add("R4 changed from 4,096 `convolveNd_wgrad_engine` launches to two cuDNN xmma wgrad kernels. "
        "All five observed R2 shape families have zero pathological group-wise launches after autotuning. "
        "Some R2 sizes select one efficient grouped-direct kernel rather than xmma; the evidence does **not** claim every autotuned shape uses the same engine.\n")
    add(f"The new cost is distributed: checkpoint replay is **{replay:.3f} s** (about {100*replay/wm['backward_only']:.1f}% of backward), "
        "with Adaptive preparation/restoration, repeated elementwise/copy work, native projections and shared FFN still material. "
        "There is no observed MATH-SDPA fallback. Additional cache/checkpoint experiments below are diagnostic prototypes, **not production changes**.\n")
    add("## 2. Production Change\n")
    add("File/symbol: `main/src/ltx_outpaint/train.py:967–970`, `main()`. Exact diff:\n\n```diff\n"
        + (BASE / "production.diff").read_text().rstrip() + "\n```\n")
    add("This runs before argument parsing, `load_training_config`, `prepare_run_artifacts`, cache audit, `OutpaintingTrainer` construction, "
        "model loading and every model forward. `torch` was already imported. Importing this module for inference does not execute `main()`. "
        "No pinned LTX file or config changed; no observer, per-step logging, CUDA synchronization, layout conversion or profiler was added to production.\n")
    add("The diagnostic invokes the actual `train.main()` and intercepts the first `torch.nn.functional.conv2d/conv3d` calls. "
        "Both recorded `benchmark=True`, `deterministic=False`, and deterministic-algorithms disabled. First Conv2D input was "
        "`[2844,4096,2,2]`; first Conv3D input was `[135,4096,9,2,2]`, both BF16. "
        "See `production_benchmark/startup.json` and final `result.json` (the startup snapshot itself intentionally says RUNNING).\n")
    add("Why fresh process matters: PyTorch's cuDNN-v8 convolution implementation looks up a cached plan before choosing its heuristic/benchmark branch; "
        "the cache key does not distinguish the benchmark flag. A late flag switch can therefore reuse an old plan. "
        "[PyTorch 2.8 cache key](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cudnn/Conv_v8.cpp#L168-L218), "
        "[lookup and execution](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cudnn/Conv_v8.cpp#L974-L1068).\n")
    add("## 3. Static Correctness Verification\n")
    add(table(["Check", "Result / evidence"], [
        ["Production source/config integrity", "PASS: 335-file pre-change SHA256 snapshot; only train.py differs, by the one line above"],
        ["Python compile/import and YAML parse", "PASS: static_verification.json; imports do not initialize CUDA"],
        ["Actual startup ordering", "PASS: first main AST statement; CPU --help executes it; actual first-convolution hook confirms it"],
        ["Regression", "13 CPU tests passed: main/tests/test_training_runtime.py; cpu_regression.log"],
        ["Ruff / final integrity", "See final_integrity.json and final_checks.log; no production semantic edits besides the flag"],
        ["Backend flag overwrite", "No later False assignment in active Hrlrvo/pinned trainer path"],
        ["Determinism", "No active deterministic=True/use_deterministic_algorithms(True)/CUBLAS_WORKSPACE_CONFIG policy; none changed"],
        ["Accelerate seed", "Installed set_seed defaults deterministic=False; only explicit True calls torch.use_deterministic_algorithms"],
        ["Workers", "CPU data loading does not mutate parent-process cuDNN settings; no worker configuration change"],
        ["Custom monitoring", "--diagnostics remains false; no train_metrics.jsonl or observer reinstated"],
        ["Checkpoint semantics", "Save/resume implementation and checkpoint configuration byte-identical; not a new save/resume execution test"],
        ["Protected resources", "Production run, pinned source and configuration hashes retained; probes create no Anchor entries"],
    ]) + "\n")
    add("Runtime library versions: PyTorch `2.8.0+cu128`, CUDA build `12.8`, runtime cuDNN integer `91002`, NVIDIA driver `580.126.20`. "
        "The shell's `CUDNN_VERSION=9.8.0.87` is not the cuDNN version PyTorch actually reports. "
        "`benchmark_limit=10`, cuDNN enabled, allow_tf32=True were left unchanged. Existing `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, "
        "`CUDA_CACHE_DISABLE=1`, and `CUDA_MODULE_LOADING=LAZY` were not modified. Probes use the user's production OMP/MKL thread count 8.\n")
    add("`benchmark=True` chooses among measured convolution algorithms; it does not change the Conv3D definition or disable deterministic policy. "
        "Algorithm selection/reduction order can vary across process, hardware and shape; bitwise reproducibility is not promised. "
        "[Backend flag documentation](https://docs.pytorch.org/docs/2.8/backends.html#torch.backends.cudnn.benchmark), "
        "[reproducibility notes](https://docs.pytorch.org/docs/2.8/notes/randomness.html#cuda-convolution-benchmarking).\n")
    add(table(["Production property", "Verified value (unchanged)"], [
        ["Architecture", "v8; 48 actual LTX blocks; existing Native/Adaptive/Global and hierarchical fusion"],
        ["Geometry", "3840×2176 internal / 3840×2160 visible; GT [1,128,9,68,120]; Anchor [1,128,9,34,60]"],
        ["Temporal", "FIRST 65 RGB frames; 9 latent frames; prefix 17 RGB / 3 latent, probability 0.5"],
        ["Batch", "microbatch=1, accumulation=1, world_size=1"],
        ["Precision / checkpoint", "BF16; all 48 blocks checkpointed, use_reentrant=False"],
        ["Optimizer / scheduler", "Actual AdamW; all groups base LR 1e-4; LinearLR 1.0→0.1 over 30,000 steps"],
        ["Clipping", "Actual Accelerator.clip_grad_norm_, max_grad_norm=1.0"],
        ["Sampling / loss", "Unchanged shifted-logit-normal; masked final flow-matching MSE; known Anchor OR prefix excluded"],
        ["Seed / checkpoint interval", "33333333; 500 optimizer steps"],
        ["Counts", "Trainable 280,363,008; frozen 18,988,838,144; installed 19,269,201,152"],
    ]) + "\n")
    add("## 4. Cold vs Warm Training Timing\n")
    add("Method: fresh process; actual production entrypoint/model/`OutpaintingTrainer._training_step`, actual AdamW/LinearLR/Accelerator backward and clipping. "
        "Only the outer `run_training` callback is replaced in memory with a bounded fixed-record diagnostic loop. "
        "Real `OutpaintingCacheDataset` and collator are used; parallel DataLoader scheduling and the native console/progress loop are not benchmarked. "
        "Stage boundaries synchronize CUDA **only in the diagnostic harness**. Gradient/snapshot audits occur outside reported phase sums. "
        "These are real updates, not forward-only or a toy model; no production checkpoint is written. "
        "The measured phases still include bounded CUDA-event scopes and R-shape hooks; they are conservative diagnostic timings, not zero-overhead production throughput.\n")
    add(table(["Step", "Kind", "Forward s", "Backward only s", "Clip s", "Adam s", "Phase sum s", "Loss", "Sigma", "Grad norm"], [
        [s["step"], "cold/autotune" if s["step"] == 1 else "warm", f"{phases(s)['forward']:.6f}",
         f"{phases(s)['backward_only']:.6f}", f"{phases(s)['clip_grad_norm_only']:.6f}",
         f"{phases(s)['optimizer_step']:.6f}", f"{s['phase_sum_seconds']:.6f}", f"{s['loss']:.9f}",
         f"{s['sigma'][0]:.9f}", f"{s['global_grad_norm_before_clip']:.9f}"] for s in steps
    ]) + "\n")
    add(f"Warm mean/median/min/max: **{total:.6f} / {median:.6f} / {min(s['phase_sum_seconds'] for s in warm):.6f} / "
        f"{max(s['phase_sum_seconds'] for s in warm):.6f} s**. Mean forward **{wm['forward']:.6f}**, mean backward **{wm['backward_only']:.6f} s**. "
        f"Backward/forward is now **{wm['backward_only']/wm['forward']:.2f}×**, not 12–15×.\n")
    add(table(["Other measured phase", "Warm mean seconds"], [[k, f"{v:.6f}"] for k, v in wm.items() if k not in ("forward", "backward_only", "audit_OUTSIDE_PHASE_SUM")]) + "\n")
    add(f"Cold phase sum exceeds warm mean by **{steps[0]['phase_sum_seconds']-total:.3f} s**. This is a combined cold/algorithm-search/first-Adam-allocation cost, "
        "not a pure measurement of search alone. Initial model/config/cache setup was additionally "
        f"{training['startup_seconds']:.3f} s. Warm diagnostic wall times including health work were "
        + ", ".join(f"{s['wall_including_diagnostics']:.3f}" for s in warm) + " s. "
        "Four synchronized warm samples are encouraging, not a 30,000-step statistical throughput guarantee.\n")
    add(f"Fresh pre-change control: F={phases(control['steps'][0])['forward']:.3f} s, B={phases(control['steps'][0])['backward_only']:.3f} s, "
        f"phase sum={control['steps'][0]['phase_sum_seconds']:.3f} s. This agrees with the earlier production 350–380 s backward. "
        "Do not interpret cold-False versus warm-True as a rigorously identical warm-state A/B; first-step state and cold costs differ. "
        "The dispatch microtraces independently establish the pathological-kernel replacement.\n")
    add("## 5. Multi-Sample / Multi-Shape Validation\n")
    add(table(["Step", "Actual matching relative record ID", "Prefix mode", "Known tokens", "Total s"], [
        [s["step"], s["record"], "17f prefix" if s["input"]["known_token_count"] == 36720 else "prefix-free",
         s["input"]["known_token_count"], f"{s['phase_sum_seconds']:.3f}"] for s in steps
    ]) + "\n")
    add("All three records use the production-fixed Anchor rectangle y=[17:51), x=[30:90) and the same 68×120 canonical canvas. "
        "There are **not three different spatial Anchor geometries** in this test: inventing them would change the requested production contract. "
        "Instead every real sample exercises both K4/K5 partitions and all five R2 query-count shapes, while the three records and two prefix states "
        "test reuse across content/noise/conditioning. Step 2 resets seed to match the earlier comparison draw but retains step-1 updated parameters and live Adam state; "
        "steps 3–5 continue the RNG. Thus this is controlled sample selection, not a shuffled production-epoch benchmark.\n")
    add(table(["Refinement input N,C,T,H,W", "Occurrences per original forward", "Occurrences during replay", "Observed reuse"], [
        ["R2 [135,4096,9,2,2]", 288, 288, "K4; same in all 5 steps"],
        ["R2 [84,4096,9,2,2]", 384, 384, "K5; same in all 5 steps"],
        ["R2 [39,4096,9,2,2]", 96, 96, "K5"],
        ["R2 [24,4096,9,2,2]", 48, 48, "K5"],
        ["R2 [21,4096,9,2,2]", 48, 48, "K5"],
        ["R4 [1,4096,9,68,120]", 48, 48, "All blocks"],
    ]) + "\n")
    add("All shape/stride/dtype observations are in each `step_*/R_shapes.json`. No warmed-step cold spike appears when record or prefix mode changes. "
        "This demonstrates process-local plan reuse for the tested production family, not direct introspection of cuDNN cache-hit counters. "
        "Unseen batch/shape/layout/device/dtype would need its own search; none is silently padded or reduced here.\n")
    add("## 6. R2/R4 Kernel Dispatch Verification\n")
    add("Bounded fresh-False/fresh-True operator probes use the actual unchanged `SpatialDecompressor(4096)`, original strides, BF16, "
        "full temporal extent 9, and spatial R2/R4 shapes. These synthetic operator-level VJPs are separate from the real-cache full training test. "
        "Each has a cold run, three unprofiled warm runs, then a bounded trace. Reported wgrad time is a CUDA interval **union**, "
        "not a sum of concurrent group streams.\n")
    add(table(["Case", "False storm kernels", "True storm kernels", "False dW union ms", "True dW union ms", "True selected dW engine", "Warm True full R backward ms"], [
        [name, kernels['false'][name]['groupwise_convolveNd_wgrad_count'], row['groupwise_convolveNd_wgrad_count'],
         f"{kernels['false'][name]['wgrad_kernel_union_ms']:.3f}", f"{row['wgrad_kernel_union_ms']:.3f}",
         "2 × xmma (main + split-K)" if any('xmma' in k['name'] for k in row['wgrad_kernels']) else "1 × grouped-direct",
         f"{1000*st.mean(t['backward'] for t in row['warm_times']):.3f}"] for name, row in kernels['true'].items()
    ]) + "\n")
    add("Exact names include:\n"
        "- Old: `cudnn::cnn::convolveNd_wgrad_engine<__nv_bfloat16,3,...>` ×4096.\n"
        "- R4/R2-q135: `sm80_xmma_wgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_..._execute_kernel__5x_cudnn` + split-K.\n"
        "- Some smaller R2: `cudnn::cnn::wgrad2d_grouped_direct_kernel<true,true,int,__nv_bfloat16,float,float>` ×1.\n\n"
        "The grouped-direct name does not imply one kernel per group; it handles the operation in a single launch. "
        "R2-q84 already had a fast default xmma plan; its true dW-only probe is not faster in isolation, while total R backward still is slightly faster. "
        "Autotuning need not improve every subkernel to eliminate the pathological family.\n")
    add(f"The independent **warmed full48 trace** contains no `convolveNd_wgrad_engine` kernel at all. "
        f"All wgrad-name kernel interval union is **{profile['kernel_union_seconds'].get('backward_wgrad', 0):.3f} s**, "
        "including compressors, not just R2/R4. Raw evidence: `kernels_false/`, `kernels_true/`, "
        "`warmed_profile/step_02/trace.json` and `raw_summary.json`.\n")
    add("## 7. Gradient / Numerical Validation\n")
    add(table(["Group", "Parameters", "Grad norm min over 5", "Grad norm max over 5", "Missing / nonfinite", "Representative updated"], [
        [g, f"{steps[0]['gradient_health']['groups'][g]['parameter_count']:,}",
         f"{min(s['gradient_health']['groups'][g]['norm'] for s in steps):.8f}",
         f"{max(s['gradient_health']['groups'][g]['norm'] for s in steps):.8f}", "0 / 0", "5/5"]
        for g in ("op_lora", "a_lora", "g_lora", "f_lora", "c2", "c4_shared", "r2", "r4", "local_fusion", "global_fusion")
    ]) + "\n")
    add("All 1,728 trainable tensors had gradients; zero-B LoRA A gradients may individually be zero, which is not treated as a disconnect. "
        "All five losses/sigmas finite; frozen Transformer gradient violations zero. Four representative frozen SA-Q/O/Text-CA/FFN slices were unchanged "
        "after every step. This is **not a bytewise comparison of every 19B frozen weight**; full frozen-gradient enumeration and unchanged optimizer membership "
        "complement representative value comparisons. The empty connector remains frozen; no text/VAE training or cache encoding is performed.\n")
    add("Actual clipping executes with max_grad_norm=1.0. All measured preclip norms are below 1, so these steps exercise the clipping path but not a norm>1 rescale. "
        "Adam states materialized; optimizer state tensor elements after five steps: "
        f"{training['optimizer_state_numel']:,} = 2 × 280,363,008 + 1,728 scalar step counters. "
        "LR after step1→5: " + ", ".join(f"{s['lr_after_step'][0]:.9g}" for s in steps) + ". "
        "The production save/resume format is unchanged and covered by CPU regression/static integrity; this task did not add another actual save/resume round trip.\n")
    comp = collected["numeric_comparison"]
    add(f"Fresh False/True step1 same input SHA256: `{steps[0]['input']['latent_sha256']}`; same record, seed, sigma and "
        "ten representative initial weight hashes. The baseline main is compiled in memory with only the new assignment omitted, not toggled after convolutions. "
        f"Loss is **exactly {comp['loss_true']:.12f}** in both processes; global norm False/True: "
        f"**{comp['global_norm_false']:.12f}/{comp['global_norm_true']:.12f}**. "
        f"The largest ten-group **norm** relative difference is {100*max(v['relative_difference'] for v in comp['norms'].values()):.5f}%.\n")
    add(table(["Selected full dW", "False vs True relative L2", "Max absolute difference", "Finite"], [
        [r['name'].replace('transformer_blocks.', 'b').replace('.attn1.decompressors.', '.R').replace('.refine.', '.'),
         f"{100*r['relative_l2']:.6f}%", f"{r['max_absolute']:.8g}", r['finite']]
        for r in gradients if r['name'].endswith('weight')
    ]) + "\n")
    add("**Do not confuse similar gradient norms with identical vectors.** Early-block selected R2/R4 dW differ by up to about 1.46% relative L2, "
        "while late-block differences are smaller. Full trainable tensors/activations were not all snapshotted. "
        "Different BF16 convolution reduction order and propagated upstream differences are permitted here; no universal BF16 tolerance or long-run quality equivalence is asserted.\n")
    add(table(["Direct operator dW vs CPU FP64 (8 selected channels)", "False relative L2", "True relative L2"], [
        [name, f"{kernels['false'][name]['dW_relative_l2_vs_fp64']:.8f}", f"{row['dW_relative_l2_vs_fp64']:.8f}"]
        for name, row in kernels['true'].items()
    ]) + "\n")
    add("These channel-wise FP64 references check the exact operator on the same rounded BF16 input/upstream gradient, not end-to-end training quality. "
        "True errors are approximately 0.15–0.17%; R4 is essentially unchanged versus False and pathological R2 cases improve. "
        "Together with five-step finite health, this supports numerical sanity, **not bitwise reproducibility**.\n")
    add("## 8. Memory Usage\n")
    add(table(["Step", "Peak allocated GiB", "Peak reserved GiB", "End physical free GiB"], [
        [s['step'], f"{s['memory']['peak_allocated_gib']:.6f}", f"{s['memory']['peak_reserved_gib']:.6f}", f"{s['memory']['free_gib']:.6f}"] for s in steps
    ]) + "\n")
    add("NVML total 143,771 MiB (140.401 GiB); `torch.cuda.mem_get_info` total 139.801 GiB. These APIs expose slightly different totals; "
        "the report does not confuse the marketed 141 GB class with GiB. GPU initially had no compute process. "
        "Warm live allocated peak leaves ~10.55 GiB relative to PyTorch total; reserved peak leaves ~3.84 GiB in that same accounting, "
        "and actual free is ~3.07 GiB after non-allocator overhead. Reserved includes reusable cache, not all live tensors. "
        "No OOM occurred, but concurrent jobs and additional long-lived activation/PE caches require explicit revalidation.\n")
    add("## 9. New Warmed Bottleneck Breakdown\n")
    add(f"Unprofiled warm timing is the baseline ({total:.3f} s), not the slower Kineto run. "
        f"The separately profiled warm step measured F={phases(profile_run['steps'][1])['forward']:.3f}, "
        f"B={phases(profile_run['steps'][1])['backward_only']:.3f}, total={profile_run['steps'][1]['phase_sum_seconds']:.3f} s. "
        f"Its {profile['event_count']:,} trace events / {profile['categories']['kernel']:,} kernel launches create observable overhead. "
        "Record-shapes/stack/memory profiling were disabled to bound that overhead; peak memory comes from allocator counters instead.\n")
    add("### Unprofiled CUDA-event scopes (mean of warm steps 2–5)\n")
    scope_rows = []
    for branch in ("Native", "Adaptive", "Global"):
        vals = [s['parent_seconds'] for s in collected['warm_scopes'].values()]
        scope_rows.append([branch + " (inclusive)", f"{st.mean(v['forward'][branch] for v in vals):.3f}", f"{st.mean(v['backward_only'][branch] for v in vals):.3f}"])
    for name in ("Adaptive/C2", "Adaptive/C4", "Adaptive/R2", "Global/C4", "Global/R4", "SharedO/O", "TextCA/TextCA", "FFN/FFN", "LocalFusion/LocalFusion", "GlobalFusion/GlobalFusion", "Shared/block"):
        vals = [s['child_seconds'] for s in collected['warm_scopes'].values()]
        scope_rows.append([name, f"{st.mean(v['forward'][name] for v in vals):.3f}", f"{st.mean(v['backward_only'][name] for v in vals):.3f}"])
    add(table(["Scope", "Original forward s", "Backward replay s"], scope_rows) + "\n")
    add(f"All 48 blocks replay **once** in each warm step; no nested/window-level checkpoint or accidental multiple replay was observed. "
        f"Replay {replay:.3f} s is about {100*replay/wm['backward_only']:.1f}% of backward. "
        f"B wall minus block replay interval sum is ~{wm['backward_only']-replay:.3f} s; that subtraction is a useful bound, not an exclusive kernel attribution. "
        "Parent/child event scopes overlap and must NOT be added together.\n")
    add("### Warm trace: nearest-scope CUDA kernel time\n")
    add(table(["Scope", "Forward kernel sum s", "Replay kernel sum s", "Derivative kernel sum s"], [
        [name, f"{cpu_gpu['forward'][name]:.3f}", f"{cpu_gpu['recompute'][name]:.3f}", f"{cpu_gpu['gradient'][name]:.3f}"]
        for name in ("Native", "Adaptive", "Global", "SharedO", "TextCA", "FFN", "LocalFusion", "GlobalFusion", "Shared", "UNATTRIBUTED")
    ]) + "\n")
    add("Attribution uses CUDA external IDs → nearest CPU operator/scope, plus only unambiguous original-forward autograd sequence numbers for derivatives. "
        f"There are {profile['ambiguous_forward_sequence_keys']:,} ambiguous forward sequence keys; ~{cpu_gpu['gradient']['UNATTRIBUTED']:.3f} s of derivative kernels "
        "remain explicitly unattributed. Kernel sums can overlap across streams and are not exclusive branch wall times. "
        f"Whole-trace kernel interval unions: F={profile['kernel_union_seconds']['forward']:.3f}, B={profile['kernel_union_seconds']['backward']:.3f} s. "
        "The difference from wall time also includes launch gaps, CPU work, profiler overhead and memcpy/memset categories omitted from this kernel-only union; "
        "it is not all removable Python overhead.\n")
    add("Adaptive is the largest measured branch grouping after the fix. R2/R4 restoration still has interpolation/layout/dgrad work, "
        f"with derivative kernel sums R2={labels['gradient/Adaptive/R2']:.3f} s and R4={labels['gradient/Global/R4']:.3f} s, "
        "but it is no longer a hundreds-of-seconds wgrad storm. Copies/casts and elementwise work are now major operator families. "
        "Only geometry/position-independent-of-hidden work is a low-risk cache target; the many hidden-dependent copies and BF16/FP32 casts cannot simply be deleted.\n")
    add("### Actual attention backend\n")
    add("All observed Native (984 calls/pass), Adaptive (864), Global (48), and Text CA (48) use cuDNN SDPA: "
        "1,944 original forward calls, 1,944 replay calls, 1,944 backward calls. "
        "Profiler operators `_cudnn_attention_forward` and `_scaled_dot_product_cudnn_attention_backward`, with "
        "`cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_...` kernels, establish dispatch. "
        "The internal kernel name containing `flash`/`f16` does not imply the external flash-attn package or a changed FP16 policy. "
        "No observed MATH/EFFICIENT fallback; installing FlashAttention is not evidence-based treatment for this measured remaining bottleneck.\n")
    add("### Top 35 operators in the warmed trace\n")
    add(table(["Operator", "Calls", "Direct CUDA kernel sum s", "CPU self s"], [
        [r['name'], f"{r['calls']:,}", f"{r['kernel_sum_us']/1e6:.6f}", f"{r['self_cpu_us']/1e6:.6f}"] for r in profile['top_operators_by_kernel_sum'][:35]
    ]) + "\n")
    add("CPU self and CUDA columns overlap asynchronously and are not additive. For example copy_ CPU time can include synchronization/waiting; "
        "5.68 s of copy CUDA kernels is not a promise that removing a metadata allocation saves 5.68 s. "
        "`raw_summary.json` also contains top kernels, CPU operators, backend attribution and exact call counts.\n")
    add("## 10. Remaining Optimization Candidates\n")
    add("The current code already LRU-caches Python Local windows, Adaptive layouts and Global cells (`geometry.py::local_windows/adaptive_layout/global_cells`). "
        "Caching those functions again is not a new optimization. Candidate work is device tensor materialization/validation and constant positional preparation "
        "inside repeated block/view calls.\n")
    add(table(["Candidate", "Current code / cost targeted", "Bounded experiment / disposition"], [
        ["Immutable device metadata plan", "compression.py::SpatialCompressor.forward cell validation/index tensors; Adaptive qi/locations/write indices; Native and assembly indices; ViewContext.subset B=1 copies", "metadata_cache A/B below; no hidden or trainable tensor cached"],
        ["Mapped Adaptive PMA positions", "position.py::target_query_positions; coordinates independent of current hidden and weights", "metadata+mapped-position A/B; call-scoped, exact context positions key"],
        ["Final Adaptive/Global PE cache", "target_query_rope/phase_rope repeated precompute_freqs_cis in forward/replay", "metadata+final-PE A/B; alternative to mapped-position-only; higher memory"],
        ["Selective checkpoint", f"{replay:.2f} s replay over 48 blocks", "Exact spatial/temporal single-block CP ON/OFF below; whole-model 4/8 OFF not blindly attempted"],
        ["Hidden layout/copy changes", "copy/cast, interpolation, convolution tensor transforms", "Not changed; constants are the safe subset. R2/R4 NCDHW explicitly excluded from production"],
        ["Window batching/vectorization", "984 Native +864 Adaptive attention calls/pass, variable bucket shapes", "Static feasibility only; no measured full-step speedup; requires VRAM/reduction-order audit"],
        ["torch.compile", "Potential Python/launch work", "Current model.py::_forward_with_native_gate explicitly rejects torch.compiler.is_compiling(); not applied or advertised as ready"],
        ["CUDA Graph", "Stable finite shape family, many small launches", "Not established: host ContextVar/control, checkpoint capture and memory-pool lifetime need work. Dynamic record alone is not a sufficient reason to reject"],
        ["Optimizer/data", f"Adam {wm['optimizer_step']:.3f} s; read/H2D small", "Not a worthwhile target; unchanged"],
    ]) + "\n")
    add("Cache safety boundary: cache only detached immutable indices/masks/coordinates/PE, never current hidden, packed tokens, QKV, compressor/restored outputs, "
        "gates, loss or gradients. A correct key includes actual position/geometry, device, dtype, scale and RoPE settings; prefix/time/fps or batch changes must invalidate it. "
        "Build eagerly before checkpoint entry, then keep the same cache-hit path in original forward and replay. "
        "Do not mutate a cache for the first time during replay and violate non-reentrant saved-tensor equivalence.\n")
    add("## 11. Selective Checkpoint Investigation\n")
    add("Production remains all48 checkpoint ON, official non-reentrant block scope "
        "`models/LTX2.3/packages/ltx-core/src/ltx_core/model/transformer/model.py::_process_transformer_blocks`. "
        "The following uses actual block0/K4 and block1/K5 with real first-block hidden `[1,73440,4096]`, all48 weights resident. "
        "Block1 is a controlled isolated fixture, **not the actual block0 output**. No reduced frame/resolution/model substitute is used, "
        "but fixed VJP is the captured hidden rather than the final diffusion loss, and there is no optimizer in these isolated tests. "
        "The isolated input is forced `requires_grad_(True)` to compare input gradients, so block0's input-gradient boundary is not identical "
        "to the frozen-preprocessor boundary in full-model training.\n")
    cp_rows = []
    for i in (0, 1):
        for mode in ("baseline", "checkpoint_off"):
            r = ab_summary[i, mode]
            cp_rows.append([f"b{i}/K{4+i}", mode, f"{r['forward']:.6f}", f"{r['backward']:.6f}", f"{r['total']:.6f}",
                            f"{r['peak_allocated']:.3f}", f"{r['post_forward_allocated']:.3f}", r['status']])
    add(table(["Block", "Mode", "F s", "B s", "F+B s", "Peak allocated GiB", "Post-F allocated GiB", "Strict bitwise status (all repeats)"], cp_rows) + "\n")
    add("MISMATCH in this column includes unchanged baseline repetitions: outputs are identical and gradients finite, but backward is not bitwise "
        "repeatable even without a candidate change. It is not a claim that checkpoint ON/OFF changed the architecture; see section12's numerical controls.\n")
    add("Measured checkpoint-OFF post-forward retention increases: "
        + "; ".join(
            f"K{4+i}: **+{ab_summary[i, 'checkpoint_off']['post_forward_allocated']-ab_summary[i, 'baseline']['post_forward_allocated']:.3f} GiB** "
            f"for **{ab_summary[i, 'baseline']['total']-ab_summary[i, 'checkpoint_off']['total']:.3f} s** isolated F+B saving"
            for i in (0, 1)
        ) + ". The isolated peak can be almost unchanged because only one block is differentiated, but retained-forward state is what overlaps "
        "with other blocks' backward peaks in the real 48-block graph. This distinction is why the isolated peak is not a safe full-model memory forecast.\n")
    add("A checkpoint-free block retains tens of GiB more activation state, whereas full warm training has a ~129.25 GiB allocated peak. "
        "Four/eight checkpoint-free blocks therefore cannot be justified by multiplying a single-block timing speedup while ignoring live activation overlap. "
        "An early block's retained state survives most of the reverse pass; a late block has a different lifetime and could behave differently. "
        "We did **not** claim every possible one-block tail selection is impossible, and did not repeatedly OOM the full model to test arbitrary subsets. "
        "No selective checkpoint policy is recommended for immediate production; a tail-lifetime memory proof would be a separate experiment, for a subsecond-per-block ceiling.\n")
    add("## 12. Additional A/B Experiments\n")
    add("`probe_constant_cache.py` is an invocation-local prototype, never imported by production. "
        "For each actual block K4/K5 it runs baseline, metadata, metadata+mapped Adaptive positions, metadata+final Adaptive/Global PE, checkpoint OFF, baseline again. "
        "Caches are eagerly built in an untimed no-grad pass; an additional untimed forward/backward warms wgrad. "
        "Each row then has two synchronized measured repetitions. Forward autocast and backward-outside-autocast match production ordering. "
        "CPU snapshots of output/input gradient/all block parameter gradients are compared **outside timings**. "
        "Allocator cache is cleared between isolated runs; thus the block timings are not the exact allocation lifetime of full48 training.\n")
    add(table(["Block", "Mode", "F mean s", "B mean s", "F+B s", "Peak alloc GiB", "Output relL2", "Input-grad relL2", "Max parameter-grad relL2", "Exact/finite"], [
        [f"b{r['block']}/K{4+r['block']}", r['mode'], f"{r['forward']:.6f}", f"{r['backward']:.6f}", f"{r['total']:.6f}",
         f"{r['peak_allocated']:.3f}", f"{r['max_output_relative_l2']:.3g}", f"{r['max_input_grad_relative_l2']:.3g}",
         f"{r['max_parameter_grad_relative_l2']:.3g}", f"{r['all_exact']}/{r['all_finite']}"] for r in collected['constant_cache']
    ]) + "\n")
    add("The strict probe status requires bitwise equality, which is stronger than the benchmark algorithm-change acceptance criterion. "
        "Baseline-again and within-mode repeatability distinguish intrinsic nondeterminism from a candidate-specific problem; nonzero differences are disclosed rather than renamed PASS. "
        f"Aggregate probe status: **{ab_manifest['status']}**. Details: `constant_cache_probe/summary.json` and each mode's result.\n")
    add("In fact **every baseline and candidate output is bitwise equal**, while even unchanged baseline repetitions have non-bitwise gradients. "
        "Across candidates, maximum reported parameter-gradient relative L2 is about 1.67e-4, similar in order to the unchanged K5 baseline repeat "
        "(1.35e-4) and baseline-again (1.50e-4); all are finite. This is evidence of small nondeterministic gradient variation, "
        "not proof of exact candidate equivalence. Additional optimizations therefore remain **INCONCLUSIVE for strict gradient identity**, not production-approved PASS. "
        "Their timing signal is useful for prioritization; a future implementation needs a declared numerical tolerance/repeatability control and full-model validation.\n")
    cache_rows = []
    for r in ab:
        if 'cache' not in r['configuration'] and 'positions' not in r['configuration']:
            continue
        details = []
        for key in ('metadata_cache', 'adaptive_mapped_position_cache', 'position_cache'):
            if r.get(key):
                details.append(key + '=' + json.dumps(r[key], separators=(',', ':')))
        cache_rows.append([r['block_index'], r['configuration'], '<br>'.join(details), r['cache_safety_pass'], r['same_configuration_repeatability_pass']])
    add(table(["Block", "Mode", "Measured cache metadata", "No grad/alias", "Within-mode exact repeat"], cache_rows) + "\n")
    add("`build_seconds` inside each cache summary is CPU enqueue time, **not synchronized GPU build time**. The synchronized `warm_and_build_seconds` "
        "in each raw mode result covers a whole-block warm/build, so it is also not a pure cache-build cost. "
        "This prototype only intercepts literal `torch.tensor/arange` requests and the verified all-batch subset identity; "
        "it does not implement the complete proposed geometry validation/index/count plan.\n")
    metadata_bytes = sum(r['metadata_cache']['bytes'] for r in ab if r['configuration'] == 'metadata_cache')
    mapped_bytes = sum(r['adaptive_mapped_position_cache']['bytes'] for r in ab if r['configuration'] == 'metadata_and_adaptive_mapped_positions')
    pe_bytes = sum(r['position_cache']['bytes'] for r in ab if r['configuration'] == 'metadata_and_final_pe_cache')
    add("Measured per-probe storage, **calculated K4+K5 concurrent sum** (the probes actually run sequentially): "
        f"metadata {metadata_bytes:,} bytes = {metadata_bytes/2**20:.4f} MiB; mapped Adaptive coordinates extra "
        f"{mapped_bytes:,} bytes = {mapped_bytes/2**20:.4f} MiB; final Adaptive/Global PE "
        f"{pe_bytes:,} bytes = {pe_bytes/2**30:.4f} GiB. A full-call implementation would have to share these constants across repeated same-geometry blocks "
        "rather than multiply storage by48. Native per-window PE was intentionally not cached. "
        "These arithmetic capacity estimates are not measured full48 peak deltas; per-call lifetimes and allocator fragmentation matter. "
        "Mapped-position and final-PE timings are **nested alternatives**; their improvements must not be added. "
        "Cache first-build cost is excluded from the two-repeat table and must be paid once per new invocation if using the safest per-call ownership.\n")
    add("## 13. Recommended Next Optimizations\n")
    add("Only the benchmark flag is approved/applied. The next-three assessment is based on the new warmed baseline, not 370 s backward. "
        "Full-model estimates below are **conditional extrapolations**, not measured complete training results. "
        "They use 24 K4 +24 K5 isolated-block deltas relative to bracketing baseline and are capped by actual remaining cost; "
        "two repeats do not establish confidence intervals. First-build cost, changed activation lifetimes and allocator overhead can erase the estimate.\n")
    candidate_rows = []
    for rank, mode, label, risk in (
        (1, 'metadata_cache', 'Immutable index/mask plan + B=1 subset identity', 'Low math risk; validate indices/alias/geometry keys; no hidden cache'),
        (2, 'metadata_and_adaptive_mapped_positions', 'Metadata + mapped Adaptive PMA positions', 'Low–medium implementation risk; exact time/fps/scale invalidation'),
        (3, 'metadata_and_final_pe_cache', 'Metadata + final Adaptive/Global PE', 'Medium memory risk; no Native PE; call-scoped prebuild required'),
    ):
        savings = []
        for base_mode in ('baseline', 'baseline_again'):
            savings.append(24*sum(ab_summary[i, base_mode]['total'] - ab_summary[i, mode]['total'] for i in (0, 1)))
        candidate_rows.append([rank, label,
                               ' / '.join(f"K{4+i}: {ab_summary[i, mode]['total']:.3f}s" for i in (0, 1)),
                               f"{min(savings):+.3f}…{max(savings):+.3f}s projected saving before build/lifetime cost",
                               f"{total-max(savings):.2f}…{total-min(savings):.2f}s conditional", risk, 'Candidate only; no production application'])
    add(table(["Rank", "Optimization", "Measured isolated F+B", "24+24-block arithmetic (NOT full-step measurement)", "Conditional sec/step", "Risk", "Disposition"], candidate_rows) + "\n")
    add("If a candidate's delta is within baseline drift, it is inconclusive rather than a confirmed speedup. "
        "A safe next implementation would share only immutable per-call constants across equal-geometry blocks, then validate checkpoint replay and every gradient again. "
        "Do not globally cache by shape alone; equal shape can still mean different canonical positions or frame-rate/time coordinates. "
        "For operational planning, keep **54.61s/step** until a full-model candidate run proves otherwise; the conditional values above are investigation targets.\n")
    add("## 14. Expected 30K Training Time\n")
    add(table(["Scenario", "Seconds/step", "30,000-step hours", "Days", "Status"], [
        ['Previous observed production', '372–403', '3100–3358.3', '129.17–139.93', 'Historical measurement range'],
        ['Autotune measured warm mean', f'{total:.6f}', f'{total*30000/3600:.3f}', f'{total*30000/86400:.3f}', '5 real steps; warm n=4'],
        ['55–70s planning range', '55–70', f'{55*30000/3600:.2f}–{70*30000/3600:.2f}', f'{55*30000/86400:.2f}–{70*30000/86400:.2f}', 'Planning, not measured 30K run'],
    ]) + "\n")
    add("Add model/cache startup, cold autotune (~195 s above warm for this first step), checkpoint I/O every 500 optimizer steps, restarts, "
        "validation if later enabled and hardware contention. The estimate does not imply production was started. "
        "No additive speed claim is made for earlier NCDHW/cuDNN-off results; those addressed the same removed pathology and are alternatives, not extra gains.\n")
    add("## 15. Risks / Rejected Optimizations\n")
    add("- **Memory:** current workload passes alone but has tight physical headroom; do not run competing GPU jobs or retain many geometry/PE caches.\n"
        "- **Numerics:** benchmark can select different reductions/plans across fresh processes. Loss/finite health and FP64 probes pass; bitwise and long-training quality equality are not promised.\n"
        "- **Cold search:** first step and genuinely new convolution signatures may be slow. Restarting frequently discards process-local plans. Do not classify the first slow step as a regression.\n"
        "- **Determinism:** no deterministic-policy change was made. If a launcher later enables deterministic algorithms or changes backend defaults, re-audit actual dispatch.\n"
        "- **No production monitoring resurrection:** all events, scans, clones, .cpu comparisons and Kineto instrumentation live only under tools and diagnostic output roots.\n"
        "- **No R layout/Conv change:** NCDHW-contiguous, disabling cuDNN, channel/group/kernel changes, alternate R layers, precision changes and interpolation changes were not applied.\n"
        "- **Selective CP:** no full48 arbitrary 4/8-block OFF OOM loop; large retained activations make a naive policy unsafe.\n"
        "- **Compile:** current eager native-gate contract explicitly rejects compilation; removing its guard without a separate semantic design/validation is not a safe flag tweak.\n"
        "- **CUDA Graph / batching:** unmeasured with actual capture/lifetime semantics; not given invented speedup figures.\n"
        "- **Dataset coverage:** 3 actual records and the 5 present R2 shape families, not every possible future geometry. Cache-created count=0.\n"
        "- **Checkpoint:** existing files/format preserved; this task's save/load claim is static+CPU regression, not another GPU resume test.\n")
    add("## 16. Final Verdict\n")
    add("**CUDNN BENCHMARK PRODUCTION CHANGE: PASS.**\n\n"
        "The A–G acceptance conditions are met: onlyflagdiff; before-first-conv activation; actual pathological-kernel elimination; "
        "warm backward below60 s and total below75 s on all four warm steps; finite loss/gradients; intended group updates; noOOM. "
        "Memory remains tight, and additional optimizations remain unapproved experiments.\n")
    answers = [
        ['Q1: Before first convolution?', 'Yes: first statement in actual train.main plus actual first Conv2D/3D flag assertions.'],
        ['Q2: Other production semantic changes?', 'None: exactly one added assignment; protected source/config/run integrity checked.'],
        ['Q3: Cold step?', f"{steps[0]['phase_sum_seconds']:.6f}s phase sum; setup {training['startup_seconds']:.3f}s separate."],
        ['Q4: Warm steps2–5 mean/median?', f'{total:.6f}/{median:.6f}s.'],
        ['Q5: Warm backward-only mean?', f"{wm['backward_only']:.6f}s."],
        ['Q6: Multiple production R2 shapes?', 'Yes: q135/84/39/24/21, in 3 real records with prefix-free/17f prefix. All fixed spatial geometry.'],
        ['Q7: Group-wise pathology gone?', 'Yes in all representative True probes and full warmed trace. R4/q135 xmma; smaller R2 often grouped-direct.'],
        ['Q8: Loss/grad health?', '5/5 finite; all10 groups nonzero; no missing trainable gradients; frozen violations0; representative updates5/5.'],
        ['Q9: VRAM safe?', f'NoOOM at {alloc:.3f} allocated/{reserved:.3f} reserved GiB; ~3.07GiB actual free is tight, not broad headroom.'],
        ['Q10: New #1 bottleneck?', f'{replay:.3f}s block replay plus distributed Adaptive/data-movement/elementwise work; not a lone 370s Conv kernel.'],
        ['Q11: Next3 safest investigations?', 'Immutable metadata; metadata+mapped PMA coordinates; metadata+final Adaptive/Global PE (nested alternatives). See measured caveats.'],
        ['Q12: Expected time/risk?', 'See section13 conditional block extrapolations, not full-model speedup claims. No extra optimization applied.'],
    ]
    add(table(["Question", "Answer"], answers) + "\n")
    add("### Reproducibility / artifacts\n\n"
        "- Production diff/snapshots: `production.diff`, `train.py.before`, `protected_before.json`, `final_integrity.json`.\n"
        "- Static: `static_verification.json`, `backend_defaults_cpu.json`, `cpu_regression.log`, `final_checks.log`.\n"
        "- Real5steps: `production_benchmark/result.json`, `step_*/{result,scopes,R_shapes,sample_runtime}.json`.\n"
        "- False control: `fresh_false_control/result.json`, `fresh_full_gradient_comparison.json`, selected dW `.pt` files.\n"
        "- Dispatch: `kernels_false/results.json`, `kernels_true/results.json`, per-case traces.\n"
        "- Warm full profile: `warmed_profile/step_02/trace.json` (~2.2GB), `raw_summary.json`.\n"
        "- Extra A/B: `constant_cache_probe/{summary,manifest}.json` and per-mode result files.\n"
        "- CPU result aggregation: `collected_summary.json`; diagnostic scripts are all under `tools/cudnn_benchmark_validation/`.\n\n"
        "All paths above are relative to the evidence root unless stated otherwise. "
        "No production process was launched or resumed; `hrlrvo_train` remains available for the user's later explicit instruction.\n")
    path = ROOT / "CUDNN_BENCHMARK_VALIDATION_AND_SPEEDUP.md"
    path.write_text("\n".join(text))
    print(path)


if __name__ == "__main__":
    main()
