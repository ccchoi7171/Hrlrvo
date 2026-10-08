# Inference-only line-art Global bias experiment

Baseline: strength_1p0 FHD recurrence; identical C0 RGB/prefix/final clean hybrid-carry/RNG state.
No production or prior experiment writes. All new files live here.

1. Preserve baseline copied source hashes and snapshot; copy input losslessly.
2. Derive static C0 first-frame Canny large-edge prior, visible-to-encoded pad mapping, Global9x15 token strengths.
3. Add independently disabled Global-only additive-logit bias; LATEST USER: scale0.5 at steps0-9,0.3 at10-19,0.1 at20-29; C0 disabled.
4. CPU regression tests before GPU: OFF exact route; formula/shape/RoPE/projection; repeated CFG calls same schedule; fixed prefix/anchor restrictions.
5. Fork exact C0 to private store; C1-C4 smoke against identical baseline chunks. Continue toC23 only if no obvious regression.
6. Save per-chunk RGB/resume states, previews, comparisons and patch.

Heuristic g_i*g_j encourages high-edge token co-attention, not true line connectivity, orientation, or a guarantee of straight geometry. Fixed image-space prior may conflict with camera motion.

## Exact C0 state recovery
Baseline alternating FP32 prefix/RNG slots overwritten by C22/C23. Replay C0 once using frozen copied baseline source, require original segment pixel SHA, clean carry file SHA, and FP32-prefix+RNG serialized SHA to match before any C1 launch. Then preserve immutable C0 bootstrap archive here for future branches. Never reconstruct FP32 prefix from MP4/uint8.
