# Isolation and rollback

Only this experiment source copy is modified. Original VideoOutpainting, pipeline, and preceding FHD baseline source/results are read-only. No production rollback is required.

The model change is Global-only additive logit bias; no parameters/training changes. Local/Adaptive/Cross-Attn/FFN/inference.py remain byte-identical to copied baseline. Global restore of known positions protects prefix/anchor despite the existing decompressor temporal/spatial refinement.

Flags: --lineart-global-prior / --no-lineart-global-prior, --lineart-bias-scale (default0.5). OFF or scale0 bypasses the bias and masked attention path exactly. Fixed zero-based30-step schedule:0-9=1x,10-19=.6x,20-29=.2x.

Rollback: run with --no-lineart-global-prior in a DIFFERENT output mode; do not resume an existing ON run with changed configuration. Alternatively restore ONLY this copy's global_attention.py from patches/original_src/ltx_outpaint/global_attention.py. Never edit the original or baseline tree.

C0 frame is from saved lossless uint8 encoder-input segment, NOT an MP4 decode. C0 prefix recovery must pass report/c0_recovery_verification.json before bootstrap.

## Selected detailed reference launch

User-selected detailed.png maps exactly to reference/candidates/lineart/detailed_global_line_strength.npy (9x15). Config JSON/YAML now explicitly select this file; runner validates SHA before GPU model load. Original coarse reference files remain unchanged. See report/detailed_reference_launch_validation.json.

Launch: bash commands/run_lineart_detailed_C23.sh (bootstrap exact C0, generate C1-C23); never run this again over committed output. Resume: bash commands/resume_lineart_C23.sh.
