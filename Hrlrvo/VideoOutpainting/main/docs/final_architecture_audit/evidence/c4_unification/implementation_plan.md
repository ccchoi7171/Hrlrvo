# C4 unification — bounded implementation plan

1. Snapshot latest production source/config and the unchanged R2/R4 class before editing.
2. Add CPU regressions before production edits: full-cell arithmetic/order, canonical 68×120 grid, invalid geometry/cell rejection, C4 ownership and two-consumer gradient sum, 41 Adaptive layouts.
3. Remove Global edge-cell construction and compressor valid-tap normalization. Reject invalid full-canvas geometry before token construction; preserve the same per-block C4 object and Conv2d initialization.
4. Keep Global attention/R4 arithmetic, all LoRA/Fusion/Adaptive/conditioning paths unchanged. Valid full-grid compressed H/W uses exact integer division, not ceil.
5. Run only selected CPU unit/geometry tests, compile and Ruff. No model forward, production training/inference, optimizer step, CUDA or process/cache manipulation.
6. Update audit documentation to the user's corrected fixed Global-grid R4 definition. Preserve historical evidence; write current test/diff evidence in this directory.

Architecture version remains v8: no parameter namespace, shape, initialization or accepted full-cell numerical semantics changes. Invalid geometry now fails closed instead of exercising the removed legacy fallback. Existing strict checkpoint metadata/key/provenance validation remains untouched.
