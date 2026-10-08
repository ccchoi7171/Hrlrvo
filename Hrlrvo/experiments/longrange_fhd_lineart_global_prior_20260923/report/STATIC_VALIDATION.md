# Static validation evidence

- 100 copied baseline regression tests PASS: logs/baseline_regression_tests.log.
- 11 Global prior unit tests PASS: logs/lineart_core_tests.log.
- FHD geometry, hard-known prefix, noising/timestep, hybrid carry semantics, stitch counts PASS: logs/lineart_recurrence_preflight.log.
- Ruff critical syntax/undefined-name checks PASS: logs/static_lint.log.
- Only baseline module changed: global_attention.py. New module owns bias/runtime; runner/scoped call does not alter sample_chunk.
- C4 9x15 x9time=1215 tokens. Bias rows: generated-touching C4 cells with gi>0, time slices>=3. All global keys retained. After dual decompression, canonical known anchor/prefix output restored exactly.
- Prefix latent handling and RGB writer unchanged; C0 exact replay hash gate still required before C1.
- Sigma matched to official30-step schedule; repeated CFG/STG calls do not advance step. Batch-aware cache, one sigma read per immutable forward context.
- Latest schedule lambda0=.5:0..9=.5,10..19=.3,20..29=.1. No latent strength attenuation.

## Limitations
Rank-1 gi*gj is salience, not connectivity/orientation. Reference includes actor/light/reflection edges. Additive mask can select a different SDPA kernel, and strict known-output restoration requires extra Global attention/decompression. Actual runtime and geometry benefit require smoke results.
