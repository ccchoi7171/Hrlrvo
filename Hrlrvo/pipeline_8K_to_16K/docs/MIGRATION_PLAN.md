# Migration execution plan
1. Read-only source/baseline inventory and free-space check.
2. Copy packaged source/models/manifests plus input/preprocessed data; SHA source and destination. No symlinks, no original writes.
3. Reuse all model computation; adapt only runner/stage layout; add shared stage-only dispatch and independent FIFO queue.
4. CPU run actual Stage0001 for05 into validation directory; compare first65 decoded rgb24 frame hashes against copied historical PreProc.
5. ONLY if this exact identity gate passes: evaluate eligible05 baseline, then at most1 new Outpainting inference (or sequential reference+new if baseline unavailable). Never FlashVSR inference in this task.
6. On identity FAIL/OOM: stop only GPU validation, finish static/CPU tests and honest docs.
7. Review hashes/diffs, configs, schema, path protections, locks and queue tests; preserve reference data.

Observed pre-run mismatch to investigate: historical05 PreProc is CRF18; audited preprocessing code is CRF1. Neither is modified to hide the mismatch.
