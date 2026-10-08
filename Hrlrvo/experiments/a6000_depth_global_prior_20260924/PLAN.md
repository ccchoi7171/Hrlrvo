# A6000 GPU0 D5 depth Global prior

User-approved invariants: isolated new experiment; only GPU0; other three inference processes and H200 lineart source untouched; base seed71634910 + absolute frame; checkpoint7500; strength1.0; latent carry ON; temporal rebase ON; 65/17/48 geometry and24fps FHD.

1. Freeze exact H200 lineart inference modules. Keep Global logit bias, fine write mask, sigma schedule0.5/0.3/0.1 and all other model source byte-identical. Only replace reference g with normalized D5 inverse-depth values; no new similarity kernel/global boost/halo.
2. On A6000 read baseline arm01 committed C0 lossless segment frame0; reuse exact C0 prefix/RNG/carry, not H200 images and not baseline C1+.
3. Reproduce D5_v2_base_518: pinned Depth-Anything-V2-Base-hf revision b1958afc87fb45a9e3746cb387596094de553ed8, CPU float32, processor518 (actual518x924), original-size postprocess, percentile0.5/99.5 normalized white-near float map. Pad36px top/bottom, area-mean32x32 then4x4 ->9x15. No line-density divisor0.05 (not meaningful for depth). Preserve raw map and provenance.
4. CPU tests: exact known-output protection and disabled-path baseline, mapping/normalization, fail-closed C0 digest and file identity checks, runner scope and pinned config.
5. Copy-only bootstrap C0; restore C0 RNG after model load; generate C1..C23 with same sequence seed rules. Verify no C1+ baseline records reused.
6. Launch only in remote tmux hrlrvo, occupancy/file-lock guards, CPU preview watcher. Verify real GPU0 producer and C1/prior activation evidence, other PIDs unaffected; report ongoing not completed.

Risk: rank1 gi*gj now emphasizes relative inverse-depth salience (nearer positions), not learned depth conditioning/same-depth affinity. Benefit unproven. Dual Global attention/decompression overhead inherited unchanged. Existing baseline C0-C3 remain preserved.

## Recovery finding / updated safe execution
Baseline streaming_video uses previous_prefix_{chunk%2}.pt. Actual originalC0 prefix digest differs because C2 overwrote slot0. OriginalC0 pixels+carry remain valid. Recover only the missing exact FP32prefix/RNG by C0-only GPU0 replay in isolated recovery, with original baseline22modules and only end_chunk0 runner change. All originalC0 pixel/prefix/carry digests and segment file bytes must match before bootstrap. No H200state, no uint8 prefix reconstruction, no relaxation of C0 provenance.
