# Original vs. cleaned Anchor sparse contact diagnostic

This is a read-only record of the input prepared for Experiment 2. It is not a
quality gate and does not stop the running 1025-frame inference.

## Bound inputs

- Original: `longrange_960x540_1025f.mkv`, SHA-256
  `34cc03832936d03dccc469fbd7b3666c7e3ff57253501f92a7cf62560dae9835`
- Cleaned FFV1 Anchor: `experiment_2/clean_anchor_1025f.mkv`, SHA-256
  `1f18cc5aee11b2f22de1108c5874e7fb91a66fc5cb291f401e23c6cf33c1bc77`
- Sampled frames: 0, 73, 120, 250, 480, 600, 800, 960, 1024.
- F120 was added as the camera/equipment inspection frame. The original image
  visibly contains the upper-left camera/jib and the combined cleanup mask is
  elevated there. This is direct image evidence, not an assertion that the mask
  is semantic ground truth.

## Direct observations

- The sampled performers are absent in the cleaned frames. At F120 the large
  upper-left camera/jib is also absent, and the replacement follows the broad
  ceiling appearance rather than leaving a solid camera silhouette.
- Differences are concentrated in the recorded cleanup support. Outside that
  support, sampled mean absolute RGB differences are 0.02–1.04 DN; inside it
  they are 34.51–58.60 DN. This shows that the operation was spatially focused,
  while still including registration/interpolation and color-fit effects.
- Broad cool-blue stage illumination remains visible. Across these sparse
  samples, cleaned full-frame mean RGB ranges approximately R 125.1–127.1,
  G 144.2–147.6, B 208.3–208.9. The original full-frame means are more variable,
  but those means are confounded by performers and equipment occupying the image.
- Small dark edge fragments remain visible near the lower-left edge at F600 and
  near the lower edge at F960/F1024. They are recorded as residual candidates;
  the sparse images alone cannot distinguish equipment, floor reflection, mask
  miss, or legitimate stage content.

## Interpretation limits

- The detector/equipment union mask is not foreground ground truth.
- A plausible donor fill at nine frames does not prove temporal stability.
- Full-frame color statistics cannot isolate lighting from removed foreground.
- These contacts document the actual cleaned Anchor used by the already-running
  full inference. Minor residuals are diagnostic and are not a stop condition.

## Artifacts

- `original_vs_clean_all_frames.jpg`: compact 9-frame overview.
- `original_vs_clean_early_camera.jpg`: larger F0/F73/F120/F250/F480 comparison.
- `original_vs_clean_late.jpg`: larger F600/F800/F960/F1024 comparison.
- `camera_frame120_upper_crop.jpg`: native-width upper-half camera comparison.
- `report.json`: input hashes and per-frame mask/difference/color measurements.
