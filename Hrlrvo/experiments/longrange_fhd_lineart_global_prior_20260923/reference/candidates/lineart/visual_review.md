# Native-resolution thin LineArt candidate review

## Recommendation

- **Visual-detail recommendation: `detailed`** — best balance of continuous architectural outlines and retained dark-stage structure. It adds materially more wall/window/arch continuity than `balanced` without the dense microtexture of `very_detailed`.
- **Conservative first-injection fallback: `balanced`** — safer if the attention bias proves overly rigid or performer/reflection regions dominate. It retains principal structural boundaries while producing substantially lower global-cell density.
- `very_detailed` is **not recommended for the first inference**: foliage, people, reflections, signage texture and small lighting edges become dense, so the map is less specifically architectural.
- `clahe_lowlight` recovers low-light edges, but visually resembles `detailed` while adding low-light texture. It is useful only if dark wall boundaries are shown to be absent from `detailed`.

## Visual evidence

- `contact_sheet_fullview.png`: earlier coarse reference plus all four native-resolution candidates.
- `contact_sheet_crops.png`: left window, right window, center stage and floor comparisons at native-map detail.
- Individual black-on-white views and exact binary crops are stored alongside each candidate.

## Important limitation for the proposed current bias

All candidate maps contain native one-pixel Canny lines, but the current proposed Global bias reduces them to one scalar density per **128×128-pixel Global cell** and then forms `g_i * g_j`. Therefore:

1. thinness, orientation and connectivity do not survive the 9×15 reduction;
2. “more detailed” can become merely “higher cell density” rather than a better structural relation;
3. people and floor reflections also increase the same scalar density;
4. line continuity would require a richer orientation/connectivity relation, which is outside this minimal experiment.

For the exact `g_i * g_j` experiment, `detailed` is the preferred visual candidate, but compare against `balanced` if the bias shows excessive scene locking. Start with the requested conservative attention scale; do not infer that a denser line map is necessarily a stronger structural prior.
