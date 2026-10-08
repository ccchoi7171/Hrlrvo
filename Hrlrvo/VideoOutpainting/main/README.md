# LTX-2.3 Video Outpainting

Single canonical hidden state; **Local System versus Global**, installed only
at the video self-attention (`attn1`) of every native LTX block. No alternate
transformer, tracker, flow, memory, retrieval, FFN or cross-attention branch.

## Status and data boundary

The missing-Anchor prerequisite is replaced by a **lazy independent RGB Anchor
cache**. Existing 4K GT latents stay read-only and supply the noisy/loss target
(and the existing causal clean-prefix condition). Only a selected, matching raw
RGB interval is read to crop the observed Anchor **before** VAE encoding. No
full-GT latent crop is used as an Anchor fallback. Cache misses are encoded by
the owning training rank, never by CUDA DataLoader workers; valid hits skip VAE
encoding. Mismatched sources, temporal intervals and corrupt entries fail closed.

Architecture/checkpoint **v8** changes only Adaptive token construction. Each
Adaptive rectangle is expanded in-canvas to a complete local C4 lattice; a pure
Bridge parent remains one C4 context token, while every other parent is promoted
to four complete C2 tokens. Only C2 cells touching the Target are queries.
Local fusion is applied only on their restored Target pixels; every other Local
pixel is the exact Native result. V8 keeps v7's 50:50 gate initialization and
identity-initialized decompressor refinements, plus
the unchanged v6 LoRA layout and parameter count. Pre-v8 checkpoints are
rejected rather than silently adopting different token semantics.

All three self-attention views share one zero-delta Outpainting Q/K/V LoRA,
while Adaptive and Global add independent Q/K/V specializers. The final shared
output projection receives its shared Outpainting O delta, and the one native
FFN receives a separate shared FFN LoRA. There is no Native-specific adapter.
The released official IC-LoRA, appended-reference protocol and RGB sentinel
conditioning remain unsupported; the Hrlrvo shared Outpainting LoRA is trained
from scratch and is not that artifact. V8 PMA uses only the nominal C2/C4 grid;
there is no centroid, support-offset or component-splitting position path.
Inference/validation now independently encodes Anchor and full-known prefix RGB
before latent placement, rather than encoding a constant-filled RGB canvas.
Training cache, masks/noise/loss, prefix strategy and attention/fusion hierarchy
are unchanged. Verification of this Adaptive change is **code-level,
meta-device and small CPU only**,
including small view-graph gradient ownership checks and mocked conditioning
encoders. No new VAE encode/cache creation, full LTX forward/backward,
optimizer step, training, generation or GPU benchmark is performed for this
change. Earlier GPU smoke results do not validate v8 Adaptive tokenization.

## Tree

```text
main/
  README.md
  configs/
    train_4k.yaml
    inference_4k.yaml
  src/ltx_outpaint/
    __init__.py
    lora.py
    model.py
    outpainting_adapter.py
    geometry.py
    native_local_attention.py
    adaptive_local_attention.py
    global_attention.py
    compression.py
    position.py
    anchor_cache.py
    data.py
    train.py
    inference.py
```

## LongRange inference memory and resume

Production LongRange inference uses `ltx_outpaint.streaming_video` rather than
the upstream whole-video `read_video`/`save_video` helpers. Source frames are
decoded sequentially into a config-derived RGB24 overlap ring. Each generated
chunk keeps the existing decode, spatial composite, and exact temporal-prefix
overwrite order; the next prefix remains CPU FP32, while only finalized frames
are atomically committed as lossless encoder-ready uint8 `.npy` segments.

After generation, the segments are memory-mapped in order and fed one frame at
a time to the same `libx264`/`yuv420p`/CRF 18 PyAV encoder settings. The uint8
segment format is intentional: it is exactly the pixel representation accepted
by the old final encoder, is four times smaller than FP32, and avoids a lossy
intermediate decode/re-encode. The recurrent prefix is never reconstructed from
segments. Resume state also stores the post-chunk CPU/CUDA RNG state because the
native diffusion VAE decoder may consume the global device RNG.

The default segment directory is `.<output filename>.segments` beside the final
MP4 and is removed only after successful encoding and manifest writes. Use
`--keep-intermediates` to retain it, and rerun the identical command with
`--resume` (and the same `--segment-dir`, when explicitly set) after an
interruption. Atomic chunk records are the completion markers; partial files are
never treated as completed chunks.

### Optional inference GPU-memory switches

`configs/inference_4k.yaml` keeps all three `memory_optimization` switches
**false** by default, so the checkpoint load, FFN, and upstream VAE decode path
remain the production reference. Each can be overridden independently with
`--video-only-gpu-weights` / `--no-video-only-gpu-weights`,
`--vae-decode-streaming` / `--no-vae-decode-streaming`, and
`--ffn-token-chunking` / `--no-ffn-token-chunking` (`--ffn-chunk-tokens N`).

* Video-only residency runs the existing full base loader and validated adapter
  load on CPU first, without dropping checkpoint keys. Only video/shared
  parameters move to GPU; enumerated Audio/AV modules skipped when `audio=None`
  remain on CPU. Do not subsequently call `.to(cuda)` on the whole transformer,
  which would undo this placement.
* VAE streaming calls the same tile planner and unchanged `tiled_decode`
  reconstruction. It normalizes each completed temporal region with the same
  BF16 expression and transfers FP32 regions to CPU before the next yield,
  avoiding full-video GPU `cat` and FP32 conversion. The returned CPU chunk
  still enters the existing spatial composite, FP32 prefix, segment, and resume
  path. The original upstream `_decode_video` remains the OFF path.
* FFN chunking invokes the same native video FFN and its installed LoRA on
  consecutive **token-axis** slices, then writes them into original order.
  It is inference-only and OFF by default. A controlled BF16 comparison of the
  active checkpoint's 73,440-token block-1 FFN found non-bitwise output at
  4,096, 8,192, and 16,384 tokens (8,192: max absolute difference 0.0625).
  Different GEMM shapes therefore change rounding despite identical weights
  and mathematical operations; leave this switch OFF when exact generation
  equivalence is required.

The active LTX-2.3 checkpoint uses the convolutional VAE decoder and the
upstream fixed `48/24` temporal, `384/192` spatial tile plan. Other checkpoints
with a diffusion VAE use a free-VRAM-aware planner; moving unused weights can
then change its selected plan, so numeric equivalence must be checked for each
enabled combination.
`torch.cuda.empty_cache()` is not an enabled optimization here and cannot lower
the live allocated FFN peak.

### Optional decoded-prefix modes

The default `normal_long_range` mode remains unchanged. `prefix_seed_search`
generates the first config-defined `prefix_frames` (17 for the active config)
from source frames 0–16 with the same chunk sampler and memory flags. It loads
the model once for all requested seeds and saves per seed:

* `seed_<seed>_prefix.pt`: exact decoded full-canvas **CPU FP32 RGB FCHW**
  tensor for continuation (about 1.576 GiB per 3840×2160/17f candidate);
* `seed_<seed>_preview.mp4`: visual-only x264 preview, never used as a condition;
* `seed_<seed>_meta.json`: seed/source/checkpoint/config lineage, tensor shape,
  dtype, SHA-256, selected memory flags, and GPU peak measurements.

Metadata is committed after both artifacts. A continuation loads the `.pt`
tensor with `weights_only=True`, verifies SHA-256, CPU FP32 geometry, source
content and checkpoint adapter, then passes it into **chunk 0's actual
conditioning**. It is not a post-generation replacement. The original
FP32 decoded-prefix overwrite and 65/17/48 recurrent LongRange semantics
remain in force; resume identity includes the selected tensor hash. No prefix
is read from the lossy MP4 preview.

```bash
CKPT=../runs/hrlrvo_main_train_v1/checkpoints/checkpoint-007500
SOURCE=/path/to/1920x1080_24fps_source.mp4

# Existing normal path, with the independently verified exact memory switches.
python -m ltx_outpaint.inference --config configs/inference_4k.yaml \
  --anchor-video "$SOURCE" --output outputs/normal.mp4 \
  --checkpoint "$CKPT" --seed 33333333 --steps 30 \
  --video-only-gpu-weights --vae-decode-streaming --no-ffn-token-chunking

# Consecutive seeds: --seed BASE --num-prefix-samples N; alternatively use
# --prefix-seeds 33333333 33333334 ... for an explicit candidate list.
python -m ltx_outpaint.inference --config configs/inference_4k.yaml \
  --mode prefix_seed_search --anchor-video "$SOURCE" \
  --prefix-output-dir outputs/prefix_candidates \
  --checkpoint "$CKPT" --seed 33333333 --num-prefix-samples 2 --steps 30 \
  --video-only-gpu-weights --vae-decode-streaming --no-ffn-token-chunking

python -m ltx_outpaint.inference --config configs/inference_4k.yaml \
  --mode long_range_from_prefix --anchor-video "$SOURCE" \
  --initial-prefix outputs/prefix_candidates/seed_33333333_prefix.pt \
  --output outputs/from_prefix.mp4 --checkpoint "$CKPT" \
  --seed 33333333 --steps 30 \
  --video-only-gpu-weights --vae-decode-streaming --no-ffn-token-chunking
```

To keep segment/RNG checkpoints after success, pass `--keep-intermediates`;
after interruption rerun the **identical** command with `--resume`. The
continuation rejects an initial tensor from a different source/checkpoint or
with a mismatched hash/shape. Search candidates require enough disk capacity:
200 uncompressed FP32 4K/17f tensors require roughly 315 GiB before previews.

## Source provenance

The upstream LTX checkout is reused in place, without modifications.

| Reference | Pinned source and reused component |
|---|---|
| LTX-2.3 | [`Lightricks/LTX-2@598ab41247a77dbfe29b5186e915bcf4f9040ec7`](https://github.com/Lightricks/LTX-2/tree/598ab41247a77dbfe29b5186e915bcf4f9040ec7): `packages/ltx-core/src/ltx_core/model/transformer/{model,transformer,attention,transformer_args,rope,ops}.py`; trainer `trainer.py`, `training_strategies/flexible.py`, `datasets.py`, `model_loader.py`, `validation_runner.py`; VAE temporal mapping and patchifier. This checkout supports other versions too; the local **2.3 checkpoint metadata**, not newer model defaults, selects this model. |
| UltraGen | [`wan_video_dit.py@4f9ec33f99d49cd533435ee12e73b83c26aa6c53`](https://github.com/sjtuplayer/UltraGen/blob/4f9ec33f99d49cd533435ee12e73b83c26aa6c53/diffsynth/models/wan_video_dit.py): QKV adapters (128–160), C/R and timestep-channel gates (161–183), hierarchical/global paths (242–331), two fusions and shared O (343–372), single CA/FFN after SA (449–475). |
| Foveated Diffusion | [`pipeline.py@694f493f1ac34375a43b0c243dd51a9f7bdae1b9`](https://github.com/bchao1/foveated_diffusion/blob/694f493f1ac34375a43b0c243dd51a9f7bdae1b9/src/diffsynth_fov/pipeline.py), packing/unpacking masks; [`dit_video.py`](https://github.com/bchao1/foveated_diffusion/blob/694f493f1ac34375a43b0c243dd51a9f7bdae1b9/src/diffsynth_fov/dit_video.py), matched feature/coordinate packing and query-scale attention. |
| Phase-Aligned RoPE | [`wan/transformer_wan.py@0825c5aac01db94f229cfcd868c006fb6d347200`](https://github.com/cvlab-stonybrook/mixed-res-dit/blob/0825c5aac01db94f229cfcd868c006fb6d347200/wan/transformer_wan.py), query-specific attention (134–255) and coordinate maps (668–701); [paper §4.1, Eq. 7–8](https://arxiv.org/html/2511.19778). |
| Official In/Outpainting IC-LoRA | [HF revision `554060ba18950e5fd5e91e6a9ab2f55852fd05a4`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/tree/554060ba18950e5fd5e91e6a9ab2f55852fd05a4); existing local artifact, inspected but **not integrated**. [Official Comfy workflow](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d/example_workflows/2.3/LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled.json) and the same revision's `iclora.py`, `vanish_nodes.py` establish its reference preprocessing. |

### Public-source discrepancies, not silently copied

UltraGen's executed bespoke QKV LoRA has rank 64, biased A/B Linear layers,
default PyTorch initialization and **no alpha**. Its training CLI's alpha 32 is
not wired to these adapters. **That biased/random-B LoRA is no longer used.**
V6 follows the pinned LTX trainer's standard PEFT convention described below.
UltraGen's C/R and fusion layers also have no custom initialization. Therefore the
required **average-initialized C** is an explicit specification-driven repair,
not an upstream initializer. V7 also explicitly identity-initializes R's
depthwise refinement; this is a Hrlrvo initialization policy, not an UltraGen
default. The spatial interpolation and Conv3d architecture remain unchanged.
The large Wan fusion MLPs are replaced by native-LTX-gate channel
affines described below; denoising conditioning, per-block channel weights and
the two fusion stages are retained. The global base-K typo (`self.q`), batch=1 assumption,
resizing to divisible dimensions, fixed layer exceptions and D=1536 constant
are not copied. All blocks participate, with D and block count read from LTX.

Foveated/PMA's aligned implementations do not specify irregular-support centroids
or component splitting. These were **Hrlrvo-specific v3 rules**, not exact
methods attributed to either paper. The query-native scale principle is reused.
The centroid rule was retained in v4 and removed in v5 because its phases were
not identical to v2 origin-based phases. V8 keeps nominal-grid phase positions
for its complete C2/C4 cells; incompatible checkpoints are not converted.

## Architecture and coordinates

`install_outpainting` freezes every pretrained parameter and preserves the
native Linear objects and state keys. It installs the hierarchical `attn1`
path plus delta-only LoRA modules on `attn1` and the two existing FFN Linear
targets; it does not clone the pretrained projections. The original
`BasicAVTransformerBlock` still performs native normalization/modulation,
SA residual, **one text CA and one FFN**. Native head gating is applied on the
fused canonical output immediately before the single shared `to_out` call.
STG keeps the official V-bypass/control-flow semantics while v8's shared
Outpainting V and O deltas remain active on that path.

```text
Local [per-window Native + Adaptive -> Local Fusion]
    -> normalized coordinate assembly -> LocalFull
Global [C_S -> Global SA -> R_S] -> GlobalFull
LocalFull + GlobalFull -> Local-Global Fusion
    -> shared (W_O + Delta_OUT,O), exactly once
    -> native SA residual -> frozen Text CA, exactly once
    -> shared (frozen W_FFN + Delta_FFN), exactly once -> next block
```

This remains a two-level hierarchy, not a Native/Adaptive/Global three-way
fusion. Text CA parameters are frozen without a `torch.no_grad` boundary, so
autograd still propagates through the operation to preceding trainables.

`OutpaintingModality` carries an immutable `CanvasGeometry`. The preprocessor
places its positions, sigma and native RoPE in the checkpointed argument object;
backward recomputation does not consult mutable per-forward geometry.

### Local Fusion — per target window

Even blocks use K² windows and odd blocks (K+1)²: **16/25** for K=4. For each
target `L_j`, Native reads exactly that window using frozen base Q/K/V plus the
shared Outpainting Q/K/V LoRA. Adaptive reads
the minimum rectangle containing a same-sized geometry-selected Anchor patch
and `L_j`, then minimally expands it toward the canvas center to a complete,
in-canvas C4 lattice. Anchor wins role classification over Target, and Target
wins over Bridge. Pure-Bridge parents remain C4; all Anchor- or Target-touching
parents become four complete C2 cells. Adaptive computes Q only for C2 cells
that touch Target; every emitted C2/C4 token remains K/V context, including the
Target queries themselves. Each query output is restored to its complete 2x2
footprint, but only pixels resolved as Target are written into the Local output.
The same write mask gates Local fusion: Target pixels evaluate
`a_local * Native_j + (1-a_local) * AdaptiveTarget_j`, while every non-Target
pixel passes through `Native_j` exactly. The mask does not clamp, detach or
freeze the source hidden state.
All windows share the same base and adapter objects within a block. Native,
Adaptive and Global reuse the **same** Outpainting Q/K/V LoRA parameters;
Adaptive and Global additionally own separate branch-specialization LoRA.

### Global Fusion — full canvas

After **all per-window fusions**, results are scattered to original coordinates
and divided by overlap counts. This gives one `LocalFull[B,T,H,W,D]`.
`C_S(full X) -> Global SA -> R_S` independently restores precisely the same
full-canvas shape. Then `a_global * LocalFull + (1-a_global) * GlobalFull` is
computed and the frozen shared/native `W_O` plus its shared Outpainting O LoRA
is invoked **once**. No branch owns an output projection or O adapter.

Each block has two separate channel gates, using its **already computed native
`vgate_msa`**, not a second timestep encoder:

```text
g_i = native get_ada_values(..., video.timesteps, slice(0, 3))[2]
alpha_local_i  = sigmoid(gamma_local_i  * g_i + beta_local_i)
alpha_global_i = sigmoid(gamma_global_i * g_i + beta_global_i)
gamma_*: [D], initialized to 0
beta_*:  [D], initialized to 0 (logit(0.5))

local_fusion_init_native_weight: 0.5
global_fusion_init_local_weight: 0.5
```

Native `AdaLayerNormSingle` prepares `video.timesteps`; the block adds its own
`scale_shift_table` and unpacks `vshift_msa/vscale_msa/vgate_msa` in
[`transformer.py:191–200,272–286`](https://github.com/Lightricks/LTX-2/blob/598ab41247a77dbfe29b5186e915bcf4f9040ec7/packages/ltx-core/src/ltx_core/model/transformer/transformer.py#L191-L286).
For per-token conditioning, `g_i` has shape `[B,T*H*W,D]`; the native `[B,1,D]`
case broadcasts without pooling. Neither the gate nor its coordinates is
spatially compressed. Local alpha is sliced at the exact restored Target window
to `[B,T,H_j,W_j,D]`; Global alpha is `[B,T,H,W,D]`. For every finite native gate,
the initial weight is exactly 0.5 in every token/channel:

```text
Local[p] = 0.5 Native[p] + 0.5 Adaptive[p]  if p is resolved Target
Local[p] = Native[p]                        otherwise
Final = 0.5 LocalFull + 0.5 GlobalFull
```

On resolved Target support, the effective **25% Native / 25% Adaptive / 50%
Global** contribution is the result of **two hierarchical 50:50 fusions**, not
a flat three-way mixture. Outside resolved Target support, Local is exact Native,
so the initial final mixture is **50% Native / 50% Global**. Local windows are
still assembled with the same normalized overlap rule before Global fusion.
The config names identify the favored operand independently: Native in Local
Fusion, LocalFull in Local–Global Fusion. Their defaults are both 0.5.
Both beta and gamma remain trainable; after initialization the unchanged
`sigmoid(beta + gamma * vgate_msa)` can learn channel-wise, token/timestep-dependent
ratios. There is no hand-crafted sigma schedule or removed native conditioning.

Two method bindings scope capture of the native gate around the unchanged
native block forward. The block class/state keys and the original residual gate
are preserved; no AdaLN computation is repeated. A resettable `ContextVar`
isolates invocations, including non-reentrant checkpoint recomputation, without
retaining a gate tensor on the module. This integration uses eager native blocks;
`torch.compile` capture is explicitly rejected rather than silently losing the
gate. Compiled/CUDA-graph execution has not been validated.

### Partition and Anchor rules

For length L and count P, window size is `ceil(L/P)`. For P>1, starts are
`floor((j*(L-size) + floor((P-1)/2))/(P-1))`. Thus the first start is zero,
the last end is L, and only the unavoidable `P*ceil(L/P)-L` overlap occurs.
For L=68,P=5: size 14, starts **0,14,27,41,54**. There is no discarded token,
spatial resize or feature-map padding. Overlap weights are uniform counts.

On each axis, a contained target preserves its position (clamped inside Anchor).
Otherwise the **Target-center versus Anchor-center direction** chooses the
corresponding Anchor boundary. The selected patch is exactly Local-sized. An
Anchor too small to contain that patch is explicitly rejected; it is not resized.

### Mixed-density packing and C/R

K and compression scale remain independent; v8 keeps the configured `S=4` and
uses an explicit **C4-first, C2-promotion** policy. The raw Adaptive bbox remains
`MinBBox(selected Anchor patch, Target Local)`. Each axis is minimally expanded
to a length divisible by four, preferring expansion toward the canvas center
and staying inside the canonical canvas; exact center ties expand right/down.
The C4 lattice starts at this expanded
support's actual canonical origin; it is not snapped to a separate global grid.
Content is not padded, resized or cropped. Geometry that cannot produce the
required complete in-canvas C4 parents fails closed.

Every canonical pixel in the aligned bbox receives exactly one semantic role in
priority order **Anchor > Target > Bridge**. This is role resolution, not the
known/generation mask and not a source-hidden freeze. After the selected patch
defines the initial bbox, Anchor role covers the full observed-Anchor
intersection with the expanded support, not only that selected patch. Each
complete 4x4 parent is then handled atomically:

* if all 16 pixels resolve to Bridge, emit one full C4 Bridge context token;
* otherwise split the parent into its four complete 2x2 children and emit all
  four C2 tokens;
* a promoted C2 becomes a query exactly when at least one of its four resolved
  pixels is Target;
* every emitted token, including query tokens, participates in K/V.

Adaptive compression therefore never creates partial tokens, field-fragment
tokens, zero-filled supports or padding tokens. A promoted C2 summarizes all
four current-block hidden vectors in its nominal cell; its Anchor/Target/Bridge
pixel roles control query selection and writeback, not which source hidden is
allowed into that full-cell feature. Tokens keep canonical temporal coordinates
unchanged, and packed sequence indices are never used as spatial positions.

There is exactly one compressor for each scale **per block**:

* `C_2`: shared by all promoted Adaptive C2 tokens.
* `C_4`: shared by Adaptive pure-Bridge C4 tokens and Global.

C uses depthwise `Conv2d(D,D,k=s,stride=s,padding=0,groups=D,bias=True)`;
weights `1/s²`, bias zero. Every C2/C4 cell is complete and executes the actual
learned convolution over all 4/16 current hidden vectors. Adaptive and Global
call the **same per-block C4 module and forward primitive**; only the subsequent
attention interpretation differs. Partial, masked, edge-specific and
valid-count-normalized compression are not supported. Global geometry requires
`H % 4 == W % 4 == 0` before emitting cells: canonical `68×120` gives `17×30`
full C4 cells. Nondivisible geometry fails closed instead of padding, cropping
or dropping a remainder. C2 and C4 remain separate learned modules.

R uses spatial bilinear interpolation to the **explicit original H/W**
(`align_corners=False, antialias=True`) followed by depthwise
`Conv3d(D,D,kernel=3,stride=1,padding=1,groups=D,bias=True)` with Hrlrvo's
identity-preserving refinement initialization: all weights zero except
`weight[:,0,1,1,1]=1`, and bias zero. Thus `refine(x)=x` initially, including
padding boundaries. The complete R is **not** an identity on the original
high-resolution hidden: it remains existing spatial upsampling followed by
identity refinement. This applies independently to Adaptive's `R_2` and
Global's `R_4`; no new sharing or module is introduced. Time length is
unchanged. Adaptive reconstructs only query tokens, one complete 2x2 cell at a
time, and scatters only Target-role pixels. Anchor/Bridge contribution counts
are exactly zero. Global is decompressed in full. No ConvTranspose/PixelShuffle.

**Temporal-refinement audit:** both R paths have kernel **(3,3,3)**, stride
**(1,1,1)**, padding **(1,1,1)** and groups **D**, in `[B,D,T,H,W]` order after
per-frame spatial interpolation. Their trainable kernels can mix latent-time
`t-1,t,t+1` without compressing T; v7/v8 start with zero off-center taps, so the
refinement introduces no neighbor mixing until those taps learn. The operators match UltraGen's
[`Conv3d` definitions](https://github.com/sjtuplayer/UltraGen/blob/4f9ec33f99d49cd533435ee12e73b83c26aa6c53/diffsynth/models/wan_video_dit.py#L176-L183)
and [Resize → Conv3d execution](https://github.com/sjtuplayer/UltraGen/blob/4f9ec33f99d49cd533435ee12e73b83c26aa6c53/diffsynth/models/wan_video_dit.py#L324-L331).
Upstream leaves torchvision Resize defaults implicit and does not pin a single
torchvision version; the explicit local antialias setting is not claimed to
match every historical upstream environment.

The 17-RGB-frame prefix occupies three latent frames, so the interior boundary
is latent index **2 ↔ 3**. Both directions of hidden-feature mixing are valid:
native LTX attention is explicitly
[`is_causal=False`](https://github.com/Lightricks/LTX-2/blob/598ab41247a77dbfe29b5186e915bcf4f9040ec7/packages/ltx-core/src/ltx_core/model/transformer/attention.py#L97-L102).
Known conditioning constrains the denoising state/timesteps/loss, not every
Transformer hidden state. Post-prefix outer tokens are noisy targets during
training and evolving generated states during inference, not clean future GT
conditions. Mixing those hidden states is not clean-GT conditioning leakage.
This differs from the causal VAE encoding requirement for a clean prefix.
No conflicting prefix semantics or new clean-future-GT path was found in the
source audit, so both decompressor **operators remain unchanged**; v7 changes
only their refinement initialization to identity.

Native LTX's selected attention interface has no varlen/cu_seqlens argument.
Adaptive evaluations therefore use geometry/shape buckets, batching equal
Anchor geometry across arbitrary B. There is no spatial or sequence padding.

### Target-touching C2 queries and nominal-grid PMA (v8)

LTX uses per-token `[start,end)` positions, temporal coordinates in seconds,
and midpoint coordinates normalized by the checkpoint's max positions. It
generates its own log-spaced frequencies and applies the checkpoint's SPLIT
head layout. Packed sequence indices are never spatial positions.

Q is projected only for promoted C2 cells whose resolved role map touches at
least one Target pixel. K/V contains every emitted pure-Bridge C4 token and all
promoted C2 tokens, including the query tokens themselves. Base QKV, the shared
Outpainting QKV LoRA and the same per-block Adaptive LoRA are reused; this is a
query-restricted row selection from Adaptive self-attention, not a new
cross-attention module. Pure Anchor C2 and pure Bridge C4 tokens are context
only. No C4 query or fine-key subsampling pass exists.

For nominal canonical cell origin `o`, key scale `r_k` and every C2 query's
fixed scale `r_q=2`:

```text
r_query = 2
P_key = o / r_key + 1/2
P_key_PMA = (P_key - 1/2) * (r_key / r_query) + 1/2
          = o / r_query + 1/2                 # x and y independently
```

Every query and C2 key therefore uses its native nominal C2 point; each C4 key
is phase-aligned into the C2 query scale by the factor `4/2`. Positions depend
only on the support-origin nominal grid in canonical coordinates, never packed
index, semantic mixture, support centroid or target coverage fraction. Native
temporal bounds, LTX frequency/base/max-position and head layout remain unchanged.
Global `phase_rope` is unchanged; its compressor accepts only full C4 cells and
rejects nondivisible Global geometry.

For each query spatial cell, its `[B,T,1,D]` output is passed through the
existing `R_2` to an exact `[B,T,2,2,D]` footprint. Scatter/writeback intersects
that footprint with the resolved Target pixels and the original Local bounds.
Anchor and Bridge pixels contribute zero to the output map; disjoint C2 cells
write each resolved Target pixel exactly once. No restored Anchor/Bridge output
reaches the canonical canvas. If role resolution leaves no Target pixel,
Adaptive returns an all-zero Local-shaped contribution and an all-false write
mask before tokenization/attention. Local fusion uses that mask to return the
Native Local result exactly. That early
exit does **not** detach or freeze the source hidden state; Native/Global paths
and later blocks continue to process the canonical hidden normally.

**Historical phase note:** v3/v4 placed fine/coarse origin cells at
`(0.75,1.25)` for query scale 2 because they used actual-support centroids. V5
restored the literal v2 phases `(0.5,0.5)`. V8 keeps that nominal-grid phase
definition but removes partial Adaptive cells entirely through bbox alignment
and whole-parent promotion. Older centroid/partial-support checkpoints are
incompatible and are never converted.

The earlier 17-case partial-support CPU audit belongs to the retired v5-v7
Adaptive layout; it is not evidence that v8's promotion/writeback path runs.
V8 adds no parameters and preserves the existing C/R/fusion module objects and
hierarchical call sites. No GPU/full-model success is claimed for this change.

## Conditioning, cache and LongRange

Audit on 2026-09-10:

* 30,024 target `.pt` files; exhaustive filename/metadata-ID parity; all metadata
  records have at least 65 source/cached frames, at 24 fps.
* Official keys: `latents,num_frames,height,width,fps`. Audited tensors are
  BF16 `[128,F,68,120]`; variable temporal F, not pre-fixed 65-frame clips.
* Cached RGB extent 3840×2176 comes from 3840×2160 with 8 normalized-zero rows
  top/bottom. This existing pixel/VAE boundary policy is not SA padding.
* The actual local pretrained checkpoint SHA-256 was read in full and matches
  the cache contract: `7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f`.
* Checkpoint-derived VAE factors are `(time=8,height=32,width=32)`; encoder output
  is already per-channel normalized. No second scaling is applied.
* No text `conditions/` cache exists. Configuration explicitly selects the empty
  constant prompt. Official Gemma feature extraction is run once and preserves
  both its video-side and audio-side preconnector features plus the attention
  mask. The audio-side text feature is required by LTX-2.3's AV embeddings
  connector; it does not activate audio generation.
* Header/probe checks do not certify every tensor's contents in the ~958 GiB
  cache; records are also checked when read. Source ordering/scaling provenance
  comes from the cache contract/metadata, not from reading raw videos.

65 RGB frames correspond to **9 latent frames**, not `65//8`. Random aligned
temporal slicing uses official `map_temporal_slice`: latent `[i,i+9)` maps to
RGB `[8i,8i+65)`. Cropped cached slabs are not claimed to be identical to newly
encoding independent RGB crops. A 17-frame prefix corresponds to **3 latent
frames**. Spatial known Anchor and Bernoulli-selected full-frame prefix masks
are unioned at runtime. Known tokens receive clean condition, timestep zero,
and no generation loss. Unknown outer tokens alone contribute the official
flow-matching target loss. The density map never depends on these masks.

### Lazy independent 2K Anchor cache

Configured roots have separate roles:

* `/home/work/ai-graphics_33/Hrlrvo/cache_Data/train`: existing GT cache, read-only.
* `/home/work/ai-graphics_33/Hrlrvo/Data/4K/train`: RGB source for Anchor crops only.
* `/home/work/ai-graphics_33/Hrlrvo/cache_Data_Anchor_2K/train`: new/reused Anchor entries.

GT metadata's relative filename matches the relocated raw source. Its producer's
fingerprint is checked exactly: first MiB, last MiB when file size exceeds one
MiB, then ASCII decimal file size, all SHA-256 hashed. Size and digest must match;
mtime alone is not identity after relocation. Selected video timestamps must map
to exact rational CFR frame indices. PyAV seeks the preceding keyframe, skips to
the requested start, crops each decoded RGB frame, and stops after the requested
interval. It does not decode/store the whole video or encode any full 4K frame.

`temporal_sampling: random_aligned` (default) retains the seeded `(record,epoch)`
random latent start. `first` explicitly selects start zero. Both use the same
official `map_temporal_slice`, producing matching GT slice and RGB intervals.

Default pixel geometry, in `[y0,x0,y1,x1]` order:

| Coordinate space | Rectangle / size |
|---|---|
| Exact source RGB Anchor | `[540,960,1620,2880]`, H1080×W1920 |
| Content on padded GT canvas | `[548,960,1628,2880]` |
| Independent VAE input | H1088×W1920; reflected top/bottom 4, left/right 0 |
| Encoded support on GT canvas | `[544,960,1632,2880]` |
| Predicted Anchor latent | `[128,9,34,60]` |
| Canonical latent storage rectangle | `[17,30,51,90]` |
| Placement / known mask / attention Anchor geometry | `[17,30,51,90]` |

The input size is derived from checkpoint spatial factors and pixel origin,
**not** by central insertion based on latent shape. Official `align_resolution`
round-up and `resize_and_reflect_pad` supply the padding operation; two
no-resize passes place reflection on the required leading/trailing sides. This
canonical-side adaptation is explicit: upstream's default helper pads trailing
sides only, which alone would introduce a 4-pixel origin mismatch. There is no
resize or crop of the requested Anchor. Inference derives one latent rectangle
from the aligned origin plus the actual encoded Anchor tensor H/W, and reuses
it for placement, known/noise masking and attention geometry. Training uses the
same validated cache extent for conditioning, zero timesteps and loss exclusion.
The last latent row `y=50` is known; flooring the RGB content end to `y1=50`
would incorrectly exclude it.

Cache entries use the existing official latent fields plus provenance/checksum:

```text
latents/<relative parent>/<source>__start_<RGB start>__f65__anchor_1080x1920__<identity hash>.pt
```

The hash covers source/GT identity and source fingerprint, RGB/latent interval,
frame counts/FPS, source/requested/encoded geometry, crop/padding offsets,
canonical placement and known-mask policy, VAE identifier/checksum, GT contract,
preprocessing version, dtype and tiling parameters. Latent bytes have a separate
checksum; malformed/nonfinite/wrong-shape entries are rejected, never replaced.

Miss path: check → per-entry `flock` → recheck → exact RGB crop → independent
official BF16 `VideoEncoder.tiled_encode` → exclusive temporary file → flush/
fsync → atomic rename → directory fsync → unlock. Persistent empty lock files
avoid inode-replacement races. Only the new Anchor cache is written. Encoder
parameters are frozen, `eval`/`inference_mode` is used, RNG is preserved, and the
VAE alone is CPU-offloaded afterward by default. The transformer and optimizer
are never moved by caching. Workers use spawn, read cache hits on CPU, and defer
misses to the rank before native training. Batching supports mixed hits/misses.

The actual 2.3 encoder uses causal temporal convolutions and pixel-wise channel
normalization, so the first three cached latent frames cannot depend on RGB
after the selected prefix's frame 16. This is independent of the spatial VAE
leakage problem. Nonzero cached slabs retain preceding causal history; they are
not claimed numerically equal to a newly encoded independent RGB crop. The
sample's RGB/latent interval and prefix endpoints still use the official grid.

The loaded encoder configuration is `CausalVideoAutoencoder` with PixelNorm,
residual spatial/temporal downsamplers and no encoder attention. Causal
convolutions pad only with past/initial values; each temporal downsampler pairs
the prepended first frame then consecutive past/current inputs. Three temporal
x2 stages give latest dependency RGB `8*j` for latent j. Thus a cached slab's
first three prefix latents cannot depend on RGB after `8*start+16`. This follows
the selected path through `model_configurator.py`, `video_vae.py`, `resnet.py`,
`sampling.py` and channel-only `normalization.py`, not merely a class name.
Nonzero random cached crops can retain up to **209 preceding RGB frames** of
structurally possible causal encoder history, unlike the reset independent
Anchor encode. Each temporal downsampler contributes three current-stride
history steps: causal kernel-3 plus the retained two-phase channel pair. This is a
documented history/distribution mismatch, **not future leakage**; neither the
random sampling policy nor the safe independent Anchor pipeline is changed.

Inference uses native `ValidationRunner._run_denoising`, `LTX2Scheduler`, Euler,
CFG/STG, `VideoConditionByMask` and VAE loaders. First chunk is 65f; subsequent
chunks use previous clean RGB output's last 17f plus 48 new Anchor frames,
stride 48. Prefix/Anchor are re-encoded from observed pixels only. Anchor H/W
is read from input, centered and never silently resized; oversized Anchor/FPS
mismatch is rejected. Explicit final-chunk repeat-last temporal padding is
removed from returned output, or can be rejected by config. Pixel/VAE padding
is removed after decode; final observed Anchor/prefix pixels are composited
without freezing transformer hidden states.

## Training policy and parameter counts

Native loaders, block/residual/AdaLN, QK norms, head gate, CA, optimizer, LR
scheduler, AMP, activation checkpointing, accumulation, Accelerate DDP/FSDP
and logging remain upstream components. Hrlrvo specializes strategy inputs,
adapter installation, the numerically stable shifted-logit-normal calculation,
safe cache access, validation geometry and adapter-state serialization. The
native FFN call remains in place; only delta LoRA is attached to its two Linear
targets. No EMA is added: this official trainer has none.

The baseline is official
[`configs/video_outpainting_lora.yaml`](https://github.com/Lightricks/LTX-2/blob/598ab41247a77dbfe29b5186e915bcf4f9040ec7/packages/ltx-trainer/configs/video_outpainting_lora.yaml):
2,000 steps, AdamW 1e-4, linear factor 1→0.1 without a separate warmup, clipping
1.0, BF16, seed 42, checkpoints every 250. Sampling uses 30 steps, video CFG 3,
STG 1 on block 28 and rescale 0.7. Official validation interval is 100; this
config disables it until isolated validation Anchor media are supplied.

The current Hrlrvo production config overrides the upstream baseline schedule:
**30,000 optimizer steps**, the user's fixed budget covering30,000 of30,024 compatible
cached FIRST clips at micro-batch1 × accumulation1 × world-size1 (24 not reached).
Matching Anchor record IDs:30,024;
missing:0 (2026-09-11 inventory). Each video supplies RGB[0:65)/latent[0:9), not
all temporal windows of the full-length cache. AdamW base LR remains1e-4 with
LinearLR1→0.1 over30,000 steps. Checkpoints are every500 optimizer steps, ending
at checkpoint-030000. Old2,000-step run snapshots remain unchanged;
resuming them requires their original config because horizon mismatches fail closed.
See `../docs/FINAL_TRAIN_INFERENCE_CLI.md` for the current commands.

`per_device_batch_size`, `gradient_accumulation_steps`, `num_workers` are exposed.
Neither attention geometry nor fusion assumes B=1. The video-file CLI processes
one input video; that is not a model batch-size restriction.

Main v8 training is deliberately prompt-free: `conditions_dir: null` and
`constant_prompt: ""` are validated together so an external caption cache cannot
be selected accidentally. The cached constant condition contains
`video_prompt_embeds`, `audio_prompt_embeds` and `prompt_attention_mask`, exactly
the inputs expected by the native LTX-2.3 embeddings processor. The training
strategy still has `audio is None`; text CA remains a frozen video-path operation
and is not wrapped in `torch.no_grad`, so hidden-state gradients pass through it.
A CPU autograd check confirmed this input-gradient path while connector
parameters remained gradient-free; the text encoder and persistent/temporary
embeddings processors are kept in eval mode with `requires_grad=False`.
A CPU wiring check with batch size 3 produced video features
`[3,1024,4096]`, audio features `[3,1024,2048]` and mask `[3,1024]`.

The configured timestep mode remains `shifted_logit_normal`. Hrlrvo keeps the
official shift, normal sample, percentile-stretch, reflection, clamp and uniform
mixture semantics, but evaluates the sigmoid and percentile arithmetic in
float64 before casting the sampled sigma to the requested runtime dtype. This
avoids the float32 `1.0 - 1.0` percentile denominator at the 4K sequence length
`9 * 68 * 120 = 73,440`. A finite assertion runs before model forward;
the pinned LTX source is not modified and the sampler is not replaced by uniform.
A 32,768-sample CPU check at that sequence length returned zero NaN/Inf,
variance `0.03771618` and 22,042 distinct float32 values. At native-safe lengths
1024/2048/4096, the stable/native maximum error was at most
`1.7881393432617188e-7` with identical RNG state progression.

Actual 2.3 metadata construction: **48 blocks, D=4096**, neither hardcoded.

| V6-v8 trainables across all blocks (unchanged) | Parameters |
|---|---:|
| Shared Outpainting Q/K/V/O LoRA, r32 | 50,331,648 |
| Adaptive QKV LoRA | 75,497,472 |
| Global QKV LoRA | 75,497,472 |
| Shared FFN LoRA, r32 | 62,914,560 |
| C_(S/2) = C2 | 983,040 |
| C_S = C4 | 3,342,336 |
| R_(S/2) = R2 | 5,505,024 |
| R_S = R4 | 5,505,024 |
| Local native-gate channel fusion | 393,216 |
| Global native-gate channel fusion | 393,216 |
| **Existing C/R/fusion subtotal** | **16,121,856** |
| **Total trainable** | **280,363,008** |
| Frozen native transformer | 18,988,838,144 |
| **Installed total** | **19,269,201,152** |

Fusion parameters decrease from **3,323,068,416** (two literal UltraGen MLPs per
block) to **786,432** (`4 * D * blocks`); they remain channel-wise, not scalar.
Fusion and C/R counts are unchanged. Base Q/K/V/O, FFN, text CA, AdaLN and all
other pretrained parameters remain frozen. V8 changes no parameter identity or
count, but its Adaptive execution contract differs from v7. All pre-v8 metadata
is rejected by the strict loader; adapters are not converted, partially loaded
or silently resumed.

### Shared Outpainting plus branch specialization LoRA (v6)

The pinned LTX trainer uses `LoraConfig(..., init_lora_weights=True)` and PEFT
([`trainer.py:449–482`](https://github.com/Lightricks/LTX-2/blob/598ab41247a77dbfe29b5186e915bcf4f9040ec7/packages/ltx-trainer/src/ltx_trainer/trainer.py#L449-L482)).
It declares `peft>=0.14.0`; this environment's **PEFT 0.20.0** was inspected at
`peft/tuners/lora/layer.py:249–279,328–341,1030–1071`. The local
`LinearLoRA(in_features, out_features, rank, alpha)` follows those standard
Linear conventions without wrapping or duplicating the shared base projection.
It supports square attention and non-square FFN projections:

```text
Native QKV   = W_QKV + Delta_OUT,QKV
Adaptive QKV = W_QKV + Delta_OUT,QKV + Delta_A,QKV
Global QKV   = W_QKV + Delta_OUT,QKV + Delta_G,QKV
Final O      = W_O + Delta_OUT,O                 # called once after fusion
Shared FFN   = frozen W_FFN + Delta_FFN          # one native FFN path

Delta(x) = (alpha/rank) * B(A(x))
A: bias=False, Kaiming-uniform(a=sqrt(5))
B: bias=False, zeros
dropout=0
```

Shared Outpainting Q/K/V has rank/alpha 32/32 and one Q/K/V adapter set per
block, owned by `attn1.shared_outpainting_lora` and reused by Native, Adaptive
and Global. Adaptive and Global own independent rank/alpha 64/64 Q/K/V adapters.
The final `attn1.to_out.0` owns one shared rank/alpha 32/32 Outpainting O
adapter. The existing `ff.net.0.proj` and `ff.net.2` each own one shared
rank/alpha 32/32 `ffn_lora`. `project_with_loras(base, x, adapters)` computes
the frozen base projection once and adds each adapter delta from the **same
input**; adapters are never chained sequentially.

There is no Native-specific, per-window, text-CA or branch-specific O/FFN
adapter. The official top-level `LoraConfig` describes the shared r32 Q/K/V/O
and FFN task adapter for trainer integration, but does not install PEFT wrappers.
`ArchitectureConfig` is the single source of truth for custom v8 construction,
preventing duplicate adapters. The optimizer's `requires_grad` selection jointly
includes only these LoRA sets plus existing C/R/fusion trainables. Base
parameters stay frozen BF16; adapters/new trainables use FP32 masters. As in
PEFT, delta input is cast to A's dtype and the final sum is cast back to the
base result dtype. Scaling is applied exactly once.

Adapter keys keep ownership explicit, including
`transformer_blocks.0.attn1.shared_outpainting_lora.q.lora_A.weight`,
`adaptive_lora`, `global_lora`, the single output Linear's
`shared_outpainting_lora`, and each FFN target's `ffn_lora`. Adapter-only
safetensors checkpointing and optimizer/scheduler/RNG resume remain in place
with strict v8 architecture/Adaptive/initialization metadata.

The active architecture contract is explicit rather than overloading one rank:

```yaml
architecture_version: 8
adaptive_attention_type: target_touching_c2_query_only
position_type: pma_v2_nominal_grid
adaptive_support_policy: c4_first_c2_promotion
adaptive_role_priority: anchor_target_bridge
adaptive_bbox_alignment: canvas_center_minimal_expand
adaptive_restore_policy: per_query_full_cell_target_write
adaptive_fusion_policy: target_masked_native_passthrough
local_fusion_init_native_weight: 0.5
global_fusion_init_local_weight: 0.5
decompressor_refine_initialization: depthwise_identity
lora_type: shared_outpainting_plus_branch_specialization
lora_conditioning: sentinel_free_native_hrlrvo
lora_initialization: peft_kaiming_a_zero_b
lora_targets: shared_qkvo_branch_qkv_ffn
lora_bias: false
lora_dropout: 0.0
shared_outpainting_lora_enabled: true
shared_outpainting_lora_rank: 32
shared_outpainting_lora_alpha: 32
native_lora_enabled: false
adaptive_lora_enabled: true
adaptive_lora_rank: 64
adaptive_lora_alpha: 64
global_lora_enabled: true
global_lora_rank: 64
global_lora_alpha: 64
ffn_lora_enabled: true
ffn_lora_rank: 32
ffn_lora_alpha: 32
official_outpainting_ic_lora_enabled: false
rgb_sentinel_conditioning: false
```

V8 metadata-derived construction of both train/eval 48-block LTX models matched
the table exactly: 280,363,008 trainable and 18,988,838,144 frozen parameters.
Their 1,728 adapter keys had zero name/shape/dtype/trainability mismatch. A
nonzero small-v8 adapter round trip was exact, while v7 metadata was rejected.

A 41-geometry sweep covered **1,681 Local windows**. C4-parent coverage,
full-cell reads, Target-only writes and write uniqueness all passed; emitted
partial Adaptive cells were exactly **0**. This is a geometry/tokenization
validation; the production 22B GPU path was not run for this change.

An actual two-block native LTX CPU BF16 smoke used
`B=2,T=3,H=16,W=20,D=32`. Train forward/backward, evaluation forward and native
gradient-checkpoint recomputation passed with prediction error **0.0**. All
expected new-module gradients were finite/nonzero and frozen pretrained
parameters received zero gradients. This validates the small execution graph,
not a production 22B GPU run.

Neutral-initialization CPU checks found exactly 0.5 gate weights for all tested
finite native gates and nonzero, finite beta/gamma gradients. Maximum FP32
arithmetic errors were `1.1920928955078125e-7` (Local) and
`2.384185791015625e-7` (Global and hierarchical 25/25/50 on tested
resolved-Target support). Both refinement
convolutions matched their inputs exactly, including padding boundaries;
complete decompression matched the unchanged bilinear upsampling with error
**0.0**, including odd target sizes. The entire pre/post meta-model parameter
name/shape/dtype/trainability list is unchanged, not merely its total count.

The historical `src/diagnostics/check_fullft_vram.py` targets an older parameter
map and is not a v8 benchmark. It fails closed on v8 rather than reporting stale
counts; no replacement GPU benchmark is run for this Adaptive change.

### Historical official IC-LoRA audit — unsupported for Hrlrvo

The earlier audit inspected the existing artifact; it is not a v8 dependency:
`models/LTX2.3/models/LTX-2.3-22b-IC-LoRA-In-Outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors`.
SHA-256: `73dd0841c0d4f0eb26fb1f017781b841b2752021944ac5ecefe57917f6dae6b5`,
matching official HF LFS metadata. It contains **960 BF16 tensors / 480 A/B pairs
/ 654,311,424 parameters**, rank 128 throughout. Metadata declares model_version
2.3 and reference_downscale_factor 1, with no alpha/scaling tensor. In every
one of the 48 blocks it targets:

* video `attn1.to_q/to_k/to_v/to_out.0`;
* video `attn2.to_q/to_k/to_v/to_out.0`;
* `ff.net.0.proj` and `ff.net.2`.

There are no other targets. Every A/B product shape matches the actual dev base
checkpoint weight; only native `LTXV_LORA_COMFY_RENAMING_MAP` removes the official
prefix. The native loader applies **explicit strength × B@A**, not a filename
coefficient or an inferred alpha/rank. Official workflow node 5011 uses strength
1.0 despite the filename suffix 0.9. HF declares the LTX-2.3 family; the referenced
workflow uses dev + an additional distilled LoRA. It does not establish exact
training-base provenance for a dev-only adapter stack.

**Conditioning is incompatible with the current input contract.** The official
workflow first paints the generate-mask RGB region green (`#66FF00`) through
[`vanish_nodes.py:92–146`](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d/vanish_nodes.py#L92-L146),
then independently encodes that full reference and appends it as guide tokens.
Its green pixels carry mask information; no extra input channel is concatenated.
The native Python counterpart
[`reference_video_cond.py:96–104`](https://github.com/Lightricks/LTX-2/blob/598ab41247a77dbfe29b5186e915bcf4f9040ec7/packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py#L96-L104)
appends reference tokens/positions and their clean/noise state. Our independent
Anchor is instead inserted at existing canonical known locations; unknown outer
tokens remain noisy. `N=T*H*W` admits no second reference sequence. Omitting that
reference or treating the existing Anchor as equivalent would silently change
the official task protocol, so **no official IC-LoRA weights are loaded or merged**.

`outpainting_adapter.py` retains the historical inspection utility and an
explicit `supported_for_hrlrvo=False` guard for the **official IC-LoRA**. An IC
checkpoint path, IC reference-token protocol or RGB-sentinel request fails with
a conditioning-protocol error before model loading. That guard does not reject
v8's scratch-initialized `shared_outpainting_lora`, which is an ordinary Hrlrvo
delta module and consumes no official IC weights. The active YAMLs and
checkpoint provenance do not require the IC filename/hash/source. Frozen-base/
source integrity checks remain independent. No IC weights are loaded, partially
selected, merged or used in staged training. No upscaler or postprocessing model
was added.

Loaded official IC-LoRA parameters = **0**. The v8 Hrlrvo model instead has
**280,363,008** scratch-initialized trainables, **18,988,838,144** frozen native
parameters and **19,269,201,152** total installed parameters.

### Sentinel-free inference/validation conditioning

Only observed Anchor RGB is reflect-aligned to the canonical VAE lattice and
independently encoded. Its latent is placed into a zero **latent** buffer at the
same coordinates used by the unchanged lazy training cache. No unknown RGB
canvas is constructed or VAE-encoded. Optional previous clean full-canvas prefix
RGB is independently reflect-padded/encoded and replaces just its first three
latent positions for 17 frames. Binary masks, clean t=0/no-noise/loss exclusion
and ordinary unknown-region noising remain unchanged.

For the current 4K policy, Anchor rect is `[548,960,1628,2880]` in the padded
canvas, input is `1088x1920` with 4 reflected rows above/below the true 1080-row
Anchor, and latent placement is `[17,30,51,90]`. Reflect padding contains only
observed content, not a constant sentinel or outer GT. Small CPU mocked-encoder
checks verify input support, placement and prefix overwrite without executing
the VAE. Training cache format/keys/provenance and data source handling are unchanged.

### Fusion state and activation audit

The actual training path, not only the API, builds `[B,N]` timesteps:
known Anchor and prefix=0; unknown outer=sigma. Native timestep embedding
flattens every token, then restores `[B,N,9D]`; native block modulation produces
`vgate_msa[B,N,D]`. Its token differences are not collapsed. Known Anchor and
prefix share the same clean-state gate; noisy outer uses its own sigma-conditioned
gate (no numerical guarantee that every channel differs). Initial beta=gamma=0
makes alpha=0.5 everywhere in both fusions, even though g is token-specific.

At B=1, T=9,H=68,W=120,D=4096, one canonical tensor has **300,810,240 elements**:
BF16 **573.75 MiB**, FP32 **1,147.5 MiB**, scaling linearly with B. Native gate
reshape/expand and Local slices are views; gamma/beta broadcast as `[D]`, with
no repeats, clones or per-window full-canvas gate copies. Native AdaLN itself
has larger backing tensors; these figures describe one D-wide tensor, not VRAM.

Fusion now uses eager `addcmul(beta, g, gamma)` to avoid an explicit full FP32
copy of g and a separate multiplication temporary while preserving FP32 affine/
sigmoid and master-parameter precision. Sigmoid reuses only that fresh, private
non-leaf affine buffer, never g or a branch/canonical state. Its backward uses
the sigmoid result; addcmul's backward uses its original inputs. This removes
the second full FP32 sigmoid allocation. The remaining FP32 alpha is
**1,147.5 MiB**; casting to feature dtype adds **573.75 MiB** BF16. Feature fusion
uses `lerp(second, first, alpha)`, equivalent to the weighted sum, without
explicit full-sized complement and product tensors. LocalFull, GlobalFull and
the fused result remain canonical-sized. Local alpha has only its window's
support (35.859375 MiB BF16 at K4;
23.625 MiB at K5). Python releases previous-window references promptly, but
autograd may retain required saved tensors. No GPU memory/profile claim is made.

The eager-only guard is specifically due to unverified `ContextVar` state
transport around instance-bound native methods, outside compiled tensor inputs.
K/K+1 is fixed per block and is not by itself the reason. Python geometry/set/
packing control flow creates additional compile questions, not proven failures.
Removing the guard alone would not establish graph correctness. Eager-only is
retained as a future performance item, not a correctness blocker.

Checkpoints contain `adapter_model.safetensors`, training/base/architecture
metadata, optimizer, scheduler, optional AMP scaler, global step and rank-local
Accelerator RNG states. FSDP uses official optimizer/state helpers because
Accelerate's normal FSDP save writes full base weights before save hooks. Frozen
base weights are not replicated on disk. The adapter keyset is exactly shared
Outpainting Q/K/V/O, Adaptive Q/K/V, Global Q/K/V, shared FFN, C/R and both
fusion modules. Load validates exact adapter completeness, architecture/base
identity and distributed runtime compatibility. Missing/unexpected keys fail;
pre-v8 adapters cannot be partially loaded into v8. Weights-only initialization requires
explicit `checkpoints.no_resume: true`. Supported serialization is the
adapter-only safetensors/state-dict path followed by native runtime
reinstallation. Full `torch.save(model)` pickle is unsupported because the
native block carries runtime-bound instrumentation; this is not the checkpoint
format used by training.

Version 8 retains the v6 LoRA layout, v7 initialization,
`fusion_type: ltx_native_vgate` and independent K/S. The initialization contract
stores `local_fusion_init_native_weight: 0.5`,
`global_fusion_init_local_weight: 0.5` and
`decompressor_refine_initialization: depthwise_identity` in the exact
architecture metadata. The old generic fusion field has no fallback/alias.
Its Adaptive identity is
`adaptive_attention_type: target_touching_c2_query_only`,
`position_type: pma_v2_nominal_grid`,
`adaptive_support_policy: c4_first_c2_promotion`,
`adaptive_role_priority: anchor_target_bridge`,
`adaptive_bbox_alignment: canvas_center_minimal_expand`, and
`adaptive_restore_policy: per_query_full_cell_target_write`, with
`adaptive_fusion_policy: target_masked_native_passthrough`. The retired
centroid/component/partial-support fields are not accepted aliases.

Its adaptation identity is
`lora_type: shared_outpainting_plus_branch_specialization` with targets
`shared_qkvo_branch_qkv_ffn`, PEFT-compatible Kaiming-A/zero-B initialization,
bias disabled and dropout zero. Shared Outpainting and FFN adapters are enabled
at rank/alpha 32/32; Native-specific LoRA is disabled; Adaptive and Global QKV
adapters are enabled at 64/64. Official IC-LoRA and RGB sentinel conditioning
are disabled. A separate strict provenance record stores only base ID/filename/
SHA-256 and imported pinned LTX source revision, also in the resume manifest.
Launch verifies the base hash and clean pinned package checkout. Missing or
different identity is rejected before adapter tensors are read; v7/v6/v5/v4/v3/v2,
unversioned and giant-MLP adapters are rejected rather than converted.

Adapter checkpoint precision defaults to FP32 (instead of the upstream BF16
export default) to retain new trainable master weights alongside optimizer
state. Epoch ordering and crop selection use independent seeded generators,
with `(record,epoch)` crop keys. Resume reconstructs the consumed global offset
before native Accelerate batch sharding; worker prefetch does not choose crops.
Epoch tails are dropped at complete global accumulation boundaries. The manifest
also checks seed, batch, accumulation, dataset size and ordered-file fingerprint.
These resume paths are implemented but have not been exercised in a training run.

## Commands

Use the existing **Hrlrvo** environment and installed local official packages;
no new dependencies were installed. From this directory:

```bash
conda activate Hrlrvo
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Full training-state launch; Anchor cache misses are generated lazily.
python -m ltx_outpaint.train configs/train_4k.yaml

# Same native trainer with an appropriate Accelerate DDP/FSDP config.
accelerate launch --config_file ../models/LTX2.3/packages/ltx-trainer/configs/accelerate/fsdp.yaml \
  -m ltx_outpaint.train configs/train_4k.yaml

# Requires a trained adapter at model.adapter_path in this configuration.
# Positive and negative text are both fixed-empty; nonempty --prompt is rejected.
python -m ltx_outpaint.inference --config configs/inference_4k.yaml \
  --anchor-video /path/to/observed_anchor_24fps.mp4 \
  --output outputs/outpaint.mp4
```

Prompt-free does not bypass text conditioning: the native empty-text encoder,
both video/audio preconnector feature streams, and frozen Text Cross-Attention
remain active. Audio generation remains disabled. CFG/STG settings are unchanged;
both positive and negative text inputs to CFG are `""`, without the upstream
descriptive negative prompt. Training-validation rejects nonempty text as well.

The upstream FSDP example requests four processes; set the launch process count
to the actual GPU allocation. This session exposes one H200. No training or
inference launch above was executed as part of implementation.
