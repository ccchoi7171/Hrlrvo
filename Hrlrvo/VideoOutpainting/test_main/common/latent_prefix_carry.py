"""Isolated final-denoised three-latent exterior carry; native RGB history stays unchanged."""

from __future__ import annotations

import hashlib
import inspect
import json

import torch

MODES = ("disabled", "exterior_last3")
NATIVE_SHAPE = (1, 128, 9, 36, 60)
NATIVE_RECT = (9, 15, 27, 45)


def latent_sha(tensor):
    value = tensor.detach().contiguous().cpu()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode()
    return hashlib.sha256(
        header + value.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def rect_tuple(rect):
    return (rect.y0, rect.x0, rect.y1, rect.x1)


class LatentPrefixCarry:
    """Minimal argument-preserving native sample wrapper, with active-only conditioning seam."""

    def __init__(
        self,
        native_sample,
        native_condition,
        mode,
        *,
        shape=NATIVE_SHAPE,
        rect=NATIVE_RECT,
        folder=None,
    ):
        if mode not in MODES:
            raise ValueError("Unknown latent-carry mode")
        if len(shape) != 5 or shape[0] != 1 or shape[2] != 9:
            raise ValueError(
                "Native65 topology must contain exactly nine temporal latent rows"
            )
        if not (
            0 <= rect[0] < rect[2] <= shape[3] and 0 <= rect[1] < rect[3] <= shape[4]
        ):
            raise ValueError("Latent anchor rectangle outside canonical grid")
        self.native_sample = native_sample
        self.native_condition = native_condition
        self.condition_signature = inspect.signature(native_condition)
        self.mode = mode
        self.shape = shape
        self.rect = rect
        self.folder = folder
        self.calls = 0
        self.pending = None
        self._in_chunk = False
        self._bank = None
        self._bank_from_start = None
        self._final_state = None
        self._condition_calls = 0
        self._denoise_calls = 0
        self._finalize_calls = 0

    @property
    def bank(self):
        """Inspection returns a copy; callers cannot mutate the saved recurrence bank."""
        return None if self._bank is None else self._bank.clone()

    def condition(self, *args, **kwargs):
        if self.mode != "exterior_last3" or not self._in_chunk or self.pending is None:
            raise RuntimeError("Active condition hook called outside its native chunk")
        self._condition_calls += 1
        if self._condition_calls != 1:
            raise RuntimeError("Exactly one native condition result per chunk required")
        bound = self.condition_signature.bind(*args, **kwargs).arguments
        prefix = bound["previous_prefix"]
        sf = bound["scale_factors"]
        if sf.time != 8 or tuple(bound["latent_shape"].to_torch_shape()) != self.shape:
            raise RuntimeError("Native factor8/65-to9 latent geometry changed")
        # Finish native RGB prefix VAE encoding and ORIGINAL-source anchor reassertion first.
        condition, rect = self.native_condition(*args, **kwargs)
        if (
            tuple(condition.shape) != self.shape
            or condition.dtype != torch.bfloat16
            or rect_tuple(rect) != self.rect
        ):
            raise RuntimeError(
                "Native padded latent layout/dtype/encoded anchor rectangle changed"
            )
        if not torch.isfinite(condition).all():
            raise FloatingPointError("Native condition contains nonfinite values")
        self.pending["native_condition_shape"] = list(condition.shape)
        self.pending["native_latent_anchor_rect_yxyx"] = list(self.rect)
        self.pending["stock_RGB_prefix_encode_and_fresh_anchor_reassert_completed"] = (
            True
        )
        if self.pending["absolute_start_frame"] == 0:
            if prefix is not None or self._bank is not None:
                raise RuntimeError(
                    "Initial chunk must remain native without previous prefix/carry"
                )
            self.pending["carry_applied"] = False
            self.pending["fresh_current_anchor_preserved"] = True
            return condition, rect
        if prefix is None or len(prefix) != 17 or self._bank is None:
            raise RuntimeError(
                "Recurrent chunk requires actual RGB17 and completed prior latent bank"
            )
        start = self.pending["absolute_start_frame"]
        if self._bank_from_start != start - 48 or self._bank.shape != (
            self.shape[0],
            self.shape[1],
            3,
            *self.shape[3:],
        ):
            raise RuntimeError(
                "Carry must originate from immediately preceding native65 chunk"
            )
        if self._bank.device.type != "cpu" or self._bank.dtype != torch.bfloat16:
            raise RuntimeError("Saved bank must be exact CPU BF16")
        y0, x0, y1, x1 = self.rect
        exterior = torch.ones(self.shape[3:], device=condition.device, dtype=torch.bool)
        exterior[y0:y1, x0:x1] = False
        source_bank_sha = latent_sha(self._bank)
        fresh_anchor = condition[:, :, :, y0:y1, x0:x1].clone()
        carried = self._bank.to(device=condition.device, dtype=condition.dtype)
        difference = (
            condition[:, :, :3, exterior].float() - carried[:, :, :, exterior].float()
        )
        rmse = float(difference.square().mean().sqrt().item())
        result = condition.clone()
        result[:, :, :3, exterior] = carried[:, :, :, exterior]
        # Reassert the already-native fresh source encoding over ALL9 anchor rows.
        result[:, :, :, y0:y1, x0:x1] = fresh_anchor
        if not torch.equal(result[:, :, :, y0:y1, x0:x1], fresh_anchor):
            raise RuntimeError("Current observed anchor was modified by latent carry")
        if not torch.equal(result[:, :, 3:], condition[:, :, 3:]):
            raise RuntimeError("Carry reached future/generated temporal rows3:9")
        if not torch.equal(result[:, :, :3, exterior], carried[:, :, :, exterior]):
            raise RuntimeError("Exterior carry is not bitwise exact")
        if latent_sha(self._bank) != source_bank_sha:
            raise RuntimeError("Saved prior bank was mutated")
        self.pending.update(
            carry_applied=True,
            carry_source_chunk_start=self._bank_from_start,
            source_latent_rows=[6, 9],
            target_latent_rows=[0, 3],
            carried_bank_sha256=source_bank_sha,
            fresh_current_anchor_preserved=True,
            future_latent_rows3to9_unchanged=True,
            carried_vs_RGB_reencoded_prefix_exterior_RMSE=rmse,
            RMSE_is_diagnostic_not_quality_score=True,
        )
        return result, rect

    def __call__(self, **kwargs):
        if self.pending is not None or self._in_chunk:
            raise RuntimeError(
                "Prior chunk lineage must be consumed before next native call"
            )
        start = kwargs["absolute_start_frame"]
        prefix = kwargs["previous_prefix"]
        anchor = kwargs["anchor_frames"]
        if type(start) is not int or start != 48 * self.calls:
            raise RuntimeError("Native source time/48-stride changed")
        if (
            len(anchor) != 65
            or anchor.dtype != torch.float32
            or anchor.device.type != "cpu"
        ):
            raise RuntimeError("Native source65 RGB FP32 contract changed")
        if (start == 0 and prefix is not None) or (
            start > 0
            and (
                prefix is None
                or len(prefix) != 17
                or prefix.dtype != torch.float32
                or prefix.device.type != "cpu"
            )
        ):
            raise RuntimeError("Native RGB recent17 recurrence must remain intact")
        for key in (
            "spatial_boundary_geometry",
            "static_reference_runtime",
            "preview_step",
            "preview_frame_indices",
        ):
            if kwargs.get(key) is not None:
                raise RuntimeError(
                    "Reference/hybrid/preview cannot combine with latent carry"
                )
        if kwargs.get("boundary_blend_width_pixels", 0) != 0:
            raise RuntimeError("Native exact observed RGB center requires blend0")
        self.pending = {
            "mode": self.mode,
            "chunk_index": self.calls,
            "absolute_start_frame": start,
            "actual_RGB_prefix_frames": 0 if prefix is None else 17,
            "RGB_prefix_replacement": False,
            "source_RGB_replacement": False,
            "carry_applied": False,
            "known_mask_function_modified": False,
            "Gaussian_PMA_or_sampling_schedule_modified": False,
        }
        self._condition_calls = self._denoise_calls = self._finalize_calls = 0
        self._final_state = None
        self._in_chunk = True
        runner = kwargs["runner"]
        native_denoise = runner._run_denoising
        native_finalize = runner._finalize_modality

        def denoise(*args, **kw):
            self._denoise_calls += 1
            if self._denoise_calls != 1 or self._condition_calls != 1:
                raise RuntimeError(
                    "Capture requires one complete native conditioning/denoising path"
                )
            result = native_denoise(*args, **kw)
            if not isinstance(result, tuple) or len(result) != 2 or result[0] is None:
                raise RuntimeError("Native final video-state return contract changed")
            self._final_state = result[0]
            return result

        def finalize(*args, **kw):
            self._finalize_calls += 1
            state = args[0] if args else kw["state"]
            tools = args[1] if len(args) > 1 else kw["tools"]
            if (
                self._finalize_calls != 1
                or self._final_state is None
                or state is not self._final_state
            ):
                raise RuntimeError(
                    "Capture only the completed video denoise state, never STG/intermediate state"
                )
            if (
                tuple(tools.target_shape.to_torch_shape()) != self.shape
                or tools.scale_factors.time != 8
            ):
                raise RuntimeError(
                    "Capture tools do not match actual native65 latent topology"
                )
            # This is the same condition-clear/unpatchify route as native final decode.
            clean = tools.unpatchify(tools.clear_conditioning(state)).latent
            if (
                tuple(clean.shape) != self.shape
                or clean.dtype != torch.bfloat16
                or not torch.isfinite(clean).all()
            ):
                raise RuntimeError(
                    "Final video latent must be finite native BF16[1,128,9,36,60]"
                )
            saved = clean[:, :, 6:9].detach().contiguous().cpu().clone()
            self._bank = saved
            self._bank_from_start = start
            self.pending.update(
                capture_source="exact_returned_final_video_state_before_native_decode",
                captured_latent_rows=[6, 9],
                captured_shape=list(saved.shape),
                captured_dtype=str(saved.dtype),
                captured_bank_sha256=latent_sha(saved),
                captures_after_complete_denoise=1,
                bank_stored_CPU_only=True,
            )
            if self.folder is not None:
                path = self.folder / f"latent_tail_chunk_{self.calls:03d}_bf16.pt"
                if path.exists():
                    raise FileExistsError(path)
                torch.save(saved, path)
                self.pending["captured_bank_file"] = str(path)
            return native_finalize(*args, **kw)

        if self.mode == "exterior_last3":
            runner._run_denoising = denoise
            runner._finalize_modality = finalize
        try:
            # No argument is replaced, including current source and displayed RGB prefix.
            generated = self.native_sample(**kwargs)
            if (
                generated.ndim != 4
                or generated.shape[:2] != (65, 3)
                or generated.dtype != torch.float32
                or generated.device.type != "cpu"
            ):
                raise RuntimeError("Native returned RGB must remain CPU FP32 full65")
            if self.mode == "exterior_last3" and (
                self._condition_calls,
                self._denoise_calls,
                self._finalize_calls,
            ) != (1, 1, 1):
                raise RuntimeError("Missing/duplicate active native latent-carry seams")
            if prefix is not None and not torch.equal(generated[:17], prefix):
                raise RuntimeError(
                    "Native displayed RGB first17 must remain actual previous RGB17"
                )
            boundary = kwargs["boundary"]
            rect = kwargs["anchor_rect"]
            y0, x0 = rect[0] - boundary.top, rect[1] - boundary.left
            if not torch.equal(
                generated[:, :, y0 : y0 + anchor.shape[-2], x0 : x0 + anchor.shape[-1]],
                anchor,
            ):
                raise RuntimeError(
                    "Native current-source observed RGB output was changed"
                )
            self.pending.update(
                completed_native_sample_verified=True,
                native_sample_arguments_not_replaced=True,
                generated_prefix_RGB_exact=True,
                current_source_RGB_exact=True,
                active_condition_calls=self._condition_calls,
                active_finalize_calls=self._finalize_calls,
            )
            self.calls += 1
            return generated
        finally:
            if self.mode == "exterior_last3":
                runner._run_denoising = native_denoise
                runner._finalize_modality = native_finalize
            self._final_state = None
            self._in_chunk = False

    def install(self, inference):
        inference.sample_chunk = self
        if self.mode == "exterior_last3":
            inference._condition_latents = self.condition

    def consume(self):
        if (
            self.pending is None
            or self._in_chunk
            or self.pending.get("completed_native_sample_verified") is not True
        ):
            raise RuntimeError("No completed latent-carry lineage available")
        row = self.pending
        self.pending = None
        if self.folder is not None:
            with (self.folder / "latent_carry_lineage.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return row
