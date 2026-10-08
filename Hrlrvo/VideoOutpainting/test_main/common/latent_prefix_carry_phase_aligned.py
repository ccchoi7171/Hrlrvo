"""Phase-aligned exterior latent carry preserving native singleton row zero."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from latent_prefix_carry import (
    NATIVE_RECT,
    NATIVE_SHAPE,
    LatentPrefixCarry,
    latent_sha,
    rect_tuple,
)

DISABLED = "disabled"
ACTIVE = "exterior_last2_keep_native_row0"
MODES = (DISABLED, ACTIVE)


class PhaseAlignedLatentPrefixCarry(LatentPrefixCarry):
    """Reuse the audited capture wrapper, retaining old rows7:9 for next rows1:3."""

    def __init__(
        self,
        native_sample,
        native_condition,
        mode,
        *,
        shape=NATIVE_SHAPE,
        rect=NATIVE_RECT,
        folder: Path | None = None,
    ):
        if mode not in MODES:
            raise ValueError("Unknown phase-aligned latent-carry mode")
        internal_mode = "disabled" if mode == DISABLED else "exterior_last3"
        super().__init__(
            native_sample,
            native_condition,
            internal_mode,
            shape=shape,
            rect=rect,
            folder=None,
        )
        self.phase_mode = mode
        self.phase_folder = folder

    def condition(self, *args, **kwargs):
        if self.phase_mode != ACTIVE or not self._in_chunk or self.pending is None:
            raise RuntimeError("Active phase condition called outside native chunk")
        self._condition_calls += 1
        if self._condition_calls != 1:
            raise RuntimeError("Exactly one native condition result per chunk required")
        bound = self.condition_signature.bind(*args, **kwargs).arguments
        prefix = bound["previous_prefix"]
        factors = bound["scale_factors"]
        if (
            factors.time != 8
            or tuple(bound["latent_shape"].to_torch_shape()) != self.shape
        ):
            raise RuntimeError("Native factor8/65-to9 latent geometry changed")

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
        self.pending.update(
            mode=self.phase_mode,
            native_condition_shape=list(condition.shape),
            native_latent_anchor_rect_yxyx=list(self.rect),
            stock_RGB_prefix_encode_and_fresh_anchor_reassert_completed=True,
        )
        if self.pending["absolute_start_frame"] == 0:
            if prefix is not None or self._bank is not None:
                raise RuntimeError(
                    "Initial chunk must remain native without previous prefix/carry"
                )
            self.pending.update(
                carry_applied=False,
                fresh_current_anchor_preserved=True,
                native_singleton_row0_preserved=True,
            )
            return condition, rect
        if prefix is None or len(prefix) != 17 or self._bank is None:
            raise RuntimeError(
                "Recurrent chunk requires actual RGB17 and completed prior latent bank"
            )
        start = self.pending["absolute_start_frame"]
        expected = (self.shape[0], self.shape[1], 2, *self.shape[3:])
        if self._bank_from_start != start - 48 or tuple(self._bank.shape) != expected:
            raise RuntimeError(
                "Phase carry must be prior native65 final rows7:9 from immediate chunk"
            )
        if self._bank.device.type != "cpu" or self._bank.dtype != torch.bfloat16:
            raise RuntimeError("Saved phase bank must be exact CPU BF16")

        y0, x0, y1, x1 = self.rect
        exterior = torch.ones(self.shape[3:], device=condition.device, dtype=torch.bool)
        exterior[y0:y1, x0:x1] = False
        source_bank_sha = latent_sha(self._bank)
        fresh_row0 = condition[:, :, 0].clone()
        fresh_anchor = condition[:, :, :, y0:y1, x0:x1].clone()
        carried = self._bank.to(device=condition.device, dtype=condition.dtype)
        difference = (
            condition[:, :, 1:3, exterior].float()
            - carried[:, :, :, exterior].float()
        )
        rmse = float(difference.square().mean().sqrt().item())
        result = condition.clone()
        result[:, :, 1:3, exterior] = carried[:, :, :, exterior]
        result[:, :, :, y0:y1, x0:x1] = fresh_anchor

        if not torch.equal(result[:, :, 0], fresh_row0):
            raise RuntimeError("Native singleton target row0 was modified")
        if not torch.equal(result[:, :, :, y0:y1, x0:x1], fresh_anchor):
            raise RuntimeError("Current observed anchor was modified by phase carry")
        if not torch.equal(result[:, :, 3:], condition[:, :, 3:]):
            raise RuntimeError("Phase carry reached future/generated rows3:9")
        if not torch.equal(result[:, :, 1:3, exterior], carried[:, :, :, exterior]):
            raise RuntimeError("Phase-aligned exterior carry is not bitwise exact")
        if latent_sha(self._bank) != source_bank_sha:
            raise RuntimeError("Saved prior phase bank was mutated")
        self.pending.update(
            carry_applied=True,
            carry_source_chunk_start=self._bank_from_start,
            source_latent_rows=[7, 9],
            target_latent_rows=[1, 3],
            carried_bank_sha256=source_bank_sha,
            native_singleton_row0_preserved=True,
            fresh_current_anchor_preserved=True,
            future_latent_rows3to9_unchanged=True,
            carried_vs_RGB_reencoded_prefix_exterior_RMSE=rmse,
            RMSE_is_diagnostic_not_quality_score=True,
            temporal_phase_contract={
                "native_row0_RGB_support": [0, 1],
                "carried_old_RGB_support": [[49, 57], [57, 65]],
                "target_RGB_support": [[1, 9], [9, 17]],
            },
        )
        return result, rect

    def __call__(self, **kwargs):
        generated = super().__call__(**kwargs)
        if self.phase_mode == DISABLED:
            self.pending["mode"] = self.phase_mode
            return generated
        if self._bank is None or self.pending is None:
            raise RuntimeError("Audited final-state capture did not produce a bank")
        full_tail = self._bank
        if tuple(full_tail.shape) != (
            self.shape[0],
            self.shape[1],
            3,
            *self.shape[3:],
        ):
            raise RuntimeError("Audited base capture did not return old rows6:9")
        retained = full_tail[:, :, 1:3].detach().contiguous().cpu().clone()
        self._bank = retained
        self.pending.update(
            mode=self.phase_mode,
            capture_source="exact_returned_final_video_state_before_native_decode",
            captured_latent_rows=[7, 9],
            captured_shape=list(retained.shape),
            captured_dtype=str(retained.dtype),
            captured_bank_sha256=latent_sha(retained),
            base_capture_rows_transient_not_carried=[6, 9],
            bank_stored_CPU_only=True,
        )
        if self.phase_folder is not None:
            path = self.phase_folder / f"phase_tail_chunk_{self.calls - 1:03d}_bf16.pt"
            if path.exists():
                raise FileExistsError(path)
            torch.save(retained, path)
            self.pending["captured_bank_file"] = str(path)
        return generated

    def install(self, inference):
        inference.sample_chunk = self
        if self.phase_mode == ACTIVE:
            inference._condition_latents = self.condition

    def consume(self):
        row = super().consume()
        if self.phase_folder is not None:
            path = self.phase_folder / "phase_aligned_latent_carry_lineage.jsonl"
            with path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return row
