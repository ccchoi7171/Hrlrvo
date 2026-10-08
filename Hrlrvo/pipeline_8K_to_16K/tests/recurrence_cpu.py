"""Run separately: CPU tensors only; no checkpoint/CUDA/model loading."""
# ruff: noqa: E402 -- install package-relative and vendored LTX import paths first.

import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/video_outpainting/main/src")]
from ltx_outpaint import inference as inf

inf._install_official_sources(ROOT / "src/video_outpainting/models/LTX2.3")
from ltx_core.types import VideoLatentShape
from ltx_outpaint.geometry import CanvasGeometry, Rect, global_cells
from ltx_outpaint.lineart_global_prior import (
    LineArtGlobalPrior,
    build_attention_bias,
    build_fine_write_mask,
    current_prior,
)
from ltx_outpaint.lineart_reference import direct_global_maps, reference_from_segment
from ltx_outpaint.recurrence_artifacts import load_carry, save_carry


class RecurrenceCPU(unittest.TestCase):
    def prior(self):
        return LineArtGlobalPrior(
            torch.ones(17, 30), tuple(float(30 - i) for i in range(30))
        )

    def test_direct_4k_global_footprints(self):
        lines = np.zeros((2160, 3840), np.uint8)
        # +8 encoded y padding: visible y119 is the final row of cell0;
        # y120 is the first row of cell1. x127/x128 cross the same boundary.
        lines[119, 127] = 255
        lines[120, 128] = 255
        density, strength = direct_global_maps(lines)
        self.assertEqual(density.shape, (17, 30))
        self.assertEqual(int(np.count_nonzero(density)), 2)
        self.assertEqual(float(density[0, 0]), 1 / 16384)
        self.assertEqual(float(density[1, 1]), 1 / 16384)
        self.assertTrue(np.array_equal(strength, np.clip(density / 0.05, 0, 1)))
        density, strength = direct_global_maps(np.full_like(lines, 255))
        self.assertTrue((density[0] == 120 / 128).all())
        self.assertTrue((density[-1] == 120 / 128).all())
        self.assertTrue((density[1:-1] == 1).all())
        self.assertTrue((strength == 1).all())

    def test_c0_scope_never_resolves_sigma_or_bias(self):
        p = self.prior()
        with (
            p.chunk_scope(0),
            patch.object(p, "_step_for_sigma", side_effect=AssertionError),
        ):
            self.assertIsNone(current_prior(object()))
        self.assertEqual(p.summary()["active_calls"], 0)

    def test_sigma_schedule_not_call_counter(self):
        p = self.prior()
        for step, strength in (
            (0, 0.5),
            (9, 0.5),
            (10, 0.3),
            (19, 0.3),
            (20, 0.1),
            (29, 0.1),
        ):
            context = SimpleNamespace(sigma=torch.tensor([30.0 - step]))
            with p.chunk_scope(1):
                for _ in range(3):
                    self.assertEqual(current_prior(context)[2:], (step, strength))

    def test_bias_and_fine_known_mask(self):
        geometry = CanvasGeometry(4, 68, 120, (Rect(17, 30, 51, 90),))
        context = SimpleNamespace(geometry=geometry, sigma=torch.tensor([30.0]))
        p = self.prior()
        with p.chunk_scope(1, 3):
            _, state, _, strength = current_prior(context)
            bias, eligible = build_attention_bias(
                p,
                state,
                context,
                global_cells(68, 120, 4),
                torch.float32,
                torch.device("cpu"),
                strength,
            )
            self.assertFalse(eligible[0, : 3 * 510].any())
            self.assertTrue(
                torch.equal(bias[0, : 3 * 510], torch.zeros_like(bias[0, : 3 * 510]))
            )
            self.assertEqual(float(bias.max()), 0.5)
            mask = build_fine_write_mask(state, context, torch.device("cpu"))
            self.assertFalse(mask[:, :3].any())
            self.assertFalse(mask[:, :, 17:51, 30:90].any())
            self.assertTrue(mask[:, 3, 0, 0].all())

    def test_global_original_route_and_known_restore(self):
        module = importlib.import_module("ltx_outpaint.global_attention")
        grid = torch.zeros(1, 4, 68, 120, 1)
        geometry = CanvasGeometry(4, 68, 120, (Rect(17, 30, 51, 90),))
        context = SimpleNamespace(geometry=geometry, sigma=torch.tensor([30.0]))

        def decompress(tokens, coarse, fine):
            # Spread every coarse-token difference spatially, like learned refinement.
            return (
                tokens.mean((2, 3), keepdim=True)
                .reshape(1, 4, 1, 1, 1)
                .expand(1, 4, 68, 120, 1)
            )

        attention = SimpleNamespace(
            scale=4,
            heads=1,
            compressors={"4": lambda g, c: torch.zeros(1, 4, 510, 1)},
            decompressors={"4": decompress},
            shared_outpainting_lora={k: None for k in "qkv"},
            global_lora={k: None for k in "qkv"},
            to_q=None,
            to_k=None,
            to_v=None,
            preattention_function=lambda q, k, *a: (q, k),
            attention_function=lambda q, k, v, h: torch.ones_like(q),
            masked_attention_function=lambda q, k, v, h, b: torch.full_like(q, 7),
        )
        with (
            patch.object(module, "phase_rope", return_value=None),
            patch.object(module, "attend", return_value=torch.ones(1, 2040, 1)),
            patch.object(
                module, "project_with_loras", side_effect=lambda proj, x, loras: x
            ),
        ):
            original = module.global_attention(attention, grid, context)
            with self.prior().chunk_scope(0):
                self.assertTrue(
                    torch.equal(
                        original, module.global_attention(attention, grid, context)
                    )
                )
            with self.prior().chunk_scope(1, 3):
                changed = module.global_attention(attention, grid, context)
            self.assertTrue(torch.equal(original[:, :3], changed[:, :3]))
            self.assertTrue(
                torch.equal(original[:, :, 17:51, 30:90], changed[:, :, 17:51, 30:90])
            )
            self.assertGreater(float(changed[0, 3, 0, 0, 0]), 1.0)

    def test_hybrid_single_frame_keyframe_and_anchor_authority(self):
        boundary = inf.PixelBoundary(128, 128, 128, 128, 0, 0)
        shape = VideoLatentShape(batch=1, channels=128, frames=9, height=4, width=4)
        sf = SimpleNamespace(time=8, height=32, width=32)
        anchor = torch.zeros(65, 3, 64, 64)
        prefix = torch.zeros(17, 3, 128, 128)
        carry = torch.full((1, 128, 3, 4, 4), 5.0)
        calls = []

        def encode(rgb, *args, **kwargs):
            calls.append((rgb.shape[2], kwargs))
            return torch.full(
                (
                    1,
                    128,
                    (rgb.shape[2] - 1) // 8 + 1,
                    rgb.shape[-2] // 32,
                    rgb.shape[-1] // 32,
                ),
                2.0 if kwargs else 7.0,
            )

        with patch.object(inf, "_encode_condition_rgb", side_effect=encode):
            condition, rect = inf._condition_latents(
                anchor,
                prefix,
                boundary,
                (32, 32, 96, 96),
                shape,
                sf,
                None,
                None,
                torch.device("cpu"),
                torch.float32,
                prefix_latent_override=carry,
            )
        self.assertEqual([x[0] for x in calls], [65, 1])
        self.assertTrue((condition[:, :, 0, 0, 0] == 7).all())
        self.assertTrue((condition[:, :, 1:3, 0, 0] == 5).all())
        self.assertTrue(
            (condition[:, :, :3, rect.y0 : rect.y1, rect.x0 : rect.x1] == 2).all()
        )
        self.assertTrue((carry == 5).all())

    def test_c0_anchor_condition_matches_original_function(self):
        original = (
            ROOT
            / "state/recurrence_lineart/before/src/video_outpainting/main/src/ltx_outpaint/inference.py"
        ).read_text()
        text = original[
            original.index("def _condition_latents(") : original.index(
                "\n\ndef _known_mask"
            )
        ]
        ns = dict(vars(inf))
        exec("from __future__ import annotations\n" + text, ns)
        args = (
            torch.zeros(65, 3, 64, 64),
            None,
            inf.PixelBoundary(128, 128, 128, 128, 0, 0),
            (32, 32, 96, 96),
            VideoLatentShape(1, 128, 9, 4, 4),
            SimpleNamespace(time=8, height=32, width=32),
            None,
            None,
            torch.device("cpu"),
            torch.float32,
        )

        def encode(*a, **k):
            return torch.ones(1, 128, 9, 2, 2)

        with patch.object(inf, "_encode_condition_rgb", side_effect=encode):
            ns["_encode_condition_rgb"] = encode
            a, ar = ns["_condition_latents"](*args)
            b, br = inf._condition_latents(*args)
        self.assertEqual(ar, br)
        self.assertTrue(torch.equal(a, b))

    def test_carry_atomic_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensor = torch.ones(1, 128, 3, 4, 4, dtype=torch.bfloat16)
            record = {
                "chunk_index": 0,
                "causal_latent_carry": save_carry(root, 0, tensor),
            }
            self.assertTrue(
                torch.equal(
                    load_carry(root, record, tensor.shape, tensor.dtype), tensor
                )
            )
            (root / record["causal_latent_carry"]["file"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "digest"):
                load_carry(root, record)

    def test_reference_roundtrip_source_identity_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = root / "segment.npy"
            frame = np.zeros((1, 2160, 3840, 3), np.uint8)
            frame[:, :, 1920:1924] = 255
            np.save(segment, frame, allow_pickle=False)
            strength, meta, digest = reference_from_segment(
                root / "lineart", segment, {"source": "c0"}
            )
            loaded, _, again = reference_from_segment(
                root / "lineart", segment, {"source": "c0"}
            )
            self.assertTrue(torch.equal(strength, loaded))
            self.assertEqual(digest, again)
            self.assertEqual(meta["reference_shape"], [2160, 3840, 3])
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                reference_from_segment(root / "lineart", segment, {"source": "wrong"})
            (root / "lineart/lineart_4k.png").write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "Corrupt"):
                reference_from_segment(root / "lineart", segment, {"source": "c0"})

    def test_all_operational_contracts(self):
        from cert_pipeline.recurrence import DEFAULT_RECURRENCE, validate_recurrence

        for family in ("certification", "quick"):
            for path in (ROOT / "configs" / family).glob("*.json"):
                self.assertEqual(
                    validate_recurrence(
                        json.loads(path.read_text())["outpainting"]["recurrence"]
                    ),
                    DEFAULT_RECURRENCE,
                )

    def test_reference_rejects_joint_artifact_and_sidecar_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = root / "segment.npy"
            np.save(segment, np.zeros((1, 2160, 3840, 3), np.uint8), allow_pickle=False)
            reference = root / "lineart"
            reference_from_segment(reference, segment, {"source": "c0"})
            corrupted = reference / "global_line_strength.npy"
            np.save(corrupted, np.ones((17, 30), np.float32), allow_pickle=False)
            metadata_path = reference / "lineart_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["files"][corrupted.name]["sha256"] = hashlib.sha256(
                corrupted.read_bytes()
            ).hexdigest()
            metadata["files"][corrupted.name]["size_bytes"] = corrupted.stat().st_size
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(RuntimeError, "density mismatch"):
                reference_from_segment(reference, segment, {"source": "c0"})

    def test_strength_contract_rejects_invalid_values(self):
        from cert_pipeline.recurrence import DEFAULT_RECURRENCE, validate_recurrence

        for value in (-0.1, 1.1, float("nan"), True, "1.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_recurrence(
                    dict(DEFAULT_RECURRENCE, generated_prefix_strength=value)
                )
        self.assertEqual(
            validate_recurrence(
                dict(DEFAULT_RECURRENCE, generated_prefix_strength=0.8)
            )["generated_prefix_strength"],
            0.8,
        )

    def test_carry_rejects_shape_and_dtype_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensor = torch.zeros(1, 128, 3, 4, 4, dtype=torch.bfloat16)
            record = {
                "chunk_index": 0,
                "causal_latent_carry": save_carry(root, 0, tensor),
            }
            with self.assertRaisesRegex(RuntimeError, "geometry mismatch"):
                load_carry(root, record, (1, 128, 3, 5, 4))
            with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
                load_carry(root, record, expected_dtype=torch.float32)


if __name__ == "__main__":
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    unittest.main()
