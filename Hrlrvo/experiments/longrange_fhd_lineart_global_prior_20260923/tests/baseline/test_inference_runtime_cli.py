import inspect
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from ltx_outpaint import inference

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (72, [([0, 65], None, [0, 65]), ([48, 72], [48, 65], [65, 72])]),
        (94, [([0, 65], None, [0, 65]), ([48, 94], [48, 65], [65, 94])]),
        (
            142,
            [
                ([0, 65], None, [0, 65]),
                ([48, 113], [48, 65], [65, 113]),
                ([96, 142], [96, 113], [113, 142]),
            ],
        ),
    ],
)
def test_recurrent_chunk_lineage_has_exact_17f_prefix_without_gap_or_duplicate(
    frames, expected
):
    spec = inference.LongRangeSpec()
    records = inference._plan_long_range_chunks(frames, spec, temporal_factor=8)
    assert [
        (
            record["anchor_source_frame_range"],
            record["prefix_output_frame_range"],
            record["new_output_frame_range"],
        )
        for record in records
    ] == expected
    assert inference._lineage_health(records, frames) == (0, 0)
    assert all(
        record["prefix_output_frame_range"] is None
        or record["prefix_output_frame_range"][1]
        - record["prefix_output_frame_range"][0]
        == 17
        for record in records
    )


def test_final_partial_chunk_records_repeat_last_without_appending_padding():
    records = inference._plan_long_range_chunks(
        72, inference.LongRangeSpec(), temporal_factor=8
    )
    assert records[-1]["real_input_frame_count"] == 24
    assert records[-1]["working_frame_count"] == 25
    assert records[-1]["repeat_last_padding_count"] == 1
    assert records[-1]["output_appended_count"] == 7
    assert records[-1]["final_partial_repeat_last"] is True


@pytest.mark.parametrize(
    ("requested_frames", "working_frames"),
    [(1, 1), (9, 9), (10, 17), (17, 17), (18, 25),
     (49, 49), (57, 57), (58, 65), (65, 65)],
)
def test_short_input_uses_smallest_vae_aligned_working_length(
    requested_frames: int, working_frames: int
) -> None:
    records = inference._plan_long_range_chunks(
        requested_frames, inference.LongRangeSpec(), temporal_factor=8
    )
    assert len(records) == 1
    record = records[0]
    assert record["working_frame_count"] == working_frames
    assert record["repeat_last_padding_count"] == working_frames - requested_frames
    assert (working_frames - 1) % 8 == 0
    assert inference._lineage_health(records, requested_frames) == (0, 0)


@pytest.mark.parametrize(
    ("requested_frames", "last_working_frames"),
    [(66, 25), (72, 25), (73, 25), (80, 33), (89, 41),
     (94, 49), (113, 65), (160, 65)],
)
def test_recurrent_tail_keeps_prefix_and_uses_aligned_working_length(
    requested_frames: int, last_working_frames: int
) -> None:
    spec = inference.LongRangeSpec()
    records = inference._plan_long_range_chunks(
        requested_frames, spec, temporal_factor=8
    )
    assert records[-1]["working_frame_count"] == last_working_frames
    assert all(
        record["working_frame_count"] >= record["real_input_frame_count"]
        and (record["working_frame_count"] - 1) % 8 == 0
        and (record["chunk_index"] == 0 or record["working_frame_count"] >= spec.prefix_frames)
        for record in records
    )
    assert inference._lineage_health(records, requested_frames) == (0, 0)


def test_reject_policy_fails_on_partial_chunk():
    with pytest.raises(ValueError, match="Final chunk"):
        inference._plan_long_range_chunks(
            72, inference.LongRangeSpec(final_chunk_policy="reject"), temporal_factor=8
        )


def test_checkpoint_path_accepts_file_or_exact_checkpoint_directory(tmp_path):
    checkpoint = tmp_path / "checkpoint-000005"
    checkpoint.mkdir()
    adapter = checkpoint / "adapter_model.safetensors"
    adapter.write_bytes(b"test")
    assert inference._resolve_adapter_path(checkpoint) == adapter.resolve()
    assert inference._resolve_adapter_path(adapter) == adapter.resolve()


def test_checkpoint_path_never_falls_back_when_adapter_is_missing(tmp_path):
    checkpoint = tmp_path / "checkpoint-000005"
    checkpoint.mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        inference._resolve_adapter_path(checkpoint)


def test_cli_passes_explicit_checkpoint_seed_and_steps(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint-000005"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").touch()
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference",
            "--config",
            "config.yaml",
            "--anchor-video",
            "anchor.mp4",
            "--output",
            "generated.mp4",
            "--checkpoint",
            str(checkpoint),
            "--seed",
            "123",
            "--steps",
            "30",
        ],
    )
    monkeypatch.setattr(inference, "_load_settings", lambda *_: {"loaded": True})
    calls = []
    monkeypatch.setattr(inference, "sample_long_range", lambda *args, **kwargs: calls.append((args, kwargs)))
    inference.main()
    assert calls == [
        (
            (
                {"loaded": True},
                Path("anchor.mp4"),
                Path("generated.mp4"),
                "",
                None,
                checkpoint,
                False,
                123,
                30,
                False,
            ),
            {
                "negative_prompt": "",
                "segment_dir": None,
                "resume": False,
                "keep_intermediates": False,
            },
        )
    ]


def test_cli_passes_streaming_resume_options(monkeypatch, tmp_path):
    segment_dir = tmp_path / "segments"
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference",
            "--config",
            "config.yaml",
            "--anchor-video",
            "anchor.mp4",
            "--output",
            "generated.mp4",
            "--initialized-adapters",
            "--segment-dir",
            str(segment_dir),
            "--resume",
            "--keep-intermediates",
        ],
    )
    monkeypatch.setattr(inference, "_load_settings", lambda *_: {})
    calls = []
    monkeypatch.setattr(
        inference,
        "sample_long_range",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    inference.main()

    assert calls[0][1] == {
        "negative_prompt": "",
        "segment_dir": segment_dir,
        "resume": True,
        "keep_intermediates": True,
    }


def test_cli_passes_existing_first_chunk_and_manifest_directory(
    monkeypatch, tmp_path
):
    bootstrap = tmp_path / "01_12345678_65f_30step.mp4"
    manifest_dir = tmp_path / "segments"
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference",
            "--config",
            "config.yaml",
            "--anchor-video",
            "anchor.mp4",
            "--output",
            "generated.mp4",
            "--initialized-adapters",
            "--bootstrap-first-chunk",
            str(bootstrap),
            "--manifest-dir",
            str(manifest_dir),
        ],
    )
    monkeypatch.setattr(inference, "_load_settings", lambda *_: {})
    calls = []
    monkeypatch.setattr(
        inference,
        "sample_long_range",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    inference.main()

    assert calls[0][1]["bootstrap_first_chunk_path"] == bootstrap
    assert calls[0][1]["manifest_dir"] == manifest_dir


def test_cli_prefix_search_does_not_require_normal_output(monkeypatch, tmp_path):
    from ltx_outpaint import prefix_modes

    candidate_dir = tmp_path / "candidates"
    monkeypatch.setattr(
        "sys.argv",
        ["inference", "--config", "config.yaml", "--anchor-video", "source.mp4",
         "--mode", "prefix_seed_search", "--prefix-output-dir", str(candidate_dir),
         "--prefix-seeds", "31", "32", "--video-only-gpu-weights",
         "--vae-decode-streaming"],
    )
    monkeypatch.setattr(inference, "_load_settings", lambda *_: {})
    calls = []
    monkeypatch.setattr(
        prefix_modes, "sample_prefix_seed_search",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    inference.main()
    assert len(calls) == 1
    assert calls[0][0][:4] == (
        {"memory_optimization": {
            "video_only_gpu_weights": True, "vae_decode_streaming": True,
        }}, Path("source.mp4"), candidate_dir, Path("config.yaml")
    )
    assert calls[0][1]["prefix_seeds"] == [31, 32]


def test_cli_step_preview_keeps_30_step_schedule_and_has_no_mp4(monkeypatch, tmp_path):
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference",
            "--config",
            "config.yaml",
            "--anchor-video",
            "source.mp4",
            "--mode",
            "normal_long_range",
            "--num-frames",
            "65",
            "--initialized-adapters",
            "--seed",
            "33333333",
            "--steps",
            "30",
            "--negative-prompt",
            "people, person, human, staff, audience, crowd, backstage, seats",
            "--seed-search-output-dir",
            str(output_dir),
            "--preview-step",
            "10",
            "--preview-frames",
            "0",
            "32",
            "64",
            "--video-only-gpu-weights",
            "--vae-decode-streaming",
        ],
    )
    monkeypatch.setattr(inference, "_load_settings", lambda *_: {})
    calls = []
    monkeypatch.setattr(
        inference,
        "sample_seed_preview",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    inference.main()

    assert len(calls) == 1
    assert calls[0][0] == (
        {
            "memory_optimization": {
                "video_only_gpu_weights": True,
                "vae_decode_streaming": True,
            }
        },
        Path("source.mp4"),
        output_dir,
    )
    assert calls[0][1]["num_frames"] == 65
    assert calls[0][1]["inference_steps"] == 30
    assert calls[0][1]["preview_step"] == 10
    assert calls[0][1]["preview_frames"] == (0, 32, 64)


def test_cli_prefix_continuation_passes_only_new_optional_argument(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["inference", "--config", "config.yaml", "--anchor-video", "source.mp4",
         "--output", "generated.mp4", "--mode", "long_range_from_prefix",
         "--initial-prefix", "seed_31_prefix.pt"],
    )
    monkeypatch.setattr(inference, "_load_settings", lambda *_: {})
    calls = []
    monkeypatch.setattr(
        inference, "sample_long_range", lambda *args, **kwargs: calls.append(kwargs),
    )
    inference.main()
    assert calls[0]["initial_prefix_path"] == Path("seed_31_prefix.pt")


def test_memory_optimizations_default_to_legacy_path() -> None:
    options = inference._memory_optimization_settings({})
    assert options == {
        "video_only_gpu_weights": False,
        "vae_decode_streaming": False,
        "ffn_token_chunking": False,
        "ffn_chunk_tokens": 8192,
    }


@pytest.mark.parametrize("name", [
    "video_only_gpu_weights", "vae_decode_streaming", "ffn_token_chunking",
])
def test_memory_optimization_flags_are_independent(name: str) -> None:
    options = inference._memory_optimization_settings({
        "memory_optimization": {name: True},
    })
    assert options[name] is True
    assert sum(options[key] for key in (
        "video_only_gpu_weights", "vae_decode_streaming", "ffn_token_chunking"
    )) == 1


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_ffn_chunk_tokens_must_be_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="ffn_chunk_tokens"):
        inference._memory_optimization_settings({
            "memory_optimization": {"ffn_chunk_tokens": value},
        })


def test_cli_memory_flags_override_config_independently(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", [
        "inference", "--config", "config.yaml", "--anchor-video", "anchor.mp4",
        "--output", "generated.mp4", "--video-only-gpu-weights",
        "--no-vae-decode-streaming", "--ffn-token-chunking",
        "--ffn-chunk-tokens", "4096",
    ])
    monkeypatch.setattr(inference, "_load_settings", lambda *_: {
        "memory_optimization": {"vae_decode_streaming": True},
    })
    calls = []
    monkeypatch.setattr(inference, "sample_long_range", lambda *args, **kwargs: calls.append((args, kwargs)))

    inference.main()

    assert calls[0][0][0]["memory_optimization"] == {
        "video_only_gpu_weights": True,
        "vae_decode_streaming": False,
        "ffn_token_chunking": True,
        "ffn_chunk_tokens": 4096,
    }


def test_cli_initialized_adapter_mode_is_explicit_and_exclusive(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference",
            "--config",
            "config.yaml",
            "--anchor-video",
            "anchor.mp4",
            "--output",
            "generated.mp4",
            "--initialized-adapters",
        ],
    )
    monkeypatch.setattr(inference, "_load_settings", lambda *_: {})
    calls = []
    monkeypatch.setattr(inference, "sample_long_range", lambda *args, **kwargs: calls.append((args, kwargs)))
    inference.main()
    assert calls[0][0][6] is True
    assert calls[0][1]["negative_prompt"] == ""


def test_cli_early_global_boost_is_explicit_and_defaults_off(monkeypatch):
    common = [
        "inference",
        "--config",
        "config.yaml",
        "--anchor-video",
        "anchor.mp4",
        "--output",
        "generated.mp4",
    ]
    monkeypatch.setattr(inference, "_load_settings", lambda *_: {})
    calls = []
    monkeypatch.setattr(inference, "sample_long_range", lambda *args, **kwargs: calls.append((args, kwargs)))

    monkeypatch.setattr("sys.argv", common)
    inference.main()
    monkeypatch.setattr("sys.argv", [*common, "--early-global-boost"])
    inference.main()

    assert calls[0][0][-1] is False
    assert calls[1][0][-1] is True
    assert calls[0][1]["negative_prompt"] == calls[1][1]["negative_prompt"] == ""


def test_early_global_boost_metadata_is_reproducible() -> None:
    assert inference._early_global_boost_metadata(False) == {"enabled": False}
    assert inference._early_global_boost_metadata(True) == {
        "enabled": True,
        "start_scale": 1.4,
        "duration_steps": 10,
        "curve": "smoothstep",
    }


def test_early_global_boost_schedule_is_exact_and_monotonic() -> None:
    scales = [inference._early_global_boost_scale(index) for index in range(10)]

    assert scales[0] == 1.4
    assert scales[9] == 1.0
    assert inference._early_global_boost_scale(10) is None
    assert inference._early_global_boost_scale(29) is None
    assert all(left >= right for left, right in pairwise(scales))
    assert all(scale >= 1.0 for scale in scales)


def test_scheduler_sigma_identifies_outer_step_without_forward_counting() -> None:
    inference._install_official_sources(ROOT.parent / "models/LTX2.3")
    from ltx_core.components.schedulers import LTX2Scheduler

    runtime = inference._EarlyGlobalBoostRuntime.from_inference_steps(30)
    sigmas = LTX2Scheduler().execute(steps=30)

    assert runtime.active_scale_for_sigma(sigmas[0:1]) == 1.4
    repeated = [runtime.active_scale_for_sigma(sigmas[1:2]) for _ in range(4)]
    assert repeated == [repeated[0]] * 4
    assert runtime.active_scale_for_sigma(sigmas[8:9]) > 1.0
    # Scheduled step 10 reaches 1.0, but gate override is deliberately absent.
    assert runtime.active_scale_for_sigma(sigmas[9:10]) is None
    assert runtime.active_scale_for_sigma(sigmas[10:11]) is None
    assert runtime.active_scale_for_sigma(sigmas[29:30]) is None


def test_early_global_boost_defaults_are_inference_only_and_training_untouched() -> None:
    assert inspect.signature(inference.sample_long_range).parameters[
        "early_global_boost"
    ].default is False
    assert inspect.signature(inference._load_models).parameters[
        "early_global_boost"
    ].default is False
    assert inspect.signature(inference.GeometryValidationRunner.create).parameters[
        "early_global_boost"
    ].default is False
    assert "early_global_boost" not in (
        ROOT / "src/ltx_outpaint/train.py"
    ).read_text(encoding="utf-8")
    assert "early_global_boost" not in (
        ROOT / "configs/train_4k.yaml"
    ).read_text(encoding="utf-8")


def test_inward_feather_width_zero_is_exact_legacy_hard_composite() -> None:
    decoded = torch.linspace(0.0, 1.0, 2 * 3 * 8 * 10).reshape(2, 3, 8, 10)
    anchor = torch.linspace(1.0, 0.0, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    bbox = (2, 3, 6, 8)
    expected = decoded.clone()
    expected[:, :, 2:6, 3:8] = anchor

    actual = inference._composite_anchor_rgb(
        decoded.clone(),
        anchor,
        anchor_bbox=bbox,
        boundary_blend_width_pixels=0,
    )

    assert torch.equal(actual, expected)
    assert inspect.signature(inference.sample_chunk).parameters[
        "boundary_blend_width_pixels"
    ].default == 0


def test_inward_feather_is_single_frame_invariant_alpha_and_preserves_outside_and_interior() -> None:
    decoded = torch.zeros(2, 3, 10, 12)
    anchor = torch.ones(2, 3, 6, 6)
    bbox = (2, 3, 8, 9)

    alpha = inference._inward_anchor_boundary_alpha(
        anchor_bbox=bbox,
        canvas_height=decoded.shape[-2],
        canvas_width=decoded.shape[-1],
        boundary_blend_width_pixels=2,
        device=decoded.device,
        dtype=decoded.dtype,
    )
    actual = inference._composite_anchor_rgb(
        decoded.clone(),
        anchor,
        anchor_bbox=bbox,
        boundary_blend_width_pixels=2,
    )

    assert alpha.shape == (6, 6)
    assert alpha.dtype == decoded.dtype and alpha.device == decoded.device
    assert torch.equal(actual[:, :, :2], decoded[:, :, :2])
    assert torch.equal(actual[:, :, 8:], decoded[:, :, 8:])
    assert torch.equal(actual[:, :, 2:8, :3], decoded[:, :, 2:8, :3])
    assert torch.equal(actual[:, :, 2:8, 9:], decoded[:, :, 2:8, 9:])
    assert torch.equal(actual[:, :, 4:6, 5:7], anchor[:, :, 2:4, 2:4])
    assert torch.equal(actual[0], actual[1])
    # Distance one from both top and left uses one nearest-edge alpha (0.5),
    # rather than applying two edge blends at the corner intersection.
    assert torch.equal(actual[:, :, 3, 4], torch.full((2, 3), 0.5))
    assert float(actual.min()) >= 0.0 and float(actual.max()) <= 1.0


def test_inward_feather_ignores_canvas_edges_without_generated_margin() -> None:
    decoded = torch.zeros(1, 1, 8, 8)
    anchor = torch.ones(1, 1, 6, 6)
    bbox = (0, 0, 6, 6)

    actual = inference._composite_anchor_rgb(
        decoded,
        anchor,
        anchor_bbox=bbox,
        boundary_blend_width_pixels=2,
    )

    # Top/left touch the canvas, so they are not Anchor-Generated boundaries.
    assert actual[0, 0, 0, 0] == 1.0
    assert actual[0, 0, 5, 5] == 0.0
    assert actual[0, 0, 4, 4] == 0.5
    assert torch.equal(actual[:, :, 6:, :], torch.zeros(1, 1, 2, 8))
    assert torch.equal(actual[:, :, :6, 6:], torch.zeros(1, 1, 6, 2))


@pytest.mark.parametrize("width", [8, 16, 32])
def test_inference_boundary_blend_width_is_config_driven(width: int) -> None:
    assert inference._inference_boundary_blend_width(
        {"video": {"boundary_blend_width_pixels": width}}
    ) == width


def test_inference_boundary_blend_width_defaults_to_sixteen_and_rejects_invalid() -> None:
    assert inference._inference_boundary_blend_width({"video": {}}) == 16
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="boundary_blend_width_pixels"):
            inference._inference_boundary_blend_width(
                {"video": {"boundary_blend_width_pixels": invalid}}
            )
