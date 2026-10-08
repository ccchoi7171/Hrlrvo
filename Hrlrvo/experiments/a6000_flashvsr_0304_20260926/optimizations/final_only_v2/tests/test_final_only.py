from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


OVERLAY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = OVERLAY_ROOT / "final_only.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("a6000_final_only_v2", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def final_only():
    return _load_module()


def _args(tmp_path: Path) -> argparse.Namespace:
    paths = {
        "input_4k": tmp_path / "03_Outpaint_4K.mp4",
        "original_8k": tmp_path / "03_Input_8K.mp4",
        "outpainting_config": tmp_path / "03.yaml",
        "flashvsr_reference": tmp_path / "upscale.py",
        "aggregate_manifest": tmp_path / "aggregate.json",
    }
    for path in paths.values():
        path.write_bytes(b"trusted")
    return argparse.Namespace(
        video_id="03",
        input_4k=paths["input_4k"],
        original_8k=paths["original_8k"],
        model_root=tmp_path / "model",
        flashvsr_reference=paths["flashvsr_reference"],
        outpainting_config=paths["outpainting_config"],
        output=tmp_path / "final.mp4",
        raw_output=tmp_path / "legacy_raw_target.mp4",
        final_manifest=tmp_path / "final_manifest.json",
        work_dir=tmp_path / "work",
        aggregate_manifest=paths["aggregate_manifest"],
        expected_frames=1952,
        tile_size=256,
        tile_overlap=32,
        min_tile_size=256,
        sparse_ratio=2.0,
        local_range=11,
        seed=71634910,
        gpu_index=0,
        ffmpeg_threads=32,
        encoder="libx265",
        compose_workers=1,
        renderer="original",
        dry_run=False,
        self_test=False,
    )


def _headers(width: int, height: int) -> dict[str, object]:
    return {
        "path": "input.mp4",
        "width": width,
        "height": height,
        "avg_frame_rate": "16/1",
        "r_frame_rate": "16/1",
        "fps": 16.0,
        "frame_count": 1952,
        "duration": 122.0,
        "decoded_frame_count_performed": False,
    }


def _prepare_happy_path(final_only, monkeypatch, args):
    fc = final_only.flash_composite
    aggregate = {
        "input_identity": {"path": str(args.input_4k), "size": 7, "sha256": "a" * 64},
        "anchor_identity": {"path": str(args.original_8k), "size": 7, "sha256": "b" * 64},
        "partial_report": {"configuration": {"precision": "torch.bfloat16 (original implementation)"}},
    }
    reference = SimpleNamespace(
        MODEL_ROOT=str(args.model_root),
        SCALE=4,
        read_stage_manifest=lambda _path: aggregate,
        execute_encode_stage=lambda *_a, **_kw: pytest.fail("raw encode must never run"),
    )
    media_utils = SimpleNamespace(
        select_encoder=lambda *_a, **_kw: (
            {"extension": ".mp4", "name": "libx265"}, [{"result": "PASS"}]
        ),
        probe_media=lambda *_a, **_kw: SimpleNamespace(audio_present=False),
    )
    monkeypatch.setattr(final_only.shutil, "which", lambda _name: "/usr/bin/fake")
    monkeypatch.setattr(fc, "required_assets", lambda *_a: [])
    monkeypatch.setattr(fc, "require_approved_reference", lambda _path: "c" * 64)
    monkeypatch.setattr(fc, "load_reference_module", lambda _path: reference)
    monkeypatch.setattr(
        fc,
        "probe_video_headers",
        lambda path: _headers(3840, 2160) if path == args.input_4k else _headers(7680, 4320),
    )
    monkeypatch.setattr(fc, "validate_video_spec", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        fc,
        "validate_trusted_source_identity",
        lambda path, _identity, **_kw: {
            "path": str(path), "size": path.stat().st_size, "sha256": "d" * 64
        },
    )
    monkeypatch.setattr(fc, "read_outpainting_feather_width", lambda _path: 16)
    monkeypatch.setattr(
        fc, "validate_aggregate_contract", lambda *_a, **_kw: ([SimpleNamespace(index=0)], {0: tmp_path_placeholder(args)}, args.work_dir)
    )
    monkeypatch.setattr(fc, "load_media_utils", lambda _path: media_utils)
    monkeypatch.setattr(
        fc,
        "bounded_output_evidence",
        lambda *_a, **_kw: {
            "policy": "bounded_no_full_decode", "decoded_full_frame_count": False
        },
    )
    monkeypatch.setattr(
        fc,
        "stage_request_payload",
        lambda *_a, **_kw: {"configuration": {"seed": args.seed}},
    )
    return aggregate


def tmp_path_placeholder(args):
    return args.work_dir / "tile_000"


def test_final_only_skips_raw_encode_and_records_truthful_manifest(
    final_only, monkeypatch, tmp_path
):
    args = _args(tmp_path)
    _prepare_happy_path(final_only, monkeypatch, args)
    args.raw_output.write_bytes(b"pre-existing legacy raw is untouched")
    calls = []

    def encoder(**kwargs):
        calls.append(kwargs)
        kwargs["partial_output"].write_bytes(b"encoded-final")
        return {"seconds": 1.25, "frames_written": 1952, "vertical_band_count": 19}

    manifest = final_only.execute_final_only(args, encode_function=encoder)

    assert len(calls) == 1
    assert calls[0]["feather_width"] == 64
    assert args.raw_output.read_bytes() == b"pre-existing legacy raw is untouched"
    assert not (args.work_dir / "raw_encode_report.json").exists()
    assert manifest["status"] == "PASS"
    assert manifest["encoding"]["raw_status"] == "SKIPPED"
    assert manifest["encoding"]["raw_16k_intermediate_video"] is False
    assert manifest["encoding"]["raw_skip_reason"] == final_only.RAW_SKIP_REASON
    assert manifest["verification_policy"]["raw_output"]["status"] == "SKIPPED"
    assert manifest["verification_policy"]["full_16k_decode_performed"] is False
    assert json.loads(args.final_manifest.read_text())["output_file"]["path"] == str(args.output)


def test_final_only_refuses_existing_final_or_manifest(final_only, tmp_path):
    args = _args(tmp_path)
    args.output.write_bytes(b"existing")
    with pytest.raises(final_only.flash_composite.PipelineError, match="fresh-only"):
        final_only.require_fresh_final_targets(args.output, args.final_manifest)

    args.output.unlink()
    args.final_manifest.write_text("{}")
    with pytest.raises(final_only.flash_composite.PipelineError, match="fresh-only"):
        final_only.require_fresh_final_targets(args.output, args.final_manifest)


def test_encoder_failure_leaves_no_final_or_manifest(final_only, monkeypatch, tmp_path):
    args = _args(tmp_path)
    _prepare_happy_path(final_only, monkeypatch, args)

    def fail_encoder(**kwargs):
        kwargs["partial_output"].write_bytes(b"partial")
        raise RuntimeError("encoder exploded")

    with pytest.raises(RuntimeError, match="encoder exploded"):
        final_only.execute_final_only(args, encode_function=fail_encoder)
    assert not args.output.exists()
    assert not args.final_manifest.exists()
    assert not list(tmp_path.glob(".*partial*.mp4"))


def test_destination_created_during_encode_is_never_overwritten(final_only, monkeypatch, tmp_path):
    args = _args(tmp_path)
    _prepare_happy_path(final_only, monkeypatch, args)

    def encoder(**kwargs):
        kwargs["partial_output"].write_bytes(b"new partial")
        args.output.write_bytes(b"existing result from another writer")
        return {"frames_written": 1952}

    with pytest.raises((FileExistsError, final_only.flash_composite.PipelineError)):
        final_only.execute_final_only(args, encode_function=encoder)
    assert args.output.read_bytes() == b"existing result from another writer"
    assert not args.final_manifest.exists()
    assert not list(tmp_path.glob(".*partial*.mp4"))


def test_dangling_destination_symlink_is_not_a_fresh_target(final_only, tmp_path):
    output = tmp_path / "output.mp4"
    output.symlink_to(tmp_path / "missing.mp4")
    with pytest.raises(final_only.flash_composite.PipelineError, match="fresh-only"):
        final_only.require_fresh_final_targets(output, tmp_path / "manifest.json")


@pytest.mark.parametrize("field", ["output", "final_manifest"])
def test_cli_resolution_cannot_hide_dangling_destination_symlink(final_only, tmp_path, field):
    args = _args(tmp_path)
    getattr(args, field).symlink_to(tmp_path / "unapproved_destination")
    with pytest.raises(final_only.flash_composite.PipelineError, match="symlink"):
        final_only.resolve_paths(args)
    assert not (tmp_path / "unapproved_destination").exists()


def test_manifest_collision_after_video_publish_preserves_foreign_manifest(
    final_only, monkeypatch, tmp_path
):
    args = _args(tmp_path)
    _prepare_happy_path(final_only, monkeypatch, args)

    def encoder(**kwargs):
        kwargs["partial_output"].write_bytes(b"our encoded video")
        return {"frames_written": 1952}

    def evidence(*_args, **_kwargs):
        args.final_manifest.write_bytes(b"foreign manifest")
        return {"decoded_full_frame_count": False}

    monkeypatch.setattr(final_only.flash_composite, "bounded_output_evidence", evidence)
    with pytest.raises(FileExistsError):
        final_only.execute_final_only(args, encode_function=encoder)
    assert args.final_manifest.read_bytes() == b"foreign manifest"
    assert not args.output.exists()
    orphans = list((args.work_dir / "orphaned_outputs").glob("*.mp4"))
    assert len(orphans) == 1 and orphans[0].read_bytes() == b"our encoded video"


def test_temporary_unlink_failure_does_not_invalidate_published_pass(
    final_only, monkeypatch, tmp_path
):
    args = _args(tmp_path)
    _prepare_happy_path(final_only, monkeypatch, args)
    original_unlink = Path.unlink

    def fail_temporary_unlink(path, *pos, **kw):
        if ".partial." in path.name:
            raise OSError("injected temporary unlink failure")
        return original_unlink(path, *pos, **kw)

    def encoder(**kwargs):
        kwargs["partial_output"].write_bytes(b"encoded video")
        return {"frames_written": 1952}

    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)
    result = final_only.execute_final_only(args, encode_function=encoder)
    assert result["status"] == "PASS"
    assert args.output.read_bytes() == b"encoded video"
    assert json.loads(args.final_manifest.read_text())["status"] == "PASS"
    assert not (args.work_dir / "orphaned_outputs").exists()


def test_audio_mux_failure_leaves_no_final_or_manifest(final_only, monkeypatch, tmp_path):
    args = _args(tmp_path)
    _prepare_happy_path(final_only, monkeypatch, args)
    media_utils = SimpleNamespace(
        select_encoder=lambda *_a, **_kw: (
            {"extension": ".mp4", "name": "libx265"}, []
        ),
        probe_media=lambda *_a, **_kw: SimpleNamespace(audio_present=True),
    )
    monkeypatch.setattr(final_only.flash_composite, "load_media_utils", lambda _path: media_utils)
    monkeypatch.setattr(
        final_only.flash_composite,
        "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=9, stderr="mux exploded"),
    )

    def encoder(**kwargs):
        kwargs["partial_output"].write_bytes(b"partial")
        return {"frames_written": 1952}

    with pytest.raises(final_only.flash_composite.PipelineError, match="mux exploded"):
        final_only.execute_final_only(args, encode_function=encoder)
    assert not args.output.exists()
    assert not args.final_manifest.exists()
    assert not list(tmp_path.glob(".*partial*.mp4"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("video_id", "05"),
        ("expected_frames", 1951),
        ("tile_size", 192),
        ("tile_overlap", 16),
        ("min_tile_size", 192),
        ("seed", 0),
        ("sparse_ratio", 1.5),
        ("local_range", 9),
        ("compose_workers", 0),
        ("compose_workers", 17),
        ("encoder", "hevc_nvenc"),
        ("ffmpeg_threads", 24),
    ],
)
def test_contract_rejects_altered_production_parameters(
    final_only, tmp_path, field, value
):
    args = _args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(final_only.flash_composite.PipelineError):
        final_only.validate_args_contract(args)


def test_original_renderer_defaults_to_frozen_encoder(final_only, monkeypatch, tmp_path):
    args = _args(tmp_path)
    sentinel = object()
    monkeypatch.setattr(final_only.flash_composite, "encode_staged_tiles", sentinel)
    assert final_only.select_encode_function(args) is sentinel


def test_threaded_renderer_imports_only_sibling_overlay(final_only, monkeypatch, tmp_path):
    args = _args(tmp_path)
    args.renderer = "threaded"
    called = []
    sentinel = object()
    monkeypatch.setattr(final_only, "load_sibling_parallel_renderer", lambda: called.append(True) or sentinel)
    assert final_only.select_encode_function(args) is sentinel
    assert called == [True]


def test_frozen_compositor_cache_path_poisoning_is_rejected(final_only, monkeypatch, tmp_path):
    poisoned = SimpleNamespace(__file__=str(tmp_path / "flash_composite.py"))
    monkeypatch.setitem(
        final_only.sys.modules, "_a6000_frozen_flash_composite_v1", poisoned
    )
    with pytest.raises(RuntimeError, match="unexpected file"):
        final_only._load_frozen_composite()
