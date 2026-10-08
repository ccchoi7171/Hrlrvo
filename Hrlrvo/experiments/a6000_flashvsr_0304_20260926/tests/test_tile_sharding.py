from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


worker = load("test_tile_worker", "tools/tile_worker.py")
merge = load("test_merge_shards", "tools/merge_shards.py")


@dataclass(frozen=True)
class FakeTile:
    index: int
    x: int = 0
    y: int = 0
    width: int = 256
    height: int = 256


def fake_make_tiles(*_args):
    return [FakeTile(index=index, x=index * 10) for index in range(170)]


def test_global_shard_selection_preserves_indices_and_seed_derivation():
    selected = worker.select_global_tiles(fake_make_tiles, 85, 169)(3840, 2160, 256, 32)
    assert [tile.index for tile in selected] == list(range(85, 170))
    assert [worker.BASE_SEED + tile.index for tile in selected[:2]] == [71634995, 71634996]
    assert selected[0] == fake_make_tiles()[85]


def test_private_manifest_rejects_renumbered_or_wrong_seed():
    request = {"selection": {"tile_start": 85, "tile_end": 85}, "mode": "smoke"}
    manifest = {
        "state": "COMPLETE", "processed_input_frames": 29,
        "tile_artifacts": [{"tile": {"index": 85}, "tile_seed": worker.BASE_SEED + 85}],
    }
    worker._validate_private_manifest(request, manifest)
    manifest["tile_artifacts"][0]["tile_seed"] = worker.BASE_SEED
    with pytest.raises(RuntimeError, match="global-index seed"):
        worker._validate_private_manifest(request, manifest)


@pytest.mark.parametrize(
    "video_id,input_name,anchor_name",
    [("03", "04_Outpaint_4K.mp4", "03_Input_8K.mp4"),
     ("04", "04_Outpaint_4K.mp4", "03_Input_8K.mp4")],
)
def test_video_anchor_mapping_fails_closed(video_id, input_name, anchor_name):
    with pytest.raises(ValueError):
        worker.assert_video_paths(video_id, Path(input_name), Path(anchor_name))


def minimal_receipt(work: Path, *, state="SHARD_COMPLETE"):
    return {
        "schema": "a6000_flashvsr_tile_shard/v1", "state": state,
        "completion_scope": "PARTIAL_SPATIAL_TILES", "video_id": "04", "mode": "full",
        "subprocess": {"exit_code": 0, "log_path": str(work / "worker.log")},
        "parameters": {
            "tile_size": 256, "tile_overlap": 32, "base_seed": worker.BASE_SEED,
            "sparse_ratio": 2.0, "local_range": 11,
            "precision": "torch.bfloat16 (original implementation)",
            "ffmpeg_threads": 32, "encoder": "libx265",
            "benchmark_mode": "full", "max_frames": None,
        },
        "input_identity": {"path": "/inputs/04_Outpaint_4K.mp4", "size": 1, "sha256": "a" * 64},
        "anchor_identity": {"path": "/inputs/04_Input_8K.mp4", "size": 1, "sha256": "b" * 64},
        "upscale_sha256": worker.APPROVED_UPSCALE_SHA256,
        "work_dir": str(work), "request_path": str(work / "request.json"),
        "private_partial_manifest": str(work / "partial.json"),
        "request_sha256": "x", "private_partial_manifest_sha256": "y",
    }


def test_fake_full_complete_receipt_is_rejected(tmp_path):
    receipt = minimal_receipt(tmp_path / "shard", state="COMPLETE")
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="bounded partial"):
        merge.validate_shard_receipt(path, video_id="04", aggregate_work_dir=tmp_path,
                                     expected_layout=[{}] * 170)


def test_path_escape_is_rejected_before_artifact_use(tmp_path):
    receipt = minimal_receipt(tmp_path / "shard")
    receipt["work_dir"] = str(tmp_path.parent / "escape")
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="escapes"):
        merge.validate_shard_receipt(path, video_id="04", aggregate_work_dir=tmp_path,
                                     expected_layout=[{}] * 170)


def test_frame_mismatch_is_rejected(tmp_path):
    shard = tmp_path / "shard"
    shard.mkdir()
    request = shard / "request.json"
    request.write_text("{}")
    partial = shard / "partial.json"
    partial.write_text(json.dumps({
        "state": "COMPLETE", "model": "FlashVSR_v1.1", "processed_input_frames": 29,
        "input": {"path": "/inputs/04_Outpaint_4K.mp4"}, "tile_artifacts": [],
    }))
    (shard / "worker.log").write_text("ok")
    receipt = minimal_receipt(shard)
    receipt["request_sha256"] = merge.sha256_file(request)
    receipt["private_partial_manifest_sha256"] = merge.sha256_file(partial)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="1952"):
        merge.validate_shard_receipt(path, video_id="04", aggregate_work_dir=tmp_path,
                                     expected_layout=[{}] * 170)


def synthetic_receipt(video="04"):
    return {
        "input_identity": {"path": "/inputs/04_Outpaint_4K.mp4", "size": 10, "sha256": "a" * 64},
        "anchor_identity": {"path": "/inputs/04_Input_8K.mp4", "size": 20, "sha256": "b" * 64},
        "parameters": {"same": True}, "upscale_sha256": worker.APPROVED_UPSCALE_SHA256,
        "model": "FlashVSR_v1.1", "repository_commit": "commit",
        "processed_input_frames": 1952, "video_id": video,
    }


def synthetic_manifest(indices):
    artifacts = [{
        "tile": {"index": index, "x": index, "y": 0, "width": 256, "height": 256},
        "directory": f"/tiles/{index}", "tile_seed": worker.BASE_SEED + index,
        "temporal_unit_metadata": [f"/metadata/{index}.json"],
    } for index in indices]
    return {
        "input": {"path": "/inputs/04_Outpaint_4K.mp4", "frame_count": 1952},
        "padded_model_frames": 1953, "tile_artifacts": artifacts,
        "encode_defaults": {"ffmpeg_threads": 1, "encoder": "auto"},
        "partial_report": {"configuration": {}, "timings": {}},
    }


def test_aggregate_builds_deterministic_full_global_layout(tmp_path, monkeypatch):
    work = tmp_path.resolve()
    shard_paths = [work / "gpu0/receipt.json", work / "gpu1/receipt.json"]
    for path in shard_paths:
        path.parent.mkdir()
        path.write_text("{}")
    pairs = [(synthetic_receipt(), synthetic_manifest(range(0, 85))),
             (synthetic_receipt(), synthetic_manifest(range(85, 170)))]
    monkeypatch.setattr(merge, "validate_shard_receipt", lambda path, **kwargs: pairs.pop(0))
    monkeypatch.setattr(merge, "_expected_layout", lambda path: [item["tile"] for item in synthetic_manifest(range(170))["tile_artifacts"]])
    output = work / "aggregate/full.json"
    merge.aggregate_shards(video_id="04", shard_paths=shard_paths, work_dir=work,
                           output_manifest=output, output_target=work / "raw/output.mkv")
    value = json.loads(output.read_text())
    assert value["state"] == "COMPLETE"
    assert value["upscale_sha256"] == worker.APPROVED_UPSCALE_SHA256
    assert value["aggregation"]["completion_scope"] == "FULL_SPATIAL_LAYOUT"
    assert [item["tile"]["index"] for item in value["tile_artifacts"]] == list(range(170))
    assert [item["tile_seed"] for item in value["tile_artifacts"]] == [worker.BASE_SEED + i for i in range(170)]


@pytest.mark.parametrize("second_indices", [range(84, 169), range(86, 170)])
def test_aggregate_rejects_duplicate_or_gap(tmp_path, monkeypatch, second_indices):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        path.write_text("{}")
    pairs = [(synthetic_receipt(), synthetic_manifest(range(0, 85))),
             (synthetic_receipt(), synthetic_manifest(second_indices))]
    monkeypatch.setattr(merge, "validate_shard_receipt", lambda path, **kwargs: pairs.pop(0))
    monkeypatch.setattr(merge, "_expected_layout", lambda path: [{}] * 170)
    with pytest.raises(ValueError, match="gap-free"):
        merge.aggregate_shards(video_id="04", shard_paths=paths, work_dir=tmp_path,
                               output_manifest=tmp_path / "full.json", output_target=tmp_path / "raw.mkv")


def test_unexpected_names_are_rejected():
    with pytest.raises(RuntimeError, match="unexpected=.*debug.tmp"):
        merge._require_exact_names({"000000000.png", "debug.tmp"}, {"000000000.png"}, "tile")


def test_worker_rejects_unapproved_upscale_code(tmp_path):
    changed = tmp_path / "upscale.py"
    changed.write_text("# modified code\n")
    with pytest.raises(RuntimeError, match="not the approved production code"):
        worker.require_approved_upscale(changed)


def test_merge_rejects_unapproved_reference_code(tmp_path):
    changed = tmp_path / "upscale.py"
    changed.write_text("# modified code\n")
    with pytest.raises(RuntimeError, match="not the approved FlashVSR"):
        merge.require_approved_upscale(changed)


def test_internal_request_cannot_bypass_fixed_tile_size():
    request = {
        "schema": "a6000_flashvsr_shard_request/v1", "video_id": "04", "gpu_index": 0,
        "input": "/inputs/04_Outpaint_4K.mp4", "anchor": "/inputs/04_Input_8K.mp4",
        "assignment": {"tile_start": 0, "tile_end": 84},
        "selection": {"tile_start": 0, "tile_end": 84}, "mode": "full",
        "parameters": {
            "tile_size": 192, "tile_overlap": 32, "base_seed": worker.BASE_SEED,
            "sparse_ratio": 2.0, "local_range": 11,
            "precision": "torch.bfloat16 (original implementation)",
            "ffmpeg_threads": 32, "encoder": "libx265",
            "benchmark_mode": "full", "max_frames": None,
        },
    }
    with pytest.raises(ValueError, match="approved FlashVSR inference parameter"):
        worker.validate_request_contract(request)
