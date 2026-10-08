from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = EXPERIMENT_ROOT / "package"
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cert_pipeline import flash_composite as composite  # noqa: E402


def _headers(*, fps: str = "16/1", frames: int = 1952) -> dict[str, object]:
    return {
        "width": 15360,
        "height": 8640,
        "avg_frame_rate": fps,
        "r_frame_rate": fps,
        "frame_count": frames,
        "duration": 122.0,
        "decoded_frame_count_performed": False,
    }


@pytest.mark.parametrize(
    ("video_id", "input_name", "anchor_name"),
    [
        ("03", "04_Outpaint_4K.mp4", "03_Input_8K.mp4"),
        ("03", "03_Outpaint_4K.mp4", "04_Input_8K.mp4"),
        ("04", "03_Outpaint_4K.mp4", "03_Input_8K.mp4"),
    ],
)
def test_wrong_anchor_or_input_pair_is_rejected(video_id, input_name, anchor_name):
    with pytest.raises(composite.PipelineError, match="basename|requires"):
        composite.assert_video_identity_pair(
            video_id, Path(input_name), Path(anchor_name)
        )


def test_missing_full_tile_proof_is_rejected():
    args = argparse.Namespace(
        video_id="03", tile_size=256, tile_overlap=32, local_range=11,
        sparse_ratio=2.0, seed=71634910, original_8k=Path("03_Input_8K.mp4"),
    )
    manifest = {
        "video_id": "03",
        "upscale_sha256": composite.APPROVED_UPSCALE_SHA256,
        "tile_count": 170,
        "aggregation": {
            "completion_scope": "PARTIAL_SPATIAL_TILES",
            "global_tile_indices": list(range(170)),
            "source_shard_receipts": ["a.json", "b.json"],
        },
    }
    with pytest.raises(composite.PipelineError, match="full spatial layout"):
        composite.validate_aggregate_contract(
            manifest, args=args,
            input_metadata={"width": 3840, "height": 2160},
            reference=SimpleNamespace(),
        )


@pytest.mark.parametrize("existing_name", [
    "final.mp4", "raw.mp4", "final_manifest.json", "raw_encode_report.json",
])
def test_postprocess_refuses_every_preexisting_output_or_report(tmp_path, existing_name):
    paths = {
        "output": tmp_path / "final.mp4",
        "raw_output": tmp_path / "raw.mp4",
        "final_manifest": tmp_path / "final_manifest.json",
        "raw_report": tmp_path / "raw_encode_report.json",
    }
    (tmp_path / existing_name).write_bytes(b"stale")
    with pytest.raises(composite.PipelineError, match="fresh-only"):
        composite.require_fresh_postprocess_targets(**paths)


def test_postprocess_requires_approved_actual_reference_hash(tmp_path, monkeypatch):
    reference = tmp_path / "upscale.py"
    reference.write_text("unchanged production snapshot")
    monkeypatch.setattr(composite, "sha256_file", lambda _path: "0" * 64)
    with pytest.raises(composite.PipelineError, match="not approved"):
        composite.require_approved_reference(reference)


def test_aggregate_requires_approved_inference_source_hash():
    args = argparse.Namespace(video_id="03")
    with pytest.raises(composite.PipelineError, match="inference-source hash"):
        composite.validate_aggregate_contract(
            {"upscale_sha256": "0" * 64}, args=args,
            input_metadata={}, reference=SimpleNamespace(),
        )


def test_nonzero_output_is_required(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.touch()
    with pytest.raises(composite.PipelineError, match="missing or empty"):
        composite.bounded_output_evidence(
            empty, _headers(), expected_frames=1952,
            encoder_frames_written=1952, encoder_exit_code=0, role="final",
        )


@pytest.mark.parametrize(
    ("metadata", "counter", "exit_code", "message"),
    [
        (_headers(fps="15/1"), 1952, 0, "frame_rate"),
        (_headers(frames=1951), 1952, 0, "frames"),
        (_headers(), 0, 0, "encoder wrote"),
        (_headers(), 1952, 9, "exit code"),
    ],
)
def test_bad_header_or_encoder_evidence_fails(
    tmp_path, metadata, counter, exit_code, message
):
    output = tmp_path / "output.mp4"
    output.write_bytes(b"not-empty")
    with pytest.raises(composite.PipelineError, match=message):
        composite.bounded_output_evidence(
            output, metadata, expected_frames=1952,
            encoder_frames_written=counter, encoder_exit_code=exit_code, role="final",
        )


def test_decoder_flag_absence_is_rejected(tmp_path):
    output = tmp_path / "output.mp4"
    output.write_bytes(b"not-empty")
    headers = _headers()
    del headers["decoded_frame_count_performed"]
    with pytest.raises(composite.PipelineError, match="bounded header-only"):
        composite.bounded_output_evidence(
            output, headers, expected_frames=1952,
            encoder_frames_written=1952, encoder_exit_code=0, role="final",
        )


def test_bounded_evidence_never_claims_full_decode(tmp_path):
    output = tmp_path / "output.mp4"
    output.write_bytes(b"not-empty")
    evidence = composite.bounded_output_evidence(
        output, _headers(), expected_frames=1952,
        encoder_frames_written=1952, encoder_exit_code=0, role="final",
    )
    assert evidence["decoded_full_frame_count"] is False
    assert "full decode was not performed" in evidence["claim"]


def test_header_probe_never_requests_count_frames_or_frame_decode(tmp_path, monkeypatch):
    output = tmp_path / "output.mp4"
    output.write_bytes(b"not-empty")
    seen = []

    def fake_run(command, **_kwargs):
        seen.extend(command)
        payload = {
            "streams": [{
                "width": 15360, "height": 8640,
                "avg_frame_rate": "16/1", "r_frame_rate": "16/1",
                "nb_frames": "1952", "duration": "122.0",
            }],
            "format": {"duration": "122.0"},
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(composite, "run", fake_run)
    metadata = composite.probe_video_headers(output)
    assert "-count_frames" not in seen
    assert "-show_frames" not in seen
    assert metadata["decoded_frame_count_performed"] is False
