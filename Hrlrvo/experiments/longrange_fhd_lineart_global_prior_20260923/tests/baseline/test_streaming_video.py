from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest
import torch

from ltx_outpaint import inference, streaming_video
from ltx_outpaint.streaming_video import (
    SegmentedOutputStore,
    StreamingAnchorReader,
    encode_segments,
    iter_segment_frames,
)

ROOT = Path(__file__).resolve().parents[1]


def _records(requested_frames: int = 72) -> list[dict[str, object]]:
    return inference._plan_long_range_chunks(
        requested_frames,
        inference.LongRangeSpec(),
        temporal_factor=8,
    )


def _frames(count: int, *, height: int = 2, width: int = 4) -> torch.Tensor:
    values = torch.arange(count, dtype=torch.float32).remainder(256).div(255)
    return values[:, None, None, None].expand(count, 3, height, width).clone()


def _commit(
    store: SegmentedOutputStore,
    record: dict[str, object],
    *,
    prefix: torch.Tensor,
    cpu_rng_state: torch.Tensor,
) -> dict[str, object]:
    return store.commit_chunk(
        record,
        _frames(int(record["output_appended_count"])),
        prefix,
        chunk_seed=100 + int(record["absolute_start_frame"]),
        cpu_rng_state=cpu_rng_state,
        cuda_rng_state=None,
        cuda_device=None,
    )


def test_lossless_segment_pixels_match_legacy_prepare_video_array(tmp_path: Path) -> None:
    inference._install_official_sources(Path("/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/models/LTX2.3"))
    from ltx_trainer.video_utils import _prepare_video_array

    frames = torch.linspace(0, 1, 5 * 3 * 4 * 6, dtype=torch.float32).reshape(
        5, 3, 4, 6
    )
    path = tmp_path / "segment.npy"

    streaming_video._write_uint8_segment(path, frames)
    actual = np.load(path, allow_pickle=False)
    expected = _prepare_video_array(frames, video_format="FCHW")

    assert actual.dtype == expected.dtype == np.uint8
    assert actual.shape == expected.shape == (5, 4, 6, 3)
    assert np.array_equal(actual, expected)


def test_atomic_commit_roundtrips_exact_fp32_prefix_and_cpu_rng(
    tmp_path: Path,
) -> None:
    store = SegmentedOutputStore(tmp_path / "segments")
    store.prepare({"request_fingerprint": "test"}, resume=False)
    record = _records()[0]
    prefix = torch.linspace(0, 1, 17 * 3 * 2 * 4, dtype=torch.float32).reshape(
        17, 3, 2, 4
    )
    cpu_rng_state = torch.get_rng_state().clone()

    committed = _commit(
        store,
        record,
        prefix=prefix,
        cpu_rng_state=cpu_rng_state,
    )
    resumed = store.load_resume_state(
        0,
        prefix_frames=17,
        output_height=2,
        output_width=4,
    )

    assert committed["next_prefix_dtype"] == "float32"
    assert torch.equal(resumed.previous_prefix, prefix)
    assert resumed.previous_prefix.dtype == torch.float32
    assert torch.equal(resumed.cpu_rng_state, cpu_rng_state)
    assert resumed.cuda_rng_state is None
    assert resumed.cuda_device is None
    assert not list(store.root.glob(".*.partial"))


def test_bootstrap_video_commits_chunk_zero_and_exact_decoded_prefix(
    tmp_path: Path,
) -> None:
    bootstrap = tmp_path / "bootstrap.mp4"
    with av.open(str(bootstrap), mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = 4
        stream.height = 2
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for frame_id in range(65):
            pixels = np.full((2, 4, 3), frame_id * 3, dtype=np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(pixels, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    store = SegmentedOutputStore(tmp_path / "segments")
    store.prepare({"request_fingerprint": "bootstrap-test"}, resume=False)
    record = _records(65)[0]
    record["bootstrap_first_chunk"] = {"path": str(bootstrap)}
    cpu_rng_state = torch.get_rng_state().clone()

    committed, prefix = store.commit_bootstrap_video(
        record,
        bootstrap,
        prefix_frames=17,
        expected_fps=24.0,
        fps_tolerance=0.001,
        output_height=2,
        output_width=4,
        chunk_seed=100,
        cpu_rng_state=cpu_rng_state,
        cuda_rng_state=None,
        cuda_device=None,
    )

    segment = np.load(store.root / "segment_000000.npy", allow_pickle=False)
    expected_prefix = (
        torch.from_numpy(segment[-17:].copy())
        .float()
        .div(255.0)
        .permute(0, 3, 1, 2)
    )
    resumed = store.load_resume_state(
        0,
        prefix_frames=17,
        output_height=2,
        output_width=4,
    )

    assert segment.shape == (65, 2, 4, 3)
    assert committed["bootstrap_first_chunk"] == {"path": str(bootstrap)}
    assert torch.equal(prefix, expected_prefix)
    assert torch.equal(resumed.previous_prefix, expected_prefix)
    assert torch.equal(resumed.cpu_rng_state, cpu_rng_state)
    assert store.scan_completed(
        [record], output_height=2, output_width=4, base_seed=100
    ) == (1, 1)


def test_resume_rejects_prefix_state_with_missing_cpu_rng(tmp_path: Path) -> None:
    store = SegmentedOutputStore(tmp_path / "segments")
    store.prepare({"request_fingerprint": "test"}, resume=False)
    record = _records()[0]
    prefix = torch.zeros(17, 3, 2, 4)
    committed = _commit(
        store,
        record,
        prefix=prefix,
        cpu_rng_state=torch.get_rng_state().clone(),
    )
    prefix_path = store.root / str(committed["next_prefix_file"])
    state = torch.load(prefix_path, map_location="cpu", weights_only=True)
    del state["cpu_rng_state"]
    torch.save(state, prefix_path)
    record_path = store.root / "chunk_000000.json"
    metadata = streaming_video._load_json(record_path)
    metadata["next_prefix_byte_size"] = prefix_path.stat().st_size
    metadata["next_prefix_state_sha256"] = streaming_video.file_sha256(prefix_path)
    streaming_video._atomic_json(record_path, metadata)

    with pytest.raises(RuntimeError, match="CPU RNG state is missing"):
        store.load_resume_state(
            0,
            prefix_frames=17,
            output_height=2,
            output_width=4,
        )


def test_resume_rejects_valid_but_altered_fp32_prefix_state(tmp_path: Path) -> None:
    store = SegmentedOutputStore(tmp_path / "segments")
    store.prepare({"request_fingerprint": "test"}, resume=False)
    record = _records()[0]
    committed = _commit(
        store,
        record,
        prefix=torch.zeros(17, 3, 2, 4),
        cpu_rng_state=torch.get_rng_state().clone(),
    )
    prefix_path = store.root / str(committed["next_prefix_file"])
    state = torch.load(prefix_path, map_location="cpu", weights_only=True)
    state["prefix"] = torch.ones_like(state["prefix"])
    torch.save(state, prefix_path)
    record_path = store.root / "chunk_000000.json"
    metadata = streaming_video._load_json(record_path)
    metadata["next_prefix_byte_size"] = prefix_path.stat().st_size
    streaming_video._atomic_json(record_path, metadata)

    with pytest.raises(RuntimeError, match="state digest does not match"):
        store.load_resume_state(
            0,
            prefix_frames=17,
            output_height=2,
            output_width=4,
        )


def test_resume_scan_accepts_only_a_contiguous_committed_prefix(tmp_path: Path) -> None:
    records = _records(114)
    store = SegmentedOutputStore(tmp_path / "segments")
    store.prepare({"request_fingerprint": "test"}, resume=False)
    prefix = torch.zeros(17, 3, 2, 4)
    rng = torch.get_rng_state().clone()
    _commit(store, records[0], prefix=prefix, cpu_rng_state=rng)

    assert store.scan_completed(
        records,
        output_height=2,
        output_width=4,
        base_seed=100,
    ) == (1, 3)

    _commit(store, records[2], prefix=prefix, cpu_rng_state=rng)
    with pytest.raises(RuntimeError, match="not a contiguous prefix"):
        store.scan_completed(
            records,
            output_height=2,
            output_width=4,
            base_seed=100,
        )


def test_resume_rejects_same_size_corrupt_committed_segment(tmp_path: Path) -> None:
    records = _records()
    store = SegmentedOutputStore(tmp_path / "segments")
    store.prepare({"request_fingerprint": "test"}, resume=False)
    committed = _commit(
        store,
        records[0],
        prefix=torch.zeros(17, 3, 2, 4),
        cpu_rng_state=torch.get_rng_state().clone(),
    )
    segment_path = store.root / str(committed["segment_file"])
    with segment_path.open("r+b") as handle:
        handle.seek(-1, 2)
        final_byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final_byte[0] ^ 0xFF]))

    with pytest.raises(RuntimeError, match="segment is corrupt"):
        store.scan_completed(
            records,
            output_height=2,
            output_width=4,
            base_seed=100,
        )


def test_cpu_rng_resume_restore_replays_next_random_values() -> None:
    original = torch.get_rng_state()
    try:
        torch.manual_seed(1234)
        saved = torch.get_rng_state().clone()
        expected = torch.rand(8)
        torch.rand(32)

        inference._restore_post_chunk_rng(
            saved,
            None,
            None,
            expected_device="cpu",
        )

        assert torch.equal(torch.rand(8), expected)
    finally:
        torch.set_rng_state(original)


def test_alternating_prefix_slot_preserves_latest_commit_if_next_record_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(114)
    store = SegmentedOutputStore(tmp_path / "segments")
    store.prepare({"request_fingerprint": "test"}, resume=False)
    rng = torch.get_rng_state().clone()
    prefix0 = torch.full((17, 3, 2, 4), 0.25)
    prefix1 = torch.full((17, 3, 2, 4), 0.5)
    _commit(store, records[0], prefix=prefix0, cpu_rng_state=rng)
    _commit(store, records[1], prefix=prefix1, cpu_rng_state=rng)
    original_atomic_json = streaming_video._atomic_json

    def fail_completion_marker(path: Path, value: object) -> None:
        if path.name == "chunk_000002.json":
            raise OSError("simulated crash before completion marker")
        original_atomic_json(path, value)

    monkeypatch.setattr(streaming_video, "_atomic_json", fail_completion_marker)
    with pytest.raises(OSError, match="simulated crash"):
        _commit(
            store,
            records[2],
            prefix=torch.full((17, 3, 2, 4), 0.75),
            cpu_rng_state=rng,
        )

    resumed = store.load_resume_state(
        1,
        prefix_frames=17,
        output_height=2,
        output_width=4,
    )
    assert torch.equal(resumed.previous_prefix, prefix1)
    assert not (store.root / "chunk_000002.json").exists()
    assert store.scan_completed(
        records,
        output_height=2,
        output_width=4,
        base_seed=100,
    ) == (2, 3)


def test_segment_iterator_preserves_committed_record_and_frame_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "segments"
    root.mkdir()
    expected_ids = [0, 1, 2, 3, 4, 5]
    records = []
    for chunk_index, ids in enumerate(([0, 1, 2], [3], [4, 5])):
        path = root / f"segment_{chunk_index:06d}.npy"
        pixels = np.asarray(ids, dtype=np.uint8)[:, None, None, None]
        np.save(path, np.broadcast_to(pixels, (len(ids), 2, 4, 3)))
        records.append({"segment_file": path.name})

    actual_ids = [int(frame[0, 0, 0]) for frame in iter_segment_frames(records, root)]

    assert actual_ids == expected_ids


def test_streaming_anchor_reader_matches_sequential_rgb24_decode_and_bounds_ring(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    source = [
        np.full((2, 4, 3), frame_id * 3, dtype=np.uint8)
        for frame_id in range(72)
    ]
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = 4
        stream.height = 2
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for pixels in source:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(video_path)) as container:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]

    records = _records(72)
    with StreamingAnchorReader(video_path, stride_frames=48) as reader:
        for record in records:
            actual = reader.read_chunk(record)
            start, end = record["anchor_source_frame_range"]
            expected = (
                torch.from_numpy(np.stack(decoded[start:end]))
                .float()
                .div(255)
                .permute(0, 3, 1, 2)
            )
            assert torch.equal(actual, expected)
            assert reader.buffered_rgb24_frames <= 17


def test_encode_segments_cpu_smoke_preserves_frame_count_and_codec_options(
    tmp_path: Path,
) -> None:
    root = tmp_path / "segments"
    root.mkdir()
    segment = np.stack(
        [np.full((2, 4, 3), value, dtype=np.uint8) for value in (0, 64, 128, 255)]
    )
    np.save(root / "segment_000000.npy", segment)
    records = [{"segment_file": "segment_000000.npy"}]
    output = tmp_path / "output.mp4"

    encoded = encode_segments(
        records,
        root,
        output,
        fps=24,
        width=4,
        height=2,
        expected_frames=4,
    )
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(video=0))

    assert encoded == len(decoded) == 4
    assert stream.codec_context.name == "h264"
    assert stream.codec_context.pix_fmt == "yuv420p"
    assert int(stream.average_rate) == 24
