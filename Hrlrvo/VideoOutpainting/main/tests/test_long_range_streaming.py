from __future__ import annotations

import inspect
from itertools import pairwise

import pytest

from ltx_outpaint import inference, streaming_video


def _emitted_frame_ids(records: list[dict[str, object]]) -> list[int]:
    emitted: list[int] = []
    for record in records:
        begin, end = record["new_output_frame_range"]
        emitted.extend(range(begin, end))
    return emitted


def test_production_long_range_path_has_no_full_sequence_frame_accumulator() -> None:
    source = inspect.getsource(inference.sample_long_range)

    assert "chunks.append" not in source
    assert "torch.cat(chunks" not in source
    assert "read_video(" not in source
    assert "save_video(" not in source
    assert "_iter_long_range_chunks" in source
    assert "StreamingAnchorReader" in source
    assert "encode_segments" in source


def test_final_encoder_converts_and_encodes_one_frame_at_a_time() -> None:
    source = inspect.getsource(streaming_video.encode_segments)

    assert "for frame_array in iter_segment_frames" in source
    assert "VideoFrame.from_ndarray(frame_array" in source
    assert "_prepare_video_array" not in source


@pytest.mark.parametrize(
    "requested_frames",
    [1, 8, 9, 16, 17, 18, 48, 49, 57, 64, 65, 66, 72, 112, 113, 114, 160],
)
def test_chunk_plan_emits_every_requested_frame_exactly_once(
    requested_frames: int,
) -> None:
    records = inference._plan_long_range_chunks(
        requested_frames,
        inference.LongRangeSpec(),
        temporal_factor=8,
    )

    emitted = _emitted_frame_ids(records)

    assert emitted == list(range(requested_frames))
    assert len(emitted) == len(set(emitted)) == requested_frames
    assert inference._lineage_health(records, requested_frames) == (0, 0)


@pytest.mark.parametrize(
    ("spec", "temporal_factor", "requested_frames"),
    [
        (inference.LongRangeSpec(clip_frames=9, prefix_frames=1, stride_frames=8), 8, 34),
        (inference.LongRangeSpec(clip_frames=17, prefix_frames=9, stride_frames=8), 8, 42),
        (inference.LongRangeSpec(clip_frames=33, prefix_frames=9, stride_frames=24), 8, 83),
    ],
)
def test_chunk_plan_uses_configured_chunk_prefix_and_stride(
    spec: inference.LongRangeSpec,
    temporal_factor: int,
    requested_frames: int,
) -> None:
    spec.validate(temporal_factor)
    records = inference._plan_long_range_chunks(
        requested_frames,
        spec,
        temporal_factor=temporal_factor,
    )

    assert _emitted_frame_ids(records) == list(range(requested_frames))
    for previous, current in pairwise(records):
        assert current["absolute_start_frame"] - previous["absolute_start_frame"] == (
            spec.stride_frames
        )
        source_begin, _ = current["anchor_source_frame_range"]
        prefix_begin, prefix_end = current["prefix_output_frame_range"]
        assert source_begin == prefix_begin
        assert prefix_end - prefix_begin == spec.prefix_frames
        assert current["new_output_frame_range"][0] == prefix_end


@pytest.mark.parametrize("requested_frames", [66, 72, 73, 80, 89, 94, 114, 159])
def test_partial_tail_padding_never_changes_emitted_frame_count(
    requested_frames: int,
) -> None:
    temporal_factor = 8
    records = inference._plan_long_range_chunks(
        requested_frames,
        inference.LongRangeSpec(),
        temporal_factor=temporal_factor,
    )
    tail = records[-1]

    assert tail["working_frame_count"] >= tail["real_input_frame_count"]
    assert (tail["working_frame_count"] - 1) % temporal_factor == 0
    assert tail["repeat_last_padding_count"] == (
        tail["working_frame_count"] - tail["real_input_frame_count"]
    )
    assert tail["output_appended_count"] == (
        tail["new_output_frame_range"][1] - tail["new_output_frame_range"][0]
    )
    assert _emitted_frame_ids(records) == list(range(requested_frames))


@pytest.mark.parametrize("requested_frames", [66, 72, 113, 114, 160, 257])
def test_source_windows_are_ring_buffer_compatible(requested_frames: int) -> None:
    """Each source window reuses only the configured overlap, then advances."""
    spec = inference.LongRangeSpec()
    records = inference._plan_long_range_chunks(
        requested_frames,
        spec,
        temporal_factor=8,
    )

    for previous, current in pairwise(records):
        previous_begin, previous_end = previous["anchor_source_frame_range"]
        current_begin, current_end = current["anchor_source_frame_range"]
        overlap = max(0, previous_end - current_begin)

        assert current_begin - previous_begin == spec.stride_frames
        assert overlap <= spec.prefix_frames
        assert current_end - current_begin <= spec.clip_frames
