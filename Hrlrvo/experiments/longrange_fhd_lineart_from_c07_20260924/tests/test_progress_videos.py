from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "watch_progress_videos.py"
SPEC = importlib.util.spec_from_file_location("watch_progress_videos", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
watcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watcher
SPEC.loader.exec_module(watcher)


def _write_commits(root: Path, through: int, *, height: int = 2, width: int = 4) -> Path:
    segment_root = root / "chunks" / "test_mode"
    segment_root.mkdir(parents=True)
    start = 0
    for chunk in range(through + 1):
        count = 65 if chunk == 0 else 48
        segment_name = f"segment_{chunk:06d}.npy"
        np.save(
            segment_root / segment_name,
            np.full((count, height, width, 3), chunk, dtype=np.uint8),
            allow_pickle=False,
        )
        record = {
            "chunk_index": chunk,
            "segment_file": segment_name,
            "output_appended_count": count,
            "new_output_frame_range": [start, start + count],
        }
        (segment_root / f"chunk_{chunk:06d}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        start += count
    return segment_root


def test_commit_validation_and_milestone_frame_accounting(tmp_path: Path) -> None:
    segment_root = _write_commits(tmp_path, 4)

    records, missing, geometry = watcher.load_committed_records(segment_root, 4)

    assert records is not None
    assert missing is None
    assert geometry == (4, 2)
    assert sum(record["output_appended_count"] for record in records) == 257
    assert watcher.expected_frames_through(4) == 257
    assert watcher.expected_frames_through(23) == 1169
    assert watcher.output_name(4) == "C00_C04_257f.mp4"


def test_cpu_encoder_output_has_exact_video_spec(tmp_path: Path) -> None:
    segment = np.arange(5 * 4 * 8 * 3, dtype=np.uint8).reshape(5, 4, 8, 3)
    np.save(tmp_path / "segment.npy", segment, allow_pickle=False)
    output = tmp_path / "progress.mp4"

    encoded = watcher.encode_segments(
        [{"segment_file": "segment.npy"}],
        tmp_path,
        output,
        width=8,
        height=4,
        expected_frames=5,
    )
    metadata = watcher.validate_video(output, expected_frames=5, width=8, height=4)

    assert encoded == 5
    assert metadata == watcher.VideoMetadata(5, 24.0, 8, 4)
    assert not list(tmp_path.glob(".*.partial.mp4"))


def test_wait_checks_commits_again_after_producer_exit(tmp_path: Path, monkeypatch) -> None:
    segment_root = tmp_path / "chunks"
    calls = 0
    expected = ([{"chunk_index": 0}], None, (4, 2))

    def load(_root: Path, _milestone: int):
        nonlocal calls
        calls += 1
        return (None, 0, None) if calls == 1 else expected

    monkeypatch.setattr(watcher, "load_committed_records", load)
    records, geometry = watcher.wait_for_records(
        segment_root, 4, 999999, poll_seconds=0.001, is_alive=lambda _pid: False
    )

    assert records == expected[0]
    assert geometry == (4, 2)
    assert calls == 2


def test_watcher_creates_then_restart_validates_and_reuses(tmp_path: Path) -> None:
    _write_commits(tmp_path, 4)
    encoded: list[tuple[int, Path]] = []
    known_outputs: set[Path] = set()

    def encode(records, _segment_root, output_path, **kwargs):
        output_path.touch()
        known_outputs.add(output_path)
        encoded.append((len(records), output_path))
        return kwargs["expected_frames"]

    def validate(path, *, expected_frames, width, height, fps):
        if path not in known_outputs:
            raise FileNotFoundError(path)
        return watcher.VideoMetadata(expected_frames, fps, width, height)

    first = watcher.run_watcher(
        root=tmp_path,
        mode="test_mode",
        end_chunk=4,
        producer_pid=1,
        poll_seconds=0.001,
        encoder=encode,
        validator=validate,
    )
    second = watcher.run_watcher(
        root=tmp_path,
        mode="test_mode",
        end_chunk=4,
        producer_pid=1,
        poll_seconds=0.001,
        encoder=encode,
        validator=validate,
    )

    assert encoded == [(5, tmp_path / "outputs/test_mode/progress/C00_C04_257f.mp4")]
    assert first[0]["disposition"] == "created"
    assert second[0]["disposition"] == "reused"
    latest = json.loads(
        (tmp_path / "outputs/test_mode/progress/latest.json").read_text(encoding="utf-8")
    )
    assert latest["event"] == "progress_watcher_complete"
    assert latest["frame_count"] == 257
    assert latest["status"] == "complete"
    index = (tmp_path / "outputs/test_mode/progress/index.html").read_text(
        encoding="utf-8"
    )
    assert 'http-equiv="refresh" content="60"' in index
    assert 'src="C00_C04_257f.mp4"' in index
