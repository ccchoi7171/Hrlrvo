"""Build bounded-memory cumulative progress videos from committed chunk segments.

This watcher is deliberately independent of the CUDA producer.  A chunk is
visible only after its JSON commit record exists, and frames are read from the
lossless ``.npy`` segments with memory mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MILESTONES = (4, 9, 14, 19, 23)
FPS = 24.0


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps: float
    width: int
    height: int


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _expected_chunk_count(chunk_index: int) -> int:
    return 65 if chunk_index == 0 else 48


def expected_frames_through(chunk_index: int) -> int:
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    return 65 + 48 * chunk_index


def _segment_path(segment_root: Path, record: Mapping[str, Any]) -> Path:
    name = record.get("segment_file")
    if not isinstance(name, str) or not name:
        raise ValueError("Commit record has no valid segment_file")
    path = segment_root / name
    if path.parent.resolve() != segment_root.resolve():
        raise ValueError(f"segment_file must name a file directly under {segment_root}: {name}")
    return path


def load_committed_records(
    segment_root: Path, end_chunk: int
) -> tuple[list[dict[str, Any]] | None, int | None, tuple[int, int] | None]:
    """Load and validate commits through ``end_chunk``, or report the first missing one."""
    records: list[dict[str, Any]] = []
    geometry: tuple[int, int] | None = None
    for chunk_index in range(end_chunk + 1):
        commit_path = segment_root / f"chunk_{chunk_index:06d}.json"
        if not commit_path.exists():
            return None, chunk_index, None
        with commit_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if not isinstance(record, dict):
            raise TypeError(f"Expected JSON object in {commit_path}")
        if int(record.get("chunk_index", -1)) != chunk_index:
            raise ValueError(f"Wrong chunk_index in {commit_path}")
        count = int(record.get("output_appended_count", -1))
        expected_count = _expected_chunk_count(chunk_index)
        if count != expected_count:
            raise ValueError(
                f"C{chunk_index} committed {count} frames, expected {expected_count}"
            )
        expected_range = [expected_frames_through(chunk_index - 1) if chunk_index else 0,
                          expected_frames_through(chunk_index)]
        if record.get("new_output_frame_range") != expected_range:
            raise ValueError(
                f"C{chunk_index} has frame range {record.get('new_output_frame_range')}, "
                f"expected {expected_range}"
            )

        segment_path = _segment_path(segment_root, record)
        if not segment_path.is_file():
            raise RuntimeError(f"Committed segment is missing: {segment_path}")
        segment = np.load(segment_path, mmap_mode="r", allow_pickle=False)
        try:
            if segment.dtype != np.uint8 or segment.ndim != 4 or segment.shape[-1] != 3:
                raise ValueError(
                    f"Invalid segment array {segment_path}: shape={segment.shape}, "
                    f"dtype={segment.dtype}"
                )
            current_geometry = (int(segment.shape[2]), int(segment.shape[1]))
            if segment.shape[0] != count:
                raise ValueError(
                    f"Segment {segment_path} has {segment.shape[0]} frames, expected {count}"
                )
            if geometry is None:
                geometry = current_geometry
            elif current_geometry != geometry:
                raise ValueError(
                    f"Segment {segment_path} geometry {current_geometry} differs from {geometry}"
                )
        finally:
            del segment
        records.append(record)
    return records, None, geometry


def producer_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return len(fields) >= 3 and fields[2] != "Z"


def wait_for_records(
    segment_root: Path,
    milestone: int,
    producer_pid: int,
    *,
    poll_seconds: float = 10.0,
    is_alive: Callable[[int], bool] = producer_alive,
) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    while True:
        records, missing, geometry = load_committed_records(segment_root, milestone)
        if records is not None and geometry is not None:
            return records, geometry
        if not is_alive(producer_pid):
            # Close the last-chunk race: a producer may publish its atomic commit
            # immediately before exiting, after the first directory observation.
            records, missing, geometry = load_committed_records(segment_root, milestone)
            if records is not None and geometry is not None:
                return records, geometry
            raise RuntimeError(
                f"Producer {producer_pid} exited before C{missing} commit "
                f"(needed for C{milestone} progress video)"
            )
        time.sleep(poll_seconds)


def probe_video(path: Path) -> VideoMetadata:
    frame_count = 0
    width: int | None = None
    height: int | None = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or FPS)
        for frame in container.decode(video=0):
            current = (int(frame.width), int(frame.height))
            if width is None:
                width, height = current
            elif current != (width, height):
                raise ValueError(f"Variable-resolution video: {path}")
            frame_count += 1
    if not frame_count or width is None or height is None:
        raise ValueError(f"No decodable video frames: {path}")
    return VideoMetadata(frame_count, fps, width, height)


def validate_video(
    path: Path, *, expected_frames: int, width: int, height: int, fps: float = FPS
) -> VideoMetadata:
    metadata = probe_video(path)
    expected = VideoMetadata(expected_frames, fps, width, height)
    if (
        metadata.frame_count != expected.frame_count
        or metadata.width != expected.width
        or metadata.height != expected.height
        or abs(metadata.fps - expected.fps) > 0.001
    ):
        raise ValueError(f"Video validation failed for {path}: {metadata}, expected {expected}")
    return metadata


def encode_segments(
    records: Sequence[Mapping[str, Any]],
    segment_root: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    expected_frames: int,
    fps: float = FPS,
) -> int:
    """Stream committed arrays into an atomic x264 MP4 with two CPU threads."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    temporary.unlink(missing_ok=True)
    encoded_frames = 0
    try:
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream("libx264", rate=int(fps))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "threads": "2"}
            for record in records:
                segment = np.load(
                    _segment_path(segment_root, record), mmap_mode="r", allow_pickle=False
                )
                try:
                    for pixels in segment:
                        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                        encoded_frames += 1
                finally:
                    del segment
            for packet in stream.encode():
                container.mux(packet)
        if encoded_frames != expected_frames:
            raise RuntimeError(
                f"Encoded {encoded_frames} frames, expected {expected_frames}"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return encoded_frames


def output_name(milestone: int) -> str:
    return f"C00_C{milestone:02d}_{expected_frames_through(milestone)}f.mp4"


def write_index(progress_root: Path, events: Sequence[Mapping[str, Any]]) -> None:
    """Atomically publish a tiny self-refreshing index of ready milestones."""
    cards = "\n".join(
        f'''<section><h2>C0–C{int(event["milestone_chunk"])}</h2>
<video controls preload="metadata" src="{Path(str(event["path"])).name}"></video>
<p><a href="{Path(str(event["path"])).name}">Download MP4</a> · {int(event["frame_count"])} frames</p></section>'''
        for event in events
    )
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>Cumulative inference progress</title>
<style>body{{font:16px sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}
video{{width:100%;height:auto;background:#111}}section{{margin:2rem 0}}</style></head>
<body><h1>Cumulative inference progress</h1>{cards or "<p>Waiting for C4…</p>"}</body></html>
'''
    path = progress_root / "index.html"
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(html)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_watcher(
    *,
    root: Path,
    mode: str,
    end_chunk: int,
    producer_pid: int,
    poll_seconds: float = 10.0,
    encoder: Callable[..., int] = encode_segments,
    validator: Callable[..., VideoMetadata] = validate_video,
) -> list[dict[str, Any]]:
    if end_chunk not in MILESTONES:
        raise ValueError(f"--end-chunk must be one of {MILESTONES}")
    segment_root = root / "chunks" / mode
    progress_root = root / "outputs" / mode / "progress"
    progress_root.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    write_index(progress_root, events)

    for milestone in (value for value in MILESTONES if value <= end_chunk):
        records, (width, height) = wait_for_records(
            segment_root,
            milestone,
            producer_pid,
            poll_seconds=poll_seconds,
        )
        frame_count = expected_frames_through(milestone)
        output_path = progress_root / output_name(milestone)
        disposition = "reused"
        try:
            metadata = validator(
                output_path,
                expected_frames=frame_count,
                width=width,
                height=height,
                fps=FPS,
            )
        except (FileNotFoundError, OSError, ValueError, av.error.FFmpegError):
            encoder(
                records,
                segment_root,
                output_path,
                width=width,
                height=height,
                expected_frames=frame_count,
                fps=FPS,
            )
            metadata = validator(
                output_path,
                expected_frames=frame_count,
                width=width,
                height=height,
                fps=FPS,
            )
            disposition = "created"

        event: dict[str, Any] = {
            "event": "progress_video",
            "status": "ready",
            "mode": mode,
            "milestone_chunk": milestone,
            "chunks_included": milestone + 1,
            "frame_count": frame_count,
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "path": str(output_path),
            "disposition": disposition,
        }
        _atomic_json(progress_root / "latest.json", event)
        write_index(progress_root, [*events, event])
        print(json.dumps(event, sort_keys=True), flush=True)
        events.append(event)

    complete = dict(events[-1])
    complete.update({"event": "progress_watcher_complete", "status": "complete"})
    _atomic_json(progress_root / "latest.json", complete)
    print(json.dumps(complete, sort_keys=True), flush=True)
    return events


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--end-chunk", required=True, type=int, choices=MILESTONES)
    parser.add_argument("--producer-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=10.0, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.producer_pid <= 0:
        raise ValueError("--producer-pid must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    run_watcher(
        root=ROOT,
        mode=args.mode,
        end_chunk=args.end_chunk,
        producer_pid=args.producer_pid,
        poll_seconds=args.poll_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
