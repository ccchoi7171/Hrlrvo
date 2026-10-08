"""Bounded-memory video I/O for LongRange inference only.

The training and upstream LTX media utilities intentionally remain unchanged.
LongRange inference uses this module to keep source RGB24 frames in a small
ring, commit finalized output frames as lossless uint8 NumPy segments, retain
only the latest decoded FP32 prefix state, and encode the final MP4 one frame at
a time.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import av
import numpy as np
import torch


@dataclass(frozen=True)
class VideoMetadata:
    """Exact decoded input metadata collected without retaining frame pixels."""

    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class ResumeState:
    """Exact recurrent pixels and global RNG state after one completed chunk."""

    previous_prefix: torch.Tensor
    cpu_rng_state: torch.Tensor
    cuda_rng_state: torch.Tensor | None
    cuda_device: str | None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def file_sha256(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Hash a file with constant memory for resume lineage/integrity checks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def probe_video(video_path: Path) -> VideoMetadata:
    """Decode frame headers exactly while keeping only scalar input metadata."""
    frame_count = 0
    width = height = None
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate or video_stream.base_rate or 24)
        for frame in container.decode(video=0):
            if width is None:
                width, height = int(frame.width), int(frame.height)
            elif (frame.width, frame.height) != (width, height):
                raise ValueError("Variable-resolution input video is not supported")
            frame_count += 1
    if frame_count == 0 or width is None or height is None:
        raise ValueError(f"Input video has no decodable frames: {video_path}")
    return VideoMetadata(
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )


class StreamingAnchorReader:
    """Sequential PyAV reader retaining at most one source overlap as RGB24."""

    def __init__(self, video_path: Path, *, stride_frames: int) -> None:
        if stride_frames <= 0:
            raise ValueError("stride_frames must be positive")
        self._video_path = video_path
        self._stride_frames = stride_frames
        self._container: Any | None = None
        self._decoder: Iterator[Any] | None = None
        self._decoded_count = 0
        self._buffer_start = 0
        self._buffer: deque[np.ndarray] = deque()

    def __enter__(self) -> Self:
        self._container = av.open(str(self._video_path))
        self._decoder = iter(self._container.decode(video=0))
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._container is not None:
            self._container.close()
        self._container = None
        self._decoder = None
        self._buffer.clear()

    @property
    def buffered_rgb24_frames(self) -> int:
        return len(self._buffer)

    def _decode_until(self, absolute_end: int, *, retain_from: int) -> None:
        if self._decoder is None:
            raise RuntimeError("StreamingAnchorReader must be used as a context manager")
        while self._decoded_count < absolute_end:
            try:
                frame = next(self._decoder)
            except StopIteration as error:
                raise RuntimeError(
                    f"Input decode ended at frame {self._decoded_count}, before {absolute_end}"
                ) from error
            index = self._decoded_count
            self._decoded_count += 1
            if index >= retain_from:
                self._buffer.append(frame.to_ndarray(format="rgb24"))

    def read_chunk(self, record: Mapping[str, Any]) -> torch.Tensor:
        """Return one legacy-equivalent FP32 FCHW source chunk.

        Conversion deliberately matches ``ltx_trainer.video_utils.read_video``:
        RGB24 arrays are stacked in FHWC layout, converted to FP32, divided by
        255, and only then permuted to FCHW.
        """
        start = int(record["absolute_start_frame"])
        real_count = int(record["real_input_frame_count"])
        end = start + real_count
        if start < self._buffer_start:
            raise RuntimeError("Source chunks must be requested in increasing order")

        while self._buffer_start < start and self._buffer:
            self._buffer.popleft()
            self._buffer_start += 1
        if self._buffer_start < start:
            self._decode_until(start, retain_from=start)
            self._buffer_start = start
        self._decode_until(end, retain_from=start)
        if len(self._buffer) < real_count:
            raise RuntimeError("Input ring does not contain the requested source chunk")

        frames_np = np.stack(tuple(self._buffer)[:real_count], axis=0)
        frames = torch.from_numpy(frames_np).float().div(255.0).permute(0, 3, 1, 2)

        next_start = start + self._stride_frames
        while self._buffer_start < next_start and self._buffer:
            self._buffer.popleft()
            self._buffer_start += 1
        return frames


def _write_uint8_segment(
    path: Path,
    frames: torch.Tensor,
) -> tuple[list[int], int, str]:
    """Atomically write FCHW FP32 [0,1] frames as encoder-equivalent RGB24."""
    if frames.device.type != "cpu" or frames.dtype != torch.float32:
        raise TypeError("Finalized segment frames must be CPU FP32")
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("Finalized segment frames must be [F,3,H,W]")
    if frames.shape[0] < 1:
        raise ValueError("Cannot write an empty output segment")
    if not bool(torch.isfinite(frames).all().item()):
        raise FloatingPointError("Finalized segment contains NaN or Inf")
    if frames.min().item() < 0.0 or frames.max().item() > 1.0:
        raise ValueError("Finalized segment must remain in the decoded [0,1] range")

    frame_count, _channels, height, width = map(int, frames.shape)
    shape = [frame_count, height, width, 3]
    temporary = path.with_name(f".{path.name}.partial")
    mapped: np.memmap | None = None
    pixel_digest = hashlib.sha256()
    try:
        mapped = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.uint8,
            shape=tuple(shape),
        )
        for index, frame in enumerate(frames):
            # This is the same arithmetic and cast as upstream save_video(), but
            # bounded to one frame instead of allocating temporaries for all N.
            frame_array = frame.mul(255).permute(1, 2, 0).to(torch.uint8).numpy()
            mapped[index] = frame_array
            pixel_digest.update(mapped[index].tobytes(order="C"))
        mapped.flush()
        del mapped
        mapped = None
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if mapped is not None:
            del mapped
        temporary.unlink(missing_ok=True)
        raise
    return shape, path.stat().st_size, pixel_digest.hexdigest()


def _write_bootstrap_video_segment(
    path: Path,
    video_path: Path,
    *,
    expected_frames: int,
    prefix_frames: int,
    expected_fps: float,
    fps_tolerance: float,
    output_height: int,
    output_width: int,
) -> tuple[list[int], int, str, torch.Tensor]:
    """Stream one existing first-chunk MP4 into the native segment format.

    Only the recurrent FP32 prefix is retained in RAM. The full bootstrap video
    is written directly to the same uint8 FHWC ``.npy`` format used by normal
    generated chunks, so later resume and final assembly need no special path.
    """
    if expected_frames < 1:
        raise ValueError("Bootstrap frame count must be positive")
    if prefix_frames < 1 or prefix_frames > expected_frames:
        raise ValueError("Bootstrap prefix_frames must be within the first chunk")
    if output_height < 1 or output_width < 1:
        raise ValueError("Bootstrap output dimensions must be positive")

    shape = [expected_frames, output_height, output_width, 3]
    temporary = path.with_name(f".{path.name}.partial")
    mapped: np.memmap | None = None
    pixel_digest = hashlib.sha256()
    next_prefix = torch.empty(
        (prefix_frames, 3, output_height, output_width), dtype=torch.float32
    )
    prefix_start = expected_frames - prefix_frames
    decoded_frames = 0
    try:
        mapped = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.uint8,
            shape=tuple(shape),
        )
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                raise ValueError(f"Bootstrap video has no video stream: {video_path}")
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.base_rate or expected_fps)
            if abs(fps - expected_fps) > fps_tolerance:
                raise ValueError(
                    f"Bootstrap FPS {fps:g} does not match configured FPS "
                    f"{expected_fps:g}"
                )
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index >= expected_frames:
                    raise ValueError(
                        f"Bootstrap video must contain exactly {expected_frames} frames"
                    )
                frame_array = frame.to_ndarray(format="rgb24")
                expected_shape = (output_height, output_width, 3)
                if frame_array.shape != expected_shape:
                    raise ValueError(
                        "Bootstrap frame resolution does not match the configured "
                        f"canvas: {frame_array.shape} != {expected_shape}"
                    )
                mapped[frame_index] = frame_array
                pixel_digest.update(frame_array.tobytes(order="C"))
                if frame_index >= prefix_start:
                    prefix_frame = next_prefix[frame_index - prefix_start]
                    prefix_frame.copy_(torch.from_numpy(frame_array).permute(2, 0, 1))
                    prefix_frame.div_(255.0)
                decoded_frames += 1

        if decoded_frames != expected_frames:
            raise ValueError(
                f"Bootstrap video has {decoded_frames} frames; expected {expected_frames}"
            )
        mapped.flush()
        del mapped
        mapped = None
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if mapped is not None:
            del mapped
        temporary.unlink(missing_ok=True)
        raise
    return shape, path.stat().st_size, pixel_digest.hexdigest(), next_prefix


def _write_prefix_state(
    path: Path,
    chunk_index: int,
    prefix: torch.Tensor,
    *,
    cpu_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor | None,
    cuda_device: str | None,
) -> tuple[int, str]:
    if prefix.device.type != "cpu" or prefix.dtype != torch.float32:
        raise TypeError("Previous-prefix state must be CPU FP32")
    if prefix.ndim != 4 or prefix.shape[1] != 3:
        raise ValueError("Previous-prefix state must be [F,3,H,W]")
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("wb") as handle:
            torch.save(
                {
                    "chunk_index": chunk_index,
                    "prefix": prefix,
                    "cpu_rng_state": cpu_rng_state,
                    "cuda_rng_state": cuda_rng_state,
                    "cuda_device": cuda_device,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        state_digest = file_sha256(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path.stat().st_size, state_digest


class SegmentedOutputStore:
    """Atomic chunk journal and lossless encoder-input segment store."""

    _SCHEMA_VERSION = 1

    def __init__(self, root: Path) -> None:
        self.root = root
        self.session_path = root / "run.json"
        self.model_ready_path = root / "model_ready.json"

    def prepare(self, identity: Mapping[str, Any], *, resume: bool) -> None:
        expected = {"schema_version": self._SCHEMA_VERSION, **dict(identity)}
        if resume:
            if not self.session_path.is_file():
                raise FileNotFoundError(
                    f"Resume metadata does not exist: {self.session_path}"
                )
            actual = _load_json(self.session_path)
            if actual != expected:
                raise RuntimeError("Resume metadata does not match this inference request")
            return

        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise FileExistsError(
                f"Segment directory is not empty: {self.root}. "
                "Use --resume or choose a different --segment-dir."
            )
        _atomic_json(self.session_path, expected)

    def record_model_ready(self, adapter_report: Mapping[str, Any]) -> None:
        value = {"adapter_load": dict(adapter_report)}
        if self.model_ready_path.is_file():
            if _load_json(self.model_ready_path) != value:
                raise RuntimeError("Resume model report does not match the committed run")
            return
        _atomic_json(self.model_ready_path, value)

    def load_model_ready(self) -> dict[str, Any]:
        if not self.model_ready_path.is_file():
            raise RuntimeError("Completed segments have no committed model report")
        value = _load_json(self.model_ready_path)
        report = value.get("adapter_load")
        if not isinstance(report, dict):
            raise TypeError("Committed model report is invalid")
        return report

    def _record_path(self, chunk_index: int) -> Path:
        return self.root / f"chunk_{chunk_index:06d}.json"

    def _segment_path(self, chunk_index: int) -> Path:
        return self.root / f"segment_{chunk_index:06d}.npy"

    def _prefix_path(self, chunk_index: int) -> Path:
        # Alternating slots guarantee that an uncommitted chunk cannot overwrite
        # the prefix referenced by the latest committed chunk record.
        return self.root / f"previous_prefix_{chunk_index % 2}.pt"

    def commit_chunk(
        self,
        record: Mapping[str, Any],
        emitted_frames: torch.Tensor,
        next_prefix: torch.Tensor,
        *,
        chunk_seed: int,
        cpu_rng_state: torch.Tensor,
        cuda_rng_state: torch.Tensor | None,
        cuda_device: str | None,
    ) -> dict[str, Any]:
        chunk_index = int(record["chunk_index"])
        segment_path = self._segment_path(chunk_index)
        segment_shape, segment_bytes, segment_sha256 = _write_uint8_segment(
            segment_path, emitted_frames
        )
        return self._commit_prefix_and_record(
            record,
            next_prefix,
            chunk_seed=chunk_seed,
            cpu_rng_state=cpu_rng_state,
            cuda_rng_state=cuda_rng_state,
            cuda_device=cuda_device,
            segment_path=segment_path,
            segment_shape=segment_shape,
            segment_bytes=segment_bytes,
            segment_sha256=segment_sha256,
        )

    def commit_bootstrap_video(
        self,
        record: Mapping[str, Any],
        bootstrap_video: Path,
        *,
        prefix_frames: int,
        expected_fps: float,
        fps_tolerance: float,
        output_height: int,
        output_width: int,
        chunk_seed: int,
        cpu_rng_state: torch.Tensor,
        cuda_rng_state: torch.Tensor | None,
        cuda_device: str | None,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Commit an existing first-chunk MP4 as completed chunk zero."""
        chunk_index = int(record["chunk_index"])
        if chunk_index != 0:
            raise ValueError("A bootstrap video can only commit chunk zero")
        expected_frames = int(record["output_appended_count"])
        segment_path = self._segment_path(chunk_index)
        segment_shape, segment_bytes, segment_sha256, next_prefix = (
            _write_bootstrap_video_segment(
                segment_path,
                bootstrap_video,
                expected_frames=expected_frames,
                prefix_frames=prefix_frames,
                expected_fps=expected_fps,
                fps_tolerance=fps_tolerance,
                output_height=output_height,
                output_width=output_width,
            )
        )
        committed = self._commit_prefix_and_record(
            record,
            next_prefix,
            chunk_seed=chunk_seed,
            cpu_rng_state=cpu_rng_state,
            cuda_rng_state=cuda_rng_state,
            cuda_device=cuda_device,
            segment_path=segment_path,
            segment_shape=segment_shape,
            segment_bytes=segment_bytes,
            segment_sha256=segment_sha256,
        )
        return committed, next_prefix

    def _commit_prefix_and_record(
        self,
        record: Mapping[str, Any],
        next_prefix: torch.Tensor,
        *,
        chunk_seed: int,
        cpu_rng_state: torch.Tensor,
        cuda_rng_state: torch.Tensor | None,
        cuda_device: str | None,
        segment_path: Path,
        segment_shape: list[int],
        segment_bytes: int,
        segment_sha256: str,
    ) -> dict[str, Any]:
        chunk_index = int(record["chunk_index"])
        prefix_path = self._prefix_path(chunk_index)
        prefix_bytes, prefix_state_sha256 = _write_prefix_state(
            prefix_path,
            chunk_index,
            next_prefix,
            cpu_rng_state=cpu_rng_state,
            cuda_rng_state=cuda_rng_state,
            cuda_device=cuda_device,
        )

        committed = dict(record)
        committed.update(
            {
                "chunk_seed": chunk_seed,
                "segment_file": segment_path.name,
                "segment_shape_fhwc": segment_shape,
                "segment_dtype": "uint8",
                "segment_byte_size": segment_bytes,
                "segment_pixel_sha256": segment_sha256,
                "next_prefix_file": prefix_path.name,
                "next_prefix_shape_fchw": list(next_prefix.shape),
                "next_prefix_dtype": str(next_prefix.dtype).removeprefix("torch."),
                "next_prefix_byte_size": prefix_bytes,
                "next_prefix_state_sha256": prefix_state_sha256,
            }
        )
        # The record is the completion marker and is always committed last.
        _atomic_json(self._record_path(chunk_index), committed)
        return committed

    def load_record(
        self,
        expected: Mapping[str, Any],
        *,
        output_height: int,
        output_width: int,
        chunk_seed: int,
    ) -> dict[str, Any] | None:
        chunk_index = int(expected["chunk_index"])
        record_path = self._record_path(chunk_index)
        if not record_path.is_file():
            return None
        record = _load_json(record_path)
        for key in (
            "chunk_index",
            "absolute_start_frame",
            "anchor_source_frame_range",
            "prefix_output_frame_range",
            "new_output_frame_range",
            "output_appended_count",
            "real_input_frame_count",
            "working_frame_count",
            "repeat_last_padding_count",
            "final_partial_repeat_last",
        ):
            if record.get(key) != expected.get(key):
                raise RuntimeError(f"Resume chunk {chunk_index} disagrees on {key}")
        if record.get("chunk_seed") != chunk_seed:
            raise RuntimeError(f"Resume chunk {chunk_index} seed does not match")

        segment_path = self.root / str(record.get("segment_file", ""))
        if not segment_path.is_file():
            raise RuntimeError(f"Completed chunk {chunk_index} has no segment file")
        if segment_path.stat().st_size != record.get("segment_byte_size"):
            raise RuntimeError(f"Completed chunk {chunk_index} segment is truncated")
        segment = np.load(segment_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (
            int(expected["output_appended_count"]),
            output_height,
            output_width,
            3,
        )
        if segment.dtype != np.uint8 or segment.shape != expected_shape:
            raise RuntimeError(f"Completed chunk {chunk_index} segment is incompatible")
        pixel_digest = hashlib.sha256()
        for frame in segment:
            pixel_digest.update(frame.tobytes(order="C"))
        if pixel_digest.hexdigest() != record.get("segment_pixel_sha256"):
            raise RuntimeError(f"Completed chunk {chunk_index} segment is corrupt")
        del segment
        return record

    def scan_completed(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        output_height: int,
        output_width: int,
        base_seed: int,
    ) -> tuple[int, int]:
        completed = 0
        total = 0
        found_gap = False
        for expected in records:
            total += 1
            chunk_seed = base_seed + int(expected["absolute_start_frame"])
            record = self.load_record(
                expected,
                output_height=output_height,
                output_width=output_width,
                chunk_seed=chunk_seed,
            )
            if record is None:
                found_gap = True
            elif found_gap:
                raise RuntimeError("Resume chunk records are not a contiguous prefix")
            else:
                completed += 1
        return completed, total

    def load_resume_state(
        self,
        chunk_index: int,
        *,
        prefix_frames: int,
        output_height: int,
        output_width: int,
    ) -> ResumeState:
        record = _load_json(self._record_path(chunk_index))
        prefix_path = self.root / str(record.get("next_prefix_file", ""))
        if not prefix_path.is_file():
            raise RuntimeError(f"Completed chunk {chunk_index} has no FP32 prefix state")
        if prefix_path.stat().st_size != record.get("next_prefix_byte_size"):
            raise RuntimeError(f"Completed chunk {chunk_index} prefix state is truncated")
        if file_sha256(prefix_path) != record.get("next_prefix_state_sha256"):
            raise RuntimeError("FP32 prefix/RNG state digest does not match its chunk record")
        state = torch.load(prefix_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("chunk_index") != chunk_index:
            raise RuntimeError("FP32 prefix state does not match the latest completed chunk")
        prefix = state.get("prefix")
        cpu_rng_state = state.get("cpu_rng_state")
        cuda_rng_state = state.get("cuda_rng_state")
        cuda_device = state.get("cuda_device")
        expected_shape = (prefix_frames, 3, output_height, output_width)
        if (
            not isinstance(prefix, torch.Tensor)
            or prefix.device.type != "cpu"
            or prefix.dtype != torch.float32
            or tuple(prefix.shape) != expected_shape
        ):
            raise RuntimeError("FP32 prefix state has incompatible shape or dtype")
        if (
            not isinstance(cpu_rng_state, torch.Tensor)
            or cpu_rng_state.device.type != "cpu"
            or cpu_rng_state.dtype != torch.uint8
            or cpu_rng_state.ndim != 1
        ):
            raise RuntimeError("CPU RNG state is missing or incompatible")
        if cuda_rng_state is not None and (
            not isinstance(cuda_rng_state, torch.Tensor)
            or cuda_rng_state.device.type != "cpu"
            or cuda_rng_state.dtype != torch.uint8
            or cuda_rng_state.ndim != 1
            or not isinstance(cuda_device, str)
        ):
            raise RuntimeError("CUDA RNG state is incompatible")
        if cuda_rng_state is None and cuda_device is not None:
            raise RuntimeError("CUDA RNG device exists without a saved RNG state")
        return ResumeState(
            previous_prefix=prefix,
            cpu_rng_state=cpu_rng_state,
            cuda_rng_state=cuda_rng_state,
            cuda_device=cuda_device,
        )

    def iter_records(self, chunk_count: int) -> Iterator[dict[str, Any]]:
        for chunk_index in range(chunk_count):
            path = self._record_path(chunk_index)
            if not path.is_file():
                raise RuntimeError(f"Missing completed chunk record: {path}")
            yield _load_json(path)

    def cleanup(self) -> None:
        """Remove only files owned by this run; never recursively delete a user path."""
        owned = [self.session_path, self.model_ready_path]
        owned.extend(self.root.glob("chunk_*.json"))
        owned.extend(self.root.glob("segment_*.npy"))
        owned.extend(self.root.glob("previous_prefix_*.pt"))
        owned.extend(self.root.glob(".*.partial"))
        for path in owned:
            path.unlink(missing_ok=True)
        try:
            self.root.rmdir()
        except OSError:
            pass


def iter_segment_frames(
    records: Iterable[Mapping[str, Any]], root: Path
) -> Iterator[np.ndarray]:
    """Yield finalized RGB24 frames in committed chunk order."""
    for record in records:
        segment_path = root / str(record["segment_file"])
        segment = np.load(segment_path, mmap_mode="r", allow_pickle=False)
        yield from segment
        del segment


def encode_segments(
    records: Iterable[Mapping[str, Any]],
    segment_root: Path,
    output_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    expected_frames: int,
) -> int:
    """Encode committed RGB24 segments with the legacy PyAV/x264 settings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    encoded_frames = 0
    try:
        with av.open(str(temporary), mode="w") as container:
            video_stream = container.add_stream("libx264", rate=int(fps))
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"crf": "18"}
            for frame_array in iter_segment_frames(records, segment_root):
                frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in video_stream.encode(frame):
                    container.mux(packet)
                encoded_frames += 1
            for packet in video_stream.encode():
                container.mux(packet)
        if encoded_frames != expected_frames:
            raise RuntimeError(
                f"Segment assembly produced {encoded_frames} frames, expected {expected_frames}"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        _fsync_directory(output_path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return encoded_frames


def session_fingerprint(value: Mapping[str, Any]) -> str:
    """Stable digest used to bind resume artifacts to generation settings."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_record_manifest(
    path: Path,
    header: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Atomically stream a JSON object with a ``chunks`` array from disk records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("{\n")
            items = list(header.items())
            for key, value in items:
                handle.write(f"  {json.dumps(str(key))}: ")
                json.dump(value, handle, sort_keys=True)
                handle.write(",\n")
            handle.write('  "chunks": [\n')
            first = True
            for record in records:
                if not first:
                    handle.write(",\n")
                handle.write("    ")
                json.dump(record, handle, sort_keys=True)
                first = False
            handle.write("\n  ]\n}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
