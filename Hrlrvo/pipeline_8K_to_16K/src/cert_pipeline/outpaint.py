"""Certification wrapper around the packaged production LongRange outpainter.

This module deliberately does not implement temporal generation.  It loads the
packaged inference YAML and calls :func:`ltx_outpaint.inference.sample_long_range`
with the recorded C1+ recurrence contract. The wrapper supplies package-relative paths, persists audit
artifacts at the native chunk commit boundary, and retimes the already emitted
frame sequence from its 24 fps processing clock to a 16 fps storage clock.

Importing this module performs no model loading and launches no subprocesses.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any


class OutpaintError(RuntimeError):
    """Raised when the outpainting stage or one of its audit artifacts fails."""


PROCESSING_FPS = Fraction(24, 1)
STORED_FPS = Fraction(16, 1)
OUTPUT_SIZE = (3840, 2160)
FINAL_RETIME_FILTER = "settb=expr=1/16,setpts=N"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutpaintError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise OutpaintError(f"Expected a JSON object: {path}")
    return value


def _fraction(value: Any, field: str) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise OutpaintError(f"Invalid {field}: {value!r}") from exc
    if result <= 0:
        raise OutpaintError(f"Invalid {field}: {value!r}")
    return result


def _video_stream(probe: Mapping[str, Any]) -> Mapping[str, Any]:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise OutpaintError("ffprobe document has no streams array")
    videos = [item for item in streams if item.get("codec_type") == "video"]
    if len(videos) != 1:
        raise OutpaintError(f"Expected one video stream, found {len(videos)}")
    return videos[0]


def _frame_count(stream: Mapping[str, Any]) -> int:
    for field in ("nb_read_frames", "nb_frames"):
        value = stream.get(field)
        if value not in (None, "", "N/A"):
            try:
                count = int(str(value))
            except ValueError as exc:
                raise OutpaintError(f"Invalid {field}: {value!r}") from exc
            if count > 0:
                return count
    raise OutpaintError("ffprobe did not report a positive decoded frame count")


def _probe(path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(completed.stdout)
    except subprocess.CalledProcessError as exc:
        raise OutpaintError(
            f"ffprobe failed for {path}: {(exc.stderr or '').strip()}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise OutpaintError(f"ffprobe returned invalid JSON for {path}") from exc
    if not isinstance(value, dict):
        raise OutpaintError(f"ffprobe returned an invalid document for {path}")
    return value


def _validate_video(
    probe: Mapping[str, Any],
    *,
    path: Path,
    expected_frames: int,
    expected_fps: Fraction,
) -> dict[str, Any]:
    video = _video_stream(probe)
    size = (int(video.get("width", 0)), int(video.get("height", 0)))
    if size != OUTPUT_SIZE:
        raise OutpaintError(
            f"{path} is {size[0]}x{size[1]}, expected {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}"
        )
    fps = _fraction(video.get("avg_frame_rate"), "avg_frame_rate")
    if fps != expected_fps:
        raise OutpaintError(f"{path} is {fps} fps, expected {expected_fps} fps")
    rate = _fraction(video.get("r_frame_rate"), "r_frame_rate")
    if rate != expected_fps:
        raise OutpaintError(f"{path} r_frame_rate is {rate}, expected {expected_fps}")
    frames = _frame_count(video)
    if frames != expected_frames:
        raise OutpaintError(
            f"{path} contains {frames} frames, expected {expected_frames}"
        )
    return {
        "path": str(path),
        "width": size[0],
        "height": size[1],
        "fps": str(fps),
        "r_frame_rate": str(rate),
        "frames": frames,
        "codec_name": video.get("codec_name"),
        "pixel_format": video.get("pix_fmt"),
    }


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> str:
    """Copy an immutable native state before its alternating slot is reused."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_stream, temporary.open("wb") as target:
            while block := source_stream.read(4 * 1024 * 1024):
                target.write(block)
                digest.update(block)
            target.flush()
            os.fsync(target.fileno())
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise OutpaintError(
                f"Native prefix state digest changed before persistence: {source}"
            )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return actual
    finally:
        temporary.unlink(missing_ok=True)


def _mirror_pipe(pipe: Any) -> None:
    for raw_line in iter(pipe.readline, b""):
        print(raw_line.decode("utf-8", errors="replace"), end="", flush=True)
    pipe.close()


def _run_ffmpeg(command: list[str], log_path: Path) -> None:
    # The top-level runner installs a stdout/stderr Tee.  Mirror child output
    # through Python stdout instead of opening the same log a second time.
    print(
        "[OUTPAINT_FFMPEG] command=" + " ".join(map(str, command)),
        flush=True,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    reader = threading.Thread(target=_mirror_pipe, args=(process.stdout,), daemon=True)
    reader.start()
    return_code = process.wait()
    reader.join()
    if return_code:
        raise OutpaintError(f"ffmpeg exited with status {return_code}; see {log_path}")


def _encode_segment_mp4(
    segment_path: Path,
    output_path: Path,
    *,
    expected_frames: int,
    log_path: Path,
) -> dict[str, Any]:
    # NumPy is already an inference dependency.  Import it only on execution so
    # a static import of this orchestration module remains lightweight.
    import numpy as np

    segment = np.load(segment_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (expected_frames, OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3)
    if segment.dtype != np.uint8 or tuple(segment.shape) != expected_shape:
        raise OutpaintError(
            f"Native segment {segment_path} has {segment.dtype} {segment.shape}, "
            f"expected uint8 {expected_shape}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.partial.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-xerror",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}",
        "-framerate",
        str(PROCESSING_FPS.numerator),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx265",
        "-crf",
        "1",
        "-pix_fmt",
        "yuv420p10le",
        "-frames:v",
        str(expected_frames),
        "-y",
        str(temporary),
    ]
    try:
        print(
            "[OUTPAINT_CHUNK_FFMPEG] command=" + " ".join(map(str, command)),
            flush=True,
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdin is not None and process.stdout is not None
        reader = threading.Thread(
            target=_mirror_pipe, args=(process.stdout,), daemon=True
        )
        reader.start()
        try:
            for frame in segment:
                process.stdin.write(frame.tobytes(order="C"))
            process.stdin.close()
            return_code = process.wait()
            reader.join()
        except BaseException:
            process.kill()
            process.wait()
            reader.join()
            raise
        if return_code:
            raise OutpaintError(
                f"Chunk ffmpeg exited with status {return_code}; see {log_path}"
            )
        verified = _validate_video(
            _probe(temporary),
            path=temporary,
            expected_frames=expected_frames,
            expected_fps=PROCESSING_FPS,
        )
        os.replace(temporary, output_path)
        _fsync_directory(output_path.parent)
        verified["path"] = str(output_path)
        return verified
    finally:
        del segment
        temporary.unlink(missing_ok=True)


def _chunk_mp4_path(mp4_dir: Path, record: Mapping[str, Any]) -> Path:
    begin, end = map(int, record["new_output_frame_range"])
    if end <= begin:
        raise OutpaintError(f"Invalid native chunk range: {[begin, end]}")
    return mp4_dir / (
        f"chunk_{int(record['chunk_index']) + 1:04d}_"
        f"frames_{begin:06d}_{end - 1:06d}.mp4"
    )


def _persist_committed_chunk(
    record: Mapping[str, Any],
    segment_path: Path,
    *,
    native_segment_dir: Path,
    mp4_dir: Path,
    latent_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Persist audit artifacts after the native completion marker is committed."""

    chunk_index = int(record["chunk_index"])
    artifact_dir = latent_dir / f"chunk_{chunk_index:06d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    generated_latent = artifact_dir / "generated_latent.pt"
    generated_metadata = artifact_dir / "metadata.json"
    if not generated_latent.is_file() or not generated_metadata.is_file():
        raise OutpaintError(
            f"Chunk {chunk_index} committed without its generated latent artifact"
        )
    latent_metadata = _load_json(generated_metadata)
    if _sha256(generated_latent) != latent_metadata.get("sha256"):
        raise OutpaintError(f"Generated latent digest failed for chunk {chunk_index}")
    # The native latent writer commits the tensor atomically.  Rewrite its
    # small metadata document atomically at the first safe post-commit hook.
    _atomic_json(generated_metadata, latent_metadata)

    carry_path = artifact_dir / "carry_prefix_state.pt"
    expected_prefix_sha = str(record["next_prefix_state_sha256"])
    if carry_path.is_file():
        if _sha256(carry_path) != expected_prefix_sha:
            raise OutpaintError(f"Carry state digest failed for chunk {chunk_index}")
    else:
        native_prefix = native_segment_dir / str(record["next_prefix_file"])
        if not native_prefix.is_file():
            raise OutpaintError(
                f"Native carry state is unavailable for chunk {chunk_index}: {native_prefix}"
            )
        _copy_verified(native_prefix, carry_path, expected_prefix_sha)

    causal_artifact = None
    if record.get("recurrence", {}).get("use_latent_carry"):
        item = record.get("causal_latent_carry", {})
        expected_name = f"causal_latent_carry_{chunk_index:06d}.pt"
        if item.get("file") != expected_name:
            raise OutpaintError(f"Missing causal latent carry for chunk {chunk_index}")
        native_carry = native_segment_dir / expected_name
        if not native_carry.is_file() or _sha256(native_carry) != item.get("sha256"):
            raise OutpaintError(f"Corrupt native causal carry: {native_carry}")
        causal_path = artifact_dir / "causal_latent_carry.pt"
        if causal_path.exists():
            if _sha256(causal_path) != item["sha256"]:
                raise OutpaintError(f"Corrupt archived causal carry: {causal_path}")
        else:
            _copy_verified(native_carry, causal_path, item["sha256"])
        causal_artifact = dict(item, path=str(causal_path))

    begin, end = map(int, record["new_output_frame_range"])
    expected_frames = end - begin
    mp4_path = _chunk_mp4_path(mp4_dir, record)
    if mp4_path.is_file():
        mp4_metadata = _validate_video(
            _probe(mp4_path),
            path=mp4_path,
            expected_frames=expected_frames,
            expected_fps=PROCESSING_FPS,
        )
    else:
        mp4_metadata = _encode_segment_mp4(
            segment_path,
            mp4_path,
            expected_frames=expected_frames,
            log_path=log_path,
        )

    artifact_metadata = {
        "status": "PASS",
        "chunk_index": chunk_index,
        "absolute_start_frame": int(record["absolute_start_frame"]),
        "new_output_frame_range_exclusive": [begin, end],
        "native_record": str(native_segment_dir / f"chunk_{chunk_index:06d}.json"),
        "native_lossless_segment": str(segment_path),
        "causal_latent_carry": causal_artifact,
        "chunk_mp4": mp4_metadata,
        "generated_latent": {
            "path": str(generated_latent),
            "sha256": latent_metadata["sha256"],
            "shape": latent_metadata.get("latent_shape"),
            "dtype": latent_metadata.get("latent_dtype"),
        },
        "carry_prefix_state": {
            "path": str(carry_path),
            "sha256": expected_prefix_sha,
            "shape": record.get("next_prefix_shape_fchw"),
            "dtype": record.get("next_prefix_dtype"),
            "contains_native_rng_state": True,
        },
    }
    _atomic_json(artifact_dir / "certification_artifacts.json", artifact_metadata)
    return artifact_metadata


def _load_native_records(native_segment_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(native_segment_dir.glob("chunk_*.json"))
    records = [_load_json(path) for path in paths]
    for expected_index, record in enumerate(records):
        if int(record.get("chunk_index", -1)) != expected_index:
            raise OutpaintError("Native LongRange chunk records are not contiguous")
    return records


def _write_certification_manifests(
    records: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    *,
    manifests_dir: Path,
    frame_count: int,
) -> None:
    if len(records) != len(artifacts):
        raise OutpaintError("Chunk records and certification artifacts differ in count")
    next_frame = 0
    plan: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for record, artifact in zip(records, artifacts, strict=True):
        begin, end = map(int, record["new_output_frame_range"])
        if begin != next_frame or end <= begin:
            raise OutpaintError(
                f"Native LongRange output lineage is not contiguous at chunk "
                f"{record['chunk_index']}"
            )
        next_frame = end
        plan.append(
            {
                key: record.get(key)
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
                )
            }
        )
        results.append({"native_record": record, "certification_artifacts": artifact})
    if next_frame != frame_count:
        raise OutpaintError(
            f"Native LongRange lineage ends at {next_frame}, expected {frame_count}"
        )
    _atomic_json(
        manifests_dir / "chunk_plan.json",
        {
            "source": "native LongRange committed records",
            "requested_frames": frame_count,
            "chunk_count": len(plan),
            "chunks": plan,
        },
    )
    _atomic_jsonl(manifests_dir / "chunk_results.jsonl", results)


def _decoded_rgb_hashes(path: Path, frame_count: int) -> list[str]:
    """Sequential CPU decode; timing is deliberately not part of pixel identity."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-hwaccel", "none", "-i", str(path),
         "-map", "0:v:0", "-an", "-pix_fmt", "rgb24", "-fps_mode:v", "passthrough",
         "-f", "framemd5", "-"],
        check=True, capture_output=True, text=True,
    )
    hashes = [line.split(",")[-1].strip() for line in result.stdout.splitlines()
              if line.strip() and not line.startswith("#")]
    if len(hashes) != frame_count:
        raise OutpaintError(f"RGB identity decode returned {len(hashes)} frames, expected {frame_count}")
    return hashes


def _retime_to_16fps(
    processing_path: Path,
    final_path: Path,
    *,
    frame_count: int,
    log_path: Path,
    resume: bool,
) -> dict[str, Any]:
    """Final postprocess ONLY: one complete merged master, no recurrent state.

    Historical fps_4K.sh uses encoder timebase1:16 and lossless libx264 CRF0.
    Lossy HEVC CRF1 cannot satisfy exact decoded RGB-sequence identity.
    """
    source_metadata = _validate_video(
        _probe(processing_path), path=processing_path,
        expected_frames=frame_count, expected_fps=PROCESSING_FPS,
    )
    if source_metadata["pixel_format"] != "yuv420p":
        raise OutpaintError("Historical lossless retime requires the native yuv420p merged master")
    source_hashes = _decoded_rgb_hashes(processing_path, frame_count)

    def verify_retime(path: Path) -> dict[str, Any]:
        verified = _validate_video(
            _probe(path), path=path, expected_frames=frame_count, expected_fps=STORED_FPS,
        )
        output_hashes = _decoded_rgb_hashes(path, frame_count)
        if source_hashes != output_hashes:
            mismatch = next(index for index, pair in enumerate(zip(source_hashes, output_hashes, strict=True))
                            if pair[0] != pair[1])
            raise OutpaintError(f"Final retime changed decoded RGB at frame {mismatch}")
        return verified

    if resume and final_path.is_file():
        verified = verify_retime(final_path)
    else:
        verified = None

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(
        f".{final_path.stem}.{os.getpid()}.partial{final_path.suffix}"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-xerror",
        "-i",
        str(processing_path),
        "-map",
        "0:v:0",
        "-vf",
        FINAL_RETIME_FILTER,
        "-fps_mode:v",
        "passthrough",
        "-enc_time_base:v",
        "1:16",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-frames:v",
        str(frame_count),
        "-y",
        str(temporary),
    ]
    try:
        if verified is None:
            _run_ffmpeg(command, log_path)
            verified = verify_retime(temporary)
            os.replace(temporary, final_path)
            _fsync_directory(final_path.parent)
        verified["path"] = str(final_path)
        _atomic_json(final_path.with_suffix(".retime.json"), {
            "status": "PASS", "phase": "postprocess_after_all_chunks_and_merge",
            "native_24fps_master": source_metadata,
            "final_16fps_deliverable": verified,
            "native_master_sha256": _sha256(processing_path),
            "deliverable_sha256": _sha256(final_path),
            "frame_count": frame_count, "frame_order_unchanged": True,
            "decoded_rgb_sequence_exact": True, "decoded_rgb_framemd5": source_hashes,
            "fps_resampling": False, "retime_filter": FINAL_RETIME_FILTER,
            "encoder_time_base": "1:16", "codec": "libx264", "crf": 0,
            "reference": "historical config/01..05/<ID>_fps_4K.sh",
        })
        return verified
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_settings(root: Path, runtime_config: Path) -> dict[str, Any]:
    # PyYAML is part of the reference runtime.  Delaying the import keeps the
    # certification orchestrator importable for preflight without model deps.
    import yaml

    try:
        settings = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OutpaintError(
            f"Cannot load outpaint runtime config: {runtime_config}"
        ) from exc
    if not isinstance(settings, dict):
        raise OutpaintError("Outpaint runtime config must contain a mapping")
    settings["ltx_repo"] = str((root / str(settings["ltx_repo"])).resolve())
    model = settings.get("model")
    if not isinstance(model, dict):
        raise OutpaintError("Outpaint runtime config has no model mapping")
    for key in ("base_model_path", "text_encoder_path", "adapter_path"):
        model[key] = str((root / str(model[key])).resolve())
    if model.get("video_vae_path"):
        model["video_vae_path"] = str((root / str(model["video_vae_path"])).resolve())
    return settings


def run_outpaint(
    root: Path,
    config: dict[str, Any],
    preprocess_path: Path,
    stage_dir: Path,
    frame_count: int,
    resume: bool,
) -> Path:
    """Run and verify the packaged production outpainting stage.

    The caller must already have completed config/input preflight, established
    global RNG seeds, and arranged terminal/file log teeing.  ``frame_count`` is
    the resolved outpaint limit, never a hard-coded source length.
    """

    root = root.resolve()
    preprocess_path = preprocess_path.resolve()
    stage_dir = stage_dir.resolve()
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise OutpaintError("frame_count must be a positive integer")
    if not preprocess_path.is_file():
        raise FileNotFoundError(
            f"Preprocessed anchor video does not exist: {preprocess_path}"
        )
    if not isinstance(resume, bool):
        raise TypeError("resume must be a boolean")

    outpaint = config.get("outpainting")
    input_config = config.get("input")
    if not isinstance(outpaint, dict) or not isinstance(input_config, dict):
        raise OutpaintError("Config requires input and outpainting mappings")
    video_id = str(input_config["video_id"])
    runtime_config = (root / str(outpaint["runtime_config"])).resolve()
    checkpoint = (root / str(outpaint["checkpoint"])).resolve()
    adapter_file = checkpoint / "adapter_model.safetensors"
    if not runtime_config.is_file():
        raise FileNotFoundError(runtime_config)
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    if int(outpaint["steps"]) != 30:
        raise OutpaintError("Certification outpainting must use exactly 30 steps")
    if tuple(map(int, outpaint["input_resolution"])) != (1920, 1080):
        raise OutpaintError("Outpaint input_resolution must be 1920x1080")
    if tuple(map(int, outpaint["output_resolution"])) != OUTPUT_SIZE:
        raise OutpaintError("Outpaint output_resolution must be 3840x2160")
    if int(outpaint["alpha_feather_px"]) != 16:
        raise OutpaintError("Outpaint alpha_feather_px must be 16")
    if _fraction(outpaint["processing_fps"], "processing_fps") != PROCESSING_FPS:
        raise OutpaintError("Outpaint processing_fps must be 24")
    if _fraction(outpaint["stored_fps"], "stored_fps") != STORED_FPS:
        raise OutpaintError("Outpaint stored_fps must be 16")

    chunks_dir = stage_dir / "chunks"
    native_segment_dir = chunks_dir / "native"
    mp4_dir = chunks_dir / "rgb"
    latent_dir = chunks_dir / "latents"
    manifests_dir = chunks_dir / "manifests"
    metadata_dir = stage_dir / "metadata"
    log_path = stage_dir / "logs" / "outpaint.log"
    final_path = stage_dir / "final" / f"{video_id}_Outpaint_4K.mp4"
    processing_path = native_segment_dir / f"{video_id}_Outpaint_4K_24fps.mp4"
    for directory in (
        native_segment_dir,
        mp4_dir,
        latent_dir,
        manifests_dir,
        metadata_dir,
        log_path.parent,
        final_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    settings = _resolve_settings(root, runtime_config)
    settings["sampling"]["inference_steps"] = 30
    settings["sampling"]["seed"] = int(config["seed"])
    settings["video"]["fps"] = 24.0
    settings["video"]["canvas"] = [3840, 2160]
    settings["video"]["expected_anchor"] = [1920, 1080]
    settings["video"]["boundary_blend_width_pixels"] = 16
    memory = outpaint.get("memory_optimization")
    if not isinstance(memory, dict):
        raise OutpaintError("outpainting.memory_optimization must be a mapping")
    settings["memory_optimization"] = dict(memory)
    from cert_pipeline.recurrence import validate_recurrence

    settings["recurrence"] = validate_recurrence(outpaint["recurrence"])
    runtime_src = root / "src/video_outpainting/main/src/ltx_outpaint"
    settings["recurrence_implementation"] = {
        name: _sha256(runtime_src / name) for name in (
            "inference.py", "global_attention.py", "lineart_global_prior.py",
            "lineart_reference.py", "recurrence_artifacts.py",
        )
    }
    print("[OUTPAINT_RECURRENCE] " + json.dumps(settings["recurrence"], sort_keys=True), flush=True)

    # A global resume may start after preprocessing but before LongRange has
    # created its own session.  Native resume is valid only once run.json has
    # been atomically committed; do not pass resume=True to an empty store.
    native_session = native_segment_dir / "run.json"
    native_resume = resume and native_session.is_file()
    if resume and not native_resume and any(native_segment_dir.iterdir()):
        raise OutpaintError(
            f"Native LongRange has artifacts but no session manifest: {native_segment_dir}"
        )

    # Reconcile every already committed native record before model loading.  A
    # crash can occur after the native record is atomically committed but before
    # this wrapper finishes its MP4/carry copies.  The currently referenced
    # alternating prefix slot can repair that boundary; an older overwritten
    # slot or a missing pre-commit generated latent is intentionally fatal
    # rather than silently treating an incomplete audit trail as resumable.
    if native_resume:
        for existing_record in _load_native_records(native_segment_dir):
            existing_segment = native_segment_dir / str(existing_record["segment_file"])
            if not existing_segment.is_file():
                raise OutpaintError(
                    f"Committed native chunk has no segment: {existing_segment}"
                )
            _persist_committed_chunk(
                existing_record,
                existing_segment,
                native_segment_dir=native_segment_dir,
                mp4_dir=mp4_dir,
                latent_dir=latent_dir,
                log_path=log_path,
            )

    # Import the reference implementation only at execution time.  This exact
    # function owns LongRange planning, causal VAE handling, denoising,
    # prefix propagation, overlap emission, and native segment assembly.
    outpaint_src = root / "src" / "video_outpainting" / "main" / "src"
    if str(outpaint_src) not in sys.path:
        sys.path.insert(0, str(outpaint_src))
    from ltx_outpaint.inference import (
        LongRangeSpec,
        _iter_long_range_chunks,
        sample_long_range,
    )

    # Ask the packaged planner itself for the display total.  Generation below
    # independently invokes the same planner after reading the real VAE scale
    # factors; this value is instrumentation only and never drives generation.
    planned_chunk_count = sum(
        1
        for _ in _iter_long_range_chunks(
            frame_count, LongRangeSpec(**settings["long_range"])
        )
    )

    inference_started = time.monotonic()

    def committed_observer(record: dict[str, Any], segment_path: Path) -> None:
        artifact = _persist_committed_chunk(
            record,
            segment_path,
            native_segment_dir=native_segment_dir,
            mp4_dir=mp4_dir,
            latent_dir=latent_dir,
            log_path=log_path,
        )
        completed_frames = int(record["new_output_frame_range"][1])
        elapsed = time.monotonic() - inference_started
        eta = (
            elapsed * (frame_count - completed_frames) / completed_frames
            if completed_frames
            else 0.0
        )
        print(
            "[OUTPAINT_CHUNK_DONE] "
            f"chunk={int(record['chunk_index']) + 1}/{planned_chunk_count} "
            f"completed_frames={completed_frames}/{frame_count} "
            f"mp4={artifact['chunk_mp4']['path']} "
            f"latent={artifact['generated_latent']['path']} "
            f"carry={artifact['carry_prefix_state']['path']} "
            f"elapsed={elapsed:.1f}s eta_total={eta:.1f}s",
            flush=True,
        )

    sample_long_range(
        settings=settings,
        anchor_video=preprocess_path,
        output_path=processing_path,
        prompt=str(outpaint["positive_prompt"]),
        negative_prompt=str(outpaint["negative_prompt"]),
        num_frames=frame_count,
        checkpoint_path=checkpoint,
        seed=int(config["seed"]),
        inference_steps=30,
        segment_dir=native_segment_dir,
        resume=native_resume,
        keep_intermediates=True,
        manifest_dir=manifests_dir,
        certification_latent_dir=latent_dir,
        chunk_committed_observer=committed_observer,
    )

    processing_probe = _probe(processing_path)
    processing_metadata = _validate_video(
        processing_probe,
        path=processing_path,
        expected_frames=frame_count,
        expected_fps=PROCESSING_FPS,
    )

    records = _load_native_records(native_segment_dir)
    artifacts: list[dict[str, Any]] = []
    for record in records:
        segment_path = native_segment_dir / str(record["segment_file"])
        if not segment_path.is_file():
            raise OutpaintError(f"Missing native segment: {segment_path}")
        artifacts.append(
            _persist_committed_chunk(
                record,
                segment_path,
                native_segment_dir=native_segment_dir,
                mp4_dir=mp4_dir,
                latent_dir=latent_dir,
                log_path=log_path,
            )
        )
    _write_certification_manifests(
        records,
        artifacts,
        manifests_dir=manifests_dir,
        frame_count=frame_count,
    )

    # Test-only output selection changes postprocessing, never the shared sampler.
    # Both operational Full and Quick finish their entire requested sequence first.
    native_only = config.get("validation", {}).get("native_outpaint_only", False)
    _atomic_json(metadata_dir / "ffprobe_processing_24fps.json", processing_probe)
    if native_only:
        final_metadata = None
        result_path = processing_path
        print("[OUTPAINT_NATIVE_MASTER_READY] "
              f"frames={frame_count} fps=24 retime=SKIPPED_VALIDATION path={processing_path}", flush=True)
    else:
        final_metadata = _retime_to_16fps(
            processing_path, final_path, frame_count=frame_count,
            log_path=log_path, resume=resume,
        )
        result_path = final_path
        _atomic_json(metadata_dir / "ffprobe_final.json", _probe(final_path))
    _atomic_json(
        metadata_dir / "inference_config.json",
        {
            "runtime_config": str(runtime_config),
            "reference_entrypoint": "ltx_outpaint.inference.sample_long_range",
            "requested_frames": frame_count,
            "processing_output": processing_metadata,
            "stored_output": final_metadata,
            "processing_fps": 24,
            "stored_fps": 24 if native_only else 16,
            "output_role": "native_24fps_validation_master" if native_only else "final_16fps_deliverable",
            "native_24fps_master": processing_metadata,
            "final_retime_performed": not native_only,
            "long_range_domain_fps": 24,
            "frame_index_mapping": "processing[k] -> stored[k]",
            "frame_count_preserved": True,
            "fps_resampling": False,
            "retime_filter": None if native_only else FINAL_RETIME_FILTER,
            "long_range": settings["long_range"],
            "memory_optimization": settings["memory_optimization"],
            "recurrence": settings["recurrence"],
            "recurrence_implementation": settings["recurrence_implementation"],
            "positive_prompt": str(outpaint["positive_prompt"]),
            "negative_prompt": str(outpaint["negative_prompt"]),
            "inference_steps": 30,
            "alpha_feather_px": 16,
        },
    )
    _atomic_json(
        metadata_dir / "checkpoint_info.json",
        {
            "checkpoint_argument": str(checkpoint),
            "adapter_file": str(adapter_file),
            "adapter_size_bytes": adapter_file.stat().st_size,
            "adapter_sha256": _sha256(adapter_file),
        },
    )
    _atomic_json(
        metadata_dir / "seed.json",
        {
            "seed": int(config["seed"]),
            "seed_mode": "global_rng_and_reference_chunk_seed",
            "reference_chunk_seed": "base_seed + absolute_start_frame",
        },
    )
    return result_path


__all__ = ["OutpaintError", "run_outpaint"]
