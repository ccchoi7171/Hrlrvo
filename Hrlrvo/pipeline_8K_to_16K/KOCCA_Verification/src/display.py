"""Concise KOCCA terminal events sourced from actual stage output."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

def elapsed_text(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([a-zA-Z_]+)=([^\s]+)", line))


class StageDisplay:
    def __init__(
        self, log: Any, request: dict[str, Any], run_dir: Path, source_frames: int,
    ) -> None:
        self.log = log
        self.request = request
        self.run_dir = run_dir
        self.source_frames = source_frames
        self.outpaint_frames = source_frames
        self.upscale_frames = source_frames
        self.outpaint_completed = 0
        self.outpaint_chunks = 0
        self.current_chunk: int | None = None
        self.upscale_completed = 0
        self.current_upscale_chunk: int | None = None
        self.current_upscale_range = ""
        self.preprocess_completed = -1

    def begin(self, stage: str) -> None:
        self.log.banner({
            "preprocess": "PreProcess",
            "outpaint": "VideoOutPainting",
            "upscale": "Upscale",
        }[stage])
        if stage == "preprocess":
            source = Path(self.request["input"]["path"])
            self.log.event("INPUT_LOAD", path=source)
            self.log.event("INPUT_CHECK", resolution="7680x4320",
                           frames=self.source_frames, status="PASS")

    def consume(self, stage: str, line: str) -> dict[str, Any] | None:
        line = line.strip()
        if stage == "preprocess":
            if "[PREPROCESS_PROGRESS]" in line:
                item = fields(line)
                completed, total = int(item["frame"]), int(item["total"])
                if completed > self.preprocess_completed:
                    self.log.progress("PREPROCESS_PROGRESS", completed, total,
                                      suffix=f"frames={completed}/{total}")
                    self.preprocess_completed = completed
                return {"phase": "preprocess", "completed": completed, "total": total}
            return None
        if stage == "outpaint":
            return self._outpaint(line)
        return self._upscale(line)

    def _outpaint(self, line: str) -> dict[str, Any] | None:
        if "[OUTPAINT_INPUT_CHECK]" in line:
            item = fields(line)
            self.log.event("INPUT_CHECK", resolution="1920x1080",
                           frames=item["frames"], status="PASS")
            return None
        if "[OUTPAINT_CHUNK_DONE]" in line:
            item = fields(line)
            completed = int(item["completed_frames"].split("/")[0])
            index = int(item["chunk"].split("/")[0]) - 1
            self._committed(index, completed)
            return {"phase": "outpaint_committed", "chunk": index,
                    "completed": completed, "total": self.source_frames}
        if "[OUTPAINT_ASSEMBLED]" in line:
            item = fields(line)
            self.outpaint_frames = int(item["frames"])
            self.upscale_frames = self.outpaint_frames
            chunks = int(item["chunks"])
            # A stop can land after native commit but before the wrapper's log.
            while self.outpaint_chunks < chunks:
                index = self.outpaint_chunks
                end = min(self.outpaint_frames, 65 + index * 48)
                self._committed(index, end)
            self.log.event("OUTPAINT_ASSEMBLED", frames=self.outpaint_frames,
                           chunks=chunks)
            return {"phase": "outpaint_assembled", "completed": self.outpaint_frames,
                    "chunks": chunks}
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(item, dict):
            return None
        if item.get("event") == "recurrence_chunk_start":
            index = int(item["chunk_index"])
            begin = 0 if index == 0 else 65 + (index - 1) * 48
            end = min(self.source_frames, 65 + index * 48) - 1
            self.current_chunk = index
            steps = int(self.request["outpainting"]["steps"])
            self.log.progress("OUTPAINT_PROGRESS", 0, steps,
                              prefix=f"chunk={index} frames={begin}-{end}",
                              suffix=f"step=0/{steps}")
            return {"phase": "outpaint_denoise", "chunk": index,
                    "step": 0, "total_steps": steps}
        if item.get("event") == "inference_denoise_step":
            index = int(item["chunk_index"])
            step, steps = int(item["step"]), int(item["total_steps"])
            begin = 0 if index == 0 else 65 + (index - 1) * 48
            end = min(self.source_frames, 65 + index * 48) - 1
            self.current_chunk = index
            self.log.progress("OUTPAINT_PROGRESS", step, steps,
                              prefix=f"chunk={index} frames={begin}-{end}",
                              suffix=f"step={step}/{steps}")
            return {"phase": "outpaint_denoise", "chunk": index,
                    "step": step, "total_steps": steps}
        if item.get("event") == "lineart_reference_ready":
            self.log.event("LINEART_READY", chunk=0, frame=0)
        return None

    def _committed(self, index: int, frames: int) -> None:
        if index < self.outpaint_chunks:
            return
        if index != self.outpaint_chunks:
            raise RuntimeError("OutPainting chunk completion log is not contiguous")
        self.outpaint_chunks += 1
        self.outpaint_completed = frames
        self.log.event("CHUNK_DONE", chunk=index, completed_frames=frames)

    def _upscale(self, line: str) -> dict[str, Any] | None:
        if "[UPSCALE_INPUT_CHECK]" in line:
            item = fields(line)
            self.upscale_frames = int(item["frames"])
            self.log.event("INPUT_CHECK", resolution="3840x2160",
                           frames=self.upscale_frames, status="PASS")
            return None
        if "[UPSCALE_CHUNK_START]" in line:
            item = fields(line)
            self.current_upscale_chunk = int(item["chunk"])
            self.current_upscale_range = item["frames"]
            self.log.event("UPSCALE_CHUNK_START", chunk=self.current_upscale_chunk,
                           frames=self.current_upscale_range)
            self.log.progress("UPSCALE_PROGRESS", 0, 1,
                              prefix=f"chunk={self.current_upscale_chunk} frames={self.current_upscale_range}")
            return {"phase": "upscale_chunk", "chunk": self.current_upscale_chunk}
        if "[UPSCALE_CHUNK_DONE]" in line:
            item = fields(line)
            self.upscale_completed = int(item["completed_frames"])
            self.log.event("UPSCALE_CHUNK_DONE", chunk=item["chunk"],
                           completed_frames=self.upscale_completed)
            return {"phase": "upscale_committed", "chunk": int(item["chunk"]),
                    "completed": self.upscale_completed, "total": self.outpaint_frames}
        if "[UPSCALE_CHUNK_EXCLUDED]" in line:
            item = fields(line)
            self.log.event("UPSCALE_CHUNK_EXCLUDED", chunk=item["chunk"],
                           reason=item["reason"])
            return None
        if "[UPSCALE_ASSEMBLED]" in line:
            item = fields(line)
            self.upscale_frames = int(item["frames"])
            self.log.event("UPSCALE_ASSEMBLED", frames=self.upscale_frames,
                           chunks=item["chunks"])
            return {"phase": "upscale_assembled", "completed": self.upscale_frames,
                    "chunks": int(item["chunks"])}
        if "[UPSCALE] tile=" in line:
            item = fields(line)
            tile, tiles = map(int, item["tile"].split("/"))
            frame, frames = map(int, item["frame"].split("/"))
            completed = (tile - 1) * frames + frame
            total = tiles * frames
            prefix = (f"chunk={self.current_upscale_chunk} frames={self.current_upscale_range}"
                      if self.current_upscale_chunk is not None else "")
            self.log.progress("UPSCALE_PROGRESS", completed, total,
                              prefix=prefix, suffix=f"tile={tile}/{tiles}")
            return {"phase": "flash_inference", "tile": tile, "tiles": tiles,
                    "frame": frame, "frames_per_tile": frames,
                    "completed_units": completed, "total_units": total}
        return None
