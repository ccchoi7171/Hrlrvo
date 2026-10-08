"""Read-only discovery and incremental tailing for training JSONL files."""
from __future__ import annotations

import json
import os
import threading
from hashlib import sha256
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReaderSnapshot:
    path: str
    records: tuple[dict[str, Any], ...]
    byte_offset: int
    pending_bytes: int
    malformed_lines: int
    reset_count: int
    file_mtime: float | None
    last_error: str | None


class IncrementalJsonlReader:
    """Append-only JSONL reader that never locks or writes its source file."""

    def __init__(self, path: str | Path, *, max_records: int | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.max_records = max_records
        self._offset = 0
        self._pending = b""
        self._records: list[dict[str, Any]] = []
        self._identity: tuple[int, int] | None = None
        self._malformed_lines = 0
        self._reset_count = 0
        self._mtime: float | None = None
        self._head_len = 0
        self._head_digest: str | None = None
        self._last_error: str | None = None
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._offset = 0
            self._pending = b""
            self._records.clear()
            self._identity = None
            self._mtime = None
            self._head_len = 0
            self._head_digest = None
            self._last_error = None
            self._reset_count += 1

    def refresh(self) -> ReaderSnapshot:
        with self._lock:
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                self._last_error = f"log unavailable: {self.path}"
                return self.snapshot()
            except OSError as exc:
                self._last_error = f"log stat failed: {exc}"
                return self.snapshot()

            identity = (stat.st_dev, stat.st_ino)
            prefix_changed = False
            if self._identity == identity and self._head_len:
                try:
                    with self.path.open("rb") as handle:
                        prefix_changed = sha256(handle.read(self._head_len)).hexdigest() != self._head_digest
                except OSError as exc:
                    self._last_error = f"log read failed: {exc}"
                    return self.snapshot()
            if self._identity is not None and (
                identity != self._identity or stat.st_size < self._offset or prefix_changed
            ):
                self._offset = 0
                self._pending = b""
                self._records.clear()
                self._malformed_lines = 0
                self._head_len = 0
                self._head_digest = None
                self._reset_count += 1
            self._identity = identity
            self._mtime = stat.st_mtime

            try:
                with self.path.open("rb") as handle:
                    handle.seek(self._offset)
                    chunk = handle.read()
                    self._offset = handle.tell()
            except OSError as exc:
                self._last_error = f"log read failed: {exc}"
                return self.snapshot()

            if chunk:
                parts = (self._pending + chunk).split(b"\n")
                self._pending = parts.pop()
                for raw in parts:
                    if not raw.strip():
                        continue
                    try:
                        value = json.loads(raw.decode("utf-8"))
                        if isinstance(value, dict):
                            self._records.append(value)
                        else:
                            self._malformed_lines += 1
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        self._malformed_lines += 1
                if self.max_records is not None and len(self._records) > self.max_records:
                    self._records = self._records[-self.max_records :]
            if self._head_len == 0 and self._offset:
                self._head_len = min(self._offset, 4096)
                try:
                    with self.path.open("rb") as handle:
                        self._head_digest = sha256(handle.read(self._head_len)).hexdigest()
                except OSError as exc:
                    self._last_error = f"log read failed: {exc}"
                    return self.snapshot()
            self._last_error = None
            return self.snapshot()

    def snapshot(self) -> ReaderSnapshot:
        with self._lock:
            return ReaderSnapshot(
                path=str(self.path),
                records=tuple(self._records),
                byte_offset=self._offset,
                pending_bytes=len(self._pending),
                malformed_lines=self._malformed_lines,
                reset_count=self._reset_count,
                file_mtime=self._mtime,
                last_error=self._last_error,
            )


class ReaderRegistry:
    """Keeps one independent byte offset/cache per selectable run."""

    def __init__(self, *, max_records: int | None = None) -> None:
        self.max_records = max_records
        self._readers: dict[str, IncrementalJsonlReader] = {}
        self._lock = threading.Lock()

    def get(self, path: str | Path) -> IncrementalJsonlReader:
        key = str(Path(path).expanduser().resolve())
        with self._lock:
            if key not in self._readers:
                self._readers[key] = IncrementalJsonlReader(key, max_records=self.max_records)
            return self._readers[key]


def _last_valid_record(path: Path, max_bytes: int = 262_144) -> dict[str, Any] | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return None
    for raw in reversed(data.splitlines()):
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _format_mtime(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def discover_runs(
    runs_root: str | Path | None,
    *,
    direct_log: str | Path | None = None,
    mock: bool = False,
) -> list[dict[str, Any]]:
    """Return selector records newest-first without opening files for write."""
    if direct_log is not None:
        path = Path(direct_log).expanduser().resolve()
        candidates = [path]
    elif runs_root is None:
        candidates = []
    else:
        root = Path(runs_root).expanduser().resolve()
        candidates = list(root.glob("*/train_metrics.jsonl")) if root.is_dir() else []
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

    result = []
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        latest = _last_valid_record(path) or {}
        step = latest.get("optimizer_step", latest.get("global_step", latest.get("step", "N/A")))
        run_name = "MOCK DATA" if mock else path.parent.name
        prefix = "[MOCK] " if mock else ""
        result.append(
            {
                "label": f"{prefix}{run_name} | {_format_mtime(mtime) if mtime else 'unavailable'} | step {step}",
                "value": str(path),
                "run_name": run_name,
                "mtime": mtime,
                "latest_step": step,
            }
        )
    return result


def read_last_lines(path: str | Path, count: int = 100, max_bytes: int = 1_048_576) -> str:
    path = Path(path)
    if not path.is_file():
        return "console log unavailable"
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
        lines = data.decode("utf-8", errors="replace").splitlines()
        return "\n".join(lines[-count:]) or "console log empty"
    except OSError as exc:
        return f"console log unavailable: {exc}"


def read_text_artifact(path: str | Path, max_bytes: int = 1_048_576) -> str:
    path = Path(path)
    if not path.is_file():
        return "unavailable"
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        suffix = "\n... truncated by monitor ..." if len(data) > max_bytes else ""
        return data[:max_bytes].decode("utf-8", errors="replace") + suffix
    except OSError as exc:
        return f"unavailable: {exc}"
