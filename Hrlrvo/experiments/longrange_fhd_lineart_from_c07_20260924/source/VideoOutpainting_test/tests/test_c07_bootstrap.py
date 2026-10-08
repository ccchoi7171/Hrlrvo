from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "main" / "src"))
SPEC = importlib.util.spec_from_file_location(
    "run_lineart_chain", PROJECT / "main" / "scripts" / "run_lineart_chain.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_archive(root: Path) -> tuple[Path, dict[str, bytes]]:
    archive = root / "archive"
    archive.mkdir()
    original_records: dict[str, bytes] = {}
    for index in range(7):
        (archive / f"segment_{index:06d}.npy").write_bytes(f"segment-{index}".encode())
        (archive / f"carry_{index:06d}.pt").write_bytes(f"carry-{index}".encode())
        record = {
            "chunk_index": index,
            "segment_file": f"segment_{index:06d}.npy",
            "carry_file": f"carry_{index:06d}.pt",
            # Older alternating prefix references are intentionally stale in
            # this transfer; only the latest C6 resume state is required.
            "next_prefix_file": f"previous_prefix_{index % 2}.pt",
        }
        name = f"chunk_{index:06d}.json"
        payload = (json.dumps(record, indent=2) + "\n").encode()
        (archive / name).write_bytes(payload)
        original_records[name] = payload
    (archive / "previous_prefix_0.pt").write_bytes(b"reconstructed-c6-prefix-and-rng")
    (archive / "model_ready.json").write_text('{"adapter_load": {}}\n')
    manifest = {
        path.name: _sha256(path)
        for path in archive.iterdir()
        if path.is_file() and path.name != "provenance.json"
    }
    provenance = {
        "status": "PASS",
        "bootstrap_through": 6,
        "source_root": "/read-only/baseline",
        "rgb_prefix_recovery": "uint8_lossless_segment_tail_div255",
        "bit_exact_fp32_prefix": False,
        "rgb_error_bound_normalized": 1 / 255,
        "latent_carry_exact": True,
        "rng_recovery": "verified",
        "file_sha256": manifest,
    }
    (archive / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return archive, original_records


def test_import_publishes_seven_commits_and_carries_without_rewriting_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, original_records = _make_archive(tmp_path)
    segments, latent = tmp_path / "chunks", tmp_path / "latent"
    published: list[str] = []
    real_atomic_copy = RUNNER.atomic_copy

    def recording_copy(source: Path, destination: Path) -> None:
        real_atomic_copy(source, destination)
        published.append(destination.name)

    monkeypatch.setattr(RUNNER, "atomic_copy", recording_copy)

    provenance = RUNNER.import_baseline_bootstrap(archive, segments, latent, 6)

    assert provenance["status"] == "PASS"
    assert [path.name for path in sorted(segments.glob("chunk_*.json"))] == [
        f"chunk_{index:06d}.json" for index in range(7)
    ]
    assert [path.name for path in sorted(latent.glob("carry_*.pt"))] == [
        f"carry_{index:06d}.pt" for index in range(7)
    ]
    assert (segments / "previous_prefix_0.pt").read_bytes() == (archive / "previous_prefix_0.pt").read_bytes()
    assert (segments / "model_ready.json").is_file()
    assert (segments / "bootstrap_provenance.json").read_bytes() == (archive / "provenance.json").read_bytes()
    for name, payload in original_records.items():
        assert (segments / name).read_bytes() == payload
    first_record = next(index for index, name in enumerate(published) if name.startswith("chunk_"))
    assert set(published[:first_record]) == {
        *(f"segment_{index:06d}.npy" for index in range(7)),
        *(f"carry_{index:06d}.pt" for index in range(7)),
        "previous_prefix_0.pt",
        "model_ready.json",
        "bootstrap_provenance.json",
    }
    assert published[-1] == "chunk_000006.json"


@pytest.mark.parametrize("failure", ["tampered", "missing"])
def test_import_rejects_invalid_hash_manifest_before_publishing(tmp_path: Path, failure: str) -> None:
    archive, _ = _make_archive(tmp_path)
    target = archive / "segment_000003.npy"
    if failure == "tampered":
        target.write_bytes(b"tampered")
    else:
        target.unlink()
    segments, latent = tmp_path / "chunks", tmp_path / "latent"

    with pytest.raises(RuntimeError, match="manifest|missing or corrupt"):
        RUNNER.import_baseline_bootstrap(archive, segments, latent, 6)

    assert not list(segments.glob("chunk_*.json"))
    assert not list(latent.glob("carry_*.pt"))
