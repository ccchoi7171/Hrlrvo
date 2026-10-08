"""Validate replay against original C0 digests; preserve an immutable branch point."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "longrange_fhd_recurrence_test_20260923"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    old_dir = BASE / "chunks/strength_1p0"
    new_dir = ROOT / "chunks/c0_recovery"
    old = json.loads((old_dir / "chunk_000000.json").read_text())
    new = json.loads((new_dir / "chunk_000000.json").read_text())
    checks = {}
    for field in ("segment_pixel_sha256", "next_prefix_state_sha256", "carry_sha256"):
        checks[field] = {"expected": old[field], "actual": new[field], "match": old[field] == new[field]}
    actual_files = {
        "next_prefix_state_sha256": new_dir / new["next_prefix_file"],
        "carry_sha256": ROOT / "latent_states/c0_recovery" / new["carry_file"],
    }
    for field, path in actual_files.items():
        checks[field]["actual_file_sha256"] = digest(path)
        checks[field]["match"] &= checks[field]["actual_file_sha256"] == old[field]
    checks["segment_file_bytes"] = {
        "expected": digest(old_dir / old["segment_file"]),
        "actual": digest(new_dir / new["segment_file"]),
    }
    checks["segment_file_bytes"]["match"] = checks["segment_file_bytes"]["expected"] == checks["segment_file_bytes"]["actual"]
    report = {"status": "PASS" if all(c["match"] for c in checks.values()) else "FAIL", "checks": checks}
    report_path = ROOT / "report/c0_recovery_verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS":
        raise RuntimeError("C0 replay differs from original; C1 branch launch is forbidden. See " + str(report_path))
    archive = ROOT / "bootstrap/c0"
    archive.mkdir(parents=True, exist_ok=True)
    files = [
        old_dir / "chunk_000000.json",
        old_dir / old["segment_file"],
        new_dir / new["next_prefix_file"],
        BASE / "latent_states/strength_1p0" / old["carry_file"],
        old_dir / "model_ready.json",
        BASE / "outputs/strength_1p0/chunk_previews/C00_full65.mp4",
    ]
    for src in files:
        dst = archive / src.name
        if dst.exists():
            if digest(dst) != digest(src):
                raise RuntimeError("Existing bootstrap artifact changed: " + str(dst))
        else:
            shutil.copy2(src, dst)
    (archive / "provenance.json").write_text(json.dumps({
        "baseline": str(BASE), "c0_record": str(old_dir / "chunk_000000.json"),
        "prefix_state": "Recovered bit-identically by frozen-source C0 replay; serialized digest includes FP32 prefix plus CPU/CUDA RNG.",
        "verification": str(report_path),
        "reason": "Original run kept only alternating latest-prefix slots. Never bootstrap recurrence from MP4 or uint8 prefix.",
        "files": {p.name: digest(p) for p in files},
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "archive": str(archive), "verification": str(report_path)}))


if __name__ == "__main__":
    main()
