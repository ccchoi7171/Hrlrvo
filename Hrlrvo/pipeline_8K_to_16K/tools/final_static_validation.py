"""CPU/static-only evidence. No model imports or media processing."""

import ast
import difflib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state/migration"
REFERENCE = ROOT.parent / "pipeline"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def schema_check(value, schema):
    """Check only the keywords present in our schemas; no external dependency."""
    known = {
        "$schema",
        "type",
        "const",
        "enum",
        "oneOf",
        "properties",
        "required",
        "additionalProperties",
        "minimum",
        "minItems",
        "items",
        "pattern",
    }
    assert not set(schema) - known, (
        f"unsupported schema keywords: {set(schema) - known}"
    )
    if "oneOf" in schema:
        successes = 0
        for option in schema["oneOf"]:
            try:
                schema_check(value, option)
                successes += 1
            except AssertionError:
                pass
        assert successes == 1
    if "const" in schema:
        assert type(value) is type(schema["const"]) and value == schema["const"]
    if "enum" in schema:
        assert any(
            type(value) is type(option) and value == option for option in schema["enum"]
        )
    kinds = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "boolean": bool,
    }
    if "type" in schema:
        assert type(value) is kinds[schema["type"]]
    if "pattern" in schema:
        assert re.search(schema["pattern"], value)
    if "minimum" in schema:
        assert value >= schema["minimum"]
    if isinstance(value, dict):
        assert set(schema.get("required", [])) <= set(value)
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            assert set(value) <= set(properties)
        for key, subschema in properties.items():
            if key in value:
                schema_check(value[key], subschema)
    if isinstance(value, list):
        assert len(value) >= schema.get("minItems", 0)
        if "items" in schema:
            for item in value:
                schema_check(item, schema["items"])


def main():
    records = [
        json.loads(line)
        for line in (STATE / "copy_integrity.jsonl").read_text().splitlines()
    ]
    mismatches, source_changes, dest_changes = [], [], []
    group_counts = Counter()
    approved_layout = "src/cert_pipeline/outpaint.py"
    for r in records:
        source, dest = Path(r["source"]), ROOT / r["destination"]
        before = (r["size_bytes"], r["source_mtime_ns"])
        s, d = source.stat(), dest.stat()
        if (s.st_size, s.st_mtime_ns) != before:
            source_changes.append(str(source))
        if r["status"] != "PASS" or r["destination_sha256"] != r["sha256"]:
            mismatches.append(r["destination"])
        # Every file was read and SHA-checked at copy time. Rehash all small files
        # here, and verify unchanged stat for large assets rather than reread 78GB.
        current = sha(dest) if d.st_size <= 8 * 1024**2 else r["destination_sha256"]
        if r["destination"] != approved_layout and (
            (d.st_size, d.st_mtime_ns) != before or current != r["sha256"]
        ):
            dest_changes.append(r["destination"])
        for group in (
            "src/video_outpainting/",
            "src/flashvsr/",
            "src/models/",
            "src/manifests/",
            "data/00_Input_8K_7680x4320/",
            "data/01_PreProc_2K_1920x1080/",
        ):
            if r["destination"].startswith(group):
                group_counts[group] += 1
    (STATE / "final_integrity.json").write_text(
        json.dumps(
            {
                "copy_files": len(records),
                "copy_bytes": sum(r["size_bytes"] for r in records),
                "copy_hash_mismatches": mismatches,
                "source_stat_changes_since_copy": source_changes,
                "destination_changes_outside_approved_layout": dest_changes,
                "approved_layout_change": approved_layout,
                "groups": dict(group_counts),
                "large_asset_method": "Full source and destination SHA at copy time; unchanged size/mtime now",
                "status": "PASS"
                if not (mismatches or source_changes or dest_changes)
                else "FAIL",
            },
            indent=2,
        )
        + "\n"
    )

    old = REFERENCE / approved_layout
    new = ROOT / approved_layout
    expected = (
        old.read_text()
        .replace('chunks_dir / "mp4"', 'chunks_dir / "rgb"')
        .replace("_Outpaint_4K_16fps.mp4", "_Outpaint_4K.mp4")
    )
    assert new.read_text() == expected, "Unexpected outpaint wrapper changes"
    (STATE / "outpaint_layout.diff").write_text(
        "".join(
            difflib.unified_diff(
                old.read_text().splitlines(True),
                new.read_text().splitlines(True),
                fromfile="audited/src/cert_pipeline/outpaint.py",
                tofile="new/src/cert_pipeline/outpaint.py",
            )
        )
    )
    (STATE / "runner_orchestration.diff").write_text(
        "".join(
            difflib.unified_diff(
                (REFERENCE / "run_cert.py").read_text().splitlines(True),
                (ROOT / "run_cert.py").read_text().splitlines(True),
                fromfile="audited/run_cert.py",
                tofile="new/run_cert.py",
            )
        )
    )

    parsed, syntax_failures = [], []
    for path in [
        ROOT / "run_cert.py",
        ROOT / "run_batch.py",
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "tools").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
    ]:
        try:
            compile(path.read_bytes(), str(path), "exec")
            parsed.append(str(path.relative_to(ROOT)))
        except (SyntaxError, UnicodeError) as error:
            syntax_failures.append(
                {"path": str(path.relative_to(ROOT)), "error": str(error)}
            )

    spec = importlib.util.spec_from_file_location("static_runner", ROOT / "run_cert.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    configs, config_changes = [], []
    schema = json.loads((ROOT / "configs/schemas/job.schema.json").read_text())
    for name in ("certification", "quick", "validation"):
        for path in sorted((ROOT / "configs" / name).glob("*.json")):
            config, *_ = runner.validate_config(path)
            schema_check(config, schema)
            if name != "validation":
                baseline = json.loads(
                    (ROOT / "docs/reference_configs" / name / path.name).read_text()
                )
                reduced = {
                    k: v
                    for k, v in config.items()
                    if k not in ("output_root", "artifacts")
                }
                old_reduced = {k: v for k, v in baseline.items() if k != "output_root"}
                if reduced != old_reduced:
                    config_changes.append(str(path.relative_to(ROOT)))
            configs.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha(path),
                    "status": "PASS",
                }
            )
    schema_check(
        json.loads((ROOT / "configs/batch/certification.json").read_text()),
        json.loads((ROOT / "configs/schemas/batch.schema.json").read_text()),
    )
    assert "torch" not in sys.modules, "Model import in static validation"

    old_functions = {
        n.name: n
        for n in ast.parse((REFERENCE / "run_cert.py").read_text()).body
        if isinstance(n, ast.FunctionDef)
    }
    new_functions = {
        n.name: n
        for n in ast.parse((ROOT / "run_cert.py").read_text()).body
        if isinstance(n, ast.FunctionDef)
    }
    helper_identity = {}
    for name in (
        "preflight_source",
        "verify_video",
        "seed_all",
        "run_logged",
        "first_frame_checksums",
        "upscale_input",
    ):
        helper_identity[name] = ast.dump(old_functions[name]) == ast.dump(
            new_functions[name]
        )
    old_flash, new_flash = (
        old_functions["run_flash"],
        new_functions["_run_flash_native"],
    )
    old_flash.name = new_flash.name
    old_flash.body[0] = new_flash.body[0]  # sole expected output-stage path adaptation
    helper_identity["run_flash_only_stage_path_changed"] = ast.dump(
        old_flash
    ) == ast.dump(new_flash)

    forbidden = [
        str(REFERENCE),
        str(ROOT.parent / "VideoOutpainting"),
        str(ROOT.parent / "Upscaler"),
    ]
    runtime_hits = []
    active = [
        ROOT / "run_cert.py",
        ROOT / "run_cert.sh",
        ROOT / "run_batch.py",
        ROOT / "run_batch.sh",
        *sorted((ROOT / "src/cert_pipeline").glob("*.py")),
        *sorted((ROOT / "src/video_outpainting/main/src").rglob("*.py")),
        *sorted((ROOT / "src/video_outpainting/main/configs").glob("*.yaml")),
        *sorted((ROOT / "src/flashvsr").rglob("*.py")),
        *sorted((ROOT / "configs").rglob("*.json")),
    ]
    for path in active:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if any(value in line for value in forbidden):
                runtime_hits.append(
                    {"path": str(path.relative_to(ROOT)), "line": number, "text": line}
                )
    missing_dirs = []
    for video in ("01", "02", "03", "04", "05"):
        for tail in ("", "/quick"):
            for sub in (
                "final",
                "chunks/rgb",
                "chunks/latents",
                "chunks/native",
                "chunks/manifests",
                "metadata",
                "logs",
            ):
                path = ROOT / f"data/02_Outpaint_4K_3840x2160/{video}{tail}/{sub}"
                if not path.is_dir():
                    missing_dirs.append(str(path.relative_to(ROOT)))
            for sub in (
                "final",
                "raw_flashvsr",
                "segments/tiles",
                "segments/metadata",
                "metadata",
                "logs",
            ):
                path = ROOT / f"data/03_Upscale_16K_15360x8640/{video}{tail}/{sub}"
                if not path.is_dir():
                    missing_dirs.append(str(path.relative_to(ROOT)))
    shell = subprocess.run(
        ["bash", "-n", str(ROOT / "run_cert.sh"), str(ROOT / "run_batch.sh")],
        capture_output=True,
        text=True,
    )
    symlinks = [
        str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*") if p.is_symlink()
    ]
    report = {
        "syntax_files": len(parsed),
        "syntax_failures": syntax_failures,
        "shell_returncode": shell.returncode,
        "shell_stderr": shell.stderr,
        "configs": configs,
        "config_computation_changes": config_changes,
        "schema_method": "All keywords present in local schemas checked by stdlib validator; runtime validate_config also called for all11 jobs",
        "helper_ast_identity": helper_identity,
        "external_active_runtime_path_hits": runtime_hits,
        "src_symlinks": symlinks,
        "missing_layout_directories": missing_dirs,
        "torch_imported": "torch" in sys.modules,
        "status": "PASS"
        if not (
            syntax_failures
            or shell.returncode
            or config_changes
            or runtime_hits
            or symlinks
            or missing_dirs
        )
        and all(helper_identity.values())
        else "FAIL",
    }
    (STATE / "static_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert report["status"] == "PASS"
    assert not (mismatches or source_changes or dest_changes)


if __name__ == "__main__":
    main()
