#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/work/ai-graphics_33/Hrlrvo/Upscaler"
MODEL_ROOT="$ROOT/models/DOVE"
REPOSITORY="$MODEL_ROOT/repository"
CHECKPOINT="$MODEL_ROOT/checkpoint"
ENV_PREFIX="/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-DOVE"
CONDA="/home/work/ai-graphics_33/miniconda3/bin/conda"
DOVE_CODE_REV="0cd4240442cb5d122839c279977142cb6d648987"
DOVE_MODEL_REV="55ea86b7b2fbad77019a300e785b7fe1be065924"
BASE_MODEL_REV="fdc5267c90b5c06492985b966e43aae984e189e0"

mkdir -p "$MODEL_ROOT" "$CHECKPOINT"

if [[ ! -d "$REPOSITORY/.git" ]]; then
  git clone --filter=blob:none https://github.com/zhengchen1999/DOVE.git "$REPOSITORY"
fi
git -C "$REPOSITORY" fetch --quiet origin "$DOVE_CODE_REV"
git -C "$REPOSITORY" checkout --quiet --detach "$DOVE_CODE_REV"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA" create -y -n Hrlrvo-DOVE --clone Hrlrvo-FlashVSR
fi
env -u PIP_CONSTRAINT "$ENV_PREFIX/bin/python" -m pip install --quiet "diffusers==0.32.2"

MODEL_ROOT="$MODEL_ROOT" CHECKPOINT="$CHECKPOINT" \
DOVE_CODE_REV="$DOVE_CODE_REV" DOVE_MODEL_REV="$DOVE_MODEL_REV" \
BASE_MODEL_REV="$BASE_MODEL_REV" "$ENV_PREFIX/bin/python" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

checkpoint = Path(os.environ["CHECKPOINT"])
snapshot_download(
    repo_id="ZhengChen1999/DOVE",
    revision=os.environ["DOVE_MODEL_REV"],
    local_dir=checkpoint,
    allow_patterns=[
        "model_index.json",
        "scheduler/**",
        "tokenizer/**",
        "text_encoder/**",
        "transformer/**",
    ],
    max_workers=4,
)
snapshot_download(
    repo_id="zai-org/CogVideoX1.5-5B",
    revision=os.environ["BASE_MODEL_REV"],
    local_dir=checkpoint,
    allow_patterns=["vae/**"],
    max_workers=2,
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = {}
for path in sorted(checkpoint.rglob("*")):
    if path.is_file() and ".cache" not in path.parts and path.name != "provenance.json":
        files[str(path.relative_to(checkpoint))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

manifest = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "official_code": {
        "repo": "https://github.com/zhengchen1999/DOVE",
        "revision": os.environ["DOVE_CODE_REV"],
    },
    "checkpoint_assembly": {
        "reason": "official Stage-2 Google Drive artifact was quota-blocked",
        "dove_components": {
            "repo": "ZhengChen1999/DOVE",
            "revision": os.environ["DOVE_MODEL_REV"],
            "paths": ["transformer", "text_encoder", "tokenizer", "scheduler"],
        },
        "vae_component": {
            "repo": "zai-org/CogVideoX1.5-5B",
            "revision": os.environ["BASE_MODEL_REV"],
            "paths": ["vae"],
        },
        "equivalence_note": (
            "Both component publishers are official, but byte-equivalence to the "
            "quota-blocked DOVE.zip has not been established."
        ),
    },
    "files": files,
}
temporary = checkpoint / ".provenance.json.tmp"
temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
temporary.replace(checkpoint / "provenance.json")
print(json.dumps({
    "status": "PASS",
    "checkpoint": str(checkpoint),
    "file_count": len(files),
    "total_bytes": sum(item["bytes"] for item in files.values()),
}, indent=2))
PY

"$ENV_PREFIX/bin/python" -m pip freeze > "$MODEL_ROOT/environment.freeze.txt"
echo "DOVE assets prepared at $MODEL_ROOT"
