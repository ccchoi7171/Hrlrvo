#!/bin/bash
# Isolated, fail-closed preparation. No H200 paths or production process signals.
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
CODE_SHA="${1:?trusted local code-manifest SHA required}"
RUNTIME_SHA="${2:?trusted local runtime-manifest SHA required}"
ENV_SHA=10130bf43fcf5488371a4e84b42fdb7bac2441420781ab40ad1e6c5c29f7ac14
BSA_SHA=7f617e49b8c65879585a0be0ae8df93ff43e9a4ad75be2c09b8afb6a3329164d
WEIGHT_SHA=bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be
ENV_ARCHIVE="$R/transfer/flash_env.tar.gz"
WHEEL="$R/transfer/block_sparse_attn-0.0.2-cp311-cp311-linux_x86_64.whl"
mkdir -p "$R/logs" "$R/verification"
exec 9>"$R/verification/prepare.lock"
flock -n 9 || { echo "Another preparation process owns this run"; exit 1; }
trap 'code=$?; printf "exit=%s time=%s\n" "$code" "$(date -u +%FT%TZ)" > "$R/verification/preparation_exit.txt"' EXIT
echo "$CODE_SHA  $R/verification/code_manifest.json" | sha256sum -c -
echo "$RUNTIME_SHA  $R/verification/ffmpeg_bundle_manifest.json" | sha256sum -c -
printf '[PREPARE] started=%s pid=%s\n' "$(date -u +%FT%TZ)" "$$"

# rsync publishes these final names only after the complete transfer.
printf '[WAIT] packed environment archive and Ampere wheel\n'
until [ -f "$ENV_ARCHIVE" ] && [ -f "$WHEEL" ]; do sleep 30; done
echo "$ENV_SHA  $ENV_ARCHIVE" | sha256sum -c -
echo "$BSA_SHA  $WHEEL" | sha256sum -c -
if [ -e "$R/env" ]; then
  echo "Existing environment directory: inspect manually; refusing overwrite/resume"
  exit 1
fi
mkdir "$R/env"
tar xzf "$ENV_ARCHIVE" -C "$R/env"
env -u PYTHONPATH -u LD_PRELOAD LD_LIBRARY_PATH="$R/env/lib" \
  "$R/env/bin/python" -I "$R/env/bin/conda-unpack"
env -u PYTHONPATH -u LD_PRELOAD LD_LIBRARY_PATH="$R/env/lib" \
  "$R/env/bin/python" -I -m pip install --no-index --no-deps --force-reinstall "$WHEEL"
# conda-pack cannot relocate an editable package whose source is outside the
# original environment prefix. Point the existing same-version metadata to the
# byte-identical packaged repository, never to a stale H200 absolute path.
env -u PYTHONPATH -u LD_PRELOAD LD_LIBRARY_PATH="$R/env/lib" \
  "$R/env/bin/python" -I - "$R" <<'RELOCATE_PY'
from pathlib import Path
import os
import sys
r = Path(sys.argv[1]).resolve()
site = r / "env/lib/python3.11/site-packages"
old = "/home/work/ai-graphics_33/Hrlrvo/Upscaler/models/FlashVSR_v1.1/repository"
new = str(r / "package/src/models/flashvsr_v1_1/repository")
changes = []
for name in ("diffsynth.egg-link", "easy-install.pth"):
    path = site / name
    if not path.resolve().is_relative_to((r / "env").resolve()):
        raise RuntimeError("Editable metadata escapes isolated environment")
    content = path.read_text()
    if old not in content:
        raise RuntimeError(f"Expected original editable reference is missing: {path}")
    changes.append((path, content.replace(old, new)))
for path, content in changes:
    temporary = path.with_name(path.name + ".relocating")
    with temporary.open("x") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
print("EDITABLE_RELOCATION_PASS", new, flush=True)
RELOCATE_PY
printf '[ENV_INSTALLED] %s original_archive=%s ampere_wheel=%s\n' \
  "$(date -u +%FT%TZ)" "$ENV_SHA" "$BSA_SHA" | tee "$R/verification/environment_install.txt"

printf '[WAIT] all approved source assets and four input/anchor files\n'
while true; do
  # The direct public download was staged separately from rsync's model directory.
  D="$R/package/src/models/flashvsr_v1_1"
  if [ -f "$R/transfer/main_weight_download.PASS" ] && \
     [ -d "$D/checkpoint" ] && \
     [ ! -e "$D/checkpoint/diffusion_pytorch_model_streaming_dmd.safetensors" ] && \
     [ -f "$D/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors" ]; then
    echo "$WEIGHT_SHA  $D/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors" | sha256sum -c -
    mv -n "$D/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors" \
      "$D/checkpoint/diffusion_pytorch_model_streaming_dmd.safetensors"
  fi
  if /usr/bin/python3 - "$R" <<'PY'
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
manifest = json.loads((r / "verification/source_provenance.json").read_text())
for entry in manifest["files"]:
    p = r / "package" / entry["path"]
    if not p.is_file() or p.stat().st_size != entry["bytes"]:
        sys.exit(1)
PY
  then break; fi
  sleep 30
done

control() {
  env -u PYTHONPATH -u LD_PRELOAD LD_LIBRARY_PATH="$R/env/lib" \
    "$R/env/bin/python" -I "$R/tools/remote_control.py" "$@"
}
control verify-assets --expected-code-manifest-sha256 "$CODE_SHA" \
  --expected-runtime-manifest-sha256 "$RUNTIME_SHA"
control environment
control smoke
control launch-full
printf '[FULL_LAUNCH_PASS] %s\n' "$(date -u +%FT%TZ)"
