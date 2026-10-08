#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "${CERT_OUTPAINT_PYTHON:-}" && -n "$conda_base" && -x "$conda_base/envs/Hrlrvo/bin/python" ]]; then
    CERT_OUTPAINT_PYTHON="$conda_base/envs/Hrlrvo/bin/python"
  fi
  if [[ -z "${CERT_FLASH_PYTHON:-}" && -n "$conda_base" && -x "$conda_base/envs/Hrlrvo-FlashVSR-py31115/bin/python" ]]; then
    export CERT_FLASH_PYTHON="$conda_base/envs/Hrlrvo-FlashVSR-py31115/bin/python"
  fi
fi
export CERT_FLASH_PYTHON="${CERT_FLASH_PYTHON:-$root/../../miniconda3/envs/Hrlrvo-FlashVSR-py31115/bin/python}"
exec "${CERT_OUTPAINT_PYTHON:-$root/../../miniconda3/envs/Hrlrvo/bin/python}" "$root/run_cert.py" "$@"
