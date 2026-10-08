#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${CERT_OUTPAINT_PYTHON:-$ROOT/../../miniconda3/envs/Hrlrvo/bin/python}" "$ROOT/run_batch.py" "$@"
