#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${CERT_ENTRY_PYTHON:-$here/../../../miniconda3/envs/Hrlrvo/bin/python}" "$here/src/run.py" "$@"
