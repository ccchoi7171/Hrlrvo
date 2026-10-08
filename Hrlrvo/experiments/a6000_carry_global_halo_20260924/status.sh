#!/usr/bin/env bash
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
date -u
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
for ARM in "$ROOT"/arms/*; do
 echo "=== $(basename "$ARM") ==="
 cat "$ARM/status.json" 2>/dev/null || true
 cat "$ARM/queue_status.json" 2>/dev/null || true
 find "$ARM/chunks" -maxdepth 1 -name 'chunk_*.json' -printf '%f\n' 2>/dev/null | sort | tail -1 || true
 tail -n 3 "$ARM/logs/latest.log" 2>/dev/null || true
 echo
 done
