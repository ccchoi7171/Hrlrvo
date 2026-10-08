#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_from_c07_20260924
cd "$ROOT"
bash "$ROOT/commands/run_lineart_from_c07.sh" > "$ROOT/logs/run_from_c07.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$ROOT/logs/run_from_c07.exit"
date -u +%FT%TZ > "$ROOT/logs/run_from_c07.ended_utc"
exit "$rc"
