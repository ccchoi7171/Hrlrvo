#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/work/ai-graphics_33/Hrlrvo/experiments/longrange_fhd_lineart_global_prior_20260923
cd "$ROOT"
bash "$ROOT/commands/run_lineart_detailed_C23.sh" > "$ROOT/logs/run_detailed.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$ROOT/logs/run_detailed.exit"
date -u +%FT%TZ > "$ROOT/logs/run_detailed.ended_utc"
exit "$rc"
