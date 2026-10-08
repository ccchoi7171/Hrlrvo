#!/usr/bin/env bash
set -Eeuo pipefail

config_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

jobs=(
  "05:$config_root/05/05_VideoOutPaint_4K.sh"
  "04:$config_root/04/04_VideoOutPaint_4K.sh"
  "01:$config_root/01/01_VideoOutPaint_4K.sh"
  "02:$config_root/02/02_VideoOutPaint_4K.sh"
  "03:$config_root/03/03_VideoOutPaint_4K.sh"
)

current_child_pid=""
current_video_id=""
current_script=""

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

forward_signal() {
  local signal_name="$1"
  local exit_code="$2"

  # Prevent a repeated signal from re-entering this handler while the current
  # child process group is shutting down.
  trap - INT TERM
  if [[ -n "$current_child_pid" ]]; then
    if kill -0 -- "-$current_child_pid" 2>/dev/null; then
      kill -s "$signal_name" -- "-$current_child_pid" 2>/dev/null || true
    elif kill -0 "$current_child_pid" 2>/dev/null; then
      kill -s "$signal_name" "$current_child_pid" 2>/dev/null || true
    fi
    wait "$current_child_pid" 2>/dev/null || true
  fi

  printf '[ERROR] time=%s video_id=%s script=%s signal=%s exit_code=%d\n' \
    "$(timestamp)" "${current_video_id:-none}" "${current_script:-none}" \
    "$signal_name" "$exit_code" >&2
  exit "$exit_code"
}

trap 'forward_signal INT 130' INT
trap 'forward_signal TERM 143' TERM

# Validate every target before starting the first long-running job.
for job in "${jobs[@]}"; do
  video_id="${job%%:*}"
  script_path="${job#*:}"
  if [[ ! -f "$script_path" ]]; then
    printf '[ERROR] time=%s video_id=%s script=%s reason=missing\n' \
      "$(timestamp)" "$video_id" "$script_path" >&2
    exit 1
  fi
  if [[ ! -x "$script_path" ]]; then
    printf '[ERROR] time=%s video_id=%s script=%s reason=not-executable\n' \
      "$(timestamp)" "$video_id" "$script_path" >&2
    exit 1
  fi
done

printf '[MASTER START] time=%s order=05,04,01,02,03\n' "$(timestamp)"

# Job control gives each asynchronous child its own process group. The master
# still waits immediately, so at most one VideoOutPaint process can run.
set -m
for job in "${jobs[@]}"; do
  current_video_id="${job%%:*}"
  current_script="${job#*:}"

  printf '[START] time=%s video_id=%s script=%s\n' \
    "$(timestamp)" "$current_video_id" "$current_script"

  "$current_script" &
  current_child_pid=$!
  if wait "$current_child_pid"; then
    child_exit_code=0
  else
    child_exit_code=$?
  fi
  current_child_pid=""

  if (( child_exit_code != 0 )); then
    printf '[ERROR] time=%s video_id=%s script=%s exit_code=%d\n' \
      "$(timestamp)" "$current_video_id" "$current_script" \
      "$child_exit_code" >&2
    exit "$child_exit_code"
  fi

  printf '[DONE] time=%s video_id=%s script=%s\n' \
    "$(timestamp)" "$current_video_id" "$current_script"
done

current_video_id=""
current_script=""
printf '[ALL DONE] time=%s order=05,04,01,02,03\n' "$(timestamp)"
