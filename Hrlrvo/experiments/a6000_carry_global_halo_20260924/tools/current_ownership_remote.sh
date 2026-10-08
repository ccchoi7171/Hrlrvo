set -eu
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
bash "$ROOT/status.sh"
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | paste -sd, -)
if [ -n "$PIDS" ]; then ps -p "$PIDS" -o user:20,pid,ppid,etime,args; fi
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do
 echo "=== $arm latest events ==="
 grep -E 'chunk_started|chunk_committed|model_ready|OutOfMemoryError|Loading text|Pre-computing|unloaded' "$ROOT/arms/$arm/logs/latest.log" | tail -4 || true
done
