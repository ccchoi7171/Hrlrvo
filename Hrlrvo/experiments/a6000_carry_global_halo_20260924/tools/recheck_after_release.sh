set -eu
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
bash "$ROOT/status.sh"
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | paste -sd, -)
if [ -n "$PIDS" ]; then ps -p "$PIDS" -o user:20,pid,ppid,etime,args; fi
ps -u oem -o pid,ppid,stat,etime,args | grep -E 'queue_arm|start_arm|run_arm.py|preflight.py|watch_chunk_previews' | grep -v grep || true
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do
 echo "=== queue $arm ==="
 tail -n 5 "$ROOT/arms/$arm/logs/queue.log" || true
 echo "=== recent $arm ==="
 tail -n 6 "$ROOT/arms/$arm/logs/latest.log" || true
done
