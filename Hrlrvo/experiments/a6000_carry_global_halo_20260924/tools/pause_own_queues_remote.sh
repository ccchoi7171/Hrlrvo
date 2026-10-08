set -eu
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do
 pane=$(cat "$ROOT/arms/$arm/queue_tmux_pane.txt")
 tmux send-keys -t "$pane" C-c
done
sleep 2
ps -u oem -o pid,ppid,stat,args | grep -E 'queue_arm|start_arm|run_arm.py|preflight.py|watch_chunk_previews' | grep -v grep || true
