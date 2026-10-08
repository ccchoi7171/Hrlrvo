set -eu
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
bash "$ROOT/status.sh"
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do
 echo "=== pane $arm ==="
 pane=$(cat "$ROOT/arms/$arm/tmux_pane.txt")
 tmux capture-pane -p -S -16 -t "$pane"
done
