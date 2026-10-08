set -euo pipefail
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
cd "$ROOT"
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do
 pane=$(tmux new-window -d -P -F '#{pane_id}' -t hrlrvo -n "queue_$arm" "bash -lc 'bash $ROOT/queue_arm.sh $arm; echo QUEUE_EXIT=\$?; exec bash'")
 echo "$arm queue=$pane"
 echo "$pane" > "arms/$arm/queue_tmux_pane.txt"
done
sleep 5
bash status.sh
