set -euo pipefail
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
cd "$ROOT"
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do
 test -f "verification/preflight_${arm}.json"
 test ! -e "arms/$arm/producer.pid"
done
# One fresh all-GPU check before launching any arm; do not touch other processes.
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
 echo 'GPU compute process present; aborting launch without interference'; exit 4
fi
for item in '01_baseline exp01_baseline' '02_no_carry exp02_no_carry' '03_global_boost exp03_global_boost' '04_native_halo exp04_native_halo'; do
 read -r arm name <<< "$item"
 pane=$(tmux new-window -d -P -F '#{pane_id}' -t hrlrvo -n "$name" "bash -lc 'bash $ROOT/start_arm.sh $arm; echo RUNNER_SHELL_EXIT=\$?; exec bash'")
 printf '%s %s %s\n' "$arm" "$name" "$pane"
 printf '%s\n' "$pane" > "arms/$arm/tmux_pane.txt"
done
sleep 8
bash status.sh
