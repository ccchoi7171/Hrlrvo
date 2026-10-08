set -eu
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
bash "$ROOT/status.sh"
ps -eo user,pid,ppid,etime,pcpu,pmem,args | grep -E 'run_arm|run_fhd|python|1191458' | grep -v grep | head -55
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do
 echo "=== $arm ==="; tail -n 18 "$ROOT/arms/$arm/logs/latest.log"
done
