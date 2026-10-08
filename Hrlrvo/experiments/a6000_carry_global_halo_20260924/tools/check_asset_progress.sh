set -eu
date -u
ps -eo pid,etime,pcpu,pmem,stat,args | grep -E 'verify_assets|preflight.py|relocate_checkpoint' | grep -v grep || true
find /home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/verification -maxdepth 1 -type f -printf '%f\n'
tail -n 8 /home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924/admin/four_arm_preflight_*.log
