set -euo pipefail
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do
 CUDA_VISIBLE_DEVICES='' /home/oem/hrlrvo/conda_env/bin/python preflight.py --arm "$arm"
done
bash tools/launch_four_remote.sh
