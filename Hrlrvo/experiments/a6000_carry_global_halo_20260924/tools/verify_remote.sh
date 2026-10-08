set -euo pipefail
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
cd "$ROOT"
# No inference has started. This updates only our new isolated experiment files.
test ! -e arms/01_baseline/producer.pid
tar -xzf admin/deploy_final.tgz
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=/home/oem/hrlrvo/conda_env/bin/python
"$PY" tools/verify_assets.py
"$PY" tools/relocate_checkpoint.py
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do
  "$PY" preflight.py --arm "$arm"
done
