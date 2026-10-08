set -euo pipefail
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
cd "$ROOT"
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do test ! -e "arms/$arm/producer.pid"; done
tar -xzf admin/deploy_halo1.tgz
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=/home/oem/hrlrvo/conda_env/bin/python
"$PY" tools/verify_assets.py
"$PY" tools/relocate_checkpoint.py
for arm in 01_baseline 02_no_carry 03_global_boost 04_native_halo; do
 "$PY" preflight.py --arm "$arm"
done
