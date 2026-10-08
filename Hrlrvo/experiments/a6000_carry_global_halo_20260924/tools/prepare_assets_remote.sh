set -euo pipefail
ROOT=/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924
cd "$ROOT"
test ! -e run_arm.py
tar -xzf admin/deploy_initial.tgz
mkdir -p verification
ln -s /home/oem/hrlrvo/portable_fhd assets
git clone shared/ltx_pinned.bundle shared/LTX2.3
test "$(git -C shared/LTX2.3 rev-parse HEAD)" = 598ab41247a77dbfe29b5186e915bcf4f9040ec7
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/home/oem/hrlrvo/conda_env/bin/python tools/verify_assets.py
/home/oem/hrlrvo/conda_env/bin/python tools/relocate_checkpoint.py
