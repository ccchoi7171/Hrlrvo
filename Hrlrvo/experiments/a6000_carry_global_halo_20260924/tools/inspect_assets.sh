set -eu
cat /home/oem/hrlrvo/portable_fhd/file_inventory.json
cat /home/oem/hrlrvo/portable_fhd/configs/baseline_05_run.readonly_copy.json | head -45
find /home/oem/hrlrvo/portable_fhd -maxdepth 3 -type f \( -iname '*sha*' -o -iname '*manifest*' -o -name 'model_inventory.json' \)
find /home/oem/hrlrvo/portable_fhd/models -maxdepth 3 -type d | head -40
cat /home/oem/hrlrvo/portable_fhd/environment/README* 2>/dev/null || true
/home/oem/hrlrvo/conda_env/bin/python -c 'import torch; print("torch",torch.__version__,"cuda",torch.version.cuda)'
