#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python"
ENV_DIR="/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo-ImageSR"
REAL_COMMIT="a4abfb2979a7bbff3f69f58f58ae324608821e27"
HAT_COMMIT="1638a9a822581657811867bf670717f8371fc3e5"
REAL_SHA256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
HAT_SHA256="f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7"

clone_and_pin() {
  local url="$1" destination="$2" commit="$3"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$url" "$destination"
  fi
  git -C "$destination" fetch --quiet origin "$commit"
  git -C "$destination" checkout --quiet --detach "$commit"
}

mkdir -p "$ROOT/models/Real-ESRGAN/weights" "$ROOT/models/HAT/weights"
clone_and_pin \
  https://github.com/xinntao/Real-ESRGAN.git \
  "$ROOT/models/Real-ESRGAN/repository" "$REAL_COMMIT"
clone_and_pin \
  https://github.com/XPixelGroup/HAT.git \
  "$ROOT/models/HAT/repository" "$HAT_COMMIT"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ENV_DIR"
fi

# The host image pins its CUDA PyTorch build through a global pip constraint.
# Reuse that tested build and install only the official inference packages.
"$ENV_DIR/bin/python" -m pip install --no-deps \
  'basicsr==1.4.2' 'realesrgan==0.3.0' \
  addict future lmdb pyyaml requests tb-nightly yapf \
  gdown beautifulsoup4 filelock tqdm soupsieve

REAL_WEIGHT="$ROOT/models/Real-ESRGAN/weights/RealESRGAN_x4plus.pth"
HAT_WEIGHT="$ROOT/models/HAT/weights/Real_HAT_GAN_SRx4.pth"
if [[ ! -s "$REAL_WEIGHT" ]]; then
  curl -L --fail --retry 3 -o "$REAL_WEIGHT" \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
fi
if [[ ! -s "$HAT_WEIGHT" ]]; then
  "$ENV_DIR/bin/gdown" 1Ma12vCWT27P9M99-s2RXnynKN-OQsBrv -O "$HAT_WEIGHT"
fi

echo "$REAL_SHA256  $REAL_WEIGHT" | sha256sum --check --strict
echo "$HAT_SHA256  $HAT_WEIGHT" | sha256sum --check --strict
echo "Frame-image SR models are prepared in $ROOT/models"
