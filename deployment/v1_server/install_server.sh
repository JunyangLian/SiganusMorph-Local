#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Run this installer as the dedicated application user, not root."
  exit 1
fi

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# CPU-only PyTorch keeps the deployment suitable for a 2-core cloud server.
python -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements-server.txt

# Install Ultralytics without its opencv-python dependency. The project needs
# opencv-contrib-python-headless for ArUco/ChArUco support.
python -m pip install ultralytics==8.3.252 --no-deps

mkdir -p results/formal_v1_user_outputs
python verify_server_runtime.py

echo "SiganusMorph V1.0 server environment installed successfully."
