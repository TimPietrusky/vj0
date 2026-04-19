#!/usr/bin/env bash
# One-shot bring-up for the Z-Image Turbo (Nunchaku FP4) worker on a fresh RunPod 5090.
# Image: runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
# Pod ports: 22/tcp, 3000/http, 10000/udp, 10001/udp, 10002/udp
# Run from /workspace on the pod after `runpodctl pod create` and SCP'ing this dir.

set -euo pipefail

PIP="pip install --break-system-packages -q"
PROJECT_DIR=/workspace/vj0-zimage

echo "[1/6] upgrade pip"
$PIP --upgrade pip

echo "[2/6] base python deps"
$PIP numpy pillow safetensors sentencepiece protobuf accelerate huggingface_hub
$PIP --upgrade transformers
$PIP --upgrade "git+https://github.com/huggingface/diffusers.git"
$PIP torchvision

echo "[3/6] nunchaku (precompiled for Blackwell sm_120; auto-upgrades torch to 2.10)"
# Pin: cu13.0+torch2.10 wheel matches the base image's CUDA 12.8 driver
NUNCHAKU_WHL="https://github.com/nunchaku-tech/nunchaku/releases/download/v1.3.0dev20260306/nunchaku-1.3.0.dev20260306%2Bcu13.0torch2.10-cp312-cp312-linux_x86_64.whl"
$PIP "$NUNCHAKU_WHL"

echo "[4/6] node.js (for WebRTC server)"
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

echo "[5/6] project dir + node deps"
mkdir -p "$PROJECT_DIR"
# Expects server.js, package.json, inference_server.py to be SCP'd into $PROJECT_DIR
if [ -f "$PROJECT_DIR/package.json" ]; then
  ( cd "$PROJECT_DIR" && npm install --silent )
fi

echo "[6/6] pre-download Z-Image Turbo + Nunchaku FP4 weights (≈25GB so first request is fast)"
python3 -c "
from huggingface_hub import snapshot_download, hf_hub_download
print('fetching Tongyi-MAI/Z-Image-Turbo...')
snapshot_download('Tongyi-MAI/Z-Image-Turbo')
print('fetching nunchaku-ai/nunchaku-z-image-turbo (fp4_r128 only)...')
hf_hub_download('nunchaku-ai/nunchaku-z-image-turbo',
                'svdq-fp4_r128-z-image-turbo.safetensors')
print('models cached')
"

echo
echo "DONE. To start the server:"
echo "  cd $PROJECT_DIR && nohup node server.js > server.log 2>&1 < /dev/null &"
echo "Health check (from outside):"
echo "  curl https://<POD_ID>-3000.proxy.runpod.net/healthz"
echo
echo "Pod signaling URL for the vj0 app:"
echo "  https://<POD_ID>-3000.proxy.runpod.net/webrtc/offer"
