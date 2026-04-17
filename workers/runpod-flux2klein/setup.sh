#!/usr/bin/env bash
# One-shot bring-up for the FLUX.2 Klein worker on a fresh RunPod 5090 / Romania.
# Image: runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
# Pod ports: 22/tcp, 3000/http, 10000/udp, 10001/udp, 10002/udp
# Run from /workspace on the pod after `runpodctl pod create`.

set -euo pipefail

PIP="pip install --break-system-packages -q"
PROJECT_DIR=/workspace/vj0-flux2klein

echo "[1/5] python deps"
$PIP --upgrade pip
$PIP numpy pillow safetensors sentencepiece protobuf accelerate huggingface_hub
$PIP --upgrade transformers
$PIP --upgrade "git+https://github.com/huggingface/diffusers.git"
$PIP torchvision

# bitsandbytes 0.49 wants libnvJitLink.so.13 which isn't on this image — DO NOT install
# torchao 0.17 + sageattention installed but not used in the production server (kept for benches)

echo "[2/5] node.js (for WebRTC server)"
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

echo "[3/5] project files"
mkdir -p "$PROJECT_DIR"
# expects server.js, package.json, inference_server.py to be SCP'd into $PROJECT_DIR

echo "[4/5] node deps"
if [ -f "$PROJECT_DIR/package.json" ]; then
  ( cd "$PROJECT_DIR" && npm install --silent )
fi

echo "[5/5] pre-download Klein + small decoder (≈8GB) so first request is fast"
python3 -c "
from huggingface_hub import snapshot_download
for r in ['black-forest-labs/FLUX.2-klein-4B', 'black-forest-labs/FLUX.2-small-decoder']:
    print(f'fetching {r}...')
    snapshot_download(r)
print('models cached')
"

echo
echo "DONE. To start the server:"
echo "  cd $PROJECT_DIR && nohup node server.js > server.log 2>&1 < /dev/null &"
echo "Health check (from outside):"
echo "  curl https://<POD_ID>-3000.proxy.runpod.net/healthz"
