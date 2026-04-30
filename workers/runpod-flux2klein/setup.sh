#!/usr/bin/env bash
# One-shot bring-up for the FLUX.2 Klein worker on a fresh RunPod 5090 / Romania.
# Image: runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
# Pod ports: 22/tcp, 3000/http, 10000/udp, 10001/udp, 10002/udp
# Run from /workspace on the pod after `runpodctl pod create`.
#
# Install order matters — see comments in requirements.txt.
#
# To persist model cache across pod recreates, attach a network volume
# (mounted at /workspace) and `export HF_HOME=/workspace/hf-cache` before
# running this script.

set -euo pipefail

PIP="pip install --break-system-packages -q"
PROJECT_DIR=/workspace/vj0-flux2klein
HF_CACHE_DIR=${HF_HOME:-/workspace/hf-cache}
export HF_HOME=$HF_CACHE_DIR
mkdir -p "$HF_CACHE_DIR"

echo "[1/6] torch 2.11.0+cu128 (driver 570 hosts can't run cu130 — see BENCH-2026-04-30.md)"
# --ignore-installed cryptography needed because the system cryptography has no RECORD file
$PIP --ignore-installed cryptography \
  torch==2.11.0 torchvision \
  --extra-index-url https://download.pytorch.org/whl/cu128

echo "[2/6] python deps (pinned diffusers commit, transformers 5.5+, etc.)"
$PIP -r "$(dirname "$0")/requirements.txt"

echo "[3/6] torchao for fp8 dynamic-activation quantization (Phase 3 winning piece)"
# Install matching torchao for cu128. 0.15+cu128 / 0.17+cu128 both work on Blackwell sm_120
# under torch 2.11+cu128 — Float8DynamicActivationFloat8WeightConfig is the path.
$PIP torchao --extra-index-url https://download.pytorch.org/whl/cu128

echo "[4/6] node.js (for WebRTC server)"
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

echo "[5/6] project files"
mkdir -p "$PROJECT_DIR"
# expects server.js, package.json, inference_server.py to be SCP'd into $PROJECT_DIR

echo "[6/6] node deps + pre-download Klein + small decoder (≈8GB) into HF_HOME=$HF_HOME"
if [ -f "$PROJECT_DIR/package.json" ]; then
  ( cd "$PROJECT_DIR" && npm install --silent )
fi

python3 -c "
from huggingface_hub import snapshot_download
for r in ['black-forest-labs/FLUX.2-klein-4B', 'black-forest-labs/FLUX.2-small-decoder']:
    print(f'fetching {r}...')
    snapshot_download(r)
print('models cached')
"

echo
echo "Verify:"
python3 -c "import torch, torchao; print('torch=', torch.__version__, 'cuda_avail=', torch.cuda.is_available()); print('torchao=', torchao.__version__)"
python3 -c "from diffusers import Flux2KleinKVPipeline; print('klein KV imports OK')"

echo
echo "DONE. To start the server:"
echo "  cd $PROJECT_DIR && nohup node server.js > server.log 2>&1 < /dev/null &"
echo "Health check (from outside):"
echo "  curl https://<POD_ID>-3000.proxy.runpod.net/healthz"
