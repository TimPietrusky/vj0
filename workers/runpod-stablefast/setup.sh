#!/bin/bash
# Complete setup script for RunPod stable-fast VJ server
# This script installs all dependencies and sets up the environment

set -e

echo "=== Setting up stable-fast VJ server ==="

# Install system dependencies
echo "Installing system dependencies..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get update
apt-get install -y nodejs python3 python3-pip wget

# Install Python dependencies
echo "Installing Python dependencies..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.27.2 transformers==4.37.2 accelerate==0.26.1 peft==0.8.2 huggingface_hub==0.21.4

# Install stable-fast
echo "Installing stable-fast..."
wget https://github.com/chengzeyi/stable-fast/releases/download/v1.0.1/stable_fast-1.0.1+torch212cu121-cp310-cp310-manylinux2014_x86_64.whl
pip install stable_fast-1.0.1+torch212cu121-cp310-cp310-manylinux2014_x86_64.whl

# Download models
echo "Downloading AI models..."
mkdir -p /workspace/models
cd /workspace/models

echo "Downloading SD-Turbo..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/sd-turbo', local_dir='sd-turbo')"

echo "Downloading Tiny VAE..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('madebyollin/taesd', local_dir='taesd')"

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
cd /workspace/vj-server
npm install express @roamhq/wrtc

echo "=== Setup complete! ==="
echo "Run './restart.sh' to start the server"