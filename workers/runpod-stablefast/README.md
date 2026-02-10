# RunPod Stable-Fast VJ Server

WebRTC server with stable-fast inference for real-time AI visual generation.

## Files

- `setup.sh` - Complete installation script
- `server.js` - Node.js WebRTC server with queue management
- `inference_server.py` - Python stable-fast inference process
- `restart.sh` - Server restart script
- `package.json` - Node.js dependencies

## Quick Setup (on RunPod A100)

```bash
# SSH into pod
ssh root@<pod-ip> -p <ssh-port>

# Run setup (installs everything)
chmod +x setup.sh && ./setup.sh

# Start server
./restart.sh
```

## Manual Setup Steps

If setup.sh fails, run these commands manually:

```bash
# 1. Install system deps
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get update && apt-get install -y nodejs python3 python3-pip wget

# 2. Install Python deps
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.27.2 transformers==4.37.2 accelerate==0.26.1 peft==0.8.2 huggingface_hub==0.21.4

# 3. Install stable-fast
wget https://github.com/chengzeyi/stable-fast/releases/download/v1.0.1/stable_fast-1.0.1+torch212cu121-cp310-cp310-manylinux2014_x86_64.whl
pip install stable_fast-1.0.1+torch212cu121-cp310-cp310-manylinux2014_x86_64.whl

# 4. Download models
mkdir -p /workspace/models
cd /workspace/models
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/sd-turbo', local_dir='sd-turbo')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('madebyollin/taesd', local_dir='taesd')"

# 5. Install Node.js deps
cd /workspace/vj-server
npm install express @roamhq/wrtc

# 6. Start server
./restart.sh
```

## Performance

- **256x256**: 14-20ms (50-72 FPS)
- **512x512**: ~20ms (49 FPS)
- **GPU**: NVIDIA A100-SXM4-80GB required
- **CUDA**: 12.1+ with PyTorch 2.1.2

## WebRTC Protocol

**Client → Server (JSON settings):**
```json
{
  "prompt": "colorful abstract art",
  "seed": 42,
  "captureWidth": 128,
  "captureHeight": 128,
  "width": 256,
  "height": 256
}
```

**Client → Server (Binary):** Raw RGB frame bytes

**Server → Client (Binary):** JPEG image bytes

**Server → Client (JSON stats):**
```json
{"type": "stats", "gen_time_ms": 18.5, "width": 256, "height": 256}
```

## Queue Management

The server implements smart queue management:
- **Parameter changes** clear queued frames to prevent stale processing
- **Client pauses** frame sending during settings updates
- **Backpressure handling** prevents buffer overflow

## Health Check

```bash
curl https://<pod-id>-3000.proxy.runpod.net/healthz
# Returns: {"ok":true,"inferenceReady":true}
```