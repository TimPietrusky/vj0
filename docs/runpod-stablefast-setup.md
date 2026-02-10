# RunPod Stable-Fast VJ Setup

## Overview

This document describes the setup for running stable-fast on RunPod for live AI visual generation.

**Final Performance Results:**
| Resolution | Time | FPS |
|------------|------|-----|
| 256x256 | 14-20ms | **50-72 FPS** |
| 512x512 | 20ms | **49 FPS** |

## Pod Configuration

### Romania Pod (UDP Support)
- **Pod ID:** `3746utbd1i3x73`
- **GPU:** NVIDIA A100-SXM4-80GB
- **Location:** EU-RO-1 (Romania) - **Required for UDP ports**
- **Cost:** $1.49/hr
- **vCPUs:** 32 | **RAM:** 250GB
- **Image:** `timpietruskyblibla/pytorch:2.1.2-py3.10-cuda12.2.2-devel-ubuntu22.04`

### Port Mappings
| Internal | External | Protocol |
|----------|----------|----------|
| 22 | 12971 | TCP (SSH) |
| 3000 | (proxy) | HTTP |
| 10000 | 12972 | **UDP** |
| 10001 | 12973 | **UDP** |
| 10002 | 12974 | **UDP** |

### Connection
```bash
ssh root@213.173.102.4 -p 12971
```

## Critical Learnings

### 1. IPv6 Routing Issue with RunPod Proxy
**Browser gets `ERR_CONNECTION_RESET` due to broken IPv6 on RunPod's proxy!**

**Fix:** Add hosts file entry to force IPv4:
```
# Add to C:\Windows\System32\drivers\etc\hosts (run notepad as Admin)
104.18.6.228 3746utbd1i3x73-3000.proxy.runpod.net
```

To verify with curl (IPv4 only):
```bash
curl.exe -4 https://3746utbd1i3x73-3000.proxy.runpod.net/healthz
```

### 2. Image Preprocessing is the Bottleneck
**PIL/numpy preprocessing adds ~400ms overhead!**

```python
# SLOW (400ms+):
img_pil = Image.open('input.png')
result = pipe(image=img_pil, ...)

# FAST (14-20ms):
img_tensor = torch.rand(1, 3, 512, 512, dtype=torch.float16, device='cuda')
result = pipe(image=img_tensor, ...)
```

### 3. getImageData is Slow on Client
`canvas.getImageData()` causes GPU sync and blocks main thread (5-20ms).

**Mitigations:**
- Use smaller capture size (128x128 instead of 256x256) = 4x less data
- Server upscales to output resolution
- Debug canvas updates only every 10 frames

### 4. Seed Consistency
For consistent AI output style, use fixed seed:
```python
torch.manual_seed(seed)  # NOT seed + frame_count
```

### 5. WebRTC Backpressure
Check `bufferedAmount` before sending to prevent queue overflow:
```typescript
canSend(maxBuffered = 256 * 1024): boolean {
  return channel.bufferedAmount < maxBuffered;
}
```

### 6. Required PyTorch/CUDA Versions
stable-fast 1.0.1 requires specific versions:
- **PyTorch:** 2.1.2+cu121
- **CUDA:** 12.1 (driver 12.4 is compatible)
- **Python:** 3.10

### 7. UDP Requires EU-RO-1 Datacenter
Only the Romania datacenter supports UDP port exposure.

## Installation Steps

### 1. Install Dependencies
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.27.2 transformers==4.37.2 accelerate==0.26.1 peft==0.8.2 huggingface_hub==0.21.4
```

### 2. Install stable-fast
```bash
cd /workspace
wget https://github.com/chengzeyi/stable-fast/releases/download/v1.0.1/stable_fast-1.0.1+torch212cu121-cp310-cp310-manylinux2014_x86_64.whl
pip install stable_fast-1.0.1+torch212cu121-cp310-cp310-manylinux2014_x86_64.whl
```

### 3. Download Models
```bash
mkdir -p /workspace/models
cd /workspace/models

# SD-Turbo
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/sd-turbo', local_dir='sd-turbo')"

# Tiny VAE (fast decoder)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('madebyollin/taesd', local_dir='taesd')"
```

### 4. Install Node.js (for WebRTC server)
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
cd /workspace/vj-server
npm install express @roamhq/wrtc
```

## WebRTC Architecture

```
┌─────────────────┐                              ┌─────────────────┐
│   vj0 Browser   │                              │  RunPod A100    │
│                 │                              │                 │
│  Canvas Capture ├──── WebRTC DataChannel ─────►│  Node.js Server │
│  (128x128 RGB)  │                              │       │         │
│                 │◄─── JPEG images back ────────│       ▼         │
│  AI Generated   │                              │  Python Infer   │
│  (256x256)      │                              │  (stable-fast)  │
└─────────────────┘                              └─────────────────┘
```

## Server Files

### 1. `inference_server.py` - Python inference process
- Loads and compiles stable-fast pipeline
- Warms up at 128, 256, 512 resolutions
- Receives small capture image, upscales to output size
- Uses fixed seed for consistent style
- Returns JPEG via stdout JSON

### 2. `server.js` - Node.js WebRTC server
- CORS enabled for all origins
- Handles WebRTC signaling (`POST /webrtc/offer`)
- Health check (`GET /healthz`)
- Bridges WebRTC DataChannel to Python process

### Starting the Server

```bash
# SSH into pod
ssh root@213.173.102.4 -p 12971

# Kill any existing processes and start fresh
pkill -f inference_server.py
pkill node

# Start server (foreground to see logs)
cd /workspace/vj-server
node server.js

# Or background:
nohup node server.js > server.log 2>&1 &
```

### Server Endpoints

- `GET /healthz` - Returns `{ ok: true, inferenceReady: true/false }`
- `POST /webrtc/offer` - WebRTC signaling, accepts `{ sdp: RTCSessionDescriptionInit }`

### WebRTC DataChannel Protocol

**Client → Server (JSON):**
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

**Client → Server (Binary):**
- Raw RGB bytes: `captureWidth * captureHeight * 3` bytes

**Server → Client (Binary):**
- JPEG image bytes (output resolution)

**Server → Client (JSON):**
```json
{ "type": "stats", "gen_time_ms": 18.5, "width": 256, "height": 256 }
```

## Client Implementation

### Key Files
| File | Purpose |
|------|---------|
| `app/vj/VJApp.tsx` | Main VJ app with AI panel |
| `src/lib/ai/webrtc-transport.ts` | WebRTC client with backpressure |
| `src/lib/ai/transport.ts` | Transport interface |
| `.env.local` | `NEXT_PUBLIC_VJ0_WEBRTC_SIGNALING_URL` |

### Frame Capture Loop
- Uses `requestAnimationFrame` (not setInterval)
- Ref-based state (no React re-renders in hot path)
- Crops center square from 4:1 canvas
- Captures at small size (128x128 default)
- Converts RGBA → RGB
- Checks backpressure before sending

### UI Controls
- **Capture size:** 64, 128, 256 (affects getImageData speed)
- **Output size:** 256, 512 (what AI generates)
- **Target FPS:** 10, 20, 30, 60
- **Seed:** Fixed value + Randomize button
- **Prompt:** Text input for AI style

## Deploy Commands (Windows PowerShell)

```powershell
# Copy file to pod (PowerShell pipe method)
Get-Content workers/runpod-stablefast/inference_server.py -Raw | ssh root@213.173.102.4 -p 12971 "cat > /workspace/vj-server/inference_server.py"

# Or use sed to modify in-place on pod
ssh root@213.173.102.4 -p 12971 "sed -i 's/old_text/new_text/' /workspace/vj-server/inference_server.py"

# Check health (force IPv4)
curl.exe -4 https://3746utbd1i3x73-3000.proxy.runpod.net/healthz
```

## Quick Commands

```bash
# SSH into pod
ssh root@213.173.102.4 -p 12971

# Check server status
curl https://3746utbd1i3x73-3000.proxy.runpod.net/healthz

# View server logs on pod
tail -f /workspace/vj-server/server.log

# Restart server on pod
pkill -f inference_server.py; pkill node; cd /workspace/vj-server && node server.js

# Or background:
pkill -f inference_server.py; pkill node; cd /workspace/vj-server && nohup node server.js > server.log 2>&1 &

# Check what's running
ps aux | grep -E 'node|inference' | grep -v grep

# Verify inference_server.py content
grep 'capture_width' /workspace/vj-server/inference_server.py
```

## Current Status (Jan 22, 2026)

### Working
1. WebRTC connection established
2. Frames being sent and processed
3. AI generating images at 17-110ms per frame
4. Seed control for consistent style
5. Separate capture/output resolution
6. Backpressure handling

### Known Issues
1. **Performance varies 17ms-110ms** - likely CUDA graph caching
2. **Debug canvas updates slowly** - intentional (every 10 frames)
3. **Server needs manual restart** after code changes

### TODO
- [ ] Handle audio features → prompt conversion
- [ ] Test end-to-end latency
- [ ] Optimize CUDA graph stability
- [ ] Auto-reconnect on disconnect
