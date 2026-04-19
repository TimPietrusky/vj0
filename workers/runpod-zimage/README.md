# runpod-zimage — Z-Image Turbo (Nunchaku FP4) img2img worker

WebRTC server for live VJ visuals powered by Z-Image Turbo, INT4/FP4 quantized via [Nunchaku](https://github.com/nunchaku-tech/nunchaku) SVDQuant. Target GPU: RTX 5090 (Blackwell sm_120).

## Performance (RTX 5090, 256², n=3, strength=0.95)

| Config | Latency | FPS |
|---|---|---|
| Nunchaku FP4_r128 + compile | **~60 ms** | **~16.7** |

Output quality: photorealistic, input influences noise seed but is invisible (SD-Turbo semantics).

See `../runpod-flux2klein/zimage-bench-extreme/results.json` for the full 48-cell sweep this worker's config was derived from.

## Reproducing the pod

### 1. Create the pod

```bash
runpodctl pod create \
  --image "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404" \
  --gpu-id "NVIDIA GeForce RTX 5090" \
  --data-center-ids "EU-RO-1" \
  --container-disk-in-gb 100 \
  --ports "22/tcp,3000/http,10000/udp,10001/udp,10002/udp" \
  --name "vj0-zimage-5090-ro"
```

**Why EU-RO-1**: only RunPod DC that supports inbound UDP — required for WebRTC.

### 2. SCP the worker files

```bash
POD_ID=<your-pod-id>
SSH_INFO=$(runpodctl ssh info "$POD_ID")
POD_IP=$(echo "$SSH_INFO" | jq -r .ip)
POD_PORT=$(echo "$SSH_INFO" | jq -r .port)
KEY=/Users/$USER/.runpod/ssh/RunPod-Key-Go

scp -i "$KEY" -P "$POD_PORT" \
  inference_server.py server.js package.json setup.sh requirements.txt \
  "root@$POD_IP:/workspace/"
```

### 3. Run setup on the pod (≈5 min — model download is the long part)

```bash
ssh -i "$KEY" "root@$POD_IP" -p "$POD_PORT" \
  "cd /workspace && bash setup.sh && cp inference_server.py server.js package.json /workspace/vj0-zimage/"
```

### 4. Start the server (long-running, background)

```bash
ssh -i "$KEY" "root@$POD_IP" -p "$POD_PORT" \
  "cd /workspace/vj0-zimage && nohup node server.js > server.log 2>&1 < /dev/null & disown"
```

First `/healthz` call will return `inferenceReady: false` for ~90s while torch.compile runs on first frame shapes. Subsequent calls are ~60ms.

### 5. Verify

```bash
curl "https://$POD_ID-3000.proxy.runpod.net/healthz"
# {"ok":true,"inferenceReady":true}
```

### 6. Plug into the vj0 app

1. Open the app.
2. Select **Z-Image Turbo** in the AI backend dropdown — the signaling URL `https://<POD_ID>-3000.proxy.runpod.net/webrtc/offer` is hardcoded in `src/lib/stores/ai-settings-store.ts::AI_BACKEND_URLS`.

If you created a new pod, update the `zimage` URL in that map.

## Files

- `inference_server.py` — Python worker. Speaks the stdin/stdout JSON protocol shared with flux2klein.
- `server.js` — Node WebRTC bridge (identical to flux2klein's `server.js` — frame-in/frame-out over a DataChannel; spawns the Python worker).
- `setup.sh` — pod provisioning (deps, model download, node install).
- `requirements.txt` — pinned Python deps.
- `package.json` — node deps (`@roamhq/wrtc`, `express`).

## Config knobs (in `inference_server.py`)

| Name | Default | Effect |
|---|---|---|
| `RANK` | 128 | Nunchaku low-rank rank. 32 ≈ 4ms faster, softer. 128 = quality. 256 = INT4 only. |
| `DEFAULT_N_STEPS` | 3 | Denoise steps. 2 is half-denoised — 3 is the clean-output floor. 4-6 for higher quality. |
| `DEFAULT_ALPHA` | 0.05 | In `[0,1]`. `strength = 1 - alpha`. 0.05 → strength 0.95 → SD-Turbo style (prompt dominates). |
| `DEFAULT_WIDTH` / `HEIGHT` | 256 | Pipeline resolution. 192² and 256² are ~same speed; 512² ≈ 2× slower. |

## Known pitfalls

- **Nunchaku 1.2.1 is broken** on Z-Image img2img — "Cannot access data pointer" storage error. Use 1.3.0dev (per setup.sh).
- **Nunchaku 1.3.0dev has a forward-signature bug** with current diffusers — `inference_server.py` monkey-patches `NunchakuZImageTransformer2DModel.forward` at import time to fix it.
- **Hub-based attention kernels** (sage_hub, _flash_3_hub) lack sm_120 kernels — don't try to enable them. The Nunchaku transformer uses its own optimized CUDA kernels; `torch.compile` mode=default is the winning compile setting.
- **`torch.compile(mode="max-autotune")` makes things slower** on Blackwell. Triton templates want 131–196KB shared memory; 5090 has 101KB/SM → templates fail → fallback to ATen. Stay on mode=default.
- **n=2 looks half-denoised.** Visually there's a lightning *shape* but it's textured and artifacty. n=3 is the clean-output floor.
