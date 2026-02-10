#!/usr/bin/env python3
"""
Persistent stable-fast inference server.
Communicates via stdin/stdout JSON for integration with Node.js WebRTC server.
"""
import sys
import json
import torch
import numpy as np
import time
import threading
import queue
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderTiny
from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig
import base64


def log(msg):
    print(json.dumps({"log": msg}), flush=True)


def send_status(status, **kwargs):
    print(json.dumps({"status": status, **kwargs}), flush=True)


def setup_pipeline():
    log("Loading pipeline...")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "/workspace/models/sd-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    )
    pipe.to("cuda")

    pipe.vae = AutoencoderTiny.from_pretrained("/workspace/models/taesd").to(
        device=pipe.device, dtype=pipe.dtype
    )

    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    pipe.set_progress_bar_config(disable=True)

    log("Compiling with stable-fast...")
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipe = compile(pipe, config)

    return pipe


def warmup(pipe, width, height):
    log(f"Warming up at {width}x{height}...")
    img_tensor = torch.rand(1, 3, height, width, dtype=torch.float16, device="cuda")
    for i in range(10):
        _ = pipe(
            prompt="warmup",
            image=img_tensor,
            height=height,
            width=width,
            num_inference_steps=1,
            strength=1.0,
            guidance_scale=0,
        ).images[0]
    log("Pipeline warm and ready!")


def bytes_to_tensor(raw_bytes, width, height):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(dtype=torch.float16, device="cuda") / 255.0
    return tensor


def tensor_to_bytes(tensor):
    arr = (tensor.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
    return arr.tobytes()


def pil_to_jpeg_bytes(img, quality=85):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def main():
    capture_width = 128  # Input size from client
    capture_height = 128
    width = 256  # Output size for AI
    height = 256
    prompt = "colorful abstract art, vibrant colors"
    seed = 42
    last_size = (width, height)

    pipe = setup_pipeline()
    
    # Warmup at common resolutions
    for res in [128, 256, 512]:
        warmup(pipe, res, res)

    send_status("ready", width=width, height=height)

    # Input queue for requests
    request_queue = queue.Queue(maxsize=2)
    shutdown = threading.Event()

    def stdin_reader():
        for line in sys.stdin:
            if shutdown.is_set():
                break
            try:
                data = json.loads(line.strip())
                if data.get("command") == "shutdown":
                    shutdown.set()
                    break
                request_queue.put(data)
            except json.JSONDecodeError:
                pass

    reader_thread = threading.Thread(target=stdin_reader, daemon=True)
    reader_thread.start()

    # Pre-allocate tensor for efficiency
    img_tensor = torch.rand(1, 3, height, width, dtype=torch.float16, device="cuda")

    log("Entering main loop...")

    frame_count = 0
    while not shutdown.is_set():
        try:
            # Get request (non-blocking with timeout)
            try:
                request = request_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            # Update parameters
            if "prompt" in request:
                prompt = request["prompt"]
                log(f"Prompt updated: {prompt[:50]}...")
            if "seed" in request:
                seed = request["seed"]
            if "captureWidth" in request:
                capture_width = request["captureWidth"]
                capture_height = request.get("captureHeight", capture_width)
            if "width" in request:
                width = request["width"]
                height = request.get("height", width)
                
            # Check if resolution changed and needs re-warmup
            new_size = (width, height)
            if new_size != last_size:
                log(f"Resolution changed to {width}x{height}, warming up...")
                warmup(pipe, width, height)
                last_size = new_size
                img_tensor = torch.rand(
                    1, 3, height, width, dtype=torch.float16, device="cuda"
                )

            # Handle image input
            if "image_base64" in request:
                raw = base64.b64decode(request["image_base64"])
                try:
                    # Decode at capture size
                    img_tensor = bytes_to_tensor(raw, capture_width, capture_height)
                    # Upscale to output size if different
                    if capture_width != width or capture_height != height:
                        img_tensor = torch.nn.functional.interpolate(
                            img_tensor, size=(height, width), mode='bilinear', align_corners=False
                        )
                except Exception as e:
                    log(f"Error decoding image: {e}, raw size: {len(raw)}, expected: {capture_width*capture_height*3}")
                    continue

            # Generate
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            torch.manual_seed(seed)
            result = pipe(
                prompt=prompt,
                image=img_tensor,
                height=height,
                width=width,
                num_inference_steps=1,
                strength=1.0,
                guidance_scale=0,
            ).images[0]

            torch.cuda.synchronize()
            gen_time = (time.perf_counter() - t0) * 1000
            frame_count += 1

            # Convert to JPEG and send back
            jpeg_bytes = pil_to_jpeg_bytes(result, quality=80)
            jpeg_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

            print(
                json.dumps(
                    {
                        "status": "frame",
                        "image_base64": jpeg_b64,
                        "gen_time_ms": round(gen_time, 1),
                        "width": width,
                        "height": height,
                    }
                ),
                flush=True,
            )

        except Exception as e:
            send_status("error", message=str(e))
            import traceback
            log(traceback.format_exc())

    send_status("shutdown")


if __name__ == "__main__":
    main()
