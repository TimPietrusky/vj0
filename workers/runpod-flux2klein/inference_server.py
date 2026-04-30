#!/usr/bin/env python3
"""
FLUX.2 Klein img2img inference server.

stdin/stdout JSON protocol — same shape as workers/runpod-stablefast/inference_server.py
so workers/runpod-stablefast/server.js can drive it unchanged (just point INFERENCE_SCRIPT
at this file).

Winning config (see BENCH-2026-04-30.md for the full grind; see RESULTS.md for the
2-step lineage that this builds on):
  Flux2KleinKVPipeline + FLUX.2-small-decoder, bf16
  + TorchAO Float8DynamicActivationFloat8WeightConfig on transformer linears
    (skipping pe_embedder/norms/embeds — same filter shape as Pruna's smash_config)
  + torch.compile(transformer + vae.encoder + vae.decoder, mode="default")
    (quantize_ MUST be applied before compile so kernels capture fp8 ops)
  + pre-encoded prompt embeds, max_seq_len=64
  + alpha-blend img2img: noisy = α·image_latents + (1-α)·noise
                         sigmas = linspace(1-α, 0, N_STEPS)
  + 4 inference steps default (production quality target — flip to 2 for speed-first)
  + alpha 0.05-0.18 (subtle SDXL-turbo-like input bias; up to 0.5 for hallucinated img2img)

Per-frame latency on RTX 5090 (driver 580, includes VAE encode of input):
  256² / 4 steps ≈ 38-45 ms (22-26 fps)   ← production target
  384² / 4 steps ≈ 65-75 ms (13-15 fps)
  512² / 4 steps ≈ 105-123 ms (8-10 fps)
  256² / 2 steps ≈ 25-30 ms (33-40 fps)   ← speed-first preset
  512² / 2 steps ≈ 75-100 ms (10-13 fps)

VRAM ≈ 12.3 GB (vs 16 GB without fp8) — 3.7 GB freed for batched inference / bigger res.

Protocol (JSON, one message per line):
  client → server:
    {
      "prompt": "neon city street at night",   # optional, triggers re-encode
      "seed": 42,                              # optional
      "captureWidth": 256, "captureHeight": 256,  # raw input image dimensions
      "width": 256, "height": 256,             # AI generation dimensions
      "alpha": 0.10,                           # 0=text2img, 1=copy input
      "n_steps": 2,                            # 1-4 (2 is the sweet spot)
      "image_base64": "..."                    # raw RGB bytes captureW*captureH*3, base64
    }
  server → client:
    {"log": "..."}                             # diagnostic
    {"status": "ready", "width": ..., "height": ...}
    {"status": "frame", "image_base64": "<jpg>", "gen_time_ms": 32.4, "width": ..., "height": ...}
    {"status": "error", "message": "..."}
    {"status": "shutdown"}
"""
import base64
import io
import json
import math
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

# ---- defaults ---- #
KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
DEFAULT_PROMPT = "vibrant neon cyberpunk city street at night, rain, reflections, wide angle"
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_ALPHA = 0.10
DEFAULT_N_STEPS = 4   # production quality target. Flip to 2 for the speed-first preset.
DEFAULT_SEED = 42
MAX_SEQ_LEN = 64
JPEG_QUALITY = 80
WARMUP_ITERS = 4
USE_FP8 = True        # flip to False to disable fp8 quantization (debug only)


# ---- output helpers (line-buffered JSON to stdout) ---- #
def emit(**msg):
    print(json.dumps(msg), flush=True)


def log(text):
    emit(log=text)


# ---- pipeline setup ---- #
def _fp8_filter(module, fqn):
    """Quantize transformer linear layers only. Skip pe_embedder, norms, embeds —
    same shape as Pruna's smash_config to keep precision-sensitive layers in bf16."""
    if not isinstance(module, torch.nn.Linear):
        return False
    if "transformer" not in fqn and "single_transformer_blocks" not in fqn:
        return False
    bad = ("pe_embedder", "norm_", "_norm", "embed", "out_proj")
    return not any(b in fqn.lower() for b in bad)


def setup_pipeline():
    log(f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    log(f"loading {KLEIN_REPO}...")
    t0 = time.perf_counter()
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    log(f"loaded in {time.perf_counter()-t0:.1f}s, "
        f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    if USE_FP8:
        # MUST be applied BEFORE torch.compile so the compiler captures the
        # fp8 ops in its kernel graph. See BENCH-2026-04-30.md Phase 3 for the
        # 24-30% latency win this delivers on Blackwell sm_120.
        log("applying TorchAO Float8DynamicActivationFloat8WeightConfig (PerRow) on transformer...")
        try:
            from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
            from torchao.quantization.granularity import PerRow
            try:
                cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
            except TypeError:
                cfg = Float8DynamicActivationFloat8WeightConfig()
            t_q = time.perf_counter()
            quantize_(pipe.transformer, cfg, filter_fn=_fp8_filter)
            log(f"fp8 applied in {time.perf_counter()-t_q:.1f}s, "
                f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
        except Exception as e:
            log(f"WARNING: fp8 quantization failed ({type(e).__name__}: {e}); "
                f"continuing in bf16. Set USE_FP8=False to skip this attempt.")

    log("compiling transformer + vae.encoder + vae.decoder (mode=default)...")
    t1 = time.perf_counter()
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
    log(f"compile stubs registered in {time.perf_counter()-t1:.1f}s "
        f"(actual JIT happens on first call per (height, width, steps) shape; "
        f"with fp8 expect ~80-160 s extra warmup at the first new resolution)")

    return pipe


class PromptCache:
    """Cache prompt embeds per (prompt, max_seq_len) — text encoder runs once per change."""
    def __init__(self, pipe):
        self.pipe = pipe
        self.embeds = None
        self.last_prompt = None
        self.last_max_seq = None

    def get(self, prompt: str, max_seq_len: int = MAX_SEQ_LEN):
        if prompt == self.last_prompt and max_seq_len == self.last_max_seq and self.embeds is not None:
            return self.embeds
        log(f"encoding prompt (len={len(prompt)} chars, max_seq={max_seq_len})")
        t0 = time.perf_counter()
        r = self.pipe.encode_prompt(
            prompt=prompt, device="cuda",
            num_images_per_prompt=1, max_sequence_length=max_seq_len,
        )
        self.embeds = r[0] if isinstance(r, tuple) else r
        self.last_prompt = prompt
        self.last_max_seq = max_seq_len
        log(f"prompt encoded in {(time.perf_counter()-t0)*1000:.0f}ms "
            f"shape={tuple(self.embeds.shape)}")
        return self.embeds


def encode_image_to_latents(pipe, img_pil, width, height):
    """Convert PIL RGB → patched VAE latents matching the pipeline's internal format.
    Latent spatial shape derives from (width, height); passing a square here when the
    pipeline is configured for a non-square output produces a square output regardless
    of the height/width args to pipe(...) — the latents shape wins."""
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents
    if img_pil.size != (width, height):
        img_pil = img_pil.resize((width, height), Image.LANCZOS)
    arr = np.asarray(img_pil, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
    patch = pipe._patchify_latents(raw)
    m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
    s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
    return (patch - m) / s


def generate(pipe, image_latents, prompt_embeds, alpha, n_steps, height, width, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn(image_latents.shape, generator=gen,
                        dtype=image_latents.dtype, device="cuda")
    noisy = alpha * image_latents + (1 - alpha) * noise
    sigmas = np.linspace(1 - alpha, 0.0, n_steps).tolist()
    return pipe(
        image=None, prompt=None, prompt_embeds=prompt_embeds,
        latents=noisy, sigmas=sigmas,
        height=height, width=width, num_inference_steps=n_steps,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]


def warmup(pipe, prompt_cache, width, height, alpha, n_steps):
    log(f"warmup at {width}×{height}, n_steps={n_steps} ({WARMUP_ITERS} iters; first one triggers compile)")
    embeds = prompt_cache.get(DEFAULT_PROMPT)
    fake_img = Image.new("RGB", (width, height), (32, 32, 32))
    lat = encode_image_to_latents(pipe, fake_img, width, height)
    for i in range(WARMUP_ITERS):
        t0 = time.perf_counter()
        _ = generate(pipe, lat, embeds, alpha, n_steps, height, width, DEFAULT_SEED)
        torch.cuda.synchronize()
        log(f"warmup {i+1}/{WARMUP_ITERS}: {(time.perf_counter()-t0)*1000:.0f}ms")


def pil_to_jpeg_bytes(img: Image.Image, quality=JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def bytes_to_pil(raw: bytes, w: int, h: int) -> Image.Image:
    """Accept either an encoded image (JPEG/PNG/WebP) or raw RGB at w*h*3 bytes.
    PIL auto-detects encoded formats from the magic bytes; raw RGB is the fallback."""
    # JPEG magic: ff d8 ff ; PNG: 89 50 4e 47 ; WebP: 52 49 46 46 ... 57 45 42 50
    if len(raw) >= 4 and (
        raw[:3] == b"\xff\xd8\xff"
        or raw[:4] == b"\x89PNG"
        or (raw[:4] == b"RIFF" and len(raw) >= 12 and raw[8:12] == b"WEBP")
    ):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    expected = w * h * 3
    if len(raw) != expected:
        raise ValueError(f"raw RGB length {len(raw)} != expected {expected} for {w}×{h} "
                         f"(and not a recognized image format)")
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    return Image.fromarray(arr, mode="RGB")


def main():
    pipe = setup_pipeline()
    prompt_cache = PromptCache(pipe)

    # current settings (mutated by client requests)
    state = {
        "prompt": DEFAULT_PROMPT,
        "seed": DEFAULT_SEED,
        "alpha": DEFAULT_ALPHA,
        "n_steps": DEFAULT_N_STEPS,
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "capture_width": DEFAULT_WIDTH,
        "capture_height": DEFAULT_HEIGHT,
    }

    # warmup at default size so first real frame is fast
    warmup(pipe, prompt_cache, state["width"], state["height"],
           state["alpha"], state["n_steps"])

    emit(status="ready", width=state["width"], height=state["height"])

    # incoming requests queue (so we can drop stale frames if behind)
    request_queue = queue.Queue(maxsize=2)
    shutdown = threading.Event()

    def reader():
        for line in sys.stdin:
            if shutdown.is_set():
                break
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if data.get("command") == "shutdown":
                shutdown.set()
                break
            # newest frame wins — drop older queued frame if queue full
            if request_queue.full():
                try: request_queue.get_nowait()
                except queue.Empty: pass
            request_queue.put(data)

    threading.Thread(target=reader, daemon=True).start()

    last_size = (state["width"], state["height"])
    frame_count = 0
    log("entering main loop")

    while not shutdown.is_set():
        try:
            req = request_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        # update settings from request (non-image fields)
        if "prompt" in req:
            state["prompt"] = req["prompt"]
        if "seed" in req:
            state["seed"] = int(req["seed"])
        if "alpha" in req:
            state["alpha"] = float(req["alpha"])
        if "n_steps" in req:
            state["n_steps"] = int(req["n_steps"])
        if "width" in req:
            state["width"] = int(req["width"])
            state["height"] = int(req.get("height", req["width"]))
        if "captureWidth" in req:
            state["capture_width"] = int(req["captureWidth"])
            state["capture_height"] = int(req.get("captureHeight", req["captureWidth"]))

        # if resolution or step count changed, re-warmup at new shape
        new_size = (state["width"], state["height"])
        if new_size != last_size:
            log(f"resolution changed → {state['width']}×{state['height']}, re-warming")
            try:
                warmup(pipe, prompt_cache, state["width"], state["height"],
                       state["alpha"], state["n_steps"])
                last_size = new_size
            except Exception as e:
                emit(status="error", message=f"warmup failed: {e}")
                continue

        # need an image to generate
        if "image_base64" not in req:
            continue

        try:
            raw = base64.b64decode(req["image_base64"])
            input_img = bytes_to_pil(
                raw, state["capture_width"], state["capture_height"]
            )
        except Exception as e:
            emit(status="error", message=f"decode input failed: {e}")
            continue

        try:
            embeds = prompt_cache.get(state["prompt"])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            lat = encode_image_to_latents(pipe, input_img, state["width"], state["height"])
            out = generate(pipe, lat, embeds,
                           state["alpha"], state["n_steps"],
                           state["height"], state["width"], state["seed"])
            torch.cuda.synchronize()
            gen_ms = (time.perf_counter() - t0) * 1000
            frame_count += 1

            jpg = pil_to_jpeg_bytes(out, JPEG_QUALITY)
            emit(
                status="frame",
                image_base64=base64.b64encode(jpg).decode("ascii"),
                gen_time_ms=round(gen_ms, 1),
                width=state["width"], height=state["height"],
            )
        except Exception as e:
            import traceback
            log(traceback.format_exc())
            emit(status="error", message=str(e))

    emit(status="shutdown")


if __name__ == "__main__":
    main()
