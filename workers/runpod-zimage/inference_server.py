#!/usr/bin/env python3
"""
Z-Image Turbo img2img inference server (Nunchaku FP4 on Blackwell).

Same stdin/stdout JSON protocol as workers/runpod-flux2klein/inference_server.py
so the shared server.js WebRTC bridge works unchanged — only INFERENCE_SCRIPT differs.

Winning config (see ../runpod-flux2klein/RESULTS.md "Z-Image" section + this dir's README):
  ZImageImg2ImgPipeline with Nunchaku FP4-quantized transformer (rank 128)
  + torch.compile(transformer + vae.decoder, mode="default")
  + pre-encoded prompt embeds via PromptCache
  + guidance_scale=0 (Turbo is distilled, CFG disabled)

Per-frame latency on RTX 5090 (256², n=3, strength=0.95):
  ~60 ms (16.7 fps)

Protocol (JSON, one message per line):
  client → server:
    {
      "prompt": "a bright lightning bolt against a black sky",  # optional, triggers re-encode
      "seed": 42,                                # optional
      "captureWidth": 256, "captureHeight": 256, # raw input image dimensions
      "width": 256, "height": 256,               # AI generation dimensions
      "alpha": 0.05,                             # 0=text2img feel, 1=copy input.
                                                 # We pass strength = 1 - alpha to the pipe.
                                                 # 0.05 → strength=0.95 → invisible input.
      "n_steps": 3,                              # 3 is the floor for clean; 4-6 for higher Q
      "image_base64": "..."                      # raw RGB bytes captureW*captureH*3, base64
    }
  server → client:
    {"log": "..."}                               # diagnostic
    {"status": "ready", "width": ..., "height": ...}
    {"status": "frame", "image_base64": "<jpg>", "gen_time_ms": 61.2, "width": ..., "height": ...}
    {"status": "error", "message": "..."}
    {"status": "shutdown"}
"""
import base64
import io
import json
import queue
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image

# --- Monkey-patch for nunchaku ↔ diffusers signature mismatch -----------------
# Nunchaku 1.3.0dev calls super().forward() positionally against a signature
# that changed in diffusers — forces kwargs to make it match.
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel as _ZT
from nunchaku.models.transformers.transformer_zimage import (
    NunchakuZImageTransformer2DModel as _NZT,
    NunchakuZImageRopeHook as _RopeHook,
)


def _fixed_forward(self, x, t, cap_feats, patch_size=2, f_patch_size=1, return_dict=True, **_kw):
    rope_hook = _RopeHook()
    self.register_rope_hook(rope_hook)
    try:
        return _ZT.forward(
            self, x, t, cap_feats,
            return_dict=return_dict,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )
    finally:
        self.unregister_rope_hook()
        del rope_hook


_NZT.forward = _fixed_forward

# --- defaults ----------------------------------------------------------------
ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
NUNCHAKU_REPO = "nunchaku-ai/nunchaku-z-image-turbo"
RANK = 128                             # r128 = quality; r32 ≈ 4ms faster, slightly softer
DEFAULT_PROMPT = "a bright white lightning bolt against a pitch black night sky, dramatic"
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_ALPHA = 0.05                   # strength = 1 - alpha = 0.95 (SD-Turbo semantics)
DEFAULT_N_STEPS = 3                    # floor for clean output; 4-6 for higher Q
DEFAULT_SEED = 42
MAX_SEQ_LEN = 256
JPEG_QUALITY = 80
WARMUP_ITERS = 4


# --- output helpers (line-buffered JSON to stdout) ---------------------------
def emit(**msg):
    print(json.dumps(msg), flush=True)


def log(text):
    emit(log=text)


# --- pipeline setup ----------------------------------------------------------
def setup_pipeline():
    log(f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from diffusers import ZImageImg2ImgPipeline
    from nunchaku import NunchakuZImageTransformer2DModel
    from nunchaku.utils import get_precision

    precision = get_precision()  # 'fp4' on sm_120 (Blackwell), 'int4' elsewhere
    filename = f"svdq-{precision}_r{RANK}-z-image-turbo.safetensors"
    log(f"loading nunchaku transformer {precision}_r{RANK} from {NUNCHAKU_REPO}...")
    t0 = time.perf_counter()
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        f"{NUNCHAKU_REPO}/{filename}", torch_dtype=torch.bfloat16,
    )
    log(f"transformer loaded in {time.perf_counter()-t0:.1f}s")

    log(f"loading ZImageImg2ImgPipeline (tokenizer/text_encoder/scheduler/vae from {ZIMAGE_REPO})...")
    t0 = time.perf_counter()
    pipe = ZImageImg2ImgPipeline.from_pretrained(
        ZIMAGE_REPO, transformer=transformer,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    log(f"pipe loaded in {time.perf_counter()-t0:.1f}s, "
        f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    log("compiling transformer + vae.decoder (mode=default)...")
    t1 = time.perf_counter()
    pipe.transformer = torch.compile(pipe.transformer, mode="default",
                                     fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default",
                                     fullgraph=False, dynamic=False)
    log(f"compile stubs registered in {time.perf_counter()-t1:.1f}s "
        f"(actual JIT happens on first call per shape)")

    return pipe


class PromptCache:
    """Cache prompt embeds per prompt — Qwen3-4B text encoder runs once per change."""
    def __init__(self, pipe):
        self.pipe = pipe
        self.embeds = None
        self.last_prompt = None

    def get(self, prompt: str):
        if prompt == self.last_prompt and self.embeds is not None:
            return self.embeds
        log(f"encoding prompt ({len(prompt)} chars)")
        t0 = time.perf_counter()
        r = self.pipe.encode_prompt(
            prompt=prompt, device="cuda",
            do_classifier_free_guidance=False, max_sequence_length=MAX_SEQ_LEN,
        )
        self.embeds = r[0] if isinstance(r, tuple) else r
        self.last_prompt = prompt
        log(f"prompt encoded in {(time.perf_counter()-t0)*1000:.0f}ms")
        return self.embeds


def generate(pipe, input_img, prompt_embeds, alpha, n_steps, height, width, seed):
    """Run img2img. `alpha` is flux2klein-style: 0=text2img feel, 1=copy input.
    We pass strength = 1 - alpha to the Z-Image pipeline."""
    strength = max(0.01, min(1.0, 1.0 - alpha))
    return pipe(
        prompt=None, prompt_embeds=prompt_embeds,
        image=input_img,
        strength=strength,
        num_inference_steps=n_steps,
        guidance_scale=0.0,
        height=height, width=width,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]


def warmup(pipe, prompt_cache, width, height, alpha, n_steps):
    log(f"warmup at {width}×{height}, n_steps={n_steps} ({WARMUP_ITERS} iters; first triggers compile)")
    embeds = prompt_cache.get(DEFAULT_PROMPT)
    fake = Image.new("RGB", (width, height), (32, 32, 32))
    for i in range(WARMUP_ITERS):
        t0 = time.perf_counter()
        _ = generate(pipe, fake, embeds, alpha, n_steps, height, width, DEFAULT_SEED)
        torch.cuda.synchronize()
        log(f"warmup {i+1}/{WARMUP_ITERS}: {(time.perf_counter()-t0)*1000:.0f}ms")


def pil_to_jpeg_bytes(img: Image.Image, quality=JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def bytes_to_pil(raw: bytes, w: int, h: int) -> Image.Image:
    """Accept either an encoded image (JPEG/PNG/WebP) or raw RGB at w*h*3 bytes."""
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

    warmup(pipe, prompt_cache, state["width"], state["height"],
           state["alpha"], state["n_steps"])

    emit(status="ready", width=state["width"], height=state["height"])

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
            if request_queue.full():
                try:
                    request_queue.get_nowait()
                except queue.Empty:
                    pass
            request_queue.put(data)

    threading.Thread(target=reader, daemon=True).start()

    last_shape = (state["width"], state["height"], state["n_steps"])
    log("entering main loop")

    while not shutdown.is_set():
        try:
            req = request_queue.get(timeout=0.05)
        except queue.Empty:
            continue

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

        new_shape = (state["width"], state["height"], state["n_steps"])
        if new_shape != last_shape:
            log(f"shape changed → {state['width']}×{state['height']} n={state['n_steps']}, re-warming")
            try:
                warmup(pipe, prompt_cache, state["width"], state["height"],
                       state["alpha"], state["n_steps"])
                last_shape = new_shape
            except Exception as e:
                emit(status="error", message=f"warmup failed: {e}")
                continue

        if "image_base64" not in req:
            continue

        try:
            raw = base64.b64decode(req["image_base64"])
            input_img = bytes_to_pil(
                raw, state["capture_width"], state["capture_height"]
            )
            if input_img.size != (state["width"], state["height"]):
                input_img = input_img.resize((state["width"], state["height"]), Image.LANCZOS)
        except Exception as e:
            emit(status="error", message=f"decode input failed: {e}")
            continue

        try:
            embeds = prompt_cache.get(state["prompt"])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = generate(pipe, input_img, embeds,
                           state["alpha"], state["n_steps"],
                           state["height"], state["width"], state["seed"])
            torch.cuda.synchronize()
            gen_ms = (time.perf_counter() - t0) * 1000

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
