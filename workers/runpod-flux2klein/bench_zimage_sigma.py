#!/usr/bin/env python3
"""Sigma-injection img2img: pre-encode input once, do flow-matching noise mix per frame.

Same winning trick as flux2klein journal — bypasses the img2img pipeline's per-frame
VAE encode by encoding the input image once and feeding pre-noised latents directly.

Compare to pipeline baseline at 256² n=3 s=0.95 fp4_r128.
"""
from pathlib import Path
import time, json
import numpy as np
import torch
from PIL import Image

# Patch nunchaku forward signature
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel as _ZT
from nunchaku.models.transformers.transformer_zimage import (
    NunchakuZImageTransformer2DModel as _NZT,
    NunchakuZImageRopeHook as _RopeHook,
)
def _fixed_forward(self, x, t, cap_feats, patch_size=2, f_patch_size=1, return_dict=True, **kw):
    rope_hook = _RopeHook()
    self.register_rope_hook(rope_hook)
    try:
        return _ZT.forward(self, x, t, cap_feats, return_dict=return_dict,
                           patch_size=patch_size, f_patch_size=f_patch_size)
    finally:
        self.unregister_rope_hook()
        del rope_hook
_NZT.forward = _fixed_forward

OUT_DIR = Path("/workspace/zimage-bench-sigma")
WAVE_DIR = Path("/workspace/waveforms")
PROMPT = "a bright white lightning bolt against a pitch black night sky, dramatic"

SIZE = 256
STEPS = 3
STRENGTH = 0.95
RANK = 128
SEED = 42
N_WARMUP = 4
N_BENCH = 20
REPO = "nunchaku-ai/nunchaku-z-image-turbo"


def build_pipe():
    from diffusers import ZImageImg2ImgPipeline
    from nunchaku import NunchakuZImageTransformer2DModel
    from nunchaku.utils import get_precision

    precision = get_precision()
    filename = f"svdq-{precision}_r{RANK}-z-image-turbo.safetensors"
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        f"{REPO}/{filename}", torch_dtype=torch.bfloat16,
    )
    pipe = ZImageImg2ImgPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    return pipe


def encode_image_to_latents(pipe, pil_img, size):
    """Pre-encode PIL image to VAE latent space. Called once, reused every frame."""
    arr = np.asarray(pil_img.resize((size, size), Image.LANCZOS), dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    vae = pipe.vae
    with torch.no_grad():
        posterior = vae.encode(t).latent_dist
        latents = posterior.sample()
        # Apply scale/shift to match the pipeline's expectation
        if hasattr(vae.config, "scaling_factor"):
            latents = (latents - getattr(vae.config, "shift_factor", 0.0)) * vae.config.scaling_factor
    return latents


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    wave_pil = Image.open(WAVE_DIR / "waveform_1.png").convert("RGB")
    wave_in = wave_pil.resize((SIZE, SIZE), Image.LANCZOS)

    pipe = build_pipe()
    print(f"[load] done, vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    t0 = time.perf_counter()
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=PROMPT, device="cuda", do_classifier_free_guidance=False, max_sequence_length=256,
    )
    print(f"[encode-prompt] {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

    # Pre-encode input latents
    t0 = time.perf_counter()
    input_latents = encode_image_to_latents(pipe, wave_pil, SIZE)
    print(f"[encode-image-latents] {(time.perf_counter()-t0)*1000:.0f}ms, shape={input_latents.shape}", flush=True)

    results = {}

    # === A: pipeline baseline (per-frame VAE encode) ===
    print("\n### A: pipeline baseline (per-frame VAE encode) ###", flush=True)
    for w in range(N_WARMUP):
        tw = time.perf_counter()
        _ = pipe(prompt=None, prompt_embeds=prompt_embeds,
                 image=wave_in, strength=STRENGTH,
                 num_inference_steps=STEPS, guidance_scale=0.0,
                 height=SIZE, width=SIZE,
                 generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
        torch.cuda.synchronize()
        print(f"  warmup {w}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)
    times = []
    for i in range(N_BENCH):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pipe(prompt=None, prompt_embeds=prompt_embeds,
                   image=wave_in, strength=STRENGTH,
                   num_inference_steps=STEPS, guidance_scale=0.0,
                   height=SIZE, width=SIZE,
                   generator=torch.Generator("cuda").manual_seed(SEED + i)).images[0]
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        if i < 1: out.save(OUT_DIR / f"A_pipeline_i{i}.png")
    arr = np.array(times)
    results["A_pipeline"] = {"mean_ms": round(arr.mean(),1), "p95_ms": round(np.percentile(arr,95),1),
                             "min_ms": round(arr.min(),1), "fps": round(1000/arr.mean(),2)}
    print(f"  A: {results['A_pipeline']}", flush=True)

    # === B: sigma injection via pipeline latents= arg (skip per-frame encode) ===
    # The pipeline accepts `latents=` for pre-noised init. We compute noisy latents
    # once, feed them in every call. VAE encode step is skipped internally.
    print("\n### B: pre-encoded latents (pipeline latents= arg) ###", flush=True)
    # Build noisy latents per frame: x_t = (1-s) * x + s * noise
    def noisy_latents(seed):
        g = torch.Generator("cuda").manual_seed(seed)
        noise = torch.randn(input_latents.shape, generator=g,
                            dtype=input_latents.dtype, device="cuda")
        return (1.0 - STRENGTH) * input_latents + STRENGTH * noise

    # When passing custom `latents=`, we still need the pipeline scheduler to use
    # sigmas from strength=STRENGTH trajectory. Pass strength so it picks right start.
    for w in range(N_WARMUP):
        tw = time.perf_counter()
        nl = noisy_latents(SEED + 1000 + w)
        _ = pipe(prompt=None, prompt_embeds=prompt_embeds,
                 image=wave_in,       # pipeline still wants an image arg; ignored when latents= given
                 latents=nl,
                 strength=STRENGTH,
                 num_inference_steps=STEPS, guidance_scale=0.0,
                 height=SIZE, width=SIZE,
                 generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
        torch.cuda.synchronize()
        print(f"  warmup {w}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)
    times = []
    for i in range(N_BENCH):
        nl = noisy_latents(SEED + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pipe(prompt=None, prompt_embeds=prompt_embeds,
                   image=wave_in,
                   latents=nl,
                   strength=STRENGTH,
                   num_inference_steps=STEPS, guidance_scale=0.0,
                   height=SIZE, width=SIZE,
                   generator=torch.Generator("cuda").manual_seed(SEED + i)).images[0]
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        if i < 1: out.save(OUT_DIR / f"B_latents_i{i}.png")
    arr = np.array(times)
    results["B_latents"] = {"mean_ms": round(arr.mean(),1), "p95_ms": round(np.percentile(arr,95),1),
                            "min_ms": round(arr.min(),1), "fps": round(1000/arr.mean(),2)}
    print(f"  B: {results['B_latents']}", flush=True)

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done]", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v['mean_ms']}ms p95={v['p95_ms']} min={v['min_ms']} → {v['fps']} fps", flush=True)


if __name__ == "__main__":
    main()
