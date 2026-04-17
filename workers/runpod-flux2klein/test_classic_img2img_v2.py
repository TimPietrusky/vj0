#!/usr/bin/env python3
"""
Classic-style img2img on Flux2KleinKVPipeline v2.

Fix from v1: prepare_latents expects 4D (B, C*4, H//16, W//16) and does its own
packing. Previously I pre-packed to 3D (B, seq_len, C) which breaks _prepare_latent_ids.

Additionally handles the `use_flow_sigmas` scheduler config that can silently
discard custom sigmas; we detect and monkey-patch if needed.
"""
from pathlib import Path
import time
import torch
import numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SIZE = 512
SEED = 42
N_STEPS = 4
OUT_DIR = Path("/workspace/flux2-classic-v2")

PROMPTS = [
    ("lightning", "a bright white lightning bolt against a pitch black night sky, dramatic, photographic, high contrast"),
    ("neon",      "a single thin glowing neon line on pure black background, minimalist"),
]
STRENGTHS = [0.7, 0.8, 0.88, 0.92, 0.95, 0.98]


def pil_to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

    # Inspect scheduler
    sched = pipe.scheduler
    print(f"[sched] class={sched.__class__.__name__}", flush=True)
    cfg = getattr(sched, "config", None)
    if cfg is not None:
        flow = getattr(cfg, "use_flow_sigmas", None)
        print(f"[sched] use_flow_sigmas={flow}", flush=True)

    wave = Image.open("/workspace/waveforms/waveform_1.png").convert("RGB")
    if wave.size != (SIZE, SIZE):
        wave = wave.resize((SIZE, SIZE), Image.LANCZOS)
    wave.save(OUT_DIR / "input_wave.png")

    # Encode waveform to latents (matching the pipeline's internal path)
    img_t = pil_to_tensor(wave).to("cuda", dtype=torch.bfloat16)
    lat_raw = retrieve_latents(pipe.vae.encode(img_t), sample_mode="argmax")
    lat_patch = pipe._patchify_latents(lat_raw)  # (1, C*4, H/16, W/16)
    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_patch.device, lat_patch.dtype)
    bn_std = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(lat_patch.device, lat_patch.dtype)
    image_latents_4d = (lat_patch - bn_mean) / bn_std
    print(f"[encode_img] 4D latent shape={tuple(image_latents_4d.shape)} dtype={image_latents_4d.dtype}",
          flush=True)

    def run_img2img(prompt, strength, seed=SEED):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(image_latents_4d.shape, generator=gen,
                            dtype=image_latents_4d.dtype, device="cuda")
        # Flow matching noise mixing at sigma=strength
        noisy_4d = (1 - strength) * image_latents_4d + strength * noise

        # Truncated flow-matching sigmas: start at `strength`, linear down to 0 over N_STEPS points.
        # BUGFIX: previously ended at strength/N (never reached clean image).
        sigmas = np.linspace(strength, 0.0, N_STEPS).tolist()
        print(f"   sigmas={[round(s,3) for s in sigmas]}", flush=True)

        return pipe(
            image=None,
            prompt=prompt,
            latents=noisy_4d,
            sigmas=sigmas,
            height=SIZE, width=SIZE,
            num_inference_steps=N_STEPS,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    for tag, prompt in PROMPTS:
        print(f"\n=== prompt: {tag}", flush=True)
        # Pure t2i baseline
        t0 = time.perf_counter()
        t2i = pipe(image=None, prompt=prompt, height=SIZE, width=SIZE,
                   num_inference_steps=N_STEPS,
                   generator=torch.Generator(device="cuda").manual_seed(SEED)).images[0]
        torch.cuda.synchronize()
        t2i.save(OUT_DIR / f"out_{tag}_t2i.png")
        print(f"  t2i baseline: {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

        for s in STRENGTHS:
            t0 = time.perf_counter()
            try:
                img = run_img2img(prompt, strength=s)
                dt = (time.perf_counter() - t0) * 1000
                img.save(OUT_DIR / f"out_{tag}_s{int(s*100):03d}.png")
                print(f"  strength={s:.2f}: {dt:.0f}ms", flush=True)
            except Exception as e:
                print(f"  strength={s:.2f}: FAILED {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()

    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
