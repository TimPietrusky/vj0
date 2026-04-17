#!/usr/bin/env python3
"""
Classic-style img2img on Flux2KleinKVPipeline via manual sigmas + noisy starting latents.

Why: the pipeline's native reference conditioning via `image=` turned out to be
too weak to drive the output shape in our tests (identical outputs whether a
waveform or blank was passed). This script bypasses that mechanism and does
real img2img the classic way:

  1. Encode input image to latents via VAE (skip add noise yet)
  2. Compute flow-matching sigmas for the shortened schedule starting at `strength`
  3. Mix: noisy_latents = (1 - sigma_start) * image_latents + sigma_start * noise
  4. Pass latents= + sigmas= to the pipeline; image= stays None (we don't want
     the Kontext-style reference path, we want the latents to *be* the image)

Strengths tested: 0.5, 0.65, 0.8, 0.9, 1.0 (1.0 == pure text2img reproduction baseline)
Input: real B&W waveform from the VJ app (/workspace/waveforms/waveform_1.png)
Prompt: same lightning prompt that failed the reference-conditioning test —
if classic img2img works we should see the bolt follow the horizontal
waveform contour at low strengths, and diverge at strength=1.0.
"""
from pathlib import Path
import math
import time
import torch
import numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SIZE = 512
SEED = 42
N_STEPS = 4
OUT_DIR = Path("/workspace/flux2-classic-img2img")

PROMPTS = [
    ("lightning", "a bright white lightning bolt against a pitch black night sky, dramatic, photographic, high contrast"),
    ("neon",      "a single thin glowing neon line on pure black background, minimalist"),
    ("river",     "aerial photograph of a silver river winding through dark terrain at night"),
]
STRENGTHS = [0.5, 0.65, 0.8, 0.9, 1.0]


def load_waveform(path: str, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return t


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    t0 = time.perf_counter()
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    print(f"[load] {time.perf_counter()-t0:.1f}s  vram={torch.cuda.memory_allocated()/1e9:.2f}GB",
          flush=True)

    wave = load_waveform("/workspace/waveforms/waveform_1.png", SIZE)
    wave.save(OUT_DIR / "input_wave.png")

    # Encode input image to VAE latents — replicate pipeline's _encode_vae_image behavior
    img_tensor = pil_to_tensor(wave).to("cuda", dtype=torch.bfloat16)
    # NOTE: the pipeline's VAE also runs a batch-norm rescaling inside _encode_vae_image.
    # We need to match that to get latents on the right scale. Replicate here:
    def encode_to_latents(x_chw_bf16):
        from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents
        lat = retrieve_latents(pipe.vae.encode(x_chw_bf16), sample_mode="argmax")
        lat = pipe._patchify_latents(lat)  # (1, C_pack, H//ps, W//ps)
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        bn_std = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(lat.device, lat.dtype)
        lat = (lat - bn_mean) / bn_std
        return lat

    image_latents_spatial = encode_to_latents(img_tensor)  # (1, C, H', W')
    print(f"[encode_img] image_latents spatial shape={tuple(image_latents_spatial.shape)} dtype={image_latents_spatial.dtype}",
          flush=True)

    # Pack into (1, seq_len, C) the way the pipeline expects for `latents=` arg.
    # prepare_latents returns a packed latents tensor of shape (batch, seq_len, channels).
    # We reuse pipe.prepare_latents to get the target shape, then overwrite with image latents.
    # Inspect: prepare_latents uses _pack_latents internally.
    B, C, H, W = image_latents_spatial.shape
    # seq_len = (H // 2) * (W // 2); packed channels = C * 4
    packed = image_latents_spatial.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, C, H/2, W/2, 2, 2)
    packed = packed.permute(0, 2, 3, 1, 4, 5).contiguous()           # (B, H/2, W/2, C, 2, 2)
    packed = packed.view(B, (H//2) * (W//2), C * 4)                  # (B, seq_len, C*4)
    print(f"[pack] packed latents shape={tuple(packed.shape)}", flush=True)

    # Sanity: ask pipeline to prepare a fresh random latents to verify shape match
    num_channels_latents = pipe.transformer.config.in_channels // 4
    fresh_latents, _ = pipe.prepare_latents(
        batch_size=1, num_latents_channels=num_channels_latents,
        height=SIZE, width=SIZE, dtype=torch.bfloat16, device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(SEED),
        latents=None,
    )
    print(f"[sanity] pipeline fresh latents shape={tuple(fresh_latents.shape)}", flush=True)
    if packed.shape != fresh_latents.shape:
        print(f"[WARN] packed {tuple(packed.shape)} != fresh {tuple(fresh_latents.shape)} — "
              f"padding/trunc channels to match", flush=True)
        # align channel dim if off
        if packed.shape[-1] < fresh_latents.shape[-1]:
            pad = fresh_latents.shape[-1] - packed.shape[-1]
            packed = torch.cat([packed, torch.zeros(*packed.shape[:-1], pad,
                                dtype=packed.dtype, device=packed.device)], dim=-1)
        elif packed.shape[-1] > fresh_latents.shape[-1]:
            packed = packed[..., :fresh_latents.shape[-1]]
        print(f"[align] packed now {tuple(packed.shape)}", flush=True)

    image_latents_packed = packed

    def img2img_generate(prompt, strength, seed=SEED):
        """Classic img2img: start from (1-s)*image + s*noise, denoise for `strength` fraction."""
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(image_latents_packed.shape, generator=generator,
                            dtype=image_latents_packed.dtype, device="cuda")
        # Flow matching mix: x_sigma = (1 - sigma) * x_clean + sigma * noise
        noisy = (1 - strength) * image_latents_packed + strength * noise
        # Custom sigmas for the truncated schedule: go from `strength` down to 0.
        sigmas = list(np.linspace(strength, 1.0 / N_STEPS, N_STEPS)) + [0.0]
        # Last element is appended by the scheduler; our N_STEPS sigmas drive it.
        sigmas = list(np.linspace(strength, strength / N_STEPS, N_STEPS))

        return pipe(
            image=None,
            prompt=prompt,
            latents=noisy,
            sigmas=sigmas,
            height=SIZE, width=SIZE,
            num_inference_steps=N_STEPS,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    for tag, prompt in PROMPTS:
        print(f"\n=== prompt: {tag} — {prompt!r}", flush=True)
        # Pure text2img (no image, no latents injected) as baseline
        t0 = time.perf_counter()
        t2i = pipe(image=None, prompt=prompt, height=SIZE, width=SIZE,
                   num_inference_steps=N_STEPS,
                   generator=torch.Generator(device="cuda").manual_seed(SEED)).images[0]
        torch.cuda.synchronize()
        t2i.save(OUT_DIR / f"out_{tag}_t2i.png")
        print(f"  t2i: {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

        for s in STRENGTHS:
            t0 = time.perf_counter()
            try:
                img = img2img_generate(prompt, strength=s)
                dt = (time.perf_counter() - t0) * 1000
                img.save(OUT_DIR / f"out_{tag}_s{int(s*100):03d}.png")
                print(f"  strength={s:.2f}: {dt:.0f}ms -> out_{tag}_s{int(s*100):03d}.png",
                      flush=True)
            except Exception as e:
                print(f"  strength={s:.2f}: FAILED {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()

    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
