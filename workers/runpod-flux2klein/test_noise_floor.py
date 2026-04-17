#!/usr/bin/env python3
"""Replicate SD-Turbo's 'noise floor' behavior in Klein.

Hypothesis: SD-Turbo at strength=1,N=1 succeeds because of its DDPM noise schedule —
even at t_max, sqrt(alpha_bar_t_max) ≈ 2% > 0, so the image always leaks 2% signal.
Klein flow-matching at sigma=1 is literally 0% image.

Test: Klein with VERY LOW image blend (tiny image bias) + FULL sigma schedule starting
at 1.0 (pure noise, all steps denoise). The per-step sigma list:
   sigmas = linspace(1.0, 0, N)
Blend strategy: noisy = alpha * image + (1-alpha) * noise, where alpha ∈ {0.005, 0.01, 0.02, 0.05, 0.10}
This is SD-Turbo's math applied inside flow-matching.

If this works, we get text2img output (since the model denoises from sigma=1) with
a subtle compositional bias from the image — exactly like SD-Turbo does.
"""
from pathlib import Path
import time, torch, numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SIZE = 512
SEED = 42
N_STEPS = 4
OUT_DIR = Path("/workspace/flux2-noise-floor")

PROMPTS = {
    "dog":    "a golden retriever puppy sitting in grass, studio light, photograph",
    "beach":  "colorful wooden house at the beach at sunset, palm trees, warm tones, photograph",
    "city":   "vibrant neon cyberpunk city street at night, rain, reflections",
}
ALPHAS = [0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30]


def pil2t(img):
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[init] {torch.cuda.get_device_name(0)}", flush=True)

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

    def encode(img):
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.LANCZOS)
        t = pil2t(img).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    wave1 = encode(Image.open("/workspace/waveforms/waveform_1.png").convert("RGB"))
    wave2 = encode(Image.open("/workspace/waveforms/waveform_2.png").convert("RGB"))
    blank = encode(Image.new("RGB", (SIZE, SIZE), (0, 0, 0)))
    inputs = {"wave1": wave1, "wave2": wave2, "blank": blank}

    embeds = {}
    for tag, p in PROMPTS.items():
        r = pipe.encode_prompt(prompt=p, device="cuda",
                               num_images_per_prompt=1, max_sequence_length=128)
        embeds[tag] = r[0] if isinstance(r, tuple) else r

    def run(lat, tag, alpha, seed=SEED):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device="cuda")
        # SD-Turbo-like blend: noisy = alpha * image + (1-alpha) * noise
        noisy = alpha * lat + (1 - alpha) * noise
        # Run full sigma schedule — model has all 4 steps to denoise from ~noise
        # sigma_start = 1-alpha so the scheduler sees this as "near-pure-noise"
        sigma_start = 1.0 - alpha
        sigmas = np.linspace(sigma_start, 0.0, N_STEPS).tolist()
        return pipe(image=None, prompt=None, prompt_embeds=embeds[tag],
                    latents=noisy, sigmas=sigmas,
                    height=SIZE, width=SIZE, num_inference_steps=N_STEPS,
                    generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]

    # warmup
    for w in range(2):
        _ = run(wave1, "dog", 0.02)
        torch.cuda.synchronize()
    print("[warmup] done", flush=True)

    # t2i reference (alpha=0)
    print("\n[t2i refs]", flush=True)
    for tag in PROMPTS:
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        sigmas = np.linspace(1.0, 0.0, N_STEPS).tolist()
        noise = torch.randn(wave1.shape, generator=gen, dtype=wave1.dtype, device="cuda")
        out = pipe(image=None, prompt=None, prompt_embeds=embeds[tag],
                   latents=noise, sigmas=sigmas,
                   height=SIZE, width=SIZE, num_inference_steps=N_STEPS,
                   generator=torch.Generator(device="cuda").manual_seed(SEED)).images[0]
        out.save(OUT_DIR / f"{tag}_t2i.png")

    # Matrix
    print("\n[matrix] 3 prompts × 3 inputs × 7 alphas = 63 generations", flush=True)
    for tag in PROMPTS:
        for a in ALPHAS:
            for in_name, lat in inputs.items():
                t0 = time.perf_counter()
                out = run(lat, tag, a)
                torch.cuda.synchronize()
                out.save(OUT_DIR / f"{tag}_{in_name}_a{int(a*1000):03d}.png")
                print(f"  {tag} {in_name} alpha={a:.3f}: {(time.perf_counter()-t0)*1000:.0f}ms",
                      flush=True)

    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
