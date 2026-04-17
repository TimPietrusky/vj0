#!/usr/bin/env python3
"""
Rigorous proof that the waveform input actually INFLUENCES generation when
the prompt has nothing to do with a waveform shape.

Matrix:
  inputs: waveform_1, waveform_2, blank (black control)
  prompts: beach house, fantasy castle, golden retriever, ramen, cityscape — all
           things with nothing waveform-shaped about them
  strengths: 0.50, 0.60, 0.70, 0.80 — sweep the range where prompt dominates but
             input should still tilt the composition
  seed: fixed so wave1-vs-wave2-vs-blank differences isolate to the input

Success criterion: for each prompt/strength, wave1 output and wave2 output
should be visibly different from each other AND from the blank control, with
the prompt's content (a beach house, a dog, etc.) clearly rendered.
"""
from pathlib import Path
import time
import torch
import numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SIZE = 512
N_STEPS = 4
SEED = 42
OUT_DIR = Path("/workspace/flux2-unrelated")

PROMPTS = {
    "beach":    "a colorful wooden house at the beach at sunset, palm trees, warm tones, photograph",
    "castle":   "fantasy castle on a mountain, bright blue sky, detailed towers, oil painting",
    "dog":      "a golden retriever puppy sitting in grass, studio light, sharp focus, photograph",
    "ramen":    "macro photograph of a bowl of steaming ramen, garnish, wooden table, warm light",
    "city":     "vibrant neon cyberpunk city street at night, rain, reflections, wide angle",
}
STRENGTHS = [0.50, 0.60, 0.70, 0.80]


def pil_to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


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

    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)

    # Encode prompts
    embeds = {}
    for tag, prompt in PROMPTS.items():
        res = pipe.encode_prompt(prompt=prompt, device="cuda",
                                 num_images_per_prompt=1, max_sequence_length=128)
        embeds[tag] = res[0] if isinstance(res, tuple) else res
    print(f"[encode] {len(embeds)} prompts cached", flush=True)

    # Prepare inputs
    def encode(img):
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.LANCZOS)
        t = pil_to_tensor(img).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    wave1 = Image.open("/workspace/waveforms/waveform_1.png").convert("RGB")
    wave2 = Image.open("/workspace/waveforms/waveform_2.png").convert("RGB")
    blank = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    inputs = {"wave1": encode(wave1), "wave2": encode(wave2), "blank": encode(blank)}
    print(f"[encode_img] {len(inputs)} inputs cached", flush=True)

    def run(lat, prompt_tag, strength, seed=SEED):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device="cuda")
        noisy = (1 - strength) * lat + strength * noise
        sigmas = np.linspace(strength, 0.0, N_STEPS).tolist()
        return pipe(image=None, prompt=None, prompt_embeds=embeds[prompt_tag],
                    latents=noisy, sigmas=sigmas,
                    height=SIZE, width=SIZE, num_inference_steps=N_STEPS,
                    generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]

    # warmup (triggers compile)
    print("[warmup] 3 calls", flush=True)
    for w in range(3):
        tw = time.perf_counter()
        _ = run(inputs["wave1"], "beach", 0.70)
        torch.cuda.synchronize()
        print(f"  {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)

    # Also run pure text2img baselines for reference
    for tag in PROMPTS:
        out = pipe(image=None, prompt=None, prompt_embeds=embeds[tag],
                   height=SIZE, width=SIZE, num_inference_steps=N_STEPS,
                   generator=torch.Generator(device="cuda").manual_seed(SEED)).images[0]
        out.save(OUT_DIR / f"{tag}_t2i.png")
        print(f"[t2i] {tag} saved", flush=True)

    # Full matrix
    print("\n[matrix] 5 prompts × 3 inputs × 4 strengths = 60 generations", flush=True)
    for tag in PROMPTS:
        for s in STRENGTHS:
            for in_name, in_lat in inputs.items():
                t0 = time.perf_counter()
                out = run(in_lat, tag, s)
                torch.cuda.synchronize()
                out.save(OUT_DIR / f"{tag}_{in_name}_s{int(s*100):03d}.png")
                print(f"  {tag} {in_name} s={s}: {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
