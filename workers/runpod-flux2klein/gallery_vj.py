#!/usr/bin/env python3
"""Real VJ-relevant gallery at the winning config (2-step, 512², compiled alpha-blend img2img).

Diverse prompts a VJ would actually use — not lightning. Plus two alpha values to show
the range: α=0.05 = subtle SDXL-turbo-style bias, α=0.15 = clearer input influence.
Seed varies across prompts to avoid seed-locked lookalikes.
"""
from pathlib import Path
import time, torch, numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SIZE = 512
N_STEPS = 2
OUT_DIR = Path("/workspace/flux2-gallery-vj")

# VJ-relevant prompts, mixing moods/styles
PROMPTS = {
    "beach_sunset":     "a beach at sunset, palm trees silhouettes, warm orange sky, photographic",
    "forest_misty":     "misty pine forest in early morning, sunlight filtering through fog, cinematic",
    "neon_city":        "cyberpunk tokyo street at night, neon signs, rain reflections, wide angle",
    "ocean_wave":       "crashing ocean wave, spray, aerial view, dramatic, sony a7",
    "galaxy":           "spiral galaxy with pink and blue nebula, dust clouds, deep space, astrophotography",
    "abstract_ink":     "black ink dropping into water, slow motion, white background, macro photography",
    "mountain_peak":    "alpine mountain peak at golden hour, snow, dramatic light, landscape photography",
    "neon_portrait":    "close-up portrait of a woman lit by neon pink and cyan, moody, cinematic",
    "underwater":       "underwater caustics, sun rays piercing blue water, silhouettes of fish, dreamy",
    "aurora":           "aurora borealis over snowy mountains, green and purple sky, long exposure",
    "flame":            "flames dancing in the dark, high speed photograph, black background",
    "waterfall":        "lush tropical waterfall in a rainforest, long exposure, silky water",
}

ALPHAS = [0.05, 0.15]
WAVEFORMS = ["waveform_1", "waveform_2"]


def pil2t(img):
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

    t0 = time.perf_counter()
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
    print(f"[load+compile stubs] {time.perf_counter()-t0:.1f}s", flush=True)

    # Encode prompts once
    embeds = {}
    for tag, p in PROMPTS.items():
        r = pipe.encode_prompt(prompt=p, device="cuda",
                               num_images_per_prompt=1, max_sequence_length=64)
        embeds[tag] = r[0] if isinstance(r, tuple) else r

    # Load waveforms
    wavs = {}
    for wv in WAVEFORMS:
        img = Image.open(f"/workspace/waveforms/{wv}.png").convert("RGB")
        wavs[wv] = img.resize((SIZE, SIZE), Image.LANCZOS)
        wavs[wv].save(OUT_DIR / f"_input_{wv}.png")

    def encode_img(img_pil):
        t = pil2t(img_pil).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    # Pre-encode waveform latents (waveform doesn't change during gallery)
    wav_lats = {k: encode_img(v) for k, v in wavs.items()}

    def run(wav_lat, embeds, alpha, n_steps, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(wav_lat.shape, generator=gen, dtype=wav_lat.dtype, device="cuda")
        noisy = alpha * wav_lat + (1 - alpha) * noise
        sigmas = np.linspace(1 - alpha, 0.0, n_steps).tolist()
        return pipe(image=None, prompt=None, prompt_embeds=embeds,
                    latents=noisy, sigmas=sigmas,
                    height=SIZE, width=SIZE, num_inference_steps=n_steps,
                    generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]

    # Warmup (triggers first-call compile)
    print("[warmup] 3 calls", flush=True)
    for w in range(3):
        tw = time.perf_counter()
        _ = run(wav_lats["waveform_1"], embeds["beach_sunset"], 0.10, N_STEPS, 42)
        torch.cuda.synchronize()
        print(f"  {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)

    # Also do a few t2i baselines (alpha=0 = pure prompt, no input)
    print("\n[t2i references]", flush=True)
    for tag in PROMPTS:
        gen = torch.Generator(device="cuda").manual_seed(hash(tag) & 0xFFFF)
        noise = torch.randn(wav_lats["waveform_1"].shape, generator=gen,
                            dtype=wav_lats["waveform_1"].dtype, device="cuda")
        sigmas = np.linspace(1.0, 0.0, N_STEPS).tolist()
        out = pipe(image=None, prompt=None, prompt_embeds=embeds[tag],
                   latents=noise, sigmas=sigmas,
                   height=SIZE, width=SIZE, num_inference_steps=N_STEPS,
                   generator=torch.Generator(device="cuda").manual_seed(hash(tag) & 0xFFFF)).images[0]
        out.save(OUT_DIR / f"{tag}_t2i.png")
        print(f"  {tag}: saved", flush=True)

    # Gallery matrix
    total = len(PROMPTS) * len(ALPHAS) * len(WAVEFORMS)
    print(f"\n[gallery] {total} generations", flush=True)
    lats = []
    for tag in PROMPTS:
        seed = hash(tag) & 0xFFFF
        for alpha in ALPHAS:
            for wv in WAVEFORMS:
                t = time.perf_counter()
                img = run(wav_lats[wv], embeds[tag], alpha, N_STEPS, seed)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t) * 1000
                lats.append(dt)
                img.save(OUT_DIR / f"{tag}_{wv}_a{int(alpha*100):03d}.png")
                print(f"  {tag} {wv} α={alpha}: {dt:.0f}ms", flush=True)

    print(f"\n[stats] mean={sum(lats)/len(lats):.1f}ms  min={min(lats):.1f}ms  max={max(lats):.1f}ms",
          flush=True)
    print(f"[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
