#!/usr/bin/env python3
"""
Final winning-stack bench:
  Flux2KleinKVPipeline + FLUX.2-small-decoder
  + pre-encoded prompt embeds (max_seq_len=128)
  + torch.compile(transformer, vae.decoder, mode="default")
  + manual img2img via sigma-injection: latents = (1-s)·image_latents + s·noise,
    sigmas = linspace(s, 0, N_STEPS)
  + native image= arg set to None (the pipeline's reference hook is too weak)

Matrix:
  inputs: waveform_1, waveform_2 (real B&W from VJ app)
  prompts: lightning, neon_line, river, crack, ink, fire
  strengths: 0.88, 0.90, 0.92
  resolutions: 512 (one compile) — can extend later
"""
import argparse, json, math, time
from pathlib import Path
import torch
import numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SIZE = 512
N_STEPS = 4
SEED = 42
OUT_DIR = Path("/workspace/flux2-final-bench")

PROMPTS = {
    "lightning":   "a bright white lightning bolt against a pitch black night sky, dramatic, photographic, high contrast",
    "neon_line":   "a single thin glowing neon line on pure black background, minimalist, studio light, cinematic",
    "river":       "aerial photograph of a silver river winding through dark terrain at night, moonlight",
    "crack":       "a bright white crack running across black ice, macro photo, dramatic contrast",
    "ink":         "flowing white ink streak on pitch black paper, macro photograph, high detail",
    "fire":        "a white-hot flame trail against black sky, long exposure photograph, cinematic",
}
WAVEFORMS = ["waveform_1", "waveform_2"]
STRENGTHS = [0.88, 0.90, 0.92]


def pil_to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def percentile(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


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
    print(f"[load] {time.perf_counter()-t0:.1f}s  vram={torch.cuda.memory_allocated()/1e9:.2f}GB",
          flush=True)

    # Compile the winning stack
    print("[compile] transformer + vae.decoder (mode=default, dynamic=False)", flush=True)
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)

    # Encode all prompts once
    prompt_embeds_cache = {}
    for tag, prompt in PROMPTS.items():
        res = pipe.encode_prompt(prompt=prompt, device="cuda",
                                 num_images_per_prompt=1, max_sequence_length=128)
        prompt_embeds_cache[tag] = res[0] if isinstance(res, tuple) else res
    print(f"[encode] cached {len(prompt_embeds_cache)} prompt embeds", flush=True)

    # Encode waveform inputs once
    def encode_waveform(path):
        img = Image.open(path).convert("RGB")
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.LANCZOS)
        t = pil_to_tensor(img).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        bn_std = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return img, (patch - bn_mean) / bn_std

    waveform_data = {}
    for w in WAVEFORMS:
        src = Path(f"/workspace/waveforms/{w}.png")
        img, lat = encode_waveform(src)
        img.save(OUT_DIR / f"_input_{w}.png")
        waveform_data[w] = lat
    print(f"[encode_img] cached {len(waveform_data)} waveform latents at {SIZE}x{SIZE}", flush=True)

    def run_one(wave_latent, prompt_embeds, strength, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(wave_latent.shape, generator=gen,
                            dtype=wave_latent.dtype, device="cuda")
        noisy = (1 - strength) * wave_latent + strength * noise
        sigmas = np.linspace(strength, 0.0, N_STEPS).tolist()
        return pipe(
            image=None, prompt=None, prompt_embeds=prompt_embeds,
            latents=noisy, sigmas=sigmas,
            height=SIZE, width=SIZE, num_inference_steps=N_STEPS,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    # Warmup (triggers compile for first call)
    print("[warmup] 4 runs at first config (triggers compile)", flush=True)
    for w in range(4):
        tw = time.perf_counter()
        _ = run_one(waveform_data[WAVEFORMS[0]], prompt_embeds_cache[list(PROMPTS)[0]],
                    strength=STRENGTHS[1], seed=SEED)
        torch.cuda.synchronize()
        print(f"  warmup {w+1}/4: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)

    # The full matrix, timed
    print("\n[matrix] running full wave x prompt x strength combinations", flush=True)
    results = {}
    lats = []
    for w in WAVEFORMS:
        for tag in PROMPTS:
            for s in STRENGTHS:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                img = run_one(waveform_data[w], prompt_embeds_cache[tag], s, seed=SEED)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                lats.append(dt)
                key = f"{w}_{tag}_s{int(s*100):03d}"
                img.save(OUT_DIR / f"{key}.png")
                results[key] = round(dt, 1)
                print(f"  {key}: {dt:.0f}ms", flush=True)

    # Stats across matrix
    summary = {
        "config": f"Klein + small-decoder + compile(T,VAE.dec) + sigma-inject img2img, bf16, {N_STEPS} steps, {SIZE}×{SIZE}",
        "matrix_runs": len(lats),
        "mean_ms": round(sum(lats) / len(lats), 2),
        "p50_ms": round(percentile(lats, 50), 2),
        "p95_ms": round(percentile(lats, 95), 2),
        "min_ms": round(min(lats), 2),
        "max_ms": round(max(lats), 2),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "latencies_ms": results,
    }
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}", flush=True)
    for k, v in summary.items():
        if k != "latencies_ms":
            print(f"  {k}: {v}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
