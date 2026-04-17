#!/usr/bin/env python3
"""Resolution sweep of the winning sigma-injection img2img stack."""
import argparse, json, math, time
from pathlib import Path
import torch
import numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
N_STEPS = 4
SEED = 42
STRENGTH = 0.90
PROMPT = "a bright white lightning bolt against a pitch black night sky, dramatic, photographic, high contrast"
OUT_DIR = Path("/workspace/flux2-final-res")
RESOLUTIONS = [256, 384, 512]


def pil_to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def percentile(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


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

    print("[compile] transformer + vae.decoder", flush=True)
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)

    res = pipe.encode_prompt(prompt=PROMPT, device="cuda",
                             num_images_per_prompt=1, max_sequence_length=128)
    prompt_embeds = res[0] if isinstance(res, tuple) else res

    def encode_img(path, size):
        img = Image.open(path).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.LANCZOS)
        t = pil_to_tensor(img).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return img, (patch - m) / s

    def run(lat, size, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device="cuda")
        noisy = (1 - STRENGTH) * lat + STRENGTH * noise
        sigmas = np.linspace(STRENGTH, 0.0, N_STEPS).tolist()
        return pipe(image=None, prompt=None, prompt_embeds=prompt_embeds,
                    latents=noisy, sigmas=sigmas,
                    height=size, width=size, num_inference_steps=N_STEPS,
                    generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]

    results = {}
    for size in RESOLUTIONS:
        print(f"\n=== {size}×{size} ===", flush=True)
        img_pil, lat = encode_img("/workspace/waveforms/waveform_1.png", size)
        img_pil.save(OUT_DIR / f"input_{size}.png")

        # warmup (first call per resolution triggers recompile)
        for w in range(4):
            tw = time.perf_counter()
            out = run(lat, size, SEED)
            torch.cuda.synchronize()
            print(f"  warmup {w+1}/4: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)
        out.save(OUT_DIR / f"out_{size}.png")

        lats = []
        for r in range(10):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = run(lat, size, SEED + r)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t) * 1000)

        results[size] = {
            "mean_ms": round(sum(lats) / len(lats), 2),
            "p50_ms": round(percentile(lats, 50), 2),
            "p95_ms": round(percentile(lats, 95), 2),
            "min_ms": round(min(lats), 2),
            "max_ms": round(max(lats), 2),
            "fps": round(1000 / (sum(lats) / len(lats)), 2),
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        }
        r = results[size]
        print(f"  >>> mean={r['mean_ms']}ms p50={r['p50_ms']} p95={r['p95_ms']} min={r['min_ms']} "
              f"fps={r['fps']} vram={r['vram_gb']}GB", flush=True)

    print(f"\n{'='*70}\nSIGMA-INJECTION IMG2IMG @ s={STRENGTH}, {N_STEPS} steps\n{'='*70}", flush=True)
    print(f"{'res':>6} {'mean':>9} {'p50':>9} {'p95':>9} {'min':>9} {'fps':>7}")
    for size, r in results.items():
        print(f"{size:>5}² {r['mean_ms']:>7}ms {r['p50_ms']:>7}ms {r['p95_ms']:>7}ms "
              f"{r['min_ms']:>7}ms {r['fps']:>6}")

    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
