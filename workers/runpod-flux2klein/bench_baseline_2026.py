#!/usr/bin/env python3
"""Phase 0 baseline — production winning config.

Pins the current shipping configuration as the comparison row for all later
optimization phases (Pruna, TeaCache, FORA, ...). This is exactly what
inference_server.py runs in prod:

  Flux2KleinKVPipeline + FLUX.2-small-decoder, bf16
  + torch.compile(transformer + vae.encoder + vae.decoder, mode="default")
  + alpha-blend img2img (alpha=0.10)
  + 2 inference steps (sweet-spot)

Output:
  /workspace/bench-2026-04-30/baseline/summary.json
  /workspace/bench-2026-04-30/baseline/{config}.png  (last frame of each cell)

Bench includes the VAE encode of the input in the timed region (real per-frame
cost for live-VJ workloads). Each cell: 5 warmup iters, then 20 timed iters.
"""
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
WAVE_PATH = "/workspace/waveforms/waveform_1.png"
OUT_DIR = Path("/workspace/bench-2026-04-30/baseline")
SEED = 42
ALPHA = 0.10
MAX_SEQ_LEN = 64
PROMPT = (
    "a bright white lightning bolt against a pitch black night sky, "
    "dramatic, photographic, high contrast"
)
WARMUP_ITERS = 5
TIMED_ITERS = 20


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pil2t(img):
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def build_runner(pipe, wave_pil, prompt_embeds, alpha, n_steps, size):
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

    def encode_img(img_pil):
        img = img_pil if img_pil.size == (size, size) else img_pil.resize((size, size), Image.LANCZOS)
        t = pil2t(img).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    def run_one(seed):
        lat = encode_img(wave_pil)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device="cuda")
        noisy = alpha * lat + (1 - alpha) * noise
        sigmas = np.linspace(1 - alpha, 0.0, n_steps).tolist()
        return pipe(
            image=None, prompt=None, prompt_embeds=prompt_embeds,
            latents=noisy, sigmas=sigmas,
            height=size, width=size, num_inference_steps=n_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    return run_one


def bench_cell(pipe, wave_pil, prompt_embeds, alpha, size, n_steps):
    print(f"\n--- cell: {size}² / {n_steps}-step / α={alpha} ---", flush=True)
    runner = build_runner(pipe, wave_pil, prompt_embeds, alpha, n_steps, size)

    for w in range(WARMUP_ITERS):
        t0 = time.perf_counter()
        _ = runner(SEED)
        torch.cuda.synchronize()
        print(f"  warmup {w+1}/{WARMUP_ITERS}: {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

    lats = []
    last_img = None
    for r in range(TIMED_ITERS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        last_img = runner(SEED + r)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)

    out = {
        "resolution": f"{size}x{size}",
        "size_px": size,
        "steps": n_steps,
        "alpha": alpha,
        "dtype": "bfloat16",
        "compile": "default (transformer+vae.encoder+vae.decoder)",
        "cache_method": "none",
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "mean_ms": round(sum(lats) / len(lats), 2),
        "p50_ms": round(percentile(lats, 50), 2),
        "p95_ms": round(percentile(lats, 95), 2),
        "p99_ms": round(percentile(lats, 99), 2),
        "min_ms": round(min(lats), 2),
        "max_ms": round(max(lats), 2),
        "fps_mean": round(1000 / (sum(lats) / len(lats)), 2),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }
    print(
        f"  result: mean={out['mean_ms']}ms p95={out['p95_ms']}ms "
        f"fps={out['fps_mean']} vram={out['vram_gb']}GB",
        flush=True,
    )
    return out, last_img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}", flush=True)
    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    import diffusers, transformers
    print(f"[init] diffusers={diffusers.__version__} transformers={transformers.__version__}", flush=True)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    print(f"[load] {KLEIN_REPO}", flush=True)
    t0 = time.perf_counter()
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    print("[compile] transformer + vae.encoder + vae.decoder (mode=default)", flush=True)
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)

    wave_pil = Image.open(WAVE_PATH).convert("RGB")

    print(f"[prompt] encoding (max_seq_len={MAX_SEQ_LEN})", flush=True)
    r = pipe.encode_prompt(
        prompt=PROMPT, device="cuda",
        num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN,
    )
    prompt_embeds = r[0] if isinstance(r, tuple) else r
    print(f"[prompt] embeds shape={tuple(prompt_embeds.shape)}", flush=True)

    # The matrix: cells are deliberately small so this finishes in ~5-7 min.
    cells = [
        (256, 2),
        (256, 3),
        (384, 2),
        (384, 3),
        (512, 2),
        (512, 3),
    ]

    results = []
    for size, n_steps in cells:
        cell_name = f"res{size}_step{n_steps}"
        try:
            out, img = bench_cell(pipe, wave_pil, prompt_embeds, ALPHA, size, n_steps)
            out["name"] = cell_name
            results.append(out)
            if img is not None:
                img.save(OUT_DIR / f"{cell_name}.png")
        except Exception as e:
            print(f"[FAIL] {cell_name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            results.append({
                "name": cell_name,
                "resolution": f"{size}x{size}",
                "steps": n_steps,
                "error": f"{type(e).__name__}: {str(e)[:300]}",
            })

    summary = {
        "date": "2026-04-30",
        "phase": "0_baseline",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "config": {
            "pipeline": "Flux2KleinKVPipeline",
            "decoder": "FLUX.2-small-decoder",
            "dtype": "bfloat16",
            "compile_mode": "default",
            "compiled_modules": ["transformer", "vae.encoder", "vae.decoder"],
            "alpha": ALPHA,
            "max_seq_len": MAX_SEQ_LEN,
            "prompt": PROMPT,
        },
        "cells": results,
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*78}\nBASELINE SUMMARY\n{'='*78}", flush=True)
    print(f"{'cell':<22} {'mean':>10} {'p95':>10} {'p99':>10} {'fps':>8} {'vram':>8}")
    print("-" * 78)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<22}  FAIL: {r['error'][:50]}")
        else:
            print(
                f"{r['name']:<22} "
                f"{r['mean_ms']:>8.2f}ms "
                f"{r['p95_ms']:>8.2f}ms "
                f"{r['p99_ms']:>8.2f}ms "
                f"{r['fps_mean']:>7.2f} "
                f"{r['vram_gb']:>6.2f}GB"
            )

    print(f"\n[done] {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
