#!/usr/bin/env python3
"""
FLUX.2-klein img2img smoke test.

Loads Flux2KleinKVPipeline (4-step distilled, with KV prompt cache),
swaps in FLUX.2-small-decoder as VAE, runs img2img on a synthetic
input image, times warm-state generation, and saves outputs.
"""
import os
import time
import math
import json
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"


def make_input_image(size: int) -> Image.Image:
    """Synthetic 'audio-reactive' input: radial gradient + bars (waveform-ish)."""
    img = Image.new("RGB", (size, size), (8, 8, 16))
    px = img.load()
    cx = cy = size / 2
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            d = math.sqrt(dx * dx + dy * dy) / (size / 2)
            r = int(max(0, 255 * (1 - d) * 0.9 + 30))
            g = int(max(0, 255 * (1 - d) * 0.3 + 20))
            b = int(max(0, 255 * (1 - d * 0.6) * 0.8 + 40))
            px[x, y] = (r, g, b)
    draw = ImageDraw.Draw(img)
    bar_w = max(2, size // 24)
    for i in range(12):
        h = int((0.3 + 0.6 * abs(math.sin(i * 0.7))) * size * 0.8)
        x0 = int(size * 0.1 + i * (size * 0.07))
        y0 = (size - h) // 2
        col = (255, 80 + i * 12, 200 - i * 10)
        draw.rectangle([x0, y0, x0 + bar_w, y0 + h], fill=col)
    return img


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/flux2-smoke-out")
    ap.add_argument("--resolutions", default="384,512")
    ap.add_argument("--prompt", default="cyberpunk neon abstract waveform, vibrant colors, glowing edges")
    ap.add_argument("--neg-prompt", default="blurry, low quality")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-small-decoder", action="store_true", default=True)
    ap.add_argument("--no-small-decoder", dest="use_small_decoder", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    t0 = time.perf_counter()
    print(f"[load] {KLEIN_REPO} (this downloads ~8GB on first run)...")
    pipe = Flux2KleinKVPipeline.from_pretrained(
        KLEIN_REPO,
        torch_dtype=torch.bfloat16,
    )
    print(f"[load] base pipeline loaded in {time.perf_counter()-t0:.1f}s")

    if args.use_small_decoder:
        t1 = time.perf_counter()
        print(f"[load] swapping VAE for {DECODER_REPO}")
        small_vae = AutoencoderKLFlux2.from_pretrained(
            DECODER_REPO, torch_dtype=torch.bfloat16
        )
        pipe.vae = small_vae
        print(f"[load] small decoder swapped in {time.perf_counter()-t1:.1f}s")

    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[load] pipeline on cuda. vram allocated: "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    # Save the synthetic input once (largest size)
    sizes = [int(s) for s in args.resolutions.split(",")]
    max_size = max(sizes)
    input_img_full = make_input_image(max_size)
    input_img_full.save(out_dir / "input.png")
    print(f"[input] wrote {out_dir/'input.png'} ({max_size}x{max_size})")

    results = {}
    for size in sizes:
        print(f"\n=== resolution {size}x{size} ===")
        input_img = input_img_full.resize((size, size), Image.LANCZOS)

        # warmup
        for w in range(args.warmup):
            tw = time.perf_counter()
            _ = pipe(
                image=input_img,
                prompt=args.prompt,
                height=size,
                width=size,
                num_inference_steps=4,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
            ).images[0]
            torch.cuda.synchronize()
            print(f"[warmup {w+1}/{args.warmup}] {(time.perf_counter()-tw)*1000:.0f} ms")

        # timed runs (same prompt → exercises KV cache)
        latencies = []
        for r in range(args.runs):
            torch.cuda.synchronize()
            tr = time.perf_counter()
            out = pipe(
                image=input_img,
                prompt=args.prompt,
                height=size,
                width=size,
                num_inference_steps=4,
                generator=torch.Generator(device="cuda").manual_seed(args.seed + r),
            ).images[0]
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - tr) * 1000
            latencies.append(dt_ms)
            out.save(out_dir / f"out_{size}_run{r}.png")
            print(f"[run {r+1}/{args.runs}] {dt_ms:.0f} ms")

        # one extra run with a *different* prompt, to confirm cache path
        tdp = time.perf_counter()
        _ = pipe(
            image=input_img,
            prompt="oil painting of stormy ocean at night, dramatic clouds",
            height=size,
            width=size,
            num_inference_steps=4,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]
        torch.cuda.synchronize()
        diff_prompt_ms = (time.perf_counter() - tdp) * 1000

        results[size] = {
            "latencies_ms": latencies,
            "mean_ms": sum(latencies) / len(latencies),
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "diff_prompt_ms": diff_prompt_ms,
            "vram_gb": torch.cuda.memory_allocated() / 1e9,
        }
        print(f"[stats] mean={results[size]['mean_ms']:.0f} ms  "
              f"p50={results[size]['p50_ms']:.0f}  p95={results[size]['p95_ms']:.0f}  "
              f"diff-prompt={diff_prompt_ms:.0f} ms  vram={results[size]['vram_gb']:.2f} GB")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {summary_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
