#!/usr/bin/env python3
"""
Resolution sweep with the proven optimal stack:
  pre-encoded prompt + compile(transformer, default) + compile(vae.decoder, default)

Each resolution requires its own compile (dynamic=False), so setup is ~30s/res.
"""
import argparse, json, math, time
from pathlib import Path
import torch
from PIL import Image, ImageDraw

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"


def make_input_image(size: int) -> Image.Image:
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
        draw.rectangle([x0, y0, x0 + bar_w, y0 + h],
                       fill=(255, 80 + i * 12, 200 - i * 10))
    return img


def percentile(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/flux2-sweep-out")
    ap.add_argument("--resolutions", default="256,320,384,448,512")
    ap.add_argument("--prompt", default="cyberpunk neon abstract waveform, vibrant colors, glowing edges")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}",
          flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    t0 = time.perf_counter()
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    print(f"[load] {time.perf_counter()-t0:.1f}s  vram={torch.cuda.memory_allocated()/1e9:.2f}GB",
          flush=True)

    res = pipe.encode_prompt(prompt=args.prompt, device="cuda",
                             num_images_per_prompt=1, max_sequence_length=args.max_seq_len)
    prompt_embeds = res[0] if isinstance(res, tuple) else res
    print(f"[encode] {tuple(prompt_embeds.shape)} {prompt_embeds.dtype}", flush=True)

    print("[compile] transformer + vae.decoder (mode=default, dynamic=False)", flush=True)
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)

    sizes = [int(s) for s in args.resolutions.split(",")]
    results = {}
    for size in sizes:
        print(f"\n=== {size}x{size} ===", flush=True)
        img = make_input_image(size)
        img.save(out_dir / f"input_{size}.png")

        # warmup (also triggers per-shape compile)
        tw0 = time.perf_counter()
        for w in range(args.warmup):
            tw = time.perf_counter()
            _ = pipe(image=img, prompt_embeds=prompt_embeds,
                     height=size, width=size, num_inference_steps=4,
                     generator=torch.Generator(device="cuda").manual_seed(args.seed)).images[0]
            torch.cuda.synchronize()
            print(f"  warmup {w+1}/{args.warmup}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)
        warmup_total = time.perf_counter() - tw0

        lats = []
        for r in range(args.runs):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = pipe(image=img, prompt_embeds=prompt_embeds,
                       height=size, width=size, num_inference_steps=4,
                       generator=torch.Generator(device="cuda").manual_seed(args.seed + r)).images[0]
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t) * 1000)

        out.save(out_dir / f"out_{size}.png")
        results[size] = {
            "lats_ms": [round(x, 1) for x in lats],
            "mean_ms": round(sum(lats) / len(lats), 2),
            "p50_ms": round(percentile(lats, 50), 2),
            "p95_ms": round(percentile(lats, 95), 2),
            "min_ms": round(min(lats), 2),
            "max_ms": round(max(lats), 2),
            "fps_at_mean": round(1000 / (sum(lats) / len(lats)), 2),
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "warmup_total_s": round(warmup_total, 1),
        }
        r = results[size]
        print(f"  >>> mean={r['mean_ms']}ms p50={r['p50_ms']} p95={r['p95_ms']} "
              f"min={r['min_ms']} max={r['max_ms']} fps={r['fps_at_mean']}", flush=True)

    print(f"\n{'='*70}\nRESOLUTION SWEEP — 4 steps, bf16, compiled\n{'='*70}", flush=True)
    print(f"{'res':>6} {'mean':>10} {'p50':>10} {'p95':>10} {'min':>10} {'fps':>7}")
    print("-" * 70)
    for size, r in results.items():
        print(f"{size:>5}² {r['mean_ms']:>8}ms {r['p50_ms']:>8}ms {r['p95_ms']:>8}ms "
              f"{r['min_ms']:>8}ms {r['fps_at_mean']:>6}")

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
