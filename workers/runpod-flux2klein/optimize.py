#!/usr/bin/env python3
"""
Stack-and-measure optimization bench for FLUX.2-klein-4B img2img.

Each stage applies one optimization on top of the previous and re-benchmarks.
Stages that fail are skipped; the run continues so we always get a final summary.

Stages:
  A. baseline (Flux2KleinKVPipeline + small decoder, 4 steps, prompt encoded per call)
  B. + pre-encoded prompt embeddings (no text encoder per frame)
  C. + channels_last memory format on transformer
  D. + torch.compile(transformer, mode="reduce-overhead")  -- triggers CUDA graphs
  E. + torch.compile(vae.decoder)
  F. + fp8 weight-only quant on transformer (torchao, Blackwell)
"""
import argparse
import json
import math
import os
import time
import traceback
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
        col = (255, 80 + i * 12, 200 - i * 10)
        draw.rectangle([x0, y0, x0 + bar_w, y0 + h], fill=col)
    return img


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bench(pipe, image, size, prompt, prompt_embeds, neg_embeds, runs=8, warmup=3, seed=42):
    """Run timed inference; if prompt_embeds is provided, skip text encoder.
    Klein is distilled (no CFG) and does not accept negative_prompt_embeds."""
    kwargs = dict(
        image=image,
        height=size,
        width=size,
        num_inference_steps=4,
    )
    if prompt_embeds is not None:
        kwargs["prompt_embeds"] = prompt_embeds
    else:
        kwargs["prompt"] = prompt

    for w in range(warmup):
        _ = pipe(
            **kwargs,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize()

    lats = []
    for r in range(runs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = pipe(
            **kwargs,
            generator=torch.Generator(device="cuda").manual_seed(seed + r),
        ).images[0]
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)

    return {
        "lats_ms": [round(x, 1) for x in lats],
        "mean_ms": round(sum(lats) / len(lats), 1),
        "p50_ms": round(percentile(lats, 50), 1),
        "p95_ms": round(percentile(lats, 95), 1),
        "min_ms": round(min(lats), 1),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "last_image": out,
    }


def stage(name, fn, results, *args, **kwargs):
    print(f"\n{'='*60}\n[STAGE] {name}\n{'='*60}", flush=True)
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
        out["setup_s"] = round(time.perf_counter() - t0, 1)
        results[name] = {k: v for k, v in out.items() if k != "last_image"}
        if "last_image" in out:
            results[name]["_image"] = out["last_image"]
        print(f"[STAGE] {name} OK in {out['setup_s']}s  "
              f"mean={out.get('mean_ms','-')}ms  min={out.get('min_ms','-')}ms",
              flush=True)
        return out
    except Exception as e:
        print(f"[STAGE] {name} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        results[name] = {"error": f"{type(e).__name__}: {e}"}
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/flux2-opt-out")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompt", default="cyberpunk neon abstract waveform, vibrant colors, glowing edges")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-fp8", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}",
          flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    print(f"[load] {KLEIN_REPO} ...", flush=True)
    t0 = time.perf_counter()
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    print(f"[load] base in {time.perf_counter()-t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[load] small decoder + cuda in {time.perf_counter()-t1:.1f}s  "
          f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    image = make_input_image(args.size)
    image.save(out_dir / "input.png")

    results = {}

    # ---- A: baseline ---- #
    a = stage(
        "A_baseline",
        bench, results, pipe, image, args.size, args.prompt, None, None,
        runs=args.runs, warmup=args.warmup, seed=args.seed,
    )
    if a and "last_image" in a:
        a["last_image"].save(out_dir / "A_baseline.png")

    # ---- B: pre-encoded prompt + low max_seq_len ---- #
    def stage_b():
        # Try pipe.encode_prompt; fall back to direct text encoder call.
        prompt_embeds = neg_embeds = None
        try:
            sig = pipe.encode_prompt
            # encode_prompt signature varies; just try common shapes
            res = pipe.encode_prompt(
                prompt=args.prompt,
                device=pipe._execution_device if hasattr(pipe, "_execution_device") else "cuda",
                num_images_per_prompt=1,
                max_sequence_length=args.max_seq_len,
            )
            if isinstance(res, tuple):
                prompt_embeds = res[0]
                if len(res) > 1 and isinstance(res[1], torch.Tensor):
                    neg_embeds = res[1]
            else:
                prompt_embeds = res
            print(f"[B] encoded via pipe.encode_prompt -> {tuple(prompt_embeds.shape)}", flush=True)
        except Exception as e:
            print(f"[B] encode_prompt failed: {e}; doing manual encode", flush=True)
            te = pipe.text_encoder
            tk = pipe.tokenizer
            tokens = tk(args.prompt, return_tensors="pt", truncation=True,
                        padding="max_length", max_length=args.max_seq_len).to("cuda")
            with torch.no_grad():
                prompt_embeds = te(**tokens).last_hidden_state.to(torch.bfloat16)
            print(f"[B] manual encoded -> {tuple(prompt_embeds.shape)}", flush=True)

        # Verify embeds path actually works once before timing
        _ = pipe(
            image=image, prompt_embeds=prompt_embeds,
            height=args.size, width=args.size, num_inference_steps=4,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]

        return bench(pipe, image, args.size, args.prompt, prompt_embeds, neg_embeds,
                     runs=args.runs, warmup=args.warmup, seed=args.seed)

    b = stage("B_preencoded_prompt", stage_b, results)
    if b and "last_image" in b:
        b["last_image"].save(out_dir / "B_preencoded.png")

    prompt_embeds = None
    neg_embeds = None
    # Re-encode for downstream stages (even if B's bench failed, the encode probably worked)
    try:
        res = pipe.encode_prompt(
            prompt=args.prompt, device="cuda",
            num_images_per_prompt=1, max_sequence_length=args.max_seq_len,
        )
        prompt_embeds = res[0] if isinstance(res, tuple) else res
        if isinstance(res, tuple) and len(res) > 1 and isinstance(res[1], torch.Tensor):
            neg_embeds = res[1]
    except Exception as e:
        print(f"[downstream] could not re-encode for stages C+: {e}", flush=True)

    # ---- C: channels_last on transformer ---- #
    def stage_c():
        try:
            pipe.transformer.to(memory_format=torch.channels_last)
            print("[C] transformer set to channels_last", flush=True)
        except Exception as e:
            print(f"[C] channels_last failed: {e}", flush=True)
        try:
            pipe.vae.to(memory_format=torch.channels_last)
            print("[C] vae set to channels_last", flush=True)
        except Exception as e:
            print(f"[C] vae channels_last failed (ok): {e}", flush=True)
        return bench(pipe, image, args.size, args.prompt, prompt_embeds, neg_embeds,
                     runs=args.runs, warmup=args.warmup, seed=args.seed)

    c = stage("C_channels_last", stage_c, results)
    if c and "last_image" in c:
        c["last_image"].save(out_dir / "C_channels_last.png")

    # ---- D: torch.compile transformer ---- #
    if not args.skip_compile:
        def stage_d():
            print("[D] compiling transformer (this can take 30-90s on first call)", flush=True)
            pipe.transformer = torch.compile(
                pipe.transformer, mode="reduce-overhead", fullgraph=False, dynamic=False
            )
            return bench(pipe, image, args.size, args.prompt, prompt_embeds, neg_embeds,
                         runs=args.runs, warmup=max(args.warmup, 5), seed=args.seed)

        d = stage("D_compile_transformer", stage_d, results)
        if d and "last_image" in d:
            d["last_image"].save(out_dir / "D_compile.png")

        # ---- E: torch.compile VAE decoder ---- #
        def stage_e():
            print("[E] compiling vae decoder", flush=True)
            try:
                pipe.vae.decoder = torch.compile(
                    pipe.vae.decoder, mode="reduce-overhead", fullgraph=False, dynamic=False
                )
            except Exception as e:
                print(f"[E] decoder compile failed, trying full vae: {e}", flush=True)
                pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=False, dynamic=False)
            return bench(pipe, image, args.size, args.prompt, prompt_embeds, neg_embeds,
                         runs=args.runs, warmup=max(args.warmup, 4), seed=args.seed)

        e = stage("E_compile_vae", stage_e, results)
        if e and "last_image" in e:
            e["last_image"].save(out_dir / "E_compile_vae.png")

    # ---- F: fp8 quant on transformer ---- #
    if not args.skip_fp8:
        def stage_f():
            print("[F] installing torchao if needed + applying fp8 weight-only quant", flush=True)
            try:
                import torchao
                print(f"[F] torchao {torchao.__version__}", flush=True)
            except ImportError:
                import subprocess
                subprocess.check_call(["pip", "install", "--break-system-packages", "-q", "torchao"])
                import torchao
                print(f"[F] installed torchao {torchao.__version__}", flush=True)

            from torchao.quantization import quantize_, Float8WeightOnlyConfig

            # Need to use the *uncompiled* transformer for quantization
            transformer = pipe.transformer
            if hasattr(transformer, "_orig_mod"):
                print("[F] unwrapping compiled transformer", flush=True)
                transformer = transformer._orig_mod
                pipe.transformer = transformer

            print("[F] applying Float8WeightOnlyConfig...", flush=True)
            quantize_(transformer, Float8WeightOnlyConfig())
            torch.cuda.empty_cache()
            print(f"[F] vram after quant: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

            # Re-compile after quant
            print("[F] re-compiling quantized transformer", flush=True)
            pipe.transformer = torch.compile(
                pipe.transformer, mode="reduce-overhead", fullgraph=False, dynamic=False
            )
            return bench(pipe, image, args.size, args.prompt, prompt_embeds, neg_embeds,
                         runs=args.runs, warmup=max(args.warmup, 5), seed=args.seed)

        f = stage("F_fp8_transformer", stage_f, results)
        if f and "last_image" in f:
            f["last_image"].save(out_dir / "F_fp8.png")

    # ---- summary ---- #
    print(f"\n{'='*60}\nSUMMARY @ {args.size}x{args.size}, 4 steps\n{'='*60}", flush=True)
    print(f"{'stage':<30} {'mean':>8} {'min':>8} {'p95':>8} {'vram':>8}")
    print("-" * 60)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<30} {'FAILED: ' + r['error'][:30]}")
            continue
        print(f"{name:<30} {r.get('mean_ms','-'):>7}ms {r.get('min_ms','-'):>7}ms "
              f"{r.get('p95_ms','-'):>7}ms {r.get('vram_gb','-'):>6}GB")

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "_image"} for k, v in results.items()},
        indent=2,
    ))
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
