#!/usr/bin/env python3
"""
v3 optimization bench. Builds on lessons from v1/v2:
- pre-encoded prompt is the single biggest win (-85ms)
- channels_last is a wash on Flux (linear-heavy), skip it
- mode="reduce-overhead" (CUDA graphs) crashes on PEFT-wrapped layers
- use mode="max-autotune-no-cudagraphs" for the speed without the crash

Stages (all on top of pre-encoded prompt at max_seq_len=128):
  1. preencoded_baseline
  2. + compile(transformer, max-autotune-no-cudagraphs)
  3. + compile(vae.decoder, max-autotune-no-cudagraphs)
  4. + fp8 weight-only quant (Float8WeightOnlyConfig)
  5. fp8 dynamic activation+weight quant (true fp8 matmul on Blackwell)
"""
import argparse, json, math, time, traceback
from pathlib import Path

import torch
from PIL import Image, ImageDraw

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
COMPILE_MODE = "default"  # max-autotune produces thousands of triton OOM warnings on Blackwell sm_120 (smaller smem); default mode uses safe heuristics that compile cleanly.


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


def bench(pipe, image, size, prompt_embeds, runs, warmup, seed=42):
    for w in range(warmup):
        _ = pipe(image=image, prompt_embeds=prompt_embeds,
                 height=size, width=size, num_inference_steps=4,
                 generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
        torch.cuda.synchronize()
    lats = []
    for r in range(runs):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = pipe(image=image, prompt_embeds=prompt_embeds,
                   height=size, width=size, num_inference_steps=4,
                   generator=torch.Generator(device="cuda").manual_seed(seed + r)).images[0]
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)
    return {
        "lats_ms": [round(x, 1) for x in lats],
        "mean_ms": round(sum(lats) / len(lats), 1),
        "p50_ms": round(percentile(lats, 50), 1),
        "p95_ms": round(percentile(lats, 95), 1),
        "min_ms": round(min(lats), 1),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "_image": out,
    }


def stage(name, fn, results):
    print(f"\n{'='*60}\n[STAGE] {name}\n{'='*60}", flush=True)
    t0 = time.perf_counter()
    try:
        out = fn()
        out["setup_s"] = round(time.perf_counter() - t0, 1)
        img = out.pop("_image", None)
        results[name] = out
        print(f"[STAGE] {name} OK ({out['setup_s']}s)  "
              f"mean={out['mean_ms']}ms  min={out['min_ms']}ms  "
              f"p95={out['p95_ms']}ms  vram={out['vram_gb']}GB", flush=True)
        return img
    except Exception as e:
        print(f"[STAGE] {name} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        traceback.print_exc()
        results[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/flux2-opt3-out")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompt", default="cyberpunk neon abstract waveform, vibrant colors, glowing edges")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
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

    print(f"[load] {KLEIN_REPO}", flush=True)
    t0 = time.perf_counter()
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    print(f"[load] ready in {time.perf_counter()-t0:.1f}s  "
          f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    image = make_input_image(args.size); image.save(out_dir / "input.png")

    print("[encode] pre-encoding prompt...", flush=True)
    res = pipe.encode_prompt(prompt=args.prompt, device="cuda",
                             num_images_per_prompt=1, max_sequence_length=args.max_seq_len)
    prompt_embeds = res[0] if isinstance(res, tuple) else res
    print(f"[encode] prompt_embeds shape={tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}",
          flush=True)

    results = {}

    img1 = stage("1_preencoded_baseline",
                 lambda: bench(pipe, image, args.size, prompt_embeds, args.runs, args.warmup),
                 results)
    if img1: img1.save(out_dir / "1_preencoded.png")

    def s2():
        print(f"[2] compiling transformer mode={COMPILE_MODE}", flush=True)
        pipe.transformer = torch.compile(
            pipe.transformer, mode=COMPILE_MODE, fullgraph=False, dynamic=False
        )
        return bench(pipe, image, args.size, prompt_embeds, args.runs, max(args.warmup, 5))

    img2 = stage("2_compile_transformer", s2, results)
    if img2: img2.save(out_dir / "2_compile_transformer.png")

    def s3():
        print(f"[3] compiling vae.decoder mode={COMPILE_MODE}", flush=True)
        pipe.vae.decoder = torch.compile(
            pipe.vae.decoder, mode=COMPILE_MODE, fullgraph=False, dynamic=False
        )
        return bench(pipe, image, args.size, prompt_embeds, args.runs, max(args.warmup, 4))

    img3 = stage("3_compile_vae", s3, results)
    if img3: img3.save(out_dir / "3_compile_vae.png")

    def s4():
        from torchao.quantization import quantize_, Float8WeightOnlyConfig
        # unwrap compiled transformer for quant
        t = pipe.transformer
        if hasattr(t, "_orig_mod"):
            t = t._orig_mod; pipe.transformer = t
        print("[4] applying Float8WeightOnlyConfig", flush=True)
        quantize_(t, Float8WeightOnlyConfig())
        torch.cuda.empty_cache()
        print(f"[4] vram after quant: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
        print(f"[4] re-compiling mode={COMPILE_MODE}", flush=True)
        pipe.transformer = torch.compile(
            pipe.transformer, mode=COMPILE_MODE, fullgraph=False, dynamic=False
        )
        return bench(pipe, image, args.size, prompt_embeds, args.runs, max(args.warmup, 5))

    img4 = stage("4_fp8_weight_only", s4, results)
    if img4: img4.save(out_dir / "4_fp8_weight_only.png")

    def s5():
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
        t = pipe.transformer
        if hasattr(t, "_orig_mod"):
            t = t._orig_mod; pipe.transformer = t
        print("[5] applying Float8DynamicActivationFloat8WeightConfig (true fp8 matmul)", flush=True)
        # already weight-fp8 quantized — re-quant overrides with dyn-act version on a fresh copy
        # so reload the transformer to start from bf16 weights
        from diffusers import Flux2Transformer2DModel
        print("[5] reloading transformer in bf16 to re-quantize from scratch", flush=True)
        pipe.transformer = Flux2Transformer2DModel.from_pretrained(
            KLEIN_REPO, subfolder="transformer", torch_dtype=torch.bfloat16
        ).to("cuda")
        torch.cuda.empty_cache()
        quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
        torch.cuda.empty_cache()
        print(f"[5] vram after quant: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
        print(f"[5] compiling mode={COMPILE_MODE}", flush=True)
        pipe.transformer = torch.compile(
            pipe.transformer, mode=COMPILE_MODE, fullgraph=False, dynamic=False
        )
        return bench(pipe, image, args.size, prompt_embeds, args.runs, max(args.warmup, 5))

    img5 = stage("5_fp8_dyn_act_weight", s5, results)
    if img5: img5.save(out_dir / "5_fp8_dyn_act_weight.png")

    print(f"\n{'='*60}\nSUMMARY @ {args.size}x{args.size}, 4 steps\n{'='*60}", flush=True)
    print(f"{'stage':<28} {'mean':>9} {'min':>8} {'p95':>8} {'vram':>8} {'setup':>8}")
    print("-" * 75)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<28}  FAILED: {r['error'][:50]}")
        else:
            print(f"{name:<28} {r['mean_ms']:>7}ms {r['min_ms']:>7}ms "
                  f"{r['p95_ms']:>7}ms {r['vram_gb']:>6}GB {r.get('setup_s','-'):>6}s")

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
