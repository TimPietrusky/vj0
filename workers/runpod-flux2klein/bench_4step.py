#!/usr/bin/env python3
"""Phase 0b + 3b — 4-step baselines (production target).

User runs FLUX.2 Klein at **4 inference steps** in live shows for quality.
Earlier 2-step benches are still archived but don't reflect production.

This bench produces clean 4-step rows for both:
  - bf16 + compile (control / "what we ship today at 4 steps")
  - fp8_dynamic_act + compile (the Phase 3 winning quantization)

At 256² / 384² / 512², 5 warmup + 20 timed iters per cell, alpha=0.10.

Output: /workspace/bench-2026-04-30/4step/summary.json
"""
import gc
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
OUT_DIR = Path("/workspace/bench-2026-04-30/4step")
SEED = 42
ALPHA = 0.10
MAX_SEQ_LEN = 64
PROMPT = (
    "a bright white lightning bolt against a pitch black night sky, "
    "dramatic, photographic, high contrast"
)
WARMUP_ITERS = 5
TIMED_ITERS = 20
N_STEPS = 4
SIZES = [256, 384, 512]


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pil2t(img):
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def filter_fn(module, fqn):
    if not isinstance(module, torch.nn.Linear):
        return False
    if "transformer" not in fqn and "single_transformer_blocks" not in fqn:
        return False
    bad = ("pe_embedder", "norm_", "_norm", "embed", "out_proj")
    return not any(b in fqn.lower() for b in bad)


def load_pipe():
    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def apply_compile(pipe):
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)


def apply_fp8(pipe):
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    from torchao.quantization.granularity import PerRow
    try:
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    except TypeError:
        cfg = Float8DynamicActivationFloat8WeightConfig()
    quantize_(pipe.transformer, cfg, filter_fn=filter_fn)


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


def bench_cell(pipe, wave_pil, prompt_embeds, alpha, size, n_steps, label):
    print(f"\n--- {label} @ {size}² / {n_steps}-step ---", flush=True)
    runner = build_runner(pipe, wave_pil, prompt_embeds, alpha, n_steps, size)
    for w in range(WARMUP_ITERS):
        t0 = time.perf_counter()
        _ = runner(SEED)
        torch.cuda.synchronize()
        print(f"  warmup {w+1}/{WARMUP_ITERS}: {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)
    lats = []
    last = None
    for r in range(TIMED_ITERS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        last = runner(SEED + r)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)
    out = {
        "name": label,
        "size_px": size,
        "steps": n_steps,
        "alpha": alpha,
        "mean_ms": round(sum(lats) / len(lats), 2),
        "p50_ms": round(percentile(lats, 50), 2),
        "p95_ms": round(percentile(lats, 95), 2),
        "p99_ms": round(percentile(lats, 99), 2),
        "min_ms": round(min(lats), 2),
        "max_ms": round(max(lats), 2),
        "fps_mean": round(1000 / (sum(lats) / len(lats)), 2),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }
    print(f"  result: mean={out['mean_ms']}ms p95={out['p95_ms']}ms fps={out['fps_mean']} vram={out['vram_gb']}GB", flush=True)
    return out, last


def free(pipe):
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_variant(name, apply_quant_fn, wave_pil, results):
    print(f"\n{'='*78}\n[VARIANT] {name}\n{'='*78}", flush=True)
    try:
        pipe = load_pipe()
        if apply_quant_fn is not None:
            print(f"  applying quantization: {name}", flush=True)
            apply_quant_fn(pipe)
        print(f"  applying torch.compile (mode=default)", flush=True)
        apply_compile(pipe)
        r = pipe.encode_prompt(prompt=PROMPT, device="cuda",
                               num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN)
        embeds = r[0] if isinstance(r, tuple) else r
        for size in SIZES:
            cell_label = f"{name}_res{size}_step{N_STEPS}"
            try:
                out, img = bench_cell(pipe, wave_pil, embeds, ALPHA, size, N_STEPS, cell_label)
                out["variant"] = name
                results.append(out)
                if img is not None:
                    img.save(OUT_DIR / f"{cell_label}.png")
            except Exception as e:
                print(f"  cell FAILED: {e}", flush=True)
                results.append({"name": cell_label, "variant": name, "error": str(e)[:300]})
        free(pipe)
    except Exception as e:
        print(f"[VARIANT] {name} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        results.append({"variant": name, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        try: free(pipe)
        except Exception: pass


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    import diffusers, transformers, torchao
    print(f"[init] diffusers={diffusers.__version__} transformers={transformers.__version__} torchao={torchao.__version__}", flush=True)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    wave_pil = Image.open(WAVE_PATH).convert("RGB")
    results = []

    run_variant("bf16_4step", None, wave_pil, results)
    run_variant("fp8_dynamic_act_4step", apply_fp8, wave_pil, results)

    summary = {
        "date": "2026-04-30",
        "phase": "0b_4step_baseline + 3b_fp8_4step",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torchao": torchao.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "config": {
            "n_steps": N_STEPS,
            "alpha": ALPHA,
            "max_seq_len": MAX_SEQ_LEN,
            "prompt": PROMPT,
            "compile_mode": "default",
        },
        "cells": results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*78}\n4-STEP SUMMARY\n{'='*78}", flush=True)
    print(f"{'cell':<38} {'mean':>10} {'p95':>10} {'fps':>8} {'vram':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r.get('name', r.get('variant', '?')):<38}  FAIL: {r['error'][:40]}")
        else:
            print(f"{r['name']:<38} {r['mean_ms']:>8.2f}ms {r['p95_ms']:>8.2f}ms {r['fps_mean']:>7.2f} {r['vram_gb']:>6.2f}GB")
    print(f"\n[done] {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
