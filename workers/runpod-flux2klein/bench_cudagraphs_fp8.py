#!/usr/bin/env python3
"""Phase 5 — CUDA graphs stacked on top of fp8_dynamic_act.

Current best: fp8_dynamic_act + compile(mode="default") = 27.39 ms / 36.5 fps @ 256².
Hypothesis: Python launch overhead (~1-3 ms per step) is unhidden — replacing
those launches with a CUDA graph replay might claw back a few ms per step.

Two attempts, in order:
  1) torch.compile(mode="reduce-overhead", fullgraph=False) — uses CUDA graphs
     internally for the captured regions. RESULTS.md says this crashed on PEFT
     wrappers with bf16; the fp8 quantize_ may have replaced the Linear layers
     with non-PEFT-wrapped quantized layers, possibly fixing it. Worth trying.
  2) If (1) fails: manual graph capture via torch.cuda.graph(). More fragile but
     bypasses the compile path entirely.

Test only at 256² / 2-step (the production target).

Output: /workspace/bench-2026-04-30/cudagraphs_fp8/summary.json
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
OUT_DIR = Path("/workspace/bench-2026-04-30/cudagraphs_fp8")
SEED = 42
ALPHA = 0.10
MAX_SEQ_LEN = 64
PROMPT = (
    "a bright white lightning bolt against a pitch black night sky, "
    "dramatic, photographic, high contrast"
)
WARMUP_ITERS = 5
TIMED_ITERS = 20
N_STEPS = 2
SIZE = 256


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


def bench_cell(pipe, wave_pil, prompt_embeds, label):
    print(f"\n--- {label} @ {SIZE}² / {N_STEPS}-step ---", flush=True)
    runner = build_runner(pipe, wave_pil, prompt_embeds, ALPHA, N_STEPS, SIZE)
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
        "size_px": SIZE,
        "steps": N_STEPS,
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    wave_pil = Image.open(WAVE_PATH).convert("RGB")
    results = []

    # --- variant A: compile(default) — control reproducing Phase 3 winning row
    print(f"\n{'='*78}\n[A] fp8_dynamic_act + compile(mode=default)\n{'='*78}", flush=True)
    try:
        pipe = load_pipe()
        apply_fp8(pipe)
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
        r = pipe.encode_prompt(prompt=PROMPT, device="cuda", num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN)
        embeds = r[0] if isinstance(r, tuple) else r
        out, img = bench_cell(pipe, wave_pil, embeds, "fp8_compile_default")
        results.append(out)
        if img: img.save(OUT_DIR / "fp8_compile_default.png")
        free(pipe)
    except Exception as e:
        print(f"[A] FAILED: {e}", flush=True)
        traceback.print_exc()
        results.append({"name": "fp8_compile_default", "error": str(e)[:300]})
        try: free(pipe)
        except Exception: pass

    # --- variant B: compile(reduce-overhead) — uses CUDA graphs internally
    print(f"\n{'='*78}\n[B] fp8_dynamic_act + compile(mode=reduce-overhead)\n{'='*78}", flush=True)
    try:
        pipe = load_pipe()
        apply_fp8(pipe)
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=False, dynamic=False)
        pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="reduce-overhead", fullgraph=False, dynamic=False)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="reduce-overhead", fullgraph=False, dynamic=False)
        r = pipe.encode_prompt(prompt=PROMPT, device="cuda", num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN)
        embeds = r[0] if isinstance(r, tuple) else r
        out, img = bench_cell(pipe, wave_pil, embeds, "fp8_compile_reduce_overhead")
        results.append(out)
        if img: img.save(OUT_DIR / "fp8_compile_reduce_overhead.png")
        free(pipe)
    except Exception as e:
        print(f"[B] FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc()
        results.append({"name": "fp8_compile_reduce_overhead", "error": f"{type(e).__name__}: {str(e)[:300]}"})
        try: free(pipe)
        except Exception: pass

    # --- variant C: compile(default) on transformer only with reduce-overhead on VAE
    print(f"\n{'='*78}\n[C] fp8_dynamic_act + transformer:default + vae:reduce-overhead\n{'='*78}", flush=True)
    try:
        pipe = load_pipe()
        apply_fp8(pipe)
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="reduce-overhead", fullgraph=False, dynamic=False)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="reduce-overhead", fullgraph=False, dynamic=False)
        r = pipe.encode_prompt(prompt=PROMPT, device="cuda", num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN)
        embeds = r[0] if isinstance(r, tuple) else r
        out, img = bench_cell(pipe, wave_pil, embeds, "fp8_split_compile")
        results.append(out)
        if img: img.save(OUT_DIR / "fp8_split_compile.png")
        free(pipe)
    except Exception as e:
        print(f"[C] FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc()
        results.append({"name": "fp8_split_compile", "error": f"{type(e).__name__}: {str(e)[:300]}"})
        try: free(pipe)
        except Exception: pass

    summary = {
        "date": "2026-04-30",
        "phase": "5_cudagraphs_on_fp8",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cells": results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*78}\nCUDA-GRAPH-ON-FP8 SUMMARY\n{'='*78}", flush=True)
    print(f"{'cell':<38} {'mean':>10} {'p95':>10} {'fps':>8} {'vram':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<38}  FAIL: {r['error'][:40]}")
        else:
            print(f"{r['name']:<38} {r['mean_ms']:>8.2f}ms {r['p95_ms']:>8.2f}ms {r['fps_mean']:>7.2f} {r['vram_gb']:>6.2f}GB")
    print(f"\n[done] {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
