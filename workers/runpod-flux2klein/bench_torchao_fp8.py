#!/usr/bin/env python3
"""Phase 3 — TorchAO fp8 quantization on the transformer, then compile.

This replicates the quantization slice of Pruna's smashed-model recipe
(per smash_config.json: TorchAO fp8wo on `*single_transformer_blocks.*`),
without Pruna's runtime overhead and env conflicts.

We test three quantization variants:
  - bf16 (control — same as baseline)
  - fp8 weight-only (Float8WeightOnlyConfig): fp8 storage, dequant-to-bf16 per matmul
  - fp8 dynamic-activation (Float8DynamicActivationFloat8WeightConfig): true fp8 matmul

For each: apply quantize_(transformer, config) BEFORE torch.compile, so the
compiler captures the quantized ops in its kernels. We DO NOT quantize the VAE
encoder/decoder — the small decoder is precision-sensitive.

Bench at 256² and 512² × 2-step (the production sweet spot and quality cap).

Output: /workspace/bench-2026-04-30/torchao_fp8/summary.json
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
OUT_DIR = Path("/workspace/bench-2026-04-30/torchao_fp8")
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
SIZES = [256, 512]


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pil2t(img):
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def filter_only_transformer_blocks(module, fqn):
    """Quantize only the transformer's single_transformer_blocks linear layers,
    matching Pruna's smash_config.json behavior. Keeps embeddings, norms, and
    attention positional layers in bf16 for stability."""
    if not isinstance(module, torch.nn.Linear):
        return False
    # accept the full transformer linear set; could narrow further if needed
    if "transformer" not in fqn and "single_transformer_blocks" not in fqn:
        return False
    # exclude PE embedder, norm, embed projection
    bad = ("pe_embedder", "norm_", "_norm", "embed", "out_proj")
    if any(b in fqn.lower() for b in bad):
        return False
    return True


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


def apply_fp8wo(pipe):
    from torchao.quantization import quantize_, Float8WeightOnlyConfig
    cfg = Float8WeightOnlyConfig()
    quantize_(pipe.transformer, cfg, filter_fn=filter_only_transformer_blocks)


def apply_fp8dqrow(pipe):
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    from torchao.quantization.granularity import PerRow
    # PerRow weight granularity, full fp8 matmul (act+weight)
    try:
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    except TypeError:
        cfg = Float8DynamicActivationFloat8WeightConfig()
    quantize_(pipe.transformer, cfg, filter_fn=filter_only_transformer_blocks)


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
    return out, last


def free_pipe(pipe):
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_variant(name, apply_quant_fn, wave_pil, results, prompt_embeds_holder):
    print(f"\n{'='*78}\n[VARIANT] {name}\n{'='*78}", flush=True)
    t_setup = time.perf_counter()
    try:
        pipe = load_pipe()
        if apply_quant_fn is not None:
            print(f"  applying quantization: {name}", flush=True)
            apply_quant_fn(pipe)
        print(f"  applying torch.compile", flush=True)
        apply_compile(pipe)
        # Encode prompt fresh — prompt cache is per-pipe
        r = pipe.encode_prompt(
            prompt=PROMPT, device="cuda",
            num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN,
        )
        embeds = r[0] if isinstance(r, tuple) else r
        prompt_embeds_holder["embeds"] = embeds  # cache for downstream

        for size in SIZES:
            cell_label = f"{name}_res{size}_step{N_STEPS}"
            out, img = bench_cell(pipe, wave_pil, embeds, ALPHA, size, N_STEPS, cell_label)
            out["variant"] = name
            out["setup_s"] = round(time.perf_counter() - t_setup, 1)
            results.append(out)
            if img is not None:
                img.save(OUT_DIR / f"{cell_label}.png")
        free_pipe(pipe)
    except Exception as e:
        print(f"[VARIANT] {name} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        results.append({"variant": name, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        try: free_pipe(pipe)  # noqa: F821
        except Exception: pass


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}", flush=True)
    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    import diffusers, transformers
    try:
        import torchao
        print(f"[init] torchao={torchao.__version__}", flush=True)
    except ImportError as e:
        print(f"[init] torchao NOT installed: {e}", flush=True)
        return
    print(f"[init] diffusers={diffusers.__version__} transformers={transformers.__version__}", flush=True)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    wave_pil = Image.open(WAVE_PATH).convert("RGB")
    results = []
    holder = {}

    # 1) bf16 control
    run_variant("bf16_control", None, wave_pil, results, holder)
    # 2) fp8 weight-only
    run_variant("fp8_weight_only", apply_fp8wo, wave_pil, results, holder)
    # 3) fp8 dynamic activation + weight (full fp8 matmul)
    run_variant("fp8_dynamic_act", apply_fp8dqrow, wave_pil, results, holder)

    summary = {
        "date": "2026-04-30",
        "phase": "3_torchao_fp8",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torchao": torchao.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "config": {
            "alpha": ALPHA,
            "n_steps": N_STEPS,
            "max_seq_len": MAX_SEQ_LEN,
            "prompt": PROMPT,
            "filter": "transformer linear layers only (skip pe_embedder, norms, embed)",
            "compile_mode": "default",
        },
        "cells": results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*78}\nTORCHAO FP8 SUMMARY\n{'='*78}", flush=True)
    print(f"{'cell':<38} {'mean':>10} {'p95':>10} {'fps':>8} {'vram':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['variant']:<38}  FAIL: {r['error'][:40]}")
        else:
            print(
                f"{r['name']:<38} "
                f"{r['mean_ms']:>8.2f}ms "
                f"{r['p95_ms']:>8.2f}ms "
                f"{r['fps_mean']:>7.2f} "
                f"{r['vram_gb']:>6.2f}GB"
            )

    print(f"\n[done] {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
