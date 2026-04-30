#!/usr/bin/env python3
"""Phase 1 — Pruna smashed model with our img2img recipe.

Loads PrunaAI/flux2-klein-4b-smashed via pruna.PrunaModel.from_pretrained,
which reapplies FORA + TorchAO fp8wo + torch.compile per the smash_config.json.
Then runs the same alpha-blend img2img schedule we use in production:
  noisy = α·image_latents + (1-α)·noise
  sigmas = linspace(1-α, 0, N_STEPS)

Tests at the resolutions × steps where FORA can actually activate (4-step):
  - 256² / 4-step
  - 384² / 4-step
  - 512² / 4-step
And one 2-step row to confirm FORA gives no benefit there (fora_start_step=4).

Output: /workspace/bench-2026-04-30/pruna_smashed/summary.json
"""
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SMASHED_REPO = "PrunaAI/flux2-klein-4b-smashed"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
WAVE_PATH = "/workspace/waveforms/waveform_1.png"
OUT_DIR = Path("/workspace/bench-2026-04-30/pruna_smashed")
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


def bench_cell(pipe, wave_pil, prompt_embeds, alpha, size, n_steps, label):
    print(f"\n--- cell: {label} ({size}² / {n_steps}-step / α={alpha}) ---", flush=True)
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
        "name": label,
        "resolution": f"{size}x{size}",
        "size_px": size,
        "steps": n_steps,
        "alpha": alpha,
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


def find_pipe_attribute(loaded_model):
    """PrunaModel wraps the underlying pipeline. Find it. We need .vae, .transformer,
    .encode_prompt, ._patchify_latents, etc. — same as Flux2KleinKVPipeline."""
    candidates = []
    for attr in ("model", "pipeline", "pipe", "_model"):
        if hasattr(loaded_model, attr):
            candidates.append((attr, getattr(loaded_model, attr)))
    for name, obj in candidates:
        if hasattr(obj, "vae") and hasattr(obj, "transformer") and hasattr(obj, "encode_prompt"):
            print(f"  found pipeline at .{name}", flush=True)
            return obj
    # PrunaModel itself may be callable like a pipeline
    if hasattr(loaded_model, "vae") and hasattr(loaded_model, "transformer"):
        print("  PrunaModel exposes pipeline interface directly", flush=True)
        return loaded_model
    raise AttributeError(
        f"can't find pipeline interface on PrunaModel (tried: model, pipeline, pipe, _model). "
        f"attrs: {[a for a in dir(loaded_model) if not a.startswith('_')][:20]}"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}", flush=True)
    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    import diffusers, transformers
    print(f"[init] diffusers={diffusers.__version__} transformers={transformers.__version__}", flush=True)
    try:
        import pruna
        print(f"[init] pruna={pruna.__version__}", flush=True)
    except ImportError as e:
        print(f"[init] PRUNA NOT INSTALLED: {e}", flush=True)
        return

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from pruna import PrunaModel
    from diffusers import AutoencoderKLFlux2

    print(f"[load] {SMASHED_REPO} via PrunaModel.from_pretrained...", flush=True)
    t0 = time.perf_counter()
    loaded = PrunaModel.from_pretrained(SMASHED_REPO, torch_dtype=torch.bfloat16)
    print(f"[load] PrunaModel loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    pipe = find_pipe_attribute(loaded)

    print(f"[load] swapping VAE for {DECODER_REPO}", flush=True)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16).to("cuda")

    if hasattr(pipe, "to"):
        pipe.to("cuda")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    print(f"[load] vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    wave_pil = Image.open(WAVE_PATH).convert("RGB")

    print(f"[prompt] encoding (max_seq_len={MAX_SEQ_LEN})", flush=True)
    r = pipe.encode_prompt(
        prompt=PROMPT, device="cuda",
        num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN,
    )
    prompt_embeds = r[0] if isinstance(r, tuple) else r
    print(f"[prompt] embeds shape={tuple(prompt_embeds.shape)}", flush=True)

    cells = [
        # FORA active here (start_step=4, interval=3)
        (256, 4, "256_4step_fora"),
        (384, 4, "384_4step_fora"),
        (512, 4, "512_4step_fora"),
        # FORA inactive — for direct comparison vs baseline
        (256, 2, "256_2step_no_fora"),
        (512, 2, "512_2step_no_fora"),
    ]

    results = []
    for size, n_steps, label in cells:
        try:
            out, img = bench_cell(pipe, wave_pil, prompt_embeds, ALPHA, size, n_steps, label)
            results.append(out)
            if img is not None:
                img.save(OUT_DIR / f"{label}.png")
        except Exception as e:
            print(f"[FAIL] {label}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            results.append({
                "name": label,
                "resolution": f"{size}x{size}",
                "steps": n_steps,
                "error": f"{type(e).__name__}: {str(e)[:300]}",
            })

    summary = {
        "date": "2026-04-30",
        "phase": "1_pruna_smashed",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "pruna": pruna.__version__,
        "config": {
            "model": SMASHED_REPO,
            "decoder": "FLUX.2-small-decoder",
            "dtype": "bfloat16",
            "applied_via_pruna": ["fora", "torchao_fp8wo", "torch_compile"],
            "alpha": ALPHA,
            "max_seq_len": MAX_SEQ_LEN,
            "prompt": PROMPT,
            "img2img_recipe": "noisy = a*lat + (1-a)*noise; sigmas = linspace(1-a, 0, N)",
        },
        "cells": results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*78}\nPRUNA SMASHED SUMMARY\n{'='*78}", flush=True)
    print(f"{'cell':<28} {'mean':>10} {'p95':>10} {'p99':>10} {'fps':>8} {'vram':>8}")
    print("-" * 78)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<28}  FAIL: {r['error'][:50]}")
        else:
            print(
                f"{r['name']:<28} "
                f"{r['mean_ms']:>8.2f}ms "
                f"{r['p95_ms']:>8.2f}ms "
                f"{r['p99_ms']:>8.2f}ms "
                f"{r['fps_mean']:>7.2f} "
                f"{r['vram_gb']:>6.2f}GB"
            )

    print(f"\n[done] {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
