#!/usr/bin/env python3
"""Phase 7 — Dual-GPU round-robin throughput bench.

Two RTX 5090s on the same host. We load identical pipelines on each, applying
fp8_dynamic_act + torch.compile (the Phase 3b winning recipe at 4-step), then
measure:

  A) Per-GPU latency — single pipeline running solo on GPU 0 (control).
  B) Per-GPU latency — single pipeline running solo on GPU 1 (sanity check).
  C) Parallel throughput — both pipelines fed alternating frames via threads.
     Reports effective FPS = 2 / (mean per-frame latency under contention).

Two threads are sufficient because PyTorch CUDA kernel launches release the
GIL, and each pipeline targets its own device (no cross-device blocking).

Output: /workspace/bench-2026-04-30/dual_gpu/summary.json
"""
import gc
import json
import math
import os
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
WAVE_PATH = "/workspace/waveforms/waveform_1.png"
OUT_DIR = Path("/workspace/bench-2026-04-30/dual_gpu")
SEED = 42
ALPHA = 0.10
MAX_SEQ_LEN = 64
PROMPT = (
    "a bright white lightning bolt against a pitch black night sky, "
    "dramatic, photographic, high contrast"
)
WARMUP_ITERS = 5
TIMED_ITERS = 40  # more iters for parallel test (2 GPUs, 20 frames each)
N_STEPS = 4
SIZES = [256, 512]


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


def load_pipe_on(device: str):
    """Build a fully-quantized + compiled pipeline pinned to a specific device."""
    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    from torchao.quantization.granularity import PerRow

    print(f"  [{device}] loading Klein + small decoder", flush=True)
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    print(f"  [{device}] applying fp8 dynamic-act quantization", flush=True)
    try:
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    except TypeError:
        cfg = Float8DynamicActivationFloat8WeightConfig()
    quantize_(pipe.transformer, cfg, filter_fn=filter_fn)

    print(f"  [{device}] applying torch.compile", flush=True)
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
    return pipe


def encode_prompt_on(pipe, device: str):
    r = pipe.encode_prompt(prompt=PROMPT, device=device, num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN)
    return r[0] if isinstance(r, tuple) else r


def make_runner(pipe, prompt_embeds, alpha, n_steps, size, device):
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents
    wave_pil = Image.open(WAVE_PATH).convert("RGB").resize((size, size), Image.LANCZOS)

    def encode_img():
        a = np.asarray(wave_pil, dtype=np.float32) / 127.5 - 1.0
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    def run_one(seed):
        lat = encode_img()
        gen = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device=device)
        noisy = alpha * lat + (1 - alpha) * noise
        sigmas = np.linspace(1 - alpha, 0.0, n_steps).tolist()
        return pipe(
            image=None, prompt=None, prompt_embeds=prompt_embeds,
            latents=noisy, sigmas=sigmas,
            height=size, width=size, num_inference_steps=n_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]

    return run_one


def time_solo(runner, device, label):
    """Single-pipeline timing on one device. Returns mean/p95/fps for the cell."""
    print(f"\n--- SOLO {label} on {device} ---", flush=True)
    for w in range(WARMUP_ITERS):
        t0 = time.perf_counter()
        _ = runner(SEED)
        torch.cuda.synchronize(device)
        print(f"  warmup {w+1}/{WARMUP_ITERS}: {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)
    lats = []
    for r in range(TIMED_ITERS // 2):  # half iters when measuring solo
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        _ = runner(SEED + r)
        torch.cuda.synchronize(device)
        lats.append((time.perf_counter() - t) * 1000)
    out = {
        "mode": "solo",
        "device": device,
        "label": label,
        "iters": len(lats),
        "mean_ms": round(sum(lats)/len(lats), 2),
        "p50_ms": round(percentile(lats, 50), 2),
        "p95_ms": round(percentile(lats, 95), 2),
        "p99_ms": round(percentile(lats, 99), 2),
        "min_ms": round(min(lats), 2),
        "max_ms": round(max(lats), 2),
        "fps_per_gpu": round(1000 / (sum(lats)/len(lats)), 2),
    }
    print(f"  result: mean={out['mean_ms']}ms p95={out['p95_ms']}ms fps={out['fps_per_gpu']}", flush=True)
    return out


def time_parallel(runner_a, runner_b, label):
    """Both runners fed alternating seeds in parallel from two threads.
    Effective throughput = total frames / total wallclock time."""
    print(f"\n--- PARALLEL {label} (cuda:0 + cuda:1, threads) ---", flush=True)

    # Warmup both in parallel (both compile their own kernels independently)
    barrier = threading.Barrier(2)

    def warmup_one(runner, dev):
        barrier.wait()
        for w in range(WARMUP_ITERS):
            _ = runner(SEED)
            torch.cuda.synchronize(dev)

    t_a = threading.Thread(target=warmup_one, args=(runner_a, "cuda:0"))
    t_b = threading.Thread(target=warmup_one, args=(runner_b, "cuda:1"))
    t_a.start(); t_b.start()
    t_a.join(); t_b.join()

    # Timed: each runner does TIMED_ITERS // 2 frames. Both run in parallel.
    half = TIMED_ITERS // 2
    barrier2 = threading.Barrier(2)
    lats_a, lats_b = [], []
    start_evt = [None]

    def run_one(runner, dev, lats, off):
        barrier2.wait()
        if dev == "cuda:0":
            start_evt[0] = time.perf_counter()
        for r in range(half):
            t0 = time.perf_counter()
            _ = runner(SEED + off + r)
            torch.cuda.synchronize(dev)
            lats.append((time.perf_counter() - t0) * 1000)

    t_a = threading.Thread(target=run_one, args=(runner_a, "cuda:0", lats_a, 0))
    t_b = threading.Thread(target=run_one, args=(runner_b, "cuda:1", lats_b, 100))
    t_a.start(); t_b.start()
    t_a.join(); t_b.join()
    end = time.perf_counter()

    wall_s = end - start_evt[0]
    total_frames = len(lats_a) + len(lats_b)
    out = {
        "mode": "parallel",
        "label": label,
        "frames_a": len(lats_a),
        "frames_b": len(lats_b),
        "wall_s": round(wall_s, 3),
        "throughput_fps": round(total_frames / wall_s, 2),
        "per_frame_a_mean_ms": round(sum(lats_a)/len(lats_a), 2),
        "per_frame_b_mean_ms": round(sum(lats_b)/len(lats_b), 2),
        "per_frame_a_p95_ms": round(percentile(lats_a, 95), 2),
        "per_frame_b_p95_ms": round(percentile(lats_b, 95), 2),
    }
    print(
        f"  result: {total_frames} frames in {wall_s:.2f}s = {out['throughput_fps']} fps effective\n"
        f"  per-frame mean A={out['per_frame_a_mean_ms']}ms B={out['per_frame_b_mean_ms']}ms",
        flush=True,
    )
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = torch.cuda.device_count()
    print(f"[init] visible CUDA devices: {n}", flush=True)
    for i in range(n):
        print(f"  cuda:{i} = {torch.cuda.get_device_name(i)}", flush=True)
    if n < 2:
        print("[FAIL] need 2 GPUs", flush=True)
        return

    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    import diffusers, transformers, torchao
    print(f"[init] diffusers={diffusers.__version__} transformers={transformers.__version__} torchao={torchao.__version__}", flush=True)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("\n[load] pipeline on cuda:0", flush=True)
    pipe_a = load_pipe_on("cuda:0")
    embeds_a = encode_prompt_on(pipe_a, "cuda:0")

    print("\n[load] pipeline on cuda:1", flush=True)
    pipe_b = load_pipe_on("cuda:1")
    embeds_b = encode_prompt_on(pipe_b, "cuda:1")

    results = []
    for size in SIZES:
        runner_a = make_runner(pipe_a, embeds_a, ALPHA, N_STEPS, size, "cuda:0")
        runner_b = make_runner(pipe_b, embeds_b, ALPHA, N_STEPS, size, "cuda:1")

        # Solo on each GPU (sanity check both behave the same)
        try:
            r = time_solo(runner_a, "cuda:0", f"res{size}_step{N_STEPS}")
            r["size_px"] = size; r["steps"] = N_STEPS; results.append(r)
        except Exception as e:
            print(f"  solo cuda:0 FAILED: {e}", flush=True); traceback.print_exc()

        try:
            r = time_solo(runner_b, "cuda:1", f"res{size}_step{N_STEPS}")
            r["size_px"] = size; r["steps"] = N_STEPS; results.append(r)
        except Exception as e:
            print(f"  solo cuda:1 FAILED: {e}", flush=True); traceback.print_exc()

        # Parallel both GPUs
        try:
            r = time_parallel(runner_a, runner_b, f"res{size}_step{N_STEPS}")
            r["size_px"] = size; r["steps"] = N_STEPS; results.append(r)
        except Exception as e:
            print(f"  parallel FAILED: {e}", flush=True); traceback.print_exc()

    summary = {
        "date": "2026-05-01",
        "phase": "7_dual_gpu",
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
            "compile_mode": "default",
            "fp8": "Float8DynamicActivationFloat8WeightConfig (PerRow)",
        },
        "cells": results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*78}\nDUAL-GPU SUMMARY\n{'='*78}", flush=True)
    print(f"{'mode':<12} {'res':>6} {'label':<22} {'fps':>10}")
    print("-" * 60)
    for r in results:
        if r["mode"] == "solo":
            print(f"{'solo':<12} {r.get('size_px','?'):>6} {r['device']+' '+r['label']:<22} {r['fps_per_gpu']:>9.2f}")
        else:
            print(f"{'parallel':<12} {r.get('size_px','?'):>6} {r['label']:<22} {r['throughput_fps']:>9.2f}")
    print(f"\n[done] {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
