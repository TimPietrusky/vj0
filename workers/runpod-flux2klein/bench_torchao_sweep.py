#!/usr/bin/env python3
"""TorchAO variant sweep — find the next quantization win.

Phase 3 found that `Float8DynamicActivationFloat8WeightConfig` (PerRow)
on transformer linears wins by 28% at 256² / 4-step over compiled bf16.
Open question: is there a better variant, or a better target-module subset?

This bench tests every (config, filter) cell in a matrix, against the same
input + prompt + seed, and reports:
  - avg latency, p95 latency
  - VRAM at steady state
  - MSE vs bf16 reference (proxy for quality drift)
  - sample image saved per cell for visual review

Cells are ordered so the bf16 baseline runs first (reference), then variants.
Each cell loads a fresh pipe (no carry-over of previous quantization state).

USAGE
  cd /workspace
  python3 bench_torchao_sweep.py             # all cells
  python3 bench_torchao_sweep.py --quick     # just the most-promising 6 cells
  python3 bench_torchao_sweep.py --size 384  # different test resolution
  python3 bench_torchao_sweep.py --steps 2   # different step count

OUTPUT
  /workspace/bench-torchao-sweep/summary.json
  /workspace/bench-torchao-sweep/<cell_label>.png  (one per cell)
  /workspace/bench-torchao-sweep/diffs/<cell>_vs_bf16_x10.png  (MSE-amplified diff)
"""
import argparse
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
OUT_DIR = Path("/workspace/bench-torchao-sweep")
SEED = 42
ALPHA = 0.10
MAX_SEQ_LEN = 64
PROMPT = (
    "a bright white lightning bolt against a pitch black night sky, "
    "dramatic, photographic, high contrast"
)
WARMUP_ITERS = 5
TIMED_ITERS = 20


# ---------- module-filter functions ---------- #

def filter_all_transformer_linears(module, fqn):
    if not isinstance(module, torch.nn.Linear):
        return False
    if "transformer" not in fqn:
        return False
    bad = ("pe_embedder", "norm_", "_norm", "embed", "out_proj")
    return not any(b in fqn.lower() for b in bad)


def filter_single_transformer_blocks_only(module, fqn):
    if not isinstance(module, torch.nn.Linear):
        return False
    if "single_transformer_blocks" not in fqn:
        return False
    bad = ("pe_embedder", "norm_", "_norm", "embed", "out_proj")
    return not any(b in fqn.lower() for b in bad)


def filter_all_linears(module, fqn):
    return isinstance(module, torch.nn.Linear)


FILTERS = {
    "all_transformer_linears": filter_all_transformer_linears,
    "single_transformer_blocks_only": filter_single_transformer_blocks_only,
    "all_linears": filter_all_linears,
}


# ---------- TorchAO config builders ---------- #
# Each returns a callable that takes (pipe, filter_fn) and applies the config.
# Returning None from a builder marks the variant as unavailable for this
# torchao version (gracefully skipped).

def build_fp8_dynact_per_row(pipe, filter_fn):
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    try:
        from torchao.quantization.granularity import PerRow
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    except Exception:
        cfg = Float8DynamicActivationFloat8WeightConfig()
    quantize_(pipe.transformer, cfg, filter_fn=filter_fn)


def build_fp8_dynact_per_tensor(pipe, filter_fn):
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    try:
        from torchao.quantization.granularity import PerTensor
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
    except Exception as e:
        raise RuntimeError(f"PerTensor granularity unavailable: {e}")
    quantize_(pipe.transformer, cfg, filter_fn=filter_fn)


def build_fp8_weight_only(pipe, filter_fn):
    from torchao.quantization import quantize_, Float8WeightOnlyConfig
    quantize_(pipe.transformer, Float8WeightOnlyConfig(), filter_fn=filter_fn)


def build_int8_weight_only(pipe, filter_fn):
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    quantize_(pipe.transformer, Int8WeightOnlyConfig(), filter_fn=filter_fn)


def build_int8_dynact_int8_weight(pipe, filter_fn):
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    quantize_(pipe.transformer, Int8DynamicActivationInt8WeightConfig(), filter_fn=filter_fn)


def build_int4_weight_only(pipe, filter_fn):
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    # group_size=128 is the typical default; some variants need power-of-2 hidden dims
    quantize_(pipe.transformer, Int4WeightOnlyConfig(group_size=128), filter_fn=filter_fn)


CONFIGS = {
    "fp8_dynact_per_row": build_fp8_dynact_per_row,        # current winner
    "fp8_dynact_per_tensor": build_fp8_dynact_per_tensor,  # fewer scale factors → faster?
    "fp8_weight_only": build_fp8_weight_only,              # Phase 3 said regresses; re-confirm
    "int8_weight_only": build_int8_weight_only,            # untested
    "int8_dynact_int8_weight": build_int8_dynact_int8_weight,  # untested true int8 matmul
    "int4_weight_only": build_int4_weight_only,            # 4-bit storage; expect regression but big VRAM win
}


# ---------- bench machinery ---------- #

def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pil2t(img):
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def load_pipe():
    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def compile_pipe(pipe):
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)


def build_runner(pipe, wave_pil, prompt_embeds, alpha, n_steps, size):
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

    def encode_img():
        img = wave_pil if wave_pil.size == (size, size) else wave_pil.resize((size, size), Image.LANCZOS)
        t = pil2t(img).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    def run_one(seed):
        lat = encode_img()
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


def free_pipe(pipe):
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def mse_vs_reference(img_pil, ref_pil):
    """Mean-squared-error between two PIL images, in [0, 1] linear-ish space.
    Reference and candidate must be the same size."""
    if img_pil.size != ref_pil.size:
        img_pil = img_pil.resize(ref_pil.size, Image.LANCZOS)
    a = np.asarray(img_pil, dtype=np.float32) / 255.0
    b = np.asarray(ref_pil, dtype=np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


def diff_image(img_pil, ref_pil, gain=10.0):
    """Per-pixel |diff| × gain, clipped to 8-bit. Useful for spotting where
    quantization actually changed the output (vs claiming "looks the same")."""
    if img_pil.size != ref_pil.size:
        img_pil = img_pil.resize(ref_pil.size, Image.LANCZOS)
    a = np.asarray(img_pil, dtype=np.float32)
    b = np.asarray(ref_pil, dtype=np.float32)
    d = np.clip(np.abs(a - b) * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(d, mode="RGB")


def bench_cell(label, apply_quant_fn, filter_name, args, results, ref_image=None):
    """Run one (config, filter) cell. apply_quant_fn=None → bf16 control."""
    print(f"\n{'='*78}\n[CELL] {label}\n{'='*78}", flush=True)
    t_setup = time.perf_counter()
    pipe = None
    try:
        pipe = load_pipe()
        if apply_quant_fn is not None:
            print(f"  applying quantization (filter={filter_name})", flush=True)
            apply_quant_fn(pipe, FILTERS[filter_name])
        print("  applying torch.compile", flush=True)
        compile_pipe(pipe)
        r = pipe.encode_prompt(prompt=PROMPT, device="cuda",
                               num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN)
        embeds = r[0] if isinstance(r, tuple) else r
        wave_pil = Image.open(WAVE_PATH).convert("RGB")
        runner = build_runner(pipe, wave_pil, embeds, ALPHA, args.steps, args.size)

        # Warmup (first iter triggers compile)
        for w in range(WARMUP_ITERS):
            t0 = time.perf_counter()
            _ = runner(SEED)
            torch.cuda.synchronize()
            print(f"    warmup {w+1}/{WARMUP_ITERS}: {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

        # Timed
        lats = []
        last = None
        for r in range(TIMED_ITERS):
            torch.cuda.synchronize()
            t = time.perf_counter()
            last = runner(SEED + r)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t) * 1000)

        out = {
            "label": label,
            "filter": filter_name,
            "size_px": args.size,
            "steps": args.steps,
            "alpha": ALPHA,
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
            "setup_s": round(time.perf_counter() - t_setup, 1),
        }

        # Save sample image (always uses seed=SEED so quality compares)
        sample = runner(SEED)
        torch.cuda.synchronize()
        sample.save(OUT_DIR / f"{label}.png")
        if ref_image is not None and label != "bf16_baseline":
            out["mse_vs_bf16"] = round(mse_vs_reference(sample, ref_image), 6)
            (OUT_DIR / "diffs").mkdir(parents=True, exist_ok=True)
            diff_image(sample, ref_image).save(OUT_DIR / "diffs" / f"{label}_vs_bf16_x10.png")
        else:
            out["mse_vs_bf16"] = 0.0

        print(f"  result: mean={out['mean_ms']}ms p95={out['p95_ms']}ms "
              f"fps={out['fps_mean']} vram={out['vram_gb']}GB "
              f"mse={out['mse_vs_bf16']}", flush=True)
        results.append(out)
        free_pipe(pipe)
        return sample
    except Exception as e:
        print(f"  CELL FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        results.append({
            "label": label,
            "filter": filter_name,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
        })
        if pipe is not None:
            try: free_pipe(pipe)
            except Exception: pass
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256, help="Test resolution (default 256)")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (default 4)")
    parser.add_argument("--quick", action="store_true",
                        help="Only the most-promising 6 cells (skip int4 and per_tensor crosses)")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    global OUT_DIR
    OUT_DIR = out_dir

    print(f"[init] device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}", flush=True)
    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    import diffusers, transformers, torchao
    print(f"[init] diffusers={diffusers.__version__} transformers={transformers.__version__} torchao={torchao.__version__}", flush=True)
    print(f"[init] size={args.size} steps={args.steps} quick={args.quick}", flush=True)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Build cell list. Order: bf16 reference first, then variants.
    if args.quick:
        cells = [
            ("bf16_baseline", None, "all_transformer_linears"),
            ("fp8_dynact_per_row__all_transformer_linears", "fp8_dynact_per_row", "all_transformer_linears"),
            ("fp8_dynact_per_row__single_transformer_blocks_only", "fp8_dynact_per_row", "single_transformer_blocks_only"),
            ("fp8_dynact_per_row__all_linears", "fp8_dynact_per_row", "all_linears"),
            ("int8_weight_only__all_transformer_linears", "int8_weight_only", "all_transformer_linears"),
            ("int8_dynact_int8_weight__all_transformer_linears", "int8_dynact_int8_weight", "all_transformer_linears"),
        ]
    else:
        cells = [("bf16_baseline", None, "all_transformer_linears")]
        for cfg_name in CONFIGS:
            for filt_name in FILTERS:
                cells.append((f"{cfg_name}__{filt_name}", cfg_name, filt_name))

    results = []
    ref_image = None

    for label, cfg_key, filt_name in cells:
        apply_fn = (lambda p, f, k=cfg_key: CONFIGS[k](p, f)) if cfg_key else None
        sample = bench_cell(label, apply_fn, filt_name, args, results, ref_image)
        if label == "bf16_baseline" and sample is not None:
            ref_image = sample.copy()

    # Summary table
    summary = {
        "date_started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "torchao": torchao.__version__,
        "diffusers": diffusers.__version__, "transformers": transformers.__version__,
        "args": vars(args),
        "cells": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*78}\nTORCHAO SWEEP SUMMARY ({args.size}² / {args.steps}-step)\n{'='*78}", flush=True)
    print(f"{'cell':<60} {'mean':>10} {'p95':>10} {'fps':>8} {'vram':>8} {'mse':>10}")
    print("-" * 110)
    # Sort: baseline first, then by mean_ms (winners up top)
    rows = [r for r in results if "error" not in r]
    rows.sort(key=lambda r: (r["label"] != "bf16_baseline", r.get("mean_ms", 1e9)))
    for r in rows:
        print(f"{r['label']:<60} "
              f"{r['mean_ms']:>8.2f}ms "
              f"{r['p95_ms']:>8.2f}ms "
              f"{r['fps_mean']:>7.2f} "
              f"{r['vram_gb']:>6.2f}GB "
              f"{r.get('mse_vs_bf16', 0):>10.6f}")
    fails = [r for r in results if "error" in r]
    if fails:
        print(f"\nFAILED ({len(fails)}):")
        for r in fails:
            print(f"  {r['label']}: {r['error'][:80]}")

    print(f"\n[done] {out_dir}/summary.json", flush=True)
    print(f"[done] sample images: {out_dir}/<cell_label>.png", flush=True)
    print(f"[done] diff vs baseline (×10 amplified): {out_dir}/diffs/", flush=True)


if __name__ == "__main__":
    main()
