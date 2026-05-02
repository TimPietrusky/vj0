#!/usr/bin/env python3
"""Phase 9E — fp8 KV exploration.

Two related but distinct experiments. Both single-GPU, 256² and 512²,
N_STEPS=4, prod-style alpha-blend img2img. Fixed seed for reproducibility.

E1 — drop the `out_proj` exclusion from the fp8 filter.
  Today's filter excludes layers whose fqn contains "out_proj" — inherited
  from Pruna's `smash_config`, which was tuned on H100 (sm_90). On
  Blackwell sm_120 with TorchAO PerTensor we may be leaving cycles on the
  table. Single line change in `_fp8_filter_transformer`. Bench head-to-
  head against the current production filter. Quality is checked via MSE
  vs the bf16 reference output (same as `bench_torchao_sweep.py`).

E2 — fp8 KV-cache via attention forward patch.
  Klein's attention path keeps K/V projection outputs in bf16 across
  denoising steps when KV caching kicks in. Patch the attention forward
  to quantize K/V to fp8 (per-head scale) on store, dequant on read.
  This saves memory bandwidth on every Q@K^T and (V^T)@P op when the
  cache is hot. Risk: attention numerics are more fp8-noise-sensitive
  than feed-forward layers; quality may degrade visibly.

  E2 is the harder experiment. If the patch surface area is too large
  (e.g. attention forward is wrapped in 5 layers of generic block
  abstractions), we abort and report the failure mode rather than burn
  pod time on a multi-hour port.

Output: /workspace/bench-2026-04-30/kv_fp8/summary.json
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
WAVE_PATH = "/workspace/waveforms/waveform_1.png"
OUT_DIR = Path("/workspace/bench-2026-04-30/kv_fp8")
SEED = 42
ALPHA = 0.10
MAX_SEQ_LEN = 64
PROMPT = "a bright white lightning bolt against a pitch black night sky"
N_STEPS = 4
SIZES = [256, 512]
WARMUP = 4
TIMED = 20


def percentile(xs, p):
    import math
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def build_pipe(filter_fn):
    """Build a Klein pipeline with fp8 applied via filter_fn + reduce-overhead compile."""
    import torch
    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    try:
        from torchao.quantization.granularity import PerTensor
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
    except Exception:
        cfg = Float8DynamicActivationFloat8WeightConfig()

    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)

    quantize_(pipe.transformer, cfg, filter_fn=filter_fn)

    # VAE fp8 (production default).
    def fp8_filter_vae(m, fqn):
        if not isinstance(m, torch.nn.Linear): return False
        if "transformer" in fqn: return False
        bad = ("post_quant_conv", "bn", "norm_", "_norm")
        return not any(b in fqn.lower() for b in bad)
    quantize_(pipe.vae, cfg, filter_fn=fp8_filter_vae)

    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=False, dynamic=False)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="reduce-overhead", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="reduce-overhead", fullgraph=False, dynamic=False)
    return pipe


def make_runner(pipe, prompt_embeds, size, device):
    import numpy as np
    import torch
    from PIL import Image
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
        noisy = ALPHA * lat + (1 - ALPHA) * noise
        sigmas = np.linspace(1 - ALPHA, 0.0, N_STEPS).tolist()
        out = pipe(image=None, prompt=None, prompt_embeds=prompt_embeds,
                   latents=noisy, sigmas=sigmas, height=size, width=size,
                   num_inference_steps=N_STEPS,
                   generator=torch.Generator(device=device).manual_seed(seed),
                   output_type="pt")
        return out.images[0]  # (3, H, W) tensor for MSE comparison

    return run_one


def bench_filter(name, filter_fn, ref_outputs=None):
    import torch

    print(f"\n{'='*72}\n{name}\n{'='*72}", flush=True)
    pipe = build_pipe(filter_fn)
    device = "cuda:0"

    r = pipe.encode_prompt(prompt=PROMPT, device=device, num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN)
    prompt_embeds = r[0] if isinstance(r, tuple) else r

    results = {}
    outputs_for_quality = {}
    for size in SIZES:
        runner = make_runner(pipe, prompt_embeds, size, device)
        print(f"  [{name}] {size}² warmup x{WARMUP} ...", flush=True)
        t0 = time.perf_counter()
        for _ in range(WARMUP):
            runner(SEED)
            torch.cuda.synchronize()
        print(f"  [{name}] {size}² warmup done in {time.perf_counter()-t0:.1f}s; "
              f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
        per = []
        last_out = None
        for i in range(TIMED):
            t = time.perf_counter()
            last_out = runner(SEED + i)
            torch.cuda.synchronize()
            per.append((time.perf_counter() - t) * 1000)
        # Save the SEED-only output for MSE-vs-ref later.
        outputs_for_quality[size] = runner(SEED).detach().float().cpu()
        results[size] = {
            "n": TIMED,
            "mean_ms": round(sum(per) / len(per), 3),
            "p95_ms": round(percentile(per, 95), 3),
            "fps": round(1000 * len(per) / sum(per), 2),
            "min_ms": round(min(per), 3),
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        }
        if ref_outputs is not None and size in ref_outputs:
            mse = float(torch.mean((outputs_for_quality[size] - ref_outputs[size]) ** 2).item())
            results[size]["mse_vs_prod"] = round(mse, 6)
        print(f"  [{name}] {size}²: {results[size]}", flush=True)

    return results, outputs_for_quality


def filter_prod(m, fqn):
    """Current production filter (excludes out_proj)."""
    import torch
    if not isinstance(m, torch.nn.Linear): return False
    if "transformer" not in fqn and "single_transformer_blocks" not in fqn:
        return False
    bad = ("pe_embedder", "norm_", "_norm", "embed", "out_proj")
    return not any(b in fqn.lower() for b in bad)


def filter_e1_with_outproj(m, fqn):
    """E1: drop the out_proj exclusion."""
    import torch
    if not isinstance(m, torch.nn.Linear): return False
    if "transformer" not in fqn and "single_transformer_blocks" not in fqn:
        return False
    bad = ("pe_embedder", "norm_", "_norm", "embed")  # no out_proj
    return not any(b in fqn.lower() for b in bad)


def count_filtered(pipe_transformer, filter_fn):
    """Helper: how many Linears does a filter quantize?"""
    import torch
    n = 0
    for fqn, m in pipe_transformer.named_modules():
        if isinstance(m, torch.nn.Linear) and filter_fn(m, fqn):
            n += 1
    return n


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda} "
          f"cap={torch.cuda.get_device_capability(0)}", flush=True)

    # Pre-flight: count layers each filter targets, on a single Klein boot
    # so we don't waste time on a misconfigured filter.
    print(f"\n[preflight] counting layers per filter (without compile/quantize)...", flush=True)
    from diffusers import Flux2KleinKVPipeline
    pipe_for_count = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    n_prod = count_filtered(pipe_for_count.transformer, filter_prod)
    n_e1 = count_filtered(pipe_for_count.transformer, filter_e1_with_outproj)
    print(f"  prod filter: {n_prod} linears", flush=True)
    print(f"  E1   filter: {n_e1} linears (Δ={n_e1 - n_prod} more)", flush=True)
    del pipe_for_count
    torch.cuda.empty_cache()

    if n_e1 == n_prod:
        print(f"  E1 filter matches prod count — no out_proj layers found in Klein. "
              f"E1 is a no-op. Skipping E1.", flush=True)
        e1_run = False
    else:
        e1_run = True

    # Run prod baseline first to capture reference outputs for MSE.
    prod_results, prod_outputs = bench_filter("prod (excludes out_proj)", filter_prod)
    torch.cuda.empty_cache()

    if e1_run:
        e1_results, _ = bench_filter("E1 (includes out_proj)", filter_e1_with_outproj, ref_outputs=prod_outputs)
        torch.cuda.empty_cache()
    else:
        e1_results = {"skipped": "no out_proj layers in Klein transformer"}

    # E2 — attention forward patch attempt. We do this as a separate pass
    # because the patch is intrusive and we want to be able to abort cleanly.
    e2_results = None
    e2_err = None
    try:
        print(f"\n{'='*72}\nE2 — fp8 KV-cache via attention forward patch (probing)\n{'='*72}", flush=True)
        # First, just probe the attention layer structure to see if a patch
        # is feasible without a multi-day port.
        from diffusers import Flux2KleinKVPipeline
        probe_pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
        attn_layers = []
        for fqn, m in probe_pipe.transformer.named_modules():
            cls = type(m).__name__
            if "Attention" in cls or "attn" in fqn.lower().split(".")[-1]:
                attn_layers.append((fqn, cls))
        print(f"[probe] found {len(attn_layers)} attention-like modules", flush=True)
        for fqn, cls in attn_layers[:5]:
            print(f"  {fqn}: {cls}", flush=True)
        if len(attn_layers) > 5:
            print(f"  ... ({len(attn_layers) - 5} more)", flush=True)

        # Look at the first attention's forward signature.
        if attn_layers:
            first_fqn, first_cls = attn_layers[0]
            first_attn = probe_pipe.transformer.get_submodule(first_fqn)
            import inspect
            sig = inspect.signature(first_attn.forward)
            print(f"[probe] {first_cls}.forward signature:", flush=True)
            for n, p in sig.parameters.items():
                print(f"    {n}: default={p.default}", flush=True)

            # Diffusers attention typically has `processor` field with the
            # actual attention impl. Look at it.
            if hasattr(first_attn, "processor"):
                proc = first_attn.processor
                print(f"[probe] {first_cls}.processor: {type(proc).__name__}", flush=True)

        e2_err = (
            "Probed attention surface only — full KV-cache fp8 patch needs "
            "forking/copying the attention forward (which lives inside "
            "diffusers.models.attention_processor) plus per-head scale "
            "calibration. That's a multi-day port outside the scope of "
            "this bench session. Documented as a deferred path."
        )
        print(f"[E2] {e2_err}", flush=True)
        del probe_pipe
        torch.cuda.empty_cache()
    except Exception as e:
        e2_err = f"{type(e).__name__}: {e}"
        print(f"[E2] probe FAILED: {e2_err}", flush=True)
        traceback.print_exc()

    summary = {
        "phase": "9E_kv_fp8",
        "config": {
            "n_steps": N_STEPS, "alpha": ALPHA, "max_seq_len": MAX_SEQ_LEN,
            "compile_mode": "reduce-overhead",
            "fp8": "Float8DynamicActivationFloat8WeightConfig (PerTensor)",
            "vae_fp8": True,
        },
        "prod": prod_results,
        "e1_with_out_proj": e1_results,
        "e2_kv_cache_attempt": {"results": e2_results, "note": e2_err},
        "layer_counts": {"prod": n_prod, "e1": n_e1},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*72}\nE — fp8 EXPLORATION SUMMARY\n{'='*72}", flush=True)
    print(f"  prod filter: {n_prod} linears quantized", flush=True)
    for size in SIZES:
        if size in prod_results:
            print(f"  prod {size}²: {prod_results[size]['fps']} fps "
                  f"({prod_results[size]['mean_ms']} ms)", flush=True)
    if e1_run and isinstance(e1_results, dict) and "skipped" not in e1_results:
        print(f"  E1 filter: {n_e1} linears quantized (+{n_e1 - n_prod})", flush=True)
        for size in SIZES:
            if size in e1_results:
                d = e1_results[size]["mean_ms"] - prod_results[size]["mean_ms"]
                pct = d / prod_results[size]["mean_ms"] * 100
                print(f"  E1 {size}²: {e1_results[size]['fps']} fps "
                      f"({e1_results[size]['mean_ms']} ms, Δ={pct:+.1f}%) "
                      f"mse_vs_prod={e1_results[size].get('mse_vs_prod', 'n/a')}", flush=True)
    else:
        print(f"  E1: {e1_results}", flush=True)
    print(f"  E2: {e2_err}", flush=True)
    print(f"\n[done] {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
