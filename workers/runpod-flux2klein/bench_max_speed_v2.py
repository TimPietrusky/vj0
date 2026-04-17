#!/usr/bin/env python3
"""Max-speed bench v2, fixed:
- SageAttention wrapper handles keyword SDPA call signature from diffusers
- VAE encode inside the timed loop (real per-frame latency for live VJ)
- Each stage isolates side effects (restores SDPA, unwraps compile) so one
  failure can't poison the rest
"""
import argparse, json, math, time, traceback, subprocess, sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SIZE = 512
SEED = 42
N_STEPS = 4
ALPHA = 0.10
PROMPT = "a bright white lightning bolt against a pitch black night sky, dramatic, photographic, high contrast"
OUT_DIR = Path("/workspace/flux2-max-speed-v2")

_ORIG_SDPA = None


def pil2t(img):
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def percentile(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pip_install(*pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", *pkgs]
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"  pip stderr: {r.stderr[-1000:]}", flush=True)
    return r.returncode == 0


def restore_sdpa():
    """Undo any monkey-patch of F.scaled_dot_product_attention."""
    global _ORIG_SDPA
    import torch.nn.functional as F
    if _ORIG_SDPA is not None:
        F.scaled_dot_product_attention = _ORIG_SDPA
        print("  [restore] SDPA reset to original", flush=True)


def unwrap_compile(pipe):
    """Get the pre-compile modules back so we can recompile cleanly."""
    if hasattr(pipe.transformer, "_orig_mod"):
        pipe.transformer = pipe.transformer._orig_mod
    if hasattr(pipe.vae.decoder, "_orig_mod"):
        pipe.vae.decoder = pipe.vae.decoder._orig_mod
    if hasattr(pipe.vae.encoder, "_orig_mod"):
        pipe.vae.encoder = pipe.vae.encoder._orig_mod


def apply_compile_base(pipe):
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)


def build_runner(pipe, wave_pil, prompt_embeds, alpha, n_steps, include_encode):
    """Returns a closure that runs one frame — optionally including VAE encode
    of the waveform in the timed region (mimics live VJ per-frame cost)."""
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

    def encode_img(img_pil):
        t = pil2t(img_pil).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    if not include_encode:
        fixed_lat = encode_img(wave_pil)

    def run_one(seed):
        lat = encode_img(wave_pil) if include_encode else fixed_lat
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device="cuda")
        noisy = alpha * lat + (1 - alpha) * noise
        sigmas = np.linspace(1 - alpha, 0.0, n_steps).tolist()
        return pipe(image=None, prompt=None, prompt_embeds=prompt_embeds,
                    latents=noisy, sigmas=sigmas,
                    height=SIZE, width=SIZE, num_inference_steps=n_steps,
                    generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]

    return run_one


def bench(pipe, wave_pil, prompt_embeds, alpha, runs, warmup, n_steps=N_STEPS,
          include_encode=False, seed=SEED):
    runner = build_runner(pipe, wave_pil, prompt_embeds, alpha, n_steps, include_encode)

    for w in range(warmup):
        tw = time.perf_counter()
        _ = runner(seed)
        torch.cuda.synchronize()
        print(f"    warmup {w+1}/{warmup}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)

    lats = []
    last = None
    for r in range(runs):
        torch.cuda.synchronize(); t = time.perf_counter()
        last = runner(seed + r)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)

    return {
        "mean_ms": round(sum(lats) / len(lats), 2),
        "p50_ms": round(percentile(lats, 50), 2),
        "p95_ms": round(percentile(lats, 95), 2),
        "min_ms": round(min(lats), 2),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "_image": last,
    }


def stage(name, fn, results, out_dir):
    print(f"\n{'='*70}\n[STAGE] {name}\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    try:
        out = fn()
        out["setup_s"] = round(time.perf_counter() - t0, 1)
        img = out.pop("_image", None)
        if img is not None:
            img.save(out_dir / f"{name}.png")
        results[name] = out
        print(f"[STAGE] {name} OK ({out['setup_s']}s)  "
              f"mean={out['mean_ms']}ms min={out['min_ms']}ms p95={out['p95_ms']}ms "
              f"vram={out['vram_gb']}GB", flush=True)
        return out
    except Exception as e:
        print(f"[STAGE] {name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        traceback.print_exc()
        results[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}",
                         "setup_s": round(time.perf_counter() - t0, 1)}
        return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    global _ORIG_SDPA
    import torch.nn.functional as F
    _ORIG_SDPA = F.scaled_dot_product_attention

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

    wave_pil = Image.open("/workspace/waveforms/waveform_1.png").convert("RGB")
    if wave_pil.size != (SIZE, SIZE):
        wave_pil = wave_pil.resize((SIZE, SIZE), Image.LANCZOS)

    r = pipe.encode_prompt(prompt=PROMPT, device="cuda",
                           num_images_per_prompt=1, max_sequence_length=128)
    prompt_embeds = r[0] if isinstance(r, tuple) else r

    results = {}

    # ---- STAGE 1: compile baseline (T + VAE.dec + VAE.enc) ---- #
    def s1():
        apply_compile_base(pipe)
        return bench(pipe, wave_pil, prompt_embeds, ALPHA,
                     runs=10, warmup=5, include_encode=True)
    stage("1_compile_all_with_encode", s1, results, OUT_DIR)

    # ---- STAGE 2: + fuse_qkv ---- #
    def s2():
        unwrap_compile(pipe)
        pipe.fuse_qkv_projections()
        print("  [2] fuse_qkv_projections applied", flush=True)
        apply_compile_base(pipe)
        return bench(pipe, wave_pil, prompt_embeds, ALPHA,
                     runs=10, warmup=5, include_encode=True)
    stage("2_fuse_qkv", s2, results, OUT_DIR)

    # ---- STAGE 3: + SageAttention (with proper kwarg wrapper) ---- #
    def s3():
        try:
            import sageattention
        except ImportError:
            if not pip_install("sageattention"):
                raise RuntimeError("sageattention pip install failed")
            import sageattention
        from sageattention import sageattn
        print(f"  [3] sageattention {getattr(sageattention, '__version__', '?')}", flush=True)

        # Wrap: diffusers passes query=, key=, value=, attn_mask=, etc. as kwargs
        # sageattn signature: sageattn(q, k, v, tensor_layout='HND', is_causal=False, sm_scale=None, ...)
        def sage_sdpa_compat(query=None, key=None, value=None,
                             attn_mask=None, dropout_p=0.0, is_causal=False,
                             scale=None, enable_gqa=False, **kwargs):
            if query is None or key is None or value is None:
                raise TypeError("sage_sdpa_compat needs query/key/value")
            return sageattn(query, key, value,
                            is_causal=is_causal, sm_scale=scale)

        F.scaled_dot_product_attention = sage_sdpa_compat
        print("  [3] monkey-patched F.scaled_dot_product_attention", flush=True)

        unwrap_compile(pipe)
        apply_compile_base(pipe)
        try:
            return bench(pipe, wave_pil, prompt_embeds, ALPHA,
                         runs=10, warmup=5, include_encode=True)
        except Exception:
            restore_sdpa()
            unwrap_compile(pipe)
            apply_compile_base(pipe)
            raise
    stage("3_sageattention", s3, results, OUT_DIR)

    # After sage stage (regardless of success), restore vanilla SDPA for
    # clean subsequent comparisons. Re-compile.
    restore_sdpa()
    unwrap_compile(pipe)
    apply_compile_base(pipe)

    # ---- STAGE 4: seqlen 128 -> 64 ---- #
    def s4():
        r_short = pipe.encode_prompt(prompt=PROMPT, device="cuda",
                                     num_images_per_prompt=1, max_sequence_length=64)
        pe_short = r_short[0] if isinstance(r_short, tuple) else r_short
        print(f"  [4] prompt_embeds {tuple(pe_short.shape)}", flush=True)
        unwrap_compile(pipe)
        apply_compile_base(pipe)
        return bench(pipe, wave_pil, pe_short, ALPHA,
                     runs=10, warmup=5, include_encode=True)
    stage("4_seqlen_64", s4, results, OUT_DIR)

    # ---- STAGE 5: 3 steps ---- #
    def s5():
        return bench(pipe, wave_pil, prompt_embeds, ALPHA,
                     runs=10, warmup=5, include_encode=True, n_steps=3)
    stage("5_steps_3", s5, results, OUT_DIR)

    # ---- STAGE 6: 2 steps ---- #
    def s6():
        return bench(pipe, wave_pil, prompt_embeds, ALPHA,
                     runs=10, warmup=5, include_encode=True, n_steps=2)
    stage("6_steps_2", s6, results, OUT_DIR)

    # Summary
    print(f"\n{'='*70}\nMAX-SPEED v2 (INCLUDES VAE ENCODE) @ {SIZE}×{SIZE}, α={ALPHA}\n{'='*70}",
          flush=True)
    print(f"{'stage':<28} {'mean':>10} {'min':>10} {'p95':>10} {'vram':>8} {'setup':>8}")
    print("-" * 80)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<28}  FAILED: {r['error'][:60]}")
        else:
            print(f"{name:<28} {r['mean_ms']:>8}ms {r['min_ms']:>8}ms "
                  f"{r['p95_ms']:>8}ms {r['vram_gb']:>6}GB {r.get('setup_s','-'):>6}s")

    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
