#!/usr/bin/env python3
"""Max-speed v3:
- fixed fuse_qkv path (transformer.fuse_qkv_projections, not pipe.)
- sage is applied BEFORE compile (so compile captures sage kernels, not SDPA)
- includes VAE encode in timed region
- 2-step baseline (proven visually acceptable) as the starting point
- resolution sweep at the final winning config
"""
import argparse, json, math, time, traceback, subprocess, sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SEED = 42
N_STEPS_DEFAULT = 2
ALPHA = 0.10
MAX_SEQ_LEN = 64
PROMPT = "a bright white lightning bolt against a pitch black night sky, dramatic, photographic, high contrast"
OUT_DIR = Path("/workspace/flux2-max-v3")

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
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"  pip stderr: {r.stderr[-600:]}", flush=True)
    return r.returncode == 0


def restore_sdpa():
    import torch.nn.functional as F
    if _ORIG_SDPA is not None:
        F.scaled_dot_product_attention = _ORIG_SDPA


def unwrap_compile(pipe):
    for name in ("transformer",):
        m = getattr(pipe, name)
        if hasattr(m, "_orig_mod"):
            setattr(pipe, name, m._orig_mod)
    for name in ("decoder", "encoder"):
        m = getattr(pipe.vae, name)
        if hasattr(m, "_orig_mod"):
            setattr(pipe.vae, name, m._orig_mod)


def apply_compile(pipe):
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)


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
        return pipe(image=None, prompt=None, prompt_embeds=prompt_embeds,
                    latents=noisy, sigmas=sigmas,
                    height=size, width=size, num_inference_steps=n_steps,
                    generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
    return run_one


def bench(pipe, wave_pil, prompt_embeds, alpha, size, runs, warmup, n_steps, seed=SEED):
    runner = build_runner(pipe, wave_pil, prompt_embeds, alpha, n_steps, size)
    for w in range(warmup):
        tw = time.perf_counter()
        _ = runner(seed); torch.cuda.synchronize()
        print(f"    warmup {w+1}/{warmup}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)
    lats = []; last = None
    for r in range(runs):
        torch.cuda.synchronize(); t = time.perf_counter()
        last = runner(seed + r); torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)
    return {
        "mean_ms": round(sum(lats) / len(lats), 2),
        "p50_ms": round(percentile(lats, 50), 2),
        "p95_ms": round(percentile(lats, 95), 2),
        "min_ms": round(min(lats), 2),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "fps": round(1000 / (sum(lats) / len(lats)), 2),
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
              f"fps={out['fps']} vram={out['vram_gb']}GB", flush=True)
        return out
    except Exception as e:
        print(f"[STAGE] {name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        traceback.print_exc()
        results[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
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

    r = pipe.encode_prompt(prompt=PROMPT, device="cuda",
                           num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN)
    prompt_embeds = r[0] if isinstance(r, tuple) else r
    print(f"[prep] prompt_embeds {tuple(prompt_embeds.shape)}", flush=True)

    results = {}

    # ---- Stage 1: 2-step baseline @ 512, all compiled, seqlen 64 ---- #
    def s1():
        apply_compile(pipe)
        return bench(pipe, wave_pil, prompt_embeds, ALPHA, 512,
                     runs=10, warmup=5, n_steps=2)
    stage("1_baseline_2step_512", s1, results, OUT_DIR)

    # ---- Stage 2: + fuse_qkv on transformer ---- #
    def s2():
        unwrap_compile(pipe)
        if hasattr(pipe.transformer, "fuse_qkv_projections"):
            pipe.transformer.fuse_qkv_projections()
            print("  [2] transformer.fuse_qkv_projections() applied", flush=True)
        else:
            raise RuntimeError("no fuse_qkv_projections on transformer")
        apply_compile(pipe)
        return bench(pipe, wave_pil, prompt_embeds, ALPHA, 512,
                     runs=10, warmup=5, n_steps=2)
    stage("2_fuse_qkv_2step", s2, results, OUT_DIR)

    # ---- Stage 3: + SageAttention (patch BEFORE compile) ---- #
    def s3():
        try:
            import sageattention
        except ImportError:
            if not pip_install("sageattention"):
                raise RuntimeError("install failed")
            import sageattention
        from sageattention import sageattn
        print(f"  [3] sageattention {getattr(sageattention, '__version__', '?')}", flush=True)

        def sage_sdpa(query=None, key=None, value=None,
                      attn_mask=None, dropout_p=0.0, is_causal=False,
                      scale=None, enable_gqa=False, **kwargs):
            return sageattn(query, key, value, is_causal=is_causal, sm_scale=scale)

        # Patch FIRST
        F.scaled_dot_product_attention = sage_sdpa
        print("  [3] patched F.scaled_dot_product_attention", flush=True)

        unwrap_compile(pipe)
        # Re-apply fuse_qkv since unwrap may've kept it, but to be safe assume state
        if hasattr(pipe.transformer, "fuse_qkv_projections"):
            try: pipe.transformer.fuse_qkv_projections()
            except Exception: pass  # may already be fused
        apply_compile(pipe)
        try:
            return bench(pipe, wave_pil, prompt_embeds, ALPHA, 512,
                         runs=10, warmup=5, n_steps=2)
        except Exception:
            restore_sdpa()
            raise
    stage("3_sage_2step", s3, results, OUT_DIR)

    # Cleanup sage patch
    restore_sdpa()
    unwrap_compile(pipe)
    try: pipe.transformer.fuse_qkv_projections()
    except Exception: pass
    apply_compile(pipe)

    # ---- Stage 4: resolution sweep at 2-step fuse_qkv (winning) ---- #
    for size in (256, 384, 512):
        def sres(sz=size):
            # Recompile will happen automatically on first new shape
            return bench(pipe, wave_pil, prompt_embeds, ALPHA, sz,
                         runs=10, warmup=5, n_steps=2)
        stage(f"4_res_{size}_2step", sres, results, OUT_DIR)

    # ---- Stage 5: 3 steps at 256 for quality margin ---- #
    def s5():
        return bench(pipe, wave_pil, prompt_embeds, ALPHA, 256,
                     runs=10, warmup=5, n_steps=3)
    stage("5_res_256_3step", s5, results, OUT_DIR)

    # ---- Stage 6: 3 steps at 384 ---- #
    def s6():
        return bench(pipe, wave_pil, prompt_embeds, ALPHA, 384,
                     runs=10, warmup=5, n_steps=3)
    stage("6_res_384_3step", s6, results, OUT_DIR)

    print(f"\n{'='*70}\nMAX-SPEED v3 SUMMARY\n{'='*70}", flush=True)
    print(f"{'stage':<28} {'mean':>10} {'min':>10} {'p95':>10} {'fps':>6} {'vram':>8}")
    print("-" * 80)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<28}  FAILED: {r['error'][:60]}")
        else:
            print(f"{name:<28} {r['mean_ms']:>8}ms {r['min_ms']:>8}ms "
                  f"{r['p95_ms']:>8}ms {r['fps']:>5} {r['vram_gb']:>6}GB")

    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
