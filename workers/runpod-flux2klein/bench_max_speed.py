#!/usr/bin/env python3
"""Max-speed optimization bench for Klein + alpha-blend img2img.

Each stage is cumulative on top of the previous; failures are isolated.
Test config: waveform_1 input, "lightning" prompt, alpha=0.10, 4 steps, 512x512.
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
OUT_DIR = Path("/workspace/flux2-max-speed")


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


def bench(pipe, image_latents, prompt_embeds, alpha, runs, warmup, seed=SEED):
    def run_one(s):
        gen = torch.Generator(device="cuda").manual_seed(s)
        noise = torch.randn(image_latents.shape, generator=gen,
                            dtype=image_latents.dtype, device="cuda")
        noisy = alpha * image_latents + (1 - alpha) * noise
        sigmas = np.linspace(1 - alpha, 0.0, N_STEPS).tolist()
        return pipe(image=None, prompt=None, prompt_embeds=prompt_embeds,
                    latents=noisy, sigmas=sigmas,
                    height=SIZE, width=SIZE, num_inference_steps=N_STEPS,
                    generator=torch.Generator(device="cuda").manual_seed(s)).images[0]

    for w in range(warmup):
        tw = time.perf_counter()
        _ = run_one(seed)
        torch.cuda.synchronize()
        print(f"    warmup {w+1}/{warmup}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)

    lats = []
    last_img = None
    for r in range(runs):
        torch.cuda.synchronize(); t = time.perf_counter()
        last_img = run_one(seed + r)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)

    return {
        "mean_ms": round(sum(lats) / len(lats), 2),
        "p50_ms": round(percentile(lats, 50), 2),
        "p95_ms": round(percentile(lats, 95), 2),
        "min_ms": round(min(lats), 2),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "_image": last_img,
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

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

    # Prep inputs
    def encode_img(path):
        img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        t = pil2t(img).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    wave_latent = encode_img("/workspace/waveforms/waveform_1.png")

    def encode_prompt(max_seq_len=128):
        r = pipe.encode_prompt(prompt=PROMPT, device="cuda",
                               num_images_per_prompt=1, max_sequence_length=max_seq_len)
        return r[0] if isinstance(r, tuple) else r

    prompt_embeds = encode_prompt(128)
    print(f"[prep] wave_latent {tuple(wave_latent.shape)} prompt {tuple(prompt_embeds.shape)}",
          flush=True)

    results = {}

    # Stage 1: baseline — compile T + VAE.dec, alpha-blend img2img
    def s1():
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
        return bench(pipe, wave_latent, prompt_embeds, ALPHA, runs=8, warmup=5)
    stage("1_baseline_compiled", s1, results, OUT_DIR)

    # Stage 2: + compile VAE encoder
    def s2():
        pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
        return bench(pipe, wave_latent, prompt_embeds, ALPHA, runs=8, warmup=3)
    stage("2_compile_vae_encoder", s2, results, OUT_DIR)

    # Stage 3: + fuse_qkv_projections
    def s3():
        t = pipe.transformer
        if hasattr(t, "_orig_mod"):
            t = t._orig_mod; pipe.transformer = t
        if hasattr(pipe, "fuse_qkv_projections"):
            pipe.fuse_qkv_projections()
            print("[3] pipe.fuse_qkv_projections() applied", flush=True)
        elif hasattr(pipe.transformer, "fuse_qkv_projections"):
            pipe.transformer.fuse_qkv_projections()
            print("[3] transformer.fuse_qkv_projections() applied", flush=True)
        else:
            raise RuntimeError("no fuse_qkv_projections on pipe or transformer")
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        return bench(pipe, wave_latent, prompt_embeds, ALPHA, runs=8, warmup=4)
    stage("3_fuse_qkv", s3, results, OUT_DIR)

    # Stage 4: + SageAttention 2
    def s4():
        try:
            import sageattention
        except ImportError:
            if not pip_install("sageattention"):
                raise RuntimeError("sageattention pip install failed")
            import sageattention
        from sageattention import sageattn
        print(f"[4] sageattention {getattr(sageattention, '__version__', '?')}", flush=True)

        import torch.nn.functional as F
        F.scaled_dot_product_attention = sageattn
        print("[4] monkey-patched F.scaled_dot_product_attention -> sageattn", flush=True)

        t = pipe.transformer
        if hasattr(t, "_orig_mod"):
            t = t._orig_mod; pipe.transformer = t
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        return bench(pipe, wave_latent, prompt_embeds, ALPHA, runs=8, warmup=5)
    stage("4_sageattention", s4, results, OUT_DIR)

    # Stage 5: max_seq_len 128 -> 64
    def s5():
        pe_short = encode_prompt(64)
        print(f"[5] prompt_embeds with max_seq=64: {tuple(pe_short.shape)}", flush=True)
        # re-compile because prompt_embeds dim changed
        t = pipe.transformer
        if hasattr(t, "_orig_mod"):
            t = t._orig_mod; pipe.transformer = t
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        return bench(pipe, wave_latent, pe_short, ALPHA, runs=8, warmup=5)
    stage("5_seqlen_64", s5, results, OUT_DIR)

    # Stage 6: try 3 steps (Klein is distilled for 4 — might degrade)
    def s6():
        global N_STEPS_LOCAL
        def run_one(s, n):
            gen = torch.Generator(device="cuda").manual_seed(s)
            noise = torch.randn(wave_latent.shape, generator=gen,
                                dtype=wave_latent.dtype, device="cuda")
            noisy = ALPHA * wave_latent + (1 - ALPHA) * noise
            sigmas = np.linspace(1 - ALPHA, 0.0, n).tolist()
            pe = encode_prompt(64)
            return pipe(image=None, prompt=None, prompt_embeds=pe,
                        latents=noisy, sigmas=sigmas,
                        height=SIZE, width=SIZE, num_inference_steps=n,
                        generator=torch.Generator(device="cuda").manual_seed(s)).images[0]

        # warmup for 3-step
        for w in range(5):
            tw = time.perf_counter()
            _ = run_one(SEED, 3)
            torch.cuda.synchronize()
            print(f"    warmup {w+1}/5 (n=3): {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)

        lats = []
        img = None
        for r in range(8):
            torch.cuda.synchronize(); t = time.perf_counter()
            img = run_one(SEED + r, 3)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t) * 1000)

        return {
            "mean_ms": round(sum(lats) / len(lats), 2),
            "p50_ms": round(percentile(lats, 50), 2),
            "p95_ms": round(percentile(lats, 95), 2),
            "min_ms": round(min(lats), 2),
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "_image": img,
        }
    stage("6_steps_3", s6, results, OUT_DIR)

    # Summary
    print(f"\n{'='*70}\nMAX-SPEED STACK @ {SIZE}×{SIZE}, α={ALPHA}\n{'='*70}", flush=True)
    print(f"{'stage':<30} {'mean':>10} {'min':>10} {'p95':>10} {'vram':>8} {'setup':>8}")
    print("-" * 80)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<30}  FAILED: {r['error'][:55]}")
        else:
            print(f"{name:<30} {r['mean_ms']:>8}ms {r['min_ms']:>8}ms "
                  f"{r['p95_ms']:>8}ms {r['vram_gb']:>6}GB {r.get('setup_s','-'):>6}s")

    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
