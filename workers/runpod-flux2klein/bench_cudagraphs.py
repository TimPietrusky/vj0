#!/usr/bin/env python3
"""Try to enable CUDA Graphs on the denoise loop via two routes:
  A. torch.compile(mode="reduce-overhead") after disabling PEFT hooks
  B. Manual torch.cuda.CUDAGraph capture of the transformer forward + step

Goal: shave another 10-20 ms off the 2-step + alpha-blend img2img stack.
"""
import argparse, json, math, time, traceback
from pathlib import Path
import torch
import numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SEED = 42
ALPHA = 0.10
N_STEPS = 2
PROMPT = "a bright white lightning bolt against a pitch black night sky, dramatic, photographic, high contrast"
OUT_DIR = Path("/workspace/flux2-cudagraphs")


def pil2t(img):
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def percentile(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bench(pipe, wave_pil, prompt_embeds, size, runs=10, warmup=5, include_encode=True, alpha=ALPHA, n_steps=N_STEPS, seed=SEED):
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

    def encode_img(img_pil):
        img = img_pil if img_pil.size == (size, size) else img_pil.resize((size, size), Image.LANCZOS)
        t = pil2t(img).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    def run_one(seed_i):
        lat = encode_img(wave_pil) if include_encode else cached_lat
        gen = torch.Generator(device="cuda").manual_seed(seed_i)
        noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device="cuda")
        noisy = alpha * lat + (1 - alpha) * noise
        sigmas = np.linspace(1 - alpha, 0.0, n_steps).tolist()
        return pipe(image=None, prompt=None, prompt_embeds=prompt_embeds,
                    latents=noisy, sigmas=sigmas,
                    height=size, width=size, num_inference_steps=n_steps,
                    generator=torch.Generator(device="cuda").manual_seed(seed_i)).images[0]

    cached_lat = encode_img(wave_pil) if not include_encode else None

    for w in range(warmup):
        tw = time.perf_counter()
        _ = run_one(seed); torch.cuda.synchronize()
        print(f"    warmup {w+1}/{warmup}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)
    lats = []; last = None
    for r in range(runs):
        torch.cuda.synchronize(); t = time.perf_counter()
        last = run_one(seed + r); torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)
    return {
        "mean_ms": round(sum(lats) / len(lats), 2),
        "p50_ms": round(percentile(lats, 50), 2),
        "p95_ms": round(percentile(lats, 95), 2),
        "min_ms": round(min(lats), 2),
        "fps": round(1000 / (sum(lats) / len(lats)), 2),
        "_image": last,
    }


def strip_peft_hooks(module):
    """Walk module tree, remove any forward/backward hooks that came from PEFT."""
    removed = 0
    for m in module.modules():
        for d in (m._forward_hooks, m._forward_pre_hooks, m._backward_hooks,
                  m._backward_pre_hooks, m._state_dict_hooks,
                  m._state_dict_pre_hooks, m._load_state_dict_pre_hooks):
            for k in list(d.keys()):
                hook = d[k]
                mod = getattr(hook, "__module__", "") or ""
                name = getattr(hook, "__qualname__", "") or ""
                if "peft" in mod.lower() or "peft" in name.lower():
                    d.pop(k); removed += 1
    return removed


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
        print(f"[STAGE] {name} OK ({out['setup_s']}s)  mean={out['mean_ms']}ms "
              f"min={out['min_ms']}ms p95={out['p95_ms']}ms fps={out['fps']}", flush=True)
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

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

    wave_pil = Image.open("/workspace/waveforms/waveform_1.png").convert("RGB")
    r = pipe.encode_prompt(prompt=PROMPT, device="cuda",
                           num_images_per_prompt=1, max_sequence_length=64)
    prompt_embeds = r[0] if isinstance(r, tuple) else r

    results = {}

    # ---- Reference: baseline compile mode=default @ 256 & 512 ---- #
    def ref_256():
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
        return bench(pipe, wave_pil, prompt_embeds, 256, runs=10, warmup=5)
    stage("A_reference_default_256", ref_256, results, OUT_DIR)

    def ref_512():
        return bench(pipe, wave_pil, prompt_embeds, 512, runs=10, warmup=5)
    stage("A_reference_default_512", ref_512, results, OUT_DIR)

    # Unwrap for next attempts
    def unwrap():
        if hasattr(pipe.transformer, "_orig_mod"):
            pipe.transformer = pipe.transformer._orig_mod
        if hasattr(pipe.vae.encoder, "_orig_mod"):
            pipe.vae.encoder = pipe.vae.encoder._orig_mod
        if hasattr(pipe.vae.decoder, "_orig_mod"):
            pipe.vae.decoder = pipe.vae.decoder._orig_mod

    # ---- Route A: strip PEFT hooks + try reduce-overhead ---- #
    def route_a_512():
        unwrap()
        r1 = strip_peft_hooks(pipe.transformer)
        r2 = strip_peft_hooks(pipe.vae.encoder)
        r3 = strip_peft_hooks(pipe.vae.decoder)
        print(f"  [A] stripped hooks: transformer={r1} enc={r2} dec={r3}", flush=True)

        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=False, dynamic=False)
        pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="reduce-overhead", fullgraph=False, dynamic=False)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="reduce-overhead", fullgraph=False, dynamic=False)
        return bench(pipe, wave_pil, prompt_embeds, 512, runs=10, warmup=7)
    stage("B_reduce_overhead_512", route_a_512, results, OUT_DIR)

    def route_a_256():
        return bench(pipe, wave_pil, prompt_embeds, 256, runs=10, warmup=7)
    stage("B_reduce_overhead_256", route_a_256, results, OUT_DIR)

    # Reset to default for next stage
    unwrap()
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)

    # ---- Route B: torch._inductor set config for cudagraphs ---- #
    def route_b_512():
        import torch._inductor.config
        try:
            torch._inductor.config.triton.cudagraphs = True
            torch._inductor.config.triton.cudagraph_trees = False
            print("  [C] cudagraphs=True, cudagraph_trees=False", flush=True)
        except Exception as e:
            print(f"  [C] config set failed: {e}", flush=True)
        unwrap()
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        return bench(pipe, wave_pil, prompt_embeds, 512, runs=10, warmup=7)
    stage("C_inductor_cudagraphs_512", route_b_512, results, OUT_DIR)

    # Summary
    print(f"\n{'='*70}\nCUDA GRAPHS ATTEMPTS SUMMARY\n{'='*70}", flush=True)
    print(f"{'stage':<32} {'mean':>10} {'min':>10} {'fps':>6}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<32}  FAILED: {r['error'][:50]}")
        else:
            print(f"{name:<32} {r['mean_ms']:>8}ms {r['min_ms']:>8}ms {r['fps']:>5}")

    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
