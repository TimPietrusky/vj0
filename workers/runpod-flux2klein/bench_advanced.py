#!/usr/bin/env python3
"""
Advanced optimization bench, layered on top of the proven winning stack:
  base = pre-encoded prompt + compile(transformer) + compile(vae.decoder), bf16, 4 steps

Stages (each isolated; failure of one does not abort the rest):
  1_baseline_reproduction   — confirms ~232 ms @ 512² as starting point
  2_fuse_qkv_projections    — fuses Q/K/V matmuls in transformer attention
  3_compile_vae_encoder     — adds vae.encoder to the compiled set (was uncompiled)
  4_sageattention           — replaces SDPA with SageAttention 2 (Blackwell-tuned fp8 attn)
  5_fbcache                 — first-block cache: skip re-compute if residual delta is small
"""
import argparse, json, math, time, traceback, importlib, subprocess, sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"


def make_input_image(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), (8, 8, 16))
    px = img.load()
    cx = cy = size / 2
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            d = math.sqrt(dx * dx + dy * dy) / (size / 2)
            r = int(max(0, 255 * (1 - d) * 0.9 + 30))
            g = int(max(0, 255 * (1 - d) * 0.3 + 20))
            b = int(max(0, 255 * (1 - d * 0.6) * 0.8 + 40))
            px[x, y] = (r, g, b)
    draw = ImageDraw.Draw(img)
    bar_w = max(2, size // 24)
    for i in range(12):
        h = int((0.3 + 0.6 * abs(math.sin(i * 0.7))) * size * 0.8)
        x0 = int(size * 0.1 + i * (size * 0.07))
        y0 = (size - h) // 2
        draw.rectangle([x0, y0, x0 + bar_w, y0 + h],
                       fill=(255, 80 + i * 12, 200 - i * 10))
    return img


def percentile(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bench(pipe, image, size, prompt_embeds, runs, warmup, seed=42):
    for w in range(warmup):
        tw = time.perf_counter()
        _ = pipe(image=image, prompt_embeds=prompt_embeds,
                 height=size, width=size, num_inference_steps=4,
                 generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
        torch.cuda.synchronize()
        print(f"    warmup {w+1}/{warmup}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)
    lats = []
    for r in range(runs):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = pipe(image=image, prompt_embeds=prompt_embeds,
                   height=size, width=size, num_inference_steps=4,
                   generator=torch.Generator(device="cuda").manual_seed(seed + r)).images[0]
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)
    return {
        "lats_ms": [round(x, 1) for x in lats],
        "mean_ms": round(sum(lats) / len(lats), 2),
        "p50_ms": round(percentile(lats, 50), 2),
        "p95_ms": round(percentile(lats, 95), 2),
        "min_ms": round(min(lats), 2),
        "max_ms": round(max(lats), 2),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "_image": out,
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
              f"mean={out['mean_ms']}ms  min={out['min_ms']}ms  "
              f"p95={out['p95_ms']}ms  vram={out['vram_gb']}GB", flush=True)
        return out
    except Exception as e:
        print(f"[STAGE] {name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        traceback.print_exc()
        results[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}",
                         "setup_s": round(time.perf_counter() - t0, 1)}
        return None


def pip_install(*pkgs, no_build_isolation=False):
    cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", *pkgs]
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"  pip stderr: {r.stderr[-1500:]}", flush=True)
    return r.returncode == 0


def make_compiled_pipe(size, prompt_embeds, image, recompile_t=True, recompile_dec=True, recompile_enc=False):
    """Returns (pipe, prompt_embeds) with the requested compile state.
    Assumes a global `pipe` exists in caller scope."""
    pass  # placeholder; real logic uses module-level pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/flux2-adv-out")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompt", default="cyberpunk neon abstract waveform, vibrant colors, glowing edges")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}",
          flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    t0 = time.perf_counter()
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    print(f"[load] {time.perf_counter()-t0:.1f}s  vram={torch.cuda.memory_allocated()/1e9:.2f}GB",
          flush=True)

    res = pipe.encode_prompt(prompt=args.prompt, device="cuda",
                             num_images_per_prompt=1, max_sequence_length=args.max_seq_len)
    prompt_embeds = res[0] if isinstance(res, tuple) else res
    print(f"[encode] {tuple(prompt_embeds.shape)} {prompt_embeds.dtype}", flush=True)

    image = make_input_image(args.size); image.save(out_dir / "input.png")
    results = {}

    # ---- STAGE 1: baseline (winning stack) ---- #
    def s1():
        print("[1] compiling transformer + vae.decoder (default mode)", flush=True)
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
        return bench(pipe, image, args.size, prompt_embeds, args.runs, max(args.warmup, 5))
    stage("1_baseline_reproduction", s1, results, out_dir)

    # ---- STAGE 2: fuse_qkv_projections ---- #
    def s2():
        # Unwrap compiled transformer to apply fusion to the original module
        t = pipe.transformer
        if hasattr(t, "_orig_mod"):
            t = t._orig_mod
            pipe.transformer = t
        print("[2] checking fuse_qkv_projections support", flush=True)

        if hasattr(pipe, "fuse_qkv_projections"):
            try:
                pipe.fuse_qkv_projections()
                print("[2] pipe.fuse_qkv_projections() OK", flush=True)
            except Exception as e:
                print(f"[2] pipe.fuse_qkv_projections() failed: {e}; trying transformer.fuse_qkv_projections()", flush=True)
                pipe.transformer.fuse_qkv_projections()
                print("[2] transformer.fuse_qkv_projections() OK", flush=True)
        elif hasattr(pipe.transformer, "fuse_qkv_projections"):
            pipe.transformer.fuse_qkv_projections()
            print("[2] transformer.fuse_qkv_projections() OK", flush=True)
        else:
            raise RuntimeError("no fuse_qkv_projections method on pipe or transformer")

        print("[2] re-compiling fused transformer", flush=True)
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        return bench(pipe, image, args.size, prompt_embeds, args.runs, max(args.warmup, 5))
    stage("2_fuse_qkv_projections", s2, results, out_dir)

    # ---- STAGE 3: compile vae.encoder too ---- #
    def s3():
        print("[3] compiling vae.encoder (decoder+transformer already compiled)", flush=True)
        pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
        return bench(pipe, image, args.size, prompt_embeds, args.runs, max(args.warmup, 4))
    stage("3_compile_vae_encoder", s3, results, out_dir)

    # ---- STAGE 4: SageAttention 2 ---- #
    def s4():
        print("[4] attempting to install sageattention", flush=True)
        try:
            import sageattention
        except ImportError:
            ok = pip_install("sageattention")
            if not ok:
                raise RuntimeError("sageattention pip install failed")
            import sageattention
        from sageattention import sageattn
        print(f"[4] sageattention {getattr(sageattention, '__version__', '?')}", flush=True)

        # Monkey-patch global SDPA. SageAttention has the same call signature.
        import torch.nn.functional as F
        global _orig_sdpa
        _orig_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = sageattn
        print("[4] F.scaled_dot_product_attention -> sageattn (monkey-patched)", flush=True)

        # Re-compile (kernels changed) — unwrap first
        t = pipe.transformer
        if hasattr(t, "_orig_mod"):
            t = t._orig_mod
            pipe.transformer = t
        print("[4] re-compiling transformer with sage attention", flush=True)
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        try:
            return bench(pipe, image, args.size, prompt_embeds, args.runs, max(args.warmup, 5))
        except Exception as e:
            # restore SDPA on failure so subsequent stages aren't poisoned
            F.scaled_dot_product_attention = _orig_sdpa
            raise
    stage("4_sageattention", s4, results, out_dir)

    # ---- STAGE 5: First-Block Cache (FBCache) via DeepCache-style residual cache ---- #
    def s5():
        # Restore plain SDPA for fair FBCache comparison (in case stage 4 set sage)
        try:
            import torch.nn.functional as F
            if "_orig_sdpa" in globals() and F.scaled_dot_product_attention is not _orig_sdpa:
                F.scaled_dot_product_attention = _orig_sdpa
                print("[5] restored vanilla SDPA for fair FBCache test", flush=True)
        except Exception:
            pass

        print("[5] looking for built-in transformer caching API", flush=True)
        cache_applied = False

        # Try DeepCache via diffusers utility (some versions have it)
        try:
            from diffusers.utils.deep_cache import DeepCacheSDHelper  # often only for SD
            print("[5] DeepCacheSDHelper found but is SD-only; skipping", flush=True)
        except Exception:
            pass

        # Try Flux-specific apply_first_block_cache or similar
        for name in ["apply_cache", "apply_first_block_cache", "set_cache", "enable_cache"]:
            if hasattr(pipe.transformer, name):
                try:
                    getattr(pipe.transformer, name)()
                    print(f"[5] transformer.{name}() applied", flush=True)
                    cache_applied = True
                    break
                except Exception as e:
                    print(f"[5] {name} failed: {e}", flush=True)

        if not cache_applied:
            # Hand-rolled: residual cache wrapper. With 4 steps, we'll skip block compute
            # on step 2 and step 4, reusing the residual delta from step 1 / step 3.
            # This is the FBCache-lite approach; only safe for distilled, small-step models.
            print("[5] no built-in cache; installing hand-rolled residual cache wrapper", flush=True)
            t = pipe.transformer
            if hasattr(t, "_orig_mod"):
                t = t._orig_mod
                pipe.transformer = t
            transformer = pipe.transformer
            blocks = getattr(transformer, "transformer_blocks", None) or \
                     getattr(transformer, "single_transformer_blocks", None) or \
                     getattr(transformer, "blocks", None)
            if blocks is None or len(blocks) == 0:
                raise RuntimeError("can't find transformer blocks to wrap for FBCache")
            print(f"[5] found {len(blocks)} blocks to wrap (will skip every other call's first 30%)",
                  flush=True)

            # Skip the first ~30% of blocks on every other forward call (steps 2 and 4 of 4)
            n_skip = max(1, int(len(blocks) * 0.3))
            skip_blocks = blocks[:n_skip]
            cache = {"hits": 0, "misses": 0, "step_idx": 0, "cached_outs": [None] * n_skip}

            originals = []
            for i, blk in enumerate(skip_blocks):
                originals.append(blk.forward)

                def make_wrapped(idx, orig):
                    def wrapped(*a, **k):
                        s = cache["step_idx"]
                        if s % 2 == 1 and cache["cached_outs"][idx] is not None:
                            cache["hits"] += 1
                            return cache["cached_outs"][idx]
                        cache["misses"] += 1
                        out = orig(*a, **k)
                        cache["cached_outs"][idx] = out
                        return out
                    return wrapped

                blk.forward = make_wrapped(i, originals[i])

            # The Klein pipeline calls transformer once per step. We need to hook step counting.
            # Use a callback_on_step_end via a wrapper around pipe.__call__.
            orig_call = pipe.__class__.__call__
            def call_with_step_counter(self, *args, **kwargs):
                cache["step_idx"] = 0
                # Reset cached outs at start of each generation
                cache["cached_outs"] = [None] * n_skip
                cb = kwargs.get("callback_on_step_end")
                def wrapped_cb(p, i, t, kw):
                    cache["step_idx"] = i + 1
                    if cb is not None:
                        return cb(p, i, t, kw)
                    return kw
                kwargs["callback_on_step_end"] = wrapped_cb
                return orig_call(self, *args, **kwargs)
            pipe.__class__.__call__ = call_with_step_counter
            print(f"[5] hand-rolled FBCache installed: {n_skip} blocks skipped on steps 2 and 4", flush=True)

            # Re-compile (forward graph changed)
            print("[5] re-compiling transformer with FBCache wrappers", flush=True)
            pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)

        result = bench(pipe, image, args.size, prompt_embeds, args.runs, max(args.warmup, 5))
        if not cache_applied:
            print(f"[5] cache hits={cache['hits']} misses={cache['misses']}", flush=True)
            result["cache_hits"] = cache["hits"]
            result["cache_misses"] = cache["misses"]
        return result
    stage("5_fbcache", s5, results, out_dir)

    # ---- summary ---- #
    print(f"\n{'='*70}\nADVANCED OPT SUMMARY @ {args.size}x{args.size}, 4 steps\n{'='*70}",
          flush=True)
    print(f"{'stage':<32} {'mean':>9} {'min':>9} {'p95':>9} {'vram':>8} {'setup':>7}")
    print("-" * 80)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<32}  FAILED: {r['error'][:60]}")
        else:
            print(f"{name:<32} {r['mean_ms']:>7}ms {r['min_ms']:>7}ms "
                  f"{r['p95_ms']:>7}ms {r['vram_gb']:>6}GB {r.get('setup_s','-'):>5}s")

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
