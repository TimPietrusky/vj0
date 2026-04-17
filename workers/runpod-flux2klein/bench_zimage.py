#!/usr/bin/env python3
"""Z-Image Turbo img2img bench on RTX 5090.

Goal: find fastest img2img config.
- bfloat16 + ZImageImg2ImgPipeline
- precompute prompt embeds once per prompt (avoids Qwen3-4B on every frame)
- torch.compile(transformer, mode=default) + vae.decoder
- flash-attn 3 attention backend
- guidance_scale=0.0 (distilled, no CFG)
- sweep resolutions [256, 384, 512] x strengths [0.5, 0.7, 0.9]
- warmup 3, bench 20 runs each cell
"""
from pathlib import Path
import time, json, sys, subprocess
import numpy as np
import torch
from PIL import Image

OUT_DIR = Path("/workspace/zimage-bench")
WAVE_DIR = Path("/workspace/waveforms")
PROMPT = "a bright white lightning bolt against a pitch black night sky, dramatic"

RESOLUTIONS = [256, 384, 512]
STRENGTHS = [0.5, 0.7, 0.9]
STEPS = 8
SEED = 42
N_WARMUP = 3
N_BENCH = 20


def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--break-system-packages", "-q", *args])


def ensure_deps():
    try:
        from diffusers import ZImageImg2ImgPipeline  # noqa
    except Exception:
        print("[deps] ZImageImg2ImgPipeline missing, upgrading diffusers from main", flush=True)
        pip("-U", "git+https://github.com/huggingface/diffusers.git")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_deps()

    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    from diffusers import ZImageImg2ImgPipeline

    print("[load] Tongyi-MAI/Z-Image-Turbo (bf16)", flush=True)
    t0 = time.perf_counter()
    pipe = ZImageImg2ImgPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Flash-attn 3 (sm_120 supports it)
    try:
        pipe.transformer.set_attention_backend("_flash_3")
        print("[attn] using flash-attn 3", flush=True)
    except Exception as e:
        try:
            pipe.transformer.set_attention_backend("flash")
            print(f"[attn] fa3 unavailable ({e}); using flash 2", flush=True)
        except Exception as e2:
            print(f"[attn] sdpa fallback ({e2})", flush=True)

    # Compile (default mode; reduce-overhead can't handle PEFT per flux2 journal)
    print("[compile] transformer + vae.decoder (mode=default)", flush=True)
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)

    # Precompute prompt embeds once (avoids Qwen3-4B on every frame — same trick as flux2)
    print(f"[encode] prompt={PROMPT!r}", flush=True)
    t0 = time.perf_counter()
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=PROMPT, device="cuda", do_classifier_free_guidance=False, max_sequence_length=256,
    )
    shape = prompt_embeds[0].shape if isinstance(prompt_embeds, list) else prompt_embeds.shape
    print(f"[encode] {(time.perf_counter()-t0)*1000:.0f}ms, embeds shape={shape}", flush=True)

    # Load inputs
    waves = {}
    for p in ["waveform_1.png", "waveform_2.png"]:
        waves[p.replace(".png", "")] = Image.open(WAVE_DIR / p).convert("RGB")
    print(f"[inputs] {list(waves.keys())}", flush=True)

    results = []

    for size in RESOLUTIONS:
        # Pre-resize inputs
        inputs = {name: img.resize((size, size), Image.LANCZOS) for name, img in waves.items()}

        for strength in STRENGTHS:
            cell = f"{size}x{size}_s{int(strength*100)}_n{STEPS}"
            print(f"\n=== {cell}", flush=True)

            # Warmup (first call triggers torch.compile per shape)
            wave_in = inputs["waveform_1"]
            for w in range(N_WARMUP):
                tw = time.perf_counter()
                _ = pipe(
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    image=wave_in,
                    strength=strength,
                    num_inference_steps=STEPS,
                    guidance_scale=0.0,
                    height=size, width=size,
                    generator=torch.Generator("cuda").manual_seed(SEED),
                ).images[0]
                torch.cuda.synchronize()
                print(f"  warmup {w}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)

            # Bench
            times_ms = []
            for i in range(N_BENCH):
                name = "waveform_1" if i % 2 == 0 else "waveform_2"
                wave_in = inputs[name]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = pipe(
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    image=wave_in,
                    strength=strength,
                    num_inference_steps=STEPS,
                    guidance_scale=0.0,
                    height=size, width=size,
                    generator=torch.Generator("cuda").manual_seed(SEED + i),
                ).images[0]
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                times_ms.append(dt)
                if i < 4:
                    out.save(OUT_DIR / f"{cell}_{name}_i{i}.png")

            arr = np.array(times_ms)
            mean, p50, p95, mn = arr.mean(), np.median(arr), np.percentile(arr, 95), arr.min()
            fps = 1000 / mean
            print(f"  mean={mean:.1f}ms p50={p50:.1f} p95={p95:.1f} min={mn:.1f} → {fps:.2f} fps", flush=True)
            results.append({
                "resolution": size, "strength": strength, "steps": STEPS,
                "mean_ms": round(mean, 1), "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1), "min_ms": round(mn, 1), "fps": round(fps, 2),
                "vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
            })
            torch.cuda.reset_peak_memory_stats()

    # Also save inputs for visual comparison
    for name, img in waves.items():
        img.resize((256, 256)).save(OUT_DIR / f"_input_{name}_256.png")

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {len(results)} cells, results → {OUT_DIR}/results.json", flush=True)
    for r in results:
        print(f"  {r['resolution']}² s={r['strength']}: {r['mean_ms']}ms ({r['fps']} fps)", flush=True)


if __name__ == "__main__":
    main()
