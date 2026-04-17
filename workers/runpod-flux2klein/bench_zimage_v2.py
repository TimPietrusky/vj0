#!/usr/bin/env python3
"""Z-Image Turbo img2img — v2: FA3, high-strength + low-step sweep.

Target: input influences but is not visible in output (SD-Turbo-like semantics).
- Strengths 0.9 / 0.95 (high, input influence via noise pattern, not composition)
- Steps 4 / 6 / 8 (minimize, Turbo is distilled)
- Resolutions 256 / 384 / 512
- Flash-Attn 3 via HF kernels-community
- torch.compile(transformer + vae.decoder, mode=default)
- precomputed prompt embeds
- guidance_scale=0
"""
from pathlib import Path
import time, json, sys, subprocess
import numpy as np
import torch
from PIL import Image

OUT_DIR = Path("/workspace/zimage-bench-fa3")
WAVE_DIR = Path("/workspace/waveforms")
PROMPT = "a bright white lightning bolt against a pitch black night sky, dramatic"

RESOLUTIONS = [256, 384, 512]
STRENGTHS = [0.95]  # prior sweep showed strength doesn't affect latency at fixed steps
STEPS_LIST = [4, 6, 8]
SEED = 42
N_WARMUP = 3
N_BENCH = 15


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    # Fast attention. Z-Image passes an attn_mask (text padding), so we need a mask-aware
    # backend. Only *varlen* hub kernels handle mask via cu_seqlens conversion.
    attn_used = "sdpa"
    for backend in ["_flash_3_varlen_hub", "_flash_varlen_hub"]:
        try:
            pipe.transformer.set_attention_backend(backend)
            attn_used = backend
            print(f"[attn] {backend} enabled", flush=True)
            break
        except Exception as e:
            print(f"[attn] {backend} FAILED: {e}", flush=True)

    # Compile
    print("[compile] transformer + vae.decoder (mode=default)", flush=True)
    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)

    # Precompute prompt embeds
    print(f"[encode] {PROMPT!r}", flush=True)
    t0 = time.perf_counter()
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=PROMPT, device="cuda", do_classifier_free_guidance=False, max_sequence_length=256,
    )
    print(f"[encode] {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

    waves = {p.replace(".png", ""): Image.open(WAVE_DIR / p).convert("RGB")
             for p in ["waveform_1.png", "waveform_2.png"]}
    print(f"[inputs] {list(waves.keys())}", flush=True)

    results = []

    for size in RESOLUTIONS:
        inputs = {n: img.resize((size, size), Image.LANCZOS) for n, img in waves.items()}
        for strength in STRENGTHS:
            for steps in STEPS_LIST:
                cell = f"{size}_s{int(strength*100)}_n{steps}"
                print(f"\n=== {cell}", flush=True)

                wave_in = inputs["waveform_1"]
                for w in range(N_WARMUP):
                    tw = time.perf_counter()
                    _ = pipe(
                        prompt=None, prompt_embeds=prompt_embeds,
                        image=wave_in, strength=strength,
                        num_inference_steps=steps, guidance_scale=0.0,
                        height=size, width=size,
                        generator=torch.Generator("cuda").manual_seed(SEED),
                    ).images[0]
                    torch.cuda.synchronize()
                    print(f"  warmup {w}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)

                times_ms = []
                for i in range(N_BENCH):
                    name = "waveform_1" if i % 2 == 0 else "waveform_2"
                    wave_in = inputs[name]
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = pipe(
                        prompt=None, prompt_embeds=prompt_embeds,
                        image=wave_in, strength=strength,
                        num_inference_steps=steps, guidance_scale=0.0,
                        height=size, width=size,
                        generator=torch.Generator("cuda").manual_seed(SEED + i),
                    ).images[0]
                    torch.cuda.synchronize()
                    times_ms.append((time.perf_counter() - t0) * 1000)
                    if i < 2:
                        out.save(OUT_DIR / f"{cell}_{name}_i{i}.png")

                arr = np.array(times_ms)
                mean, p95, mn = arr.mean(), np.percentile(arr, 95), arr.min()
                fps = 1000 / mean
                print(f"  mean={mean:.1f}ms p95={p95:.1f} min={mn:.1f} → {fps:.2f} fps", flush=True)
                results.append({
                    "resolution": size, "strength": strength, "steps": steps,
                    "mean_ms": round(mean, 1), "p95_ms": round(p95, 1),
                    "min_ms": round(mn, 1), "fps": round(fps, 2),
                    "vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
                })
                torch.cuda.reset_peak_memory_stats()

    for name, img in waves.items():
        img.resize((256, 256)).save(OUT_DIR / f"_input_{name}_256.png")

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {len(results)} cells → {OUT_DIR}/results.json", flush=True)
    for r in sorted(results, key=lambda r: r["mean_ms"]):
        print(f"  {r['resolution']:3}² s={r['strength']} n={r['steps']}: {r['mean_ms']:5}ms ({r['fps']:.2f} fps, {r['vram_gb']}GB)", flush=True)


if __name__ == "__main__":
    main()
