"""
ControlNet img2img bench for FLUX.1-schnell + InstantX Canny ControlNet.

Goal: prove the pipeline actually preserves input structure (waveform) rather than
defaulting to text2img behavior. Then bench latency.

Note: 5090 has 32GB; FLUX schnell (12B bf16) + ControlNet + T5 exceeds VRAM.
We use enable_model_cpu_offload() — adds latency but works.
"""

import os
import time
import json
import statistics
import gc
from pathlib import Path

import torch
from PIL import Image, ImageOps
from diffusers import (
    FluxControlNetImg2ImgPipeline,
    FluxControlNetPipeline,
    FluxControlNetModel,
)

OUT = Path("/workspace/controlnet-test/out")
OUT.mkdir(parents=True, exist_ok=True)
WAVE1 = "/workspace/waveforms/waveform_1.png"

SCHNELL = "Niansuh/FLUX.1-schnell"
CN_CANNY = "InstantX/FLUX.1-dev-Controlnet-Canny"
CN_UNION = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"

DEVICE = "cuda"
DTYPE = torch.bfloat16


def load_waveform(path, size=512):
    return Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)


def make_blank(size=512, color=(0, 0, 0)):
    return Image.new("RGB", (size, size), color)


def run_test(label, pipe, prompt, control_image, init_image, seed,
             steps=4, controlnet_conditioning_scale=0.7, strength=0.95,
             control_mode=None, is_img2img=True, size=512):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        control_image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=steps,
        guidance_scale=0.0,
        height=size, width=size,
        generator=g,
    )
    if is_img2img:
        kwargs["image"] = init_image
        kwargs["strength"] = strength
    if control_mode is not None:
        kwargs["control_mode"] = control_mode
    out = pipe(**kwargs).images[0]
    out.save(OUT / f"{label}.png")
    print(f"  saved {label}.png")
    return out


def bench(pipe, prompt, control_image, init_image, seed=42, steps=4,
          warmup=5, runs=8, controlnet_conditioning_scale=0.7,
          strength=0.95, control_mode=None, is_img2img=True, size=512):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def one():
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        kw = dict(
            prompt=prompt, control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=steps, guidance_scale=0.0,
            height=size, width=size, generator=g,
        )
        if is_img2img:
            kw["image"] = init_image
            kw["strength"] = strength
        if control_mode is not None:
            kw["control_mode"] = control_mode
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipe(**kw)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    print(f"  Warmup ({warmup}x)...")
    for i in range(warmup):
        ms = one()
        print(f"    warmup {i}: {ms:.1f} ms")
    print(f"  Timed ({runs}x)...")
    times = []
    for i in range(runs):
        ms = one()
        times.append(ms)
        print(f"    run {i}: {ms:.1f} ms")

    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return {
        "size": size, "steps": steps,
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[max(0, int(len(times) * 0.95) - 1)],
        "min_ms": min(times),
        "max_ms": max(times),
        "vram_mb": vram_mb,
        "all_ms": times,
    }


def main():
    results = {}

    waveform = load_waveform(WAVE1, size=512)
    waveform.save(OUT / "_input_waveform.png")
    blank = make_blank(512)
    blank.save(OUT / "_input_blank.png")

    # ===== InstantX Canny + img2img =====
    print("\n=== Loading InstantX Canny ControlNet ===")
    cn = FluxControlNetModel.from_pretrained(CN_CANNY, torch_dtype=DTYPE)
    print("=== Loading FLUX.1-schnell pipe with ControlNet (cpu offload) ===")
    pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
        SCHNELL, controlnet=cn, torch_dtype=DTYPE
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    print("\n--- Test A: Canny img2img, lightning bolt, waveform vs blank, same seed ---")
    PROMPT = "a glowing electric blue lightning bolt on a dark background, dramatic, highly detailed"
    SEED = 12345

    run_test("A1_canny_img2img_waveform", pipe, PROMPT, waveform, waveform, SEED,
             steps=4, controlnet_conditioning_scale=0.8, strength=0.95)
    run_test("A2_canny_img2img_blank", pipe, PROMPT, blank, blank, SEED,
             steps=4, controlnet_conditioning_scale=0.8, strength=0.95)

    PROMPT2 = "a thin white silhouette of a mountain ridge on black background"
    run_test("B1_canny_img2img_waveform_silhouette", pipe, PROMPT2, waveform, waveform, SEED,
             steps=4, controlnet_conditioning_scale=0.9, strength=0.95)
    run_test("B2_canny_img2img_blank_silhouette", pipe, PROMPT2, blank, blank, SEED,
             steps=4, controlnet_conditioning_scale=0.9, strength=0.95)

    # ===== Bench =====
    print("\n=== Bench: Canny img2img, 512x512, 4 steps ===")
    results["canny_img2img_512_4step_cpuoffload"] = bench(
        pipe, PROMPT, waveform, waveform, seed=SEED,
        steps=4, controlnet_conditioning_scale=0.8, strength=0.95,
        warmup=5, runs=8, size=512,
    )

    # Save results
    with open(OUT / "_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
