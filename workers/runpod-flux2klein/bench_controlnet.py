"""
FLUX.1-schnell + ControlNet Canny img2img benchmark for vj0.

Goal: prove "input influences composition, prompt dictates content".
The white waveform line should become a lightning bolt / neon strip /
river that follows the horizontal contour of the input — not a copy,
not unrelated.

Run on RTX 5090 (sm_120, CUDA 12.8). Pulls models from open mirrors:
- Base:       YuCollection/FLUX.1-schnell-Diffusers (open mirror of BFL schnell)
- ControlNet: InstantX/FLUX.1-dev-Controlnet-Canny  (open)
              Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0 (open)
"""
import os
import time
import json
import gc
import statistics

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import (
    FluxControlNetPipeline,
    FluxControlNetImg2ImgPipeline,
    FluxControlNetModel,
)

OUTDIR = "/workspace/out"
os.makedirs(OUTDIR, exist_ok=True)

BASE = "YuCollection/FLUX.1-schnell-Diffusers"
CN_INSTANTX = "InstantX/FLUX.1-dev-Controlnet-Canny"
CN_UNION = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"

DEVICE = "cuda"
DTYPE = torch.bfloat16
SEED = 1234
SIZE = 512


# --------------------------------------------------------------------------
# Step 3: build canny edges from the waveform PNGs
# --------------------------------------------------------------------------
def make_canny(src_path: str, out_path: str) -> Image.Image:
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    edges = cv2.Canny(img, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    Image.fromarray(edges_rgb).save(out_path)
    return Image.fromarray(edges_rgb)


def make_blank_canny(out_path: str) -> Image.Image:
    blank = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    Image.fromarray(blank).save(out_path)
    return Image.fromarray(blank)


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
def load_pipeline_cn_only(cn_repo: str):
    """Pure ControlNet-conditioned txt2img (no img2img latents).

    Uses model CPU offload — full bf16 schnell + CN > 32 GB on RTX 5090.
    Offload keeps VAE/text_encoder on CPU, swaps transformer + CN to GPU.
    """
    cn = FluxControlNetModel.from_pretrained(cn_repo, torch_dtype=DTYPE)
    pipe = FluxControlNetPipeline.from_pretrained(
        BASE, controlnet=cn, torch_dtype=DTYPE
    )
    pipe.enable_model_cpu_offload()
    return pipe


def load_pipeline_cn_img2img(cn_repo: str):
    """ControlNet + real img2img — what the user actually wants."""
    cn = FluxControlNetModel.from_pretrained(cn_repo, torch_dtype=DTYPE)
    pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
        BASE, controlnet=cn, torch_dtype=DTYPE
    )
    pipe.enable_model_cpu_offload()
    return pipe


# --------------------------------------------------------------------------
# Test matrix runner
# --------------------------------------------------------------------------
def run_one(pipe, control_img, prompt, scale, *, tag, steps=4, strength=None):
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    kwargs = dict(
        prompt=prompt,
        control_image=control_img,
        controlnet_conditioning_scale=scale,
        height=SIZE,
        width=SIZE,
        num_inference_steps=steps,
        guidance_scale=0.0,  # schnell is distilled (CFG-free)
        generator=g,
    )
    if strength is not None:
        # img2img path — also pass image (init latents)
        kwargs["image"] = control_img
        kwargs["strength"] = strength
    out = pipe(**kwargs).images[0]
    path = os.path.join(OUTDIR, tag + ".png")
    out.save(path)
    print("saved", path)
    return path


def matrix_pass(pipe, kind, wave1, wave2, blank):
    """Exercises the full matrix from Step 4.

    kind: "cnonly" or "img2img" — affects prompt tag only.
    """
    prompts = [
        ("lightning",   "a bright white lightning bolt on black background"),
        ("neon",        "a thin glowing neon line on black background"),
        ("river",       "aerial view of a river meandering through dark terrain"),
    ]
    scales = [0.4, 0.6, 0.8, 1.0]

    # Fix one prompt, sweep scale on wave1 → find sweet spot
    for s in scales:
        run_one(pipe, wave1, prompts[0][1], s,
                tag=f"{kind}_w1_lightning_s{int(s*10):02d}")

    # Pick a mid scale, swap inputs and prompts
    s = 0.7
    run_one(pipe, wave1,  prompts[0][1], s, tag=f"{kind}_w1_lightning_s07")
    run_one(pipe, wave2,  prompts[0][1], s, tag=f"{kind}_w2_lightning_s07")
    run_one(pipe, blank,  prompts[0][1], s, tag=f"{kind}_blank_lightning_s07")
    run_one(pipe, wave1,  prompts[1][1], s, tag=f"{kind}_w1_neon_s07")
    run_one(pipe, wave1,  prompts[2][1], s, tag=f"{kind}_w1_river_s07")


# --------------------------------------------------------------------------
# Step 5: bench
# --------------------------------------------------------------------------
def bench(pipe, control_img, prompt, scale, steps=4, n_warm=3, n_timed=6):
    print(f"\nBENCH: scale={scale} steps={steps}")
    for i in range(n_warm):
        g = torch.Generator(device=DEVICE).manual_seed(SEED + i)
        _ = pipe(
            prompt=prompt, control_image=control_img,
            controlnet_conditioning_scale=scale,
            height=SIZE, width=SIZE,
            num_inference_steps=steps, guidance_scale=0.0,
            generator=g,
        ).images[0]
    torch.cuda.synchronize()

    times = []
    for i in range(n_timed):
        g = torch.Generator(device=DEVICE).manual_seed(SEED + 100 + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = pipe(
            prompt=prompt, control_image=control_img,
            controlnet_conditioning_scale=scale,
            height=SIZE, width=SIZE,
            num_inference_steps=steps, guidance_scale=0.0,
            generator=g,
        ).images[0]
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        print(f"  run {i}: {times[-1]:.1f} ms")

    times.sort()
    out = dict(
        mean_ms=statistics.mean(times),
        p50_ms=statistics.median(times),
        p95_ms=times[int(0.95 * (len(times) - 1))],
        min_ms=min(times),
        vram_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    print(json.dumps(out, indent=2))
    return out


def main():
    # Build canny inputs
    wave1 = make_canny("/workspace/waveform_1.png", os.path.join(OUTDIR, "wave1_canny.png"))
    wave2 = make_canny("/workspace/waveform_2.png", os.path.join(OUTDIR, "wave2_canny.png"))
    blank = make_blank_canny(os.path.join(OUTDIR, "blank_canny.png"))

    print("\n========== InstantX Canny — FluxControlNetPipeline (txt2img+CN) ==========")
    pipe = load_pipeline_cn_only(CN_INSTANTX)
    matrix_pass(pipe, "instantx_cn", wave1, wave2, blank)
    del pipe; gc.collect(); torch.cuda.empty_cache()

    print("\n========== InstantX Canny — FluxControlNetImg2ImgPipeline ==========")
    try:
        pipe = load_pipeline_cn_img2img(CN_INSTANTX)
        # img2img with low strength = preserve more of input
        for strength in (0.6, 0.8, 0.95):
            run_one(pipe, wave1,
                    "a bright white lightning bolt on black background",
                    0.7,
                    tag=f"instantx_i2i_w1_lightning_s07_str{int(strength*100)}",
                    strength=strength)
        del pipe; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print("img2img failed:", e)

    print("\n========== Union Pro 2.0 (mode=canny) ==========")
    try:
        pipe = load_pipeline_cn_only(CN_UNION)
        for s in (0.4, 0.7, 1.0):
            run_one(pipe, wave1,
                    "a bright white lightning bolt on black background",
                    s, tag=f"union_w1_lightning_s{int(s*10):02d}")
        run_one(pipe, wave1,
                "a thin glowing neon line on black background",
                0.7, tag="union_w1_neon_s07")
        run_one(pipe, blank,
                "a bright white lightning bolt on black background",
                0.7, tag="union_blank_lightning_s07")
    except Exception as e:
        print("union failed:", e)
    del pipe; gc.collect(); torch.cuda.empty_cache()

    # Bench the winning config — InstantX canny @ 0.7, schnell 4 steps
    print("\n========== BENCH: InstantX canny + schnell 4-step ==========")
    pipe = load_pipeline_cn_only(CN_INSTANTX)
    bench_result = bench(
        pipe, wave1,
        "a bright white lightning bolt on black background",
        scale=0.7, steps=4,
    )
    with open(os.path.join(OUTDIR, "bench.json"), "w") as f:
        json.dump(bench_result, f, indent=2)


if __name__ == "__main__":
    main()
