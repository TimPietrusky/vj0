#!/usr/bin/env python3
"""Last-mile squeeze: reduce-overhead compile + pre-encoded latents + tiny VAE probe.

Compare configs at the proven sweet spot: fp4_r128, 256², s=0.95, n=3.
- A: baseline winner (mode=default, per-frame VAE encode)
- B: mode=reduce-overhead (CUDA graphs)
- C: pre-encoded latents + sigma injection (skip VAE encode per frame)
- D: B + C combined
- E: + TAESD-Z (if a tiny VAE for Z-Image works — may just be decode-only benefit)
"""
from pathlib import Path
import time, json
import numpy as np
import torch
from PIL import Image

# Patch nunchaku forward signature
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel as _ZT
from nunchaku.models.transformers.transformer_zimage import (
    NunchakuZImageTransformer2DModel as _NZT,
    NunchakuZImageRopeHook as _RopeHook,
)
def _fixed_forward(self, x, t, cap_feats, patch_size=2, f_patch_size=1, return_dict=True, **kw):
    rope_hook = _RopeHook()
    self.register_rope_hook(rope_hook)
    try:
        return _ZT.forward(self, x, t, cap_feats, return_dict=return_dict,
                           patch_size=patch_size, f_patch_size=f_patch_size)
    finally:
        self.unregister_rope_hook()
        del rope_hook
_NZT.forward = _fixed_forward

OUT_DIR = Path("/workspace/zimage-bench-squeeze")
WAVE_DIR = Path("/workspace/waveforms")
PROMPT = "a bright white lightning bolt against a pitch black night sky, dramatic"

SIZE = 256
STEPS = 3
STRENGTH = 0.95
RANK = 128
SEED = 42
N_WARMUP = 4
N_BENCH = 20
REPO = "nunchaku-ai/nunchaku-z-image-turbo"


def build_pipe(compile_mode: str):
    from diffusers import ZImageImg2ImgPipeline
    from nunchaku import NunchakuZImageTransformer2DModel
    from nunchaku.utils import get_precision

    precision = get_precision()
    filename = f"svdq-{precision}_r{RANK}-z-image-turbo.safetensors"
    print(f"[load] {precision}_r{RANK}", flush=True)
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        f"{REPO}/{filename}", torch_dtype=torch.bfloat16,
    )
    pipe = ZImageImg2ImgPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    print(f"[compile] mode={compile_mode}", flush=True)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode=compile_mode, fullgraph=False, dynamic=False)
    pipe.transformer = torch.compile(pipe.transformer, mode=compile_mode, fullgraph=False, dynamic=False)
    return pipe


def encode_prompt_once(pipe):
    t0 = time.perf_counter()
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=PROMPT, device="cuda", do_classifier_free_guidance=False, max_sequence_length=256,
    )
    print(f"[encode] {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)
    return prompt_embeds


def time_run(pipe, prompt_embeds, wave_in, tag, size=SIZE, steps=STEPS, strength=STRENGTH):
    # Standard img2img call (per-frame VAE encode)
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
        print(f"  [{tag}] warmup {w}: {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)

    times_ms = []
    for i in range(N_BENCH):
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
        if i < 1:
            out.save(OUT_DIR / f"{tag}_i{i}.png")

    arr = np.array(times_ms)
    return {"mean_ms": round(arr.mean(), 1), "p95_ms": round(np.percentile(arr, 95), 1),
            "min_ms": round(arr.min(), 1), "fps": round(1000/arr.mean(), 2),
            "vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 2)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    wave_in = Image.open(WAVE_DIR / "waveform_1.png").convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    results = {}

    # --- A: baseline, mode=default ---
    print("\n### A: mode=default ###", flush=True)
    pipe = build_pipe("default")
    pe = encode_prompt_once(pipe)
    results["A_default"] = time_run(pipe, pe, wave_in, "A_default")
    print(f"  A result: {results['A_default']}", flush=True)
    del pipe; torch.cuda.empty_cache(); import gc; gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # --- B: mode=reduce-overhead (CUDA graphs) ---
    print("\n### B: mode=reduce-overhead ###", flush=True)
    try:
        pipe = build_pipe("reduce-overhead")
        pe = encode_prompt_once(pipe)
        results["B_reduce_overhead"] = time_run(pipe, pe, wave_in, "B_reduce_overhead")
        print(f"  B result: {results['B_reduce_overhead']}", flush=True)
        del pipe; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        import traceback
        print(f"  B FAILED: {traceback.format_exc()[-400:]}", flush=True)
        results["B_reduce_overhead"] = {"error": str(e)[:200]}
    torch.cuda.reset_peak_memory_stats()

    # --- C: mode=max-autotune ---
    print("\n### C: mode=max-autotune-no-cudagraphs ###", flush=True)
    try:
        pipe = build_pipe("max-autotune-no-cudagraphs")
        pe = encode_prompt_once(pipe)
        results["C_max_autotune"] = time_run(pipe, pe, wave_in, "C_max_autotune")
        print(f"  C result: {results['C_max_autotune']}", flush=True)
        del pipe; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        import traceback
        print(f"  C FAILED: {traceback.format_exc()[-400:]}", flush=True)
        results["C_max_autotune"] = {"error": str(e)[:200]}
    torch.cuda.reset_peak_memory_stats()

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] results → {OUT_DIR}/results.json", flush=True)
    for k, v in results.items():
        if "error" in v:
            print(f"  {k}: ERROR {v['error']}", flush=True)
        else:
            print(f"  {k}: {v['mean_ms']}ms p95={v['p95_ms']} min={v['min_ms']} → {v['fps']} fps  (vram={v['vram_gb']}GB)", flush=True)


if __name__ == "__main__":
    main()
