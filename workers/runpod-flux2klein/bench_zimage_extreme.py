#!/usr/bin/env python3
"""Push Nunchaku FP4 img2img to the limit: compile transformer, try 2/3 steps, add 192²."""
from pathlib import Path
import time, json
import numpy as np
import torch
from PIL import Image

# Patch nunchaku forward signature (same as bench_zimage_nunchaku.py)
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

OUT_DIR = Path("/workspace/zimage-bench-extreme")
WAVE_DIR = Path("/workspace/waveforms")
PROMPT = "a bright white lightning bolt against a pitch black night sky, dramatic"

RANKS = [32, 128]
RESOLUTIONS = [192, 256, 384]
STEPS_LIST = [2, 3, 4, 6]
STRENGTH = 0.95
SEED = 42
N_WARMUP = 3
N_BENCH = 15
REPO = "nunchaku-ai/nunchaku-z-image-turbo"


def run_variant(rank: int, compile_transformer: bool, results: list):
    from diffusers import ZImageImg2ImgPipeline
    from nunchaku import NunchakuZImageTransformer2DModel
    from nunchaku.utils import get_precision

    precision = get_precision()
    filename = f"svdq-{precision}_r{rank}-z-image-turbo.safetensors"
    tag = f"{precision}_r{rank}{'_tfc' if compile_transformer else ''}"
    print(f"\n\n######## {tag} ########", flush=True)

    t0 = time.perf_counter()
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        f"{REPO}/{filename}", torch_dtype=torch.bfloat16,
    )
    print(f"[load] transformer done in {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    pipe = ZImageImg2ImgPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[load] pipe done in {time.perf_counter()-t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Compile
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
    if compile_transformer:
        pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False, dynamic=False)
        print("[compile] transformer + vae.decoder", flush=True)
    else:
        print("[compile] vae.decoder only", flush=True)

    t0 = time.perf_counter()
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=PROMPT, device="cuda", do_classifier_free_guidance=False, max_sequence_length=256,
    )
    print(f"[encode] {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

    waves = {p.replace(".png", ""): Image.open(WAVE_DIR / p).convert("RGB")
             for p in ["waveform_1.png", "waveform_2.png"]}

    for size in RESOLUTIONS:
        inputs = {n: img.resize((size, size), Image.LANCZOS) for n, img in waves.items()}
        for steps in STEPS_LIST:
            cell = f"{tag}_{size}_n{steps}"
            print(f"\n=== {cell}", flush=True)

            wave_in = inputs["waveform_1"]
            try:
                for w in range(N_WARMUP):
                    tw = time.perf_counter()
                    _ = pipe(
                        prompt=None, prompt_embeds=prompt_embeds,
                        image=wave_in, strength=STRENGTH,
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
                        image=wave_in, strength=STRENGTH,
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
                    "quant": tag, "resolution": size, "steps": steps,
                    "strength": STRENGTH,
                    "mean_ms": round(mean, 1), "p95_ms": round(p95, 1),
                    "min_ms": round(mn, 1), "fps": round(fps, 2),
                    "vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
                    "compile_transformer": compile_transformer,
                })
                torch.cuda.reset_peak_memory_stats()
            except Exception as e:
                print(f"  CELL FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)

    del pipe, transformer
    torch.cuda.empty_cache()
    import gc; gc.collect()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[gpu] {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability()}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    results = []
    # Order: compile=False first (baseline-ish), then compile=True
    for compile_transformer in [False, True]:
        for rank in RANKS:
            try:
                run_variant(rank, compile_transformer, results)
            except Exception as e:
                import traceback
                print(f"[rank {rank} compile={compile_transformer}] FAILED:\n{traceback.format_exc()}", flush=True)

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {len(results)} cells → {OUT_DIR}/results.json", flush=True)
    for r in sorted(results, key=lambda r: r["mean_ms"])[:20]:
        print(f"  {r['quant']:16} {r['resolution']:3}² n={r['steps']}: {r['mean_ms']:5}ms ({r['fps']:.2f} fps)", flush=True)


if __name__ == "__main__":
    main()
