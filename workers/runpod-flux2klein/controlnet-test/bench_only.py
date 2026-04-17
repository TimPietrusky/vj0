"""Bench-only run for FLUX.1-schnell + InstantX Canny ControlNet img2img."""
import time, json, statistics
from pathlib import Path
import torch
from PIL import Image
from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel

OUT = Path("/workspace/controlnet-test/out")
WAVE1 = "/workspace/waveforms/waveform_1.png"
SCHNELL = "Niansuh/FLUX.1-schnell"
CN_CANNY = "InstantX/FLUX.1-dev-Controlnet-Canny"
DTYPE = torch.bfloat16

waveform = Image.open(WAVE1).convert("RGB").resize((512, 512), Image.BILINEAR)
blank = Image.new("RGB", (512, 512), (0, 0, 0))

cn = FluxControlNetModel.from_pretrained(CN_CANNY, torch_dtype=DTYPE)
pipe = FluxControlNetImg2ImgPipeline.from_pretrained(SCHNELL, controlnet=cn, torch_dtype=DTYPE)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)

PROMPT = "a glowing electric blue lightning bolt on a dark background, dramatic, highly detailed"
PROMPT2 = "a thin white silhouette of a mountain ridge on black background"
SEED = 12345

# Save the silhouette comparison images we missed
def gen(label, prompt, ctrl, init, seed=SEED, scale=0.8, strength=0.95, steps=4):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(prompt=prompt, control_image=ctrl, image=init,
               controlnet_conditioning_scale=scale, strength=strength,
               num_inference_steps=steps, guidance_scale=0.0,
               height=512, width=512, generator=g).images[0]
    out.save(OUT / f"{label}.png")
    print(f"  saved {label}.png")

print("=== Generating remaining test images ===")
gen("B1_canny_img2img_waveform_silhouette", PROMPT2, waveform, waveform, scale=0.9)
gen("B2_canny_img2img_blank_silhouette", PROMPT2, blank, blank, scale=0.9)
# Also try strength=1.0 (pure controlnet+text influence; init image only used for noise pattern)
gen("E1_strength10_waveform", PROMPT, waveform, waveform, scale=0.9, strength=1.0)
gen("E2_strength10_blank", PROMPT, blank, blank, scale=0.9, strength=1.0)
# And a "river" prompt to look for line-following
PROMPT3 = "an aerial photograph of a winding river through mountains, top-down view"
gen("F1_river_waveform", PROMPT3, waveform, waveform, scale=0.85, strength=0.95)
gen("F2_river_blank", PROMPT3, blank, blank, scale=0.85, strength=0.95)

print("\n=== Bench: 512x512, 4 steps, canny img2img (cpu offload) ===")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def one_run():
    g = torch.Generator(device="cuda").manual_seed(SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pipe(prompt=PROMPT, control_image=waveform, image=waveform,
         controlnet_conditioning_scale=0.8, strength=0.95,
         num_inference_steps=4, guidance_scale=0.0,
         height=512, width=512, generator=g)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000

print("Warmup x5:")
for i in range(5):
    ms = one_run()
    print(f"  warmup {i}: {ms:.1f} ms")
print("Timed x8:")
times = []
for i in range(8):
    ms = one_run()
    times.append(ms)
    print(f"  run {i}: {ms:.1f} ms")

vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
result = {
    "config": "FLUX.1-schnell + InstantX FLUX.1-dev Canny ControlNet, img2img, 512x512, 4 steps, bf16, cpu_offload",
    "mean_ms": statistics.mean(times),
    "p50_ms": statistics.median(times),
    "p95_ms": sorted(times)[max(0, int(len(times) * 0.95) - 1)],
    "min_ms": min(times),
    "max_ms": max(times),
    "vram_mb": vram_mb,
    "all_ms": times,
}
print("\n=== RESULT ===")
print(json.dumps(result, indent=2))
with open(OUT / "_bench.json", "w") as f:
    json.dump(result, f, indent=2)
