"""Pure bench: FLUX.1-schnell + InstantX Canny ControlNet img2img, 512, 4 steps."""
import time, json, statistics
import torch
from PIL import Image
from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel

WAVE1 = "/workspace/waveforms/waveform_1.png"
SCHNELL = "Niansuh/FLUX.1-schnell"
CN_CANNY = "InstantX/FLUX.1-dev-Controlnet-Canny"
DTYPE = torch.bfloat16

waveform = Image.open(WAVE1).convert("RGB").resize((512, 512), Image.BILINEAR)

cn = FluxControlNetModel.from_pretrained(CN_CANNY, torch_dtype=DTYPE)
pipe = FluxControlNetImg2ImgPipeline.from_pretrained(SCHNELL, controlnet=cn, torch_dtype=DTYPE)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)

PROMPT = "a glowing electric blue lightning bolt on a dark background, dramatic, highly detailed"
SEED = 12345

def one():
    g = torch.Generator(device="cuda").manual_seed(SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pipe(prompt=PROMPT, control_image=waveform, image=waveform,
         controlnet_conditioning_scale=0.8, strength=0.95,
         num_inference_steps=4, guidance_scale=0.0,
         height=512, width=512, generator=g)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print("Warmup x5:")
for i in range(5):
    print(f"  warmup {i}: {one():.1f} ms")
print("Timed x8:")
times = []
for i in range(8):
    ms = one()
    times.append(ms)
    print(f"  run {i}: {ms:.1f} ms")

vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
result = {
    "config": "FLUX.1-schnell + InstantX FLUX.1-dev Canny ControlNet, img2img, 512x512, 4 steps, bf16, enable_model_cpu_offload",
    "mean_ms": round(statistics.mean(times), 1),
    "p50_ms": round(statistics.median(times), 1),
    "p95_ms": round(sorted(times)[max(0, int(len(times) * 0.95) - 1)], 1),
    "min_ms": round(min(times), 1),
    "max_ms": round(max(times), 1),
    "vram_mb": round(vram_mb, 1),
    "all_ms": [round(t, 1) for t in times],
}
print("\n=== RESULT ===")
print(json.dumps(result, indent=2))
with open("/workspace/controlnet-test/out/_bench.json", "w") as f:
    json.dump(result, f, indent=2)
