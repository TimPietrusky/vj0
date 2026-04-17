#!/usr/bin/env python3
"""Verbatim reproduction of the working stablefast SD-Turbo config, to verify
user's claim that img2img at 1-step-strength-1 actually influences output.

Matrix: wave1, wave2, blank input × dog, beach, city prompt × strength {1.0, 0.9, 0.75}."""
from pathlib import Path
import time, torch, numpy as np
from PIL import Image

OUT_DIR = Path("/workspace/sdturbo-test")
SIZE_IN = 128   # their capture size
SIZE_OUT = 256  # their output size
SEED = 42
PROMPTS = {
    "dog":     "a golden retriever puppy sitting in grass, studio light, photograph",
    "beach":   "colorful wooden house at the beach at sunset, palm trees, warm tones, photograph",
    "city":    "vibrant neon cyberpunk city street at night, rain, reflections",
    "lightning": "a bright white lightning bolt on a pitch black night sky, dramatic",
}


def prep_img(path, out_size):
    img = Image.open(path).convert("RGB").resize((out_size, out_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float16, device="cuda")
    return img, t


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Install deps if needed
    import subprocess, sys
    for pkg in ["diffusers", "transformers", "accelerate", "safetensors"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "--break-system-packages", "-q", pkg])

    from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderTiny
    from huggingface_hub import snapshot_download

    # Download models if not cached (both are small, no HF token needed)
    print("[load] fetching sd-turbo + taesd", flush=True)
    sd_path = snapshot_download("stabilityai/sd-turbo")
    tae_path = snapshot_download("madebyollin/taesd")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        sd_path, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
    )
    pipe.vae = AutoencoderTiny.from_pretrained(tae_path).to(device="cuda", dtype=torch.float16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    print(f"[load] done, vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    def prep(path):
        img_pil, img_t = prep_img(path, SIZE_IN)
        # Upscale to output as the stablefast server does
        img_t_up = torch.nn.functional.interpolate(img_t, size=(SIZE_OUT, SIZE_OUT),
                                                   mode='bilinear', align_corners=False)
        return img_pil, img_t_up

    wave1_pil, wave1_t = prep("/workspace/waveforms/waveform_1.png")
    wave2_pil, wave2_t = prep("/workspace/waveforms/waveform_2.png")

    # blank input: solid black
    blank_t = torch.zeros(1, 3, SIZE_OUT, SIZE_OUT, dtype=torch.float16, device="cuda")

    wave1_pil.save(OUT_DIR / "input_wave1_128.png")
    wave2_pil.save(OUT_DIR / "input_wave2_128.png")

    inputs = {"wave1": wave1_t, "wave2": wave2_t, "blank": blank_t}

    # warmup
    for w in range(3):
        tw = time.perf_counter()
        _ = pipe(prompt="warmup", image=wave1_t, height=SIZE_OUT, width=SIZE_OUT,
                 num_inference_steps=1, strength=1.0, guidance_scale=0).images[0]
        torch.cuda.synchronize()
        print(f"[warmup] {(time.perf_counter()-tw)*1000:.0f}ms", flush=True)

    # Matrix
    for tag, prompt in PROMPTS.items():
        print(f"\n=== {tag}: {prompt!r}", flush=True)
        for strength in [1.0, 0.9, 0.75, 0.5]:
            n_steps = 1 if strength == 1.0 else 2  # >1 step required for strength<1 in SD-Turbo
            for in_name, in_t in inputs.items():
                torch.manual_seed(SEED)
                t0 = time.perf_counter()
                try:
                    out = pipe(prompt=prompt, image=in_t,
                               height=SIZE_OUT, width=SIZE_OUT,
                               num_inference_steps=n_steps, strength=strength,
                               guidance_scale=0).images[0]
                    dt = (time.perf_counter() - t0) * 1000
                    out.save(OUT_DIR / f"{tag}_{in_name}_s{int(strength*100):03d}_n{n_steps}.png")
                    print(f"  {tag} {in_name} n={n_steps} s={strength}: {dt:.0f}ms", flush=True)
                except Exception as e:
                    print(f"  {tag} {in_name} n={n_steps} s={strength}: FAIL {e}", flush=True)

    print(f"[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
