#!/usr/bin/env python3
"""Klein with 6/8/12 steps + sigma injection. Test if more denoising budget
closes the gap for unrelated prompts (dog, beach)."""
from pathlib import Path
import time, torch, numpy as np
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SIZE = 512
SEED = 42
OUT_DIR = Path("/workspace/flux2-more-steps")
PROMPTS = {
    "dog":   "a golden retriever puppy sitting in grass, studio light, sharp focus, photograph",
    "beach": "a colorful wooden house at the beach at sunset, palm trees, warm tones, photograph",
    "city":  "vibrant neon cyberpunk city street at night, rain, reflections, wide angle",
}
STEP_STRENGTH = [(4, 0.85), (6, 0.75), (6, 0.85), (8, 0.70), (8, 0.80), (8, 0.90),
                 (12, 0.70), (12, 0.80)]


def pil2t(img):
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[init] {torch.cuda.get_device_name(0)}", flush=True)

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

    # NOTE: skipping compile — dynamic steps would cause recompiles per config

    def encode(img):
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.LANCZOS)
        t = pil2t(img).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    wave1 = encode(Image.open("/workspace/waveforms/waveform_1.png").convert("RGB"))
    wave2 = encode(Image.open("/workspace/waveforms/waveform_2.png").convert("RGB"))
    blank = encode(Image.new("RGB", (SIZE, SIZE), (0, 0, 0)))
    inputs = {"wave1": wave1, "wave2": wave2, "blank": blank}

    embeds = {}
    for tag, p in PROMPTS.items():
        r = pipe.encode_prompt(prompt=p, device="cuda",
                               num_images_per_prompt=1, max_sequence_length=128)
        embeds[tag] = r[0] if isinstance(r, tuple) else r

    def run(lat, tag, n_steps, strength, seed=SEED):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device="cuda")
        noisy = (1 - strength) * lat + strength * noise
        sigmas = np.linspace(strength, 0.0, n_steps).tolist()
        return pipe(image=None, prompt=None, prompt_embeds=embeds[tag],
                    latents=noisy, sigmas=sigmas,
                    height=SIZE, width=SIZE, num_inference_steps=n_steps,
                    generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]

    for tag in PROMPTS:
        for n_steps, strength in STEP_STRENGTH:
            for in_name, lat in inputs.items():
                t0 = time.perf_counter()
                try:
                    out = run(lat, tag, n_steps, strength)
                    dt = (time.perf_counter() - t0) * 1000
                    out.save(OUT_DIR / f"{tag}_{in_name}_n{n_steps}_s{int(strength*100):03d}.png")
                    print(f"  {tag} {in_name} n={n_steps} s={strength}: {dt:.0f}ms", flush=True)
                except Exception as e:
                    print(f"  {tag} {in_name} n={n_steps} s={strength}: FAIL {e}", flush=True)

    print(f"[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
