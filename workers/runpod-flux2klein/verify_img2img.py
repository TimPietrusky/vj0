#!/usr/bin/env python3
"""
Methodology verification: prove that img2img is actually conditioning on the input
image, not just generating from prompt.

We feed the SAME waveform-shaped synthetic input through THREE prompts that have
nothing to do with waveforms, audio, or bars. If the generated outputs preserve
the input's spatial structure (12 vertical bars on a radial gradient), then
img2img is real. If outputs ignore the bars completely and just render the
prompt content, then we were fooling ourselves with a leading prompt.

Bonus control: also generate the same prompts with a *blank* input (solid gray)
for direct A/B comparison — if outputs differ between waveform-input and gray-input,
the input is being used.
"""
import math
from pathlib import Path
import torch
from PIL import Image, ImageDraw

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SIZE = 512
SEED = 42

NEUTRAL_PROMPTS = [
    ("forest", "lush green forest in autumn, golden light filtering through leaves, peaceful, photographic"),
    ("portrait", "studio portrait of an elderly fisherman, weathered face, soft window light, oil painting"),
    ("food", "macro photograph of a bowl of ramen with steam rising, garnish on top, dark wooden table"),
]


def make_waveform_input(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), (8, 8, 16))
    px = img.load()
    cx = cy = size / 2
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            d = math.sqrt(dx * dx + dy * dy) / (size / 2)
            r = int(max(0, 255 * (1 - d) * 0.9 + 30))
            g = int(max(0, 255 * (1 - d) * 0.3 + 20))
            b = int(max(0, 255 * (1 - d * 0.6) * 0.8 + 40))
            px[x, y] = (r, g, b)
    draw = ImageDraw.Draw(img)
    bar_w = max(2, size // 24)
    for i in range(12):
        h = int((0.3 + 0.6 * abs(math.sin(i * 0.7))) * size * 0.8)
        x0 = int(size * 0.1 + i * (size * 0.07))
        y0 = (size - h) // 2
        draw.rectangle([x0, y0, x0 + bar_w, y0 + h],
                       fill=(255, 80 + i * 12, 200 - i * 10))
    return img


def make_blank_input(size: int) -> Image.Image:
    return Image.new("RGB", (size, size), (128, 128, 128))


def main():
    out_dir = Path("/workspace/flux2-verify-out"); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] device={torch.cuda.get_device_name(0)}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    print("[load] pipeline", flush=True)
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

    waveform_in = make_waveform_input(SIZE)
    blank_in = make_blank_input(SIZE)
    waveform_in.save(out_dir / "input_waveform.png")
    blank_in.save(out_dir / "input_blank.png")

    for tag, prompt in NEUTRAL_PROMPTS:
        print(f"\n[prompt:{tag}] {prompt!r}", flush=True)

        out_wave = pipe(
            image=waveform_in, prompt=prompt,
            height=SIZE, width=SIZE, num_inference_steps=4,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
        ).images[0]
        out_wave.save(out_dir / f"out_{tag}_wave_input.png")

        out_blank = pipe(
            image=blank_in, prompt=prompt,
            height=SIZE, width=SIZE, num_inference_steps=4,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
        ).images[0]
        out_blank.save(out_dir / f"out_{tag}_blank_input.png")

        print(f"[prompt:{tag}] saved both", flush=True)

    print(f"\n[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
