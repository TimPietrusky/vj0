#!/usr/bin/env python3
"""
Proper img2img methodology test with REAL b/w waveform inputs.

Flux2KleinKVPipeline uses Kontext-style reference conditioning (reference tokens
participate in attention of the first denoise step, K/V cached after). To prove
the reference is actually influencing the output, we use prompts where shape
fidelity would be unambiguous — and compare wave input vs blank input, and
the two different waveforms against each other.

Test matrix:
  inputs:  waveform_1 (real), waveform_2 (real), blank (white), blank (black)
  prompts: directive shape-following prompts where reference matters
"""
from pathlib import Path
import torch
from PIL import Image

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SIZE = 512          # upscale the 256 waveforms to 512 for Klein
SEED = 42
WAVEFORM_DIR = Path("/workspace/waveforms")
OUT_DIR = Path("/workspace/flux2-verify-real")

# Shape-following prompts — if reference conditioning works, the output contour
# will mirror the input silhouette.
PROMPTS = [
    ("lightning", "a bright white lightning bolt against a pitch black night sky, dramatic, photographic, high contrast"),
    ("neon_line", "a single thin glowing neon line on pure black background, minimalist, studio light"),
    ("river",    "aerial photograph of a winding silver river through a dark forest at night, high contrast"),
    ("crack",    "a bright white crack running across black ice, macro photo, dramatic contrast"),
]


def load_input(p: Path, size: int) -> Image.Image:
    img = Image.open(p).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2

    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

    wave1 = load_input(WAVEFORM_DIR / "waveform_1.png", SIZE)
    wave2 = load_input(WAVEFORM_DIR / "waveform_2.png", SIZE)
    black = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    white = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))

    wave1.save(OUT_DIR / "input_wave1.png")
    wave2.save(OUT_DIR / "input_wave2.png")
    black.save(OUT_DIR / "input_black.png")
    white.save(OUT_DIR / "input_white.png")

    inputs = [("wave1", wave1), ("wave2", wave2), ("black", black), ("white", white)]

    for tag, prompt in PROMPTS:
        print(f"\n[{tag}] prompt={prompt!r}", flush=True)
        for name, img in inputs:
            out = pipe(
                image=img, prompt=prompt,
                height=SIZE, width=SIZE, num_inference_steps=4,
                generator=torch.Generator(device="cuda").manual_seed(SEED),
            ).images[0]
            out.save(OUT_DIR / f"out_{tag}_{name}.png")
            print(f"  [{tag}] input={name} -> saved", flush=True)

    # Also run the SAME prompt with NO image arg to check pure text2img baseline
    print("\n[t2i] running pure text2img (no image) for each prompt as control", flush=True)
    for tag, prompt in PROMPTS:
        out = pipe(
            image=None, prompt=prompt,
            height=SIZE, width=SIZE, num_inference_steps=4,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
        ).images[0]
        out.save(OUT_DIR / f"out_{tag}_noimage.png")
        print(f"  [{tag}] no_image -> saved", flush=True)

    print(f"\n[done] {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
