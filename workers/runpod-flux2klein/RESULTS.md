# FLUX.2-klein-4B img2img — Final Results

Complete journey of every optimization attempt, correctness investigation, and the final winning configuration for live-VJ audio-reactive image generation.

## 🏆 TL;DR — Final winning numbers

**Starting point: 351 ms (text2img). Final: 32 ms @ 256² = 30.8 fps.** That's an **11× speedup** while gaining real img2img behavior and keeping Klein quality.

All numbers below **include the VAE encode of the waveform input** (real per-frame cost for live VJ, not just pipeline time):

| Resolution | 2 steps | 3 steps | 4 steps |
|---|---|---|---|
| 256² | **32.4 ms · 30.8 fps** 🔥 | 44.9 ms · 22.3 fps | ~60 ms · 17 fps |
| 384² | **57.6 ms · 17.4 fps** | 79.6 ms · 12.6 fps | ~105 ms · 9.5 fps |
| 512² | **96.1 ms · 10.4 fps** | 138.3 ms · 7.2 fps | 177.2 ms · 5.6 fps |

p95 within **0.4 ms of mean** across all configs — glassy-smooth frame timing.
VRAM: **16.6 GB / 32 GB** (plenty of headroom).

### The tradeoff ladder for the VJ UI
- **Latency-first** (fast beats, aggressive cuts): 256² / 2-step → 30 fps
- **Balanced** (standard VJ set): 384² / 2-step → 17 fps
- **Quality-first** (slow ambient sets, hero shots): 512² / 2-step → 10 fps, or 3-step → 7 fps

## The winning config

```python
from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents
import torch, numpy as np
from PIL import Image

# --- load & compile once at startup (~15s warmup per resolution) ---
pipe = Flux2KleinKVPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16
)
pipe.vae = AutoencoderKLFlux2.from_pretrained(
    "black-forest-labs/FLUX.2-small-decoder", torch_dtype=torch.bfloat16
)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
torch.backends.cuda.matmul.allow_tf32 = True

# Compile transformer + both VAE halves (encoder runs every frame)
pipe.transformer  = torch.compile(pipe.transformer,  mode="default", fullgraph=False, dynamic=False)
pipe.vae.encoder  = torch.compile(pipe.vae.encoder,  mode="default", fullgraph=False, dynamic=False)
pipe.vae.decoder  = torch.compile(pipe.vae.decoder,  mode="default", fullgraph=False, dynamic=False)

# --- encode prompt once per prompt change (skipped for hot path) ---
prompt_embeds, _ = pipe.encode_prompt(
    prompt="a bright white lightning bolt against a pitch black night sky",
    device="cuda", num_images_per_prompt=1, max_sequence_length=64,
)

# --- per-frame loop ---
ALPHA = 0.10      # 0.03-0.05 = subtle (SDXL-turbo-like); 0.08-0.15 = clear influence
N_STEPS = 2       # 2 for 30fps@256, 3 for quality, 4 for max detail
SIZE = 256

def encode_image(pil_img):
    """Convert waveform PNG → VAE latents (~8 ms @ 256²)."""
    a = np.asarray(pil_img.resize((SIZE, SIZE)), dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
    patch = pipe._patchify_latents(raw)
    m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
    s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
    return (patch - m) / s

def generate(waveform_pil, prompt_embeds, seed=42):
    lat = encode_image(waveform_pil)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device="cuda")
    # Alpha-blend noise floor — matches SD-Turbo's compositional-bias math
    noisy = ALPHA * lat + (1 - ALPHA) * noise
    sigmas = np.linspace(1 - ALPHA, 0.0, N_STEPS).tolist()
    return pipe(
        image=None, prompt=None,
        prompt_embeds=prompt_embeds,
        latents=noisy, sigmas=sigmas,
        height=SIZE, width=SIZE, num_inference_steps=N_STEPS,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]
```

## Environment

- **Hardware**: RTX 5090 (32 GB, Blackwell sm_120, capability `(12, 0)`)
- **Datacenter**: EU-RO-1 (Romania — required for UDP ingress / WebRTC)
- **Pod**: `3m90zbu8fwyyqk`, secure cloud, $0.99/hr
- **Image**: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` (CUDA 12.8 base, the only Blackwell-compatible RunPod PyTorch image)
- **Runtime**: torch `2.11.0+cu130` (auto-upgraded during install), diffusers `0.38.0.dev0` from main, transformers `5.5.4`
- **Models** (public, ungated): `black-forest-labs/FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-small-decoder`

## Methodology correctness discovery (the big gotcha)

For most of the optimization session I assumed the pipeline's `image=` kwarg did img2img. It doesn't usefully. The docstring advertises "KV-cached reference image conditioning" but rigorous A/B testing showed:

| Input | Prompt "lightning bolt against black sky" | Result |
|---|---|---|
| waveform_1 | diagonal bolt top-left→bottom-right | **identical** |
| waveform_2 | diagonal bolt top-left→bottom-right | **identical** |
| solid black | diagonal bolt top-left→bottom-right | **identical** |
| **no image at all** | diagonal bolt top-left→bottom-right | **identical** |

The native `image=` arg's reference influence is imperceptibly weak at 4 steps. The pipeline is effectively text2img.

**Fix**: bypass the native hook entirely. Inject the input image as *starting latents* with classic flow-matching noise mixing. Pass `image=None` so the pipeline doesn't activate its weak reference path.

The math for alpha-blend img2img on a flow-matching model:

```
noisy_latents = α · image_latents + (1 - α) · noise         # small α = small image bias
sigmas        = linspace(1 - α, 0, N_STEPS)                 # truncated schedule
```

This reproduces SD-Turbo's "2% image-leak at t_max" behavior. DDPM's `sqrt(alpha_bar_t_max) ≈ 0.02` is non-zero, which is why SD-Turbo at strength=1 still subtly influences composition. Flow matching at sigma=1 is literally 0% image. With our alpha-blend we control the leak explicitly.

**Verified across 63 generations** on 3 unrelated prompts (dog, beach, city) × 3 inputs (wave1, wave2, blank) × 7 alpha values. At α=0.02-0.08 you get clean prompt rendering with subtle compositional bias from the input. At α=0.10-0.25 you get visible input structure. Above α=0.40 the input dominates.

## Pipeline class choice

Three FLUX.2-related pipelines exist in diffusers main. We use `Flux2KleinKVPipeline`.

| Class | Default steps | Guidance | KV cache | Notes |
|---|---|---|---|---|
| `Flux2Pipeline` | 50 | 4.0 | ❌ | Generic, slow |
| `Flux2KleinPipeline` | 50 | 4.0 | ❌ | Klein-tuned, no cache |
| **`Flux2KleinKVPipeline`** ⭐ | 4 | none (distilled) | ✅ | What we use |

## Optimization journey

### What worked

| # | Stage | Δ @ 512² | Cumulative |
|---|---|---|---|
| — | Baseline (text2img, 4 steps, no opts) | — | 351 ms |
| 1 | Pre-encoded prompt, `max_sequence_length=128` | **−85 ms** | 266 ms |
| 2 | `torch.compile(transformer, mode="default")` | **−27 ms** | 239 ms |
| 3 | `torch.compile(vae.decoder, mode="default")` | **−7 ms** | 232 ms |
| 4 | Shape-preserving sigma-injection img2img (s=0.9) | **−63 ms** | 169 ms |
| 5 | Alpha-blend img2img (α=0.10, real SDXL-turbo-style) | **+8 ms** | 177 ms (includes VAE encode) |
| 6 | `torch.compile(vae.encoder)` — matters because encoder is in hot path | −0 to −8 ms | 168 ms |
| 7 | `max_sequence_length=128 → 64` | **−9 ms** | 159 ms |
| 8 | **4 steps → 3 steps** | **−30 ms** | 129 ms |
| 9 | **3 steps → 2 steps** | **−33 ms** | **96 ms @ 512², 32 ms @ 256²** |

At each step, p95 stays within 1-2 ms of mean — compilation locks kernel choice and alpha-blend schedule is deterministic.

### What didn't work

| Attempt | Result | Reason |
|---|---|---|
| `channels_last` memory format | +6 ms (regression) | Flux is linear-heavy, not conv. Permute overhead > benefit |
| `torch.compile(mode="reduce-overhead")` | Crash | CUDA graphs can't capture PEFT wrapper's view tensors |
| `torch.compile(mode="max-autotune-no-cudagraphs")` | Compiles but spams thousands of OOM warnings | Triton max-autotune templates assume Hopper smem (228 KB); Blackwell = 101 KB/SM; most configs infeasible → ATen fallback → no gain |
| `torchao.Float8WeightOnlyConfig` (fp8 weight storage) | 257 ms (slower than compiled bf16) | Dequant-to-bf16 cast every matmul costs more than the 4 GB VRAM saved |
| `torchao.Float8DynamicActivationFloat8WeightConfig` (true fp8 matmul) | `InternalTorchDynamoError: Polyfill cmp_eq` | Known torch 2.11 / torchao 0.17 polyfill bug |
| `pipe.fuse_qkv_projections()` | `AttributeError` | Method is on `pipe.transformer`, not `pipe`. When corrected: **−0.7 ms** (noise-level) |
| `SageAttention` (kwarg wrapper) | `AssertionError` during compile | Sage custom op doesn't cleanly compose with `torch.compile` on this torch version. Patching before compile still asserts — needs deeper integration |
| `FluxControlNetImg2ImgPipeline` on schnell (parallel subagent test) | Works structurally but **9937 ms/frame** | Schnell (12B) + CN + T5 + CLIP > 32 GB → forced `enable_model_cpu_offload` → PCIe swap dominates. ~103× slower than our Klein winning config |
| `bitsandbytes` 0.49 | Fails to load `libnvJitLink.so.13` | Wants CUDA 13 runtime libs not on this image. Uninstalled |
| Negative prompt embeds | `TypeError: unexpected kwarg` | Klein is distilled (no CFG) — pipeline literally doesn't accept them |

### Not attempted / not worth pursuing

- **TensorRT**: another potential 1.5-2× but days of ONNX-export fragility; skipped for launch
- **SDNQ 4-bit model** (`Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic`): needs custom loader framework outside standard diffusers; expected to behave like fp8 weight-only (slower due to dequant), so low ROI
- **FlashAttention 3**: Hopper-only (uses TMA + async warp specialization); falls back to FA2 on Blackwell with no speedup over SDPA
- **Manual CUDA graphs bypassing PEFT**: would likely squeeze another 10-15 ms but fiddly to maintain; only worth it if we need to push past 30 fps at 256²

## Correctness matrix

Tested **extensively** to avoid false positives:

- **Alpha=0 control**: produces pure text2img (confirms our math is right)
- **Blank input control**: at any alpha, blank input should behave identical to α=0 for that alpha+prompt combo — confirmed
- **Wave1 vs Wave2**: visibly different outputs at every α > 0 for shape-compatible prompts — img2img is genuinely conditioning
- **Shape-compatible prompts** (lightning, fire, river, ink, neon, crack, flame): ✅ prompt content clearly follows waveform's horizontal contour at α=0.10
- **Shape-incompatible prompts** (dog, beach, castle, city): only **subtle** compositional bias at α ≤ 0.08; at higher α the waveform dominates. This is a fundamental property of distilled 2-4 step models, same limitation observed by parallel ControlNet subagent. For VJ use cases we craft prompts from the shape-compatible vocabulary

## Quality check at the winning config

All sampled at `waveform_1.png` + "lightning bolt" prompt + α=0.10 + seed=42, see `max-v3/*.png`:

- `4_res_256_2step.png` (32.4 ms): small but sharp diagonal lightning following waveform
- `4_res_384_2step.png` (57.6 ms): brighter, defined core + branches, smooth edges
- `4_res_512_2step.png` (96.1 ms): crisp detail, noticeably cleaner glow
- `5_res_256_3step.png` (44.9 ms) vs 2-step @ 256: basically visually indistinguishable
- `6_res_384_3step.png` (79.6 ms) vs 2-step @ 384: similar, slight edge detail gain

2-step wins the quality/cost tradeoff everywhere. Klein was distilled for 4 but with alpha-blend img2img the 2-step output is still coherent because the starting latent already carries structural information.

## Operational lessons

- **Don't run `runpodctl doctor` after creating a pod** — it rotates your account-level SSH key, leaves pod's baked-in `PUBLIC_KEY` env stale, locks you out. Run `doctor` *before* creation or recreate pod after rotation
- **EU-RO-1 is the only RunPod DC with UDP ingress** — mandatory for WebRTC ports 10000-10002
- **`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`** is the Blackwell-compatible base; older `cu12.2` won't boot a 5090
- **PEP 668 lockdown** — use `pip install --break-system-packages` on this image
- **First `torch.compile` call per resolution** takes ~15-30 s warmup. After that, per-frame latency is dead-flat
- **Diffusers from `main` auto-upgrades torch to 2.11+cu130** — wheel bundles its own CUDA 13 runtime; works fine under driver 580
- **Sigma schedule must end at 0** — `np.linspace(sigma_start, 0, N_STEPS)` not `linspace(sigma_start, 0, N+1)[:N]` (bug I hit)
- **Pre-encoded prompt embeds persist across calls** — cache them per-prompt. Text encoder is 85 ms saved per frame
- **VAE encoder compilation only helps if encoder is in the hot path** — with live waveforms changing per frame, it is. Bench it inside the timed loop

## Files

All under `workers/runpod-flux2klein/`:

| File | Purpose |
|---|---|
| `bench_max_speed_v3.py` | Final winning bench. Reproduce with: `python bench_max_speed_v3.py` |
| `bench_final.py` | Shape-compatible prompt matrix (6 prompts × 2 waveforms × 3 strengths) |
| `test_noise_floor.py` | Alpha-blend correctness verification (7 alphas × 3 inputs × 3 prompts) |
| `test_unrelated_prompts.py` | Unrelated-prompt correctness (dog, beach, city) — shows limits of 4-step |
| `verify_real_waveform.py` | Native `image=` arg A/B that proved it doesn't work |
| `test_sdturbo.py` | SD-Turbo comparison (17 ms at 256² but low quality) |
| `verify_img2img.py` | First A/B test — exposed the methodology hole |
| `sweep_resolutions.py` | Resolution sweep without img2img (text2img baseline) |
| `optimize_v3.py` / `optimize.py` / `smoke.py` | Earlier optimization iterations |

Output galleries:

- `max-v3/` — final winning config images at every resolution × step combo
- `final-bench/` — 36-run matrix (2 waveforms × 6 prompts × 3 strengths @ 512²) for shape-compatible prompts
- `noise-floor/` — 63-generation alpha sweep showing the SDXL-turbo-style behavior
- `classic-v2/` — first sigma-injection img2img results (pre-alpha-blend discovery)
- `verify-real/` — proof that native `image=` arg is ineffective
- `sdturbo-test/` — SD-Turbo reference on pod #2 (17 ms/256² but visibly lower quality)
- `controlnet-test/` — ControlNet parallel investigation (works but 10s/frame)

## What's next (for integration)

1. **Port to `inference_server.py`** — adapt the stablefast worker's stdin/stdout JSON protocol for the Flux2 klein config. Drop-in to the existing `server.js` WebRTC bridge. Port `22/tcp, 3000/http, 10000-10002/udp` already allocated on this pod
2. **Attach a network volume** before pod destruction — the ~8 GB model cache is otherwise ephemeral
3. **Prompt library** for the VJ UI — curated shape-compatible prompts (fire, lightning, river, ink, neon, smoke, aurora, tendrils, laser, electricity, crack, flame, trails) with per-prompt suggested alpha
4. **Resolution/step presets in UI** — three clicks: Fast (256/2 = 30fps) · Balanced (384/2 = 17fps) · Quality (512/3 = 7fps)
5. **Compare against z-image-turbo** when the parallel test lands — direct A/B at the same prompts + waveforms
