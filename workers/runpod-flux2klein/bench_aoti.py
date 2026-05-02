#!/usr/bin/env python3
"""Phase 9D — AOT-Inductor: ahead-of-time compile the transformer.

Question: can `torch._inductor.aoti_compile_and_package(...)` produce a
`.pt2` artifact that loads in <1s and runs as fast (or faster) than the
JIT-compiled `torch.compile(mode="reduce-overhead")` we ship today?

Two motivations for AOT:
1. UX win — pod cold-start drops from ~3 min (load + JIT) to ~30s (load
   weights + load .pt2). Big quality-of-life upgrade for live demos.
2. Possible perf win — AOT skips Python in the hot path entirely. With
   reduce-overhead we already cudagraph-capture, but Python-side overhead
   between graphs (sigma update, scheduler step) still costs.

Approach:
1. Boot Klein, apply fp8 (production config).
2. Hook `transformer.forward` to capture the EXACT args (shapes, dtypes,
   None vs tensor) at one real inference call. This avoids guessing rope
   arg shapes etc.
3. Build a tiny `ExportableTransformer` wrapper that takes flat positional
   tensor args (torch.export doesn't love None/Optional/dict).
4. `torch.export.export(wrapper, captured_args)` → ExportedProgram.
5. `torch._inductor.aoti_compile_and_package(ep)` → /tmp/aoti_klein.pt2.
6. `torch._inductor.aoti_load_package(...)` → callable AOTIModule.
7. Bench: WARMUP_ITERS + TIMED_ITERS calls, single-GPU, fixed shape.
8. Compare vs same shape with `torch.compile(reduce-overhead)`.

Single-GPU 256²/4-step (we're benching the kernel, not the round-robin
dispatcher). KV cache disabled (None) for the bench — Klein's KV cache
between denoising steps would require shape-changing inputs, which AOT
can't handle without bake-per-shape. That's a real production caveat
documented in the report.

Output: print summary + write /workspace/bench-2026-04-30/aoti/summary.json
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
WAVE_PATH = "/workspace/waveforms/waveform_1.png"
OUT_DIR = Path("/workspace/bench-2026-04-30/aoti")
SEED = 42
ALPHA = 0.10
MAX_SEQ_LEN = 64
PROMPT = "a bright white lightning bolt against a pitch black night sky"
N_STEPS = 4
SIZE = 256
WARMUP = 5
TIMED = 30


def percentile(xs, p):
    xs = sorted(xs)
    import math
    k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from PIL import Image
    from diffusers import Flux2KleinKVPipeline, AutoencoderKLFlux2
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents
    from torchao.quantization import (
        quantize_,
        Float8DynamicActivationFloat8WeightConfig,
    )
    try:
        from torchao.quantization.granularity import PerTensor
        cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
    except Exception:
        cfg = Float8DynamicActivationFloat8WeightConfig()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda:0"

    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda} "
          f"cap={torch.cuda.get_device_capability(0)}", flush=True)

    print(f"[load] {KLEIN_REPO} ...", flush=True)
    t0 = time.perf_counter()
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s, "
          f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    def fp8_filter(m, fqn):
        if not isinstance(m, torch.nn.Linear): return False
        if "transformer" not in fqn and "single_transformer_blocks" not in fqn:
            return False
        bad = ("pe_embedder", "norm_", "_norm", "embed", "out_proj")
        return not any(b in fqn.lower() for b in bad)

    print(f"[fp8] applying PerTensor fp8 to transformer...", flush=True)
    quantize_(pipe.transformer, cfg, filter_fn=fp8_filter)

    # ---- Capture real call args via a one-shot hook. ----
    captured = {}

    def hook(module, args, kwargs):
        # Grab the first call only.
        if "args" not in captured:
            captured["args"] = tuple(args)
            captured["kwargs"] = dict(kwargs)

    h = pipe.transformer.register_forward_pre_hook(hook, with_kwargs=True)

    # Drive one real img2img inference to populate captured args.
    print(f"[capture] running one inference at {SIZE}² to capture transformer args...", flush=True)
    wave_pil = Image.open(WAVE_PATH).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    a = np.asarray(wave_pil, dtype=np.float32) / 127.5 - 1.0
    img_t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
    raw = retrieve_latents(pipe.vae.encode(img_t), sample_mode="argmax")
    patch = pipe._patchify_latents(raw)
    m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
    s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
    lat = (patch - m) / s
    gen = torch.Generator(device=device).manual_seed(SEED)
    noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device=device)
    noisy = ALPHA * lat + (1 - ALPHA) * noise
    sigmas = np.linspace(1 - ALPHA, 0.0, N_STEPS).tolist()
    r = pipe.encode_prompt(prompt=PROMPT, device=device, num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN)
    prompt_embeds = r[0] if isinstance(r, tuple) else r
    pipe(image=None, prompt=None, prompt_embeds=prompt_embeds,
         latents=noisy, sigmas=sigmas, height=SIZE, width=SIZE,
         num_inference_steps=N_STEPS,
         generator=torch.Generator(device=device).manual_seed(SEED)).images[0]
    h.remove()

    if "args" not in captured:
        raise RuntimeError("hook never fired — did the pipeline call transformer at all?")

    pos_args = captured["args"]
    kw_args = captured["kwargs"]
    print(f"[capture] got {len(pos_args)} positional args, {len(kw_args)} kwargs", flush=True)

    # Catalog the captured args for the writeup.
    arg_summary = []
    for i, a in enumerate(pos_args):
        if isinstance(a, torch.Tensor):
            arg_summary.append(f"pos[{i}]=Tensor{tuple(a.shape)} {a.dtype}")
        else:
            arg_summary.append(f"pos[{i}]={type(a).__name__}({a!r})"[:80])
    for k, a in kw_args.items():
        if isinstance(a, torch.Tensor):
            arg_summary.append(f"{k}=Tensor{tuple(a.shape)} {a.dtype}")
        elif a is None:
            arg_summary.append(f"{k}=None")
        else:
            arg_summary.append(f"{k}={type(a).__name__}({a!r})"[:80])
    print("\n[capture] real call shapes:", flush=True)
    for s in arg_summary:
        print(f"  {s}", flush=True)

    # ---- Build a wrapper that flattens to all-tensor args for export. ----
    # Filter kwargs to (name, value) pairs that are tensors. Bake non-tensor
    # values (None, scalars, dicts) as constants in the wrapper.
    tensor_kwargs = {k: v for k, v in kw_args.items() if isinstance(v, torch.Tensor)}
    const_kwargs = {k: v for k, v in kw_args.items() if not isinstance(v, torch.Tensor)}
    print(f"[wrap] tensor kwargs: {list(tensor_kwargs)}", flush=True)
    print(f"[wrap] constant kwargs: {list(const_kwargs)}", flush=True)

    transformer = pipe.transformer

    class ExportableTransformer(torch.nn.Module):
        def __init__(self, inner, const_kwargs, tensor_kwarg_names, n_pos):
            super().__init__()
            self.inner = inner
            self.const_kwargs = const_kwargs
            self.tensor_kwarg_names = tensor_kwarg_names
            self.n_pos = n_pos

        def forward(self, *all_args):
            pos = all_args[:self.n_pos]
            tk_vals = all_args[self.n_pos:]
            kw = dict(zip(self.tensor_kwarg_names, tk_vals))
            kw.update(self.const_kwargs)
            kw["return_dict"] = False
            out = self.inner(*pos, **kw)
            # Forward returns a tuple; we want the first element (sample).
            if isinstance(out, tuple):
                return out[0]
            return out

    wrapper = ExportableTransformer(
        transformer, const_kwargs, list(tensor_kwargs.keys()), len(pos_args)
    ).to(device).eval()

    flat_args = tuple(pos_args) + tuple(tensor_kwargs.values())

    # Sanity: wrapper produces same shape as the real transformer.
    print(f"[sanity] wrapper forward...", flush=True)
    t = time.perf_counter()
    sanity_out = wrapper(*flat_args)
    torch.cuda.synchronize()
    print(f"[sanity] wrapper out shape: {tuple(sanity_out.shape)} in "
          f"{(time.perf_counter()-t)*1000:.1f} ms", flush=True)

    # ---- AOT-Inductor: export + compile + load + bench. ----
    aoti_ok = False
    aoti_err = None
    aoti_results = None
    try:
        print(f"[aoti] torch.export.export ...", flush=True)
        t_exp = time.perf_counter()
        ep = torch.export.export(wrapper, flat_args, strict=False)
        print(f"[aoti] export done in {time.perf_counter()-t_exp:.1f}s; "
              f"graph nodes={sum(1 for _ in ep.graph.nodes)}", flush=True)

        print(f"[aoti] aoti_compile_and_package ...", flush=True)
        t_aoti = time.perf_counter()
        artifact_path = str(OUT_DIR / "klein_transformer_256.pt2")
        torch._inductor.aoti_compile_and_package(ep, package_path=artifact_path)
        print(f"[aoti] compile+package done in {time.perf_counter()-t_aoti:.1f}s; "
              f"artifact at {artifact_path}", flush=True)

        print(f"[aoti] aoti_load_package ...", flush=True)
        t_load = time.perf_counter()
        loaded = torch._inductor.aoti_load_package(artifact_path)
        print(f"[aoti] load done in {time.perf_counter()-t_load:.1f}s", flush=True)

        # Warmup + timing.
        print(f"[aoti] warmup x{WARMUP} ...", flush=True)
        for _ in range(WARMUP):
            _ = loaded(*flat_args)
            torch.cuda.synchronize()
        print(f"[aoti] timing x{TIMED} ...", flush=True)
        per = []
        for _ in range(TIMED):
            tt = time.perf_counter()
            _ = loaded(*flat_args)
            torch.cuda.synchronize()
            per.append((time.perf_counter() - tt) * 1000)
        aoti_results = {
            "n": TIMED,
            "mean_ms": round(sum(per) / len(per), 3),
            "p95_ms": round(percentile(per, 95), 3),
            "p99_ms": round(percentile(per, 99), 3),
            "min_ms": round(min(per), 3),
        }
        print(f"[aoti] {aoti_results}", flush=True)
        aoti_ok = True
    except Exception as e:
        aoti_err = f"{type(e).__name__}: {e}"
        print(f"[aoti] FAILED: {aoti_err}", flush=True)
        traceback.print_exc()

    # ---- Baseline: torch.compile(reduce-overhead) on the same wrapper. ----
    print(f"\n[jit] torch.compile(mode=reduce-overhead) on wrapper ...", flush=True)
    jit_results = None
    jit_err = None
    try:
        compiled = torch.compile(wrapper, mode="reduce-overhead", fullgraph=False, dynamic=False)
        print(f"[jit] warmup x{WARMUP} ...", flush=True)
        t_w = time.perf_counter()
        for _ in range(WARMUP):
            _ = compiled(*flat_args)
            torch.cuda.synchronize()
        print(f"[jit] warmup done in {time.perf_counter()-t_w:.1f}s", flush=True)
        per = []
        for _ in range(TIMED):
            tt = time.perf_counter()
            _ = compiled(*flat_args)
            torch.cuda.synchronize()
            per.append((time.perf_counter() - tt) * 1000)
        jit_results = {
            "n": TIMED,
            "mean_ms": round(sum(per) / len(per), 3),
            "p95_ms": round(percentile(per, 95), 3),
            "p99_ms": round(percentile(per, 99), 3),
            "min_ms": round(min(per), 3),
        }
        print(f"[jit] {jit_results}", flush=True)
    except Exception as e:
        jit_err = f"{type(e).__name__}: {e}"
        print(f"[jit] FAILED: {jit_err}", flush=True)
        traceback.print_exc()

    summary = {
        "phase": "9D_aoti",
        "size_px": SIZE,
        "n_steps": N_STEPS,
        "captured_args": arg_summary,
        "aoti": {"ok": aoti_ok, "results": aoti_results, "error": aoti_err},
        "jit_reduce_overhead": {"results": jit_results, "error": jit_err},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*72}\nAOT-INDUCTOR vs JIT(reduce-overhead) — single-call, {SIZE}²\n{'='*72}", flush=True)
    if aoti_ok and jit_results:
        delta = (jit_results["mean_ms"] - aoti_results["mean_ms"]) / jit_results["mean_ms"] * 100
        print(f"  AOTI: {aoti_results['mean_ms']:.2f} ms (p95 {aoti_results['p95_ms']:.2f})", flush=True)
        print(f"  JIT:  {jit_results['mean_ms']:.2f} ms (p95 {jit_results['p95_ms']:.2f})", flush=True)
        print(f"  Δ:    {delta:+.1f}% (AOTI {'faster' if delta > 0 else 'slower'} than JIT)", flush=True)
    elif aoti_ok:
        print(f"  AOTI: {aoti_results['mean_ms']:.2f} ms (JIT failed: {jit_err})", flush=True)
    elif jit_results:
        print(f"  AOTI failed: {aoti_err}", flush=True)
        print(f"  JIT: {jit_results['mean_ms']:.2f} ms", flush=True)
    else:
        print(f"  Both failed. AOTI={aoti_err}; JIT={jit_err}", flush=True)
    print(f"\n[done] {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
