#!/usr/bin/env python3
"""Phase 7b — Dual-GPU round-robin via separate processes (GIL-free).

The threaded version (`bench_dual_gpu.py`) hit 1.61x at 2 GPUs because both
threads dispatch CUDA kernels through the same Python interpreter — the GIL
serializes the Python-side pipeline glue (sigma scheduling, scheduler steps,
post-processing) between the two GPUs.

Fix: spawn one *process* per GPU. Each worker has its own interpreter and
its own GPU pinned via `CUDA_VISIBLE_DEVICES`. They run fully in parallel.
A dispatcher process feeds them seeds round-robin via mp.Pipe (lower latency
than mp.Queue for small messages) and collects per-frame timings.

Bench: 256² and 512² × 4-step. Each GPU does N/2 frames. We measure wall-clock
from when both workers report "ready" to when both finish N/2 frames.

Output: /workspace/bench-2026-04-30/dual_gpu_mp/summary.json
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
WAVE_PATH = "/workspace/waveforms/waveform_1.png"
OUT_DIR = Path("/workspace/bench-2026-04-30/dual_gpu_mp")
SEED = 42
ALPHA = 0.10
MAX_SEQ_LEN = 64
PROMPT = (
    "a bright white lightning bolt against a pitch black night sky, "
    "dramatic, photographic, high contrast"
)
WARMUP_ITERS = 3
TIMED_ITERS_PER_GPU = 20
N_STEPS = 4
SIZES = [256, 512]


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def worker_main(gpu_idx: int, ctrl_in, data_out):
    """One process pinned to a single GPU. Loads pipeline, then services requests.

    Protocol on `ctrl_in` (parent → worker):
      ("config", size: int)            — load + warmup at this resolution
      ("run", seed: int)               — run one frame with this seed
      ("shutdown",)                    — exit

    Protocol on `data_out` (worker → parent):
      ("ready", gpu_idx, info_dict)    — after config/warmup
      ("done", gpu_idx, t_ms, vram_gb) — after each run
      ("error", gpu_idx, msg)
    """
    # Pin this process to one GPU. Must happen BEFORE `import torch`.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")

    try:
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
            from torchao.quantization.granularity import PerRow
        except ImportError:
            PerRow = None

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # In this process, the only visible device is at index 0.
        device = "cuda:0"

        # Optional: monkey-patch torch SDPA → SageAttention.
        #   BENCH_USE_SAGE=1 (default off) — turn the patch on.
        #   BENCH_SAGE_VERSION=1 — pip `sageattention` (v1, int8 Q/K, bf16 PV).
        #   BENCH_SAGE_VERSION=3 — `sageattn3_blackwell` from source, FP4
        #     microscaling attention specifically tuned for sm_120. README
        #     claims 2.7x vs FA2 on RTX5090.
        #
        # Sage2 documented torch.compile compat for non-cudagraphs mode only;
        # Sage3 makes no claim. So we A/B against both compile_mode=default
        # AND compile_mode=reduce-overhead; the latter may regress more.
        sage_v = os.environ.get("BENCH_SAGE_VERSION", "1")
        if os.environ.get("BENCH_USE_SAGE", "0") == "1":
            _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
            try:
                if sage_v == "3":
                    from sageattn3 import sageattn3_blackwell as _sage_kernel

                    def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                                   is_causal=False, scale=None, enable_gqa=False):
                        # Sage3 API: sageattn3_blackwell(q, k, v, is_causal=False).
                        # No scale, no mask, no dropout, no GQA — fall back for those.
                        if attn_mask is not None or dropout_p != 0.0 or enable_gqa or scale is not None:
                            return _orig_sdpa(
                                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
                            )
                        try:
                            return _sage_kernel(query, key, value, is_causal=is_causal)
                        except Exception:
                            return _orig_sdpa(query, key, value, is_causal=is_causal)
                else:
                    import sageattention as _sa

                    def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                                   is_causal=False, scale=None, enable_gqa=False):
                        if attn_mask is not None or dropout_p != 0.0 or enable_gqa:
                            return _orig_sdpa(
                                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
                            )
                        try:
                            return _sa.sageattn(query, key, value, is_causal=is_causal, sm_scale=scale)
                        except Exception:
                            return _orig_sdpa(query, key, value, is_causal=is_causal, scale=scale)

                torch.nn.functional.scaled_dot_product_attention = _sage_sdpa
                print(f"[gpu{gpu_idx}] sage-attn v{sage_v} monkey-patch installed", flush=True)
            except Exception as e:
                print(f"[gpu{gpu_idx}] sage-attn v{sage_v} import FAILED: {e!r}", flush=True)

        pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
        pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        def fp8_filter(m, fqn):
            if not isinstance(m, torch.nn.Linear):
                return False
            if "transformer" not in fqn and "single_transformer_blocks" not in fqn:
                return False
            bad = ("pe_embedder", "norm_", "_norm", "embed", "out_proj")
            return not any(b in fqn.lower() for b in bad)

        # Production winner from torchao_sweep: PerTensor (vs PerRow) wins on
        # Blackwell sm_120 — same VRAM, 1ms faster, half the quality drift.
        try:
            from torchao.quantization.granularity import PerTensor
            cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
        except Exception:
            cfg = Float8DynamicActivationFloat8WeightConfig()
        quantize_(pipe.transformer, cfg, filter_fn=fp8_filter)

        # Optional VAE fp8 (saves ~1ms; sweep cell 7).
        if os.environ.get("BENCH_VAE_FP8", "1") != "0":
            def fp8_filter_vae(module, fqn):
                if not isinstance(module, torch.nn.Linear): return False
                if "transformer" in fqn: return False
                bad = ("post_quant_conv", "bn", "norm_", "_norm")
                return not any(b in fqn.lower() for b in bad)
            quantize_(pipe.vae, cfg, filter_fn=fp8_filter_vae)

        compile_mode = os.environ.get("BENCH_COMPILE_MODE", "default")
        pipe.transformer = torch.compile(pipe.transformer, mode=compile_mode, fullgraph=False, dynamic=False)
        pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode=compile_mode, fullgraph=False, dynamic=False)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode=compile_mode, fullgraph=False, dynamic=False)

        r = pipe.encode_prompt(prompt=PROMPT, device=device, num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN)
        prompt_embeds = r[0] if isinstance(r, tuple) else r

        wave_pil_full = Image.open(WAVE_PATH).convert("RGB")

        # Per-resolution state + runner
        runner = None
        current_size = None

        def make_runner(size):
            wave_pil = wave_pil_full.resize((size, size), Image.LANCZOS)

            def encode_img():
                a = np.asarray(wave_pil, dtype=np.float32) / 127.5 - 1.0
                t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
                raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
                patch = pipe._patchify_latents(raw)
                m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
                s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
                return (patch - m) / s

            def run_one(seed):
                lat = encode_img()
                gen = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device=device)
                noisy = ALPHA * lat + (1 - ALPHA) * noise
                sigmas = np.linspace(1 - ALPHA, 0.0, N_STEPS).tolist()
                pipe(
                    image=None, prompt=None, prompt_embeds=prompt_embeds,
                    latents=noisy, sigmas=sigmas,
                    height=size, width=size, num_inference_steps=N_STEPS,
                    generator=torch.Generator(device=device).manual_seed(seed),
                ).images[0]

            return run_one

        # Service loop
        while True:
            cmd = ctrl_in.recv()
            op = cmd[0]
            if op == "shutdown":
                break
            elif op == "config":
                size = cmd[1]
                if size != current_size:
                    runner = make_runner(size)
                    current_size = size
                    # Warmup at this size — first iter triggers JIT compile.
                    for w in range(WARMUP_ITERS):
                        runner(SEED)
                        torch.cuda.synchronize(device)
                vram_gb = torch.cuda.memory_allocated() / 1e9
                data_out.send(("ready", gpu_idx, {"size": size, "vram_gb": round(vram_gb, 2)}))
            elif op == "run":
                seed = cmd[1]
                t = time.perf_counter()
                runner(seed)
                torch.cuda.synchronize(device)
                t_ms = (time.perf_counter() - t) * 1000
                vram_gb = torch.cuda.memory_allocated() / 1e9
                data_out.send(("done", gpu_idx, round(t_ms, 3), round(vram_gb, 2)))
            else:
                data_out.send(("error", gpu_idx, f"unknown op: {op}"))
    except Exception as e:
        traceback.print_exc()
        try:
            data_out.send(("error", gpu_idx, f"{type(e).__name__}: {e}"))
        except Exception:
            pass


def bench_size(workers, ctrls, datas, size):
    print(f"\n--- size {size}² / {N_STEPS}-step ---", flush=True)

    # Tell both workers to (re)config at this size. Compile happens here.
    print("  configuring + warming up both workers...", flush=True)
    t0 = time.perf_counter()
    for c in ctrls:
        c.send(("config", size))
    info = [None, None]
    for _ in range(2):
        msg = None
        # Receive from whichever worker is ready first.
        ready_d = mp.connection.wait([d for d in datas])
        for d in ready_d:
            m = d.recv()
            if m[0] == "ready":
                idx = m[1]
                info[idx] = m[2]
            elif m[0] == "error":
                raise RuntimeError(f"worker {m[1]}: {m[2]}")
        if all(x is not None for x in info):
            break
    print(f"  both ready in {time.perf_counter()-t0:.1f}s, vram={[i['vram_gb'] for i in info]}", flush=True)

    # Per-GPU solo timing — feed N frames to ONE worker only.
    solo_results = {}
    for who in (0, 1):
        seeds = [SEED + 1000 * who + i for i in range(TIMED_ITERS_PER_GPU)]
        t = time.perf_counter()
        for s in seeds:
            ctrls[who].send(("run", s))
        per_frame = []
        for _ in seeds:
            m = datas[who].recv()
            assert m[0] == "done", m
            per_frame.append(m[2])
        wall = time.perf_counter() - t
        solo_results[who] = {
            "wall_s": round(wall, 3),
            "frames": len(per_frame),
            "fps": round(len(per_frame) / wall, 2),
            "mean_ms": round(sum(per_frame) / len(per_frame), 2),
            "p95_ms": round(percentile(per_frame, 95), 2),
            "p99_ms": round(percentile(per_frame, 99), 2),
            "min_ms": round(min(per_frame), 2),
            "vram_gb": info[who]["vram_gb"],
        }
        print(f"  solo cuda:{who}: {solo_results[who]}", flush=True)

    # Parallel timing — feed both workers concurrently, alternating.
    n_each = TIMED_ITERS_PER_GPU
    t = time.perf_counter()
    # Send all in advance — workers' Pipes have OS buffer space, this is fine
    # at our payload size (small ints). Workers process serially inside.
    for i in range(n_each):
        ctrls[0].send(("run", SEED + 5000 + 2 * i))
        ctrls[1].send(("run", SEED + 5000 + 2 * i + 1))
    # Drain — receive 2N total via wait()
    received = [0, 0]
    per_frame = {0: [], 1: []}
    while sum(received) < 2 * n_each:
        ready = mp.connection.wait([datas[0], datas[1]])
        for d_idx, d in enumerate(datas):
            if d in ready:
                m = d.recv()
                assert m[0] == "done", m
                idx = m[1]
                per_frame[idx].append(m[2])
                received[idx] += 1
    wall = time.perf_counter() - t

    parallel_result = {
        "wall_s": round(wall, 3),
        "total_frames": 2 * n_each,
        "throughput_fps": round((2 * n_each) / wall, 2),
        "per_gpu_a_mean_ms": round(sum(per_frame[0]) / len(per_frame[0]), 2),
        "per_gpu_b_mean_ms": round(sum(per_frame[1]) / len(per_frame[1]), 2),
        "per_gpu_a_p95_ms": round(percentile(per_frame[0], 95), 2),
        "per_gpu_b_p95_ms": round(percentile(per_frame[1], 95), 2),
        "scaling_vs_solo_a": round(parallel_throughput := (2 * n_each) / wall / solo_results[0]["fps"], 3),
    }
    print(f"  parallel: {parallel_result}", flush=True)

    return {"size_px": size, "n_steps": N_STEPS, "solo": solo_results, "parallel": parallel_result}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] launching dual-GPU bench (multiprocess)", flush=True)
    print(f"[init] sizes={SIZES} n_steps={N_STEPS} timed_iters_per_gpu={TIMED_ITERS_PER_GPU}", flush=True)

    ctx = mp.get_context("spawn")  # CUDA requires spawn, not fork
    parents = [ctx.Pipe(duplex=False) for _ in range(2)]  # ctrl: parent→worker
    workers_pipes = [ctx.Pipe(duplex=False) for _ in range(2)]  # data: worker→parent
    ctrls = [p[1] for p in parents]
    ctrls_recv = [p[0] for p in parents]
    workers_send = [p[1] for p in workers_pipes]
    datas = [p[0] for p in workers_pipes]

    workers = [
        ctx.Process(target=worker_main, args=(0, ctrls_recv[0], workers_send[0])),
        ctx.Process(target=worker_main, args=(1, ctrls_recv[1], workers_send[1])),
    ]
    for w in workers:
        w.start()

    results = []
    try:
        for size in SIZES:
            r = bench_size(workers, ctrls, datas, size)
            results.append(r)
    finally:
        for c in ctrls:
            try: c.send(("shutdown",))
            except Exception: pass
        for w in workers:
            w.join(timeout=30)
            if w.is_alive():
                w.terminate()

    summary = {
        "date": "2026-05-01",
        "phase": "7b_dual_gpu_mp",
        "config": {
            "n_steps": N_STEPS,
            "alpha": ALPHA,
            "max_seq_len": MAX_SEQ_LEN,
            "compile_mode": os.environ.get("BENCH_COMPILE_MODE", "default"),
            "fp8": "Float8DynamicActivationFloat8WeightConfig (PerTensor)",
            "vae_fp8": os.environ.get("BENCH_VAE_FP8", "1") != "0",
            "use_sage": os.environ.get("BENCH_USE_SAGE", "0") == "1",
            "sage_version": os.environ.get("BENCH_SAGE_VERSION", "1") if os.environ.get("BENCH_USE_SAGE", "0") == "1" else None,
            "sizes": SIZES,
            "timed_iters_per_gpu": TIMED_ITERS_PER_GPU,
        },
        "results": results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*78}\nDUAL-GPU MULTIPROCESS SUMMARY\n{'='*78}", flush=True)
    print(f"{'size':>6} {'solo_a_fps':>11} {'solo_b_fps':>11} {'parallel_fps':>14} {'scaling':>10}")
    print("-" * 70)
    for r in results:
        sa = r["solo"][0]["fps"]
        sb = r["solo"][1]["fps"]
        pf = r["parallel"]["throughput_fps"]
        sc = r["parallel"]["scaling_vs_solo_a"]
        print(f"{r['size_px']:>6} {sa:>10.2f}  {sb:>10.2f}  {pf:>13.2f}  {sc:>9.2f}x")
    print(f"\n[done] {OUT_DIR}/summary.json", flush=True)


if __name__ == "__main__":
    main()
