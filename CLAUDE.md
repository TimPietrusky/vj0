# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**vj0** is a web-based VJ tool for live visual artists. It captures audio from USB interfaces, analyzes it in real-time, renders audio-reactive visuals to canvas at 60fps, and controls stage lights via WebUSB DMX512. An AI transport layer (WebRTC) exists for future cloud-based image-to-image generation.

## Commands

```bash
pnpm dev      # Start Next.js dev server (localhost:3000)
pnpm build    # Production build
pnpm start    # Start production server
pnpm lint     # ESLint (flat config, v9)
```

No test framework is configured. Audio features require real browser microphone permissions — manual browser testing is expected.

## Architecture

### Two-Layer Design

**React UI layer** (`app/`) — Next.js 16 App Router with server components by default, client components marked `"use client"`. Handles UI state, device selection, and panel rendering.

**Framework-agnostic core** (`src/lib/`) — Pure TypeScript classes with no React dependency. All audio, rendering, and lighting logic lives here.

### Core Modules

- **AudioEngine** (`src/lib/audio-engine.ts`) — Web Audio API wrapper. Connects `getUserMedia` → `AnalyserNode` (waveform) + `AudioWorkletNode` (feature extraction). Features arrive via MessagePort from the off-main-thread worklet.
- **AudioWorklet** (`public/audio-worklet/vj0-audio-processor.js`) — Plain JS (not TypeScript). Computes RMS, peak, energy bands (low/mid/high via FFT), spectral centroid. All values normalized 0..1.
- **VisualEngine** (`src/lib/scenes/visual-engine.ts`) — Single `requestAnimationFrame` loop. Polls audio features, delegates to the active `VjScene` for rendering.
- **VjScene** (`src/lib/scenes/types.ts`) — Interface for visual scenes. Lifecycle: `init?` → `render` (every frame) → `resize?` → `destroy?`. Scenes are registered in `SCENES` array in `src/lib/scenes/index.ts`.
- **LightingEngine** (`src/lib/lighting/lighting-engine.ts`) — Samples canvas pixels at fixture mapping coordinates on a `setInterval` (30Hz, decoupled from 60fps render loop). Maps RGB to DMX channels via fixture profiles. Reuses a single `Uint8Array(512)` universe buffer.
- **DmxOutput** (`src/lib/lighting/dmx-output.ts`) — WebUSB wrapper for Arduino Leonardo-based DMX512 controllers. Graceful no-op when WebUSB unavailable.
- **AI Transport** (`src/lib/ai/`) — `AiTransport` interface with WebRTC data channel implementation. Includes backpressure via `canSend()`.

### Orchestrator

`app/vj/VJApp.tsx` is the main client component (~860 lines). It wires all engines together using `useRef` for engine instances (not useState) and manages UI state. This is the entry point for understanding how modules connect.

### State Management

Zustand is used **only** for UI persistence (fixture settings in `src/lib/stores/lighting-store.ts` with localStorage). Audio buffers, render loop data, and engine state live in refs — never in React state.

### Fixture System

Fixture profiles define DMX channel layouts (`src/lib/lighting/fixtures/`). `FixtureInstance` binds a profile to a DMX address and canvas sampling coordinates. Users manage fixtures via the Zustand store.

## Key Conventions

- **Path alias**: `@/*` maps to project root
- **File naming**: Components `PascalCase.tsx`, utilities `kebab-case.ts`, pages `page.tsx`
- **No legacy code**: Delete deprecated code immediately. No `@deprecated` annotations, no backward-compatibility shims. History lives in git.
- **No allocations in hot paths**: Reuse `Float32Array`/`Uint8Array` buffers. No `map`/`filter` in render loops.
- **Audio buffers never in useState**: Use `useRef` for engine instances and buffers. Debug panels may use throttled `useState` (e.g., 10fps).
- **Tailwind CSS 4** with PostCSS. Theming via CSS variables in `globals.css`.

## Adding a New Scene

1. Create `src/lib/scenes/my-scene.ts` implementing `VjScene` interface
2. Add instance to `SCENES` array in `src/lib/scenes/index.ts`

## Adding a New Fixture Profile

1. Create profile in `src/lib/lighting/fixtures/`
2. Add to `FIXTURE_PROFILES` array in `src/lib/lighting/fixtures/index.ts`

## FLUX.2 Klein worker — Docker deploy

The img2img worker (`workers/runpod-flux2klein/`) is shipped as a baked
Docker image: **`nerddisco/vj0-flux2klein-worker:latest`**. RunPod pods
created from this image auto-launch the WebRTC dispatcher + spawn one
Python inference worker per GPU on container start — no `setup.sh` ritual.

**Verified to boot to `inferenceReady:true` in ~7 min from absolute cold
start (no `/workspace` cache); ~30-60 s when the network volume's
torch-Inductor + HF caches are warm.**

### What's already optimized (in the image)

| Optimization | Source | Phase |
|---|---|---|
| `torch==2.11.0+cu128` (Blackwell + driver-570 compat) | Dockerfile | base |
| TorchAO `Float8DynamicActivationFloat8WeightConfig(PerTensor())` on transformer | `inference_server.py` | 3 + 3b |
| fp8 on VAE linears | `inference_server.py` (`USE_VAE_FP8=1` default in Dockerfile) | 3c |
| `compile_mode=reduce-overhead` (CUDA-graph capture) | Dockerfile (`COMPILE_MODE=reduce-overhead`) | 9A |
| Multi-GPU multiprocess round-robin (auto-detects N GPUs) | `server.js` | 7b |
| Load-aware dispatcher + dispatch-time stale drop | `server.js` (`MAX_PENDING_PER_WORKER=3`) | 7c |
| Per-stage timing (vae/transformer/jpeg ms breakdown) | `inference_server.py` | 9 polish |
| Phase events for boot UX (`loading_weights`, `applying_fp8`, …) | `inference_server.py` + `server.js` | 9 polish |
| Inductor cache hydration `/workspace ↔ /tmp` | `entrypoint.sh` | ops |
| HEALTHCHECK on `/healthz inferenceReady:true` | Dockerfile | ops |

**Pod boot recipe** — production performance on this stack: 50.67 fps @
256² / 4-step dual-GPU, 17.11 fps @ 512² / 4-step dual-GPU. See
`workers/runpod-flux2klein/BENCH-2026-04-30.md` for the full grind.

### Adding a new optimization

If you find a real perf win, ship it through the worker → image pipeline:

1. **Bench it first.** A new variant goes in `workers/runpod-flux2klein/bench_*.py`
   and runs against the existing dual-GPU bench harness. See `bench_aoti.py`
   (Phase 9D), `bench_kv_fp8.py` (Phase 9E), or `bench_dual_gpu_mp.py`
   (the canonical multi-GPU bench) for templates.
2. **Document the result** in `BENCH-2026-04-30.md` Phase 9 (the table at
   the top + a per-variant section). **Wins AND losses** — the no-go
   results (max-autotune OOM, SageAttention graph breaks, etc.) are how
   future-you doesn't waste hours re-discovering the same dead end.
3. **Wire it into `inference_server.py`** behind an env flag with a sane
   default (look at `USE_VAE_FP8`, `COMPILE_MODE` for the pattern).
4. **Update Dockerfile ENV defaults** if the new feature is on by default.
5. **Push to `main`.** The path filter on
   `.github/workflows/runpod-flux2klein-docker.yml` catches changes to
   `Dockerfile`, `entrypoint.sh`, `server.js`, `inference_server.py`,
   `package.json`, `requirements.txt` — Blacksmith builds + pushes a new
   `:latest` to Docker Hub. Subsequent pod restarts pick up the new image
   (RunPod re-pulls when `:latest` digest changes, or you can pin `:<sha>`
   for reproducibility).
6. **Smoke-test on a 1-GPU EU-RO-1 test pod** before promoting to prod.
   See "How to test the image" in `BENCH-2026-04-30.md`.

### Things deliberately NOT in the image (Phase 9 verdicts)

- **AOT-Inductor** (`bench_aoti.py`) — works, equal perf to JIT, 3.8 GB/.pt2 per shape. UX win only (deploy warmup → 0 s) — defer until cold-start matters more than disk cost.
- **fp8 KV-cache** (`bench_kv_fp8.py`) — needs `Flux2KVAttnProcessor` subclass + per-head scale calibration + `custom_op` wrapper for compile compat. Multi-day port; deferred until per-stage profile pinpoints attention.
- **SageAttention v1/v3** — both regressions on Klein due to head_dim mismatches + Python wrapper graph breaks. Documented in BENCH Phase 9C/9F. Don't re-bench unless Sage adds proper `torch.library.custom_op` registration.
- **`compile_mode=max-autotune`** — Triton wants 122-196 KB shmem, Blackwell sm_120 has 101 KB. Wait for torch 2.12+ Blackwell-aware autotune templates.

## Project Documentation

`docs/context.md` is the comprehensive project bible — read it for full architectural details, design principles, and extension points. `docs/stories/` contains user stories with detailed specs for each feature.
