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

## Project Documentation

`docs/context.md` is the comprehensive project bible — read it for full architectural details, design principles, and extension points. `docs/stories/` contains user stories with detailed specs for each feature.
