# vj0 – Project Context

> A comprehensive guide for developers and AI agents working in this codebase.

## 🎯 Project Overview

**vj0** is a web-based VJ tool designed for live visual artists. The application will eventually allow users to:

- Receive audio from USB audio interfaces
- Analyze audio in real-time
- Render visuals to canvas (2D, WebGL, WebGPU) at 60 fps
- Use visuals as input for cloud-based AI image-to-image models
- Control stage lights via WebUSB DMX512 controllers

---

## 🛠 Tech Stack

| Category        | Technology           | Version              |
| --------------- | -------------------- | -------------------- |
| Framework       | Next.js (App Router) | 16.0.5               |
| UI Library      | React                | 19.2.0               |
| Language        | TypeScript           | ^5                   |
| Styling         | Tailwind CSS         | ^4                   |
| Linting         | ESLint               | ^9                   |
| Package Manager | npm                  | (npm-based lockfile) |

### Key Technical Decisions

- **No heavy client-side state management** – Avoid libraries like Redux/Zustand for audio/rendering logic
- **Framework-agnostic core modules** – Audio and rendering logic lives in pure TypeScript classes, not React hooks
- **Modern browser APIs** – Web Audio API, Canvas 2D (upgradeable to WebGL/WebGPU), AudioWorklet for real-time analysis
- **Performance-first design** – Zero allocations in render loops, no React state for audio buffers
- **AudioWorklet for feature extraction** – Off-main-thread audio analysis with typed AudioFeatures bus

---

## 📁 Project Structure

```
vj0/
├── app/                      # Next.js App Router pages
│   ├── layout.tsx            # Root layout with fonts and metadata
│   ├── page.tsx              # Home page (currently default Next.js starter)
│   ├── globals.css           # Global styles + Tailwind config
│   ├── favicon.ico           # App favicon
│   └── vj/                   # VJ visualization route
│       ├── page.tsx          # VJ page (server component wrapper)
│       └── VJWaveform.tsx    # Waveform client component + debug panel
│
├── docs/                     # Documentation
│   ├── context.md            # This file – project context for contributors
│   └── stories/              # User stories and feature specs
│       ├── 001_init.md       # Initial story: audio waveform visualization
│       └── 002_audio_worklet.md # AudioWorklet-based audio features bus
│
├── public/                   # Static assets
│   ├── *.svg                 # Various icons (Next.js defaults)
│   └── audio-worklet/        # AudioWorklet processor modules
│       └── vj0-audio-processor.js # Real-time audio feature extraction
│
├── src/                      # Source code
│   └── lib/                  # Framework-agnostic modules
│       ├── audio-engine.ts   # Web Audio API + AudioWorklet abstraction
│       ├── audio-features.ts # AudioFeatures type definition
│       └── waveform-renderer.ts # Canvas rendering (implemented)
│
├── package.json              # Dependencies and scripts
├── tsconfig.json             # TypeScript configuration
├── next.config.ts            # Next.js configuration
├── postcss.config.mjs        # PostCSS with Tailwind plugin
├── eslint.config.mjs         # ESLint flat config with Next.js rules
└── README.md                 # Basic project readme
```

---

## 🏗 Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Next.js App (React)                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    React Components                          ││
│  │  - VJWaveform (canvas + status + debug panel)               ││
│  │  - Device selector (audio input dropdown)                   ││
│  └───────────────────────┬─────────────────────────────────────┘│
│                          │ uses (no React state for buffers)    │
│  ┌───────────────────────▼─────────────────────────────────────┐│
│  │              Framework-Agnostic Core Modules                 ││
│  │  ┌─────────────────┐    ┌─────────────────────────┐         ││
│  │  │  AudioEngine    │    │   WaveformRenderer      │         ││
│  │  │  - Web Audio API│    │   - Canvas 2D           │         ││
│  │  │  - AnalyserNode │───▶│   - requestAnimationFrame│        ││
│  │  │  - AudioWorklet │    │   (→WebGL/WebGPU)       │         ││
│  │  └────────┬────────┘    └─────────────────────────┘         ││
│  │           │ MessagePort                                      ││
│  │  ┌────────▼────────┐                                        ││
│  │  │ vj0-audio-      │ (runs in AudioWorklet thread)          ││
│  │  │ processor.js    │                                        ││
│  │  │ → AudioFeatures │                                        ││
│  │  └─────────────────┘                                        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                    Future Extensions
                              ▼
        ┌─────────────┬───────────────┬─────────────┐
        │  AI Module  │  DMX Module   │  WebGPU     │
        │  (cloud GPU)│  (WebUSB)     │  Renderer   │
        └─────────────┴───────────────┴─────────────┘
```

### Core Modules (`src/lib/`)

#### AudioEngine (`audio-engine.ts`)

- Pure TypeScript class (no React dependency)
- Handles: `getUserMedia`, `AudioContext`, `MediaStreamAudioSourceNode`, `AnalyserNode`, `AudioWorkletNode`
- Audio graph: source → analyser (waveform) + source → worklet → silent gain (features)
- API:
  - `init(deviceId?)` – Initialize audio with optional device selection, loads AudioWorklet
  - `getTimeDomainData(buffer: Float32Array)` – Fill buffer with samples (for waveform)
  - `getLatestFeatures(): AudioFeatures | null` – Get latest computed audio features
  - `bufferSize` – Getter for FFT size
  - `destroy()` – Cleanup resources
- **Extension point**: Replace MessagePort with SharedArrayBuffer for zero-copy transfer (requires COEP/COOP headers)

#### AudioFeatures (`audio-features.ts`)

- TypeScript type defining extracted audio characteristics
- Fields: `rms`, `peak`, `energyLow`, `energyMid`, `energyHigh`, `spectralCentroid`
- All values normalized 0..1
- Reserved for future: `beat`, `tempo`

#### vj0-audio-processor (`public/audio-worklet/vj0-audio-processor.js`)

- AudioWorkletProcessor running off main thread
- Computes RMS, peak, energy bands (low/mid/high via FFT), spectral centroid
- Posts AudioFeatures to main thread via MessagePort
- **Design**: Processes every quantum (~128 samples at 44.1kHz), throttles message posting

#### WaveformRenderer (`waveform-renderer.ts`)

- Pure TypeScript class (no React dependency)
- Handles: Canvas 2D context, `requestAnimationFrame` loop
- API:
  - `constructor(canvas: HTMLCanvasElement)` – Setup context
  - `start(callback, buffer)` – Begin render loop
  - `stop()` – Cancel animation frame
- **Extension point**: Swap Canvas 2D for WebGL/WebGPU renderer

### Design Principles

1. **Separation of Concerns** – React handles UI state, core modules handle audio/rendering
2. **No Allocations in Hot Paths** – Reuse `Float32Array` buffers, avoid `map`/`filter` in render loops
3. **Stable Public Interfaces** – Core module APIs designed for future extensibility without breaking changes
4. **Performance Budget** – Target 60 fps on typical laptops
5. **Off-Main-Thread Analysis** – AudioWorklet ensures stable, low-latency audio feature extraction

---

## 🎨 Styling Conventions

- **Tailwind CSS 4** with PostCSS integration
- CSS variables for theming in `globals.css`:
  - `--background` / `--foreground` (light/dark mode)
  - `--font-geist-sans` / `--font-geist-mono` (via next/font)
- **Font**: Geist (sans and mono variants) from Vercel

---

## 📋 Coding Conventions

### TypeScript

- Strict mode enabled
- Path alias: `@/*` maps to project root
- Target: ES2017
- Module resolution: bundler

### React

- App Router (Next.js 13+ style)
- Server Components by default
- Client Components marked with `"use client"` directive
- **Important**: Audio buffers MUST NOT use `useState` (no re-renders per frame)
- Debug panels can use throttled `useState` (e.g., 10fps) for feature display

### File Naming

- React components: `PascalCase.tsx`
- Utility modules: `kebab-case.ts`
- Pages: `page.tsx` (Next.js convention)
- AudioWorklet processors: `public/audio-worklet/*.js` (plain JS, not TypeScript)

---

## 📚 Documentation Structure

### `/docs/stories/`

User stories follow a structured format:

- **Title**: Feature name
- **Context**: Background and motivation
- **User Story**: "As a... I want... So that..."
- **Tech Stack Requirements**: Specific technologies to use
- **Functional Requirements**: Detailed implementation specs
- **Non-functional Requirements**: Performance, maintainability
- **Out of Scope**: Explicit exclusions
- **Deliverables**: Expected outcomes

---

## ⚠️ Important Notes for Contributors

### Testing and Development

- **No dev server or browser testing required** – Audio functionality requires real browser microphone permissions, which automated testing tools cannot grant. Code verification is sufficient; manual testing in a real browser is expected for audio features.

### Performance Critical Areas

- **Never** allocate in the render loop (`start()` callback)
- **Never** use React state for audio buffer data
- **Always** reuse `Float32Array` buffers
- Poll `getLatestFeatures()` in rAF loop, throttle React state updates

### Extension Points (marked in code)

1. **SharedArrayBuffer**: Replace MessagePort in `audio-engine.ts` for zero-copy audio features
2. **WebGL/WebGPU**: Swap Canvas 2D in `waveform-renderer.ts`
3. **DMX Module**: Consume AudioFeatures for light control
4. **AI Module**: Process canvas frames
5. **Beat Detection**: Add `beat`/`tempo` to AudioFeatures in worklet

### Browser Compatibility

- Requires modern browsers with Web Audio API and AudioWorklet support
- `getUserMedia` requires HTTPS in production
- WebGPU support is experimental (future consideration)
- SharedArrayBuffer requires Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers

---

## 🔗 Related Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)
- [AudioWorkletProcessor](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor)
- [modV](https://github.com/vcync/modv) – Related VJ project by the same author
- [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)

---

_This document should be updated as the project evolves._
