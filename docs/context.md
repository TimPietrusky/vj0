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
- **Modern browser APIs** – Web Audio API, Canvas 2D (upgradeable to WebGL/WebGPU), future AudioWorklet support
- **Performance-first design** – Zero allocations in render loops, no React state for audio buffers

---

## 📁 Project Structure

```
vj0/
├── app/                      # Next.js App Router pages
│   ├── layout.tsx            # Root layout with fonts and metadata
│   ├── page.tsx              # Home page (currently default Next.js starter)
│   ├── globals.css           # Global styles + Tailwind config
│   └── favicon.ico           # App favicon
│
├── docs/                     # Documentation
│   ├── context.md            # This file – project context for contributors
│   └── stories/              # User stories and feature specs
│       └── 001_init.md       # Initial story: audio waveform visualization
│
├── public/                   # Static assets
│   └── *.svg                 # Various icons (Next.js defaults)
│
├── src/                      # Source code (to be created)
│   └── lib/                  # Framework-agnostic modules
│       ├── audio-engine.ts   # Web Audio API abstraction
│       └── waveform-renderer.ts # Canvas rendering
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
│  │  - VJWaveform (canvas + status)                             ││
│  │  - Device selector (audio input dropdown)                   ││
│  └───────────────────────┬─────────────────────────────────────┘│
│                          │ uses (no React state for buffers)    │
│  ┌───────────────────────▼─────────────────────────────────────┐│
│  │              Framework-Agnostic Core Modules                 ││
│  │  ┌─────────────────┐    ┌─────────────────────────┐         ││
│  │  │  AudioEngine    │    │   WaveformRenderer      │         ││
│  │  │  - Web Audio API│    │   - Canvas 2D           │         ││
│  │  │  - AnalyserNode │───▶│   - requestAnimationFrame│        ││
│  │  │  (→AudioWorklet)│    │   (→WebGL/WebGPU)       │         ││
│  │  └─────────────────┘    └─────────────────────────┘         ││
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

### Core Modules (planned in `src/lib/`)

#### AudioEngine (`audio-engine.ts`)

- Pure TypeScript class (no React dependency)
- Handles: `getUserMedia`, `AudioContext`, `MediaStreamAudioSourceNode`, `AnalyserNode`
- API:
  - `init(deviceId?)` – Initialize audio with optional device selection
  - `getTimeDomainData(buffer: Float32Array)` – Fill buffer with samples
  - `bufferSize` – Getter for FFT size
  - `destroy()` – Cleanup resources
- **Extension point**: Replace `AnalyserNode` with `AudioWorklet` for advanced analysis

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

### File Naming

- React components: `PascalCase.tsx`
- Utility modules: `kebab-case.ts`
- Pages: `page.tsx` (Next.js convention)

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

### Performance Critical Areas

- **Never** allocate in the render loop (`start()` callback)
- **Never** use React state for audio buffer data
- **Always** reuse `Float32Array` buffers

### Extension Points (marked in code)

1. **AudioWorklet**: Replace `AnalyserNode` in `audio-engine.ts`
2. **WebGL/WebGPU**: Swap Canvas 2D in `waveform-renderer.ts`
3. **DMX Module**: Hook into renderer output
4. **AI Module**: Process canvas frames

### Browser Compatibility

- Requires modern browsers with Web Audio API support
- `getUserMedia` requires HTTPS in production
- WebGPU support is experimental (future consideration)

---

## 🔗 Related Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)
- [modV](https://github.com/vcync/modv) – Related VJ project by the same author
- [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)

---

_This document should be updated as the project evolves._
