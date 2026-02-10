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
| State (UI only) | Zustand              | ^5                   |
| Linting         | ESLint               | ^9                   |
| Package Manager | npm                  | (npm-based lockfile) |

### Key Technical Decisions

- **Minimal client-side state for hot paths** – Use Zustand only for UI settings/persistence, never for audio buffers or render loop state
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
│   ├── page.tsx              # Home page (VJ visualization)
│   ├── globals.css           # Global styles + Tailwind config
│   ├── favicon.ico           # App favicon
│   └── vj/                   # VJ visualization route
│       ├── page.tsx          # VJ page (server component wrapper)
│       ├── VJApp.tsx         # Main VJ client orchestrator (engines, refs)
│       └── components/       # UI components (presentational)
│           ├── StatusBar.tsx
│           ├── AudioDebugPanel.tsx
│           ├── LightingPanel.tsx
│           ├── DmxControls.tsx
│           ├── FixtureSelector.tsx
│           ├── FixtureInspector.tsx
│           ├── FeatureBar.tsx
│           └── index.ts
│
├── docs/                     # Documentation
│   ├── context.md            # This file – project context for contributors
│   └── stories/              # User stories and feature specs
│       ├── 001_init.md       # Initial story: audio waveform visualization
│       ├── 002_audio_worklet.md # AudioWorklet-based audio features bus
│       ├── 003_scenes.md     # Scene system: VjScene + VisualEngine
│       ├── 004_lighting_dmx.md # Lighting system: DMX + fixtures + WebUSB
│       └── 005_webrtc_poc.md # Remote WebRTC PoC to GPU worker (transport only)
│
├── public/                   # Static assets
│   ├── *.svg                 # Various icons (Next.js defaults)
│   └── audio-worklet/        # AudioWorklet processor modules
│       └── vj0-audio-processor.js # Real-time audio feature extraction
│
├── src/                      # Source code
│   └── lib/                  # Framework-agnostic modules
│       ├── ai/               # Remote AI transports (browser-side)
│       │   ├── transport.ts  # AiTransport interface + shared types
│       │   └── webrtc-transport.ts # WebRTC data-channel transport (PoC)
│       ├── audio-engine.ts   # Web Audio API + AudioWorklet abstraction
│       ├── audio-features.ts # AudioFeatures type definition
│       ├── scenes/           # Scene system
│       │   ├── types.ts      # VjScene interface
│       │   ├── visual-engine.ts # Scene manager + render loop
│       │   ├── waveform-scene.ts # Waveform visualization
│       │   ├── spectrum-bars-scene.ts # Spectrum analyzer bars
│       │   └── index.ts      # Scene registry + exports
│       ├── lighting/         # DMX lighting system
│       │   ├── types.ts      # DmxUniverse, FixtureProfile, FixtureInstance
│       │   ├── lighting-engine.ts # Canvas sampling + universe building
│       │   ├── dmx-output.ts # WebUSB DMX512 controller
│       │   ├── fixtures/     # Fixture profiles
│       │   │   ├── fun-gen-separ-quad.ts # Fun Generation SePar Quad profile
│       │   │   ├── stairville-wild-wash-pro.ts # Stairville Wild Wash Pro 648
│       │   │   └── index.ts  # FIXTURE_PROFILES array + exports
│       │   └── index.ts      # Lighting module exports
│       └── stores/           # Zustand stores (UI state only)
│           ├── lighting-store.ts # Fixture settings with localStorage persistence
│           └── index.ts      # Store exports
│
├── workers/                  # Standalone services (deployed separately)
│   └── webrtc-echo-worker/   # Remote WebRTC echo worker (RunPod Docker image)
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
│  │  - VJApp (orchestrator) + StatusBar, LightingPanel, etc.   ││
│  │  - Device selector (audio input dropdown)                   ││
│  └───────────────────────┬─────────────────────────────────────┘│
│                          │ uses (no React state for buffers)    │
│  ┌───────────────────────▼─────────────────────────────────────┐│
│  │              Framework-Agnostic Core Modules                 ││
│  │  ┌─────────────────┐    ┌─────────────────────────┐         ││
│  │  │  AudioEngine    │    │   VisualEngine          │         ││
│  │  │  - Web Audio API│    │   - Scene manager       │         ││
│  │  │  - AnalyserNode │───▶│   - Single rAF loop     │         ││
│  │  │  - AudioWorklet │    │   - Canvas 2D context   │         ││
│  │  └────────┬────────┘    └───────────┬─────────────┘         ││
│  │           │ MessagePort             │ delegates to           ││
│  │  ┌────────▼────────┐    ┌───────────▼─────────────┐         ││
│  │  │ vj0-audio-      │    │   VjScene (interface)   │         ││
│  │  │ processor.js    │    │   - WaveformScene       │         ││
│  │  │ → AudioFeatures │    │   - SpectrumBarsScene   │         ││
│  │  └─────────────────┘    │   - (extensible)        │         ││
│  │                         └─────────────────────────┘         ││
│  │                                     │ canvas pixels          ││
│  │  ┌──────────────────────────────────▼───────────────────┐   ││
│  │  │              Lighting System                          │   ││
│  │  │  ┌─────────────────┐    ┌─────────────────────────┐  │   ││
│  │  │  │ LightingEngine  │    │   DmxOutput             │  │   ││
│  │  │  │ - Canvas sample │───▶│   - WebUSB DMX512       │  │   ││
│  │  │  │ - Fixture map   │    │   - Arduino Leonardo    │  │   ││
│  │  │  │ - DMX universe  │    │   - 512 channels        │  │   ││
│  │  │  └─────────────────┘    └─────────────────────────┘  │   ││
│  │  └──────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                    Future Extensions
                              ▼
        ┌─────────────┬───────────────┬─────────────┐
        │  AI Module  │  Multi-Univ   │  WebGPU     │
        │  (cloud GPU)│  Art-Net/sACN │  Scenes     │
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

#### VisualEngine (`scenes/visual-engine.ts`)

- Pure TypeScript class (no React dependency)
- Manages single `requestAnimationFrame` loop, canvas context, and active scene
- Delegates rendering to current `VjScene` implementation
- API:
  - `constructor(canvas, audioEngine, scenes)` – Setup with canvas, audio source, and scene registry
  - `start()` / `stop()` – Control render loop
  - `setSceneById(id)` – Switch active scene
  - `getCurrentScene()` / `getScenes()` – Query scene state
  - `handleResize(width, height)` – Update canvas dimensions
  - `destroy()` – Cleanup resources
- **Extension point**: Add scene transitions, layer compositing, post-processing effects

#### VjScene Interface (`scenes/types.ts`)

- Contract for all visual scenes
- Properties: `id`, `name` (readonly)
- Lifecycle: `init?(canvas)`, `resize?(width, height)`, `destroy?()`
- Render: `render(ctx, features, timeDomain, dt)` – Called every frame
- **Design**: Scenes are self-contained, stateless between frames, receive all data via render params

#### Scene Registry (`scenes/index.ts`)

- `SCENES` array of instantiated `VjScene` implementations
- First scene is default
- Adding scenes: implement `VjScene`, add instance to array
- **Design**: Simple, no dynamic loading; future: lazy loading for heavy scenes

#### LightingEngine (`lighting/lighting-engine.ts`)

- Pure TypeScript class (no React dependency)
- Samples pixels from the main canvas at a configurable tick rate (default 30 Hz)
- Maps sampled colors to DMX channels via fixture profiles
- Runs on `setInterval`, decoupled from the 60 fps visual render loop
- API:
  - `constructor(canvas, fixtures, config)` – Setup with canvas, fixture array, and tick config
  - `start()` / `stop()` – Control lighting update loop
  - `onFrame(callback)` / `offFrame(callback)` – Subscribe to `LightingFrame` emissions
  - `getUniverse()` – Get current `DmxUniverse` buffer
  - `updateFixtureAddress(id, address)` – Runtime address change for fixtures
- **Design**: Reuses `Uint8Array(512)` universe buffer, no allocations per tick

#### DmxOutput (`lighting/dmx-output.ts`)

- WebUSB wrapper for DMX512 controllers (Arduino Leonardo-based)
- Handles device picker, connection, and data transmission
- API:
  - `connect()` / `disconnect()` – Manage WebUSB device lifecycle
  - `isConnected()` – Check connection status
  - `sendUniverse(universe)` – Send 512-byte DMX frame
- Sends blackout (all zeros) on disconnect
- Graceful fallback when WebUSB is not supported

#### Fixture System (`lighting/fixtures/`)

- `FixtureProfile`: Defines fixture type, mode, and channel layout (R, G, B, UV, dimmer, strobe, etc.)
- `FixtureInstance`: Concrete fixture with profile, DMX address, and canvas mapping coordinates
- `FIXTURE_PROFILES` array: Available profiles (SePar Quad, Stairville Wild Wash Pro 648)
- User fixtures managed via Zustand store with localStorage persistence
- **Extensibility**: Add new profiles and instances without modifying engine code

#### Lighting Store (`stores/lighting-store.ts`)

- Zustand store for fixture configuration
- Persisted to localStorage for session persistence
- Serializes fixtures (stores profile ID, not full profile object)
- Actions for add/remove/update fixtures and their settings

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

### No Legacy Code Policy

- **Delete** deprecated code immediately when replaced—do not keep "for reference"
- **No** `@deprecated` annotations—if code is deprecated, it should be deleted
- **No** backward-compatibility shims or fallbacks for removed features
- Code should always reflect the current architecture; history lives in git

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
2. **WebGL/WebGPU Scenes**: Create new `VjScene` implementations using WebGL/WebGPU instead of Canvas 2D
3. **Scene Transitions**: Add transition effects in `VisualEngine` when switching scenes
4. **AI Module**: Process canvas frames for cloud-based image-to-image generation
   - Browser-side transport abstraction lives under `src/lib/ai/` (currently WebRTC data channel PoC)
   - Remote workers are standalone deployables under `workers/` (Docker images for GPU hosts like RunPod)
5. **Beat Detection**: Add `beat`/`tempo` to AudioFeatures in worklet
6. **Multi-Universe DMX**: Extend lighting system to support Art-Net, sACN, or multiple USB universes
7. **Advanced Fixture Mapping**: Add multiple sample points per fixture, 2D matrix support, effects

### Browser Compatibility

- Requires modern browsers with Web Audio API and AudioWorklet support
- `getUserMedia` requires HTTPS in production
- WebGPU support is experimental (future consideration)
- SharedArrayBuffer requires Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
- WebUSB for DMX requires Chrome/Edge (not supported in Firefox/Safari); graceful fallback provided

---

## 🔗 Related Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)
- [AudioWorkletProcessor](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor)
- [WebUSB API](https://developer.mozilla.org/en-US/docs/Web/API/USB)
- [webusb-dmx512-controller](https://github.com/NERDDISCO/webusb-dmx512-controller) – Arduino-based DMX512 controller
- [modV](https://github.com/vcync/modv) – Related VJ project by the same author
- [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)

---

_This document should be updated as the project evolves._
