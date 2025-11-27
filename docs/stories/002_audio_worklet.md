# vj0 – Story V2: AudioWorklet-based audio features bus

## Context

vj0 currently:

- Uses Web Audio with `getUserMedia` + `AnalyserNode`
- Renders a real time waveform to a canvas at 60 fps via a `WaveformRenderer`
- Has an `AudioEngine` that exposes `getTimeDomainData(...)` and `bufferSize`

This works, but:

- Analysis is still tied to the main thread via `AnalyserNode`
- There is no structured `AudioFeatures` object we can map to visuals, DMX, or AI
- Longer term we want stable low latency analysis even when the UI is busy

**Goal of this iteration:**  
Move audio analysis into an `AudioWorklet` and define a simple but extensible **audio features bus** that the rest of the app can consume.

The waveform should still work exactly as before. We just add an additional way to read high level audio features from the engine.

## User story

> As a live visual artist (VJ) using vj0  
> I want vj0 to compute audio features (like loudness and simple frequency bands) in a dedicated audio worklet  
> So that I get stable, low latency data to drive visuals, lights, and AI without glitches when the UI is busy

## Tech / architectural requirements

- Keep using Next.js + TypeScript + React as in v1.
- Keep the existing `AudioEngine` / `WaveformRenderer` public interfaces working.
- Introduce:
  - An `AudioWorkletProcessor` module (e.g. `public/audio-worklet/vj0-audio-processor.js` or similar) compiled/bundled correctly so it can be loaded via `audioContext.audioWorklet.addModule(...)`.
  - A new TypeScript type `AudioFeatures` describing the extracted features.
- The worklet must run on the audio rendering thread and communicate via `MessagePort`.
- For this iteration a simple message based protocol is ok (`postMessage`). We do not need `SharedArrayBuffer` yet, but the design should not block it later.

## Audio features definition

Create a TypeScript type (for example in `src/lib/audio-features.ts`):

```ts
export type AudioFeatures = {
  rms: number; // root-mean-square loudness, 0..1 normalized
  peak: number; // peak absolute amplitude, 0..1
  energyLow: number; // energy in a low frequency band (e.g. 20–250 Hz), 0..1
  energyMid: number; // energy in a mid band (e.g. 250–4000 Hz), 0..1
  energyHigh: number; // energy in a high band (e.g. 4 kHz–20 kHz), 0..1
  spectralCentroid: number; // normalized 0..1
  // reserved for future:
  // beat: boolean;
  // tempo: number | null;
};
```

We only need basic approximations for now; they don’t need to be musically perfect, just consistent and normalized.

## Worklet implementation

Create an `AudioWorkletProcessor` module:

- Name the processor `"vj0-audio-processor"`.
- It should:

  - Receive audio from the first input channel.
  - For each `process()` call:
    - Compute `rms` and `peak` over the current block.
    - Compute a simple FFT / spectrum to get frequency bins.
    - From the spectrum, approximate:
      - `energyLow`, `energyMid`, `energyHigh`
      - `spectralCentroid`
    - Normalize all feature values to `[0, 1]`.
    - Pack them into a plain object that matches `AudioFeatures`.
    - Post the features to the main thread via `this.port.postMessage(features)`.

- It is acceptable if this posts a message for every render quantum. The engine on the main thread can just keep the latest features.

## AudioEngine changes (`src/lib/audio-engine.ts`)

Extend the existing `AudioEngine`:

- On `init()`:

  - After creating `AudioContext`, call  
    `audioContext.audioWorklet.addModule("path/to/vj0-audio-processor.js")`.
  - Create an `AudioWorkletNode` with the name `"vj0-audio-processor"`.
  - Connect the `MediaStreamAudioSourceNode` to the `AudioWorkletNode`.
  - Optionally connect the `AudioWorkletNode` to a `GainNode` with `gain = 0` to avoid feedback but keep the graph valid (or to `ctx.destination` if monitoring is desired; for now monitoring can be off).
  - Subscribe to `workletNode.port.onmessage` and store the latest `AudioFeatures` object in a private field, e.g. `this.latestFeatures`.

- Add a new method:

```ts
getLatestFeatures(): AudioFeatures | null;
```

- Returns the most recently received `AudioFeatures` (or `null` if nothing has been received yet).

- Keep existing methods working:
  - `init(deviceId?)`
  - `getTimeDomainData(target)`
  - `bufferSize`
  - `destroy()`
- If necessary, keep a separate `AnalyserNode` for the waveform path for now; we can refactor later.

## WaveformRenderer and React integration

- The `WaveformRenderer` should not change for this iteration.
- The React component (`app/vj/page.tsx` or `VJWaveform`) should continue to:
  - Use `getTimeDomainData(...)` to render the waveform.

## Debug UI for features

Add a simple debug panel for the current `AudioFeatures` values:

- Place it above or below the waveform.
- Show:
  - Numeric readouts: `rms`, `peak`, `spectralCentroid`.
  - Simple horizontal bars for `energyLow`, `energyMid`, `energyHigh`.

Implementation notes:

- Poll `getLatestFeatures()` inside a `requestAnimationFrame` loop (or reuse the existing render loop) instead of `setInterval`.
- **Do NOT** put features directly into React state at 60 fps.
  - Options:
    - Use a `useRef` to store the latest features and only update state at a throttled rate (e.g. 10–20 fps) for display, or
    - Use a minimal custom rendering function that writes into DOM elements without triggering React re-renders each frame.
- Goal: live, low overhead display of features without compromising performance.

## Non functional requirements

- Audio processing in the worklet must not allocate large objects in the hot path.
  - Reuse arrays where possible.
  - Avoid heavy math in JS loops if not necessary.
- Feature computation should be fast enough to avoid audio dropouts on a typical laptop.
- The public API of `AudioEngine` must remain simple:
  - `getTimeDomainData(target: Float32Array)`
  - `getLatestFeatures(): AudioFeatures | null`

## Out of scope for this iteration

- Beat detection and accurate tempo estimation
- Meyda integration (design should not prevent using Meyda inside the worklet later)
- WebRTC, WebCodecs, AI, WebUSB, or DMX
- Complex UI design

## Deliverables

- Updated vj0 repository with:

  - AudioWorklet processor module (e.g. `public/audio-worklet/vj0-audio-processor.js`, or another appropriate path wired into the build).
  - Updated `AudioEngine` that:
    - Loads and uses the AudioWorklet
    - Exposes `getLatestFeatures()`
  - Existing waveform still working exactly as before.
  - A small debug panel in the VJ page that displays current `AudioFeatures`.

- Short comments in the code explaining:
  - Where to extend `AudioFeatures` in the future (e.g. beat / tempo).
  - How to switch from message based communication to `SharedArrayBuffer` if needed.
  - How visuals and DMX can later consume `AudioFeatures` instead of raw waveform data.
