# 003_scenes – Scene system + audio‑reactive visuals

## Context

vj0 currently:

- Uses Web Audio + `AudioWorklet` via an `AudioEngine` that exposes:
  - `getTimeDomainData(target: Float32Array)` for waveform data
  - `getLatestFeatures(): AudioFeatures | null` for high level features (`rms`, `peak`, low/mid/high, `spectralCentroid`)
- Renders a real time waveform to a main canvas at 60 fps.
- Provides a device selector and status indicator (`RUNNING`, etc.).
- Shows an **Audio Features Debug** panel below the canvas, visualizing the current feature values.

Right now this is effectively one hard‑coded visual: a waveform view with an attached debug panel.

We want to generalize the visuals into a **scene system**, so vj0 can host multiple audio‑reactive looks while reusing the same audio engine and features.

The existing audio debug panel is very useful and should stay on the main page for now as a collapsible “Audio Features Debug” / “Audio Lab” section, similar to a mixer / meter view in Ableton.

---

## Goal of this story

Introduce a **scene abstraction and scene manager**, refactor the existing waveform into a scene, and add at least one additional audio‑reactive scene.

After this story, vj0 should let the user:

- Switch between different audio‑reactive visual scenes in the main canvas while audio is running.
- Have scenes that react both to **time‑domain data** and **AudioFeatures**.
- Keep all visuals decoupled from React and easy to port to WebGL/WebGPU later.

The audio debug panel remains on the main `/vj` view as a collapsible section.

---

## User story

> As a live visual artist (VJ) using vj0  
> I want to switch between different audio‑reactive visual scenes in the main canvas  
> So that I can perform with different looks without restarting the app or touching the audio routing

---

## Architectural requirements

### 1. Scene interface

Create a scene interface in `src/lib/scenes/types.ts`:

```ts
import type { AudioFeatures } from "../audio-features";

export interface VjScene {
  readonly id: string;
  readonly name: string;

  init?(canvas: HTMLCanvasElement): void;
  resize?(width: number, height: number): void;

  render(
    ctx: CanvasRenderingContext2D,
    features: AudioFeatures | null,
    timeDomain: Float32Array | null,
    dt: number
  ): void;

  destroy?(): void;
}
```

Notes:

- `timeDomain` is the current waveform buffer or `null`.
- `features` is the latest `AudioFeatures` from `AudioEngine`.
- `dt` is the time delta in seconds since the previous frame.
- We pass a 2D context for now, but the interface should make it easy to swap to a WebGL/WebGPU abstraction later.

### 2. Visual engine / scene manager

Create a visual engine in `src/lib/visual-engine.ts` that:

- Owns:
  - The main visual canvas element.
  - The `CanvasRenderingContext2D` for that canvas.
  - The `requestAnimationFrame` loop.
- Uses:
  - An `AudioEngine` instance for time‑domain data and features.
  - An array of `VjScene` implementations (scene registry).

Responsibilities:

- Keep track of:
  - The current scene (`currentScene`).
  - Canvas size and resize handling.
  - Last frame timestamp to compute `dt`.

- Per frame (called from `requestAnimationFrame`):
  - Fill a reusable `Float32Array` with time‑domain audio samples via `audioEngine.getTimeDomainData(timeDomainBuffer)`.
  - Read `const features = audioEngine.getLatestFeatures()`.
  - Compute `dt` in seconds.
  - Clear the canvas.
  - Call:
    ```ts
    currentScene.render(ctx, features, timeDomainBuffer, dt);
    ```

Suggested public API:

```ts
export class VisualEngine {
  constructor(
    canvas: HTMLCanvasElement,
    audioEngine: AudioEngine,
    scenes: VjScene[]
  );

  start(): void;
  stop(): void;

  setSceneById(id: string): void;
  getCurrentScene(): VjScene;

  handleResize(): void; // recompute canvas width/height and call currentScene.resize()
}
```

Constraints:

- The visual engine owns exactly **one** `requestAnimationFrame` loop for visuals.
- No React state updates are triggered from the animation loop.

### 3. Refactor existing waveform into a scene

Create `src/lib/scenes/waveform-scene.ts`:

- Implement `VjScene` using the existing waveform drawing logic from `WaveformRenderer`.
- Use `timeDomain` as the primary input:
  - Draw the waveform as it is today.
- Optionally modulate:
  - Line thickness or vertical scale with `features?.rms`.
  - Color with `features?.spectralCentroid` (e.g. lerp between two neon colors).

The behavior should be visually similar to the current main view so this becomes the **Waveform** scene.

Remove or deprecate the separate `WaveformRenderer` in favor of this scene‑based approach. Any reusable drawing utilities can be kept as pure helper functions.

### 4. Add at least one new audio‑reactive scene

Create `src/lib/scenes/spectrum-bars-scene.ts`:

- Use `AudioFeatures.energyLow`, `energyMid`, `energyHigh` to drive the visual.
- Split the canvas into a small number of vertical bars (for example 3, 5, or 8).
- For each bar:
  - Map one or more of the feature values to the bar height.
  - Apply simple smoothing / decay so bars fall more slowly than they rise (e.g. keep a per‑bar “display value” and lerp towards the current value).
- Use distinct colors, ideally aligned with the debug panel colors (e.g. red for low, orange/yellow for mid, cyan for high) to keep the vj0 aesthetic.

Optional third scene (if time allows): `radial-pulse-scene.ts`

- Center a circle (or ring) in the middle of the canvas.
- Radius driven by `rms` or `peak`.
- Color hue or brightness driven by `spectralCentroid`.
- This scene is optional for this story but should follow the same `VjScene` interface.

### 5. Scene registry

Create a small registry in `src/lib/scenes/index.ts`:

```ts
import type { VjScene } from "./types";
import { WaveformScene } from "./waveform-scene";
import { SpectrumBarsScene } from "./spectrum-bars-scene";
// import { RadialPulseScene } from "./radial-pulse-scene";

export const SCENES: VjScene[] = [
  new WaveformScene(),
  new SpectrumBarsScene(),
  // new RadialPulseScene(),
];
```

The `VisualEngine` receives `SCENES` and uses the first scene as default.

Adding future scenes should be as simple as:

1. Implement `VjScene` in its own file.
2. Add a new instance to `SCENES`.
3. The scene automatically appears in the UI selector.

---

## UI requirements

All of the following applies to the main vj page (e.g. `app/vj/page.tsx`) that vj0 already uses.

### 1. Integrate VisualEngine

- Replace the direct usage of `WaveformRenderer` with `VisualEngine`:
  - On mount (`useEffect`):
    - Create `AudioEngine` as before (already done in earlier stories).
    - After audio init and canvas reference are ready, instantiate `VisualEngine` with the main canvas, `AudioEngine`, and `SCENES`.
    - Call `visualEngine.start()`.
  - On unmount:
    - Call `visualEngine.stop()` and `audioEngine.destroy()`.

- Ensure canvas internal resolution is still set explicitly (e.g. `width = 1024`, `height = 256`) and CSS is used for scaling.

### 2. Scene selector controls

Add a simple scene selector control above or below the main canvas:

- Either:
  - A `<select>` listing all scenes by `scene.name`, or
  - A row of buttons / segmented control.

Behavior:

- When selection changes, call `visualEngine.setSceneById(id)`.
- Visually indicate the current scene (selected state, active button style, etc.).

The selector must be straightforward to extend when new scenes are added to `SCENES`.

### 3. Keyboard shortcuts (optional but nice)

- Add keyboard shortcuts so that when the vj0 page has focus:
  - Pressing `1` selects the first scene in `SCENES`.
  - Pressing `2` selects the second scene, etc.
- Implement this using `useEffect` with a keydown listener on `window`.
- Make sure listeners are cleaned up on unmount.

### 4. Audio Features Debug panel stays (collapsible)

- Keep the existing **Audio Features Debug** panel on the `/vj` page.
- The panel continues to read from `AudioEngine.getLatestFeatures()` as it does now.
- Ensure it remains **independent** of the current scene:
  - Even when switching scenes, the debug view continues to show live feature data.
- If not already implemented, ensure the panel is **collapsible** (e.g. a “Hide/Show Audio Features Debug” toggle) to reduce visual noise during performance.

This way the main view behaves a bit like Ableton for VJ:

- Main canvas = current “clip/scene” visualization.
- Debug panel = meters and analysis you can glance at when needed.

---

## Non‑functional requirements

- **Single render loop**:
  - Only `VisualEngine` should call `requestAnimationFrame` for visuals.
  - Do not keep the old waveform RAF running alongside.

- **No React state per frame**:
  - The animation loop must not trigger React re‑renders every frame.
  - Scene selection can use React state, but the state should not change on every frame.

- **Minimal allocations in render**:
  - Reuse the `Float32Array` for time‑domain data.
  - Avoid creating temporary arrays or objects inside `scene.render(...)` on each frame.
  - If smoothing/decay is required, store per‑scene state on `this`, not in freshly allocated objects.

- **Easy extensibility**:
  - Adding a new scene must not require changes in `VisualEngine` logic.
  - The only required changes should be:
    - Implementing `VjScene` for the new visual.
    - Adding it to `SCENES`.

---

## Out of scope

The following are explicitly **not** part of this story:

- 3D/WebGL/WebGPU/shader rendering (for a future story).
- DMX / WebUSB integration.
- AI / WebRTC / video streaming.
- A full mod‑matrix or UI for editing feature‑to‑parameter mappings:
  - For now, each scene hard‑codes its own simple mapping from `AudioFeatures`/`timeDomain` to visual parameters.
- Separate `/debug` or `/lab` route:
  - The existing debug UI stays on the main `/vj` view.

---

## Deliverables

- `src/lib/scenes/types.ts` with `VjScene` interface.
- `src/lib/visual-engine.ts` (or similar) that:
  - Owns the main canvas and animation loop.
  - Pulls audio data and features from `AudioEngine` once per frame.
  - Delegates drawing to the current `VjScene`.
  - Provides `start`, `stop`, `setSceneById`, `getCurrentScene`, and `handleResize` methods.
- `src/lib/scenes/waveform-scene.ts`:
  - Implements the current waveform look as a `VjScene`.
- `src/lib/scenes/spectrum-bars-scene.ts`:
  - Implements at least one new audio‑reactive visual based on low/mid/high energy.
- `src/lib/scenes/index.ts`:
  - Scene registry exporting `SCENES` array with at least the two scenes above.
- Updated `app/vj/page.tsx` (or equivalent):
  - Uses `VisualEngine` instead of the old `WaveformRenderer`.
  - Provides a scene selector UI.
  - Keeps the Audio Features Debug panel working and collapsible.

The final result should allow running vj0 locally, selecting an audio device, switching between at least two scenes, and observing both the visual change and the live audio feature debug panel on the same page.
