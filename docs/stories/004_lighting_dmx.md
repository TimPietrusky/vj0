# 004_lighting_dmx – Lighting bus, fixtures & WebUSB DMX MVP

## Context

vj0 currently:

- Has a stable **audio engine** with time-domain data and `AudioFeatures` provided by an `AudioWorklet`-based `AudioEngine`.
- Renders visuals through a **scene system** on a main canvas at 60 fps.
- Shows an **Audio Features Debug / Audio Lab** panel on the main `/vj` page.
- Has no lighting or DMX integration yet.

We now want to:

- Connect a **WebUSB DMX512 controller**.
- Control a real fixture: **Fun Generation SePar Quad LED RGB UV** in 6-channel DMX mode.
- Drive this fixture from the visual output of vj0 by sampling pixels from the main canvas.

To do this, we need:

- A **DMX universe** (512 channels).
- A **fixture layer** that understands which channels do what for a given fixture type.
- A **lighting engine** that samples the canvas and maps visual data to fixtures, filling the universe.
- A **DMX output layer** that sends the universe via WebUSB.
- A **fixture inspector** in the UI to verify per-channel output and tweak DMX addresses.

---

## Goal of this story

Add a **lighting system** that:

- Defines fixtures via profiles and instances (including the Fun Generation SePar Quad LED RGB UV in 6ch mode).
- Samples color data from the main vj0 canvas.
- Maps those colors into the correct DMX channels for one or more fixtures.
- Sends DMX frames to a WebUSB DMX512 controller at a controlled tick rate.
- Provides a UI to:
  - Connect / disconnect DMX.
  - Inspect fixture channels live (per-channel bars + numeric values).
  - Change DMX start address of fixtures at runtime.

The lighting system must be decoupled from the 60 fps visual render loop and must not block rendering.

---

## User story

> As a VJ using vj0 with my Fun Generation SePar Quad LED RGB UV and WebUSB DMX controller  
> I want vj0 to sample colors from my visuals and map them to my fixture’s DMX channels  
> So that my light reacts in sync with the visuals in real time and I can verify the output visually in the UI

---

## Architectural requirements

### 1. DMX & lighting types

Create `src/lib/lighting/types.ts` with core lighting types:

```ts
export type DmxUniverse = Uint8Array; // length 512

export type LightingFrame = {
  universe: DmxUniverse; // current DMX channel values
};

export type LightingConfig = {
  tickHz: number; // how often to update lighting per second, e.g. 20–30
};
```

These types represent the low-level lighting bus independent of DMX output or fixtures.

### 2. Fixture profiles & fixture instances

#### 2.1 FixtureProfile

In the same file or a separate `fixtures-types.ts` (up to you), define:

```ts
export type FixtureChannelKind =
  | "red"
  | "green"
  | "blue"
  | "uv"
  | "dimmer"
  | "strobe"
  | "program"
  | "programSpeed";

export interface FixtureProfile {
  id: string;             // e.g. "fun-gen-separ-quad-rgbuv-6ch"
  name: string;           // Human readable name
  mode: string;           // e.g. "6ch"
  channels: FixtureChannelKind[]; // index 0..N-1 maps to DMX channel offset
}
```

Add a profile for **Fun Generation SePar Quad LED RGB UV – 6-channel mode** in `src/lib/lighting/fixtures/fun-gen-separ-quad.ts`:

```ts
import type { FixtureProfile } from "../types";

export const SeParQuadRGBUV_6CH: FixtureProfile = {
  id: "fun-gen-separ-quad-rgbuv-6ch",
  name: "Fun Generation SePar Quad LED RGB UV (6ch)",
  mode: "6ch",
  channels: [
    "red",     // ch1
    "green",   // ch2
    "blue",    // ch3
    "uv",      // ch4
    "dimmer",  // ch5
    "strobe",  // ch6
  ],
};
```

#### 2.2 FixtureInstance

Describe concrete fixtures in the rig:

```ts
export interface FixtureInstance {
  id: string;               // e.g. "separ-1"
  profile: FixtureProfile;
  /**
   * DMX start channel, 1..512, with the constraint:
   * address + profile.channels.length - 1 <= 512
   * This value must be editable at runtime via the UI.
   */
  address: number;
  mapping: {
    x: number; // 0..1 normalized canvas X
    y: number; // 0..1 normalized canvas Y
  };
}
```

Create `src/lib/lighting/fixtures/index.ts` with a first setup that defines one fixture instance using the profile above:

```ts
import type { FixtureInstance } from "../types";
import { SeParQuadRGBUV_6CH } from "./fun-gen-separ-quad";

export const FIXTURES: FixtureInstance[] = [
  {
    id: "separ-1",
    profile: SeParQuadRGBUV_6CH,
    address: 1,                  // default DMX address (editable in UI)
    mapping: { x: 0.5, y: 0.5 }, // sample center of canvas for now
  },
];
```

For this story it is enough to support a single fixture, but the implementation should be written with arrays in mind so adding more later is easy.

### 3. LightingEngine (sampling + filling universe)

Create `src/lib/lighting/lighting-engine.ts`.

Responsibilities:

- Own:
  - A `LightingConfig` (at least `tickHz`).
  - A reference to the **main visual canvas** (the one used by the scenes).
  - A shared `DmxUniverse` buffer (`Uint8Array(512)`).
  - The array of `FixtureInstance` (`FIXTURES`).
- Expose methods to:
  - Start and stop a lighting update loop.
  - Subscribe / unsubscribe to lighting frames.

Suggested public API:

```ts
import type {
  FixtureInstance,
  LightingConfig,
  LightingFrame,
  DmxUniverse,
} from "./types";

export class LightingEngine {
  constructor(
    canvas: HTMLCanvasElement,
    fixtures: FixtureInstance[],
    config: LightingConfig
  );

  start(): void;
  stop(): void;

  onFrame(callback: (frame: LightingFrame) => void): void;
  offFrame(callback: (frame: LightingFrame) => void): void;

  getUniverse(): DmxUniverse;

  updateFixtureAddress(id: string, address: number): void;
}
```

Implementation details:

- Use a timer driven by `tickHz` (e.g. `setInterval` or `requestAnimationFrame` + internal time accumulator).
- On each tick:
  - Clear or reuse the `DmxUniverse` buffer.
  - For each `FixtureInstance`:
    - Convert `mapping.x` / `mapping.y` (0..1) to actual canvas coordinates.
    - Read a 1×1 or small region via `ctx.getImageData(x, y, 1, 1)` using the canvas’ 2D context.
    - Convert the pixel(s) to `{ r, g, b, uv }`:
      - For now, UV can be derived from brightness (e.g. average of r/g/b) or be set to 0. Implementation can start with `uv = 0` and be improved later.
    - Apply those values to the universe via a helper (see next section).
  - After processing all fixtures, emit a `LightingFrame`:
    ```ts
    const frame: LightingFrame = { universe };
    callbacks.forEach(cb => cb(frame));
    ```

### 4. Mapping fixture profiles to the universe

In `lighting-engine.ts` (or a helper module), implement a function to apply a fixture’s color data to the DMX universe:

```ts
function applyFixtureToUniverse(
  fixture: FixtureInstance,
  color: { r: number; g: number; b: number; uv: number },
  universe: Uint8Array
) {
  const base = fixture.address - 1; // convert 1-based DMX to 0-based index

  fixture.profile.channels.forEach((kind, i) => {
    const chIndex = base + i;
    switch (kind) {
      case "red":
        universe[chIndex] = color.r;
        break;
      case "green":
        universe[chIndex] = color.g;
        break;
      case "blue":
        universe[chIndex] = color.b;
        break;
      case "uv":
        universe[chIndex] = color.uv;
        break;
      case "dimmer":
        universe[chIndex] = 255; // full brightness for now
        break;
      case "strobe":
        universe[chIndex] = 0;   // no strobe for now
        break;
      default:
        // Ignore program/programSpeed for this story
        break;
    }
  });
}
```

Notes:

- For now, we assume a single fixture and no channel overlap. Validation of address ranges should be simple (e.g. clamping / rejecting invalid addresses).
- Later stories can add validation and complex rig layout; for now, keep it straightforward.

### 5. Runtime update of fixture address

The `LightingEngine.updateFixtureAddress(id: string, address: number)` method must:

- Find the corresponding `FixtureInstance` by id.
- Validate and clamp the new address so that `address + profile.channels.length - 1 <= 512`.
- Update the instance’s `address` field.
- The new address must be used in the next lighting tick for universe mapping.
- No restart of the engine should be required.

This allows the UI to provide a numeric input to change DMX start address while vj0 is running.

### 6. DMX output (WebUSB layer)

Create `src/lib/lighting/dmx-output.ts`.

Responsibilities:

- Manage WebUSB connection to the DMX512 controller.
- Send the current `DmxUniverse` whenever a new `LightingFrame` is produced.

Suggested API:

```ts
import type { DmxUniverse } from "./types";

export class DmxOutput {
  async connect(): Promise<void>;   // opens WebUSB picker, initializes device
  async disconnect(): Promise<void>;
  isConnected(): boolean;

  async sendUniverse(universe: DmxUniverse): Promise<void>;
}
```

Integration with `LightingEngine` (in the React layer):

- Subscribe once `LightingEngine` is started:

```ts
lightingEngine.onFrame((frame) => {
  if (dmxOutput.isConnected()) {
    dmxOutput.sendUniverse(frame.universe);
  }
});
```

Implementation notes:

- Gracefully handle browsers without WebUSB support (e.g. feature detection and fallback message).
- If the user cancels the device picker, handle the rejection cleanly and keep state as `DISCONNECTED`.
- On disconnect, close the device. Optionally send an “all zeros” universe once to turn fixtures off.

### 7. Separation of concerns

- `VisualEngine`:
  - Owns the main canvas and renders scenes at 60 fps.
- `LightingEngine`:
  - Reads from the main canvas at a lower rate (e.g. 20–30 Hz).
  - Builds the DMX universe from fixture instances + profiles.
- `DmxOutput`:
  - Only sends universe data to hardware when connected.
- React components:
  - Orchestrate creation/destruction of engines.
  - Provide UI for connection and inspection.

---

## UI requirements

All UI changes happen on the main vj page (e.g. `app/vj/page.tsx`), where audio, scenes, and debug panels already live.

### 1. DMX connection controls

Add a **Lighting / DMX** section with:

- A **Connect DMX** button:
  - Calls `dmxOutput.connect()`.
- A **Disconnect** button:
  - Calls `dmxOutput.disconnect()`.
- A status label showing:
  - `DMX: DISCONNECTED`
  - `DMX: CONNECTED` (optional: display basic device info if available).
- If WebUSB is not supported, show a clear message and disable the connect button.

### 2. Fixture inspector & channel monitor

Add a **Fixture Inspector** panel for debugging and verification.

For each `FixtureInstance`:

- Show:
  - **Fixture name** (`fixture.profile.name`).
  - **Fixture id** (optional).
- DMX address input:
  - A numeric `<input>` for `fixture.address` with validation:
    - Minimum: 1.
    - Maximum: `512 - fixture.profile.channels.length + 1`.
  - On change:
    - Call into `LightingEngine.updateFixtureAddress(fixture.id, newAddress)`.
    - The change must affect both the preview and DMX output starting with the next lighting tick.

- Per-channel monitor:
  - For each channel in `fixture.profile.channels` (e.g. R, G, B, UV, Dimmer, Strobe):
    - Derive its **absolute channel index** in the universe:
      - `absIndex = fixture.address - 1 + localChannelIndex`.
    - Read its current value from `LightingFrame.universe[absIndex]`.
    - Display:
      - Kind label (e.g. `R`, `G`, `B`, `UV`, `DIM`, `STR`).
      - A small horizontal bar showing value 0–255 as 0–100% width.
      - Numeric value (0–255).

Implementation notes:

- The inspector should subscribe to `LightingEngine.onFrame`.
- To avoid excessive React re-renders, you may:
  - Update a `useRef` on every frame and only propagate to state at a throttled rate (e.g. 10–20 fps), or
  - Keep the inspector simple enough that state updates at `tickHz` are acceptable.
- The goal is to visually verify that:
  - Channels are mapped where expected.
  - Changing the fixture address moves the activity to different DMX channels.

### 3. Simple fixture color preview

In addition to per-channel bars, add a small color preview for each fixture:

- A colored square whose background color is based on the current RGB values for that fixture (from the universe).
- This gives a quick at-a-glance impression of what the fixture *should* look like, even without real hardware.

---

## Non-functional requirements

- **No interference with 60 fps visuals**:
  - Lighting updates must not run on the same RAF loop as visual rendering.
  - Use `setInterval` or a separate timing mechanism based on `tickHz`.

- **Graceful fallback**:
  - If WebUSB is not available, the lighting engine and fixture inspector should still work (for preview only).
  - DMX output simply does nothing; UI should make this clear.

- **Minimal allocations**:
  - Reuse the `DmxUniverse` buffer.
  - Avoid creating new arrays or objects inside the per-tick loop; store per-fixture temporary data on the instance or engine where reasonable.

- **Safe shutdown**:
  - On page unload or teardown of the `/vj` view:
    - Stop the lighting engine.
    - Unsubscribe all `onFrame` listeners.
    - Disconnect DMX if connected, closing the WebUSB device.
    - Optionally send an “all zeros” universe once on disconnect to turn fixtures off cleanly.

- **Extensibility**:
  - It should be easy to add more fixtures later by:
    - Creating new `FixtureProfile`s.
    - Adding `FixtureInstance`s to the array.
  - No changes in `DmxOutput` should be required to support more fixtures; everything is driven by the universe buffer.

---

## Out of scope

The following are explicitly **not** part of this story:

- Multiple complex fixtures or full-rig configuration UI (we only need one fixture instance defined plus the ability to edit its address).
- Multiple DMX universes, Art-Net, sACN.
- Persisting fixture addresses or layouts to local storage or a database.
- Complex mapping (e.g. 2D matrices, multiple sample points per fixture, effects).
- Integration of lighting state with AI or remote video streaming.

---

## Deliverables

- `src/lib/lighting/types.ts` with:
  - `DmxUniverse`, `LightingFrame`, `LightingConfig`,
  - `FixtureChannelKind`, `FixtureProfile`, `FixtureInstance`.
- `src/lib/lighting/fixtures/fun-gen-separ-quad.ts`:
  - Contains `SeParQuadRGBUV_6CH` profile for the Fun Generation SePar Quad LED RGB UV in 6ch mode.
- `src/lib/lighting/fixtures/index.ts`:
  - Exports `FIXTURES` array with at least one `FixtureInstance` (`separ-1`) at a default address (1) and a mapping pointing to the center of the canvas.
- `src/lib/lighting/lighting-engine.ts`:
  - Samples the main canvas at `tickHz`.
  - Computes per-fixture color and fills the `DmxUniverse` using fixture profiles and addresses.
  - Emits `LightingFrame`s to subscribers.
  - Implements `updateFixtureAddress(id, address)`.
- `src/lib/lighting/dmx-output.ts`:
  - Manages WebUSB DMX device connection.
  - Sends the universe on each lighting frame when connected.
- Updated `app/vj/page.tsx` (or equivalent main vj page):
  - Instantiates `LightingEngine` once the main canvas and scenes are set up.
  - Hooks `LightingEngine` frames to `DmxOutput`.
  - Adds a **Lighting / DMX** section with:
    - DMX connect/disconnect controls and status label.
    - Fixture Inspector with:
      - Name, editable address, per-channel bars and numeric values.
      - Small RGB color preview per fixture.

The final result should allow running vj0 locally, connecting the WebUSB DMX controller, changing the DMX address in the UI, seeing per-channel activity in the inspector, and driving at least one Fun Generation SePar Quad LED RGB UV fixture in 6ch mode from the main visual canvas output.
