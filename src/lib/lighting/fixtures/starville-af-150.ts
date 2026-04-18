import type { FixtureProfile } from "../types";

/**
 * Starville AF-150 — 800 W compact fog machine.
 *
 * Per the user manual (Thomann c_305148_v7):
 *   - Exactly 1 DMX channel (fixed, no mode switch)
 *   - Ch1 is a proportional 0-255 intensity dimmer: 0 = no fog, 255 = 100%
 *   - No deadband, no timer/interval channels — any burst behavior has to
 *     be generated in software (we do this via LightingEngine strobe modes
 *     and/or triggerFogBurst())
 *   - DMX address is set via 9-bit DIP switches, max address 511 (SW10 unused)
 *   - 4-5 minute warm-up from cold power-on; no DMX feedback for ready state
 *
 * Software-side driving:
 *   - A manual fog burst (hotkey "0" / FOG button) overrides this channel
 *     for a configurable duration + intensity via LightingEngine's
 *     fogBurstUntil / fogBurstIntensity state.
 *   - When no burst is active, the strobe pipeline drives the channel:
 *       * strobeMode "off"  → 0 (no fog)
 *       * strobeMode "peak" → fog bursts when peaks exceed threshold
 *       * strobeMode "rms" threshold 0 → continuous ambient fog
 */
export const StarvilleAF150_1CH: FixtureProfile = {
  id: "starville-af-150-1ch",
  name: "Starville AF-150 fog (1ch)",
  mode: "1ch",
  channels: ["fog"],
};
