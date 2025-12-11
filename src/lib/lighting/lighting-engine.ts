/**
 * LightingEngine - Samples canvas pixels and builds DMX universe from fixture data
 *
 * Runs on a separate timer from the visual render loop (tickHz, typically 20-30 Hz)
 * to avoid blocking the 60fps visual rendering.
 */

import type { AudioEngine } from "../audio-engine";
import type { AudioFeatures } from "../audio-features";
import type {
  FixtureInstance,
  LightingConfig,
  LightingFrame,
  DmxUniverse,
  StrobeMode,
  ColorMode,
} from "./types";

type FrameCallback = (frame: LightingFrame) => void;

export class LightingEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D | null = null;
  private audioEngine: AudioEngine | null = null;
  private fixtures: FixtureInstance[];
  private config: LightingConfig;

  private universe: DmxUniverse;
  private intervalId: number | null = null;
  private callbacks: Set<FrameCallback> = new Set();

  // Reusable frame object to avoid allocations
  private frame: LightingFrame;

  constructor(
    canvas: HTMLCanvasElement,
    fixtures: FixtureInstance[],
    config: LightingConfig
  ) {
    this.canvas = canvas;
    this.fixtures = [...fixtures]; // Copy to allow mutation
    this.config = config;
    this.universe = new Uint8Array(512);
    this.frame = { universe: this.universe };
  }

  /**
   * Set the audio engine for audio-reactive effects
   */
  setAudioEngine(audioEngine: AudioEngine): void {
    this.audioEngine = audioEngine;
  }

  /**
   * Start the lighting update loop
   */
  start(): void {
    if (this.intervalId !== null) return;

    // Get 2D context for pixel sampling
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    if (!this.ctx) {
      console.error("LightingEngine: Failed to get 2D context");
      return;
    }

    const intervalMs = 1000 / this.config.tickHz;

    this.intervalId = window.setInterval(() => {
      this.tick();
    }, intervalMs);
  }

  /**
   * Stop the lighting update loop
   */
  stop(): void {
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  /**
   * Subscribe to lighting frames
   */
  onFrame(callback: FrameCallback): void {
    this.callbacks.add(callback);
  }

  /**
   * Unsubscribe from lighting frames
   */
  offFrame(callback: FrameCallback): void {
    this.callbacks.delete(callback);
  }

  /**
   * Get the current DMX universe buffer
   */
  getUniverse(): DmxUniverse {
    return this.universe;
  }

  /**
   * Get all fixtures
   */
  getFixtures(): FixtureInstance[] {
    return this.fixtures;
  }

  /**
   * Update a fixture's DMX start address
   */
  updateFixtureAddress(id: string, address: number): void {
    const fixture = this.fixtures.find((f) => f.id === id);
    if (!fixture) return;

    // Validate address range
    const maxAddress = 512 - fixture.profile.channels.length + 1;
    const clampedAddress = Math.max(1, Math.min(address, maxAddress));

    fixture.address = clampedAddress;
  }

  /**
   * Update a fixture's strobe mode
   */
  updateFixtureStrobeMode(id: string, mode: StrobeMode): void {
    const fixture = this.fixtures.find((f) => f.id === id);
    if (fixture) {
      fixture.strobeMode = mode;
    }
  }

  /**
   * Update a fixture's strobe threshold
   */
  updateFixtureStrobeThreshold(id: string, threshold: number): void {
    const fixture = this.fixtures.find((f) => f.id === id);
    if (fixture) {
      fixture.strobeThreshold = Math.max(0, Math.min(1, threshold));
    }
  }

  /**
   * Update a fixture's strobe max value
   */
  updateFixtureStrobeMax(id: string, max: number): void {
    const fixture = this.fixtures.find((f) => f.id === id);
    if (fixture) {
      fixture.strobeMax = Math.max(0, Math.min(255, max));
    }
  }

  /**
   * Update a fixture's color mode
   */
  updateFixtureColorMode(id: string, mode: ColorMode): void {
    const fixture = this.fixtures.find((f) => f.id === id);
    if (fixture) {
      fixture.colorMode = mode;
    }
  }

  /**
   * Update a fixture's solid color
   */
  updateFixtureSolidColor(
    id: string,
    color: { r: number; g: number; b: number }
  ): void {
    const fixture = this.fixtures.find((f) => f.id === id);
    if (fixture) {
      fixture.solidColor = {
        r: Math.max(0, Math.min(255, color.r)),
        g: Math.max(0, Math.min(255, color.g)),
        b: Math.max(0, Math.min(255, color.b)),
      };
    }
  }

  /**
   * Main tick: sample canvas, fill universe, emit frame
   */
  private tick(): void {
    if (!this.ctx) return;

    const canvasWidth = this.canvas.width;
    const canvasHeight = this.canvas.height;

    // Get audio features for audio-reactive effects
    const features = this.audioEngine?.getLatestFeatures() ?? null;

    // Process each fixture
    for (const fixture of this.fixtures) {
      let r: number, g: number, b: number;

      if (fixture.colorMode === "solid") {
        // Use solid color
        r = fixture.solidColor.r;
        g = fixture.solidColor.g;
        b = fixture.solidColor.b;
      } else {
        // Sample from canvas
        const x = Math.floor(fixture.mapping.x * canvasWidth);
        const y = Math.floor(fixture.mapping.y * canvasHeight);
        const imageData = this.ctx.getImageData(x, y, 1, 1);
        [r, g, b] = imageData.data;
      }

      // Apply fixture color to universe
      this.applyFixtureToUniverse(fixture, { r, g, b, uv: 0 }, features);
    }

    // Emit frame to all subscribers
    for (const callback of this.callbacks) {
      callback(this.frame);
    }
  }

  /**
   * Apply a fixture's color data to the DMX universe
   */
  private applyFixtureToUniverse(
    fixture: FixtureInstance,
    color: { r: number; g: number; b: number; uv: number },
    features: AudioFeatures | null
  ): void {
    const base = fixture.address - 1; // Convert 1-based DMX to 0-based index

    // Calculate strobe value from audio features
    const strobeValue = this.calculateStrobeValue(fixture, features);

    // When strobe is active and value is 0, also set dimmer to 0 (fixture off)
    const strobeActive = fixture.strobeMode !== "off";
    const dimmerValue = strobeActive && strobeValue === 0 ? 0 : 255;

    for (let i = 0; i < fixture.profile.channels.length; i++) {
      const kind = fixture.profile.channels[i];
      const chIndex = base + i;

      if (chIndex >= 512) continue; // Safety check

      switch (kind) {
        case "red":
          this.universe[chIndex] = color.r;
          break;
        case "green":
          this.universe[chIndex] = color.g;
          break;
        case "blue":
          this.universe[chIndex] = color.b;
          break;
        case "uv":
          this.universe[chIndex] = color.uv;
          break;
        case "dimmer":
          this.universe[chIndex] = dimmerValue;
          break;
        case "strobe":
          this.universe[chIndex] = strobeValue;
          break;
        default:
          // Ignore program/programSpeed for now
          break;
      }
    }
  }

  /**
   * Calculate strobe value based on fixture's strobe mode and audio features.
   * Uses the fixture profile's strobeRange if defined.
   */
  private calculateStrobeValue(
    fixture: FixtureInstance,
    features: AudioFeatures | null
  ): number {
    // Get strobe range from profile, with defaults
    const strobeRange = fixture.profile.strobeRange ?? {
      off: 0,
      min: 0,
      max: 255,
    };

    if (fixture.strobeMode === "off" || !features) {
      return strobeRange.off;
    }

    // Get the feature value based on strobe mode
    let featureValue: number;
    switch (fixture.strobeMode) {
      case "energyLow":
        featureValue = features.energyLow;
        break;
      case "energyMid":
        featureValue = features.energyMid;
        break;
      case "energyHigh":
        featureValue = features.energyHigh;
        break;
      case "peak":
        featureValue = features.peak;
        break;
      case "rms":
        featureValue = features.rms;
        break;
      default:
        return strobeRange.off;
    }

    // Apply threshold - only strobe when feature exceeds threshold
    if (featureValue < fixture.strobeThreshold) {
      return strobeRange.off;
    }

    // Map feature value to strobe range [min, max]
    // Normalize: (value - threshold) / (1 - threshold) gives 0-1 range above threshold
    const normalized =
      (featureValue - fixture.strobeThreshold) / (1 - fixture.strobeThreshold);

    // Scale to the fixture's strobe range
    // Also apply the user's strobeMax as a percentage limiter within the range
    const rangeSize = strobeRange.max - strobeRange.min;
    const userMaxPercent = fixture.strobeMax / 255; // User's strobeMax as 0-1
    const effectiveMax = strobeRange.min + rangeSize * userMaxPercent;

    return Math.floor(
      strobeRange.min + normalized * (effectiveMax - strobeRange.min)
    );
  }
}
