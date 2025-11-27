import type { VjScene } from "./types";
import type { AudioFeatures } from "../audio-features";

/**
 * Concentric circles expanding from center.
 * Size/color modulated by RMS and energy bands.
 */
export class RadialPulseScene implements VjScene {
  readonly id = "radial-pulse";
  readonly name = "Radial Pulse";

  private width = 0;
  private height = 0;
  private centerX = 0;
  private centerY = 0;

  // Ring state - reused each frame
  private readonly maxRings = 8;
  private ringPhases: number[] = [];

  init(): void {
    // Initialize ring phases with staggered offsets
    for (let i = 0; i < this.maxRings; i++) {
      this.ringPhases[i] = (i / this.maxRings) * Math.PI * 2;
    }
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.centerX = width / 2;
    this.centerY = height / 2;
  }

  render(
    ctx: CanvasRenderingContext2D,
    features: AudioFeatures | null,
    timeDomain: Float32Array | null,
    dt: number
  ): void {
    if (!this.width || !this.height) return;

    const rms = features?.rms ?? 0;
    const peak = features?.peak ?? 0;
    const energyLow = features?.energyLow ?? 0;
    const energyMid = features?.energyMid ?? 0;
    const energyHigh = features?.energyHigh ?? 0;

    // Update phases based on time
    const timeScale = dt * 0.002;
    for (let i = 0; i < this.maxRings; i++) {
      this.ringPhases[i] += timeScale * (1 + energyMid * 2);
      if (this.ringPhases[i] > Math.PI * 2) {
        this.ringPhases[i] -= Math.PI * 2;
      }
    }

    const maxRadius = Math.min(this.width, this.height) * 0.45;

    // Draw rings from back to front
    for (let i = this.maxRings - 1; i >= 0; i--) {
      const phase = this.ringPhases[i];
      const normalizedPhase = (Math.sin(phase) + 1) / 2; // 0 to 1

      // Radius grows with phase, modulated by RMS
      const baseRadius = normalizedPhase * maxRadius;
      const radius = baseRadius * (0.5 + rms * 0.5 + peak * 0.3);

      if (radius < 5) continue;

      // Color based on energy bands - cycle through hues
      const hueBase = i * 45; // Spread hues across rings
      const hue = (hueBase + energyLow * 60) % 360;
      const saturation = 70 + energyHigh * 30;
      const lightness = 40 + energyMid * 30;

      // Line width pulses with energy
      const lineWidth = 2 + energyLow * 8 + (i === 0 ? peak * 10 : 0);

      // Alpha fades as ring expands
      const alpha = 1 - normalizedPhase * 0.7;

      ctx.beginPath();
      ctx.arc(this.centerX, this.centerY, radius, 0, Math.PI * 2);
      ctx.strokeStyle = `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }

    // Center circle - pulses with peak
    const centerRadius = 10 + peak * 30;
    ctx.beginPath();
    ctx.arc(this.centerX, this.centerY, centerRadius, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${energyHigh * 60}, 100%, ${50 + rms * 30}%)`;
    ctx.fill();
  }

  destroy(): void {
    // No resources to clean up
  }
}

