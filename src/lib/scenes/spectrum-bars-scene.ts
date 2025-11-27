/**
 * SpectrumBarsScene - Frequency band visualization
 *
 * Renders 3 vertical bars representing low, mid, and high frequency energy.
 * Bar heights react to audio features with smooth interpolation.
 *
 * Extension points:
 * - Add more frequency bands by subdividing the spectrum
 * - Add peak hold indicators
 * - Add mirrored/symmetric layouts
 */

import type { AudioFeatures } from '../audio-features';
import type { VjScene } from './types';

export class SpectrumBarsScene implements VjScene {
  readonly id = 'spectrum-bars';
  readonly name = 'Spectrum Bars';

  // Canvas dimensions
  private width = 1024;
  private height = 256;

  // Bar configuration
  private readonly barCount = 3;
  private readonly barGap = 20;
  private readonly barColors = ['#ff4444', '#ffaa00', '#00ccff']; // Low, Mid, High

  // Smoothed values for interpolation (avoids jitter)
  private smoothedValues = [0, 0, 0];
  private readonly smoothingFactor = 0.15; // Lower = smoother

  // Style
  private readonly backgroundColor = '#0a0a0a';

  init(canvas: HTMLCanvasElement): void {
    this.width = canvas.width;
    this.height = canvas.height;
    this.smoothedValues = [0, 0, 0];
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  render(
    ctx: CanvasRenderingContext2D,
    features: AudioFeatures | null,
    _timeDomain: Float32Array | null,
    _dt: number
  ): void {
    const { width, height, backgroundColor, barCount, barGap, barColors } = this;

    // Clear canvas
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Get target values from features
    const targets = features
      ? [features.energyLow, features.energyMid, features.energyHigh]
      : [0, 0, 0];

    // Calculate bar dimensions
    const totalGaps = (barCount + 1) * barGap;
    const barWidth = (width - totalGaps) / barCount;
    const maxBarHeight = height * 0.85;

    // Render bars
    for (let i = 0; i < barCount; i++) {
      // Smooth interpolation
      this.smoothedValues[i] +=
        (targets[i] - this.smoothedValues[i]) * this.smoothingFactor;

      const barHeight = this.smoothedValues[i] * maxBarHeight;
      const x = barGap + i * (barWidth + barGap);
      const y = height - barHeight;

      // Draw bar
      ctx.fillStyle = barColors[i];
      ctx.fillRect(x, y, barWidth, barHeight);

      // Add subtle gradient overlay for depth
      const gradient = ctx.createLinearGradient(x, y, x, height);
      gradient.addColorStop(0, 'rgba(255, 255, 255, 0.2)');
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0.3)');
      ctx.fillStyle = gradient;
      ctx.fillRect(x, y, barWidth, barHeight);
    }
  }

  destroy(): void {
    this.smoothedValues = [0, 0, 0];
  }
}

