/**
 * WaveformScene - Audio waveform visualization
 *
 * Renders time-domain audio data as a horizontal waveform line.
 * Line thickness modulated by RMS for audio-reactive feedback.
 *
 * Extension points:
 * - Add color modulation based on spectral centroid
 * - Add multiple waveform layers with different colors
 * - Add glow/bloom effect using shadow blur
 */

import type { AudioFeatures } from '../audio-features';
import type { VjScene } from './types';

export class WaveformScene implements VjScene {
  readonly id = 'waveform';
  readonly name = 'Waveform';

  // Canvas dimensions (set on init/resize)
  private width = 1024;
  private height = 256;
  private centerY = 128;

  // Style configuration
  private readonly strokeColor = '#00ff88';
  private readonly backgroundColor = '#0a0a0a';
  private readonly baseLineWidth = 2;
  private readonly maxLineWidth = 6;

  init(canvas: HTMLCanvasElement): void {
    this.width = canvas.width;
    this.height = canvas.height;
    this.centerY = this.height / 2;
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.centerY = height / 2;
  }

  render(
    ctx: CanvasRenderingContext2D,
    features: AudioFeatures | null,
    timeDomain: Float32Array | null,
    _dt: number
  ): void {
    const { width, height, centerY, backgroundColor, strokeColor } = this;

    // Clear canvas
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Nothing to draw without audio data
    if (!timeDomain) return;

    // Modulate line width based on RMS (louder = thicker)
    const rms = features?.rms ?? 0;
    const lineWidth = this.baseLineWidth + rms * (this.maxLineWidth - this.baseLineWidth);

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();

    const bufferLength = timeDomain.length;
    const sliceWidth = width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      // Map audio sample [-1, 1] to canvas Y coordinate
      const y = centerY + timeDomain[i] * centerY * 0.8;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    ctx.stroke();
  }

  destroy(): void {
    // No resources to clean up
  }
}

