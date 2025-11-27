import type { VjScene } from "./types";
import type { AudioFeatures } from "../audio-features";

/**
 * Horizontal lines stacked vertically, each displaced by timeDomain slice.
 * Creates a 3D terrain/mountain visualization effect.
 */
export class TerrainLinesScene implements VjScene {
  readonly id = "terrain-lines";
  readonly name = "Terrain Lines";

  private width = 0;
  private height = 0;

  private readonly lineCount = 32;
  private lineOffset = 0;

  init(): void {
    this.lineOffset = 0;
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  render(
    ctx: CanvasRenderingContext2D,
    features: AudioFeatures | null,
    timeDomain: Float32Array | null,
    dt: number
  ): void {
    if (!this.width || !this.height || !timeDomain) return;

    const rms = features?.rms ?? 0;
    const peak = features?.peak ?? 0;
    const energyLow = features?.energyLow ?? 0;
    const energyMid = features?.energyMid ?? 0;
    const energyHigh = features?.energyHigh ?? 0;
    const spectralCentroid = features?.spectralCentroid ?? 0.5;

    // Scroll speed based on energy
    this.lineOffset += dt * 0.02 * (0.5 + energyMid);
    if (this.lineOffset > 1) this.lineOffset -= 1;

    const lineSpacing = this.height / this.lineCount;
    const bufferLength = timeDomain.length;
    const samplesPerLine = Math.floor(bufferLength / this.lineCount);

    // Base hue shifts with spectral centroid
    const baseHue = 180 + spectralCentroid * 120; // Cyan to magenta range

    // Draw lines from back (top) to front (bottom)
    for (let lineIdx = 0; lineIdx < this.lineCount; lineIdx++) {
      const depth = lineIdx / this.lineCount; // 0 = back, 1 = front
      const y = lineIdx * lineSpacing + this.lineOffset * lineSpacing;

      // Calculate amplitude from corresponding section of timeDomain
      const sampleStart = lineIdx * samplesPerLine;
      const sampleEnd = Math.min(sampleStart + samplesPerLine, bufferLength);

      // Perspective scale - front lines are bigger
      const perspectiveScale = 0.3 + depth * 0.7;
      const amplitudeScale = 30 + depth * 50 + rms * 40;

      // Line styling - front lines are brighter and thicker
      const alpha = 0.3 + depth * 0.7;
      const hue = (baseHue + depth * 30) % 360;
      const lightness = 30 + depth * 30 + energyHigh * 20;
      ctx.strokeStyle = `hsla(${hue}, 70%, ${lightness}%, ${alpha})`;
      ctx.lineWidth = 1 + depth * 2 + peak * 2;

      ctx.beginPath();

      const pointCount = 64;
      const xStep = this.width / (pointCount - 1);

      for (let i = 0; i < pointCount; i++) {
        const x = i * xStep;

        // Sample from timeDomain with interpolation
        const sampleIndex = sampleStart + (i / pointCount) * (sampleEnd - sampleStart);
        const sampleIndexFloor = Math.floor(sampleIndex);
        const sampleIndexCeil = Math.min(sampleIndexFloor + 1, bufferLength - 1);
        const t = sampleIndex - sampleIndexFloor;

        const sample =
          timeDomain[sampleIndexFloor] * (1 - t) + timeDomain[sampleIndexCeil] * t;

        // Displacement goes upward (negative Y)
        const displacement = sample * amplitudeScale * perspectiveScale;
        const yPos = y - Math.abs(displacement) * (0.5 + energyLow * 0.5);

        if (i === 0) {
          ctx.moveTo(x, yPos);
        } else {
          ctx.lineTo(x, yPos);
        }
      }

      ctx.stroke();
    }
  }

  destroy(): void {
    // No resources to clean up
  }
}

