import type { VjScene } from "./types";
import type { AudioFeatures } from "../audio-features";

interface Star {
  x: number;
  y: number;
  z: number;
  prevX: number;
  prevY: number;
}

/**
 * Stars flying toward/away from viewer.
 * Speed tied to peak, star density tied to RMS.
 */
export class StarfieldScene implements VjScene {
  readonly id = "starfield";
  readonly name = "Starfield";

  private width = 0;
  private height = 0;
  private centerX = 0;
  private centerY = 0;

  // Pre-allocated star pool
  private readonly maxStars = 300;
  private stars: Star[] = [];

  init(): void {
    // Pre-allocate star objects
    this.stars = [];
    for (let i = 0; i < this.maxStars; i++) {
      this.stars[i] = {
        x: 0,
        y: 0,
        z: 0,
        prevX: 0,
        prevY: 0,
      };
    }
    this.resetStars();
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.centerX = width / 2;
    this.centerY = height / 2;
    this.resetStars();
  }

  private resetStars(): void {
    for (let i = 0; i < this.maxStars; i++) {
      this.resetStar(this.stars[i], true);
    }
  }

  private resetStar(star: Star, randomZ: boolean): void {
    // Random position in 3D space centered around origin
    const spread = Math.max(this.width, this.height);
    star.x = (Math.random() - 0.5) * spread;
    star.y = (Math.random() - 0.5) * spread;
    star.z = randomZ ? Math.random() * 1000 : 1000;
    star.prevX = this.centerX;
    star.prevY = this.centerY;
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
    const spectralCentroid = features?.spectralCentroid ?? 0.5;

    // Speed increases with peak and RMS
    const baseSpeed = 0.3 + rms * 0.5 + peak * 1.5;
    const speed = baseSpeed * dt;

    // Color based on spectral centroid
    const baseHue = spectralCentroid * 240; // Blue to red range

    // Draw stars
    const activeStars = Math.floor(100 + rms * (this.maxStars - 100));

    for (let i = 0; i < activeStars; i++) {
      const star = this.stars[i];

      // Store previous screen position for trail
      const prevZ = star.z;
      star.prevX = this.centerX + (star.x / prevZ) * 500;
      star.prevY = this.centerY + (star.y / prevZ) * 500;

      // Move star toward viewer
      star.z -= speed;

      // Reset star if it passed the camera
      if (star.z <= 1) {
        this.resetStar(star, false);
        continue;
      }

      // Project to screen
      const screenX = this.centerX + (star.x / star.z) * 500;
      const screenY = this.centerY + (star.y / star.z) * 500;

      // Skip if off screen
      if (
        screenX < -10 ||
        screenX > this.width + 10 ||
        screenY < -10 ||
        screenY > this.height + 10
      ) {
        this.resetStar(star, false);
        continue;
      }

      // Size and brightness based on depth
      const depthFactor = 1 - star.z / 1000;
      const size = 1 + depthFactor * 3 + peak * 2;
      const brightness = 30 + depthFactor * 70;

      // Hue shifts slightly per star based on position
      const hue = (baseHue + (star.x + star.y) * 0.1) % 360;

      // Draw trail (line from previous to current position)
      const trailLength = depthFactor * (0.5 + energyMid);
      if (trailLength > 0.1) {
        ctx.beginPath();
        ctx.moveTo(star.prevX, star.prevY);
        ctx.lineTo(screenX, screenY);
        ctx.strokeStyle = `hsla(${hue}, 60%, ${brightness}%, ${trailLength})`;
        ctx.lineWidth = size * 0.5;
        ctx.stroke();
      }

      // Draw star point
      ctx.beginPath();
      ctx.arc(screenX, screenY, size, 0, Math.PI * 2);
      ctx.fillStyle = `hsl(${hue}, 70%, ${brightness + energyHigh * 20}%)`;
      ctx.fill();
    }
  }

  destroy(): void {
    this.stars = [];
  }
}

