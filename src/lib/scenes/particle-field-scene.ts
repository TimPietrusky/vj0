import type { VjScene } from "./types";
import type { AudioFeatures } from "../audio-features";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  hue: number;
  life: number;
}

/**
 * Floating particles that accelerate/change direction based on audio.
 * Peak triggers bursts, spectralCentroid shifts gravity direction.
 */
export class ParticleFieldScene implements VjScene {
  readonly id = "particle-field";
  readonly name = "Particle Field";

  private width = 0;
  private height = 0;

  // Pre-allocated particle pool
  private readonly maxParticles = 200;
  private particles: Particle[] = [];
  private activeCount = 0;

  init(): void {
    // Pre-allocate particle objects
    this.particles = [];
    for (let i = 0; i < this.maxParticles; i++) {
      this.particles[i] = { x: 0, y: 0, vx: 0, vy: 0, size: 0, hue: 0, life: 0 };
    }
    this.activeCount = 0;
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;

    // Reset particles on resize
    this.activeCount = 0;
  }

  private spawnParticle(x: number, y: number, hue: number, speed: number): void {
    if (this.activeCount >= this.maxParticles) return;

    const p = this.particles[this.activeCount];
    const angle = Math.random() * Math.PI * 2;

    p.x = x;
    p.y = y;
    p.vx = Math.cos(angle) * speed;
    p.vy = Math.sin(angle) * speed;
    p.size = 2 + Math.random() * 4;
    p.hue = hue;
    p.life = 1;

    this.activeCount++;
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

    const dtSeconds = dt / 1000;

    // Gravity direction based on spectral centroid (0 = down, 1 = up)
    const gravityY = (0.5 - spectralCentroid) * 200;
    const gravityX = (energyMid - 0.5) * 100;

    // Spawn particles based on energy
    const spawnRate = Math.floor(2 + rms * 10 + peak * 15);
    const baseHue = energyHigh * 360;

    for (let i = 0; i < spawnRate; i++) {
      const x = Math.random() * this.width;
      const y = Math.random() * this.height;
      const speed = 50 + peak * 200;
      this.spawnParticle(x, y, (baseHue + Math.random() * 60) % 360, speed);
    }

    // Update and render particles
    let writeIndex = 0;

    for (let i = 0; i < this.activeCount; i++) {
      const p = this.particles[i];

      // Apply gravity and damping
      p.vx += gravityX * dtSeconds;
      p.vy += gravityY * dtSeconds;
      p.vx *= 0.98;
      p.vy *= 0.98;

      // Boost velocity on peak
      if (peak > 0.5) {
        p.vx *= 1 + peak * 0.5;
        p.vy *= 1 + peak * 0.5;
      }

      // Update position
      p.x += p.vx * dtSeconds;
      p.y += p.vy * dtSeconds;

      // Decay life
      p.life -= dtSeconds * (0.3 + energyHigh * 0.5);

      // Keep alive particles, compact array
      if (p.life > 0 && p.x > -50 && p.x < this.width + 50 && p.y > -50 && p.y < this.height + 50) {
        // Swap to keep alive
        if (writeIndex !== i) {
          const temp = this.particles[writeIndex];
          this.particles[writeIndex] = p;
          this.particles[i] = temp;
        }

        // Render particle
        const alpha = p.life;
        const size = p.size * (0.5 + rms * 0.5);

        ctx.beginPath();
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 80%, ${50 + energyLow * 30}%, ${alpha})`;
        ctx.fill();

        writeIndex++;
      }
    }

    this.activeCount = writeIndex;
  }

  destroy(): void {
    this.particles = [];
    this.activeCount = 0;
  }
}

