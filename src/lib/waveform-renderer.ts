/**
 * WaveformRenderer - Framework-agnostic Canvas 2D waveform renderer
 *
 * Handles requestAnimationFrame loop and waveform drawing.
 * Uses Canvas 2D for now - designed to be replaceable with WebGL/WebGPU later.
 *
 * Extension points:
 * - Replace Canvas 2D context with WebGL/WebGPU for GPU-accelerated rendering
 * - Move to OffscreenCanvas in a Worker for background rendering
 * - Hook into DMX module by sampling pixel data from the canvas
 * - Feed canvas frames to AI image-to-image models
 */

export class WaveformRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private animationFrameId: number | null = null;
  private isRunning = false;

  // Fixed internal resolution (CSS handles visual scaling)
  private readonly width = 1024;
  private readonly height = 256;

  // Pre-calculated values to avoid allocations in render loop
  private readonly centerY: number;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;

    // Set internal canvas resolution
    this.canvas.width = this.width;
    this.canvas.height = this.height;

    const ctx = this.canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get 2D canvas context');
    }
    this.ctx = ctx;

    // Pre-calculate center for waveform rendering
    this.centerY = this.height / 2;

    // Configure rendering style once (avoid per-frame allocations)
    this.ctx.lineWidth = 2;
    this.ctx.strokeStyle = '#00ff88'; // Neon green waveform
    this.ctx.fillStyle = '#0a0a0a'; // Near-black background
  }

  /**
   * Start the render loop.
   *
   * @param fillBuffer - Callback to fill the buffer with audio data each frame
   * @param buffer - Pre-allocated Float32Array for audio samples
   */
  start(fillBuffer: (buffer: Float32Array) => void, buffer: Float32Array): void {
    if (this.isRunning) return;
    this.isRunning = true;

    const render = (): void => {
      if (!this.isRunning) return;

      // Get fresh audio data
      fillBuffer(buffer);

      // Draw the frame
      this.draw(buffer);

      // Schedule next frame
      this.animationFrameId = requestAnimationFrame(render);
    };

    this.animationFrameId = requestAnimationFrame(render);
  }

  /**
   * Stop the render loop.
   */
  stop(): void {
    this.isRunning = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  /**
   * Draw a single frame of the waveform.
   * Optimized to avoid allocations - no map/filter/forEach.
   */
  private draw(buffer: Float32Array): void {
    const { ctx, width, height, centerY } = this;
    const bufferLength = buffer.length;

    // Clear canvas with background fill
    ctx.fillRect(0, 0, width, height);

    // Draw waveform as a polyline
    ctx.beginPath();

    // Calculate step size to fit buffer across canvas width
    const sliceWidth = width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      // Map audio sample [-1, 1] to canvas Y coordinate
      // Multiply by 0.8 to leave some padding
      const y = centerY + buffer[i] * centerY * 0.8;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    ctx.stroke();
  }
}

