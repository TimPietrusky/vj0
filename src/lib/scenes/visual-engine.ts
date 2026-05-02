/**
 * VisualEngine - Scene manager and render loop coordinator
 *
 * Owns the single requestAnimationFrame loop and delegates rendering to the active scene.
 * Handles scene switching, canvas resize, and audio data flow.
 *
 * Extension points:
 * - Add scene transition effects (crossfade, wipe, etc.)
 * - Add multiple canvas layers for compositing
 * - Add post-processing effects pipeline
 */

import type { AudioEngine } from '../audio-engine';
import type { VjScene } from './types';

export class VisualEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private audioEngine: AudioEngine;
  private scenes: VjScene[];

  private currentScene: VjScene;
  private animationFrameId: number | null = null;
  private isRunning = false;
  private lastFrameTime = 0;

  // Pre-allocated buffer for time-domain data (reused every frame)
  private timeDomainBuffer: Float32Array;

  // Internal canvas resolution — driven by the AI output dimensions so the
  // pixel buffer matches what gets captured and sent. No hidden crop or
  // stretch: what you see on the input canvas is exactly what the AI sees.
  private width: number;
  private height: number;

  constructor(
    canvas: HTMLCanvasElement,
    audioEngine: AudioEngine,
    scenes: VjScene[],
    /** Initial canvas resolution. Pass the AI output width/height so the
     *  waveform renders at the same dimensions the AI will generate at. */
    initialWidth = 512,
    initialHeight = 288,
  ) {
    if (scenes.length === 0) {
      throw new Error('VisualEngine requires at least one scene');
    }

    this.width = initialWidth;
    this.height = initialHeight;

    this.canvas = canvas;
    this.audioEngine = audioEngine;
    this.scenes = scenes;
    this.currentScene = scenes[0];

    // Set canvas resolution to match the supplied dimensions
    this.canvas.width = this.width;
    this.canvas.height = this.height;

    const ctx = this.canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get 2D canvas context');
    }
    this.ctx = ctx;

    // Pre-allocate buffer for audio data
    this.timeDomainBuffer = new Float32Array(audioEngine.bufferSize);

    // Initialize first scene
    this.currentScene.init?.(this.canvas);
    this.currentScene.resize?.(this.width, this.height);
  }

  /**
   * Start the render loop.
   */
  start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.lastFrameTime = performance.now();

    const render = (timestamp: number): void => {
      if (!this.isRunning) return;

      // Calculate delta time
      const dt = timestamp - this.lastFrameTime;
      this.lastFrameTime = timestamp;

      // Get fresh audio data
      this.audioEngine.getTimeDomainData(this.timeDomainBuffer);
      const features = this.audioEngine.getLatestFeatures();

      // Delegate rendering to current scene
      this.currentScene.render(this.ctx, features, this.timeDomainBuffer, dt);

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
    this.currentScene.destroy?.();
  }

  /**
   * Switch to a scene by its ID.
   * @returns true if scene was found and switched, false otherwise
   */
  setSceneById(id: string): boolean {
    const scene = this.scenes.find((s) => s.id === id);
    if (!scene || scene === this.currentScene) {
      return scene !== undefined;
    }

    // Cleanup old scene
    this.currentScene.destroy?.();

    // Initialize new scene
    this.currentScene = scene;
    this.currentScene.init?.(this.canvas);
    this.currentScene.resize?.(this.width, this.height);

    return true;
  }

  /**
   * Get the currently active scene.
   */
  getCurrentScene(): VjScene {
    return this.currentScene;
  }

  /**
   * Get all registered scenes.
   */
  getScenes(): readonly VjScene[] {
    return this.scenes;
  }

  /**
   * Handle canvas resize. Call this when the container size changes.
   */
  handleResize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.currentScene.resize?.(width, height);
  }
}

