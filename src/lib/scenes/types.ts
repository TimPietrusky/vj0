/**
 * VjScene - Interface for all visual scenes in vj0
 *
 * Each scene is a self-contained visual that can be swapped at runtime.
 * Scenes receive audio features and time-domain data for audio-reactive rendering.
 *
 * Extension points:
 * - Add WebGL/WebGPU context support via separate interface or render method overload
 * - Add scene-specific parameters/presets via a config object
 * - Add transition effects between scenes in VisualEngine
 */

import type { AudioFeatures } from '../audio-features';

export interface VjScene {
  /** Unique identifier for scene selection */
  readonly id: string;

  /** Human-readable name for UI display */
  readonly name: string;

  /**
   * Called once when scene becomes active.
   * Use for one-time setup like creating gradients or pre-calculating values.
   */
  init?(canvas: HTMLCanvasElement): void;

  /**
   * Called when canvas dimensions change.
   * Use to recalculate layout-dependent values.
   */
  resize?(width: number, height: number): void;

  /**
   * Render a single frame.
   *
   * @param ctx - Canvas 2D rendering context
   * @param features - Latest audio features (null if not available)
   * @param timeDomain - Raw waveform samples [-1, 1] (null if not available)
   * @param dt - Delta time since last frame in milliseconds
   */
  render(
    ctx: CanvasRenderingContext2D,
    features: AudioFeatures | null,
    timeDomain: Float32Array | null,
    dt: number
  ): void;

  /**
   * Called when scene is deactivated or engine stops.
   * Use for cleanup of scene-specific resources.
   */
  destroy?(): void;
}

