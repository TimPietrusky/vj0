/**
 * AudioFeatures - Real-time audio analysis data from the AudioWorklet
 *
 * All values are normalized to [0, 1] for consistent mapping to visuals, DMX, or AI.
 * This type is designed to be extensible for future features like beat detection.
 *
 * Extension points:
 * - Add `beat: boolean` and `tempo: number | null` for beat detection
 * - Add `spectrum: Float32Array` for full frequency data (requires SharedArrayBuffer)
 * - Add `onset: boolean` for transient detection
 *
 * How visuals/DMX can consume:
 * - Poll `audioEngine.getLatestFeatures()` in requestAnimationFrame loop
 * - Map feature values directly to visual parameters (e.g., rms -> brightness)
 * - Use energyLow/Mid/High for frequency-reactive color or intensity
 */
export type AudioFeatures = {
  /** Root-mean-square loudness, 0..1 normalized */
  rms: number;

  /** Peak absolute amplitude, 0..1 */
  peak: number;

  /** Energy in low frequency band (~20-250 Hz), 0..1 */
  energyLow: number;

  /** Energy in mid frequency band (~250-4000 Hz), 0..1 */
  energyMid: number;

  /** Energy in high frequency band (~4000-20000 Hz), 0..1 */
  energyHigh: number;

  /** Spectral centroid (brightness), normalized 0..1 */
  spectralCentroid: number;

  // Reserved for future:
  // beat: boolean;
  // tempo: number | null;
};

