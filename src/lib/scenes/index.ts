/**
 * Scene Registry - Central export for all VjScene implementations
 *
 * Add new scenes here to make them available in the UI.
 * Order determines the default scene (first in array).
 */

export type { VjScene } from './types';
export { VisualEngine } from './visual-engine';
export { WaveformScene } from './waveform-scene';
export { SpectrumBarsScene } from './spectrum-bars-scene';
export { RadialPulseScene } from './radial-pulse-scene';
export { ParticleFieldScene } from './particle-field-scene';
export { TerrainLinesScene } from './terrain-lines-scene';
export { StarfieldScene } from './starfield-scene';

import type { VjScene } from './types';
import { WaveformScene } from './waveform-scene';
import { SpectrumBarsScene } from './spectrum-bars-scene';
import { RadialPulseScene } from './radial-pulse-scene';
import { ParticleFieldScene } from './particle-field-scene';
import { TerrainLinesScene } from './terrain-lines-scene';
import { StarfieldScene } from './starfield-scene';

/**
 * All available scenes. First scene is the default.
 * To add a new scene:
 * 1. Create a class implementing VjScene
 * 2. Export it above
 * 3. Add an instance here
 */
export const SCENES: VjScene[] = [
  new WaveformScene(),
  new SpectrumBarsScene(),
  new RadialPulseScene(),
  new ParticleFieldScene(),
  new TerrainLinesScene(),
  new StarfieldScene(),
];

