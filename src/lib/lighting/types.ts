/**
 * Lighting types for DMX512 control
 */

// DMX universe: 512 channels, each 0-255
export type DmxUniverse = Uint8Array;

// Audio feature source for strobe control
export type StrobeMode =
  | "off"
  | "energyLow"
  | "energyMid"
  | "energyHigh"
  | "peak"
  | "rms";

// Color source mode
export type ColorMode = "canvas" | "solid";

// Frame emitted by the lighting engine
export type LightingFrame = {
  universe: DmxUniverse;
};

// Configuration for the lighting engine
export type LightingConfig = {
  tickHz: number; // update rate, e.g. 20-30
};

// Fixture channel kinds
export type FixtureChannelKind =
  | "red"
  | "green"
  | "blue"
  | "uv"
  | "dimmer"
  | "strobe"
  | "program"
  | "programSpeed";

// Strobe range configuration for fixture profiles
export interface StrobeRange {
  /** DMX value when strobe is off (default: 0) */
  off: number;
  /** Minimum DMX value for strobe effect (default: 0) */
  min: number;
  /** Maximum DMX value for strobe effect (default: 255) */
  max: number;
}

// Fixture profile describes a fixture type and its channel layout
export interface FixtureProfile {
  id: string;
  name: string;
  mode: string;
  channels: FixtureChannelKind[];
  /**
   * Strobe DMX range for this fixture.
   * If not specified, defaults to { off: 0, min: 0, max: 255 }
   */
  strobeRange?: StrobeRange;
}

// Fixture instance is a concrete fixture in the rig
export interface FixtureInstance {
  id: string;
  profile: FixtureProfile;
  /**
   * DMX start channel, 1..512
   * Constraint: address + profile.channels.length - 1 <= 512
   */
  address: number;
  mapping: {
    x: number; // 0..1 normalized canvas X
    y: number; // 0..1 normalized canvas Y
  };
  /**
   * Which audio feature drives the strobe channel
   * Default: "off" (no strobe)
   */
  strobeMode: StrobeMode;
  /**
   * Threshold for strobe activation (0-1)
   * Strobe only triggers when feature exceeds this value
   */
  strobeThreshold: number;
  /**
   * Maximum strobe value (0-255)
   * Limits how fast the strobe can go
   */
  strobeMax: number;
  /**
   * Color source mode: "canvas" samples from visuals, "solid" uses solidColor
   */
  colorMode: ColorMode;
  /**
   * Solid color to use when colorMode is "solid"
   * RGB values 0-255
   */
  solidColor: { r: number; g: number; b: number };
}
