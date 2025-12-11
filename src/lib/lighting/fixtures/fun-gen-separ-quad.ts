import type { FixtureProfile } from "../types";

/**
 * Fun Generation SePar Quad LED RGB UV - 6-channel mode
 *
 * Channel layout:
 *   1: Red (0-255)
 *   2: Green (0-255)
 *   3: Blue (0-255)
 *   4: UV (0-255)
 *   5: Dimmer (0-255)
 *   6: Strobe (0-255)
 */
export const SeParQuadRGBUV_6CH: FixtureProfile = {
  id: "fun-gen-separ-quad-rgbuv-6ch",
  name: "Fun Generation SePar Quad LED RGB UV (6ch)",
  mode: "6ch",
  channels: [
    "red",     // ch1
    "green",   // ch2
    "blue",    // ch3
    "uv",      // ch4
    "dimmer",  // ch5
    "strobe",  // ch6
  ],
};

