import type { FixtureProfile } from "../types";

/**
 * Stairville Wild Wash Pro 648 LED RGB - 6-channel mode
 *
 * Channel layout:
 *   1: Dimmer (0-255, 0% to 100%)
 *   2: Strobe (0-5 = LEDs on controlled by dimmer, 6-10 = blackout, 11-255 = various strobe effects)
 *      We keep this at 0 by default to have LEDs on, controlled by dimmer
 *   3: Red (0-255)
 *   4: Green (0-255)
 *   5: Blue (0-255)
 *   6: Music control (0-5 = off, 6-255 = on with increasing sensitivity)
 *      We keep this at 0 to disable music control
 */
export const StairvilleWildWashPro648_6CH: FixtureProfile = {
  id: "stairville-wild-wash-pro-648-6ch",
  name: "Stairville Wild Wash Pro 648 (6ch)",
  mode: "6ch",
  channels: [
    "dimmer", // ch1 - 0-255 controls brightness
    "strobe", // ch2 - keep at 0-5 for normal operation
    "red", // ch3
    "green", // ch4
    "blue", // ch5
    "program", // ch6 - music control, keep at 0
  ],
};
