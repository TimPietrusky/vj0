import type { FixtureProfile } from "../types";

/**
 * Stairville Wild Wash Pro 648 LED RGB - 6-channel mode
 *
 * Channel layout:
 *   1: Dimmer (0-255, 0% to 100%)
 *   2: Strobe:
 *      0-5: LEDs on, controlled by dimmer
 *      6-10: Blackout
 *      11-33: Random pulses, speed increasing
 *      34-56: Random rising brightness, speed increasing
 *      57-79: Random falling brightness, speed increasing
 *      80-102: Random strobe effect, speed increasing
 *      103-127: Interruption effect, 5s to 1s
 *      128-250: Strobe effect, speed increasing (~0 Hz to 30 Hz)
 *      251-255: LEDs on, controlled by dimmer
 *   3: Red (0-255)
 *   4: Green (0-255)
 *   5: Blue (0-255)
 *   6: Music control (0-5 = off, 6-255 = on with increasing sensitivity)
 */
export const StairvilleWildWashPro648_6CH: FixtureProfile = {
  id: "stairville-wild-wash-pro-648-6ch",
  name: "Stairville Wild Wash Pro 648 (6ch)",
  mode: "6ch",
  channels: [
    "dimmer", // ch1 - 0-255 controls brightness
    "strobe", // ch2 - 0 for normal, 128-250 for strobe
    "red", // ch3
    "green", // ch4
    "blue", // ch5
    "program", // ch6 - music control, keep at 0
  ],
  strobeRange: {
    off: 0, // LEDs on, controlled by dimmer
    min: 128, // Slowest strobe
    max: 250, // Fastest strobe (~30 Hz)
  },
};
