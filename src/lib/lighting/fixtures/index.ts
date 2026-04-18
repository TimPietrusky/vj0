import type { FixtureInstance, FixtureProfile } from "../types";
import { SeParQuadRGBUV_6CH } from "./fun-gen-separ-quad";
import { StairvilleWildWashPro648_6CH } from "./stairville-wild-wash-pro";
import { StarvilleAF150_1CH } from "./starville-af-150";

export { SeParQuadRGBUV_6CH } from "./fun-gen-separ-quad";
export { StairvilleWildWashPro648_6CH } from "./stairville-wild-wash-pro";
export { StarvilleAF150_1CH } from "./starville-af-150";

/**
 * Available fixture profiles for the fixture selector
 */
export const FIXTURE_PROFILES: FixtureProfile[] = [
  SeParQuadRGBUV_6CH,
  StairvilleWildWashPro648_6CH,
  StarvilleAF150_1CH,
];

/**
 * Default fixture setup for vj0
 * Currently contains one SePar Quad LED at DMX address 1, sampling center of canvas
 */
export const DEFAULT_FIXTURES: FixtureInstance[] = [
  {
    id: "separ-1",
    profile: SeParQuadRGBUV_6CH,
    address: 1,
    mapping: { x: 0.5, y: 0.5 },
    strobeMode: "off",
    strobeThreshold: 0.2,
    strobeMax: 200,
    colorMode: "canvas",
    solidColor: { r: 255, g: 255, b: 255 },
  },
];

// Keep FIXTURES export for backwards compatibility
export const FIXTURES = DEFAULT_FIXTURES;
