import { useMemo } from "react";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  FixtureInstance,
  FixtureProfile,
  StrobeMode,
  ColorMode,
  DimmerMode,
} from "../lighting/types";
import { FIXTURE_PROFILES, DEFAULT_FIXTURES } from "../lighting/fixtures";

/**
 * Serializable fixture state for persistence
 * We store profile ID instead of the full profile object
 */
interface SerializedFixture {
  id: string;
  profileId: string;
  address: number;
  mapping: { x: number; y: number };
  strobeMode: StrobeMode;
  strobeThreshold: number;
  strobeMax: number;
  colorMode: ColorMode;
  solidColor: { r: number; g: number; b: number };
  // Optional in the type so pre-existing persisted fixtures (without these
  // fields) still load; deserialize fills in defaults ("auto", 255).
  dimmerMode?: DimmerMode;
  manualDimmer?: number;
}

interface LightingState {
  fixtures: SerializedFixture[];
  /**
   * Master switch for the entire DMX/lighting subsystem. When false, the UI
   * hides the fixture list, fog card, and DMX status — and the DMX output
   * driver is bypassed so no frames go to the physical universe. Audio
   * features and visuals keep running; only the lighting branch is silenced.
   * Persisted so a "DMX off for tonight, just visuals" choice survives reloads.
   */
  enabled: boolean;

  // Actions
  setEnabled: (value: boolean) => void;
  addFixture: (profileId: string) => void;
  removeFixture: (id: string) => void;
  updateFixtureAddress: (id: string, address: number) => void;
  updateFixtureStrobeMode: (id: string, mode: StrobeMode) => void;
  updateFixtureStrobeThreshold: (id: string, threshold: number) => void;
  updateFixtureStrobeMax: (id: string, max: number) => void;
  updateFixtureColorMode: (id: string, mode: ColorMode) => void;
  updateFixtureSolidColor: (
    id: string,
    color: { r: number; g: number; b: number }
  ) => void;
  updateFixtureProfile: (id: string, profileId: string) => void;
  updateFixtureDimmerMode: (id: string, mode: DimmerMode) => void;
  updateFixtureManualDimmer: (id: string, value: number) => void;
}

/**
 * Convert serialized fixture to full FixtureInstance
 */
export function deserializeFixture(
  serialized: SerializedFixture
): FixtureInstance | null {
  const profile = FIXTURE_PROFILES.find((p) => p.id === serialized.profileId);
  if (!profile) return null;

  return {
    id: serialized.id,
    profile,
    address: serialized.address,
    mapping: serialized.mapping,
    strobeMode: serialized.strobeMode,
    strobeThreshold: serialized.strobeThreshold,
    strobeMax: serialized.strobeMax,
    colorMode: serialized.colorMode,
    solidColor: serialized.solidColor,
    // Back-fill dimmer fields with auto/full so fixtures saved before this
    // feature existed keep their previous behaviour on load.
    dimmerMode: serialized.dimmerMode ?? "auto",
    manualDimmer: serialized.manualDimmer ?? 255,
  };
}

/**
 * Convert FixtureInstance to serializable format
 */
function serializeFixture(fixture: FixtureInstance): SerializedFixture {
  return {
    id: fixture.id,
    profileId: fixture.profile.id,
    address: fixture.address,
    mapping: fixture.mapping,
    strobeMode: fixture.strobeMode,
    strobeThreshold: fixture.strobeThreshold,
    strobeMax: fixture.strobeMax,
    colorMode: fixture.colorMode,
    solidColor: fixture.solidColor,
    dimmerMode: fixture.dimmerMode,
    manualDimmer: fixture.manualDimmer,
  };
}

/**
 * Generate unique fixture ID
 */
function generateFixtureId(profile: FixtureProfile): string {
  return `${profile.id.split("-")[0]}-${Date.now().toString(36)}`;
}

/**
 * Get default fixture values for a profile
 */
function createDefaultFixture(
  profile: FixtureProfile,
  address: number
): SerializedFixture {
  return {
    id: generateFixtureId(profile),
    profileId: profile.id,
    address,
    mapping: { x: 0.5, y: 0.5 },
    strobeMode: "off",
    strobeThreshold: 0.2,
    strobeMax: 200,
    colorMode: "canvas",
    solidColor: { r: 255, g: 255, b: 255 },
    dimmerMode: "auto",
    manualDimmer: 255,
  };
}

/**
 * Calculate next available DMX address
 */
function getNextAvailableAddress(fixtures: SerializedFixture[]): number {
  if (fixtures.length === 0) return 1;

  // Find the highest used address
  let maxEnd = 0;
  for (const f of fixtures) {
    const profile = FIXTURE_PROFILES.find((p) => p.id === f.profileId);
    if (profile) {
      const end = f.address + profile.channels.length - 1;
      if (end > maxEnd) maxEnd = end;
    }
  }

  return Math.min(maxEnd + 1, 512);
}

// Initialize with default fixtures serialized
const initialFixtures: SerializedFixture[] =
  DEFAULT_FIXTURES.map(serializeFixture);

export const useLightingStore = create<LightingState>()(
  persist(
    (set) => ({
      fixtures: initialFixtures,
      enabled: true,

      setEnabled: (value: boolean) => set({ enabled: value }),

      addFixture: (profileId: string) =>
        set((state) => {
          const profile = FIXTURE_PROFILES.find((p) => p.id === profileId);
          if (!profile) return state;

          const address = getNextAvailableAddress(state.fixtures);
          const newFixture = createDefaultFixture(profile, address);

          return { fixtures: [...state.fixtures, newFixture] };
        }),

      removeFixture: (id: string) =>
        set((state) => ({
          fixtures: state.fixtures.filter((f) => f.id !== id),
        })),

      updateFixtureAddress: (id: string, address: number) =>
        set((state) => ({
          fixtures: state.fixtures.map((f) =>
            f.id === id
              ? { ...f, address: Math.max(1, Math.min(512, address)) }
              : f
          ),
        })),

      updateFixtureStrobeMode: (id: string, mode: StrobeMode) =>
        set((state) => ({
          fixtures: state.fixtures.map((f) =>
            f.id === id ? { ...f, strobeMode: mode } : f
          ),
        })),

      updateFixtureStrobeThreshold: (id: string, threshold: number) =>
        set((state) => ({
          fixtures: state.fixtures.map((f) =>
            f.id === id
              ? { ...f, strobeThreshold: Math.max(0, Math.min(1, threshold)) }
              : f
          ),
        })),

      updateFixtureStrobeMax: (id: string, max: number) =>
        set((state) => ({
          fixtures: state.fixtures.map((f) =>
            f.id === id
              ? { ...f, strobeMax: Math.max(0, Math.min(255, max)) }
              : f
          ),
        })),

      updateFixtureColorMode: (id: string, mode: ColorMode) =>
        set((state) => ({
          fixtures: state.fixtures.map((f) =>
            f.id === id ? { ...f, colorMode: mode } : f
          ),
        })),

      updateFixtureSolidColor: (
        id: string,
        color: { r: number; g: number; b: number }
      ) =>
        set((state) => ({
          fixtures: state.fixtures.map((f) =>
            f.id === id
              ? {
                  ...f,
                  solidColor: {
                    r: Math.max(0, Math.min(255, color.r)),
                    g: Math.max(0, Math.min(255, color.g)),
                    b: Math.max(0, Math.min(255, color.b)),
                  },
                }
              : f
          ),
        })),

      updateFixtureProfile: (id: string, profileId: string) =>
        set((state) => ({
          fixtures: state.fixtures.map((f) =>
            f.id === id ? { ...f, profileId } : f
          ),
        })),

      updateFixtureDimmerMode: (id: string, mode: DimmerMode) =>
        set((state) => ({
          fixtures: state.fixtures.map((f) =>
            f.id === id ? { ...f, dimmerMode: mode } : f
          ),
        })),

      updateFixtureManualDimmer: (id: string, value: number) =>
        set((state) => ({
          fixtures: state.fixtures.map((f) =>
            f.id === id
              ? {
                  ...f,
                  manualDimmer: Math.max(
                    0,
                    Math.min(255, Math.round(value))
                  ),
                }
              : f
          ),
        })),
    }),
    {
      name: "vj0-lighting-storage",
    }
  )
);

/**
 * Hook to get deserialized fixtures. Memoised against the serialised array
 * identity so callers get the same FixtureInstance[] reference across
 * unrelated renders — critical because VJApp's engine-recreation effect
 * has `fixtures` in its deps, and a new reference on every render would
 * recreate the LightingEngine every ~16 ms, nuking runtime state like the
 * manual fog toggle before the user could see it take effect.
 */
export function useFixtures(): FixtureInstance[] {
  const serialized = useLightingStore((state) => state.fixtures);
  return useMemo(
    () =>
      serialized
        .map(deserializeFixture)
        .filter((f): f is FixtureInstance => f !== null),
    [serialized]
  );
}
