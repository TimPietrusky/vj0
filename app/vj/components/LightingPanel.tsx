import type {
  FixtureInstance,
  StrobeMode,
  ColorMode,
} from "@/src/lib/lighting";
import { DmxControls } from "./DmxControls";
import { FixtureSelector } from "./FixtureSelector";
import { FixtureInspector } from "./FixtureInspector";

type DmxStatus = "disconnected" | "connecting" | "connected" | "unsupported";

interface LightingPanelProps {
  // DMX state
  dmxStatus: DmxStatus;
  dmxSupported: boolean;
  onDmxConnect: () => void;
  onDmxDisconnect: () => void;

  // Fixture selector
  selectedProfileId: string;
  onProfileSelect: (profileId: string) => void;
  onAddFixture: () => void;

  // Fixtures
  fixtures: FixtureInstance[];
  fixtureValues: Map<string, Uint8Array>;
  onFixtureAddressChange: (id: string, addr: number) => void;
  onFixtureStrobeModeChange: (id: string, mode: StrobeMode) => void;
  onFixtureStrobeThresholdChange: (id: string, threshold: number) => void;
  onFixtureStrobeMaxChange: (id: string, max: number) => void;
  onFixtureColorModeChange: (id: string, mode: ColorMode) => void;
  onFixtureSolidColorChange: (
    id: string,
    color: { r: number; g: number; b: number }
  ) => void;
  onFixtureProfileChange: (id: string, profileId: string) => void;
  onFixtureRemove: (id: string) => void;
}

/**
 * Lighting panel - contains DMX controls and fixture management.
 */
export function LightingPanel({
  dmxStatus,
  dmxSupported,
  onDmxConnect,
  onDmxDisconnect,
  selectedProfileId,
  onProfileSelect,
  onAddFixture,
  fixtures,
  fixtureValues,
  onFixtureAddressChange,
  onFixtureStrobeModeChange,
  onFixtureStrobeThresholdChange,
  onFixtureStrobeMaxChange,
  onFixtureColorModeChange,
  onFixtureSolidColorChange,
  onFixtureProfileChange,
  onFixtureRemove,
}: LightingPanelProps) {
  return (
    <div className="font-mono text-xs flex flex-col gap-3">
      <DmxControls
        status={dmxStatus}
        supported={dmxSupported}
        onConnect={onDmxConnect}
        onDisconnect={onDmxDisconnect}
      />

      <FixtureSelector
        selectedProfileId={selectedProfileId}
        onProfileSelect={onProfileSelect}
        onAdd={onAddFixture}
      />

      <div className="space-y-4">
        {fixtures.length === 0 ? (
          <div className="text-neutral-500 text-center py-4">
            No fixtures configured. Add one above.
          </div>
        ) : (
          fixtures.map((fixture) => (
            <FixtureInspector
              key={fixture.id}
              fixture={fixture}
              values={fixtureValues.get(fixture.id)}
              onAddressChange={(addr) =>
                onFixtureAddressChange(fixture.id, addr)
              }
              onStrobeModeChange={(mode) =>
                onFixtureStrobeModeChange(fixture.id, mode)
              }
              onStrobeThresholdChange={(t) =>
                onFixtureStrobeThresholdChange(fixture.id, t)
              }
              onStrobeMaxChange={(m) => onFixtureStrobeMaxChange(fixture.id, m)}
              onColorModeChange={(mode) =>
                onFixtureColorModeChange(fixture.id, mode)
              }
              onSolidColorChange={(color) =>
                onFixtureSolidColorChange(fixture.id, color)
              }
              onProfileChange={(profileId) =>
                onFixtureProfileChange(fixture.id, profileId)
              }
              onRemove={() => onFixtureRemove(fixture.id)}
            />
          ))
        )}
      </div>
    </div>
  );
}
