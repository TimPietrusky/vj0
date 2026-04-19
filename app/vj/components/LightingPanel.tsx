import type {
  FixtureInstance,
  StrobeMode,
  ColorMode,
  DimmerMode,
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
  onDmxReconnect: () => void;

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
  onFixtureDimmerModeChange: (id: string, mode: DimmerMode) => void;
  onFixtureManualDimmerChange: (id: string, value: number) => void;
}

/**
 * Lighting panel - contains DMX controls and fixture management.
 */
export function LightingPanel({
  dmxStatus,
  dmxSupported,
  onDmxConnect,
  onDmxDisconnect,
  onDmxReconnect,
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
  onFixtureDimmerModeChange,
  onFixtureManualDimmerChange,
}: LightingPanelProps) {
  return (
    <div className="font-mono text-xs flex flex-col gap-2">
      {/* Action row: pair button on the left, fixture profile picker on the
          right. Status itself lives in the SystemsBar — don't repeat it. */}
      <div className="grid grid-cols-[auto_1fr] gap-2 items-end">
        <DmxControls
          status={dmxStatus}
          supported={dmxSupported}
          onConnect={onDmxConnect}
          onDisconnect={onDmxDisconnect}
          onReconnect={onDmxReconnect}
        />
        <FixtureSelector
          selectedProfileId={selectedProfileId}
          onProfileSelect={onProfileSelect}
          onAdd={onAddFixture}
        />
      </div>

      <div className="space-y-2">
        {fixtures.length === 0 ? (
          <div className="text-[color:var(--vj-ink-dim)] text-center py-2 text-[11px] uppercase tracking-wider">
            no fixtures yet — pick a profile and click + add
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
              onDimmerModeChange={(mode) =>
                onFixtureDimmerModeChange(fixture.id, mode)
              }
              onManualDimmerChange={(v) =>
                onFixtureManualDimmerChange(fixture.id, v)
              }
            />
          ))
        )}
      </div>
    </div>
  );
}
