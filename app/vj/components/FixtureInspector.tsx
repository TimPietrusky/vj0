import type {
  FixtureInstance,
  StrobeMode,
  ColorMode,
} from "@/src/lib/lighting";
import { FIXTURE_PROFILES } from "@/src/lib/lighting";

// Channel kind to display label mapping
const CHANNEL_LABELS: Record<string, string> = {
  red: "R",
  green: "G",
  blue: "B",
  uv: "UV",
  dimmer: "DIM",
  strobe: "STR",
  program: "PRG",
  programSpeed: "SPD",
};

// Channel kind to color mapping for bars
const CHANNEL_COLORS: Record<string, string> = {
  red: "bg-red-500",
  green: "bg-green-500",
  blue: "bg-blue-500",
  uv: "bg-violet-500",
  dimmer: "bg-amber-500",
  strobe: "bg-white",
  program: "bg-neutral-500",
  programSpeed: "bg-neutral-500",
};

// Strobe mode options for the dropdown
const STROBE_MODE_OPTIONS: { value: StrobeMode; label: string }[] = [
  { value: "off", label: "Off" },
  { value: "energyLow", label: "Bass (Low)" },
  { value: "energyMid", label: "Mid" },
  { value: "energyHigh", label: "High" },
  { value: "peak", label: "Peak" },
  { value: "rms", label: "RMS" },
];

interface FixtureInspectorProps {
  fixture: FixtureInstance;
  values?: Uint8Array;
  onAddressChange: (addr: number) => void;
  onStrobeModeChange: (mode: StrobeMode) => void;
  onStrobeThresholdChange: (threshold: number) => void;
  onStrobeMaxChange: (max: number) => void;
  onColorModeChange: (mode: ColorMode) => void;
  onSolidColorChange: (color: { r: number; g: number; b: number }) => void;
  onProfileChange: (profileId: string) => void;
  onRemove: () => void;
}

/**
 * Fixture inspector component - shows fixture info, address control, and channel values.
 * Pure presentational - all state managed by parent.
 */
export function FixtureInspector({
  fixture,
  values,
  onAddressChange,
  onStrobeModeChange,
  onStrobeThresholdChange,
  onStrobeMaxChange,
  onColorModeChange,
  onSolidColorChange,
  onProfileChange,
  onRemove,
}: FixtureInspectorProps) {
  const maxAddress = 512 - fixture.profile.channels.length + 1;
  const hasStrobe = fixture.profile.channels.includes("strobe");
  const {
    address,
    strobeMode,
    strobeThreshold,
    strobeMax,
    colorMode,
    solidColor,
  } = fixture;

  // Get RGB values for color preview
  const r = values?.[fixture.profile.channels.indexOf("red")] ?? 0;
  const g = values?.[fixture.profile.channels.indexOf("green")] ?? 0;
  const b = values?.[fixture.profile.channels.indexOf("blue")] ?? 0;

  return (
    <div className="bg-neutral-800/50 rounded p-3">
      {/* Header with profile selector and controls */}
      <div className="flex items-center justify-between mb-3 gap-3">
        <div className="flex items-center gap-3 flex-1">
          <select
            value={fixture.profile.id}
            onChange={(e) => onProfileChange(e.target.value)}
            className="bg-neutral-900 border border-neutral-600 text-neutral-200 text-xs rounded px-2 py-1 focus:outline-none focus:border-emerald-500"
          >
            {FIXTURE_PROFILES.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.name}
              </option>
            ))}
          </select>
          <span className="text-neutral-500 text-xs">ID: {fixture.id}</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-neutral-500 text-xs">DMX:</span>
            <input
              type="number"
              min={1}
              max={maxAddress}
              value={address}
              onChange={(e) =>
                onAddressChange(parseInt(e.target.value, 10) || 1)
              }
              className="w-14 bg-neutral-900 border border-neutral-600 text-neutral-200 text-xs rounded px-2 py-1 text-center focus:outline-none focus:border-emerald-500"
            />
          </div>
          <button
            onClick={onRemove}
            className="px-2 py-1 bg-red-600/20 hover:bg-red-600/40 text-red-400 rounded text-xs transition-colors"
            title="Remove fixture"
          >
            Remove
          </button>
        </div>
      </div>

      {/* Color source controls */}
      <div className="flex items-center gap-4 mb-3">
        <div className="flex items-center gap-2">
          <span className="text-neutral-500 text-xs">Color:</span>
          <button
            onClick={() => onColorModeChange("canvas")}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              colorMode === "canvas"
                ? "bg-emerald-600 text-white"
                : "bg-neutral-700 text-neutral-400 hover:bg-neutral-600"
            }`}
          >
            Canvas
          </button>
          <button
            onClick={() => onColorModeChange("solid")}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              colorMode === "solid"
                ? "bg-emerald-600 text-white"
                : "bg-neutral-700 text-neutral-400 hover:bg-neutral-600"
            }`}
          >
            Solid
          </button>
        </div>

        {colorMode === "solid" && (
          <div className="flex items-center gap-2">
            <input
              type="color"
              value={`#${solidColor.r
                .toString(16)
                .padStart(2, "0")}${solidColor.g
                .toString(16)
                .padStart(2, "0")}${solidColor.b
                .toString(16)
                .padStart(2, "0")}`}
              onChange={(e) => {
                const hex = e.target.value;
                const r = parseInt(hex.slice(1, 3), 16);
                const g = parseInt(hex.slice(3, 5), 16);
                const b = parseInt(hex.slice(5, 7), 16);
                onSolidColorChange({ r, g, b });
              }}
              className="w-8 h-6 rounded border border-neutral-600 cursor-pointer"
            />
            <span className="text-neutral-400 text-xs">
              {solidColor.r}, {solidColor.g}, {solidColor.b}
            </span>
          </div>
        )}
      </div>

      {/* Color preview and channel bars */}
      <div className="flex gap-4 mb-3">
        {/* RGB Color preview */}
        <div
          className="w-12 h-12 rounded border border-neutral-600 shrink-0"
          style={{ backgroundColor: `rgb(${r}, ${g}, ${b})` }}
          title={`RGB: ${r}, ${g}, ${b}`}
        />

        {/* Channel bars */}
        <div className="flex-1 grid grid-cols-3 sm:grid-cols-6 gap-2">
          {fixture.profile.channels.map((kind, i) => {
            const value = values?.[i] ?? 0;
            const label = CHANNEL_LABELS[kind] || kind;
            const barColor = CHANNEL_COLORS[kind] || "bg-neutral-500";

            return (
              <div key={i} className="flex flex-col gap-1">
                <div className="flex justify-between items-baseline">
                  <span className="text-neutral-500">{label}</span>
                  <span className="text-neutral-300">{value}</span>
                </div>
                <div className="h-1.5 bg-neutral-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${barColor} transition-all duration-75`}
                    style={{ width: `${(value / 255) * 100}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Strobe controls - only show if fixture has strobe channel */}
      {hasStrobe && (
        <div className="border-t border-neutral-700 pt-3 mt-3">
          <div className="text-neutral-400 text-xs uppercase tracking-wide mb-2">
            Audio-Reactive Strobe
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* Strobe mode selector */}
            <div className="flex flex-col gap-1">
              <label className="text-neutral-500 text-xs">Source</label>
              <select
                value={strobeMode}
                onChange={(e) =>
                  onStrobeModeChange(e.target.value as StrobeMode)
                }
                className="bg-neutral-900 border border-neutral-600 text-neutral-200 text-xs rounded px-2 py-1 focus:outline-none focus:border-emerald-500"
              >
                {STROBE_MODE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Threshold slider */}
            <div className="flex flex-col gap-1">
              <label className="text-neutral-500 text-xs">
                Threshold: {(strobeThreshold * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min={0}
                max={100}
                value={strobeThreshold * 100}
                onChange={(e) =>
                  onStrobeThresholdChange(parseInt(e.target.value, 10) / 100)
                }
                className="w-full accent-emerald-500"
                disabled={strobeMode === "off"}
              />
            </div>

            {/* Max speed slider */}
            <div className="flex flex-col gap-1">
              <label className="text-neutral-500 text-xs">
                Max Speed: {strobeMax}
              </label>
              <input
                type="range"
                min={0}
                max={255}
                value={strobeMax}
                onChange={(e) =>
                  onStrobeMaxChange(parseInt(e.target.value, 10))
                }
                className="w-full accent-emerald-500"
                disabled={strobeMode === "off"}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
