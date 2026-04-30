import type {
  FixtureInstance,
  StrobeMode,
  ColorMode,
  DimmerMode,
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
  fog: "FOG",
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
  fog: "bg-[color:var(--vj-info)]",
  program: "bg-neutral-500",
  programSpeed: "bg-neutral-500",
};

const STROBE_MODE_OPTIONS: { value: StrobeMode; label: string }[] = [
  { value: "off", label: "Off" },
  { value: "energyLow", label: "Bass" },
  { value: "energyMid", label: "Mid" },
  { value: "energyHigh", label: "High" },
  { value: "peak", label: "Peak" },
  { value: "rms", label: "RMS" },
];

const COLOR_CHANNEL_KINDS = new Set(["red", "green", "blue", "uv"]);

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
  onDimmerModeChange: (mode: DimmerMode) => void;
  onManualDimmerChange: (value: number) => void;
}

/**
 * Compact fixture row. Uses vj-* design tokens and only renders sections
 * that apply to the fixture's actual channels:
 *   - Color controls only for fixtures with at least one RGB/UV channel
 *     (so a single-channel fog machine doesn't get a useless color picker)
 *   - Strobe controls only when the profile has a strobe-kind channel
 * Everything else collapses so a minimal fixture is a compact row.
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
  onDimmerModeChange,
  onManualDimmerChange,
}: FixtureInspectorProps) {
  const maxAddress = 512 - fixture.profile.channels.length + 1;
  const hasStrobe = fixture.profile.channels.includes("strobe");
  const hasDimmer = fixture.profile.channels.includes("dimmer");
  const hasColor = fixture.profile.channels.some((k) =>
    COLOR_CHANNEL_KINDS.has(k as string)
  );
  const {
    address,
    strobeMode,
    strobeThreshold,
    strobeMax,
    colorMode,
    solidColor,
    dimmerMode,
    manualDimmer,
  } = fixture;

  const r = values?.[fixture.profile.channels.indexOf("red")] ?? 0;
  const g = values?.[fixture.profile.channels.indexOf("green")] ?? 0;
  const b = values?.[fixture.profile.channels.indexOf("blue")] ?? 0;

  return (
    <div className="rounded border border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-panel-2)] p-2 flex flex-col gap-2 font-mono text-xs">
      {/* Row 1 — profile, DMX address, remove. Single row, no labels. */}
      <div className="grid grid-cols-[1fr_auto_auto] gap-2 items-center">
        <select
          value={fixture.profile.id}
          onChange={(e) => onProfileChange(e.target.value)}
          className="vj-input text-[11px]"
          title="Fixture profile"
        >
          {FIXTURE_PROFILES.map((profile) => (
            <option key={profile.id} value={profile.id}>
              {profile.name}
            </option>
          ))}
        </select>
        <div
          className="flex items-center gap-1"
          title="DMX start address (1..512 minus channel count)"
        >
          <span className="text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)]">
            ch
          </span>
          <input
            type="number"
            min={1}
            max={maxAddress}
            value={address}
            onChange={(e) =>
              onAddressChange(parseInt(e.target.value, 10) || 1)
            }
            className="vj-input w-14 text-center tabular-nums"
          />
        </div>
        <button
          onClick={onRemove}
          className="vj-btn vj-btn--danger"
          title="Remove fixture"
        >
          ✕
        </button>
      </div>

      {/* Row 2 — color controls. Only shown for fixtures that have actual
          colour channels. A fog machine's 1-channel profile skips this
          entire row. */}
      {hasColor && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)]">
            color
          </span>
          <div className="inline-flex rounded border border-[color:var(--vj-edge-hot)] overflow-hidden">
            <button
              onClick={() => onColorModeChange("canvas")}
              className={`px-2 py-0.5 text-[11px] uppercase tracking-wider transition-colors ${
                colorMode === "canvas"
                  ? "bg-[color:var(--vj-live)] text-black"
                  : "bg-[color:var(--vj-bg)] text-[color:var(--vj-ink-dim)] hover:text-[color:var(--vj-info)]"
              }`}
            >
              canvas
            </button>
            <button
              onClick={() => onColorModeChange("solid")}
              className={`px-2 py-0.5 text-[11px] uppercase tracking-wider transition-colors ${
                colorMode === "solid"
                  ? "bg-[color:var(--vj-live)] text-black"
                  : "bg-[color:var(--vj-bg)] text-[color:var(--vj-ink-dim)] hover:text-[color:var(--vj-info)]"
              }`}
            >
              solid
            </button>
          </div>
          {colorMode === "solid" && (
            <>
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
                  onSolidColorChange({
                    r: parseInt(hex.slice(1, 3), 16),
                    g: parseInt(hex.slice(3, 5), 16),
                    b: parseInt(hex.slice(5, 7), 16),
                  });
                }}
                className="w-6 h-6 rounded border border-[color:var(--vj-edge-hot)] cursor-pointer bg-transparent"
              />
              <span className="tabular-nums text-[10px] text-[color:var(--vj-ink-dim)]">
                {solidColor.r},{solidColor.g},{solidColor.b}
              </span>
            </>
          )}
          {/* Live RGB preview — tiny swatch reflecting current output. */}
          <div
            className="ml-auto w-6 h-6 rounded border border-[color:var(--vj-edge-hot)]"
            style={{ backgroundColor: `rgb(${r}, ${g}, ${b})` }}
            title={`live rgb: ${r}, ${g}, ${b}`}
          />
        </div>
      )}

      {/* Dimmer row — only for fixtures that actually have a dimmer
          channel. Auto (default) lets the engine compute the value (255
          full-on, 0 during strobe blackouts). Manual lets the user pin
          the brightness via the slider; strobe blackouts do NOT dim a
          manually-driven fixture, since the user is asserting a level.
          Slider capped via vj-range--tight so it never sprawls the row. */}
      {hasDimmer && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)] w-7">
            dim
          </span>
          <div className="inline-flex rounded border border-[color:var(--vj-edge-hot)] overflow-hidden">
            <button
              onClick={() => onDimmerModeChange("auto")}
              className={`px-2 py-0.5 text-[11px] uppercase tracking-wider transition-colors ${
                dimmerMode === "auto"
                  ? "bg-[color:var(--vj-live)] text-black"
                  : "bg-[color:var(--vj-bg)] text-[color:var(--vj-ink-dim)] hover:text-[color:var(--vj-info)]"
              }`}
              title="Engine computes the dimmer (full-on, 0 during strobe blackout)"
            >
              auto
            </button>
            <button
              onClick={() => onDimmerModeChange("manual")}
              className={`px-2 py-0.5 text-[11px] uppercase tracking-wider transition-colors ${
                dimmerMode === "manual"
                  ? "bg-[color:var(--vj-live)] text-black"
                  : "bg-[color:var(--vj-bg)] text-[color:var(--vj-ink-dim)] hover:text-[color:var(--vj-info)]"
              }`}
              title="User-controlled brightness (slider) — strobe blackouts don't apply"
            >
              manual
            </button>
          </div>
          <input
            type="range"
            min={0}
            max={255}
            value={manualDimmer}
            onChange={(e) =>
              onManualDimmerChange(parseInt(e.target.value, 10))
            }
            className="vj-range vj-range--tight"
            style={
              {
                ["--vj-range-fill" as string]: `${(manualDimmer / 255) * 100}%`,
              } as React.CSSProperties
            }
            disabled={dimmerMode !== "manual"}
            title={`Manual dimmer: ${manualDimmer}`}
          />
          <span className="font-mono text-[10px] tabular-nums text-[color:var(--vj-info)] w-8 text-right ml-auto">
            {manualDimmer}
          </span>
        </div>
      )}

      {/* Row 3 — channel-value bars. Tight grid, auto-fit per fixture. */}
      <div
        className="grid gap-1.5"
        style={{
          gridTemplateColumns: `repeat(${fixture.profile.channels.length}, minmax(0, 1fr))`,
        }}
      >
        {fixture.profile.channels.map((kind, i) => {
          const value = values?.[i] ?? 0;
          const label = CHANNEL_LABELS[kind] || kind;
          const barColor = CHANNEL_COLORS[kind] || "bg-neutral-500";
          return (
            <div key={i} className="flex flex-col gap-0.5">
              <div className="flex justify-between items-baseline text-[9px]">
                <span className="text-[color:var(--vj-ink-dim)]">{label}</span>
                <span className="tabular-nums text-[color:var(--vj-ink)]">
                  {value}
                </span>
              </div>
              <div className="h-1 bg-[color:var(--vj-bg)] rounded-full overflow-hidden">
                <div
                  className={`h-full ${barColor} transition-all duration-75`}
                  style={{ width: `${(value / 255) * 100}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Row 4 — audio-reactive strobe / fog-trigger controls. Only for
          fixtures with a channel that hooks into the strobe pipeline
          (strobe or fog — both read from strobeValue). The mode select
          stays on top; thr/max sliders share a row underneath so the
          paired controls read as one cluster. */}
      {(hasStrobe || fixture.profile.channels.includes("fog")) && (
        <div className="flex flex-col gap-1.5 pt-1 border-t border-[color:var(--vj-edge)]">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)] w-7">
              src
            </span>
            <select
              value={strobeMode}
              onChange={(e) => onStrobeModeChange(e.target.value as StrobeMode)}
              className="vj-input text-[11px] flex-1"
              title="Audio feature that drives the strobe/fog channel"
            >
              {STROBE_MODE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center gap-1.5">
              <span className="text-[9px] uppercase tracking-wider text-[color:var(--vj-ink-dim)] w-7">
                thr
              </span>
              <input
                type="range"
                min={0}
                max={100}
                value={strobeThreshold * 100}
                onChange={(e) =>
                  onStrobeThresholdChange(parseInt(e.target.value, 10) / 100)
                }
                className="vj-range flex-1"
                style={
                  {
                    ["--vj-range-fill" as string]: `${strobeThreshold * 100}%`,
                  } as React.CSSProperties
                }
                disabled={strobeMode === "off"}
                title={`Trigger threshold: ${(strobeThreshold * 100).toFixed(0)}%`}
              />
              <span className="font-mono text-[9px] tabular-nums text-[color:var(--vj-info)] w-7 text-right">
                {(strobeThreshold * 100).toFixed(0)}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-[9px] uppercase tracking-wider text-[color:var(--vj-ink-dim)] w-7">
                max
              </span>
              <input
                type="range"
                min={0}
                max={255}
                value={strobeMax}
                onChange={(e) =>
                  onStrobeMaxChange(parseInt(e.target.value, 10))
                }
                className="vj-range flex-1"
                style={
                  {
                    ["--vj-range-fill" as string]: `${(strobeMax / 255) * 100}%`,
                  } as React.CSSProperties
                }
                disabled={strobeMode === "off"}
                title={`Max DMX value: ${strobeMax}`}
              />
              <span className="font-mono text-[9px] tabular-nums text-[color:var(--vj-info)] w-7 text-right">
                {strobeMax}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
