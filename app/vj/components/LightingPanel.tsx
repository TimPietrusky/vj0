"use client";

import { useEffect, useState } from "react";
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
  // Master switch
  enabled: boolean;
  onSetEnabled: (value: boolean) => void;

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
  /** How many fixtures are currently pushing non-zero DMX. Surfaced in the
   *  drawer header so the operator can verify "yes, light is actually
   *  flowing" without staring at the fixture cards. */
  dmxActiveCount: number;
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

  // Fog — folded in because fog is just a DMX channel. Toggle button +
  // intensity slider live as a single section in the master row so the
  // live cue is one glance away (a fog machine left running by accident
  // is bad).
  fogIntensity: number;
  onFogIntensityChange: (v: number) => void;
  onFogToggle: () => void;
  isFogActive: () => boolean;

  /** Dismiss the drawer. Wired to the header × button + Esc. */
  onClose: () => void;
}

/**
 * DMX console — content of the bottom drawer surfaced via the SystemsBar
 * "DMX" chip. The component renders its own header (drawer-style: title +
 * live counter chips + ON/OFF + close), then a horizontal master row
 * (DEVICE / ADD / FOG) that uses the wide drawer surface, then a
 * responsive fixture grid that scales from 1 column on a phone up to 4 on
 * a 1440-wide monitor.
 *
 * NOTE on the "OFF" state: the master toggle silences DMX output but
 * leaves fixture configuration intact, so a visual-only set still has
 * the patch ready when the user re-engages. We still show the master row
 * but greyed out, rather than hiding the body, because hiding it inside
 * a transient drawer would feel like the surface itself disappeared.
 */
export function LightingPanel({
  enabled,
  onSetEnabled,
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
  dmxActiveCount,
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
  fogIntensity,
  onFogIntensityChange,
  onFogToggle,
  isFogActive,
  onClose,
}: LightingPanelProps) {
  // Poll the engine at ~15 Hz so the fog button reflects state even when
  // toggled by the "0" hotkey (no prop change to drive a re-render).
  const [fogActive, setFogActive] = useState(false);
  useEffect(() => {
    if (!enabled) return;
    const tick = () => setFogActive(isFogActive());
    tick();
    const id = window.setInterval(tick, 66);
    return () => window.clearInterval(id);
  }, [isFogActive, enabled]);

  const dimmed = !enabled;

  return (
    <>
      {/* Drawer header — title, live counters, master toggle, close. The
          counters are intentionally telemetry-styled (count chips) rather
          than action buttons so the eye can sweep them at a glance. */}
      <div className="vj-drawer-header">
        <div className="flex items-center gap-3 min-w-0">
          <DmxSigil active={enabled && dmxActiveCount > 0} />
          <div className="flex flex-col min-w-0">
            <span className="font-mono text-[10px] tracking-[0.28em] uppercase text-[color:var(--vj-ink-dim)]">
              Console
            </span>
            <span className="font-mono text-[15px] font-bold tracking-[0.08em] uppercase text-[color:var(--vj-ink)]">
              Lighting · DMX
            </span>
          </div>

          {/* Counter chips — sit inline so the eye reads
              "console / lighting · dmx / [3 fixtures · 2 active]". */}
          <div className="hidden sm:flex items-center gap-1.5 ml-3 font-mono">
            <span
              className="vj-meter-chip"
              title={`${fixtures.length} configured fixture${fixtures.length === 1 ? "" : "s"}`}
            >
              <strong>{fixtures.length}</strong> fixt
            </span>
            {enabled && (
              <span
                className="vj-meter-chip"
                title={`${dmxActiveCount} fixture${dmxActiveCount === 1 ? "" : "s"} actively pushing DMX`}
              >
                <em>{dmxActiveCount}</em> live
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {/* Hotkey hint for closing the drawer — visible when the user
              has the keyboard at the ready, hidden on narrow widths. */}
          <kbd
            className="hidden md:inline-flex items-center justify-center px-1.5 h-5 rounded border border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-bg)] text-[color:var(--vj-ink-dim)] font-mono text-[10px]"
            title="Press Esc to close the DMX console"
          >
            esc
          </kbd>

          <button
            type="button"
            onClick={() => onSetEnabled(!enabled)}
            aria-pressed={enabled}
            className={`vj-icon-btn ${enabled ? "vj-icon-btn--on" : "vj-icon-btn--off"}`}
            title={
              enabled
                ? "Lighting active — click to disable DMX"
                : "Lighting disabled — click to re-enable"
            }
          >
            {enabled ? "ON" : "OFF"}
          </button>

          <button
            type="button"
            onClick={onClose}
            className="vj-icon-btn"
            title="Close DMX console (Esc)"
            aria-label="Close DMX console"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Drawer body — scrolls. Master row first (DEVICE / ADD / FOG),
          then the fixture grid. The grid is a CSS auto-fill grid with
          minmax(310px, 1fr) so it naturally lays 1 / 2 / 3 / 4 columns
          across phone → ultrawide without media-query gymnastics. */}
      <div
        className={`vj-drawer-body font-mono text-xs flex flex-col gap-4 ${dimmed ? "opacity-60" : ""}`}
      >
        {/* Master row — three labeled sections side by side. Collapses to
            a stack on narrow widths (<800px). */}
        <div className="grid grid-cols-1 lg:grid-cols-[auto_1fr_minmax(280px,1fr)] gap-3 mt-2">
          <div className="vj-section">
            <span className="vj-section-label">Device</span>
            <DmxControls
              status={dmxStatus}
              supported={dmxSupported}
              onConnect={onDmxConnect}
              onDisconnect={onDmxDisconnect}
              onReconnect={onDmxReconnect}
            />
          </div>

          <div className="vj-section">
            <span className="vj-section-label">Add fixture</span>
            <FixtureSelector
              selectedProfileId={selectedProfileId}
              onProfileSelect={onProfileSelect}
              onAdd={onAddFixture}
            />
          </div>

          <div className="vj-section">
            <span className="vj-section-label">Fog · cue</span>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={onFogToggle}
                disabled={!enabled}
                className={`
                  rounded-md border px-3 py-2 font-mono text-[11px] uppercase tracking-[0.22em]
                  transition-all w-24 text-center disabled:opacity-50
                  ${
                    fogActive
                      ? "border-[color:var(--vj-live)] text-[color:var(--vj-live)] bg-[color-mix(in_srgb,var(--vj-live)_22%,transparent)] shadow-[0_0_22px_-4px_var(--vj-live)]"
                      : "border-[color:var(--vj-accent)] text-[color:var(--vj-accent)] bg-[color-mix(in_srgb,var(--vj-accent)_10%,transparent)] hover:bg-[color-mix(in_srgb,var(--vj-accent)_22%,transparent)] hover:shadow-[0_0_18px_-6px_var(--vj-accent)]"
                  }
                `}
                title={
                  fogActive
                    ? "Fog ON — click to stop"
                    : "Click to turn fog ON (hotkey: 0)"
                }
              >
                {fogActive ? "● fog" : "○ fog"}
              </button>
              <input
                type="range"
                min={0}
                max={255}
                step={1}
                value={fogIntensity}
                onChange={(e) => onFogIntensityChange(Number(e.target.value))}
                disabled={!enabled}
                className="vj-range flex-1"
                style={
                  {
                    ["--vj-range-fill" as string]: `${(fogIntensity / 255) * 100}%`,
                  } as React.CSSProperties
                }
                title="DMX value on the fog channel while fog is on (0–255)"
              />
              <div className="flex items-center gap-1.5 shrink-0">
                <span className="font-mono text-[12px] tabular-nums text-[color:var(--vj-info)] w-8 text-right">
                  {fogIntensity}
                </span>
                <kbd
                  className="inline-flex items-center justify-center w-5 h-5 rounded border border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-bg)] text-[color:var(--vj-info)] text-[10px]"
                  title="Hotkey: 0"
                >
                  0
                </kbd>
              </div>
            </div>
          </div>
        </div>

        {/* Fixture grid — responsive auto-fill so wide drawers get
            multi-column layout instead of a single tall stack. The
            grid stays 1-col below ~640px (phones), 2-col on tablets,
            3-col on standard laptops, 4-col on ultrawides. */}
        {fixtures.length === 0 ? (
          <div className="border border-dashed border-[color:var(--vj-edge-hot)] rounded-md py-10 px-4 text-center">
            <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-[color:var(--vj-ink-dim)] mb-1">
              No fixtures patched
            </div>
            <div className="font-mono text-[12px] text-[color:var(--vj-ink-dim)]">
              Pick a profile in <span className="text-[color:var(--vj-accent)]">Add fixture</span>{" "}
              above and click <span className="text-[color:var(--vj-live)]">+ add</span>.
            </div>
          </div>
        ) : (
          <div
            className="grid gap-3"
            style={{
              gridTemplateColumns:
                "repeat(auto-fill, minmax(min(340px, 100%), 1fr))",
            }}
          >
            {fixtures.map((fixture) => (
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
                onStrobeMaxChange={(m) =>
                  onFixtureStrobeMaxChange(fixture.id, m)
                }
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
            ))}
          </div>
        )}
      </div>
    </>
  );
}

/**
 * Tiny lighting glyph for the drawer header — three projector cones of
 * light. Uses currentColor; goes magenta at rest, magenta + a faint live
 * pulse when fixtures are actively pushing DMX. NOT shown on the
 * SystemsBar chip — that one already has the dot indicator; this one
 * exists so the drawer header has its own piece of identity.
 */
function DmxSigil({ active }: { active: boolean }) {
  return (
    <div
      className="relative shrink-0 w-7 h-7 grid place-items-center"
      style={{ color: "var(--vj-accent)" }}
      aria-hidden
    >
      <svg viewBox="0 0 24 24" width="26" height="26" fill="none">
        <defs>
          <linearGradient id="vj-dmx-cone" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="currentColor" stopOpacity="0.9" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0.0" />
          </linearGradient>
        </defs>
        {/* Three cones fanning down from the truss line */}
        <path d="M3 4 L21 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M6 5 L3 21 L9 21 Z" fill="url(#vj-dmx-cone)" />
        <path d="M12 5 L9.5 21 L14.5 21 Z" fill="url(#vj-dmx-cone)" />
        <path d="M18 5 L15 21 L21 21 Z" fill="url(#vj-dmx-cone)" />
        <circle cx="6" cy="5" r="1.4" fill="currentColor" />
        <circle cx="12" cy="5" r="1.4" fill="currentColor" />
        <circle cx="18" cy="5" r="1.4" fill="currentColor" />
      </svg>
      {active && (
        <span
          className="absolute inset-0 rounded-full pointer-events-none"
          style={{
            boxShadow: "0 0 16px 1px var(--vj-live)",
            animation: "vj-sys-btn-pulse 2.6s ease-in-out infinite",
          }}
        />
      )}
    </div>
  );
}
