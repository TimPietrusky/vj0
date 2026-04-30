"use client";

import { useEffect, useState } from "react";
import { PanelHeader } from "./PanelHeader";

interface FogControlProps {
  intensity: number;
  onSetIntensity: (v: number) => void;
  /** Toggles fog on/off. Reads latest intensity at fire-time inside VJApp. */
  onToggle: () => void;
  /** Polled by the card so the button reflects engine state even if the
   *  toggle was fired by the hotkey rather than this button. */
  isActive: () => boolean;
}

/**
 * Dedicated FOG control card. Click the button (or hit hotkey "0") to toggle
 * fog on/off. Intensity is the DMX value used while ON — read fresh on each
 * toggle so adjustments during an active burst only apply to the next press
 * (toggle off+on to commit a new level mid-set). Kept in its own card so the
 * state is one glance away — a fog machine left running by accident is bad.
 */
export function FogControl({
  intensity,
  onSetIntensity,
  onToggle,
  isActive,
}: FogControlProps) {
  // Poll the engine at ~15 Hz so the button state reflects the toggle even
  // when triggered by the hotkey (no prop change to drive a re-render).
  const [active, setActive] = useState(false);
  useEffect(() => {
    const tick = () => setActive(isActive());
    tick();
    const id = window.setInterval(tick, 66);
    return () => window.clearInterval(id);
  }, [isActive]);

  return (
    <div className="vj-panel p-2 flex flex-col gap-2">
      <PanelHeader
        title="Fog"
        actions={
          <span
            className="flex items-center gap-1 text-[9px] uppercase tracking-wider font-mono text-[color:var(--vj-ink-dim)]"
            title="Hotkey"
          >
            <kbd className="inline-flex items-center justify-center w-5 h-5 rounded border border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-bg)] text-[color:var(--vj-info)] text-[10px]">
              0
            </kbd>
          </span>
        }
      />

      {/* Toggle + intensity on a single row so the live cue is one glance. */}
      <div className="grid grid-cols-[auto_1fr_auto] items-center gap-2">
        <button
          type="button"
          onClick={onToggle}
          className={`
            rounded-md border px-3 py-2 font-mono text-[11px] uppercase tracking-[0.2em]
            transition-colors w-24 text-center
            ${
              active
                ? "border-[color:var(--vj-live)] text-[color:var(--vj-live)] bg-[color-mix(in_srgb,var(--vj-live)_20%,transparent)] shadow-[0_0_18px_-4px_var(--vj-live)]"
                : "border-[color:var(--vj-accent)] text-[color:var(--vj-accent)] bg-[color-mix(in_srgb,var(--vj-accent)_8%,transparent)] hover:bg-[color-mix(in_srgb,var(--vj-accent)_18%,transparent)]"
            }
          `}
          title={active ? "Fog is ON — click to stop" : "Click to turn fog ON (hotkey: 0)"}
        >
          {active ? "● on" : "○ off"}
        </button>

        <input
          type="range"
          min={0}
          max={255}
          step={1}
          value={intensity}
          onChange={(e) => onSetIntensity(Number(e.target.value))}
          className="vj-range vj-range--tight"
          style={
            {
              ["--vj-range-fill" as string]: `${(intensity / 255) * 100}%`,
            } as React.CSSProperties
          }
          title="DMX value on the fog channel while fog is on (0–255)"
        />
        <span className="font-mono text-[11px] tabular-nums text-[color:var(--vj-info)] w-10 text-right">
          {intensity}
        </span>
      </div>
    </div>
  );
}
