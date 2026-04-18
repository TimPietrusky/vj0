"use client";

import { useEffect, useState } from "react";

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
 * Dedicated FOG control card. Click the big button (or hit hotkey "0") to
 * toggle fog on; click again to turn it off. Intensity slider controls the
 * DMX value while ON — it's read fresh on every toggle so adjustments during
 * an active burst don't suddenly change the stream (you'd need to toggle
 * off+on to apply). Kept in its own card so the state is one glance away
 * during a set — a fog machine left running by accident is bad.
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
      <div className="flex items-center justify-between">
        <span className="vj-panel-title">Fog</span>
        <span
          className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-mono text-[color:var(--vj-ink-dim)]"
          title="Hotkey"
        >
          <kbd className="inline-flex items-center justify-center min-w-[20px] px-1 rounded border border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-bg)] text-[color:var(--vj-info)] text-[10px]">
            0
          </kbd>
          toggle
        </span>
      </div>

      <button
        type="button"
        onClick={onToggle}
        className={`
          w-full rounded-md border py-3 font-mono text-[14px] uppercase tracking-[0.2em]
          transition-colors
          ${
            active
              ? "border-[color:var(--vj-live)] text-[color:var(--vj-live)] bg-[color-mix(in_srgb,var(--vj-live)_20%,transparent)] shadow-[0_0_24px_-4px_var(--vj-live)]"
              : "border-[color:var(--vj-accent)] text-[color:var(--vj-accent)] bg-[color-mix(in_srgb,var(--vj-accent)_8%,transparent)] hover:bg-[color-mix(in_srgb,var(--vj-accent)_18%,transparent)] hover:shadow-[0_0_18px_-4px_var(--vj-accent)]"
          }
        `}
        title={active ? "Fog is ON — click to stop" : "Click to turn fog ON (hotkey: 0)"}
      >
        {active ? "● fog on" : "○ fog off"}
      </button>

      <div className="flex items-center gap-2">
        <span className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)] w-16">
          intensity
        </span>
        <input
          type="range"
          min={0}
          max={255}
          step={1}
          value={intensity}
          onChange={(e) => onSetIntensity(Number(e.target.value))}
          className="vj-range flex-1"
          style={
            {
              ["--vj-range-fill" as string]: `${(intensity / 255) * 100}%`,
            } as React.CSSProperties
          }
          title="DMX value on the fog channel while fog is on (0–255)"
        />
        <span className="font-mono text-[11px] tabular-nums text-[color:var(--vj-info)] w-14 text-right">
          {intensity}
        </span>
      </div>
    </div>
  );
}
