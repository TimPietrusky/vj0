"use client";

import { Popover } from "./Popover";
import {
  OUTPUT_PRESETS,
  findOutputPreset,
} from "@/src/lib/stores/ai-settings-store";

interface ResPickerProps {
  width: number;
  height: number;
  onPick: (w: number, h: number) => void;
}

/**
 * Compact resolution chip — shows current size, opens a small grid of presets.
 * Replaces the verbose <select> and lets the user switch in two clicks.
 */
export function ResPicker({ width, height, onPick }: ResPickerProps) {
  const current = findOutputPreset(width, height);
  return (
    <Popover
      width={300}
      align="right"
      trigger={({ open, toggle }) => (
        <button
          type="button"
          onClick={toggle}
          className={`
            flex items-center gap-1.5 px-2 py-1 rounded
            border font-mono text-[10px] uppercase tracking-wider transition-colors
            ${
              open
                ? "border-[color:var(--vj-info)] bg-[color-mix(in_srgb,var(--vj-info)_12%,transparent)]"
                : "border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-panel-2)] hover:border-[color:var(--vj-info)]"
            }
          `}
          title="Output resolution"
        >
          <span className="text-[color:var(--vj-ink-dim)]">res</span>
          <span className="tabular-nums normal-case text-[color:var(--vj-info)]">
            {width}×{height}
          </span>
        </button>
      )}
    >
      <div className="flex flex-col gap-2">
        <div className="text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)]">
          Output resolution
        </div>
        <div className="grid grid-cols-3 gap-1.5">
          {OUTPUT_PRESETS.map((p) => {
            const active = current?.id === p.id;
            // After the v5 preset curation every non-square preset is either
            // exact 16:9 (horizontal) or exact 9:16 (vertical). Pick by
            // orientation; no need for the "~" approximation tilde anymore.
            const ratio =
              p.w === p.h ? "1:1" : p.w > p.h ? "16:9" : "9:16";
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => onPick(p.w, p.h)}
                className={`
                  flex flex-col items-start rounded border px-2 py-1.5
                  text-left font-mono transition-colors
                  ${
                    active
                      ? "border-[color:var(--vj-accent)] bg-[color-mix(in_srgb,var(--vj-accent)_15%,transparent)] text-white"
                      : "border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-bg)] text-[color:var(--vj-ink)] hover:border-[color:var(--vj-info)]"
                  }
                `}
              >
                <span className="text-[11px] tabular-nums">
                  {p.w}×{p.h}
                </span>
                <span className="text-[9px] uppercase tracking-wider text-[color:var(--vj-ink-dim)]">
                  {ratio}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </Popover>
  );
}
