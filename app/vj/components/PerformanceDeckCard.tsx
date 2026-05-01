"use client";

import type { PromptPreset } from "@/src/lib/stores/ai-settings-store";
import { HotkeyBoard } from "./HotkeyBoard";
import { PanelHeader } from "./PanelHeader";

interface PerformanceDeckCardProps {
  presets: PromptPreset[];
  activePrompt: string;
  onFirePreset: (prompt: string) => void;
  onUpdatePreset: (index: number, patch: Partial<PromptPreset>) => void;
  onRandom: () => void;
  onFireFog: () => void;
  /** Klein-only; when undefined the alpha row is hidden inside HotkeyBoard. */
  alpha?: number;
  onAlphaDelta?: (delta: number) => void;
}

/**
 * Performance Deck — promoted from inside AI Console to its own card.
 * The 9 preset caps + global triggers are the most-touched surface
 * mid-set, so they get top-level visual presence with a hot accent
 * border (.vj-panel--hot) telegraphing "live trigger surface."
 */
export function PerformanceDeckCard({
  presets,
  activePrompt,
  onFirePreset,
  onUpdatePreset,
  onRandom,
  onFireFog,
  alpha,
  onAlphaDelta,
}: PerformanceDeckCardProps) {
  return (
    <div className="vj-panel vj-panel--hot p-2 flex flex-col gap-2">
      <PanelHeader
        title="Performance Deck"
        actions={
          <span
            className="text-[9px] uppercase tracking-wider font-mono text-[color:var(--vj-ink-dim)]"
            title="1-9 fire preset · Space re-rolls seed · arrows nudge klein α · 0 fog"
          >
            live triggers
          </span>
        }
      />
      <HotkeyBoard
        presets={presets}
        activePrompt={activePrompt}
        onFirePreset={onFirePreset}
        onUpdatePreset={onUpdatePreset}
        onRandom={onRandom}
        onFireFog={onFireFog}
        alpha={alpha}
        onAlphaDelta={onAlphaDelta}
      />
    </div>
  );
}
