"use client";

import type { PromptPreset } from "@/src/lib/stores/ai-settings-store";
import { HotkeyBoard } from "./HotkeyBoard";

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
 * Performance Deck — the 9 preset caps + global triggers.
 * Most-touched surface mid-set, no chrome needed.
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
    <div className="vj-panel p-2">
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
