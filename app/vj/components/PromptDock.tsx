"use client";

import { useEffect, useState } from "react";
import { PresetChips } from "./PresetChips";
import type { PromptPreset } from "@/src/lib/stores/ai-settings-store";

interface PromptDockProps {
  activePrompt: string;
  /** Commits the typed prompt as-is (Enter in input). */
  onSetPrompt: (p: string) => void;
  /** Fires a preset (chip click). Typically also re-rolls the seed. */
  onFirePreset?: (p: string) => void;
  presets: PromptPreset[];
  onUpdatePreset: (index: number, patch: Partial<PromptPreset>) => void;
}

/**
 * Prompt input with draft/live indicator, send button, and 1-9 preset chips.
 * Enter commits the draft. Escape reverts. Click a chip fires that preset.
 */
export function PromptDock({
  activePrompt,
  onSetPrompt,
  onFirePreset,
  presets,
  onUpdatePreset,
}: PromptDockProps) {
  const fire = onFirePreset ?? onSetPrompt;
  const [draft, setDraft] = useState(activePrompt);
  useEffect(() => {
    setDraft(activePrompt);
  }, [activePrompt]);
  const dirty = draft !== activePrompt;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <div className="vj-panel-title">Prompt</div>
        <span
          className={`text-[10px] font-mono tabular-nums ${
            dirty
              ? "text-[color:var(--vj-warn)]"
              : "text-[color:var(--vj-live)]"
          }`}
        >
          {dirty ? "● unsaved — ↵ to send" : "✓ live"}
        </span>
      </div>
      <div className="flex items-stretch gap-2">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              onSetPrompt(draft);
            } else if (e.key === "Escape") {
              e.preventDefault();
              setDraft(activePrompt);
            }
          }}
          placeholder="describe the visual — press Enter to send"
          className="vj-input flex-1 font-mono text-xs py-2"
        />
        <button
          type="button"
          onClick={() => onSetPrompt(draft)}
          disabled={!dirty}
          className="vj-btn vj-btn--accent"
          title="Enter"
        >
          Send ↵
        </button>
      </div>
      <PresetChips
        presets={presets}
        activePrompt={activePrompt}
        onFire={fire}
        onEdit={onUpdatePreset}
      />
    </div>
  );
}
