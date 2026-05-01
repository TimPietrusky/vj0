"use client";

import { useEffect, useState } from "react";

interface PromptDockProps {
  activePrompt: string;
  /** Commits the typed prompt as-is (Enter in input). */
  onSetPrompt: (p: string) => void;
}

/**
 * Prompt input + send. Stripped to a single row: placeholder doubles as the
 * "Prompt" label, the live/dirty state lives as a tiny status pill on the
 * right of the input itself instead of a separate caption row above. Saves
 * a whole line of vertical space and the input still reads as the prompt
 * field — context (preview canvas right above) makes the missing label
 * unambiguous. Enter commits the draft, Escape reverts.
 */
export function PromptDock({ activePrompt, onSetPrompt }: PromptDockProps) {
  const [draft, setDraft] = useState(activePrompt);
  useEffect(() => {
    setDraft(activePrompt);
  }, [activePrompt]);
  const dirty = draft !== activePrompt;

  return (
    <div className="flex items-stretch gap-1.5">
      <div className="relative flex-1">
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
          placeholder="prompt — describe the visual"
          className="vj-input w-full font-mono text-xs py-2"
          // Inline to win over the .vj-input shorthand `padding: 0.375rem
          // 0.625rem` which (same specificity, defined after Tailwind in
          // source order) eats the `pr-*` utility. Reserves room for the
          // ✓ live / ● ↵ pill so the prompt text never scrolls under it.
          style={{ paddingRight: "3.25rem" }}
        />
        {/* Inline live/dirty pill — sits inside the input on the right so
            it doesn't claim its own row. Switches color on dirty (warn
            yellow) vs synced (live emerald) so a glance tells you whether
            the projector is showing what's typed. */}
        <span
          className={`pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[9px] font-mono uppercase tracking-wider tabular-nums ${
            dirty
              ? "text-[color:var(--vj-warn)]"
              : "text-[color:var(--vj-live)]"
          }`}
          aria-live="polite"
        >
          {dirty ? "● ↵" : "✓ live"}
        </span>
      </div>
      <button
        type="button"
        onClick={() => onSetPrompt(draft)}
        disabled={!dirty}
        className="vj-btn vj-btn--accent"
        title="Send prompt (Enter)"
      >
        send ↵
      </button>
    </div>
  );
}
