"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { PromptPreset } from "@/src/lib/stores/ai-settings-store";

interface HotkeyBoardProps {
  presets: PromptPreset[];
  activePrompt: string;
  onFirePreset: (prompt: string) => void;
  onUpdatePreset: (index: number, patch: Partial<PromptPreset>) => void;

  onRandom: () => void;
  onHideUi: () => void;
  /** Fires a manual fog burst. Bound to the "0" key and its on-screen cap. */
  onFireFog?: () => void;

  /** Klein-only controls; when undefined the alpha row is hidden. */
  alpha?: number;
  onAlphaDelta?: (delta: number) => void;
}

/**
 * Visual keyboard with every hotkey as a uniform cap. Click a cap to fire
 * the same action as the hotkey. Preset caps (1-9) reveal an edit pencil on
 * hover so labels and prompts can be tweaked inline.
 *
 * All caps share the same width/height so the board reads as a real keyboard
 * row, not a list of mismatched buttons. Two rows:
 *   1) preset triggers (1-9)
 *   2) global actions (Space, arrows, H)
 */
export function HotkeyBoard({
  presets,
  activePrompt,
  onFirePreset,
  onUpdatePreset,
  onRandom,
  onHideUi,
  onFireFog,
  alpha,
  onAlphaDelta,
}: HotkeyBoardProps) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);

  return (
    <div className="flex flex-col gap-1.5">
      {/* Row 1 — preset hotkeys. Equal-width cells via grid-cols-9. */}
      <div className="grid grid-cols-9 gap-1.5">
        {presets.map((p, i) =>
          editingIndex === i ? (
            <PresetEditor
              key={i}
              index={i}
              preset={p}
              onSave={(patch) => {
                onUpdatePreset(i, patch);
                setEditingIndex(null);
              }}
              onCancel={() => setEditingIndex(null)}
            />
          ) : (
            <Cap
              key={i}
              keyLabel={`${i + 1}`}
              caption={p.label}
              active={p.prompt === activePrompt}
              onClick={() => onFirePreset(p.prompt)}
              onEdit={() => setEditingIndex(i)}
              title={`${i + 1} · ${p.label} — ${p.prompt}`}
            />
          )
        )}
      </div>

      {/* Row 2 — global hotkeys. Same cell size as row 1 (grid-cols-9 again
          so columns line up vertically with the preset row above). */}
      <div className="grid grid-cols-9 gap-1.5">
        <div className="col-span-2">
          <Cap
            keyLabel="␣"
            keyText="space"
            caption="random"
            onClick={onRandom}
            title="Pick a random preset (re-rolls seed)"
            wide
          />
        </div>

        {alpha !== undefined && onAlphaDelta && (
          <>
            <Cap
              keyLabel="←"
              caption={`α −0.01 (${alpha.toFixed(2)})`}
              onClick={() => onAlphaDelta(-0.01)}
              title="Decrease klein alpha by 0.01"
            />
            <Cap
              keyLabel="→"
              caption="α +0.01"
              onClick={() => onAlphaDelta(0.01)}
              title="Increase klein alpha by 0.01"
            />
            <Cap
              keyLabel="↓"
              caption="α −0.02"
              onClick={() => onAlphaDelta(-0.02)}
              title="Decrease klein alpha by 0.02"
            />
            <Cap
              keyLabel="↑"
              caption="α +0.02"
              onClick={() => onAlphaDelta(0.02)}
              title="Increase klein alpha by 0.02"
            />
          </>
        )}

        {onFireFog && (
          <Cap
            keyLabel="0"
            caption="fog"
            onClick={onFireFog}
            title="Fire a fog burst (hotkey: 0)"
          />
        )}

        <Cap
          keyLabel="H"
          caption="hide ui"
          onClick={onHideUi}
          title="Toggle fullscreen output (no UI)"
        />
      </div>
    </div>
  );
}

function Cap({
  keyLabel,
  keyText,
  caption,
  active,
  onClick,
  onEdit,
  title,
  wide,
}: {
  keyLabel: string;
  keyText?: string;
  caption: string;
  active?: boolean;
  onClick: () => void;
  onEdit?: () => void;
  title?: string;
  wide?: boolean;
}) {
  return (
    <div
      className={`
        group relative flex flex-col items-center justify-between
        rounded-md border px-1 py-1.5 transition-colors
        ${
          active
            ? "border-[color:var(--vj-accent)] bg-[color-mix(in_srgb,var(--vj-accent)_18%,transparent)] shadow-[0_0_14px_-4px_var(--vj-accent)]"
            : "border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-panel-2)] hover:border-[color:var(--vj-info)]"
        }
      `}
      title={title}
    >
      <button
        type="button"
        onClick={onClick}
        className="w-full flex flex-col items-center gap-1 focus:outline-none"
      >
        <span
          className={`
            inline-flex items-center justify-center
            ${wide ? "min-w-[44px] px-2" : "w-7"} h-7 rounded
            text-[13px] font-bold tabular-nums font-mono
            ${
              active
                ? "bg-[color:var(--vj-accent)] text-black"
                : "bg-[color:var(--vj-bg)] text-[color:var(--vj-info)] border border-[color:var(--vj-edge-hot)] group-hover:border-[color:var(--vj-info)]"
            }
          `}
        >
          {keyText ?? keyLabel}
        </span>
        <span className="text-[9px] uppercase tracking-wider font-mono text-[color:var(--vj-ink-dim)] truncate w-full text-center">
          {caption}
        </span>
      </button>

      {onEdit && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onEdit();
          }}
          aria-label="Edit preset"
          className="
            absolute right-0.5 top-0.5
            w-4 h-4 rounded
            flex items-center justify-center
            opacity-0 group-hover:opacity-100
            text-[color:var(--vj-ink-dim)] hover:text-[color:var(--vj-warn)]
            hover:bg-[color:var(--vj-bg)] transition-opacity
          "
          title="Edit label & prompt"
        >
          <svg width="9" height="9" viewBox="0 0 16 16" fill="none" aria-hidden>
            <path
              d="M12 2l2 2-8 8H4v-2l8-8z"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      )}
    </div>
  );
}

function PresetEditor({
  index,
  preset,
  onSave,
  onCancel,
}: {
  index: number;
  preset: PromptPreset;
  onSave: (patch: Partial<PromptPreset>) => void;
  onCancel: () => void;
}): ReactNode {
  const [label, setLabel] = useState(preset.label);
  const [prompt, setPrompt] = useState(preset.prompt);
  const labelRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    labelRef.current?.select();
  }, []);

  const save = () => {
    const trimmed = label.trim() || preset.label;
    onSave({ label: trimmed.slice(0, 16), prompt: prompt.trim() });
  };

  return (
    <div className="col-span-9 rounded border border-[color:var(--vj-warn)] bg-[color:var(--vj-panel-2)] p-2 shadow-[0_0_18px_-4px_var(--vj-warn)] flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <span className="inline-flex w-5 h-5 items-center justify-center rounded bg-[color:var(--vj-warn)] text-black text-[10px] font-bold font-mono">
          {index + 1}
        </span>
        <input
          ref={labelRef}
          value={label}
          onChange={(e) => setLabel(e.target.value.slice(0, 16))}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              save();
            } else if (e.key === "Escape") {
              e.preventDefault();
              onCancel();
            }
          }}
          placeholder="label"
          maxLength={16}
          className="vj-input text-[11px] w-32"
        />
        <div className="ml-auto flex gap-1">
          <button
            type="button"
            onClick={save}
            className="vj-btn vj-btn--live py-1"
            title="Save (Enter)"
          >
            save
          </button>
          <button
            type="button"
            onClick={onCancel}
            className="vj-btn py-1"
            title="Cancel (Esc)"
          >
            cancel
          </button>
        </div>
      </div>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            save();
          } else if (e.key === "Escape") {
            e.preventDefault();
            onCancel();
          }
        }}
        rows={2}
        className="vj-input w-full resize-y text-[11px]"
        placeholder="full prompt text (⌘/Ctrl+Enter to save)"
      />
    </div>
  );
}
