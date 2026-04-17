"use client";

import { useEffect, useRef, useState } from "react";
import type { PromptPreset } from "@/src/lib/stores/ai-settings-store";

interface PresetChipsProps {
  presets: PromptPreset[];
  activePrompt: string;
  onFire: (prompt: string) => void;
  onEdit: (index: number, patch: Partial<PromptPreset>) => void;
}

/**
 * Hotkey preset chips (1-9).
 *  - Click chip body → fires the preset prompt immediately (same as pressing number key).
 *  - Hover reveals an edit pencil in the top-right of the chip.
 *  - Click the pencil → inline editor for label + full prompt text.
 */
export function PresetChips({
  presets,
  activePrompt,
  onFire,
  onEdit,
}: PresetChipsProps) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);

  return (
    <div className="flex flex-wrap gap-1.5">
      {presets.map((p, i) => {
        const active = p.prompt === activePrompt;
        const editing = editingIndex === i;
        if (editing) {
          return (
            <PresetEditor
              key={i}
              index={i}
              preset={p}
              onSave={(patch) => {
                onEdit(i, patch);
                setEditingIndex(null);
              }}
              onCancel={() => setEditingIndex(null)}
            />
          );
        }
        return (
          <PresetChip
            key={i}
            index={i}
            preset={p}
            active={active}
            onFire={() => onFire(p.prompt)}
            onEditRequest={() => setEditingIndex(i)}
          />
        );
      })}
    </div>
  );
}

function PresetChip({
  index,
  preset,
  active,
  onFire,
  onEditRequest,
}: {
  index: number;
  preset: PromptPreset;
  active: boolean;
  onFire: () => void;
  onEditRequest: () => void;
}) {
  return (
    <div
      className={`
        group relative flex items-center gap-1.5 rounded
        border px-2 py-1 pr-6 transition-all
        font-mono text-[11px]
        ${
          active
            ? "border-[color:var(--vj-accent)] bg-[color-mix(in_srgb,var(--vj-accent)_15%,transparent)] text-white shadow-[0_0_14px_-4px_var(--vj-accent)]"
            : "border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-panel-2)] text-[color:var(--vj-ink)] hover:border-[color:var(--vj-info)] hover:text-[color:var(--vj-info)] hover:bg-[color-mix(in_srgb,var(--vj-info)_10%,transparent)]"
        }
      `}
      title={`${index + 1} · ${preset.label} — ${preset.prompt}`}
    >
      <button
        type="button"
        onClick={onFire}
        className="flex items-center gap-1.5 focus:outline-none"
      >
        <span
          className={`
            inline-flex items-center justify-center
            w-4 h-4 rounded-sm text-[10px] font-bold tabular-nums
            ${
              active
                ? "bg-[color:var(--vj-accent)] text-black"
                : "bg-[color:var(--vj-bg)] text-[color:var(--vj-info)] border border-[color:var(--vj-edge-hot)] group-hover:border-[color:var(--vj-info)]"
            }
          `}
        >
          {index + 1}
        </span>
        <span className="tracking-wide">{preset.label}</span>
      </button>

      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onEditRequest();
        }}
        aria-label={`Edit preset ${index + 1}`}
        className="
          absolute right-1 top-1/2 -translate-y-1/2
          w-5 h-5 rounded-sm
          flex items-center justify-center
          opacity-0 group-hover:opacity-100
          text-[color:var(--vj-ink-dim)] hover:text-[color:var(--vj-warn)]
          hover:bg-[color:var(--vj-bg)]
          transition-opacity
        "
        title="Edit label & prompt"
      >
        <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden>
          <path
            d="M12 2l2 2-8 8H4v-2l8-8z"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
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
}) {
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
    <div className="flex flex-col gap-1 rounded border border-[color:var(--vj-warn)] bg-[color:var(--vj-panel-2)] p-2 shadow-[0_0_18px_-4px_var(--vj-warn)] w-full max-w-md">
      <div className="flex items-center gap-2">
        <span className="inline-flex w-4 h-4 items-center justify-center rounded-sm bg-[color:var(--vj-warn)] text-black text-[10px] font-bold">
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
          className="vj-input text-[11px] w-28"
        />
        <div className="ml-auto flex gap-1">
          <button
            type="button"
            onClick={save}
            className="vj-btn vj-btn--live py-1"
            title="Save (Enter)"
          >
            Save
          </button>
          <button
            type="button"
            onClick={onCancel}
            className="vj-btn py-1"
            title="Cancel (Esc)"
          >
            Cancel
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
        className="vj-input w-full resize-y"
        placeholder="full prompt text (⌘/Ctrl+Enter to save)"
      />
    </div>
  );
}
