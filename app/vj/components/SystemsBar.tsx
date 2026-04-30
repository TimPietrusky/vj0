"use client";

import type { AiTransportStatus } from "@/src/lib/ai/transport";
import {
  AI_BACKEND_LABELS,
  type AiBackend,
} from "@/src/lib/stores/ai-settings-store";

type Status = "idle" | "requesting" | "running" | "error";
type DmxStatus = "disconnected" | "connecting" | "connected" | "unsupported";

interface SystemsBarProps {
  audioStatus: Status;
  audioDeviceLabel?: string;
  aiStatus: AiTransportStatus;
  aiBackend: AiBackend;
  aiGenTimeMs: number | null;
  dmxStatus: DmxStatus;
  dmxFixtureCount: number;
  dmxActiveCount: number;
  /** Master DMX/lighting switch — when false the DMX block goes muted "off". */
  lightingEnabled: boolean;
  onHideUi: () => void;
  hideUi: boolean;
}

/**
 * Single horizontal systems bar — reads top-to-bottom in 0.5s.
 * [AUDIO] [VISUAL] [AI] [DMX] on the left, live-mode toggles on the right.
 */
export function SystemsBar({
  audioStatus,
  audioDeviceLabel,
  aiStatus,
  aiBackend,
  aiGenTimeMs,
  dmxStatus,
  dmxFixtureCount,
  dmxActiveCount,
  lightingEnabled,
  onHideUi,
  hideUi,
}: SystemsBarProps) {
  const audioTone = audioStatus === "error"
    ? "error"
    : audioStatus === "running"
    ? "live"
    : audioStatus === "requesting"
    ? "info"
    : "info";

  const aiTone: Tone =
    aiStatus === "connected"
      ? "live"
      : aiStatus === "error"
      ? "error"
      : aiStatus === "connecting"
      ? "info"
      : "info";

  const dmxTone: Tone = !lightingEnabled
    ? "muted"
    : dmxStatus === "connected"
    ? "live"
    : dmxStatus === "unsupported"
    ? "error"
    : dmxStatus === "connecting"
    ? "info"
    : "info";

  return (
    <header className="sticky top-0 z-30 w-full border-b border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-panel)]/95 backdrop-blur-sm">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-1 font-mono text-[10px] uppercase tracking-wider">
        <BrandSigil />

        <SysBlock
          label="Audio"
          tone={audioTone}
          value={
            audioStatus === "running"
              ? audioDeviceLabel
                ? trimLabel(audioDeviceLabel)
                : "live"
              : audioStatus === "requesting"
              ? "starting…"
              : audioStatus === "error"
              ? "error"
              : "idle"
          }
        />

        <SysBlock
          label={aiBackend}
          tone={aiTone}
          value={formatAi(aiStatus, aiGenTimeMs)}
          title={AI_BACKEND_LABELS[aiBackend]}
        />

        <SysBlock
          label="DMX"
          tone={dmxTone}
          value={
            !lightingEnabled
              ? "off"
              : dmxStatus === "unsupported"
              ? "no webusb"
              : dmxStatus === "connected"
              ? `${dmxActiveCount}/${dmxFixtureCount}`
              : dmxStatus
          }
        />

        <div className="ml-auto flex items-center gap-2">
          <span
            className="text-[9px] text-[color:var(--vj-ink-dim)] normal-case tracking-normal hidden md:inline"
            title="Hotkeys: 1-9 fire preset (re-rolls seed) · Space random preset (re-rolls seed) · ↑↓ klein alpha ±0.02 · ←→ klein alpha ±0.01 · H hide UI"
          >
            <span className="text-[color:var(--vj-info)]">1-9</span> presets ·{" "}
            <span className="text-[color:var(--vj-info)]">Space</span> random ·{" "}
            <span className="text-[color:var(--vj-info)]">H</span> hide
          </span>
          <a
            href="/vj/stage"
            target="_blank"
            rel="noreferrer"
            className="vj-btn"
            title="Open audience-only fullscreen output"
          >
            Stage ↗
          </a>
          <button
            onClick={onHideUi}
            className="vj-btn vj-btn--accent"
            title="Fullscreen the AI output only (hotkey: H)"
          >
            {hideUi ? "Show UI" : "Hide UI"}
          </button>
        </div>
      </div>
    </header>
  );
}

type Tone = "live" | "info" | "error" | "warn" | "muted";

function SysBlock({
  label,
  tone,
  value,
  title,
}: {
  label: string;
  tone: Tone;
  value: string;
  title?: string;
}) {
  const color = toneColor(tone);
  return (
    <div className="flex items-center gap-2" title={title ?? `${label}: ${value}`}>
      <span
        className={tone === "muted" ? "vj-dot vj-dot--static" : "vj-dot"}
        style={{ color }}
        aria-hidden
      />
      <span className="text-[color:var(--vj-ink-dim)]">{label}</span>
      <span
        className="tabular-nums normal-case tracking-normal"
        style={{ color }}
      >
        {value}
      </span>
    </div>
  );
}

function BrandSigil() {
  // Tiny branding glyph so the bar doesn't look headless. NOT a headline.
  return (
    <div
      className="flex items-center gap-1.5 text-[color:var(--vj-accent)]"
      aria-hidden
      title="vj0"
    >
      <svg width="14" height="14" viewBox="0 0 14 14" className="shrink-0">
        <path
          d="M2 2 L7 12 L12 2"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="7" cy="7" r="1.6" fill="currentColor" />
      </svg>
      <span className="text-[11px] font-bold tracking-[0.2em]">vj0</span>
    </div>
  );
}

function toneColor(tone: Tone): string {
  switch (tone) {
    case "live":
      return "var(--vj-live)";
    case "info":
      return "var(--vj-info)";
    case "error":
      return "var(--vj-error)";
    case "warn":
      return "var(--vj-warn)";
    case "muted":
      return "var(--vj-muted)";
  }
}

function formatAi(status: AiTransportStatus, genMs: number | null): string {
  if (status === "connected") {
    if (genMs && genMs > 0) {
      const fps = 1000 / genMs;
      return `connected · ${genMs.toFixed(0)}ms · ${fps.toFixed(0)}fps`;
    }
    return "connected";
  }
  return status;
}

function trimLabel(s: string): string {
  if (s.length <= 28) return s;
  return s.slice(0, 26) + "…";
}
