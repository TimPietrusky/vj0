"use client";

import type { AiTransportStatus } from "@/src/lib/ai/transport";
import {
  AI_BACKEND_LABELS,
  AI_BACKEND_URLS,
  RECORDING_RESOLUTIONS,
  type AiBackend,
  type RecordingResolution,
} from "@/src/lib/stores/ai-settings-store";
import { RecordingControl } from "./RecordingControl";

type Status = "idle" | "requesting" | "running" | "error";
type DmxStatus = "disconnected" | "connecting" | "connected" | "unsupported";

interface SystemsBarProps {
  audioStatus: Status;
  audioDeviceLabel?: string;
  aiStatus: AiTransportStatus;
  aiBackend: AiBackend;
  /** Switch backend (also closes any open WebRTC channel — caller's job). */
  onBackendChange: (b: AiBackend) => void;
  aiAutoConnect: boolean;
  onAutoConnectChange: (v: boolean) => void;
  /** Open the WebRTC channel. Bound to the connect button. */
  onConnect: () => void;
  /** Close the WebRTC channel. Bound to the disconnect button. */
  onDisconnect: () => void;
  aiGenTimeMs: number | null;
  /** Per-stage timing breakdown from inference_server.py. When present,
   *  hover the AI block to see vae/transformer/jpeg ms split — useful for
   *  spotting the next bottleneck without re-instrumenting the worker. */
  aiTiming?: {
    decode_in_ms?: number;
    prompt_ms?: number;
    vae_encode_ms?: number;
    transformer_plus_decode_ms?: number;
    jpeg_ms?: number;
    total_ms?: number;
  } | null;
  dmxStatus: DmxStatus;
  dmxFixtureCount: number;
  dmxActiveCount: number;
  /** Master DMX/lighting switch — when false the DMX block goes muted "off". */
  lightingEnabled: boolean;
  /** Whether the DMX console drawer is currently open. Drives the chevron
   *  rotation and the magenta accent state on the chip. */
  dmxOpen: boolean;
  /** Toggle the DMX console drawer open/closed. */
  onToggleDmx: () => void;

  // ── Recording — session-output controls ─────────────────────────────
  // Promoted out of the AI Console card so it sits next to the other
  // global session actions (Stage ↗). The card itself is a per-cue
  // surface; capturing the whole session belongs at the bar level.
  isRecording: boolean;
  isFinalizingRecording: boolean;
  recordingSupported: boolean;
  getRecordingElapsedMs: () => number;
  onStartRecording: () => void;
  onStopRecording: () => void;
  recordingResolution: RecordingResolution;
  onRecordingResolutionChange: (value: RecordingResolution) => void;
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
  onBackendChange,
  aiAutoConnect,
  onAutoConnectChange,
  onConnect,
  onDisconnect,
  aiGenTimeMs,
  aiTiming,
  dmxStatus,
  dmxFixtureCount,
  dmxActiveCount,
  lightingEnabled,
  dmxOpen,
  onToggleDmx,
  isRecording,
  isFinalizingRecording,
  recordingSupported,
  getRecordingElapsedMs,
  onStartRecording,
  onStopRecording,
  recordingResolution,
  onRecordingResolutionChange,
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

        {/* AI block — status pill + backend dropdown + auto + connect.
            Promoted from the AI Console card header so the most-touched
            session controls live next to their status indicator. */}
        <div className="flex items-center gap-2">
          <SysBlock
            label="AI"
            tone={aiTone}
            value={formatAi(aiStatus, aiGenTimeMs)}
            title={
              aiTiming
                ? `${AI_BACKEND_LABELS[aiBackend]}\n` +
                  `vae enc: ${(aiTiming.vae_encode_ms ?? 0).toFixed(1)} ms\n` +
                  `transformer+vae dec: ${(aiTiming.transformer_plus_decode_ms ?? 0).toFixed(1)} ms\n` +
                  `jpeg: ${(aiTiming.jpeg_ms ?? 0).toFixed(1)} ms\n` +
                  `decode in: ${(aiTiming.decode_in_ms ?? 0).toFixed(1)} ms\n` +
                  `prompt: ${(aiTiming.prompt_ms ?? 0).toFixed(1)} ms\n` +
                  `total: ${(aiTiming.total_ms ?? 0).toFixed(1)} ms`
                : AI_BACKEND_LABELS[aiBackend]
            }
          />
          <select
            value={aiBackend}
            onChange={(e) => onBackendChange(e.target.value as AiBackend)}
            className="vj-input vj-input--bar"
            title={`Backend · ${AI_BACKEND_LABELS[aiBackend]}`}
          >
            {(Object.keys(AI_BACKEND_URLS) as AiBackend[]).map((k) => (
              <option key={k} value={k}>
                {AI_BACKEND_LABELS[k]}
              </option>
            ))}
          </select>
          <label
            className="flex items-center gap-1 normal-case tracking-normal text-[color:var(--vj-ink-dim)]"
            title="Auto-connect on page load and on backend switch"
          >
            <input
              type="checkbox"
              checked={aiAutoConnect}
              onChange={(e) => onAutoConnectChange(e.target.checked)}
              className="vj-check"
            />
            auto
          </label>
          {aiStatus === "connected" || aiStatus === "connecting" ? (
            <button
              onClick={onDisconnect}
              className="vj-btn vj-btn--danger vj-btn--bar"
              title="Close the WebRTC channel"
            >
              ✕ disc.
            </button>
          ) : (
            <button
              onClick={onConnect}
              className="vj-btn vj-btn--live vj-btn--bar"
              title="Open WebRTC channel to the AI backend"
            >
              ▶ connect
            </button>
          )}
        </div>

        {/* DMX chip — clickable surface that toggles the console drawer.
            When the drawer is open it adopts a magenta accent (matching the
            drawer's top edge); when fixtures are actively driving DMX the
            chip pulses softly so the operator catches it peripherally. */}
        <button
          type="button"
          onClick={onToggleDmx}
          aria-expanded={dmxOpen}
          aria-controls="vj-dmx-drawer"
          className={`vj-sys-btn ${
            lightingEnabled && dmxActiveCount > 0 && !dmxOpen
              ? "vj-sys-btn--live"
              : ""
          }`}
          title={
            dmxOpen
              ? "Close DMX console (Esc)"
              : `Open DMX console · ${dmxFixtureCount} fixture${
                  dmxFixtureCount === 1 ? "" : "s"
                }${dmxActiveCount > 0 ? `, ${dmxActiveCount} active` : ""}`
          }
        >
          <DmxChipBody
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
          <svg
            className="vj-chev"
            viewBox="0 0 10 10"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden
          >
            <path d="M2 4 L5 7 L8 4" />
          </svg>
        </button>

        <div className="ml-auto flex items-center gap-2">
          {/* Recording — session output. Resolution dropdown locks while a
              clip is in progress (can't change mid-recording without
              restarting the encoder). The shortLabel ("1K"/"2K"/"4K")
              keeps the bar dense; full label is in the option text. */}
          <select
            value={recordingResolution}
            onChange={(e) =>
              onRecordingResolutionChange(e.target.value as RecordingResolution)
            }
            disabled={isRecording || isFinalizingRecording}
            className="vj-input vj-input--bar"
            title="Recording output resolution. Source canvas is composited onto a fresh offscreen canvas at this size — letterboxed if AI source aspect doesn't match 16:9."
          >
            {RECORDING_RESOLUTIONS.map((r) => (
              <option key={r.id} value={r.id}>
                {r.shortLabel}
              </option>
            ))}
          </select>
          <RecordingControl
            isRecording={isRecording}
            isSupported={recordingSupported}
            getElapsedMs={getRecordingElapsedMs}
            onStart={onStartRecording}
            onStop={onStopRecording}
            isFinalizing={isFinalizingRecording}
            compact
          />

          <a
            href="/vj/stage"
            target="_blank"
            rel="noreferrer"
            className="vj-btn vj-btn--bar"
            title="Open audience-only fullscreen output"
          >
            Stage ↗
          </a>
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
        className="vj-dot"
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

/**
 * Inner status row for the DMX chip — same visual language as SysBlock but
 * stripped of the wrapping div so it can sit inside a <button> without
 * doubling up roles. Tone-coloured dot + label + value, no extra click
 * target (the surrounding .vj-sys-btn is the click surface).
 */
function DmxChipBody({ tone, value }: { tone: Tone; value: string }) {
  const color = toneColor(tone);
  return (
    <>
      <span className="vj-dot" style={{ color }} aria-hidden />
      <span className="text-[color:var(--vj-ink-dim)]">DMX</span>
      <span
        className="tabular-nums normal-case tracking-normal"
        style={{ color }}
      >
        {value}
      </span>
    </>
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
