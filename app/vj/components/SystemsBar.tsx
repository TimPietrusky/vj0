"use client";

import React, { useState, useRef, useEffect } from "react";
import type { AiTransportStatus } from "@/src/lib/ai/transport";
import type { VjScene } from "@/src/lib/scenes/types";
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
type Tone = "live" | "info" | "error" | "warn" | "muted";
type ActivePopover = "audio" | "ai" | null;

interface AudioDevice {
  deviceId: string;
  label: string;
}

interface SystemsBarProps {
  // ── Audio ─────────────────────────────────────────────────────
  audioStatus: Status;
  audioDeviceLabel?: string;
  /** Available audio input devices for the pop-over device picker. */
  devices: AudioDevice[];
  /** Currently selected device ID ("" = default, systemAudioValue = system). */
  selectedDeviceId: string;
  onDeviceChange: (deviceId: string) => void;
  /** Whether getDisplayMedia is available in this browser. */
  systemAudioSupported: boolean;
  /** Sentinel value for the "System audio" dropdown entry. */
  systemAudioValue: string;

  // ── Scene ─────────────────────────────────────────────────────
  scenes: readonly VjScene[];
  currentSceneId: string;
  onSceneChange: (sceneId: string) => void;

  // ── AI ────────────────────────────────────────────────────────
  aiStatus: AiTransportStatus;
  aiBackend: AiBackend;
  /** Switch backend (also closes any open WebRTC channel — caller's job). */
  onBackendChange: (b: AiBackend) => void;
  aiAutoConnect: boolean;
  onAutoConnectChange: (v: boolean) => void;
  /** Open the WebRTC channel. */
  onConnect: () => void;
  /** Close the WebRTC channel. */
  onDisconnect: () => void;
  aiGenTimeMs: number | null;
  /** Actual received frames per second — real canvas output rate. */
  aiRecvFps: number | null;
  /** Per-stage timing breakdown from inference_server.py. */
  aiTiming?: {
    decode_in_ms?: number;
    prompt_ms?: number;
    vae_encode_ms?: number;
    transformer_plus_decode_ms?: number;
    jpeg_ms?: number;
    total_ms?: number;
  } | null;

  // ── DMX ───────────────────────────────────────────────────────
  dmxStatus: DmxStatus;
  dmxFixtureCount: number;
  dmxActiveCount: number;
  /** Master DMX/lighting switch — when false the DMX block goes muted "off". */
  lightingEnabled: boolean;
  /** Whether the DMX console drawer is currently open. */
  dmxOpen: boolean;
  /** Toggle the DMX console drawer open/closed. */
  onToggleDmx: () => void;

  // ── Recording ─────────────────────────────────────────────────
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
 * Minimal systems bar — icon-driven top bar with pop-over menus.
 *
 * Surface shows only: logo, 3 icon buttons (Audio / AI / DMX), a compact
 * recording indicator when active, the live FPS counter, and a Stage link.
 * Everything else (device selection, scene, backend, latency breakdown,
 * recording controls) lives in the pop-overs — one click away but never
 * cluttering the bar during a live set.
 */
export function SystemsBar({
  audioStatus,
  audioDeviceLabel,
  devices,
  selectedDeviceId,
  onDeviceChange,
  systemAudioSupported,
  systemAudioValue,
  scenes,
  currentSceneId,
  onSceneChange,
  aiStatus,
  aiBackend,
  onBackendChange,
  aiAutoConnect,
  onAutoConnectChange,
  onConnect,
  onDisconnect,
  aiRecvFps,
  aiTiming,
  dmxStatus,
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
  const [activePopover, setActivePopover] = useState<ActivePopover>(null);
  const controlsRef = useRef<HTMLDivElement>(null);

  const togglePopover = (which: ActivePopover) => {
    setActivePopover((prev) => (prev === which ? null : which));
  };

  // ── Tones ─────────────────────────────────────────────────────
  const audioTone: Tone =
    audioStatus === "error"
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

  // ── Outside click + Escape closes pop-over ────────────────────
  useEffect(() => {
    if (!activePopover) return;
    const onDown = (e: MouseEvent) => {
      if (
        controlsRef.current &&
        !controlsRef.current.contains(e.target as Node)
      ) {
        setActivePopover(null);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setActivePopover(null);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [activePopover]);

  return (
    <header className="sticky top-0 z-30 w-full border-b border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-panel)]/95 backdrop-blur-sm">
      <div className="flex items-center gap-4 px-4 py-2">
        {/* ── Left: Logo + Icon Buttons + Pop-overs ─────────────── */}
        <div className="relative flex items-center gap-3" ref={controlsRef}>
          <BrandSigil />

          {/* Text buttons with status dots */}
          <div className="flex items-center gap-1">
            <TextButton
              label="audio"
              tone={audioTone}
              active={activePopover === "audio"}
              onClick={() => togglePopover("audio")}
            />
            <TextButton
              label="ai"
              tone={aiTone}
              active={activePopover === "ai"}
              onClick={() => togglePopover("ai")}
            />
            <TextButton
              label="show control"
              tone={dmxTone}
              active={dmxOpen}
              onClick={onToggleDmx}
            />
          </div>

          {/* Recording indicator — compact stop in the bar when active,
              so stopping mid-set never requires opening a menu. */}
          {isRecording && (
            <RecordingControl
              isRecording
              isSupported
              getElapsedMs={getRecordingElapsedMs}
              onStart={onStartRecording}
              onStop={onStopRecording}
              isFinalizing={isFinalizingRecording}
              compact
            />
          )}

          {/* ── Pop-over panel ────────────────────────────────────── */}
          {activePopover && (
            <div className="absolute left-0 top-full mt-2 w-72 rounded-lg border border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-panel)] shadow-[0_18px_48px_-12px_rgba(0,0,0,0.7)] z-50">
              <div className="space-y-3 p-4 font-mono text-[11px] uppercase tracking-wider">
                {activePopover === "audio" ? (
                  <>
                    {/* ── Audio status ─────────────────────────── */}
                    <PopoverSection title="Audio">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <span
                            className="vj-dot"
                            style={{ color: toneColor(audioTone) }}
                          />
                          <span className="text-[color:var(--vj-ink)]">
                            {audioStatus === "running"
                              ? "Active"
                              : audioStatus === "requesting"
                              ? "Starting…"
                              : audioStatus === "error"
                              ? "Error"
                              : "Idle"}
                          </span>
                        </div>

                        {audioDeviceLabel && (
                          <div className="text-[10px] text-[color:var(--vj-ink-dim)] normal-case">
                            {audioDeviceLabel}
                          </div>
                        )}
                      </div>
                    </PopoverSection>

                    {/* ── Device selection ─────────────────────── */}
                    <PopoverSection title="Device">
                      <select
                        value={selectedDeviceId}
                        onChange={(e) => onDeviceChange(e.target.value)}
                        className="vj-input w-full"
                      >
                        {systemAudioSupported && (
                          <option value={systemAudioValue}>
                            System audio (share screen/tab)
                          </option>
                        )}
                        <option value="">Default device</option>
                        {devices.map((d) => (
                          <option key={d.deviceId} value={d.deviceId}>
                            {d.label}
                          </option>
                        ))}
                      </select>
                    </PopoverSection>

                    {/* ── Scene selection ──────────────────────── */}
                    <PopoverSection title="Scene">
                      <select
                        value={currentSceneId}
                        onChange={(e) => onSceneChange(e.target.value)}
                        className="vj-input w-full"
                      >
                        {scenes.map((s) => (
                          <option key={s.id} value={s.id}>
                            {s.name}
                          </option>
                        ))}
                      </select>
                    </PopoverSection>
                  </>
                ) : (
                  <>
                    {/* ── AI Backend ───────────────────────────── */}
                    <PopoverSection title="AI Backend">
                      <div className="space-y-2">
                        <select
                          value={aiBackend}
                          onChange={(e) =>
                            onBackendChange(e.target.value as AiBackend)
                          }
                          className="vj-input w-full"
                        >
                          {(Object.keys(AI_BACKEND_URLS) as AiBackend[]).map(
                            (k) => (
                              <option key={k} value={k}>
                                {AI_BACKEND_LABELS[k]}
                              </option>
                            )
                          )}
                        </select>

                        <label className="flex items-center gap-2 text-[color:var(--vj-ink-dim)]">
                          <input
                            type="checkbox"
                            checked={aiAutoConnect}
                            onChange={(e) =>
                              onAutoConnectChange(e.target.checked)
                            }
                            className="vj-check"
                          />
                          <span>Auto-connect</span>
                        </label>

                        {aiStatus === "connected" ||
                        aiStatus === "connecting" ? (
                          <button
                            onClick={onDisconnect}
                            className="vj-btn vj-btn--danger w-full"
                          >
                            Disconnect
                          </button>
                        ) : (
                          <button
                            onClick={onConnect}
                            className="vj-btn vj-btn--live w-full"
                          >
                            Connect
                          </button>
                        )}
                      </div>
                    </PopoverSection>

                    {/* ── Latency breakdown ────────────────────── */}
                    {aiTiming && (
                      <PopoverSection title="Latency">
                        <div className="space-y-1 text-[10px] text-[color:var(--vj-info)] normal-case">
                          <div>
                            VAE encode:{" "}
                            {(aiTiming.vae_encode_ms ?? 0).toFixed(1)} ms
                          </div>
                          <div>
                            Transformer + decode:{" "}
                            {(
                              aiTiming.transformer_plus_decode_ms ?? 0
                            ).toFixed(1)}{" "}
                            ms
                          </div>
                          <div>
                            JPEG: {(aiTiming.jpeg_ms ?? 0).toFixed(1)} ms
                          </div>
                          <div>
                            Prompt: {(aiTiming.prompt_ms ?? 0).toFixed(1)} ms
                          </div>
                          <div className="font-bold text-[color:var(--vj-ink)]">
                            Total: {(aiTiming.total_ms ?? 0).toFixed(1)} ms
                          </div>
                        </div>
                      </PopoverSection>
                    )}

                    {/* ── Recording ────────────────────────────── */}
                    <PopoverSection title="Recording">
                      <div className="space-y-2">
                        <select
                          value={recordingResolution}
                          onChange={(e) =>
                            onRecordingResolutionChange(
                              e.target.value as RecordingResolution
                            )
                          }
                          disabled={isRecording || isFinalizingRecording}
                          className="vj-input w-full"
                        >
                          {RECORDING_RESOLUTIONS.map((r) => (
                            <option key={r.id} value={r.id}>
                              {r.label}
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
                        />
                      </div>
                    </PopoverSection>
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        {/* ── Center: FPS metric ──────────────────────────────── */}
        <div className="flex-1 flex items-center justify-center gap-4 text-[11px] font-mono uppercase tracking-wider text-[color:var(--vj-ink-dim)]">
          {aiRecvFps != null && (
            <div className="flex items-center gap-1.5">
              <span
                className="vj-dot"
                style={{ color: "var(--vj-info)" }}
              />
              <span>{aiRecvFps.toFixed(0)} fps</span>
            </div>
          )}
        </div>

        {/* ── Right: Stage ────────────────────────────────────── */}
        <a
          href="/vj/stage"
          target="_blank"
          rel="noreferrer"
          className="vj-btn vj-btn--bar"
          title="Open audience-only fullscreen output"
        >
          Stage
        </a>
      </div>
    </header>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   Sub-components
   ═══════════════════════════════════════════════════════════════════ */

function PopoverSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-2 text-[10px] font-bold text-[color:var(--vj-accent)] tracking-[0.15em]">
        {title}
      </div>
      <div className="text-[color:var(--vj-ink)]">{children}</div>
    </div>
  );
}

/** Text button with a status dot — reads as a label you can click. */
function TextButton({
  label,
  tone,
  active = false,
  onClick,
}: {
  label: string;
  tone: Tone;
  active?: boolean;
  onClick: () => void;
}) {
  const color = toneColor(tone);
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1.5 text-[11px] font-mono uppercase tracking-wider transition-all ${
        active
          ? "bg-[color:var(--vj-accent)]/10 ring-1 ring-[color:var(--vj-accent)] text-[color:var(--vj-ink)]"
          : "text-[color:var(--vj-ink-dim)] hover:bg-[color:var(--vj-edge)] hover:text-[color:var(--vj-ink)]"
      }`}
    >
      <span
        className="vj-dot"
        style={{ color }}
      />
      <span>{label}</span>
    </button>
  );
}

/** vj0 wordmark — text only, no icon. */
function BrandSigil() {
  return (
    <span
      className="text-[12px] font-bold tracking-[0.22em] text-[color:var(--vj-accent)]"
      title="vj0"
    >
      vj0
    </span>
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
