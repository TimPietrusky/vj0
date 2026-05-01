"use client";

import type { RefObject } from "react";
import type { AudioFeatures } from "@/src/lib/audio-features";
import type { VjScene } from "@/src/lib/scenes/types";
import { AudioDebugPanel } from "./AudioDebugPanel";
import { Field } from "./Field";
import { PanelHeader } from "./PanelHeader";

type Status = "idle" | "requesting" | "running" | "error";

interface AudioDevice {
  deviceId: string;
  label: string;
}

interface WaveformSourceCardProps {
  /** Canvas ref the visual engine paints the waveform/scene into. */
  canvasRef: RefObject<HTMLCanvasElement | null>;
  status: Status;

  // Scene picker
  scenes: readonly VjScene[];
  currentSceneId: string;
  onSceneChange: (sceneId: string) => void;

  // Audio source
  systemAudioSupported: boolean;
  systemAudioValue: string;
  selectedDeviceId: string;
  devices: AudioDevice[];
  onDeviceChange: (deviceId: string) => void;

  // Audio features debug disclosure (two-way binds to persisted store)
  showDebug: boolean;
  setShowDebug: (open: boolean) => void;
  debugFeatures: AudioFeatures | null;
}

/**
 * INPUT column card — owns the waveform canvas, scene picker, audio
 * source picker, and (collapsed by default) audio features debug
 * readout. Was previously two cards; merged because Audio Features
 * rarely matters live.
 */
export function WaveformSourceCard({
  canvasRef,
  status,
  scenes,
  currentSceneId,
  onSceneChange,
  systemAudioSupported,
  systemAudioValue,
  selectedDeviceId,
  devices,
  onDeviceChange,
  showDebug,
  setShowDebug,
  debugFeatures,
}: WaveformSourceCardProps) {
  return (
    <div className="vj-panel p-2 flex flex-col gap-2">
      <PanelHeader
        title="Waveform / Source"
        actions={
          <span
            className="flex items-center gap-1.5 text-[10px] text-[color:var(--vj-ink-dim)] normal-case h-full"
            title={`Audio status: ${status}`}
          >
            <span
              className="vj-dot"
              style={{
                color:
                  status === "running"
                    ? "var(--vj-live)"
                    : status === "error"
                    ? "var(--vj-error)"
                    : "var(--vj-info)",
              }}
            />
            <span className="tracking-wider uppercase">
              {status === "running" ? "live" : status}
            </span>
          </span>
        }
      />

      <div className="vj-canvas-frame aspect-[4/1] w-full">
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ imageRendering: "pixelated" }}
        />
      </div>

      {/* Selects stack vertically — col-2 is too narrow for two-up. */}
      <div className="flex flex-col gap-2">
        <Field label="Scene">
          <select
            value={currentSceneId}
            onChange={(e) => onSceneChange(e.target.value)}
            className="vj-input"
          >
            {scenes.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        </Field>

        <Field
          label="Audio In"
          hint={
            systemAudioSupported
              ? "Mic / line-in device, or 🖥 System audio (browser will prompt to share a tab/window/screen with audio)"
              : "Pick which audio input drives the visuals"
          }
        >
          <select
            value={selectedDeviceId}
            onChange={(e) => onDeviceChange(e.target.value)}
            className="vj-input"
          >
            {systemAudioSupported && (
              <option value={systemAudioValue}>
                🖥 System audio (share screen/tab)
              </option>
            )}
            <option value="">Default device</option>
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label}
              </option>
            ))}
          </select>
        </Field>
      </div>

      {/* Audio Features — demoted from full card to a disclosure. It's a
          debug overlay; live-set use is rare. The <details> open state
          two-way-binds to the persisted showDebug flag so the polling
          effect tied to showDebug still gates correctly (no polling when
          collapsed). */}
      <details
        className="text-xs"
        open={showDebug}
        onToggle={(e) => setShowDebug(e.currentTarget.open)}
      >
        <summary className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)] cursor-pointer hover:text-[color:var(--vj-info)] py-1">
          audio features
        </summary>
        <div className="mt-2">
          <AudioDebugPanel features={debugFeatures} />
        </div>
      </details>
    </div>
  );
}
