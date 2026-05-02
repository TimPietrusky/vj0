"use client";

import type { RefObject } from "react";
import type { AudioFeatures } from "@/src/lib/audio-features";
import { AudioDebugPanel } from "./AudioDebugPanel";

type Status = "idle" | "requesting" | "running" | "error";

interface WaveformSourceCardProps {
  /** Canvas ref the visual engine paints the waveform/scene into. */
  canvasRef: RefObject<HTMLCanvasElement | null>;
  status: Status;

  // Audio features debug disclosure (two-way binds to persisted store)
  showDebug: boolean;
  setShowDebug: (open: boolean) => void;
  debugFeatures: AudioFeatures | null;
}

/**
 * INPUT column card — owns the waveform canvas and (collapsed by default)
 * audio features debug readout. Aspect ratio matches the AI preview output
 * (16:9) so input and output feel visually linked.
 */
export function WaveformSourceCard({
  canvasRef,
  showDebug,
  setShowDebug,
  debugFeatures,
}: WaveformSourceCardProps) {
  return (
    <div className="vj-panel p-2 flex flex-col gap-2">
      <div className="vj-canvas-frame aspect-video w-full">
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ imageRendering: "pixelated" }}
        />
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
