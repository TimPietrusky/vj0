"use client";

import { useEffect, useState } from "react";

interface RecordingControlProps {
  /** True while a recording is in progress. */
  isRecording: boolean;
  /** Whether MediaRecorder + a supported MIME type are available in this browser. */
  isSupported: boolean;
  /**
   * Polled every ~250 ms while `isRecording` is true. Pulled from the
   * RecordingEngine via ref — kept out of React state to avoid re-rendering
   * the whole VJApp on every tick.
   */
  getElapsedMs: () => number;
  onStart: () => void;
  onStop: () => void;
  /** Set while we're flushing chunks into a Blob & triggering download. */
  isFinalizing?: boolean;
  /**
   * When true, the button uses the .vj-btn--bar size variant so it docks
   * cleanly inside the SystemsBar next to the other inline session
   * controls (connect / disconnect / Stage). Default: false.
   */
  compact?: boolean;
}

/**
 * Record / stop button. Two visual states:
 *
 *   idle      → red ● REC button (vj-btn--danger)
 *   recording → ■ STOP button + live elapsed timer pulsing red
 *
 * Lives in the global SystemsBar — recording is a session-output action
 * (like Stage ↗), not a per-cue control, so it sits next to the other
 * global actions instead of buried inside the AI Console card header.
 *
 * The button stays hit-testable through every state so the user can stop a
 * recording immediately. Disabled only when MediaRecorder isn't supported,
 * with a tooltip explaining why.
 */
export function RecordingControl({
  isRecording,
  isSupported,
  getElapsedMs,
  onStart,
  onStop,
  isFinalizing,
  compact = false,
}: RecordingControlProps) {
  const sizeClass = compact ? " vj-btn--bar" : "";
  // 4 Hz tick force-renders this component while recording so the elapsed
  // time below picks up fresh values from the engine. When idle no interval
  // runs — the label is simply frozen at "00:00" via the conditional below.
  // Deriving instead of storing avoids the "setState inside an effect" lint
  // (and saves us from forgetting to reset the label when recording stops).
  const [, forceTickRender] = useState(0);
  useEffect(() => {
    if (!isRecording) return;
    const id = window.setInterval(() => forceTickRender((n) => n + 1), 250);
    return () => window.clearInterval(id);
  }, [isRecording]);

  const elapsedLabel = isRecording
    ? formatElapsed(getElapsedMs())
    : "00:00";

  if (!isSupported) {
    return (
      <button
        type="button"
        disabled
        className={`vj-btn${sizeClass}`}
        title="Recording requires MediaRecorder + an MP4/WebM encoder. Your browser doesn't expose either."
      >
        ● rec n/a
      </button>
    );
  }

  if (isRecording) {
    return (
      <button
        type="button"
        onClick={onStop}
        className={`vj-btn vj-btn--danger${sizeClass}`}
        title="Stop recording and download the file"
      >
        <span
          aria-hidden
          className="inline-block w-2 h-2 rounded-full bg-[color:var(--vj-error)] animate-pulse"
        />
        <span className="tabular-nums">{elapsedLabel}</span>
        <span>■ stop</span>
      </button>
    );
  }

  if (isFinalizing) {
    return (
      <button
        type="button"
        disabled
        className={`vj-btn${sizeClass}`}
        title="Saving recording…"
      >
        ⏳ saving…
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={onStart}
      className={`vj-btn vj-btn--danger${sizeClass}`}
      title="Record the AI output canvas + live audio to a video file"
    >
      ● rec
    </button>
  );
}

/** mm:ss for short clips, hh:mm:ss past the hour mark. */
function formatElapsed(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const hh = Math.floor(totalSec / 3600);
  const mm = Math.floor((totalSec % 3600) / 60);
  const ss = totalSec % 60;
  const pad = (n: number) => n.toString().padStart(2, "0");
  if (hh > 0) return `${pad(hh)}:${pad(mm)}:${pad(ss)}`;
  return `${pad(mm)}:${pad(ss)}`;
}
