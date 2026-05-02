"use client";

import type { Ref, RefObject } from "react";
import type { AiTransport, AiTransportStatus } from "@/src/lib/ai/transport";
import type { UpscaleMode } from "@/src/lib/stores/ai-settings-store";
import { CompileOverlay, type BootPhase } from "./CompileOverlay";
import { PromptDock } from "./PromptDock";
import { ResPicker } from "./ResPicker";
import { StageRenderer, type StageRendererHandle } from "./StageRenderer";

interface AiCompileState {
  phase: BootPhase;
  width?: number;
  height?: number;
  n_steps?: number;
  iter?: number;
  total_iters?: number;
  elapsed_ms?: number;
  est_seconds?: number;
  started_at: number;
}

interface AiServerState {
  workerCount: number;
  readyCount: number;
}

interface AiConsoleCardProps {
  /** Used only to gate the ▶ generate button on isConnected(). Connect /
   *  disconnect controls themselves live on the SystemsBar. */
  aiTransport: AiTransport;
  /** Drives the placeholder text inside the preview canvas. */
  aiStatus: AiTransportStatus;

  // Preview canvas
  previewRendererRef: Ref<StageRendererHandle>;
  aiImageUrl: string | null;
  aiCompile: AiCompileState | null;
  aiServer: AiServerState | null;

  // Live cues
  aiSendFrames: boolean;
  onSendFramesChange: (v: boolean) => void;
  aiOutputWidth: number;
  aiOutputHeight: number;
  onOutputSizeChange: (w: number, h: number) => void;
  aiBackendKlein: boolean;
  aiKleinAlpha: number;
  onKleinAlphaChange: (v: number) => void;
  aiKleinSteps: number;
  onKleinStepsChange: (v: number) => void;
  // Stage FX (display passes — only what the preview canvas needs)
  aiStageSharpen: number;
  aiStagePixelate: boolean;
  aiStagePixelateSize: number;

  // Prompt
  aiPrompt: string;
  onPromptChange: (p: string) => void;

  // Set-and-forget params
  aiFrameRate: number;
  onFrameRateChange: (v: number) => void;
  aiSeed: number;
  onSeedChange: (v: number) => void;
  aiUpscaleMode: UpscaleMode;
  onUpscaleModeChange: (v: UpscaleMode) => void;

  // Diagnostics
  aiShowCaptureDebug: boolean;
  onShowCaptureDebugChange: (v: boolean) => void;
  aiDebugCanvasRef: RefObject<HTMLCanvasElement | null>;
  aiLogs: string[];
}

/**
 * AI CONSOLE — generation pipeline surface. Owns the preview canvas + the
 * controls a VJ touches mid-set: ▶ generate, output resolution, klein α/
 * steps, prompt. Set-and-forget params + diagnostics tuck into disclosures
 * below.
 *
 * Deliberately headless: no panel title, no card-level recording control.
 * The preview canvas is the visual anchor — slapping "AI CONSOLE" above it
 * burned a row to restate what's already obvious. Recording moved to the
 * SystemsBar (it's a session-output action, peer to "Stage ↗", not a
 * per-cue control). Connection controls also live on the SystemsBar.
 *
 * Visual language: every inline control is the same height (.vj-chip
 * matches .vj-btn), so the live toolbar reads as a single row of equal-
 * weight tools instead of four bordered "mini-panels" glued together.
 */
export function AiConsoleCard({
  aiTransport,
  aiStatus,
  previewRendererRef,
  aiImageUrl,
  aiCompile,
  aiServer,
  aiSendFrames,
  onSendFramesChange,
  aiOutputWidth,
  aiOutputHeight,
  onOutputSizeChange,
  aiBackendKlein,
  aiKleinAlpha,
  onKleinAlphaChange,
  aiKleinSteps,
  onKleinStepsChange,
  aiStageSharpen,
  aiStagePixelate,
  aiStagePixelateSize,
  aiPrompt,
  onPromptChange,
  aiFrameRate,
  onFrameRateChange,
  aiSeed,
  onSeedChange,
  aiUpscaleMode,
  onUpscaleModeChange,
  aiShowCaptureDebug,
  onShowCaptureDebugChange,
  aiDebugCanvasRef,
  aiLogs,
}: AiConsoleCardProps) {
  const sliderFillPct = Math.round((aiKleinAlpha / 0.5) * 100);

  return (
    <div className="vj-panel p-2 flex flex-col gap-2">
      {/* Body splits into a hero preview (left, ~62%) and an options
          column (right, ~38%) at lg+. Below lg, falls back to a
          stacked single column so the preview still leads. The split
          lets mid-set actions (generate, ResPicker, α / steps, prompt)
          sit inline with the preview instead of pushing it up the
          screen and forcing eyes to ping-pong between canvas and
          controls. The 1.6fr/1fr ratio (vs. an even tighter 2fr/1fr)
          buys the options column ~70px more horizontal real estate at
          1920w — enough for the four params chips (capture · fps ·
          seed · upscale) to sit on one line instead of orphaning
          upscale onto its own row. The preview is still huge — 16:9
          aspect drives height and at 1920w the canvas lands at ~870px,
          which is more than enough hero presence. */}
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)] gap-3 items-start">
        {/* Preview — hero. WebGL StageRenderer applies the same sharpen
            pass that runs on /vj/stage so what you see here is what the
            projector sees. The frame is locked to 16:9 regardless of
            output resolution; vertical content shows letterboxed,
            exactly as it'll appear on a 16:9 projector. */}
        <div className="vj-canvas-frame relative flex items-center justify-center aspect-video w-full">
          <StageRenderer
            ref={previewRendererRef}
            sharpen={aiStagePixelate ? 0 : aiStageSharpen}
            pixelate={aiStagePixelate ? aiStagePixelateSize : 0}
            className="w-full h-full"
            style={{ objectFit: "contain" }}
          />
          {aiCompile && (
            <CompileOverlay
              phase={aiCompile.phase}
              width={aiCompile.width}
              height={aiCompile.height}
              startedAt={aiCompile.started_at}
              estSeconds={aiCompile.est_seconds ?? 150}
              iter={aiCompile.iter}
              totalIters={aiCompile.total_iters}
            />
          )}
          {!aiImageUrl && !aiCompile && (
            <div className="absolute inset-0 flex items-center justify-center text-[11px] font-mono uppercase tracking-wider text-[color:var(--vj-ink-dim)] text-center px-4 bg-black">
              {/*
               * Placeholder ladder — match the UI to *what's actually
               * happening on the wire*, not just "connected/not". Order
               * matters: each branch is the most-specific true thing we
               * can report.
               *
               *   1. connecting              → negotiating webrtc
               *   2. not connected           → press ▶ connect
               *   3. workers booting         → "preparing workers x/y"
               *      (aiServer.readyCount < workerCount — server isn't
               *      able to serve frames yet even if we sent them)
               *   4. generating, no frame yet → "generating · waiting
               *      for first frame" (aiSendFrames==true, aiImageUrl
               *      still null — frames are flying but the round-trip
               *      hasn't returned one. Without this branch the user
               *      sees "press ▶ generate" *while already generating*,
               *      which is the bug they kept hitting after auto-
               *      connect.)
               *   5. ready idle              → "{n} workers ready ·
               *      press ▶ generate"
               */}
              {aiStatus === "connecting" ? (
                <span className="text-[color:var(--vj-info)]">
                  negotiating webrtc…
                </span>
              ) : aiStatus !== "connected" ? (
                <span>
                  not connected —
                  <br />
                  <span className="text-[color:var(--vj-accent)]">
                    press ▶ connect ↑
                  </span>
                </span>
              ) : aiServer && aiServer.readyCount < aiServer.workerCount ? (
                <span>
                  <span className="text-[color:var(--vj-info)]">
                    preparing workers
                  </span>
                  <br />
                  <span className="text-[color:var(--vj-accent)] tabular-nums">
                    {aiServer.readyCount} / {aiServer.workerCount} ready
                  </span>
                  <br />
                  <span className="text-[10px] mt-1 inline-block normal-case tracking-normal">
                    loading model weights and JIT-compiling kernels —
                    ~3 min on cold boot, then frames will start streaming
                  </span>
                </span>
              ) : aiSendFrames ? (
                <span>
                  <span className="text-[color:var(--vj-live)]">
                    generating
                  </span>
                  <br />
                  <span className="text-[color:var(--vj-ink-dim)]">
                    waiting for first frame…
                  </span>
                </span>
              ) : (
                <span>
                  <span className="text-[color:var(--vj-info)]">
                    {aiServer
                      ? `${aiServer.workerCount} ${aiServer.workerCount === 1 ? "worker" : "workers"} ready`
                      : "connected"}
                  </span>
                  <br />
                  <span className="text-[color:var(--vj-accent)]">
                    press ▶ generate or hit space
                  </span>
                </span>
              )}
            </div>
          )}
        </div>

        {/* Options column — two persistent toolbar rows (live cues +
            params), the prompt, then the diagnostics disclosure. Both
            toolbar rows share the same .vj-chip language so the visual
            hierarchy comes from row order (actions on top, settings
            below), not from heavier framing. min-w-0 lets the column
            actually shrink when long preset chips would otherwise
            stretch the grid. */}
        <div className="flex flex-col gap-2 min-w-0">
          {/* Row 1 — Live cues. Generate is the action so it leads in
              magenta accent; the per-cue chips (resolution, klein α /
              steps, match-to-output) follow in the neutral chip style.
              Wraps if the column gets too narrow; otherwise sits on
              one line. */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <button
              type="button"
              onClick={() => onSendFramesChange(!aiSendFrames)}
              disabled={!aiTransport.isConnected()}
              className={`vj-btn ${aiSendFrames ? "vj-btn--live" : "vj-btn--accent"}`}
              title={
                !aiTransport.isConnected()
                  ? "Connect AI first"
                  : "Toggle generation (Space)"
              }
            >
              {aiSendFrames ? "■ stop" : "▶ generate"}
            </button>
            <ResPicker
              width={aiOutputWidth}
              height={aiOutputHeight}
              onPick={onOutputSizeChange}
            />
            {aiBackendKlein && (
              <div
                className="vj-chip"
                title="0 = pure prompt · 0.5 = waveform dominates · ↑/↓ to adjust"
              >
                <span className="vj-chip__label">α</span>
                <input
                  type="range"
                  min={0}
                  max={0.5}
                  step={0.01}
                  value={aiKleinAlpha}
                  onChange={(e) => onKleinAlphaChange(Number(e.target.value))}
                  className="vj-range vj-chip__range"
                  style={
                    {
                      ["--vj-range-fill" as string]: `${sliderFillPct}%`,
                    } as React.CSSProperties
                  }
                />
                <span className="vj-chip__value">{aiKleinAlpha.toFixed(2)}</span>
              </div>
            )}
            {aiBackendKlein && (
              <label
                className="vj-chip"
                title="Klein steps — 1 fastest, 2 sweet spot, 4 max quality"
              >
                <span className="vj-chip__label">steps</span>
                <select
                  value={aiKleinSteps}
                  onChange={(e) => onKleinStepsChange(Number(e.target.value))}
                  className="vj-chip__select"
                >
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                  <option value={4}>4</option>
                </select>
              </label>
            )}
          </div>

          {/* Row 2 — Generation params. Permanent, not collapsed: the
              chip language already collapses each one to ~80–130 px so
              they fit on a single row in the available column. */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <label className="vj-chip" title="Target frame rate">
              <span className="vj-chip__label">fps</span>
              <select
                value={aiFrameRate}
                onChange={(e) => onFrameRateChange(Number(e.target.value))}
                className="vj-chip__select"
              >
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={30}>30</option>
                <option value={60}>60</option>
              </select>
            </label>
            <label className="vj-chip" title="Generation seed">
              <span className="vj-chip__label">seed</span>
              <input
                type="number"
                value={aiSeed}
                onChange={(e) =>
                  onSeedChange(Math.max(0, Math.floor(Number(e.target.value))))
                }
                className="vj-chip__input"
              />
              <button
                type="button"
                onClick={() =>
                  onSeedChange(Math.floor(Math.random() * 1_000_000))
                }
                className="vj-chip__icon"
                title="Randomize seed"
                aria-label="Randomize seed"
              >
                ⟲
              </button>
            </label>
            <label
              className="vj-chip"
              title="Interpolation when the AI output is scaled up for display"
            >
              <span className="vj-chip__label">upscale</span>
              <select
                value={aiUpscaleMode}
                onChange={(e) =>
                  onUpscaleModeChange(e.target.value as UpscaleMode)
                }
                className="vj-chip__select"
              >
                <option value="lanczos">lanczos</option>
                <option value="bilinear">bilinear</option>
              </select>
            </label>
          </div>

          <PromptDock activePrompt={aiPrompt} onSetPrompt={onPromptChange} />

          {/* Diagnostics — capture preview + log feed. Stays collapsed:
              the log list and 160-px debug canvas are bulky and only
              wanted while debugging. Unlike params (4 inline chips),
              there's no compact representation that fits in a single row. */}
          <details className="vj-disclosure">
            <summary>
              diagnostics ({aiLogs.length} log{aiLogs.length === 1 ? "" : "s"})
            </summary>
            <div className="mt-2 flex flex-col gap-2">
              <label className="flex items-center gap-2 text-[10px] font-mono text-[color:var(--vj-ink-dim)] uppercase tracking-wider">
                <input
                  type="checkbox"
                  checked={aiShowCaptureDebug}
                  onChange={(e) => onShowCaptureDebugChange(e.target.checked)}
                  className="vj-check"
                />
                show capture preview
              </label>
              {aiShowCaptureDebug && (
                <div className="flex flex-col gap-1 rounded border border-[color:var(--vj-edge-hot)] bg-black/50 p-2">
                  <div className="text-[9px] uppercase tracking-wider font-mono text-[color:var(--vj-ink-dim)]">
                    sending {aiOutputWidth}×{aiOutputHeight}
                  </div>
                  <canvas
                    ref={aiDebugCanvasRef}
                    width={aiOutputWidth}
                    height={aiOutputHeight}
                    className="rounded bg-black self-center"
                    style={{
                      width: 160,
                      height: (160 * aiOutputHeight) / aiOutputWidth,
                      imageRendering: "pixelated",
                    }}
                  />
                </div>
              )}
              <div className="max-h-28 overflow-auto font-mono text-[11px] text-[color:var(--vj-ink-dim)] bg-black/40 rounded p-2">
                {aiLogs.length === 0 ? (
                  <div className="opacity-50">(no messages)</div>
                ) : (
                  aiLogs.map((line, idx) => (
                    <div key={idx} className="whitespace-pre-wrap">
                      {line}
                    </div>
                  ))
                )}
              </div>
            </div>
          </details>
        </div>
      </div>
    </div>
  );
}
