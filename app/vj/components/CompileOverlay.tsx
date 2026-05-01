"use client";

import { useEffect, useState } from "react";

/**
 * Server-emitted boot phases. Map directly to messages from
 * inference_server.py — the worker walks this list once at startup
 * (cold boot) and again every time it sees a new (width, height)
 * shape that hasn't been JIT-compiled yet.
 */
export type BootPhase =
  | "loading_weights"
  | "applying_fp8"
  | "registering_compile_stubs"
  | "warming_up";

const BOOT_PHASE_TITLE: Record<BootPhase, string> = {
  loading_weights: "Loading the AI model",
  applying_fp8: "Optimising for speed",
  registering_compile_stubs: "Almost there",
  warming_up: "Warming up",
};

interface CompileOverlayProps {
  phase: BootPhase;
  width?: number;
  height?: number;
  startedAt: number;
  estSeconds: number;
  iter?: number;
  totalIters?: number;
}

/**
 * Overlay shown when the AI worker is JIT-compiling for a new (width,
 * height). The compile is a single ~150 s torch.compile call that emits
 * no Python-side progress, so we estimate elapsed/remaining locally from
 * `startedAt` and the server-reported `estSeconds`. A 4 Hz timer keeps
 * the bar moving even when nothing has happened on the network for tens
 * of seconds.
 */
export function CompileOverlay({
  phase,
  width,
  height,
  startedAt,
  estSeconds,
  iter,
  totalIters,
}: CompileOverlayProps) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  const elapsed = Math.max(0, (now - startedAt) / 1000);
  const pct = Math.min(0.99, elapsed / Math.max(1, estSeconds));
  const remaining = Math.max(0, Math.ceil(estSeconds - elapsed));
  // Resolution is only meaningful during warm-up; during model load there
  // isn't a "current shape" yet (server's at default until first request).
  const showSize = phase === "warming_up" && width && height;

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-4 bg-black/85 backdrop-blur-sm gap-3 font-mono">
      <div className="text-[11px] uppercase tracking-wider text-[color:var(--vj-info)] animate-pulse">
        {BOOT_PHASE_TITLE[phase]}
      </div>
      {showSize && (
        <div className="text-[28px] font-bold text-[color:var(--vj-accent)]">
          {width}×{height}
        </div>
      )}
      <div className="w-56 h-1.5 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full bg-[color:var(--vj-accent)] transition-[width] duration-200 ease-linear"
          style={{ width: `${pct * 100}%` }}
        />
      </div>
      <div className="text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)] tabular-nums">
        {Math.round(elapsed)}s elapsed · ~{remaining}s left
      </div>
      {phase === "warming_up" && iter && totalIters && (
        <div className="text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)]">
          step {iter} of {totalIters}
        </div>
      )}
    </div>
  );
}
