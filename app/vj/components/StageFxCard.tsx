"use client";

import { PanelHeader } from "./PanelHeader";

interface StageFxCardProps {
  sharpen: number;
  onSharpenChange: (v: number) => void;
  scanlines: boolean;
  onScanlinesChange: (v: boolean) => void;
  vignette: boolean;
  onVignetteChange: (v: boolean) => void;
  pixelate: boolean;
  onPixelateChange: (v: boolean) => void;
  pixelateSize: number;
  onPixelateSizeChange: (v: number) => void;
}

/**
 * OUTPUT-column FX card. Sharpen runs as a WebGL unsharp-mask pass on
 * /vj/stage; scanlines + vignette are CSS overlays matching the preview
 * frame. Pixelate is mutually exclusive with sharpen (sharpen on
 * quantised blocks just amplifies block-edge step) — when ON, the
 * StageRenderer skips the sharpen pass entirely.
 */
export function StageFxCard({
  sharpen,
  onSharpenChange,
  scanlines,
  onScanlinesChange,
  vignette,
  onVignetteChange,
  pixelate,
  onPixelateChange,
  pixelateSize,
  onPixelateSizeChange,
}: StageFxCardProps) {
  return (
    <div className="vj-panel p-2 flex flex-col gap-2">
      <PanelHeader
        title="Stage FX"
        actions={
          <span className="text-[9px] uppercase tracking-wider font-mono text-[color:var(--vj-ink-dim)]">
            /vj/stage
          </span>
        }
      />
      <div className="flex items-center gap-2">
        <span className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)] w-16">
          sharpen
        </span>
        <input
          type="range"
          min={0}
          max={10}
          step={0.1}
          value={sharpen}
          onChange={(e) => onSharpenChange(Number(e.target.value))}
          className="vj-range vj-range--tight"
          style={
            {
              ["--vj-range-fill" as string]: `${(sharpen / 10) * 100}%`,
            } as React.CSSProperties
          }
          title="WebGL unsharp-mask strength (0 = off, ~1 mild, 10 = aggressive)"
        />
        <span className="font-mono text-[11px] tabular-nums text-[color:var(--vj-info)] w-10 text-right ml-auto">
          {sharpen.toFixed(1)}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <label className="flex items-center gap-2 text-[10px] font-mono text-[color:var(--vj-ink-dim)] uppercase tracking-wider">
          <input
            type="checkbox"
            checked={scanlines}
            onChange={(e) => onScanlinesChange(e.target.checked)}
            className="vj-check"
          />
          scanlines
        </label>
        <label className="flex items-center gap-2 text-[10px] font-mono text-[color:var(--vj-ink-dim)] uppercase tracking-wider">
          <input
            type="checkbox"
            checked={vignette}
            onChange={(e) => onVignetteChange(e.target.checked)}
            className="vj-check"
          />
          vignette
        </label>
      </div>
      <div className="flex items-center gap-2">
        <label className="flex items-center gap-2 text-[10px] font-mono text-[color:var(--vj-ink-dim)] uppercase tracking-wider w-16">
          <input
            type="checkbox"
            checked={pixelate}
            onChange={(e) => onPixelateChange(e.target.checked)}
            className="vj-check"
          />
          pixelate
        </label>
        <input
          type="range"
          min={2}
          max={32}
          step={1}
          value={pixelateSize}
          onChange={(e) => onPixelateSizeChange(Number(e.target.value))}
          disabled={!pixelate}
          className="vj-range vj-range--tight"
          style={
            {
              ["--vj-range-fill" as string]: `${((pixelateSize - 2) / 30) * 100}%`,
              opacity: pixelate ? 1 : 0.3,
            } as React.CSSProperties
          }
          title="Pixelate block size in source pixels (2..32)"
        />
        <span className="font-mono text-[11px] tabular-nums text-[color:var(--vj-info)] w-10 text-right ml-auto">
          {pixelateSize}px
        </span>
      </div>
    </div>
  );
}
