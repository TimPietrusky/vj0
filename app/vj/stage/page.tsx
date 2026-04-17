"use client";

/**
 * Stage (audience) view — fullscreen AI output only, no controls.
 * Subscribes to the stage BroadcastChannel published by the /vj control tab.
 * Run /vj on your laptop (or iPad), drag this tab to the projector screen and
 * hit F11 to go fullscreen.
 */
import { useEffect, useRef, useState } from "react";
import { openStageChannel, type StageMsg } from "@/src/lib/ai/stage-channel";
import { useAiSettingsStore } from "@/src/lib/stores";

export default function StagePage() {
  // Output canvas — we draw the received JPEG into a canvas so we can control
  // the image smoothing quality (Lanczos vs bilinear) at display time.
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hasSignal, setHasSignal] = useState(false);
  const [promptOverlay, setPromptOverlay] = useState<string>("");
  const overlayTimeoutRef = useRef<number | null>(null);

  // Display-time upscale quality (synced from the store via localStorage)
  const upscaleMode = useAiSettingsStore((s) => s.upscaleMode);

  useEffect(() => {
    const ch = openStageChannel();
    if (!ch) return;

    // Announce presence so the control tab can (optionally) re-publish current state
    ch.postMessage({ type: "hello" });

    const onMessage = async (ev: MessageEvent<StageMsg>) => {
      const msg = ev.data;
      if (!msg || typeof msg !== "object") return;

      if (msg.type === "frame") {
        setHasSignal(true);
        const canvas = canvasRef.current;
        if (!canvas) return;
        try {
          const blob = new Blob([msg.bytes], { type: "image/jpeg" });
          const bitmap = await createImageBitmap(blob);
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          // Use the bitmap's actual dimensions, not what the control tab claims —
          // the server may still be returning its default size for the first few
          // frames after a settings change, and we want to render those correctly
          // (object-fit: contain in CSS handles letterboxing) instead of clipping
          // a 512×512 frame into a 256×144 canvas.
          if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
          }
          // high = Lanczos/bicubic in modern browsers; low = bilinear
          ctx.imageSmoothingEnabled = upscaleMode !== "bilinear";
          ctx.imageSmoothingQuality =
            upscaleMode === "lanczos" ? "high" : "low";
          ctx.drawImage(bitmap, 0, 0);
          bitmap.close?.();
        } catch {
          // swallow — next frame will try again
        }
      } else if (msg.type === "prompt") {
        const p = msg.prompt.trim();
        if (!p) return;
        setPromptOverlay(p);
        if (overlayTimeoutRef.current)
          window.clearTimeout(overlayTimeoutRef.current);
        overlayTimeoutRef.current = window.setTimeout(() => {
          setPromptOverlay("");
          overlayTimeoutRef.current = null;
        }, 2400);
      }
    };

    ch.onmessage = onMessage;
    return () => {
      ch.onmessage = null;
      ch.close();
      if (overlayTimeoutRef.current)
        window.clearTimeout(overlayTimeoutRef.current);
    };
  }, [upscaleMode]);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        margin: 0,
        padding: 0,
        background: "#000",
        overflow: "hidden",
        cursor: "none",
      }}
    >
      {/* Centered, contain-scaled output canvas. CSS scales pixels; the
          imageSmoothing setting above controls the interpolation. */}
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "contain",
          imageRendering: upscaleMode === "bilinear" ? "auto" : "auto",
        }}
      />

      {/* Prompt reveal overlay — fades in for ~2.4s on prompt change */}
      <div
        style={{
          position: "absolute",
          left: "4vw",
          bottom: "6vh",
          right: "4vw",
          pointerEvents: "none",
          fontFamily:
            "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
          fontSize: "2vh",
          lineHeight: 1.35,
          letterSpacing: "0.02em",
          color: "rgba(255,255,255,0.88)",
          textShadow:
            "0 0 18px rgba(0,0,0,0.9), 0 1px 3px rgba(0,0,0,0.95)",
          opacity: promptOverlay ? 1 : 0,
          transform: promptOverlay ? "translateY(0)" : "translateY(8px)",
          transition:
            "opacity 380ms ease, transform 380ms ease",
        }}
      >
        {promptOverlay}
      </div>

      {/* Idle placeholder before the first frame arrives */}
      {!hasSignal && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#444",
            fontFamily:
              "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
            fontSize: "1.4vh",
            letterSpacing: "0.15em",
            textTransform: "uppercase",
          }}
        >
          waiting for stage feed from /vj …
        </div>
      )}
    </div>
  );
}
