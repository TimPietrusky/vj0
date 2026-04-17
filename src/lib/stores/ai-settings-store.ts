import { create } from "zustand";
import { persist } from "zustand/middleware";

export type AiBackend = "klein" | "sdturbo" | "zimage";

export const AI_BACKEND_URLS: Record<AiBackend, string> = {
  klein: "https://3m90zbu8fwyyqk-3000.proxy.runpod.net/webrtc/offer",
  sdturbo: "https://3746utbd1i3x73-3000.proxy.runpod.net/webrtc/offer",
  zimage: "https://astt1jyau6hsaq-3000.proxy.runpod.net/webrtc/offer",
};

export const AI_BACKEND_LABELS: Record<AiBackend, string> = {
  klein: "FLUX.2 Klein (img2img, ~30fps@256)",
  sdturbo: "SD-Turbo (t2i-ish, ~60fps@256)",
  zimage: "Z-Image Turbo Nunchaku (img2img, ~17fps@256)",
};

interface AiSettingsState {
  backend: AiBackend;
  showAi: boolean;
  sendFrames: boolean;
  showCaptureDebug: boolean;
  prompt: string;
  captureSize: number;
  outputSize: number;
  frameRate: number;
  seed: number;
  // Klein-specific
  kleinAlpha: number;   // 0..0.5 — how much the input biases composition (0 = pure t2i)
  kleinSteps: number;   // 1..4 — inference steps (2 = sweet spot, 4 = max quality)

  setBackend: (value: AiBackend) => void;
  setShowAi: (value: boolean) => void;
  setSendFrames: (value: boolean) => void;
  setShowCaptureDebug: (value: boolean) => void;
  setPrompt: (value: string) => void;
  setCaptureSize: (value: number) => void;
  setOutputSize: (value: number) => void;
  setFrameRate: (value: number) => void;
  setSeed: (value: number) => void;
  setKleinAlpha: (value: number) => void;
  setKleinSteps: (value: number) => void;
}

export const useAiSettingsStore = create<AiSettingsState>()(
  persist(
    (set) => ({
      backend: "klein",
      showAi: true,
      sendFrames: false,
      showCaptureDebug: false,
      prompt: "colorful abstract art, vibrant neon lights, psychedelic patterns",
      captureSize: 128,
      outputSize: 256,
      frameRate: 20,
      seed: 42,
      kleinAlpha: 0.10,
      kleinSteps: 2,

      setBackend: (value) => set({ backend: value }),
      setKleinAlpha: (value) =>
        set({ kleinAlpha: Math.max(0, Math.min(0.5, value)) }),
      setKleinSteps: (value) =>
        set({ kleinSteps: Math.max(1, Math.min(4, Math.round(value))) }),
      setShowAi: (value) => set({ showAi: value }),
      setSendFrames: (value) => set({ sendFrames: value }),
      setShowCaptureDebug: (value) => set({ showCaptureDebug: value }),
      setPrompt: (value) => set({ prompt: value }),
      setCaptureSize: (value) =>
        set({ captureSize: Math.max(64, Math.min(1024, value)) }),
      setOutputSize: (value) =>
        set({
          outputSize: [256, 512, 768, 1024].includes(value) ? value : 256,
        }),
      setFrameRate: (value) =>
        set({ frameRate: [10, 20, 30, 60].includes(value) ? value : 20 }),
      setSeed: (value) => set({ seed: Math.max(0, Math.floor(value)) }),
    }),
    {
      name: "vj0-ai-settings-storage",
    }
  )
);
