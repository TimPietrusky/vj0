import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AiSettingsState {
  showAi: boolean;
  sendFrames: boolean;
  showCaptureDebug: boolean;
  prompt: string;
  captureSize: number;
  outputSize: number;
  frameRate: number;
  seed: number;

  setShowAi: (value: boolean) => void;
  setSendFrames: (value: boolean) => void;
  setShowCaptureDebug: (value: boolean) => void;
  setPrompt: (value: string) => void;
  setCaptureSize: (value: number) => void;
  setOutputSize: (value: number) => void;
  setFrameRate: (value: number) => void;
  setSeed: (value: number) => void;
}

export const useAiSettingsStore = create<AiSettingsState>()(
  persist(
    (set) => ({
      showAi: true,
      sendFrames: false,
      showCaptureDebug: false,
      prompt: "colorful abstract art, vibrant neon lights, psychedelic patterns",
      captureSize: 128,
      outputSize: 256,
      frameRate: 20,
      seed: 42,

      setShowAi: (value) => set({ showAi: value }),
      setSendFrames: (value) => set({ sendFrames: value }),
      setShowCaptureDebug: (value) => set({ showCaptureDebug: value }),
      setPrompt: (value) => set({ prompt: value }),
      setCaptureSize: (value) =>
        set({ captureSize: Math.max(64, Math.min(256, value)) }),
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
