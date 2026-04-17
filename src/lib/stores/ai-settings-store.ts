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

export type UpscaleMode = "bilinear" | "lanczos";

// Default 1-9 preset prompts, bound to number-key hotkeys in the VJ app.
// Edit here and they'll update in the control panel.
export const DEFAULT_PROMPT_PRESETS: string[] = [
  "vibrant neon cyberpunk city street at night, rain, reflections, wide angle",
  "aurora borealis over snowy mountains, green and purple sky, long exposure",
  "spiral galaxy with pink and blue nebula, dust clouds, deep space, astrophotography",
  "underwater caustics, sun rays piercing blue water, silhouettes of fish, dreamy",
  "black ink dropping into water, slow motion, white background, macro photography",
  "flames dancing in the dark, high speed photograph, black background",
  "forest floor after rain, golden hour, moss and fungi, macro photograph",
  "abstract oil paint swirling in water, vibrant colors, macro",
  "bioluminescent jellyfish in deep ocean, dark blue background, ethereal, cinematic",
];

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
  // Live-performance UX
  upscaleMode: UpscaleMode;       // display-time upscale quality in the output canvas
  hideUi: boolean;                // H key: hide all panels, just show output
  promptPresets: string[];        // bound to number keys 1-9

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
  setUpscaleMode: (value: UpscaleMode) => void;
  setHideUi: (value: boolean) => void;
  setPromptPresets: (value: string[]) => void;
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
      upscaleMode: "lanczos",
      hideUi: false,
      promptPresets: DEFAULT_PROMPT_PRESETS.slice(),

      setBackend: (value) => set({ backend: value }),
      setKleinAlpha: (value) =>
        set({ kleinAlpha: Math.max(0, Math.min(0.5, value)) }),
      setKleinSteps: (value) =>
        set({ kleinSteps: Math.max(1, Math.min(4, Math.round(value))) }),
      setUpscaleMode: (value) => set({ upscaleMode: value }),
      setHideUi: (value) => set({ hideUi: value }),
      setPromptPresets: (value) => set({ promptPresets: value.slice(0, 9) }),
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
