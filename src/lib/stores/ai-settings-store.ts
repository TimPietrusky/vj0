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

// Prompt preset: a short label (shown on hotkey chips 1-9) and the full prompt
// text sent to the AI. Label is kept terse so the chip stays compact.
export type PromptPreset = {
  label: string;
  prompt: string;
};

// Output resolution presets for the AI generator. FLUX.2 requires both
// dimensions divisible by 16. We cap at 512×512 — bigger than that costs
// too much per-frame for live performance. Projector-friendly widescreen
// options dominate the small end (16:9 and near-16:9 wide formats), since
// that's what most VJ rigs actually output to.
export type OutputPreset = {
  id: string;
  label: string;
  w: number;
  h: number;
};

export const OUTPUT_PRESETS: OutputPreset[] = [
  { id: "192x112", w: 192, h: 112, label: "192×112 · ~16:9 · ultra-fast (projector)" },
  { id: "256x144", w: 256, h: 144, label: "256×144 · 16:9 · fast (projector)" },
  { id: "256x256", w: 256, h: 256, label: "256×256 · 1:1 · fast" },
  { id: "320x176", w: 320, h: 176, label: "320×176 · ~16:9 · fast (projector)" },
  { id: "384x224", w: 384, h: 224, label: "384×224 · ~16:9 · balanced (projector)" },
  { id: "384x384", w: 384, h: 384, label: "384×384 · 1:1 · balanced" },
  { id: "448x256", w: 448, h: 256, label: "448×256 · ~16:9 · balanced (projector)" },
  { id: "512x288", w: 512, h: 288, label: "512×288 · 16:9 · max (projector)" },
  { id: "512x512", w: 512, h: 512, label: "512×512 · 1:1 · max" },
];

export function findOutputPreset(w: number, h: number): OutputPreset | undefined {
  return OUTPUT_PRESETS.find((p) => p.w === w && p.h === h);
}

// Default 1-9 preset prompts, bound to number-key hotkeys in the VJ app.
// `label` shows on the hotkey chip; `prompt` is what's sent to the model.
export const DEFAULT_PROMPT_PRESETS: PromptPreset[] = [
  { label: "Cyber",  prompt: "vibrant neon cyberpunk city street at night, rain, reflections, wide angle" },
  { label: "Aurora", prompt: "aurora borealis over snowy mountains, green and purple sky, long exposure" },
  { label: "Galaxy", prompt: "spiral galaxy with pink and blue nebula, dust clouds, deep space, astrophotography" },
  { label: "Ocean",  prompt: "underwater caustics, sun rays piercing blue water, silhouettes of fish, dreamy" },
  { label: "Ink",    prompt: "black ink dropping into water, slow motion, white background, macro photography" },
  { label: "Fire",   prompt: "flames dancing in the dark, high speed photograph, black background" },
  { label: "Forest", prompt: "forest floor after rain, golden hour, moss and fungi, macro photograph" },
  { label: "Paint",  prompt: "abstract oil paint swirling in water, vibrant colors, macro" },
  { label: "Jelly",  prompt: "bioluminescent jellyfish in deep ocean, dark blue background, ethereal, cinematic" },
];

interface AiSettingsState {
  backend: AiBackend;
  showAi: boolean;
  sendFrames: boolean;
  showCaptureDebug: boolean;
  prompt: string;
  captureSize: number;
  /** Output width in pixels (must be div by 16 for FLUX.2). */
  outputWidth: number;
  /** Output height in pixels (must be div by 16 for FLUX.2). */
  outputHeight: number;
  frameRate: number;
  seed: number;
  // Klein-specific
  kleinAlpha: number;   // 0..0.5 — how much the input biases composition (0 = pure t2i)
  kleinSteps: number;   // 1..4 — inference steps (2 = sweet spot, 4 = max quality)
  // Live-performance UX
  upscaleMode: UpscaleMode;       // display-time upscale quality in the output canvas
  hideUi: boolean;                // H key: hide all panels, just show output
  autoConnect: boolean;           // auto-connect to configured backend on app load
  promptPresets: PromptPreset[];  // bound to number keys 1-9
  // Last-selected audio input device. Persisted so it survives reloads, and
  // re-auto-selected when the device gets plugged in mid-session via the
  // navigator.mediaDevices `devicechange` event listener in VJApp.
  audioDeviceId: string;
  // Visual scene currently driving the waveform canvas.
  sceneId: string;
  // Whether the Audio Features panel is expanded (also gates polling).
  showAudioFeatures: boolean;
  // Last-selected fixture profile in the Lighting card's "+ add" picker.
  selectedFixtureProfileId: string;
  // Stage-page FX. `stageSharpen` is the WebGL unsharp-mask strength
  // (0 = off, ~1 = mild, ~5 = strong, 10 = aggressive). Clamped 0..10.
  // Scanlines + vignette are CSS overlays matching the preview frame.
  stageSharpen: number;
  stageScanlines: boolean;
  stageVignette: boolean;
  // Manual fog intensity (0..255) used when the fog toggle is ON. The
  // ON/OFF state itself is not persisted — reloads always start with fog
  // off, which is the safer physical default for a 800 W heater.
  fogIntensity: number;

  setBackend: (value: AiBackend) => void;
  setShowAi: (value: boolean) => void;
  setSendFrames: (value: boolean) => void;
  setShowCaptureDebug: (value: boolean) => void;
  setPrompt: (value: string) => void;
  setCaptureSize: (value: number) => void;
  setOutputSize: (width: number, height?: number) => void;
  setFrameRate: (value: number) => void;
  setSeed: (value: number) => void;
  setKleinAlpha: (value: number) => void;
  setKleinSteps: (value: number) => void;
  setUpscaleMode: (value: UpscaleMode) => void;
  setHideUi: (value: boolean) => void;
  setAutoConnect: (value: boolean) => void;
  setPromptPresets: (value: PromptPreset[]) => void;
  updatePromptPreset: (index: number, preset: Partial<PromptPreset>) => void;
  setAudioDeviceId: (value: string) => void;
  setSceneId: (value: string) => void;
  setShowAudioFeatures: (value: boolean) => void;
  setSelectedFixtureProfileId: (value: string) => void;
  setStageSharpen: (value: number) => void;
  setStageScanlines: (value: boolean) => void;
  setStageVignette: (value: boolean) => void;
  setFogIntensity: (value: number) => void;
}

export const useAiSettingsStore = create<AiSettingsState>()(
  persist(
    (set) => ({
      backend: "klein",
      showAi: true,
      sendFrames: false,
      showCaptureDebug: false,
      prompt: "colorful abstract art, vibrant neon lights, psychedelic patterns",
      captureSize: 256,
      outputWidth: 512,
      outputHeight: 288, // 16:9 default — projector-friendly
      frameRate: 20,
      seed: 42,
      kleinAlpha: 0.10,
      kleinSteps: 2,
      upscaleMode: "lanczos",
      hideUi: false,
      autoConnect: false,
      promptPresets: DEFAULT_PROMPT_PRESETS.slice(),
      audioDeviceId: "",
      sceneId: "",
      showAudioFeatures: true,
      selectedFixtureProfileId: "",
      stageSharpen: 0.6,
      stageScanlines: true,
      stageVignette: true,
      fogIntensity: 255,

      setAudioDeviceId: (value) => set({ audioDeviceId: value }),
      setSceneId: (value) => set({ sceneId: value }),
      setShowAudioFeatures: (value) => set({ showAudioFeatures: value }),
      setSelectedFixtureProfileId: (value) =>
        set({ selectedFixtureProfileId: value }),
      setStageSharpen: (value) =>
        set({ stageSharpen: Math.max(0, Math.min(10, value)) }),
      setStageScanlines: (value) => set({ stageScanlines: value }),
      setStageVignette: (value) => set({ stageVignette: value }),
      setFogIntensity: (value) =>
        set({ fogIntensity: Math.max(0, Math.min(255, Math.round(value))) }),
      setBackend: (value) => set({ backend: value }),
      setKleinAlpha: (value) =>
        set({ kleinAlpha: Math.max(0, Math.min(0.5, value)) }),
      setKleinSteps: (value) =>
        set({ kleinSteps: Math.max(1, Math.min(4, Math.round(value))) }),
      setUpscaleMode: (value) => set({ upscaleMode: value }),
      setHideUi: (value) => set({ hideUi: value }),
      setAutoConnect: (value) => set({ autoConnect: value }),
      setPromptPresets: (value) => set({ promptPresets: value.slice(0, 9) }),
      updatePromptPreset: (index, preset) =>
        set((s) => {
          if (index < 0 || index >= s.promptPresets.length) return s;
          const next = s.promptPresets.slice();
          next[index] = { ...next[index], ...preset };
          return { promptPresets: next };
        }),
      setShowAi: (value) => set({ showAi: value }),
      setSendFrames: (value) => set({ sendFrames: value }),
      setShowCaptureDebug: (value) => set({ showCaptureDebug: value }),
      setPrompt: (value) => set({ prompt: value }),
      setCaptureSize: (value) =>
        set({ captureSize: Math.max(64, Math.min(1024, value)) }),
      setOutputSize: (width, height) => {
        // Snap to a known preset when possible (handles legacy callers passing
        // only a width). FLUX.2 needs dims divisible by 16 — the preset list
        // already respects that.
        const h = height ?? width;
        const preset = findOutputPreset(width, h);
        if (preset) {
          set({ outputWidth: preset.w, outputHeight: preset.h });
          return;
        }
        // Fallback: round both to the nearest multiple of 16, clamp to sane
        // range. This keeps us valid even if a caller hands in a custom size.
        const snap = (n: number) =>
          Math.max(112, Math.min(512, Math.round(n / 16) * 16));
        set({ outputWidth: snap(width), outputHeight: snap(h) });
      },
      setFrameRate: (value) =>
        set({ frameRate: [10, 20, 30, 60].includes(value) ? value : 20 }),
      setSeed: (value) => set({ seed: Math.max(0, Math.floor(value)) }),
    }),
    {
      name: "vj0-ai-settings-storage",
      version: 3,
      migrate: (persisted: unknown, version: number) => {
        // v0/v1 had: outputSize: number, promptPresets: string[]
        // v2 has:   outputWidth/outputHeight,  promptPresets: {label, prompt}[]
        // v3:       max output capped at 512×512 (768/1024 sizes removed for live perf)
        const s = (persisted as Record<string, unknown>) || {};
        const out: Record<string, unknown> = { ...s };

        if (version < 2) {
          const size = typeof s.outputSize === "number" ? s.outputSize : 512;
          if (!("outputWidth" in out)) out.outputWidth = size;
          if (!("outputHeight" in out)) out.outputHeight = size === 512 ? 288 : size;
          delete out.outputSize;

          if (Array.isArray(s.promptPresets) && typeof s.promptPresets[0] === "string") {
            out.promptPresets = (s.promptPresets as string[]).map((p, i) => ({
              label: DEFAULT_PROMPT_PRESETS[i]?.label ?? `P${i + 1}`,
              prompt: p,
            }));
          }
        }

        if (version < 3) {
          // Clamp legacy oversized output to 512×288 (the new max-projector preset)
          const w = typeof out.outputWidth === "number" ? out.outputWidth : 512;
          const h = typeof out.outputHeight === "number" ? out.outputHeight : 288;
          if (w > 512 || h > 512) {
            out.outputWidth = 512;
            out.outputHeight = 288;
          }
        }

        return out as unknown as AiSettingsState;
      },
    }
  )
);

// Cross-tab sync. Zustand's persist middleware writes to localStorage but
// does NOT subscribe other tabs to those writes — so by default the
// dashboard tab and the /vj/stage tab drift out of sync the moment you
// touch a setting (e.g. moving the Stage FX → sharpen slider). The
// `storage` event fires on every tab *except* the one that wrote, so we
// rehydrate from the new value when our key changes. One-shot listener,
// installed at module-load on the client only.
if (typeof window !== "undefined") {
  window.addEventListener("storage", (e) => {
    if (e.key === "vj0-ai-settings-storage") {
      void useAiSettingsStore.persist.rehydrate();
    }
  });
}
