import { create } from "zustand";
import { persist } from "zustand/middleware";

export type AiBackend = "klein" | "sdturbo" | "zimage";

export const AI_BACKEND_URLS: Record<AiBackend, string> = {
  klein: "https://lrlvdh3j31k14t-3000.proxy.runpod.net/webrtc/offer",
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

// Output resolution presets for the AI generator.
//
// HARD RULES this list satisfies:
//   1. FLUX.2 requires both source dimensions divisible by 16.
//   2. Source × INTEGER N must land at or above a standard display target
//      (2560×1440 QHD or 3840×2160 4K) so the StageRenderer canvas can
//      upscale losslessly with the source-pixel UV math kept aligned.
//
// Note about 1080p: 1080 ÷ 16 = 67.5 — so NO div-by-16 source can hit exact
// 1920×1080 with integer N. The closest path is QHD (2560×1440) which
// displays beautifully on a 1080p monitor (browser bilinear-shrinks the
// canvas; ~25% extra source pixels = imperceptible loss + overshoot
// protection on hi-DPI screens).
//
// Each row below documents the integer-N upscale path it offers.
export type OutputPreset = {
  id: string;
  label: string;
  w: number;
  h: number;
};

export const OUTPUT_PRESETS: OutputPreset[] = [
  // 16:9 horizontal — ladder from fast to true-720p quality.
  // First/last hit standard targets exactly; the middle three are aspect-true
  // 16:9-or-near-it intermediates for "I want quality between 512 and 720p"
  // testing. Each new shape pays a one-time ~150 s torch.compile cost on
  // first use, then ~20 s on cache hits forever after.

  // ~45 fps dual-GPU. 16:9 exact. ×10 → QHD, ×15 → 4K (both integer).
  { id: "256x144", w: 256, h: 144, label: "256×144 · 16:9 · fast · ×15 → 4K" },
  // ~45 fps dual-GPU. 16:9 exact. ×5 → QHD (integer).
  { id: "512x288", w: 512, h: 288, label: "512×288 · 16:9 · standard · ×5 → QHD" },
  // ~24 fps dual-GPU. ~16:9 (1.71, slightly tall). 2.3× pixels of 512×288.
  { id: "768x448", w: 768, h: 448, label: "768×448 · ~16:9 · balanced · 2.3× quality" },
  // ~13 fps dual-GPU. 16:9 exact. 4× pixels of 512×288. ×2 → 2048×1152 (~FHD).
  { id: "1024x576", w: 1024, h: 576, label: "1024×576 · 16:9 · high · 4× quality" },
  // ~7 fps dual-GPU. 16:9 exact = real 720p. 6.25× pixels. ×2 → QHD, ×3 → 4K (both integer).
  { id: "1280x720", w: 1280, h: 720, label: "1280×720 · 16:9 · 720p · ×2 → QHD" },

  // 9:16 vertical — same ladder, transposed (phone / portrait projector).
  { id: "288x512", w: 288, h: 512, label: "288×512 · 9:16 · vertical std · ×5 → QHD" },
  { id: "448x768", w: 448, h: 768, label: "448×768 · ~9:16 · vertical balanced" },
  { id: "576x1024", w: 576, h: 1024, label: "576×1024 · 9:16 · vertical high" },
  { id: "720x1280", w: 720, h: 1280, label: "720×1280 · 9:16 · vertical 720p · ×2 → QHD" },
];

export function findOutputPreset(w: number, h: number): OutputPreset | undefined {
  return OUTPUT_PRESETS.find((p) => p.w === w && p.h === h);
}

// Recording output resolution. The recording engine composites the AI
// preview canvas onto a fresh offscreen canvas at one of these sizes
// (object-fit: contain, black letterbox bars), so the produced video has
// clean, predictable dimensions regardless of how big the source canvas's
// internal pixel grid happens to be. All 16:9 — vertical/square content
// shows letterboxed; that's the right call for projector-style sets.
export type RecordingResolution = "1k" | "2k" | "4k";

export interface RecordingResolutionSpec {
  id: RecordingResolution;
  label: string;       // dropdown label
  shortLabel: string;  // header chip label (super tight)
  width: number;
  height: number;
}

export const RECORDING_RESOLUTIONS: ReadonlyArray<RecordingResolutionSpec> = [
  { id: "1k", shortLabel: "1K", label: "1K · 1920×1080", width: 1920, height: 1080 },
  { id: "2k", shortLabel: "2K", label: "2K · 2560×1440", width: 2560, height: 1440 },
  { id: "4k", shortLabel: "4K", label: "4K · 3840×2160", width: 3840, height: 2160 },
];

export function getRecordingResolutionSpec(
  id: RecordingResolution
): RecordingResolutionSpec {
  return (
    RECORDING_RESOLUTIONS.find((r) => r.id === id) ?? RECORDING_RESOLUTIONS[0]
  );
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
  // Pixelate FX. `stagePixelate` toggles the effect; `stagePixelateSize`
  // is the block size in source pixels (1 = unchanged, 8 = chunky 8×8 blocks).
  // Applies to BOTH the AI panel preview and the /vj/stage full-screen
  // output, so what you see in monitor matches what hits the projector.
  stagePixelate: boolean;
  stagePixelateSize: number;
  // Manual fog intensity (0..255) used when the fog toggle is ON. The
  // ON/OFF state itself is not persisted — reloads always start with fog
  // off, which is the safer physical default for a 800 W heater.
  fogIntensity: number;
  // Output resolution for the in-browser MediaRecorder. Persisted so the
  // user picks once and forgets. Always 16:9 — see RECORDING_RESOLUTIONS.
  recordingResolution: RecordingResolution;

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
  setStagePixelate: (value: boolean) => void;
  setStagePixelateSize: (value: number) => void;
  setFogIntensity: (value: number) => void;
  setRecordingResolution: (value: RecordingResolution) => void;
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
      frameRate: 30,
      seed: 42,
      kleinAlpha: 0.10,
      kleinSteps: 2,
      upscaleMode: "lanczos",
      autoConnect: false,
      promptPresets: DEFAULT_PROMPT_PRESETS.slice(),
      audioDeviceId: "",
      sceneId: "",
      showAudioFeatures: true,
      selectedFixtureProfileId: "",
      stageSharpen: 0.6,
      stageScanlines: true,
      stageVignette: true,
      stagePixelate: false,
      stagePixelateSize: 8,
      fogIntensity: 255,
      recordingResolution: "1k",

      setAudioDeviceId: (value) => set({ audioDeviceId: value }),
      setSceneId: (value) => set({ sceneId: value }),
      setShowAudioFeatures: (value) => set({ showAudioFeatures: value }),
      setSelectedFixtureProfileId: (value) =>
        set({ selectedFixtureProfileId: value }),
      setStageSharpen: (value) =>
        set({ stageSharpen: Math.max(0, Math.min(10, value)) }),
      setStageScanlines: (value) => set({ stageScanlines: value }),
      setStageVignette: (value) => set({ stageVignette: value }),
      setStagePixelate: (value) => set({ stagePixelate: value }),
      setStagePixelateSize: (value) =>
        // 1 = no effect (single-pixel blocks); 64 = extreme chonky pixels
        set({ stagePixelateSize: Math.max(1, Math.min(64, Math.round(value))) }),
      setFogIntensity: (value) =>
        set({ fogIntensity: Math.max(0, Math.min(255, Math.round(value))) }),
      setRecordingResolution: (value) =>
        // Defensive: any unknown id falls back to "1k" rather than throwing
        // — a stale localStorage value shouldn't kill the app.
        set({
          recordingResolution: RECORDING_RESOLUTIONS.some((r) => r.id === value)
            ? value
            : "1k",
        }),
      setBackend: (value) => set({ backend: value }),
      setKleinAlpha: (value) =>
        set({ kleinAlpha: Math.max(0, Math.min(0.5, value)) }),
      setKleinSteps: (value) =>
        set({ kleinSteps: Math.max(1, Math.min(4, Math.round(value))) }),
      setUpscaleMode: (value) => set({ upscaleMode: value }),
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
        set({ frameRate: [10, 20, 30, 60].includes(value) ? value : 30 }),
      setSeed: (value) => set({ seed: Math.max(0, Math.floor(value)) }),
    }),
    {
      name: "vj0-ai-settings-storage",
      version: 8,
      migrate: (persisted: unknown, version: number) => {
        // v0/v1 had: outputSize: number, promptPresets: string[]
        // v2 has:   outputWidth/outputHeight,  promptPresets: {label, prompt}[]
        // v3:       max output capped at 512×512 (768/1024 sizes removed for live perf)
        // v4:       default frameRate bumped 20 → 30 (dual-GPU server saturates ~28 fps)
        // v5:       output preset list trimmed to integer-upscale-clean sizes
        // v6:       hideUi field removed (the H-key fullscreen overlay was cut)
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

        if (version < 4) {
          // Bump legacy default 20 fps → 30 fps. Dual-GPU multi-worker server
          // can saturate ~28 fps at 512×288 / 4-step, so 30 is the new sweet spot.
          // Leave 10 alone (some users intentionally pick low for laptop-power
          // setups), and leave 60 alone (already smooth-by-choice).
          if (out.frameRate === 20) out.frameRate = 30;
        }

        if (version < 5) {
          // Resolution presets were trimmed to only ones that integer-upscale
          // to standard display targets (QHD 2560×1440 / 4K 3840×2160). The
          // dropped legacy presets (192×112, 320×176, 384×224, 384×384,
          // 448×256) didn't satisfy that constraint. Snap them to the nearest
          // valid preset by aspect ratio.
          const w = typeof out.outputWidth === "number" ? out.outputWidth : 512;
          const h = typeof out.outputHeight === "number" ? out.outputHeight : 288;
          const stillValid = OUTPUT_PRESETS.some(
            (p) => p.w === w && p.h === h
          );
          if (!stillValid) {
            const aspect = w / h;
            if (aspect > 1.4) {
              // 16:9-ish horizontal → 512×288 (the standard live VJ resolution)
              out.outputWidth = 512;
              out.outputHeight = 288;
            } else if (aspect < 0.7) {
              // 9:16-ish vertical → 288×512
              out.outputWidth = 288;
              out.outputHeight = 512;
            } else {
              // Square-ish → match by size (≤ 320 long side → 256², else 512²)
              const big = Math.max(w, h) > 320;
              out.outputWidth = big ? 512 : 256;
              out.outputHeight = big ? 512 : 256;
            }
          }
        }

        if (version < 6) {
          // Drop the orphaned `hideUi` flag — the H-key fullscreen overlay
          // was removed. Persist would otherwise re-hydrate it indefinitely.
          delete out.hideUi;
        }

        if (version < 7) {
          // Curated preset list: dropped 256×256 and 512×512 (1:1 squares
          // not in current use), added a 16:9 / 9:16 ladder up to 720p
          // (768×448, 1024×576, 1280×720 + their 9:16 mirrors). Re-snap any
          // persisted (w,h) that's not in the new preset list to the closest
          // surviving preset by aspect.
          const w = typeof out.outputWidth === "number" ? out.outputWidth : 512;
          const h = typeof out.outputHeight === "number" ? out.outputHeight : 288;
          const stillValid = OUTPUT_PRESETS.some(
            (p) => p.w === w && p.h === h
          );
          if (!stillValid) {
            const aspect = w / h;
            if (aspect > 1.4) {
              out.outputWidth = 512; out.outputHeight = 288;       // 16:9 default
            } else if (aspect < 0.7) {
              out.outputWidth = 288; out.outputHeight = 512;       // 9:16 default
            } else {
              // Squarish → use the area to pick a horizontal-ish substitute
              // (no 1:1 left in the list). Small → 256×144, big → 512×288.
              const big = Math.max(w, h) > 320;
              out.outputWidth = big ? 512 : 256;
              out.outputHeight = big ? 288 : 144;
            }
          }
        }

        if (version < 8) {
          // New `recordingResolution` field for the in-browser MP4/WebM
          // recorder. Default to "1k" (Full HD) — small files, fast share,
          // matches the most common projector output. Users who want
          // higher fidelity can flip to 2K/4K via the dropdown.
          if (
            typeof out.recordingResolution !== "string" ||
            !["1k", "2k", "4k"].includes(out.recordingResolution as string)
          ) {
            out.recordingResolution = "1k";
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
