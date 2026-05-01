"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { AudioEngine } from "@/src/lib/audio-engine";
import { VisualEngine, SCENES } from "@/src/lib/scenes";
import {
  LightingEngine,
  DmxOutput,
  FIXTURE_PROFILES,
} from "@/src/lib/lighting";
import {
  useLightingStore,
  useFixtures,
  useAiSettingsStore,
} from "@/src/lib/stores";
import {
  AI_BACKEND_URLS,
  getRecordingResolutionSpec,
} from "@/src/lib/stores/ai-settings-store";
import {
  openStageChannel,
  type StageMsg,
} from "@/src/lib/ai/stage-channel";
import type { AudioFeatures } from "@/src/lib/audio-features";
import { WebRtcAiTransport } from "@/src/lib/ai/webrtc-transport";
import type {
  AiIncomingFrame,
  AiTransportStatus,
} from "@/src/lib/ai/transport";
import type {
  FixtureInstance,
  LightingFrame,
  StrobeMode,
  ColorMode,
  DimmerMode,
} from "@/src/lib/lighting";
import {
  AiConsoleCard,
  LightingPanel,
  PerformanceDeckCard,
  StageFxCard,
  SystemsBar,
  WaveformSourceCard,
  type StageRendererHandle,
} from "./components";
import type { BootPhase } from "./components";
import { RecordingEngine, type RecordingResult } from "@/src/lib/recording";

type Status = "idle" | "requesting" | "running" | "error";
type DmxStatus = "disconnected" | "connecting" | "connected" | "unsupported";
type AudioSource = "device" | "system";

interface AudioDevice {
  deviceId: string;
  label: string;
}

// Sentinel deviceId value for the "🖥 System audio" entry in the audio input
// dropdown. Picking it triggers getDisplayMedia instead of getUserMedia.
const SYSTEM_AUDIO_VALUE = "__system__";

/**
 * Main VJ application orchestrator.
 *
 * Layout (signal flow, left → right):
 *  - SystemsBar (sticky, full-width) — audio / ai / dmx at a glance.
 *    The DMX chip is also the toggle for the console drawer below.
 *  - Below: 12-col responsive dashboard, grouped by signal-flow stage
 *     [INPUT col-3]      waveform + scene/audio source + features
 *                        (disclosure), STAGE FX below — both "set
 *                        before the set" surfaces, off the hot path
 *     [AI CONSOLE col-9] preview (left 2/3) + prompt / generate /
 *                        per-cue params (right 1/3), PERFORMANCE DECK
 *                        (hotkey board) below — the wide chip row
 *                        wants a full-bleed surface, not a thin column
 *  - DMX console — slide-up drawer surfaced via the DMX chip in the
 *    SystemsBar. It overlays the dashboard from the bottom (z-25, below
 *    the sticky bar at z-30), so the user can patch fixtures and tune
 *    fog without losing the canvas/preview behind it. Esc dismisses.
 *
 * Engine lifecycle via refs (not React state) to avoid blocking render loops.
 */

/**
 * Save a finished RecordingResult as a file the user can keep. We use the
 * classic anchor-with-download trick rather than a full file-system access
 * dance — works in every browser, no extra permission, lands in the user's
 * default downloads folder. The object URL gets revoked after a 1 s delay so
 * Safari has time to start the download before the URL goes away.
 */
function triggerRecordingDownload(result: RecordingResult): void {
  const url = URL.createObjectURL(result.blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `vj0-${formatRecordingTimestamp(new Date())}.${result.extension}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function formatRecordingTimestamp(date: Date): string {
  const pad = (n: number) => n.toString().padStart(2, "0");
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `_${pad(date.getHours())}-${pad(date.getMinutes())}-${pad(date.getSeconds())}`
  );
}

export function VJApp() {
  // Engine refs - NOT React state to avoid render loop interference
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioEngineRef = useRef<AudioEngine | null>(null);
  const visualEngineRef = useRef<VisualEngine | null>(null);
  const lightingEngineRef = useRef<LightingEngine | null>(null);
  const dmxOutputRef = useRef<DmxOutput | null>(null);
  // Lazily constructed on first record so we don't allocate the engine
  // (or its internal MediaRecorder probe) until the user actually wants it.
  const recordingEngineRef = useRef<RecordingEngine | null>(null);

  const aiIceServers = useMemo((): RTCIceServer[] => {
    const raw = process.env.NEXT_PUBLIC_VJ0_WEBRTC_ICE_SERVERS_JSON;
    if (!raw) return [{ urls: "stun:stun.l.google.com:19302" }];
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) return parsed as RTCIceServer[];
    } catch {
      // ignore
    }
    return [{ urls: "stun:stun.l.google.com:19302" }];
  }, []);

  const aiIceTransportPolicy = useMemo(():
    | RTCIceTransportPolicy
    | undefined => {
    const raw = process.env.NEXT_PUBLIC_VJ0_WEBRTC_ICE_TRANSPORT_POLICY;
    if (raw === "relay" || raw === "all") return raw;
    return undefined;
  }, []);

  // Backend choice (persisted) picks the signaling URL. Env var still overrides.
  const aiBackendSel = useAiSettingsStore((s) => s.backend);
  const aiSignalingUrl = useMemo(() => {
    const envOverride = process.env.NEXT_PUBLIC_VJ0_WEBRTC_SIGNALING_URL;
    if (envOverride) return envOverride;
    return AI_BACKEND_URLS[aiBackendSel] || "/api/webrtc/offer";
  }, [aiBackendSel]);

  const aiTransport = useMemo(
    () =>
      new WebRtcAiTransport({
        signalingUrl: aiSignalingUrl,
        iceServers: aiIceServers,
        iceTransportPolicy: aiIceTransportPolicy,
      }),
    [aiSignalingUrl, aiIceServers, aiIceTransportPolicy]
  );

  // /healthz URL — same origin as the signaling URL with /healthz path. We
  // poll this when not actively receiving frames so we can show
  // "preparing workers (N/M ready)" instead of "press generate" while the
  // server is still loading weights or JIT-compiling.
  const aiHealthUrl = useMemo(() => {
    try {
      const u = new URL(aiSignalingUrl, window.location.href);
      u.pathname = "/healthz";
      u.search = "";
      return u.toString();
    } catch {
      return null;
    }
  }, [aiSignalingUrl]);

  useEffect(() => {
    return () => {
      void aiTransport.stop();
    };
  }, [aiTransport]);

  // Auto-connect on app load (and on backend switch), if the user enabled it.
  const aiAutoConnectSel = useAiSettingsStore((s) => s.autoConnect);
  useEffect(() => {
    if (!aiAutoConnectSel) return;
    if (!aiTransport.isConnected()) void aiTransport.start();
  }, [aiAutoConnectSel, aiTransport]);

  // UI state
  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const persistedDeviceId = useAiSettingsStore((s) => s.audioDeviceId);
  const setPersistedDeviceId = useAiSettingsStore((s) => s.setAudioDeviceId);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>(persistedDeviceId);
  // Tracks whether we're currently fed by a microphone-style device or by a
  // system-audio capture (getDisplayMedia). NOT persisted — getDisplayMedia
  // requires a fresh user gesture each session anyway, so we always come back
  // up on the persisted device after a reload.
  const [audioSource, setAudioSource] = useState<AudioSource>("device");
  // Whether the browser supports getDisplayMedia. Used to gate the System
  // entry in the audio dropdown (Safari and older browsers don't have it).
  //
  // MUST start as `false` and be filled in via useEffect — NOT useMemo +
  // `typeof navigator !== "undefined"`. SSR has no `navigator`, so a useMemo
  // version evaluates to `false` on the server but `true` on the client's
  // first render, which causes a hydration mismatch on the <option> + the
  // Field's `title` hint. Effect-based init keeps the first client render
  // identical to SSR, then flips to `true` after hydration completes.
  const [systemAudioSupported, setSystemAudioSupported] = useState(false);
  useEffect(() => {
    setSystemAudioSupported(
      typeof navigator !== "undefined" &&
        typeof navigator.mediaDevices?.getDisplayMedia === "function"
    );
  }, []);

  // ── Recording state ──────────────────────────────────────────────────────
  // `recordingSupported` follows the same SSR-safe pattern as systemAudioSupported:
  // it starts false on the server (no MediaRecorder there) and flips after
  // hydration so the button's first client render matches the server's HTML.
  const [recordingSupported, setRecordingSupported] = useState(false);
  useEffect(() => {
    setRecordingSupported(RecordingEngine.isSupported());
  }, []);
  const [isRecording, setIsRecording] = useState(false);
  const [isFinalizingRecording, setIsFinalizingRecording] = useState(false);
  const [recordingError, setRecordingError] = useState<string>("");

  // Persisted scene + audio-features-visible toggle. We keep currentSceneId
  // as a derived selector because the VisualEngine internally tracks the
  // active scene; the store value is the source of truth across reloads.
  const persistedSceneId = useAiSettingsStore((s) => s.sceneId);
  const setPersistedSceneId = useAiSettingsStore((s) => s.setSceneId);
  const initialSceneId = persistedSceneId || SCENES[0].id;
  const [currentSceneId, setCurrentSceneId] = useState<string>(initialSceneId);

  const showDebug = useAiSettingsStore((s) => s.showAudioFeatures);
  const setShowDebug = useAiSettingsStore((s) => s.setShowAudioFeatures);

  // Debug panel state - throttled updates (100ms = ~10fps)
  const [debugFeatures, setDebugFeatures] = useState<AudioFeatures | null>(
    null
  );

  // Remote AI (WebRTC) UI state
  const {
    backend: aiBackend,
    sendFrames: aiSendFrames,
    showCaptureDebug: aiShowCaptureDebug,
    prompt: aiPrompt,
    captureSize: aiCaptureSize,
    outputWidth: aiOutputWidth,
    outputHeight: aiOutputHeight,
    frameRate: aiFrameRate,
    seed: aiSeed,
    kleinAlpha: aiKleinAlpha,
    kleinSteps: aiKleinSteps,
    upscaleMode: aiUpscaleMode,
    autoConnect: aiAutoConnect,
    promptPresets: aiPromptPresets,
    stageSharpen: aiStageSharpen,
    stageScanlines: aiStageScanlines,
    stageVignette: aiStageVignette,
    stagePixelate: aiStagePixelate,
    stagePixelateSize: aiStagePixelateSize,
    fogIntensity,
    recordingResolution,
    setBackend: setAiBackend,
    setSendFrames: setAiSendFrames,
    setShowCaptureDebug: setAiShowCaptureDebug,
    setPrompt: setAiPrompt,
    setCaptureSize: setAiCaptureSize,
    setOutputSize: setAiOutputSize,
    setFrameRate: setAiFrameRate,
    setSeed: setAiSeed,
    setKleinAlpha: setAiKleinAlpha,
    setKleinSteps: setAiKleinSteps,
    setUpscaleMode: setAiUpscaleMode,
    setAutoConnect: setAiAutoConnect,
    setStageSharpen: setAiStageSharpen,
    setStageScanlines: setAiStageScanlines,
    setStageVignette: setAiStageVignette,
    setStagePixelate: setAiStagePixelate,
    setStagePixelateSize: setAiStagePixelateSize,
    setFogIntensity,
    setRecordingResolution,
    updatePromptPreset,
  } = useAiSettingsStore();

  const aiOutputLong = Math.max(aiOutputWidth, aiOutputHeight);

  // BroadcastChannel to the /vj/stage tab
  const stageChannelRef = useRef<BroadcastChannel | null>(null);
  const stageFrameSeqRef = useRef<number>(0);
  useEffect(() => {
    const ch = openStageChannel();
    stageChannelRef.current = ch;

    const onMessage = (ev: MessageEvent<StageMsg>) => {
      if (ev.data?.type === "hello") {
        ch?.postMessage({ type: "prompt", prompt: aiPrompt });
      }
    };
    ch?.addEventListener("message", onMessage as EventListener);
    return () => {
      ch?.removeEventListener("message", onMessage as EventListener);
      ch?.close();
      stageChannelRef.current = null;
    };
  }, [aiPrompt]);

  useEffect(() => {
    stageChannelRef.current?.postMessage({ type: "prompt", prompt: aiPrompt });
  }, [aiPrompt]);

  // Build + sendText the settings payload NOW, reading the latest values
  // straight from the Zustand store. Zustand's set() is synchronous, so
  // any state mutation immediately before calling this is already visible
  // in getState(). We call this directly from hotkey / button handlers so
  // the sendText happens in the same event tick as the user action — no
  // React scheduler, no microtask, no requestAnimationFrame delay. That's
  // the difference between "click did nothing, click again works" and
  // "click registers instantly".
  const flushSettingsNow = useCallback(() => {
    if (!aiTransport.isConnected()) return;
    const s = useAiSettingsStore.getState();
    const aspect = s.outputWidth / s.outputHeight;
    const captureW =
      aspect >= 1
        ? s.captureSize
        : Math.max(16, Math.round(s.captureSize * aspect));
    const captureH =
      aspect >= 1
        ? Math.max(16, Math.round(s.captureSize / aspect))
        : s.captureSize;
    const payload: Record<string, unknown> = {
      prompt: s.prompt,
      seed: s.seed,
      captureWidth: captureW,
      captureHeight: captureH,
      width: s.outputWidth,
      height: s.outputHeight,
    };
    if (s.backend === "klein") {
      payload.alpha = s.kleinAlpha;
      payload.n_steps = s.kleinSteps;
    }
    aiTransport.sendText(JSON.stringify(payload));
  }, [aiTransport]);

  // Firing a preset = swap prompt + re-roll seed + send the payload right
  // now. Hotkeys (1-9) and chip clicks share this so hammering the same
  // preset gives fresh variations every time, and jumping between presets
  // always re-inits the noise.
  const firePreset = useCallback(
    (prompt: string) => {
      setAiPrompt(prompt);
      setAiSeed(Math.floor(Math.random() * 1_000_000));
      flushSettingsNow();
    },
    [setAiPrompt, setAiSeed, flushSettingsNow]
  );

  const fireRandomPreset = useCallback(() => {
    // Pull presets fresh from the store so rapid Space presses always see
    // the current list (e.g. right after the user edits a preset).
    const presets = useAiSettingsStore.getState().promptPresets;
    if (presets.length === 0) return;
    const idx = Math.floor(Math.random() * presets.length);
    firePreset(presets[idx].prompt);
  }, [firePreset]);

  // Re-roll the seed without changing the prompt. Spacebar uses this for
  // "give me a fresh variation of what's on screen" — keeps the user's chosen
  // mood, just shakes the noise. Number keys still swap prompts via firePreset.
  const rerollSeed = useCallback(() => {
    setAiSeed(Math.floor(Math.random() * 1_000_000));
    flushSettingsNow();
  }, [setAiSeed, flushSettingsNow]);

  // Klein α nudge — clamped 0..0.5, snapped to 2 decimals so the chip+slider
  // display the value cleanly. Used by both keyboard arrows and the on-screen
  // HotkeyBoard arrow caps. We pull the latest value from the store instead
  // of closing over `aiKleinAlpha` so rapid keypresses (especially with key
  // autorepeat held down) actually accumulate — the previous closure form
  // dropped every press that happened before React rendered the next value.
  const adjustAlpha = useCallback(
    (delta: number) => {
      const cur = useAiSettingsStore.getState().kleinAlpha;
      useAiSettingsStore
        .getState()
        .setKleinAlpha(Math.round((cur + delta) * 100) / 100);
      flushSettingsNow();
    },
    [flushSettingsNow]
  );

  // Toggle fog on/off. Reads current intensity fresh from the store at
  // toggle time so the latest slider value applies when switching on. Also
  // re-reads the engine's current state so repeated presses alternate
  // reliably regardless of intermediate UI renders.
  //
  // Debounced at 150 ms because multiple event sources can target the same
  // action in this codebase (physical "0" keydown, the HotkeyBoard cap's
  // onClick, the FogControl button's onClick). In dev with React Strict
  // Mode the keydown listener briefly registers twice during remount, and
  // a single keypress would flip the toggle back to off. 150 ms is well
  // below any realistic user intent for a fog heater.
  const lastFogToggleRef = useRef(0);
  const triggerFog = useCallback(() => {
    const now = performance.now();
    if (now - lastFogToggleRef.current < 150) return;
    lastFogToggleRef.current = now;
    const engine = lightingEngineRef.current;
    if (!engine) return;
    const wasActive = engine.isFogActive();
    const s = useAiSettingsStore.getState();
    engine.setFogActive(!wasActive, s.fogIntensity);
  }, []);

  // ── Recording control ────────────────────────────────────────────────────
  // The engine is created on first record so we don't probe MediaRecorder
  // capabilities for users who never hit the button. Deps are getter
  // functions: `getCanvas` reads the AI preview's WebGL canvas (the same
  // picture pushed to /vj/stage), `createAudioTap` opens a fresh sink on
  // the live audio graph. Both are evaluated at start() time, so the
  // engine doesn't capture stale references after a device switch.
  const handleStartRecording = useCallback(() => {
    setRecordingError("");
    if (!recordingEngineRef.current) {
      recordingEngineRef.current = new RecordingEngine({
        getCanvas: () => previewRendererRef.current?.getCanvas() ?? null,
        createAudioTap: () => audioEngineRef.current?.createAudioTap() ?? null,
      });
    }
    // Read the resolution fresh from the store so changes the user made
    // right before pressing record are honoured (the closure over the
    // hook value would lag behind any in-flight zustand update).
    const spec = getRecordingResolutionSpec(
      useAiSettingsStore.getState().recordingResolution
    );
    try {
      recordingEngineRef.current.start({
        outputWidth: spec.width,
        outputHeight: spec.height,
      });
      setIsRecording(true);
    } catch (err) {
      setRecordingError(
        err instanceof Error ? err.message : "Failed to start recording"
      );
    }
  }, []);

  const handleStopRecording = useCallback(async () => {
    const engine = recordingEngineRef.current;
    if (!engine) return;
    setIsFinalizingRecording(true);
    try {
      const result = await engine.stop();
      triggerRecordingDownload(result);
    } catch (err) {
      setRecordingError(
        err instanceof Error ? err.message : "Recording failed"
      );
    } finally {
      setIsRecording(false);
      setIsFinalizingRecording(false);
    }
  }, []);

  // Stable getter for the elapsed-time poll inside RecordingControl. Without
  // useCallback, every parent render swaps the prop identity and forces
  // RecordingControl's effect to tear down + re-establish its 4 Hz timer.
  const getRecordingElapsedMs = useCallback(
    () => recordingEngineRef.current?.getElapsedMs() ?? 0,
    []
  );

  // ── DMX console drawer ──────────────────────────────────────────────────
  // The DMX console used to be a full-width row jammed at the bottom of
  // the page. With AI preview, performance deck, and stage FX cards in
  // play it scrolled out of view, so the operator had to chase it during
  // a set. We promoted it to a slide-up drawer toggled from the SystemsBar
  // DMX chip — the panel becomes a true layer above the dashboard, never
  // out of reach, and the dashboard underneath gets its full vertical
  // budget back.
  //
  // Local state (not persisted): a fresh reload should always start with
  // the drawer closed so the user sees the full canvas immediately. The
  // DMX status chip stays visible in the top bar regardless.
  const [dmxOpen, setDmxOpen] = useState(false);
  const toggleDmxDrawer = useCallback(() => setDmxOpen((v) => !v), []);
  const closeDmxDrawer = useCallback(() => setDmxOpen(false), []);

  // Esc closes the drawer (only when open — otherwise we'd swallow Escape
  // for unrelated UI). Skip when the user is typing so a textarea
  // dismissal doesn't accidentally yank the panel away.
  useEffect(() => {
    if (!dmxOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const el = document.activeElement as HTMLElement | null;
      if (el) {
        const tag = el.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (el.isContentEditable) return;
      }
      e.preventDefault();
      setDmxOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dmxOpen]);

  // Global hotkeys for live performance. Skip when typing in input/textarea/select.
  useEffect(() => {
    const isTyping = () => {
      const el = document.activeElement as HTMLElement | null;
      if (!el) return false;
      const tag = el.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
      if (el.isContentEditable) return true;
      return false;
    };

    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTyping()) return;

      if (/^[1-9]$/.test(e.key)) {
        const idx = parseInt(e.key, 10) - 1;
        const preset = aiPromptPresets[idx];
        if (preset) {
          firePreset(preset.prompt);
          e.preventDefault();
        }
        return;
      }

      if (e.key === "0") {
        triggerFog();
        e.preventDefault();
        return;
      }

      if (e.key === " ") {
        // Re-roll the seed only — keep the current prompt. Gives the user a
        // fresh noise variation of the same intent. Number keys 1-9 still
        // change prompts. (Used to fire a random preset; was disorienting in
        // live sets where you've just dialed in a vibe and don't want it
        // replaced wholesale.)
        rerollSeed();
        e.preventDefault();
        return;
      }

      if (aiBackend === "klein") {
        if (e.key === "ArrowUp") {
          adjustAlpha(0.02);
          e.preventDefault();
        } else if (e.key === "ArrowDown") {
          adjustAlpha(-0.02);
          e.preventDefault();
        } else if (e.key === "ArrowRight") {
          adjustAlpha(0.01);
          e.preventDefault();
        } else if (e.key === "ArrowLeft") {
          adjustAlpha(-0.01);
          e.preventDefault();
        }
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    aiBackend,
    aiPromptPresets,
    firePreset,
    fireRandomPreset,
    adjustAlpha,
    triggerFog,
    rerollSeed,
  ]);

  const [aiStatus, setAiStatus] = useState<AiTransportStatus>("idle");
  const [aiLogs, setAiLogs] = useState<string[]>([]);
  const [aiImageUrl, setAiImageUrl] = useState<string | null>(null);
  const [aiGenTime, setAiGenTime] = useState<number | null>(null);
  // Per-stage timing breakdown emitted by inference_server.py — surfaces where
  // the frame budget is going (vae_encode vs transformer vs jpeg vs python glue).
  // Useful for spotting the next bottleneck without re-instrumenting the worker.
  const [aiTiming, setAiTiming] = useState<{
    decode_in_ms?: number;
    prompt_ms?: number;
    vae_encode_ms?: number;
    transformer_plus_decode_ms?: number;
    jpeg_ms?: number;
    total_ms?: number;
  } | null>(null);
  // Server-reported compile status. When the worker has to JIT-compile a new
  // (width, height) shape, every connected GPU is dark for ~150s. We surface
  // that state so the UI can show "compiling 512x288… ~150s" instead of a
  // mysteriously frozen output canvas.
  const [aiCompile, setAiCompile] = useState<{
    phase: BootPhase;
    width?: number;
    height?: number;
    n_steps?: number;
    iter?: number;
    total_iters?: number;
    elapsed_ms?: number;
    est_seconds?: number;
    started_at: number;
  } | null>(null);
  // Polled /healthz state — shows worker-readiness so the UI knows whether
  // the server is mid-bootup (still loading weights / compiling) vs ready.
  // Without this we couldn't distinguish "user hasn't pressed generate" from
  // "server is still spinning up the workers and frames physically can't flow yet".
  const [aiServer, setAiServer] = useState<{
    workerCount: number;
    readyCount: number;
  } | null>(null);

  // DMX/Lighting UI state
  const [dmxStatus, setDmxStatus] = useState<DmxStatus>("disconnected");
  const [dmxSupported, setDmxSupported] = useState<boolean>(true);
  const [fixtureValues, setFixtureValues] = useState<Map<string, Uint8Array>>(
    new Map()
  );
  const persistedProfileId = useAiSettingsStore(
    (s) => s.selectedFixtureProfileId
  );
  const setPersistedProfileId = useAiSettingsStore(
    (s) => s.setSelectedFixtureProfileId
  );
  const [selectedProfileId, setSelectedProfileIdLocal] = useState<string>(
    persistedProfileId || FIXTURE_PROFILES[0].id
  );
  // Mirror to the persisted store on every change so the "+ add" picker
  // remembers what the user last had selected across reloads.
  const setSelectedProfileId = useCallback(
    (value: string) => {
      setSelectedProfileIdLocal(value);
      setPersistedProfileId(value);
    },
    [setPersistedProfileId]
  );

  const fixtures = useFixtures();
  const lightingEnabled = useLightingStore((s) => s.enabled);
  const setLightingEnabled = useLightingStore((s) => s.setEnabled);
  const {
    addFixture,
    removeFixture,
    updateFixtureAddress,
    updateFixtureStrobeMode,
    updateFixtureStrobeThreshold,
    updateFixtureStrobeMax,
    updateFixtureColorMode,
    updateFixtureSolidColor,
    updateFixtureProfile,
    updateFixtureDimmerMode,
    updateFixtureManualDimmer,
  } = useLightingStore();

  // Count fixtures currently putting out DMX — "active" = any channel > 0.
  const dmxActiveCount = useMemo(() => {
    let n = 0;
    for (const [, vals] of fixtureValues) {
      for (const v of vals) {
        if (v > 0) {
          n++;
          break;
        }
      }
    }
    return n;
  }, [fixtureValues]);

  const selectedDeviceLabel = useMemo(() => {
    if (audioSource === "system") return "🖥 System audio";
    if (!selectedDeviceId) return devices[0]?.label;
    return devices.find((d) => d.deviceId === selectedDeviceId)?.label;
  }, [audioSource, selectedDeviceId, devices]);

  // ============================================================================
  // Device enumeration
  // ============================================================================

  const fetchDevices = useCallback(async () => {
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = allDevices
        .filter((d) => d.kind === "audioinput")
        .map((d) => ({
          deviceId: d.deviceId,
          label: d.label || `Microphone ${d.deviceId.slice(0, 8)}`,
        }));
      setDevices(audioInputs);
    } catch {
      console.error("Failed to enumerate devices");
    }
  }, []);

  // ============================================================================
  // Lighting frame handlers
  // ============================================================================

  const handleLightingFrame = useCallback(
    (frame: LightingFrame) => {
      const newValues = new Map<string, Uint8Array>();
      for (const fixture of fixtures) {
        const base = fixture.address - 1;
        const channelCount = fixture.profile.channels.length;
        const values = new Uint8Array(channelCount);
        for (let i = 0; i < channelCount; i++) {
          values[i] = frame.universe[base + i];
        }
        newValues.set(fixture.id, values);
      }
      setFixtureValues(newValues);
    },
    [fixtures]
  );

  // Read lightingEnabled fresh from the store on every frame instead of
  // closing over the React state — that way flipping the master switch
  // takes effect on the very next 30 Hz tick without needing the
  // LightingEngine to be torn down and rebuilt.
  const handleDmxFrame = useCallback((frame: LightingFrame) => {
    if (!useLightingStore.getState().enabled) return;
    const dmx = dmxOutputRef.current;
    if (dmx && dmx.isConnected()) {
      dmx.sendUniverse(frame.universe);
    }
  }, []);

  // ============================================================================
  // Audio/Visual/Lighting initialization
  // ============================================================================

  const initAudio = useCallback(
    async (
      source?: string | MediaStream,
      fixtureList?: FixtureInstance[]
    ) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      // Drop any in-flight recording before tearing down the audio graph —
      // otherwise the recorder would keep the now-orphaned audio tracks alive
      // (silent for the rest of the recording) and dangle a destination node
      // off the about-to-close AudioContext.
      if (recordingEngineRef.current?.isRecording()) {
        recordingEngineRef.current.cancel();
        setIsRecording(false);
        setIsFinalizingRecording(false);
      }

      lightingEngineRef.current?.stop();
      lightingEngineRef.current = null;
      visualEngineRef.current?.stop();
      visualEngineRef.current = null;
      audioEngineRef.current?.destroy();
      audioEngineRef.current = null;

      setStatus("requesting");
      setErrorMessage("");

      try {
        const audioEngine = new AudioEngine();
        await audioEngine.init(source);
        audioEngineRef.current = audioEngine;

        const visualEngine = new VisualEngine(canvas, audioEngine, SCENES);
        visualEngineRef.current = visualEngine;

        const lightingEngine = new LightingEngine(
          canvas,
          fixtureList ?? fixtures,
          { tickHz: 30 }
        );
        lightingEngine.setAudioEngine(audioEngine);
        lightingEngineRef.current = lightingEngine;

        lightingEngine.onFrame(handleLightingFrame);
        lightingEngine.onFrame(handleDmxFrame);

        // Restore the persisted scene before starting so the first frame
        // already runs the user's last-selected visualizer instead of the
        // default. setSceneById is a no-op if the id no longer exists.
        if (persistedSceneId) visualEngine.setSceneById(persistedSceneId);

        visualEngine.start();
        lightingEngine.start();

        setStatus("running");
        setCurrentSceneId(visualEngine.getCurrentScene().id);
        fetchDevices();
      } catch (err) {
        setStatus("error");
        setErrorMessage(
          err instanceof Error ? err.message : "Failed to initialize audio"
        );
      }
    },
    [fetchDevices, handleLightingFrame, handleDmxFrame, fixtures, persistedSceneId]
  );

  // ============================================================================
  // UI event handlers
  // ============================================================================

  // System audio capture via getDisplayMedia. Browser shows its share-picker
  // (tab / window / entire screen + an "include audio" checkbox); we keep
  // only the audio track and feed it to AudioEngine. The Chromium-specific
  // hints (`systemAudio`, `windowAudio`, `suppressLocalAudioPlayback`) tell
  // Chrome we want clean, unmuted system audio — they're ignored on engines
  // that don't recognise them, so the call still works on any browser that
  // implements getDisplayMedia.
  const initSystemAudio = useCallback(async () => {
    try {
      // Cast: TS lib.dom doesn't yet know `systemAudio` / `windowAudio` /
      // `suppressLocalAudioPlayback` (Chromium proposal). They're real and
      // documented; we just don't have types for them.
      const constraints = {
        video: true,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          suppressLocalAudioPlayback: false,
        },
        systemAudio: "include",
        windowAudio: "system",
      } as DisplayMediaStreamOptions;
      const stream = await navigator.mediaDevices.getDisplayMedia(constraints);

      // Drop video — we asked for it because Chrome requires it, but we don't
      // need it. Stop + remove so nothing else accidentally renders the
      // preview thumbnail and we save a tiny bit of decode work.
      for (const t of stream.getVideoTracks()) {
        stream.removeTrack(t);
        t.stop();
      }

      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) {
        // User picked Share but didn't tick the audio checkbox in the picker.
        stream.getTracks().forEach((t) => t.stop());
        setStatus("error");
        setErrorMessage(
          "No audio in the shared source — re-share and check the audio box (browser only offers it for tabs and full screen)."
        );
        return;
      }

      // When the user clicks the browser's "Stop sharing" bar, the audio
      // track ends. Auto-fall back to the persisted device so visuals don't
      // go silent the moment they release the share.
      const audioTrack = audioTracks[0];
      audioTrack.addEventListener("ended", () => {
        setSelectedDeviceId(persistedDeviceId);
        setAudioSource("device");
        initAudio(persistedDeviceId || undefined);
      });

      setSelectedDeviceId(SYSTEM_AUDIO_VALUE);
      setAudioSource("system");
      await initAudio(stream);
    } catch (err) {
      // NotAllowedError = user dismissed the picker. Quietly revert the
      // dropdown without changing the running audio. Anything else is real.
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setSelectedDeviceId(persistedDeviceId);
        return;
      }
      setStatus("error");
      setErrorMessage(
        err instanceof Error ? err.message : "System audio unavailable"
      );
    }
  }, [initAudio, persistedDeviceId]);

  const handleDeviceChange = useCallback(
    (deviceId: string) => {
      if (deviceId === SYSTEM_AUDIO_VALUE) {
        // Don't persist the sentinel — getDisplayMedia must be re-triggered
        // each session anyway. We update selectedDeviceId optimistically;
        // initSystemAudio reverts it on cancel.
        setSelectedDeviceId(SYSTEM_AUDIO_VALUE);
        void initSystemAudio();
        return;
      }
      setAudioSource("device");
      setSelectedDeviceId(deviceId);
      setPersistedDeviceId(deviceId);
      initAudio(deviceId || undefined);
    },
    [initAudio, initSystemAudio, setPersistedDeviceId]
  );

  const handleSceneChange = useCallback(
    (sceneId: string) => {
      const engine = visualEngineRef.current;
      if (engine && engine.setSceneById(sceneId)) {
        setCurrentSceneId(sceneId);
        setPersistedSceneId(sceneId);
      }
    },
    [setPersistedSceneId]
  );

  const handleDmxConnect = useCallback(async () => {
    if (!dmxOutputRef.current) {
      dmxOutputRef.current = new DmxOutput();
    }
    setDmxStatus("connecting");
    try {
      await dmxOutputRef.current.connect();
      setDmxStatus("connected");
    } catch {
      setDmxStatus("disconnected");
    }
  }, []);

  const handleDmxDisconnect = useCallback(async () => {
    const dmx = dmxOutputRef.current;
    if (dmx) {
      await dmx.disconnect();
      setDmxStatus("disconnected");
    }
  }, []);

  const handleDmxReconnect = useCallback(async () => {
    const dmx = dmxOutputRef.current;
    if (!dmx) return;
    setDmxStatus("connecting");
    const ok = await dmx.reconnect();
    setDmxStatus(ok ? "connected" : "disconnected");
  }, []);

  const handleFixtureAddressChange = useCallback(
    (fixtureId: string, newAddress: number) => {
      updateFixtureAddress(fixtureId, newAddress);
      lightingEngineRef.current?.updateFixtureAddress(fixtureId, newAddress);
    },
    [updateFixtureAddress]
  );

  const handleStrobeModeChange = useCallback(
    (fixtureId: string, mode: StrobeMode) => {
      updateFixtureStrobeMode(fixtureId, mode);
      lightingEngineRef.current?.updateFixtureStrobeMode(fixtureId, mode);
    },
    [updateFixtureStrobeMode]
  );

  const handleStrobeThresholdChange = useCallback(
    (fixtureId: string, threshold: number) => {
      updateFixtureStrobeThreshold(fixtureId, threshold);
      lightingEngineRef.current?.updateFixtureStrobeThreshold(
        fixtureId,
        threshold
      );
    },
    [updateFixtureStrobeThreshold]
  );

  const handleStrobeMaxChange = useCallback(
    (fixtureId: string, max: number) => {
      updateFixtureStrobeMax(fixtureId, max);
      lightingEngineRef.current?.updateFixtureStrobeMax(fixtureId, max);
    },
    [updateFixtureStrobeMax]
  );

  const handleColorModeChange = useCallback(
    (fixtureId: string, mode: ColorMode) => {
      updateFixtureColorMode(fixtureId, mode);
      lightingEngineRef.current?.updateFixtureColorMode(fixtureId, mode);
    },
    [updateFixtureColorMode]
  );

  const handleSolidColorChange = useCallback(
    (fixtureId: string, color: { r: number; g: number; b: number }) => {
      updateFixtureSolidColor(fixtureId, color);
      lightingEngineRef.current?.updateFixtureSolidColor(fixtureId, color);
    },
    [updateFixtureSolidColor]
  );

  const handleDimmerModeChange = useCallback(
    (fixtureId: string, mode: DimmerMode) => {
      updateFixtureDimmerMode(fixtureId, mode);
      lightingEngineRef.current?.updateFixtureDimmerMode(fixtureId, mode);
    },
    [updateFixtureDimmerMode]
  );

  const handleManualDimmerChange = useCallback(
    (fixtureId: string, value: number) => {
      updateFixtureManualDimmer(fixtureId, value);
      lightingEngineRef.current?.updateFixtureManualDimmer(fixtureId, value);
    },
    [updateFixtureManualDimmer]
  );

  const handleAddFixture = useCallback(() => {
    addFixture(selectedProfileId);
  }, [addFixture, selectedProfileId]);

  const handleRemoveFixture = useCallback(
    (fixtureId: string) => {
      removeFixture(fixtureId);
    },
    [removeFixture]
  );

  const handleFixtureProfileChange = useCallback(
    (fixtureId: string, profileId: string) => {
      updateFixtureProfile(fixtureId, profileId);
    },
    [updateFixtureProfile]
  );

  // ============================================================================
  // Effects
  // ============================================================================

  useEffect(() => {
    const handleStatus = (s: AiTransportStatus) => {
      setAiStatus(s);
      stageChannelRef.current?.postMessage({ type: "connection", status: s });
    };
    aiTransport.onStatusChange(handleStatus);

    const handleFrame = (frame: AiIncomingFrame) => {
      if (frame.kind === "text") {
        try {
          const data = JSON.parse(frame.message);
          if (data.type === "stats" && data.gen_time_ms) {
            setAiGenTime(data.gen_time_ms);
            if (data.timing) setAiTiming(data.timing);
            return;
          }
          // Boot-phase messages from inference_server.py: "loading_weights",
          // "applying_fp8", "registering_compile_stubs". These fire BEFORE
          // the warmup events. We surface them in the overlay so the user
          // sees "Loading the AI model" instead of a blank "preparing workers"
          // box for two minutes.
          if (data.type === "phase") {
            const stage = data.stage as string | undefined;
            // Transient phases ("loaded", anything we don't render) clear nothing.
            const PHASE_MAP: Record<string, BootPhase> = {
              loading_weights: "loading_weights",
              applying_fp8: "applying_fp8",
              registering_compile_stubs: "registering_compile_stubs",
            };
            if (stage && PHASE_MAP[stage]) {
              setAiCompile({
                phase: PHASE_MAP[stage],
                est_seconds: data.est_seconds ?? 30,
                started_at: Date.now(),
              });
            }
            return;
          }
          if (data.type === "compile") {
            // Warmup phase. Server emits "compiling" once, then
            // "compiling_progress" per warmup iteration, then "warmed" when
            // ready. We accept progress events even if we missed the initial
            // "compiling" emit (common: client connects mid-warmup).
            if (data.status === "compiling" || data.status === "compiling_progress") {
              setAiCompile((prev) => ({
                phase: "warming_up",
                width: data.width,
                height: data.height,
                n_steps: data.n_steps,
                iter: data.iter,
                total_iters: data.total_iters,
                elapsed_ms: data.elapsed_ms,
                est_seconds: data.est_seconds,
                // If we're seeing a progress event without a prior compiling
                // event, back-calculate started_at from the server-reported
                // elapsed_ms so the countdown stays accurate.
                started_at:
                  prev?.phase === "warming_up" && prev?.started_at
                    ? prev.started_at
                    : Date.now() - (typeof data.elapsed_ms === "number" ? data.elapsed_ms : 0),
              }));
            } else if (data.status === "warmed") {
              setAiCompile(null);
            }
            return;
          }
        } catch {
          // not JSON, log it
        }
        setAiLogs((prev) => [...prev, `← ${frame.message}`].slice(-20));
        return;
      }

      const url = URL.createObjectURL(frame.blob);
      setAiImageUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      frame.blob
        .arrayBuffer()
        .then((buf) => {
          stageChannelRef.current?.postMessage({
            type: "frame",
            bytes: buf,
            width: aiOutputWidth,
            height: aiOutputHeight,
            seq: ++stageFrameSeqRef.current,
          });
        })
        .catch(() => {
          /* ignore */
        });

      // Off-thread: decode the JPEG into an ImageBitmap. The single decoded
      // bitmap drives both consumers in this branch — preview WebGL renderer
      // (sharpened display) AND the 1×1 lighting source canvas (averaged
      // colour for DMX). Fire-and-forget: if a decode is slow the next frame
      // supersedes it. The bitmap is closed only after both consumers are
      // done with it.
      const aiSrc = aiSourceCanvasRef.current;
      createImageBitmap(frame.blob)
        .then((bitmap) => {
          previewRendererRef.current?.drawBitmap(bitmap);
          if (aiSrc) {
            const ctx = aiSrc.getContext("2d", { willReadFrequently: true });
            if (ctx) {
              ctx.imageSmoothingEnabled = true;
              ctx.imageSmoothingQuality = "low";
              ctx.drawImage(bitmap, 0, 0, aiSrc.width, aiSrc.height);
            }
          }
          bitmap.close?.();
        })
        .catch(() => {
          /* swallow: stale frame, next one will retry */
        });
    };

    aiTransport.onFrame(handleFrame);
    return () => {
      aiTransport.offFrame(handleFrame);
      aiTransport.offStatusChange(handleStatus);
    };
  }, [aiTransport, aiOutputWidth, aiOutputHeight]);

  useEffect(() => {
    return () => {
      if (aiImageUrl) URL.revokeObjectURL(aiImageUrl);
    };
  }, [aiImageUrl]);

  // Poll /healthz so we know whether the server's workers are ready. Without
  // this, the UI can't distinguish "user hasn't pressed generate" from
  // "workers are still booting". We poll every 2 s while connected (or
  // attempting to be) and stop when the server is fully ready *and* we've
  // received our first frame — at that point health-state is irrelevant.
  useEffect(() => {
    if (!aiHealthUrl) return;
    if (aiStatus !== "connected" && aiStatus !== "connecting") {
      setAiServer(null);
      return;
    }
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch(aiHealthUrl, { cache: "no-store" });
        if (!r.ok) throw new Error(String(r.status));
        const j = (await r.json()) as {
          workerCount?: number;
          readyCount?: number;
        };
        if (cancelled) return;
        if (
          typeof j.workerCount === "number" &&
          typeof j.readyCount === "number"
        ) {
          setAiServer({ workerCount: j.workerCount, readyCount: j.readyCount });
        }
      } catch {
        if (!cancelled) setAiServer(null);
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [aiHealthUrl, aiStatus]);

  const aiDebugCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Preview renderer — same WebGL2 sharpen pipeline as the stage page, so
  // the dashboard preview reflects exactly what the projector will look
  // like (sharpen + scanlines/vignette via the canvas-frame CSS overlay).
  const previewRendererRef = useRef<StageRendererHandle>(null);

  // Hidden 1×1 canvas that the LightingEngine samples when the AI backend
  // is connected. Each AI frame is downscaled into it via createImageBitmap
  // off the main thread, so the WebRTC frame path stays unblocked. Result:
  // every fixture takes its colour from the average colour of the AI image
  // — a "middle value" representative tint that follows whatever the model
  // is generating, without per-fixture pixel reads against a large canvas.
  const aiSourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  if (typeof document !== "undefined" && !aiSourceCanvasRef.current) {
    const c = document.createElement("canvas");
    c.width = 1;
    c.height = 1;
    aiSourceCanvasRef.current = c;
  }

  const aiFrameSenderRef = useRef<{
    running: boolean;
    captureCanvas: HTMLCanvasElement | null;
    captureCtx: CanvasRenderingContext2D | null;
    debugCtx: CanvasRenderingContext2D | null;
    lastFrameTime: number;
    resolution: number;
    frameCount: number;
    pendingEncode: boolean;
  }>({
    running: false,
    captureCanvas: null,
    captureCtx: null,
    debugCtx: null,
    lastFrameTime: 0,
    resolution: 0,
    frameCount: 0,
    pendingEncode: false,
  });

  const aiFrameLoop = useCallback(() => {
    const sender = aiFrameSenderRef.current;
    if (!sender.running) return;

    requestAnimationFrame(aiFrameLoop);

    const now = performance.now();
    const frameInterval = 1000 / aiFrameRate;

    if (now - sender.lastFrameTime < frameInterval) {
      return;
    }

    const src = canvasRef.current;
    if (!src) {
      return;
    }

    const outAspect = aiOutputWidth / aiOutputHeight;
    const capW =
      outAspect >= 1 ? aiCaptureSize : Math.max(16, Math.round(aiCaptureSize * outAspect));
    const capH =
      outAspect >= 1 ? Math.max(16, Math.round(aiCaptureSize / outAspect)) : aiCaptureSize;
    const resolutionKey = capW * 10000 + capH;
    if (!sender.captureCanvas || sender.resolution !== resolutionKey) {
      sender.captureCanvas = document.createElement("canvas");
      sender.captureCanvas.width = capW;
      sender.captureCanvas.height = capH;
      sender.captureCtx = sender.captureCanvas.getContext("2d");
      sender.resolution = resolutionKey;
    }

    const ctx = sender.captureCtx;
    if (!ctx) {
      return;
    }

    let srcCropW = src.width;
    let srcCropH = src.height;
    if (src.width / src.height > outAspect) {
      srcCropW = src.height * outAspect;
    } else {
      srcCropH = src.width / outAspect;
    }
    const srcX = (src.width - srcCropW) / 2;
    const srcY = (src.height - srcCropH) / 2;

    ctx.drawImage(
      src,
      srcX, srcY, srcCropW, srcCropH,
      0, 0, capW, capH
    );

    sender.lastFrameTime = now;
    sender.frameCount++;

    const debugCanvas = aiDebugCanvasRef.current;
    if (debugCanvas) {
      if (!sender.debugCtx || debugCanvas.width !== capW || debugCanvas.height !== capH) {
        debugCanvas.width = capW;
        debugCanvas.height = capH;
        sender.debugCtx = debugCanvas.getContext("2d");
      }
      sender.debugCtx?.drawImage(sender.captureCanvas, 0, 0);
    }

    if (!aiTransport.isConnected() || !aiTransport.canSend(256 * 1024)) {
      return;
    }
    if (sender.pendingEncode) {
      return;
    }

    sender.pendingEncode = true;
    sender.captureCanvas.toBlob(
      (blob) => {
        sender.pendingEncode = false;
        if (!blob || !aiTransport.isConnected()) return;
        blob
          .arrayBuffer()
          .then((buf) => {
            if (!aiTransport.isConnected()) return;
            aiTransport.sendBinary(buf);
          })
          .catch(() => {
            /* network hiccup; next frame will try again */
          });
      },
      "image/jpeg",
      0.85
    );
  }, [aiTransport, aiCaptureSize, aiFrameRate, aiOutputWidth, aiOutputHeight]);

  useEffect(() => {
    // Catch-all for non-hotkey state changes (slider drags, backend swap,
    // resolution pick, etc.) that still need to sync to the server. Hotkey
    // handlers send directly via flushSettingsNow, so by the time this
    // effect runs for a hotkey the payload is already on the wire — this
    // is the safety net for everything else.
    flushSettingsNow();
  }, [
    flushSettingsNow,
    aiBackend,
    aiPrompt,
    aiSeed,
    aiCaptureSize,
    aiOutputWidth,
    aiOutputHeight,
    aiKleinAlpha,
    aiKleinSteps,
    aiTransport,
    // Re-fire when status flips to "connected" so the server picks up our
    // current resolution / seed / etc. instead of using its defaults.
    // Critical for auto-connect: settings can't push before the channel exists.
    aiStatus,
  ]);

  useEffect(() => {
    const sender = aiFrameSenderRef.current;

    if (aiSendFrames || aiShowCaptureDebug) {
      if (!sender.running) {
        sender.running = true;
        sender.lastFrameTime = 0;
        requestAnimationFrame(aiFrameLoop);
      }
    } else {
      sender.running = false;
    }

    return () => {
      sender.running = false;
    };
  }, [aiSendFrames, aiShowCaptureDebug, aiStatus, aiFrameLoop, aiTransport]);

  useEffect(() => {
    if (!showDebug) return;

    const interval = setInterval(() => {
      const engine = audioEngineRef.current;
      if (engine) {
        setDebugFeatures(engine.getLatestFeatures());
      }
    }, 100);

    return () => clearInterval(interval);
  }, [showDebug]);

  useEffect(() => {
    const supported = DmxOutput.isSupported();
    setDmxSupported(supported);
    if (!supported) {
      setDmxStatus("unsupported");
      return;
    }

    const tryAutoConnect = async () => {
      if (!dmxOutputRef.current) {
        dmxOutputRef.current = new DmxOutput();
      }
      // Skip if already connected — hotplug events can fire on every USB
      // change, including unrelated devices, and we don't want to re-claim
      // an interface we already hold.
      if (dmxOutputRef.current.isConnected()) return;
      setDmxStatus("connecting");
      const connected = await dmxOutputRef.current.autoConnect();
      setDmxStatus(connected ? "connected" : "disconnected");
    };

    // Initial attempt for already-paired devices.
    tryAutoConnect();

    // Hotplug: WebUSB fires `connect` when a paired device is plugged in
    // (or its usbd re-enumerates). Re-running autoConnect picks it up
    // without the user having to click anything. `disconnect` flips the
    // status back so the UI reflects reality.
    const usb = navigator.usb;
    if (!usb?.addEventListener) return;
    const onConnect = () => {
      void tryAutoConnect();
    };
    const onDisconnect = () => {
      const dmx = dmxOutputRef.current;
      if (dmx && !dmx.isConnected()) {
        setDmxStatus("disconnected");
      }
    };
    usb.addEventListener("connect", onConnect);
    usb.addEventListener("disconnect", onDisconnect);
    return () => {
      usb.removeEventListener("connect", onConnect);
      usb.removeEventListener("disconnect", onDisconnect);
    };
  }, []);

  useEffect(() => {
    const engine = lightingEngineRef.current;
    if (engine && status === "running") {
      engine.stop();
      const canvas = canvasRef.current;
      if (canvas) {
        const newEngine = new LightingEngine(canvas, fixtures, { tickHz: 30 });
        newEngine.setAudioEngine(audioEngineRef.current!);
        newEngine.onFrame(handleLightingFrame);
        newEngine.onFrame(handleDmxFrame);
        newEngine.start();
        lightingEngineRef.current = newEngine;
      }
    }
  }, [fixtures, status, handleLightingFrame, handleDmxFrame]);

  // Swap the LightingEngine's sample source when the AI backend connects /
  // disconnects. While AI is connected we want fixtures to follow the
  // generated frame's tint (via the 1×1 aiSourceCanvas); otherwise we sample
  // the waveform canvas as before. Re-runs after fixtures change too, so a
  // freshly recreated engine picks up the right source.
  useEffect(() => {
    const engine = lightingEngineRef.current;
    if (!engine) return;
    const useAi =
      aiStatus === "connected" && aiSourceCanvasRef.current !== null;
    const target = useAi
      ? aiSourceCanvasRef.current!
      : canvasRef.current;
    if (target) engine.setSourceCanvas(target);
  }, [aiStatus, fixtures]);

  useEffect(() => {
    // Probe enumerateDevices first so we can downgrade gracefully when the
    // persisted device isn't currently connected. Otherwise getUserMedia's
    // `{ exact: deviceId }` constraint would throw OverconstrainedError and
    // we'd land in an error state on every reload until the user re-plugs.
    // The hotplug effect below picks the persisted device up the moment it
    // appears.
    (async () => {
      let initial = persistedDeviceId || undefined;
      if (initial) {
        try {
          const all = await navigator.mediaDevices.enumerateDevices();
          const found = all.some(
            (d) => d.kind === "audioinput" && d.deviceId === initial
          );
          if (!found) initial = undefined;
        } catch {
          initial = undefined;
        }
      }
      await initAudio(initial, fixtures);
      fetchDevices();
    })();

    return () => {
      // Cancel an in-flight recording first so we don't leave the audio
      // destination node connected when the AudioContext closes underneath.
      recordingEngineRef.current?.cancel();
      lightingEngineRef.current?.stop();
      visualEngineRef.current?.stop();
      audioEngineRef.current?.destroy();
      dmxOutputRef.current?.disconnect();
      void aiTransport.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-select the persisted device when it (re)appears mid-session — e.g.
  // the user plugs in their USB interface after the app was already open.
  // Browser fires `devicechange` on connect/disconnect; we re-enumerate and
  // switch over only if we're not already using that device, to avoid
  // gratuitously restarting the audio graph on unrelated hotplug events.
  useEffect(() => {
    const md = navigator.mediaDevices;
    if (!md?.addEventListener) return;
    const onChange = async () => {
      try {
        const all = await md.enumerateDevices();
        const inputs = all.filter((d) => d.kind === "audioinput");
        setDevices(
          inputs.map((d) => ({
            deviceId: d.deviceId,
            label: d.label || `Microphone ${d.deviceId.slice(0, 8)}`,
          }))
        );
        // Don't yank a live system-audio capture away just because the
        // persisted USB mic appeared: the user explicitly chose to share
        // system audio, hotplug events shouldn't override that.
        if (audioSource === "system") return;
        if (!persistedDeviceId) return;
        const present = inputs.some((d) => d.deviceId === persistedDeviceId);
        if (present && persistedDeviceId !== selectedDeviceId) {
          setSelectedDeviceId(persistedDeviceId);
          initAudio(persistedDeviceId);
        }
      } catch {
        /* enumerateDevices can throw in private windows; nothing to do */
      }
    };
    md.addEventListener("devicechange", onChange);
    return () => md.removeEventListener("devicechange", onChange);
  }, [persistedDeviceId, selectedDeviceId, audioSource, initAudio]);

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <div className="w-full min-h-screen flex flex-col">
      <SystemsBar
        audioStatus={status}
        audioDeviceLabel={selectedDeviceLabel}
        aiStatus={aiStatus}
        aiBackend={aiBackend}
        onBackendChange={(b) => {
          // Switching backend tears down the current channel — auto-connect
          // (if on) re-opens against the new URL on the next render.
          void aiTransport.stop();
          setAiBackend(b);
        }}
        aiAutoConnect={aiAutoConnect}
        onAutoConnectChange={setAiAutoConnect}
        onConnect={() => void aiTransport.start()}
        onDisconnect={() => void aiTransport.stop()}
        aiGenTimeMs={aiStatus === "connected" ? aiGenTime : null}
        aiTiming={aiStatus === "connected" ? aiTiming : null}
        dmxStatus={dmxStatus}
        dmxFixtureCount={fixtures.length}
        dmxActiveCount={dmxActiveCount}
        lightingEnabled={lightingEnabled}
        dmxOpen={dmxOpen}
        onToggleDmx={toggleDmxDrawer}
        isRecording={isRecording}
        isFinalizingRecording={isFinalizingRecording}
        recordingSupported={recordingSupported}
        getRecordingElapsedMs={getRecordingElapsedMs}
        onStartRecording={handleStartRecording}
        onStopRecording={handleStopRecording}
        recordingResolution={recordingResolution}
        onRecordingResolutionChange={setRecordingResolution}
      />

      {/* Error banner */}
      {status === "error" && errorMessage && (
        <div className="mx-4 mt-3 text-sm font-mono text-[color:var(--vj-error)] bg-[color-mix(in_srgb,var(--vj-error)_10%,transparent)] border border-[color:var(--vj-error)] rounded px-3 py-2 shadow-[0_0_18px_-6px_var(--vj-error)]">
          ⚠ {errorMessage}
        </div>
      )}

      {/* Recording errors get their own dismissible banner so the audio
          error banner above stays focused on its own concern. Click the
          message to clear it once the user has read it. */}
      {recordingError && (
        <button
          type="button"
          onClick={() => setRecordingError("")}
          className="mx-4 mt-3 text-left text-sm font-mono text-[color:var(--vj-error)] bg-[color-mix(in_srgb,var(--vj-error)_10%,transparent)] border border-[color:var(--vj-error)] rounded px-3 py-2 shadow-[0_0_18px_-6px_var(--vj-error)]"
          title="Click to dismiss"
        >
          ⚠ recording: {recordingError}
        </button>
      )}

      {/* Top section — INPUT (col-3, narrow) + MAIN (col-9, the working
          surface). Stage FX sits in the main column under the Performance
          Deck so it stays one glance away without eating its own column.
          Lighting/DMX moves below the grid as a full-width row — its
          fixture list grows tall and the wide footprint actually helps. */}
      <div className="flex-1 w-full grid grid-cols-1 xl:grid-cols-12 gap-x-4 gap-y-2 p-2">
        <section className="xl:col-span-3 flex flex-col gap-2 min-w-0">
          <WaveformSourceCard
            canvasRef={canvasRef}
            status={status}
            scenes={SCENES}
            currentSceneId={currentSceneId}
            onSceneChange={handleSceneChange}
            systemAudioSupported={systemAudioSupported}
            systemAudioValue={SYSTEM_AUDIO_VALUE}
            selectedDeviceId={selectedDeviceId}
            devices={devices}
            onDeviceChange={handleDeviceChange}
            showDebug={showDebug}
            setShowDebug={setShowDebug}
            debugFeatures={debugFeatures}
          />

          <StageFxCard
            sharpen={aiStageSharpen}
            onSharpenChange={setAiStageSharpen}
            scanlines={aiStageScanlines}
            onScanlinesChange={setAiStageScanlines}
            vignette={aiStageVignette}
            onVignetteChange={setAiStageVignette}
            pixelate={aiStagePixelate}
            onPixelateChange={setAiStagePixelate}
            pixelateSize={aiStagePixelateSize}
            onPixelateSizeChange={setAiStagePixelateSize}
          />
        </section>

        <section className="xl:col-span-9 flex flex-col gap-2 min-w-0">
          <AiConsoleCard
            aiTransport={aiTransport}
            aiStatus={aiStatus}
            previewRendererRef={previewRendererRef}
            aiImageUrl={aiImageUrl}
            aiCompile={aiCompile}
            aiServer={aiServer}
            aiSendFrames={aiSendFrames}
            onSendFramesChange={setAiSendFrames}
            aiOutputWidth={aiOutputWidth}
            aiOutputHeight={aiOutputHeight}
            onOutputSizeChange={setAiOutputSize}
            aiBackendKlein={aiBackend === "klein"}
            aiKleinAlpha={aiKleinAlpha}
            onKleinAlphaChange={setAiKleinAlpha}
            aiKleinSteps={aiKleinSteps}
            onKleinStepsChange={setAiKleinSteps}
            aiCaptureSize={aiCaptureSize}
            onCaptureSizeChange={setAiCaptureSize}
            aiOutputLong={aiOutputLong}
            aiStageSharpen={aiStageSharpen}
            aiStagePixelate={aiStagePixelate}
            aiStagePixelateSize={aiStagePixelateSize}
            aiPrompt={aiPrompt}
            onPromptChange={setAiPrompt}
            aiFrameRate={aiFrameRate}
            onFrameRateChange={setAiFrameRate}
            aiSeed={aiSeed}
            onSeedChange={setAiSeed}
            aiUpscaleMode={aiUpscaleMode}
            onUpscaleModeChange={setAiUpscaleMode}
            aiShowCaptureDebug={aiShowCaptureDebug}
            onShowCaptureDebugChange={setAiShowCaptureDebug}
            aiDebugCanvasRef={aiDebugCanvasRef}
            aiLogs={aiLogs}
          />

          <PerformanceDeckCard
            presets={aiPromptPresets}
            activePrompt={aiPrompt}
            onFirePreset={firePreset}
            onUpdatePreset={updatePromptPreset}
            onRandom={fireRandomPreset}
            onFireFog={triggerFog}
            alpha={aiBackend === "klein" ? aiKleinAlpha : undefined}
            onAlphaDelta={aiBackend === "klein" ? adjustAlpha : undefined}
          />
        </section>
      </div>

      {/* DMX Console Drawer — slide-up surface, toggled via the SystemsBar
          DMX chip. Rendered after the dashboard so its fixed-position
          drawer overlays everything else, but underneath the sticky
          SystemsBar (z-30 vs drawer z-25) so the chip stays clickable
          when the drawer is open. We always render the markup (toggling
          `--closed` class) so the slide-out animation can play; the
          aria-hidden state and pointer-events:none on the closed class
          keep it inert. */}
      <aside
        id="vj-dmx-drawer"
        role="dialog"
        aria-modal="false"
        aria-label="DMX lighting console"
        aria-hidden={!dmxOpen}
        className={`vj-drawer ${dmxOpen ? "" : "vj-drawer--closed"}`}
      >
        <LightingPanel
          enabled={lightingEnabled}
          onSetEnabled={setLightingEnabled}
          dmxStatus={dmxStatus}
          dmxSupported={dmxSupported}
          onDmxConnect={handleDmxConnect}
          onDmxDisconnect={handleDmxDisconnect}
          onDmxReconnect={handleDmxReconnect}
          selectedProfileId={selectedProfileId}
          onProfileSelect={setSelectedProfileId}
          onAddFixture={handleAddFixture}
          fixtures={fixtures}
          fixtureValues={fixtureValues}
          dmxActiveCount={dmxActiveCount}
          onFixtureAddressChange={handleFixtureAddressChange}
          onFixtureStrobeModeChange={handleStrobeModeChange}
          onFixtureStrobeThresholdChange={handleStrobeThresholdChange}
          onFixtureStrobeMaxChange={handleStrobeMaxChange}
          onFixtureColorModeChange={handleColorModeChange}
          onFixtureSolidColorChange={handleSolidColorChange}
          onFixtureProfileChange={handleFixtureProfileChange}
          onFixtureRemove={handleRemoveFixture}
          onFixtureDimmerModeChange={handleDimmerModeChange}
          onFixtureManualDimmerChange={handleManualDimmerChange}
          fogIntensity={fogIntensity}
          onFogIntensityChange={setFogIntensity}
          onFogToggle={triggerFog}
          isFogActive={() => lightingEngineRef.current?.isFogActive() ?? false}
          onClose={closeDmxDrawer}
        />
      </aside>
    </div>
  );
}
