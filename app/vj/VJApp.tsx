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
  AI_BACKEND_LABELS,
  type AiBackend,
  type UpscaleMode,
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
  AudioDebugPanel,
  LightingPanel,
  SystemsBar,
  PromptDock,
  HotkeyBoard,
  Field,
  PanelHeader,
  ResPicker,
  StageRenderer,
  type StageRendererHandle,
  FogControl,
} from "./components";

type Status = "idle" | "requesting" | "running" | "error";
type DmxStatus = "disconnected" | "connecting" | "connected" | "unsupported";

interface AudioDevice {
  deviceId: string;
  label: string;
}

/**
 * Main VJ application orchestrator.
 *
 * Layout:
 *  - SystemsBar (sticky, full-width) — audio / visual / ai / dmx status at a glance
 *  - Below: 12-column responsive dashboard grid
 *     [Left col]   waveform + scene/device popovers + audio debug
 *     [Center col] AI output preview (hero) + prompt dock
 *     [Right col]  AI settings + DMX / lighting stack
 *
 * Engine lifecycle via refs (not React state) to avoid blocking render loops.
 */
export function VJApp() {
  // Engine refs - NOT React state to avoid render loop interference
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioEngineRef = useRef<AudioEngine | null>(null);
  const visualEngineRef = useRef<VisualEngine | null>(null);
  const lightingEngineRef = useRef<LightingEngine | null>(null);
  const dmxOutputRef = useRef<DmxOutput | null>(null);

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
    hideUi: aiHideUi,
    autoConnect: aiAutoConnect,
    promptPresets: aiPromptPresets,
    stageSharpen: aiStageSharpen,
    stageScanlines: aiStageScanlines,
    stageVignette: aiStageVignette,
    fogIntensity,
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
    setHideUi: setAiHideUi,
    setAutoConnect: setAiAutoConnect,
    setStageSharpen: setAiStageSharpen,
    setStageScanlines: setAiStageScanlines,
    setStageVignette: setAiStageVignette,
    setFogIntensity,
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
        // Roulette: pick a random preset and fire it (also re-rolls the seed).
        // Stopping/starting generation is rarely needed mid-set; surfacing a
        // surprise prompt on Space is far more useful for live work.
        fireRandomPreset();
        e.preventDefault();
        return;
      }

      if (e.key === "h" || e.key === "H") {
        const s = useAiSettingsStore.getState();
        s.setHideUi(!s.hideUi);
        // No flushSettingsNow — hide UI doesn't affect the generation payload.
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
    aiHideUi,
    aiPromptPresets,
    firePreset,
    fireRandomPreset,
    adjustAlpha,
    triggerFog,
    setAiHideUi,
  ]);

  const [aiStatus, setAiStatus] = useState<AiTransportStatus>("idle");
  const [aiLogs, setAiLogs] = useState<string[]>([]);
  const [aiImageUrl, setAiImageUrl] = useState<string | null>(null);
  const [aiGenTime, setAiGenTime] = useState<number | null>(null);

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
    if (!selectedDeviceId) return devices[0]?.label;
    return devices.find((d) => d.deviceId === selectedDeviceId)?.label;
  }, [selectedDeviceId, devices]);

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
    async (deviceId?: string, fixtureList?: FixtureInstance[]) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

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
        await audioEngine.init(deviceId);
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

  const handleDeviceChange = useCallback(
    (deviceId: string) => {
      setSelectedDeviceId(deviceId);
      setPersistedDeviceId(deviceId);
      initAudio(deviceId || undefined);
    },
    [initAudio, setPersistedDeviceId]
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
  }, [persistedDeviceId, selectedDeviceId, initAudio]);

  // ============================================================================
  // Render
  // ============================================================================

  const sliderFillPct = Math.round((aiKleinAlpha / 0.5) * 100);

  return (
    <div className="w-full min-h-screen flex flex-col">
      {/* Fullscreen Hide-UI overlay */}
      {aiHideUi && (
        <div
          onClick={() => setAiHideUi(false)}
          className="fixed inset-0 z-50 bg-black flex items-center justify-center cursor-none"
        >
          {aiImageUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={aiImageUrl}
              alt="AI output"
              className="max-w-full max-h-full object-contain"
              style={{
                imageRendering: aiUpscaleMode === "bilinear" ? "auto" : "auto",
              }}
            />
          )}
        </div>
      )}

      <SystemsBar
        audioStatus={status}
        audioDeviceLabel={selectedDeviceLabel}
        aiStatus={aiStatus}
        aiBackend={aiBackend}
        aiGenTimeMs={aiStatus === "connected" ? aiGenTime : null}
        dmxStatus={dmxStatus}
        dmxFixtureCount={fixtures.length}
        dmxActiveCount={dmxActiveCount}
        lightingEnabled={lightingEnabled}
        hideUi={aiHideUi}
        onHideUi={() => setAiHideUi(!aiHideUi)}
      />

      {/* Error banner */}
      {status === "error" && errorMessage && (
        <div className="mx-4 mt-3 text-sm font-mono text-[color:var(--vj-error)] bg-[color-mix(in_srgb,var(--vj-error)_10%,transparent)] border border-[color:var(--vj-error)] rounded px-3 py-2 shadow-[0_0_18px_-6px_var(--vj-error)]">
          ⚠ {errorMessage}
        </div>
      )}

      {/* Dashboard grid — 12 cols at wide width, reflows at narrow.
          Tight gap + padding so everything fits in one viewport. */}
      <div className="flex-1 w-full grid grid-cols-1 xl:grid-cols-12 gap-2 p-2">
        {/* ============== LEFT COL: waveform + sources ============== */}
        <section className="xl:col-span-3 flex flex-col gap-2 min-w-0">
          <div className="vj-panel p-2 flex flex-col gap-2">
            <PanelHeader
              title="Waveform / Source"
              actions={
                <span
                  className="flex items-center gap-1.5 text-[10px] text-[color:var(--vj-ink-dim)] normal-case h-full"
                  title={`Audio status: ${status}`}
                >
                  <span
                    className="vj-dot vj-dot--static"
                    style={{
                      color:
                        status === "running"
                          ? "var(--vj-live)"
                          : status === "error"
                          ? "var(--vj-error)"
                          : "var(--vj-info)",
                    }}
                  />
                  <span className="tracking-wider uppercase">
                    {status === "running" ? "live" : status}
                  </span>
                </span>
              }
            />

            <div className="vj-canvas-frame aspect-[4/1] w-full">
              <canvas
                ref={canvasRef}
                className="w-full h-full"
                style={{ imageRendering: "pixelated" }}
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <Field label="Scene">
                <select
                  value={currentSceneId}
                  onChange={(e) => handleSceneChange(e.target.value)}
                  className="vj-input"
                >
                  {SCENES.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                    </option>
                  ))}
                </select>
              </Field>

              <Field
                label="Audio In"
                hint={
                  devices.length <= 1
                    ? "Only one audio device detected"
                    : "Pick which audio input drives the visuals"
                }
              >
                <select
                  value={selectedDeviceId}
                  onChange={(e) => handleDeviceChange(e.target.value)}
                  className="vj-input"
                  disabled={devices.length <= 1}
                >
                  <option value="">Default device</option>
                  {devices.map((d) => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
          </div>

          <div className="vj-panel p-2 flex flex-col gap-2">
            <PanelHeader
              title="Audio Features"
              actions={
                <button
                  type="button"
                  onClick={() => setShowDebug(!showDebug)}
                  aria-pressed={showDebug}
                  className={`vj-icon-btn ${showDebug ? "vj-icon-btn--on" : ""}`}
                  title="Hides the panel and stops the polling timer."
                >
                  {showDebug ? "ON" : "OFF"}
                </button>
              }
            />
            {showDebug && <AudioDebugPanel features={debugFeatures} />}
          </div>
        </section>

        {/* ============== CENTER COL: AI output + prompt ============== */}
        <section className="xl:col-span-5 flex flex-col gap-2 min-w-0">
          <div className="vj-panel p-2 flex flex-col gap-2 flex-1">
            <PanelHeader
              title="AI Output"
              actions={
                <button
                  type="button"
                  onClick={() => setAiSendFrames(!aiSendFrames)}
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
              }
            />

            {/* Sub-toolbar — controls live below the title row so every
                card's title baseline lines up. Resolution + (klein only)
                α slider — both are live cues you'll touch mid-set. */}
            <div className="flex items-center gap-2 flex-wrap">
              <ResPicker
                width={aiOutputWidth}
                height={aiOutputHeight}
                onPick={setAiOutputSize}
              />
              {aiBackend === "klein" && (
                <div
                  className="flex items-center gap-1.5 px-2 py-1 rounded border border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-panel-2)]"
                  title="0 = pure prompt · 0.5 = waveform dominates · ↑/↓ to adjust"
                >
                  <span className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)]">
                    α
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={0.5}
                    step={0.01}
                    value={aiKleinAlpha}
                    onChange={(e) => setAiKleinAlpha(Number(e.target.value))}
                    className="vj-range w-24"
                    style={
                      {
                        ["--vj-range-fill" as string]: `${sliderFillPct}%`,
                      } as React.CSSProperties
                    }
                  />
                  <span className="font-mono text-[11px] tabular-nums text-[color:var(--vj-accent)] w-9 text-right">
                    {aiKleinAlpha.toFixed(2)}
                  </span>
                </div>
              )}
              {aiBackend === "klein" && (
                <label
                  className="flex items-center gap-1.5 px-2 py-1 rounded border border-[color:var(--vj-edge-hot)] bg-[color:var(--vj-panel-2)] text-[10px] uppercase tracking-wider font-mono text-[color:var(--vj-ink-dim)]"
                  title="Klein steps — 1 fastest, 2 sweet spot, 4 max quality"
                >
                  <span>steps</span>
                  <select
                    value={aiKleinSteps}
                    onChange={(e) => setAiKleinSteps(Number(e.target.value))}
                    className="bg-transparent border-0 outline-none text-[color:var(--vj-info)] tabular-nums normal-case font-mono py-0 pr-1"
                  >
                    <option value={1}>1</option>
                    <option value={2}>2</option>
                    <option value={3}>3</option>
                    <option value={4}>4</option>
                  </select>
                </label>
              )}
              {aiBackend === "klein" && aiCaptureSize !== aiOutputLong && (
                <button
                  onClick={() => setAiCaptureSize(aiOutputLong)}
                  className="vj-btn vj-btn--accent"
                  title={`Match capture to output long side (${aiOutputLong}) for sharper Klein output`}
                >
                  match → {aiOutputLong}
                </button>
              )}
            </div>

            {/* Preview fills the column width; height follows the configured
                output aspect. The WebGL StageRenderer applies the same
                sharpen pass that runs on /vj/stage, so what you see here is
                what the projector sees (the .vj-canvas-frame ::after CSS
                overlay supplies the matching scanline+vignette FX). The
                placeholder text overlays the canvas until a frame arrives. */}
            <div
              className="vj-canvas-frame relative flex items-center justify-center w-full"
              style={{
                aspectRatio: `${aiOutputWidth} / ${aiOutputHeight}`,
              }}
            >
              <StageRenderer
                ref={previewRendererRef}
                sharpen={aiStageSharpen}
                className="w-full h-full"
                style={{ objectFit: "contain" }}
              />
              {!aiImageUrl && (
                <div className="absolute inset-0 flex items-center justify-center text-[11px] font-mono uppercase tracking-wider text-[color:var(--vj-ink-dim)] text-center px-4 bg-black">
                  {aiStatus === "connected" ? (
                    <span>
                      <span className="text-[color:var(--vj-info)]">
                        connected
                      </span>
                      <br />
                      press ▶ generate or hit space
                    </span>
                  ) : aiStatus === "connecting" ? (
                    <span className="text-[color:var(--vj-info)]">
                      negotiating webrtc…
                    </span>
                  ) : (
                    <span>
                      connect backend in
                      <br />
                      <span className="text-[color:var(--vj-accent)]">
                        right panel ↗
                      </span>
                    </span>
                  )}
                </div>
              )}
            </div>

            <PromptDock
              activePrompt={aiPrompt}
              onSetPrompt={setAiPrompt}
            />

            <HotkeyBoard
              presets={aiPromptPresets}
              activePrompt={aiPrompt}
              onFirePreset={firePreset}
              onUpdatePreset={updatePromptPreset}
              onRandom={fireRandomPreset}
              onHideUi={() => setAiHideUi(!aiHideUi)}
              onFireFog={triggerFog}
              alpha={aiBackend === "klein" ? aiKleinAlpha : undefined}
              onAlphaDelta={
                aiBackend === "klein" ? adjustAlpha : undefined
              }
            />
          </div>
        </section>

        {/* ============== RIGHT COL: AI settings + lighting ============== */}
        {/* ============== RIGHT COL: AI backend + Lighting ============== */}
        <aside className="xl:col-span-4 flex flex-col gap-2 min-w-0">
          {/* AI Backend — connection wiring + per-frame params. Klein-only
              live controls (α, steps, match-capture) live on the AI Output
              toolbar; Stage FX has its own card below. */}
          <div className="vj-panel p-2 flex flex-col gap-2">
            <PanelHeader
              title="AI Backend"
              actions={
                <>
                  <label
                    className="flex items-center gap-1.5 text-[10px] font-mono text-[color:var(--vj-ink-dim)] uppercase tracking-wider h-full"
                    title="Auto-connect on page load and on backend switch"
                  >
                    <input
                      type="checkbox"
                      checked={aiAutoConnect}
                      onChange={(e) => setAiAutoConnect(e.target.checked)}
                      className="vj-check"
                    />
                    auto
                  </label>
                  {aiStatus === "connected" || aiStatus === "connecting" ? (
                    <button
                      onClick={() => void aiTransport.stop()}
                      className="vj-btn vj-btn--danger"
                      title="Close the WebRTC channel"
                    >
                      ✕ disconnect
                    </button>
                  ) : (
                    <button
                      onClick={() => void aiTransport.start()}
                      className="vj-btn vj-btn--live"
                      title="Open WebRTC channel to the AI backend"
                    >
                      ▶ connect
                    </button>
                  )}
                </>
              }
            />

            <select
              value={aiBackend}
              onChange={(e) => {
                void aiTransport.stop();
                setAiBackend(e.target.value as AiBackend);
              }}
              className="vj-input"
              title={AI_BACKEND_LABELS[aiBackend]}
            >
              {(Object.keys(AI_BACKEND_URLS) as AiBackend[]).map((k) => (
                <option key={k} value={k}>
                  {AI_BACKEND_LABELS[k]}
                </option>
              ))}
            </select>

            {/* Per-frame params — two columns, dense. Seed is grouped with
                its randomize button so the pair reads as one control. */}
            <div className="grid grid-cols-2 gap-2">
              <Field label="Capture px">
                <select
                  value={aiCaptureSize}
                  onChange={(e) => setAiCaptureSize(Number(e.target.value))}
                  className="vj-input"
                  title={
                    aiBackend === "klein"
                      ? "Klein: match to output long side for best quality"
                      : "Client capture resolution"
                  }
                >
                  <option value={64}>64 · fastest</option>
                  <option value={128}>128</option>
                  <option value={256}>256</option>
                  <option value={512}>512</option>
                </select>
              </Field>
              <Field label="Target fps">
                <select
                  value={aiFrameRate}
                  onChange={(e) => setAiFrameRate(Number(e.target.value))}
                  className="vj-input"
                >
                  <option value={10}>10 fps</option>
                  <option value={20}>20 fps</option>
                  <option value={30}>30 fps</option>
                  <option value={60}>60 fps</option>
                </select>
              </Field>

              <Field label="Seed">
                <div className="grid grid-cols-[1fr_auto] gap-1">
                  <input
                    type="number"
                    value={aiSeed}
                    onChange={(e) =>
                      setAiSeed(Math.max(0, Math.floor(Number(e.target.value))))
                    }
                    className="vj-input tabular-nums"
                  />
                  <button
                    onClick={() => setAiSeed(Math.floor(Math.random() * 1_000_000))}
                    className="vj-btn"
                    title="Randomize seed"
                  >
                    ⟲
                  </button>
                </div>
              </Field>
              <Field label="Display upscale">
                <select
                  value={aiUpscaleMode}
                  onChange={(e) => setAiUpscaleMode(e.target.value as UpscaleMode)}
                  className="vj-input"
                  title="Interpolation when the AI output is scaled up for display"
                >
                  <option value="lanczos">Lanczos (sharp)</option>
                  <option value="bilinear">Bilinear (soft)</option>
                </select>
              </Field>
            </div>

            {/* Diagnostics live behind a disclosure so they don't crowd the
                live-controls in the default view. */}
            <details className="text-xs">
              <summary className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)] cursor-pointer hover:text-[color:var(--vj-info)]">
                diagnostics ({aiLogs.length} log{aiLogs.length === 1 ? "" : "s"})
              </summary>
              <div className="mt-2 flex flex-col gap-2">
                <label className="flex items-center gap-2 text-[10px] font-mono text-[color:var(--vj-ink-dim)] uppercase tracking-wider">
                  <input
                    type="checkbox"
                    checked={aiShowCaptureDebug}
                    onChange={(e) => setAiShowCaptureDebug(e.target.checked)}
                    className="vj-check"
                  />
                  show capture preview
                </label>
                {aiShowCaptureDebug && (
                  <div className="flex flex-col gap-1 rounded border border-[color:var(--vj-edge-hot)] bg-black/50 p-2">
                    <div className="text-[9px] uppercase tracking-wider font-mono text-[color:var(--vj-ink-dim)]">
                      sending {aiCaptureSize} → {aiOutputWidth}×{aiOutputHeight}
                    </div>
                    <canvas
                      ref={aiDebugCanvasRef}
                      width={aiCaptureSize}
                      height={aiCaptureSize}
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

          {/* Stage FX — its own card. Sharpen runs as a WebGL unsharp-mask
              pass on /vj/stage; scanlines + vignette are CSS overlays
              matching the preview frame. Header carries the apply-target
              hint so users see at a glance this isn't the preview. */}
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
                value={aiStageSharpen}
                onChange={(e) => setAiStageSharpen(Number(e.target.value))}
                className="vj-range vj-range--tight"
                style={
                  {
                    ["--vj-range-fill" as string]: `${(aiStageSharpen / 10) * 100}%`,
                  } as React.CSSProperties
                }
                title="WebGL unsharp-mask strength (0 = off, ~1 mild, 10 = aggressive)"
              />
              <span className="font-mono text-[11px] tabular-nums text-[color:var(--vj-info)] w-10 text-right ml-auto">
                {aiStageSharpen.toFixed(1)}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <label className="flex items-center gap-2 text-[10px] font-mono text-[color:var(--vj-ink-dim)] uppercase tracking-wider">
                <input
                  type="checkbox"
                  checked={aiStageScanlines}
                  onChange={(e) => setAiStageScanlines(e.target.checked)}
                  className="vj-check"
                />
                scanlines
              </label>
              <label className="flex items-center gap-2 text-[10px] font-mono text-[color:var(--vj-ink-dim)] uppercase tracking-wider">
                <input
                  type="checkbox"
                  checked={aiStageVignette}
                  onChange={(e) => setAiStageVignette(e.target.checked)}
                  className="vj-check"
                />
                vignette
              </label>
            </div>
          </div>

          {/* Fog — own card so the toggle button + hotkey are one glance
              away. Sits above the Lighting card because fog state is a live
              cue and shouldn't be hidden inside a fixture list. Hidden when
              the master DMX/lighting switch is off — fog rides on a DMX
              channel, so it's meaningless without the universe live. */}
          {lightingEnabled && (
            <FogControl
              intensity={fogIntensity}
              onSetIntensity={setFogIntensity}
              onToggle={triggerFog}
              isActive={() =>
                lightingEngineRef.current?.isFogActive() ?? false
              }
            />
          )}

          {/* Lighting / DMX — its own internal master toggle controls the
              entire card body. Status comes from the SystemsBar so this
              card jumps straight to the action when enabled. */}
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
          />
        </aside>
      </div>
    </div>
  );
}
