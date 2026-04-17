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
} from "@/src/lib/lighting";
import {
  AudioDebugPanel,
  LightingPanel,
  SystemsBar,
  PromptDock,
  Field,
  ResPicker,
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
    const t = window.setTimeout(() => {
      if (!aiTransport.isConnected()) void aiTransport.start();
    }, 120);
    return () => window.clearTimeout(t);
  }, [aiAutoConnectSel, aiTransport]);

  // UI state
  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [currentSceneId, setCurrentSceneId] = useState<string>(SCENES[0].id);

  // Debug panel state - throttled updates (100ms = ~10fps)
  const [debugFeatures, setDebugFeatures] = useState<AudioFeatures | null>(
    null
  );
  // Audio features start visible — user wants the dashboard fully populated.
  const [showDebug, setShowDebug] = useState<boolean>(true);

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

  // Firing a preset = swap prompt + re-roll seed. Hotkeys (1-9) and chip
  // clicks share this so hammering the same preset gives fresh variations
  // every time, and jumping between presets always re-inits the noise.
  const firePreset = useCallback(
    (prompt: string) => {
      setAiPrompt(prompt);
      setAiSeed(Math.floor(Math.random() * 1_000_000));
    },
    [setAiPrompt, setAiSeed]
  );

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

      if (e.key === " ") {
        setAiSendFrames(!aiSendFrames);
        e.preventDefault();
        return;
      }

      if (e.key === "h" || e.key === "H") {
        setAiHideUi(!aiHideUi);
        e.preventDefault();
        return;
      }

      if (aiBackend === "klein") {
        if (e.key === "ArrowUp") {
          setAiKleinAlpha(Math.round((aiKleinAlpha + 0.02) * 100) / 100);
          e.preventDefault();
        } else if (e.key === "ArrowDown") {
          setAiKleinAlpha(Math.round((aiKleinAlpha - 0.02) * 100) / 100);
          e.preventDefault();
        }
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    aiBackend,
    aiHideUi,
    aiKleinAlpha,
    aiPromptPresets,
    aiSendFrames,
    firePreset,
    setAiHideUi,
    setAiKleinAlpha,
    setAiSendFrames,
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
  const [selectedProfileId, setSelectedProfileId] = useState<string>(
    FIXTURE_PROFILES[0].id
  );

  const fixtures = useFixtures();
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

  const handleDmxFrame = useCallback((frame: LightingFrame) => {
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
    [fetchDevices, handleLightingFrame, handleDmxFrame, fixtures]
  );

  // ============================================================================
  // UI event handlers
  // ============================================================================

  const handleDeviceChange = useCallback(
    (deviceId: string) => {
      setSelectedDeviceId(deviceId);
      initAudio(deviceId || undefined);
    },
    [initAudio]
  );

  const handleSceneChange = useCallback((sceneId: string) => {
    const engine = visualEngineRef.current;
    if (engine && engine.setSceneById(sceneId)) {
      setCurrentSceneId(sceneId);
    }
  }, []);

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

  const aiSettingsResumeTimeoutRef = useRef<number | null>(null);
  const aiDebugCanvasRef = useRef<HTMLCanvasElement | null>(null);

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
    if (!aiTransport.isConnected()) return;

    const sender = aiFrameSenderRef.current;

    sender.running = false;
    if (aiSettingsResumeTimeoutRef.current !== null) {
      window.clearTimeout(aiSettingsResumeTimeoutRef.current);
      aiSettingsResumeTimeoutRef.current = null;
    }
    aiSettingsResumeTimeoutRef.current = window.setTimeout(() => {
      aiSettingsResumeTimeoutRef.current = null;
      if (!aiSendFrames || !aiTransport.isConnected()) return;
      sender.running = true;
      sender.lastFrameTime = 0;
      requestAnimationFrame(aiFrameLoop);
    }, 100);

    const aspect = aiOutputWidth / aiOutputHeight;
    const captureW =
      aspect >= 1 ? aiCaptureSize : Math.max(16, Math.round(aiCaptureSize * aspect));
    const captureH =
      aspect >= 1 ? Math.max(16, Math.round(aiCaptureSize / aspect)) : aiCaptureSize;
    const payload: Record<string, unknown> = {
      prompt: aiPrompt,
      seed: aiSeed,
      captureWidth: captureW,
      captureHeight: captureH,
      width: aiOutputWidth,
      height: aiOutputHeight,
    };
    if (aiBackend === "klein") {
      payload.alpha = aiKleinAlpha;
      payload.n_steps = aiKleinSteps;
    }
    aiTransport.sendText(JSON.stringify(payload));
  }, [
    aiBackend,
    aiPrompt,
    aiSeed,
    aiCaptureSize,
    aiOutputWidth,
    aiOutputHeight,
    aiKleinAlpha,
    aiKleinSteps,
    aiTransport,
    aiSendFrames,
    aiFrameLoop,
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
    return () => {
      if (aiSettingsResumeTimeoutRef.current !== null) {
        window.clearTimeout(aiSettingsResumeTimeoutRef.current);
      }
    };
  }, []);

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
      const connected = await dmxOutputRef.current.autoConnect();
      if (connected) {
        setDmxStatus("connected");
      }
    };
    tryAutoConnect();
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

  useEffect(() => {
    fetchDevices();
    initAudio(undefined, fixtures);

    return () => {
      lightingEngineRef.current?.stop();
      visualEngineRef.current?.stop();
      audioEngineRef.current?.destroy();
      dmxOutputRef.current?.disconnect();
      void aiTransport.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
            <div className="flex items-center justify-between gap-2">
              <div className="vj-panel-title">Waveform / Source</div>
              <div className="flex items-center gap-1.5 text-[10px] text-[color:var(--vj-ink-dim)] normal-case">
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
              </div>
            </div>

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
            <div className="flex items-center justify-between">
              <div className="vj-panel-title">Audio Features</div>
              <button
                onClick={() => setShowDebug(!showDebug)}
                className="vj-btn"
                title="Hides the panel and stops the polling timer."
              >
                {showDebug ? "Hide" : "Show"}
              </button>
            </div>
            {showDebug && <AudioDebugPanel features={debugFeatures} />}
          </div>
        </section>

        {/* ============== CENTER COL: AI output + prompt ============== */}
        <section className="xl:col-span-5 flex flex-col gap-2 min-w-0">
          <div className="vj-panel p-2 flex flex-col gap-2 flex-1">
            <div className="flex items-center justify-between gap-2">
              <div className="vj-panel-title">AI Output</div>
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
            </div>

            {/* Hero output preview — honours aspect via object-contain. Capped
                at a sensible max-height so a small generated res doesn't
                stretch into a wall-sized blurry image. The aspectRatio +
                max-height combo: height hits cap first, width derives from
                aspect, mx-auto centres the resulting box. */}
            <div
              className="vj-canvas-frame flex items-center justify-center mx-auto"
              style={{
                aspectRatio: `${aiOutputWidth} / ${aiOutputHeight}`,
                maxHeight: 320,
                maxWidth: "100%",
                width: "auto",
              }}
            >
              {aiImageUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={aiImageUrl}
                  alt="AI generated output"
                  className="w-full h-full object-contain"
                  style={{ imageRendering: "auto" }}
                />
              ) : (
                <div className="text-[11px] font-mono uppercase tracking-wider text-[color:var(--vj-ink-dim)] text-center px-4">
                  {aiStatus === "connected" ? (
                    <>
                      <span className="text-[color:var(--vj-info)]">
                        Connected
                      </span>
                      <br />
                      waiting for frames…
                    </>
                  ) : aiStatus === "connecting" ? (
                    <span className="text-[color:var(--vj-info)]">
                      Negotiating WebRTC…
                    </span>
                  ) : (
                    <>
                      Connect AI backend
                      <br />
                      <span className="text-[color:var(--vj-accent)]">
                        to preview output
                      </span>
                    </>
                  )}
                </div>
              )}
            </div>

            <PromptDock
              activePrompt={aiPrompt}
              onSetPrompt={setAiPrompt}
              onFirePreset={firePreset}
              presets={aiPromptPresets}
              onUpdatePreset={updatePromptPreset}
            />
          </div>
        </section>

        {/* ============== RIGHT COL: AI settings + lighting ============== */}
        <aside className="xl:col-span-4 grid grid-cols-1 lg:grid-cols-2 gap-3 min-w-0 content-start">
          {/* AI backend + connection */}
          <div className="vj-panel p-2 flex flex-col gap-2 lg:col-span-2">
            <div className="vj-panel-title">AI Backend</div>

            <Field label="Model">
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
            </Field>

            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={() => void aiTransport.start()}
                disabled={aiStatus === "connecting" || aiStatus === "connected"}
                className="vj-btn vj-btn--live flex-1"
                title="Open WebRTC channel to the AI backend"
              >
                {aiStatus === "connecting" ? "Connecting…" : "Connect"}
              </button>
              <button
                onClick={() => void aiTransport.stop()}
                disabled={aiStatus === "idle" || aiStatus === "disconnected"}
                className="vj-btn vj-btn--danger flex-1"
              >
                Disconnect
              </button>
            </div>

            <label
              className="flex items-center gap-2 text-[11px] font-mono text-[color:var(--vj-ink-dim)]"
              title="Automatically connect to the selected backend on page load and on backend switch."
            >
              <input
                type="checkbox"
                checked={aiAutoConnect}
                onChange={(e) => setAiAutoConnect(e.target.checked)}
                className="vj-check"
              />
              <span className="uppercase tracking-wider">Auto-connect</span>
            </label>
          </div>

          {/* Generation parameters */}
          <div className="vj-panel p-2 flex flex-col gap-2">
            <div className="vj-panel-title">Generation</div>

            <div className="grid grid-cols-2 gap-2">
              <Field
                label="Capture"
                hint={
                  aiBackend === "klein"
                    ? "For Klein best quality, match Capture to Output long side."
                    : "Client capture resolution."
                }
              >
                <select
                  value={aiCaptureSize}
                  onChange={(e) => setAiCaptureSize(Number(e.target.value))}
                  className="vj-input"
                >
                  <option value={64}>64 px · fastest</option>
                  <option value={128}>128 px</option>
                  <option value={256}>256 px</option>
                  <option value={512}>512 px</option>
                </select>
              </Field>

              <Field label="Target FPS">
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
            </div>

            <Field label="Output">
              <ResPicker
                width={aiOutputWidth}
                height={aiOutputHeight}
                onPick={setAiOutputSize}
              />
            </Field>

            {aiBackend === "klein" && aiCaptureSize !== aiOutputLong && (
              <button
                onClick={() => setAiCaptureSize(aiOutputLong)}
                className="vj-btn vj-btn--accent self-start"
                title={`Match capture to output long side (${aiOutputLong}) for sharper Klein results`}
              >
                Match capture → {aiOutputLong}
              </button>
            )}

            <div className="grid grid-cols-[1fr_auto] gap-2 items-end">
              <Field label="Seed">
                <input
                  type="number"
                  value={aiSeed}
                  onChange={(e) =>
                    setAiSeed(Math.max(0, Math.floor(Number(e.target.value))))
                  }
                  className="vj-input tabular-nums"
                />
              </Field>
              <button
                onClick={() => setAiSeed(Math.floor(Math.random() * 1000000))}
                className="vj-btn"
                title="Randomize seed"
              >
                ⟲ Random
              </button>
            </div>

            <Field
              label="Display upscale"
              hint="Interpolation when the AI output is scaled up for display."
            >
              <select
                value={aiUpscaleMode}
                onChange={(e) => setAiUpscaleMode(e.target.value as UpscaleMode)}
                className="vj-input"
              >
                <option value="lanczos">Lanczos (sharp)</option>
                <option value="bilinear">Bilinear (soft)</option>
              </select>
            </Field>
          </div>

          {/* Klein-specific */}
          {aiBackend === "klein" && (
            <div className="vj-panel p-2 flex flex-col gap-2">
              <div className="vj-panel-title">Klein Img2Img</div>

              <div className="flex flex-col gap-1">
                <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-[color:var(--vj-ink-dim)]">
                  <span>Alpha</span>
                  <span className="tabular-nums text-[color:var(--vj-accent)]">
                    {aiKleinAlpha.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={0.5}
                  step={0.01}
                  value={aiKleinAlpha}
                  onChange={(e) => setAiKleinAlpha(Number(e.target.value))}
                  className="vj-range w-full"
                  style={
                    {
                      ["--vj-range-fill" as string]: `${sliderFillPct}%`,
                    } as React.CSSProperties
                  }
                  title="0 = pure prompt · 0.1 = subtle nudge · 0.2+ = waveform shapes composition · 0.3+ = waveform dominates"
                />
                <div className="flex justify-between text-[9px] uppercase tracking-wider text-[color:var(--vj-ink-dim)]">
                  <span>prompt</span>
                  <span>waveform</span>
                </div>
              </div>

              <Field
                label="Steps"
                hint="Klein is distilled for 4 steps. 2 = sweet spot."
              >
                <select
                  value={aiKleinSteps}
                  onChange={(e) => setAiKleinSteps(Number(e.target.value))}
                  className="vj-input"
                >
                  <option value={1}>1 · fastest</option>
                  <option value={2}>2 · recommended</option>
                  <option value={3}>3 · more detail</option>
                  <option value={4}>4 · max quality</option>
                </select>
              </Field>
            </div>
          )}

          {/* Debug capture preview + logs (collapsed by default) */}
          <div className="vj-panel p-2 flex flex-col gap-2">
            <div className="flex items-center justify-between gap-2">
              <div className="vj-panel-title">Capture Debug</div>
              <label className="flex items-center gap-2 text-[11px] font-mono text-[color:var(--vj-ink-dim)]">
                <input
                  type="checkbox"
                  checked={aiShowCaptureDebug}
                  onChange={(e) => setAiShowCaptureDebug(e.target.checked)}
                  className="vj-check"
                />
                <span className="uppercase tracking-wider">On</span>
              </label>
            </div>
            {aiShowCaptureDebug && (
              <div className="flex flex-col gap-1 rounded border border-[color:var(--vj-edge-hot)] bg-black/50 p-2">
                <div className="text-[9px] uppercase tracking-wider font-mono text-[color:var(--vj-ink-dim)]">
                  Sending: {aiCaptureSize} → {aiOutputWidth}×{aiOutputHeight}
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
            <details className="text-xs">
              <summary className="font-mono text-[10px] uppercase tracking-wider text-[color:var(--vj-ink-dim)] cursor-pointer hover:text-[color:var(--vj-info)]">
                Logs ({aiLogs.length})
              </summary>
              <div className="mt-2 max-h-28 overflow-auto font-mono text-[11px] text-[color:var(--vj-ink-dim)] bg-black/40 rounded p-2">
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
            </details>
          </div>

          {/* Lighting panel — keep existing, dress the wrapper only */}
          <div className="vj-panel p-2 flex flex-col gap-2 lg:col-span-2">
            <div className="vj-panel-title">Lighting / DMX</div>
            <LightingPanel
              dmxStatus={dmxStatus}
              dmxSupported={dmxSupported}
              onDmxConnect={handleDmxConnect}
              onDmxDisconnect={handleDmxDisconnect}
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
            />
          </div>
        </aside>
      </div>
    </div>
  );
}
