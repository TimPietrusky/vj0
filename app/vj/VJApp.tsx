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
import { StatusBar, AudioDebugPanel, LightingPanel } from "./components";

type Status = "idle" | "requesting" | "running" | "error";
type DmxStatus = "disconnected" | "connecting" | "connected" | "unsupported";

interface AudioDevice {
  deviceId: string;
  label: string;
}

/**
 * Main VJ application orchestrator.
 *
 * Manages engine lifecycle via refs (not React state) to avoid blocking render loops.
 * UI state uses useState but is throttled where necessary (debug panel at ~10fps).
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
  // Use a store selector so the URL memo updates reactively on backend change.
  const aiBackendSel = useAiSettingsStore((s) => s.backend);
  const aiSignalingUrl = useMemo(() => {
    const envOverride = process.env.NEXT_PUBLIC_VJ0_WEBRTC_SIGNALING_URL;
    if (envOverride) return envOverride;
    return AI_BACKEND_URLS[aiBackendSel] || "/api/webrtc/offer";
  }, [aiBackendSel]);

  // AI transport (WebRTC) - keep as stable instance; rebuilt when signalingUrl changes
  const aiTransport = useMemo(
    () =>
      new WebRtcAiTransport({
        signalingUrl: aiSignalingUrl,
        iceServers: aiIceServers,
        iceTransportPolicy: aiIceTransportPolicy,
      }),
    [aiSignalingUrl, aiIceServers, aiIceTransportPolicy]
  );

  // When transport changes (e.g. backend switch), stop the old one cleanly.
  useEffect(() => {
    return () => {
      void aiTransport.stop();
    };
  }, [aiTransport]);

  // UI state - safe to use useState for these
  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [currentSceneId, setCurrentSceneId] = useState<string>(SCENES[0].id);

  // Debug panel state - throttled updates (100ms = ~10fps)
  const [debugFeatures, setDebugFeatures] = useState<AudioFeatures | null>(
    null
  );
  const [showDebug, setShowDebug] = useState<boolean>(true);

  // Remote AI (WebRTC) UI state
  const {
    backend: aiBackend,
    showAi,
    sendFrames: aiSendFrames,
    showCaptureDebug: aiShowCaptureDebug,
    prompt: aiPrompt,
    captureSize: aiCaptureSize,
    outputSize: aiOutputSize,
    frameRate: aiFrameRate,
    seed: aiSeed,
    kleinAlpha: aiKleinAlpha,
    kleinSteps: aiKleinSteps,
    upscaleMode: aiUpscaleMode,
    hideUi: aiHideUi,
    promptPresets: aiPromptPresets,
    setBackend: setAiBackend,
    setShowAi,
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
  } = useAiSettingsStore();

  // Prompt draft — user types freely; Enter commits to the store (and server).
  // This avoids pushing a new prompt on every keystroke during live performance.
  const [aiPromptDraft, setAiPromptDraft] = useState<string>(aiPrompt);
  useEffect(() => {
    // If the store prompt changes externally (hotkey, preset click), sync the
    // draft so the input reflects it.
    setAiPromptDraft(aiPrompt);
  }, [aiPrompt]);
  const aiPromptDirty = aiPromptDraft !== aiPrompt;

  // BroadcastChannel to the /vj/stage tab — we publish every AI frame + prompt
  // changes + transport status so the stage view can render fullscreen.
  const stageChannelRef = useRef<BroadcastChannel | null>(null);
  const stageFrameSeqRef = useRef<number>(0);
  useEffect(() => {
    const ch = openStageChannel();
    stageChannelRef.current = ch;

    // When the stage tab says "hello" (just opened), re-publish the current
    // prompt so the overlay has something to show.
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

  // Publish prompt whenever it changes (store-level, so after an Enter commit).
  useEffect(() => {
    stageChannelRef.current?.postMessage({ type: "prompt", prompt: aiPrompt });
  }, [aiPrompt]);

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
      if (e.metaKey || e.ctrlKey || e.altKey) return; // leave OS shortcuts alone
      if (isTyping()) return;

      // 1..9 → preset prompt
      if (/^[1-9]$/.test(e.key)) {
        const idx = parseInt(e.key, 10) - 1;
        const preset = aiPromptPresets[idx];
        if (preset) {
          setAiPrompt(preset);
          e.preventDefault();
        }
        return;
      }

      if (e.key === " ") {
        // Space → toggle send-frames (freeze / unfreeze the AI feed)
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
    setAiHideUi,
    setAiKleinAlpha,
    setAiPrompt,
    setAiSendFrames,
  ]);

  const [aiStatus, setAiStatus] = useState<AiTransportStatus>("idle");
  const [aiLogs, setAiLogs] = useState<string[]>([]);
  const [aiImageUrl, setAiImageUrl] = useState<string | null>(null);
  const [aiGenTime, setAiGenTime] = useState<number | null>(null);

  // DMX/Lighting UI state
  const [dmxStatus, setDmxStatus] = useState<DmxStatus>("disconnected");
  const [dmxSupported, setDmxSupported] = useState<boolean>(true);
  const [showLighting, setShowLighting] = useState<boolean>(true);
  const [fixtureValues, setFixtureValues] = useState<Map<string, Uint8Array>>(
    new Map()
  );
  const [selectedProfileId, setSelectedProfileId] = useState<string>(
    FIXTURE_PROFILES[0].id
  );

  // Zustand store for fixtures (persisted to localStorage)
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

      // Cleanup previous instances
      lightingEngineRef.current?.stop();
      lightingEngineRef.current = null;
      visualEngineRef.current?.stop();
      visualEngineRef.current = null;
      audioEngineRef.current?.destroy();
      audioEngineRef.current = null;

      setStatus("requesting");
      setErrorMessage("");

      try {
        // Create and initialize audio engine
        const audioEngine = new AudioEngine();
        await audioEngine.init(deviceId);
        audioEngineRef.current = audioEngine;

        // Create visual engine with all scenes
        const visualEngine = new VisualEngine(canvas, audioEngine, SCENES);
        visualEngineRef.current = visualEngine;

        // Create lighting engine with fixtures from store
        const lightingEngine = new LightingEngine(
          canvas,
          fixtureList ?? fixtures,
          { tickHz: 30 }
        );
        lightingEngine.setAudioEngine(audioEngine);
        lightingEngineRef.current = lightingEngine;

        // Subscribe to lighting frames
        lightingEngine.onFrame(handleLightingFrame);
        lightingEngine.onFrame(handleDmxFrame);

        // Start engines
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

  // ============================================================================
  // Fixture handlers - update both zustand store and lighting engine
  // ============================================================================

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

  // AI transport status + frame subscriptions
  useEffect(() => {
    const handleStatus = (s: AiTransportStatus) => {
      setAiStatus(s);
      stageChannelRef.current?.postMessage({ type: "connection", status: s });
    };
    aiTransport.onStatusChange(handleStatus);

    const handleFrame = (frame: AiIncomingFrame) => {
      if (frame.kind === "text") {
        // Try to parse as JSON for stats
        try {
          const data = JSON.parse(frame.message);
          if (data.type === "stats" && data.gen_time_ms) {
            setAiGenTime(data.gen_time_ms);
            return; // Don't log stats messages
          }
        } catch {
          // Not JSON, just log it
        }
        setAiLogs((prev) => [...prev, `← ${frame.message}`].slice(-20));
        return;
      }

      // Image blob: show locally + forward to stage tab (audience view).
      const url = URL.createObjectURL(frame.blob);
      setAiImageUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      // structured-clone the bytes into the channel (~10KB JPEG / frame)
      frame.blob
        .arrayBuffer()
        .then((buf) => {
          stageChannelRef.current?.postMessage({
            type: "frame",
            bytes: buf,
            width: aiOutputSize,
            height: aiOutputSize,
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
  }, [aiTransport, aiOutputSize]);

  // Revoke object URL on unmount
  useEffect(() => {
    return () => {
      if (aiImageUrl) URL.revokeObjectURL(aiImageUrl);
    };
  }, [aiImageUrl]);

  const aiSettingsResumeTimeoutRef = useRef<number | null>(null);

  // Debug canvas to show what we're sending to AI
  const aiDebugCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // AI frame sender - refs only, no React lifecycle overhead
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

  // Frame sender loop - runs outside React, uses RAF
  const aiFrameLoop = useCallback(() => {
    const sender = aiFrameSenderRef.current;
    if (!sender.running) return;

    requestAnimationFrame(aiFrameLoop);

    const now = performance.now();
    const frameInterval = 1000 / aiFrameRate;
    
    // Rate limit by frame interval
    if (now - sender.lastFrameTime < frameInterval) {
      return;
    }

    const src = canvasRef.current;
    if (!src) {
      return;
    }

    // Lazy init or resize capture canvas
    if (!sender.captureCanvas || sender.resolution !== aiCaptureSize) {
      sender.captureCanvas = document.createElement("canvas");
      sender.captureCanvas.width = aiCaptureSize;
      sender.captureCanvas.height = aiCaptureSize;
      // willReadFrequently no longer needed — we use toBlob (GPU path) instead of getImageData
      sender.captureCtx = sender.captureCanvas.getContext("2d");
      sender.resolution = aiCaptureSize;
    }

    const ctx = sender.captureCtx;
    if (!ctx) {
      return;
    }

    // Crop center square from source
    const srcSize = Math.min(src.width, src.height);
    const srcX = (src.width - srcSize) / 2;
    const srcY = (src.height - srcSize) / 2;

    ctx.drawImage(
      src,
      srcX, srcY, srcSize, srcSize,
      0, 0, aiCaptureSize, aiCaptureSize
    );

    sender.lastFrameTime = now;
    sender.frameCount++;

    // Always paint debug canvas so the preview works without a connection.
    const debugCanvas = aiDebugCanvasRef.current;
    if (debugCanvas) {
      if (!sender.debugCtx || debugCanvas.width !== aiCaptureSize) {
        debugCanvas.width = aiCaptureSize;
        debugCanvas.height = aiCaptureSize;
        sender.debugCtx = debugCanvas.getContext("2d");
      }
      sender.debugCtx?.drawImage(sender.captureCanvas, 0, 0);
    }

    // Only send over the wire when connected and there's headroom.
    if (!aiTransport.isConnected() || !aiTransport.canSend(256 * 1024)) {
      return;
    }
    // Skip if previous encode is still pending — avoids overlapping toBlob calls
    // stacking up if network or CPU stalls.
    if (sender.pendingEncode) {
      return;
    }

    // Encode the capture canvas as JPEG via the browser's native (GPU-backed)
    // encoder. Async, does not block the RAF loop. At 512² this is ~1-2ms and
    // produces ~8-15KB of bytes — fits comfortably in one SCTP datagram, unlike
    // raw RGB (196KB @ 256², which had reliability issues on the datachannel).
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
  }, [aiTransport, aiCaptureSize, aiFrameRate]);

  // Send prompt/settings to AI when they change
  useEffect(() => {
    if (!aiTransport.isConnected()) return;

    const sender = aiFrameSenderRef.current;

    // Pause frame sending temporarily to let settings take effect immediately
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
    }, 100); // Resume after settings are pushed to server.

    const payload: Record<string, unknown> = {
      prompt: aiPrompt,
      seed: aiSeed,
      captureWidth: aiCaptureSize,
      captureHeight: aiCaptureSize,
      width: aiOutputSize,
      height: aiOutputSize,
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
    aiOutputSize,
    aiKleinAlpha,
    aiKleinSteps,
    aiTransport,
    aiSendFrames,
    aiFrameLoop,
  ]);

  // Start/stop frame sender imperatively
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

  // Debug features polling - throttled to ~10fps
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

  // Check WebUSB support and auto-connect
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

  // Sync lighting engine with fixture changes
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

  // Initial setup
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

  return (
    <div className="flex flex-col gap-4 w-full max-w-4xl mx-auto p-4">
      {/* Hide-UI overlay: fullscreen AI output, sits on top when H is pressed.
          Listeners (keydown, transport, frame loop) stay mounted in the tree
          below so nothing reconnects. Click overlay or press H to exit. */}
      {aiHideUi && (
        <div
          onClick={() => setAiHideUi(false)}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 50,
            background: "#000",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "none",
          }}
        >
          {aiImageUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={aiImageUrl}
              alt="AI output"
              style={{
                maxWidth: "100%",
                maxHeight: "100%",
                objectFit: "contain",
                imageRendering: aiUpscaleMode === "bilinear" ? "auto" : "auto",
              }}
            />
          )}
        </div>
      )}

      <StatusBar
        status={status}
        scenes={SCENES}
        currentSceneId={currentSceneId}
        onSceneChange={handleSceneChange}
        devices={devices}
        selectedDeviceId={selectedDeviceId}
        onDeviceChange={handleDeviceChange}
      />

      {status === "error" && errorMessage && (
        <div className="text-red-400 text-sm font-mono bg-red-950/30 border border-red-900 rounded px-3 py-2">
          {errorMessage}
        </div>
      )}

      {/* Waveform + AI output preview */}
      <div
        className={
          aiSendFrames
            ? "grid grid-cols-1 md:grid-cols-[minmax(300px,1fr)_minmax(460px,1.6fr)] gap-3 items-start"
            : "grid grid-cols-1"
        }
      >
        <div className="relative w-full aspect-4/1 bg-neutral-950 rounded-lg overflow-hidden border border-neutral-800">
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ imageRendering: "pixelated" }}
          />
        </div>

        {aiSendFrames && (
          <div className="relative w-full min-h-[260px] bg-neutral-950 rounded-lg overflow-hidden border border-neutral-800">
            {aiImageUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={aiImageUrl}
                alt="AI generated output"
                className="w-full h-full object-contain"
                style={{
                  // pixelated forces nearest-neighbor (crisp but blocky);
                  // auto lets the browser use bicubic/Lanczos, which is what
                  // we want for either quality mode.
                  imageRendering: "auto",
                }}
              />
            ) : (
              <div className="w-full h-full text-xs font-mono text-neutral-600 flex items-center justify-center">
                {aiStatus === "connected"
                  ? "Waiting for AI frames..."
                  : "Connect AI to preview output"}
              </div>
            )}
          </div>
        )}
      </div>

      {/* AI toggle */}
      <button
        onClick={() => setShowAi(!showAi)}
        className="self-start text-xs font-mono text-neutral-500 hover:text-neutral-300 transition-colors"
      >
        {showAi ? "Hide" : "Show"} Remote AI (WebRTC)
      </button>

      {showAi && (
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3 flex flex-col gap-3">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs font-mono text-neutral-300">
              AI link:{" "}
              <span
                className={
                  aiStatus === "connected"
                    ? "text-emerald-400"
                    : aiStatus === "error"
                    ? "text-red-400"
                    : "text-neutral-400"
                }
              >
                {aiStatus}
              </span>
              {aiGenTime && aiStatus === "connected" && (
                <span className="ml-2 text-cyan-400">
                  {aiGenTime.toFixed(0)}ms ({(1000 / aiGenTime).toFixed(0)} FPS)
                </span>
              )}
            </div>

            <div className="flex items-center gap-2">
              <label className="flex items-center gap-2 text-xs font-mono text-neutral-400">
                Backend:
                <select
                  value={aiBackend}
                  onChange={(e) => {
                    void aiTransport.stop();
                    setAiBackend(e.target.value as AiBackend);
                  }}
                  className="bg-black/50 border border-neutral-800 rounded px-2 py-1 text-neutral-200 max-w-[22em]"
                  title={AI_BACKEND_LABELS[aiBackend]}
                >
                  {(Object.keys(AI_BACKEND_URLS) as AiBackend[]).map((k) => (
                    <option key={k} value={k}>
                      {AI_BACKEND_LABELS[k]}
                    </option>
                  ))}
                </select>
              </label>
              <button
                onClick={() => void aiTransport.start()}
                disabled={aiStatus === "connecting" || aiStatus === "connected"}
                className="text-xs font-mono px-3 py-1 rounded border border-emerald-900 text-emerald-400 hover:bg-emerald-950/30 disabled:opacity-40 disabled:hover:bg-transparent transition-colors"
              >
                Connect RunPod AI
              </button>
              <button
                onClick={() => void aiTransport.stop()}
                disabled={aiStatus === "idle" || aiStatus === "disconnected"}
                className="text-xs font-mono px-3 py-1 rounded border border-neutral-800 text-neutral-300 hover:bg-neutral-900/50 disabled:opacity-40 disabled:hover:bg-transparent transition-colors"
              >
                Disconnect
              </button>
            </div>
          </div>

          {/* Prompt input — draft-based. Enter commits to the store (and server).
              Hotkeys 1-9 also commit presets directly. */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between gap-2">
              <label className="text-xs font-mono text-neutral-500">AI Prompt</label>
              <span
                className={`text-[10px] font-mono ${
                  aiPromptDirty ? "text-amber-400" : "text-neutral-600"
                }`}
              >
                {aiPromptDirty ? "● unsaved — press Enter to send" : "✓ live"}
              </span>
            </div>
            <div className="flex items-stretch gap-2">
              <input
                type="text"
                value={aiPromptDraft}
                onChange={(e) => setAiPromptDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    setAiPrompt(aiPromptDraft);
                  } else if (e.key === "Escape") {
                    e.preventDefault();
                    setAiPromptDraft(aiPrompt);
                  }
                }}
                className="w-full px-3 py-2 text-sm font-mono bg-black/50 border border-neutral-800 rounded text-neutral-200 placeholder-neutral-600 focus:border-emerald-900 focus:outline-none"
                placeholder="Describe the visual style, press Enter to send…"
              />
              <button
                onClick={() => setAiPrompt(aiPromptDraft)}
                disabled={!aiPromptDirty}
                className="text-xs font-mono px-3 py-1 rounded border border-emerald-900 text-emerald-400 hover:bg-emerald-950/30 disabled:opacity-40 disabled:hover:bg-transparent transition-colors"
                title="Enter"
              >
                Send ↵
              </button>
            </div>
            {/* Preset chips (also bound to hotkeys 1-9) */}
            <div className="flex flex-wrap gap-1 pt-1">
              {aiPromptPresets.map((p, i) => {
                const active = p === aiPrompt;
                return (
                  <button
                    key={i}
                    onClick={() => setAiPrompt(p)}
                    className={`text-[10px] font-mono px-2 py-1 rounded border transition-colors ${
                      active
                        ? "border-emerald-700 bg-emerald-950/40 text-emerald-300"
                        : "border-neutral-800 text-neutral-500 hover:text-neutral-300 hover:bg-neutral-900/60"
                    }`}
                    title={p}
                  >
                    <span className="opacity-50 mr-1">{i + 1}</span>
                    {p.slice(0, 32)}
                    {p.length > 32 ? "…" : ""}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-xs font-mono text-neutral-400">
              <input
                type="checkbox"
                checked={aiSendFrames}
                onChange={(e) => setAiSendFrames(e.target.checked)}
                disabled={!aiTransport.isConnected()}
                className="accent-emerald-500"
              />
              Generate AI visuals
            </label>

            <label
              className="flex items-center gap-2 text-xs font-mono text-neutral-400"
              title={
                aiBackend === "klein"
                  ? "For Klein best quality, match Capture to Output (both 256, or both 512). Mismatch = server upscales the input and output gets blurry."
                  : "Client capture resolution. Smaller = less data to send, but blurrier input."
              }
            >
              Capture:
              <select
                value={aiCaptureSize}
                onChange={(e) => setAiCaptureSize(Number(e.target.value))}
                className="bg-black/50 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
              >
                <option value={64}>64 (fastest)</option>
                <option value={128}>128</option>
                <option value={256}>256</option>
                <option value={512}>512</option>
              </select>
              {aiBackend === "klein" && aiCaptureSize !== aiOutputSize && (
                <button
                  onClick={() => setAiCaptureSize(aiOutputSize)}
                  className="px-2 py-0.5 rounded border border-amber-900 text-[10px] text-amber-400 hover:bg-amber-950/30"
                  title={`Match capture to output (${aiOutputSize}) for sharper Klein results`}
                >
                  → {aiOutputSize}
                </button>
              )}
            </label>

            <label className="flex items-center gap-2 text-xs font-mono text-neutral-400">
              Output:
              <select
                value={aiOutputSize}
                onChange={(e) => setAiOutputSize(Number(e.target.value))}
                className="bg-black/50 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
              >
                <option value={256}>256x256</option>
                <option value={512}>512x512</option>
                <option value={768}>768x768</option>
                <option value={1024}>1024x1024</option>
              </select>
            </label>

            <label className="flex items-center gap-2 text-xs font-mono text-neutral-400">
              Target FPS:
              <select
                value={aiFrameRate}
                onChange={(e) => setAiFrameRate(Number(e.target.value))}
                className="bg-black/50 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
              >
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={30}>30</option>
                <option value={60}>60</option>
              </select>
            </label>

            <div className="flex items-center gap-2 text-xs font-mono text-neutral-400">
              Seed: {aiSeed}
              <button
                onClick={() => setAiSeed(Math.floor(Math.random() * 1000000))}
                className="px-2 py-1 rounded border border-cyan-900 text-cyan-400 hover:bg-cyan-950/30 transition-colors"
              >
                Randomize
              </button>
            </div>
          </div>

          {/* Klein-specific controls */}
          {aiBackend === "klein" && (
            <div className="flex flex-wrap items-center gap-4 pt-2 border-t border-neutral-900">
              <div className="text-[10px] font-mono uppercase tracking-wider text-neutral-600">
                Klein
              </div>
              <label className="flex items-center gap-2 text-xs font-mono text-neutral-400">
                Alpha: <span className="tabular-nums w-10 text-right">{aiKleinAlpha.toFixed(2)}</span>
                <input
                  type="range"
                  min={0}
                  max={0.5}
                  step={0.01}
                  value={aiKleinAlpha}
                  onChange={(e) => setAiKleinAlpha(Number(e.target.value))}
                  className="accent-emerald-500 w-40"
                  title="0 = pure prompt (ignore input)  ·  0.05-0.10 = subtle SDXL-turbo-like nudge  ·  0.15-0.25 = waveform shapes composition  ·  0.3+ = waveform dominates"
                />
              </label>
              <label className="flex items-center gap-2 text-xs font-mono text-neutral-400">
                Steps:
                <select
                  value={aiKleinSteps}
                  onChange={(e) => setAiKleinSteps(Number(e.target.value))}
                  className="bg-black/50 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
                  title="Klein is distilled for 4 steps. 2 = fastest with good quality. 3/4 = higher quality, slower."
                >
                  <option value={1}>1 (fastest, lower quality)</option>
                  <option value={2}>2 (recommended)</option>
                  <option value={3}>3 (more detail)</option>
                  <option value={4}>4 (max)</option>
                </select>
              </label>
            </div>
          )}

          {/* Live-performance controls */}
          <div className="flex flex-wrap items-center gap-4 pt-2 border-t border-neutral-900">
            <div className="text-[10px] font-mono uppercase tracking-wider text-neutral-600">
              Live
            </div>
            <label
              className="flex items-center gap-2 text-xs font-mono text-neutral-400"
              title="Interpolation when the AI output is scaled up for display. Lanczos (high) is sharper; Bilinear is softer."
            >
              Upscale:
              <select
                value={aiUpscaleMode}
                onChange={(e) => setAiUpscaleMode(e.target.value as UpscaleMode)}
                className="bg-black/50 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
              >
                <option value="lanczos">Lanczos (sharp)</option>
                <option value="bilinear">Bilinear (soft)</option>
              </select>
            </label>
            <a
              href="/vj/stage"
              target="_blank"
              rel="noreferrer"
              className="text-xs font-mono px-3 py-1 rounded border border-cyan-900 text-cyan-400 hover:bg-cyan-950/30 transition-colors"
              title="Open the audience-only fullscreen output in a new tab. Drag it to your projector screen + press F11."
            >
              Open Stage ↗
            </a>
            <button
              onClick={() => setAiHideUi(!aiHideUi)}
              className="text-xs font-mono px-3 py-1 rounded border border-neutral-800 text-neutral-300 hover:bg-neutral-900/50 transition-colors"
              title="Toggle a quick fullscreen-output mode on this tab (hotkey: H)"
            >
              {aiHideUi ? "Show UI (H)" : "Hide UI (H)"}
            </button>
            <div className="text-[10px] font-mono text-neutral-600 whitespace-nowrap">
              keys: <span className="text-neutral-400">1-9</span> presets ·{" "}
              <span className="text-neutral-400">Space</span> freeze ·{" "}
              <span className="text-neutral-400">↑↓</span> alpha ·{" "}
              <span className="text-neutral-400">H</span> hide UI
            </div>
          </div>

          <label className="flex items-center gap-2 text-xs font-mono text-neutral-400">
            <input
              type="checkbox"
              checked={aiShowCaptureDebug}
              onChange={(e) => setAiShowCaptureDebug(e.target.checked)}
              className="accent-cyan-500"
            />
            Show &quot;Sending to AI&quot; debug preview
          </label>

          {aiShowCaptureDebug && (
            <div className="inline-flex flex-col gap-1 rounded border border-neutral-800 bg-black/30 p-2 self-start">
              <div className="text-xs font-mono text-neutral-500">
                Sending to AI ({aiCaptureSize}x{aiCaptureSize} → {aiOutputSize}x{aiOutputSize})
              </div>
              <canvas
                ref={aiDebugCanvasRef}
                width={aiCaptureSize}
                height={aiCaptureSize}
                className="rounded bg-black"
                style={{ width: 128, height: 128, imageRendering: "pixelated" }}
              />
            </div>
          )}

          {/* Collapsible logs */}
          <details className="text-xs">
            <summary className="font-mono text-neutral-500 cursor-pointer hover:text-neutral-300">
              Debug logs ({aiLogs.length})
            </summary>
            <div className="mt-2 h-20 overflow-auto font-mono text-neutral-400 bg-black/30 rounded p-2">
              {aiLogs.map((line, idx) => (
                <div key={idx} className="whitespace-pre-wrap">{line}</div>
              ))}
            </div>
          </details>
        </div>
      )}

      {/* Debug toggle */}
      <button
        onClick={() => setShowDebug(!showDebug)}
        className="self-start text-xs font-mono text-neutral-500 hover:text-neutral-300 transition-colors"
      >
        {showDebug ? "Hide" : "Show"} Audio Features Debug
      </button>

      {showDebug && <AudioDebugPanel features={debugFeatures} />}

      {/* Lighting toggle */}
      <button
        onClick={() => setShowLighting(!showLighting)}
        className="self-start text-xs font-mono text-neutral-500 hover:text-neutral-300 transition-colors"
      >
        {showLighting ? "Hide" : "Show"} Lighting / DMX
      </button>

      {showLighting && (
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
      )}

      <p className="text-xs text-neutral-500 font-mono">
        vj0 - Audio-reactive visuals. Select a scene and connect a USB audio
        interface for best results.
      </p>
    </div>
  );
}
