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

  // AI signaling URL - defaults to local, can be overridden for RunPod
  const aiSignalingUrl = useMemo(() => {
    const url = process.env.NEXT_PUBLIC_VJ0_WEBRTC_SIGNALING_URL;
    return url || "/api/webrtc/offer";
  }, []);

  // AI transport (WebRTC) - keep as stable instance
  const aiTransport = useMemo(
    () =>
      new WebRtcAiTransport({
        signalingUrl: aiSignalingUrl,
        iceServers: aiIceServers,
        iceTransportPolicy: aiIceTransportPolicy,
      }),
    [aiSignalingUrl, aiIceServers, aiIceTransportPolicy]
  );

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
    showAi,
    sendFrames: aiSendFrames,
    showCaptureDebug: aiShowCaptureDebug,
    prompt: aiPrompt,
    captureSize: aiCaptureSize,
    outputSize: aiOutputSize,
    frameRate: aiFrameRate,
    seed: aiSeed,
    setShowAi,
    setSendFrames: setAiSendFrames,
    setShowCaptureDebug: setAiShowCaptureDebug,
    setPrompt: setAiPrompt,
    setCaptureSize: setAiCaptureSize,
    setOutputSize: setAiOutputSize,
    setFrameRate: setAiFrameRate,
    setSeed: setAiSeed,
  } = useAiSettingsStore();

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
    const handleStatus = (s: AiTransportStatus) => setAiStatus(s);
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

      // Handle image blob
      const url = URL.createObjectURL(frame.blob);
      setAiImageUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
    };

    aiTransport.onFrame(handleFrame);
    return () => {
      aiTransport.offFrame(handleFrame);
      aiTransport.offStatusChange(handleStatus);
    };
  }, [aiTransport]);

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
    rgbBuffer: Uint8Array | null;
    lastFrameTime: number;
    resolution: number;
    frameCount: number;
  }>({
    running: false,
    captureCanvas: null,
    captureCtx: null,
    debugCtx: null,
    rgbBuffer: null,
    lastFrameTime: 0,
    resolution: 0,
    frameCount: 0,
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
    if (!src || !aiTransport.isConnected()) {
      return;
    }

    // Backpressure: skip frame if send buffer is full (256KB threshold)
    // This prevents queue overflow when generation is slower than capture
    if (!aiTransport.canSend(256 * 1024)) {
      return;
    }

    // Lazy init or resize capture canvas
    if (!sender.captureCanvas || sender.resolution !== aiCaptureSize) {
      sender.captureCanvas = document.createElement("canvas");
      sender.captureCanvas.width = aiCaptureSize;
      sender.captureCanvas.height = aiCaptureSize;
      sender.captureCtx = sender.captureCanvas.getContext("2d", { willReadFrequently: true });
      sender.rgbBuffer = new Uint8Array(aiCaptureSize * aiCaptureSize * 3);
      sender.resolution = aiCaptureSize;
    }

    const ctx = sender.captureCtx;
    const rgbData = sender.rgbBuffer;
    if (!ctx || !rgbData) {
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

    // Get pixels and convert RGBA -> RGB
    const imageData = ctx.getImageData(0, 0, aiCaptureSize, aiCaptureSize);
    const rgba = imageData.data;
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      rgbData[j] = rgba[i];
      rgbData[j + 1] = rgba[i + 1];
      rgbData[j + 2] = rgba[i + 2];
    }

    // Send directly - no Blob allocation
    aiTransport.sendBinary(rgbData);
    sender.lastFrameTime = now;
    sender.frameCount++;

    // Update debug canvas less frequently (every 10 frames) to avoid perf hit
    if (sender.frameCount % 10 === 0) {
      const debugCanvas = aiDebugCanvasRef.current;
      if (debugCanvas) {
        if (!sender.debugCtx || debugCanvas.width !== aiCaptureSize) {
          debugCanvas.width = aiCaptureSize;
          debugCanvas.height = aiCaptureSize;
          sender.debugCtx = debugCanvas.getContext("2d");
        }
        sender.debugCtx?.drawImage(sender.captureCanvas!, 0, 0);
      }
    }
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

    aiTransport.sendText(
      JSON.stringify({
        prompt: aiPrompt,
        seed: aiSeed,
        captureWidth: aiCaptureSize,
        captureHeight: aiCaptureSize,
        width: aiOutputSize,
        height: aiOutputSize,
      })
    );
  }, [
    aiPrompt,
    aiSeed,
    aiCaptureSize,
    aiOutputSize,
    aiTransport,
    aiSendFrames,
    aiFrameLoop,
  ]);

  // Start/stop frame sender imperatively
  useEffect(() => {
    const sender = aiFrameSenderRef.current;
    
    if (aiSendFrames && aiTransport.isConnected()) {
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
  }, [aiSendFrames, aiStatus, aiFrameLoop, aiTransport]);

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

          {/* Prompt input */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-mono text-neutral-500">AI Prompt</label>
            <input
              type="text"
              value={aiPrompt}
              onChange={(e) => setAiPrompt(e.target.value)}
              className="w-full px-3 py-2 text-sm font-mono bg-black/50 border border-neutral-800 rounded text-neutral-200 placeholder-neutral-600 focus:border-emerald-900 focus:outline-none"
              placeholder="Describe the visual style..."
            />
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

            <label className="flex items-center gap-2 text-xs font-mono text-neutral-400">
              Capture:
              <select
                value={aiCaptureSize}
                onChange={(e) => setAiCaptureSize(Number(e.target.value))}
                className="bg-black/50 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
              >
                <option value={64}>64 (fastest)</option>
                <option value={128}>128</option>
                <option value={256}>256</option>
              </select>
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

          {aiShowCaptureDebug && (
            <div className="rounded border border-neutral-800 bg-black/30 p-2">
              <div className="text-xs font-mono text-neutral-500 mb-2">
                Sending to AI ({aiCaptureSize}x{aiCaptureSize} → {aiOutputSize}x{aiOutputSize})
              </div>
              <canvas
                ref={aiDebugCanvasRef}
                width={aiCaptureSize}
                height={aiCaptureSize}
                className="w-full h-auto rounded bg-black"
                style={{ imageRendering: "pixelated" }}
              />
            </div>
          )}

          <label className="flex items-center gap-2 text-xs font-mono text-neutral-400">
            <input
              type="checkbox"
              checked={aiShowCaptureDebug}
              onChange={(e) => setAiShowCaptureDebug(e.target.checked)}
              className="accent-cyan-500"
            />
            Show &quot;Sending to AI&quot; debug preview
          </label>

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
