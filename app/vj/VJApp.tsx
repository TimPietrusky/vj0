"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { AudioEngine } from "@/src/lib/audio-engine";
import { VisualEngine, SCENES } from "@/src/lib/scenes";
import {
  LightingEngine,
  DmxOutput,
  FIXTURE_PROFILES,
} from "@/src/lib/lighting";
import { useLightingStore, useFixtures } from "@/src/lib/stores";
import type { AudioFeatures } from "@/src/lib/audio-features";
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

      {/* Canvas container */}
      <div className="relative w-full aspect-4/1 bg-neutral-950 rounded-lg overflow-hidden border border-neutral-800">
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ imageRendering: "pixelated" }}
        />
      </div>

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
