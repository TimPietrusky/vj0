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

type Status = "idle" | "requesting" | "running" | "error";
type DmxStatus = "disconnected" | "connecting" | "connected" | "unsupported";

interface AudioDevice {
  deviceId: string;
  label: string;
}

// Channel kind to display label mapping
const CHANNEL_LABELS: Record<string, string> = {
  red: "R",
  green: "G",
  blue: "B",
  uv: "UV",
  dimmer: "DIM",
  strobe: "STR",
  program: "PRG",
  programSpeed: "SPD",
};

// Channel kind to color mapping for bars
const CHANNEL_COLORS: Record<string, string> = {
  red: "bg-red-500",
  green: "bg-green-500",
  blue: "bg-blue-500",
  uv: "bg-violet-500",
  dimmer: "bg-amber-500",
  strobe: "bg-white",
  program: "bg-neutral-500",
  programSpeed: "bg-neutral-500",
};

export function VJWaveform() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioEngineRef = useRef<AudioEngine | null>(null);
  const visualEngineRef = useRef<VisualEngine | null>(null);
  const lightingEngineRef = useRef<LightingEngine | null>(null);
  const dmxOutputRef = useRef<DmxOutput | null>(null);

  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [currentSceneId, setCurrentSceneId] = useState<string>(SCENES[0].id);

  // Debug panel state
  const [debugFeatures, setDebugFeatures] = useState<AudioFeatures | null>(
    null
  );
  const [showDebug, setShowDebug] = useState<boolean>(true);

  // DMX/Lighting state
  const [dmxStatus, setDmxStatus] = useState<DmxStatus>("disconnected");
  const [dmxSupported, setDmxSupported] = useState<boolean>(true);
  const [showLighting, setShowLighting] = useState<boolean>(true);
  const [fixtureValues, setFixtureValues] = useState<Map<string, Uint8Array>>(
    new Map()
  );
  const [selectedProfileId, setSelectedProfileId] = useState<string>(
    FIXTURE_PROFILES[0].id
  );

  // Zustand store for fixtures
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

  // Fetch available audio input devices
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

  // Handle lighting frame for fixture inspector
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

  // Handle DMX frame sending
  const handleDmxFrame = useCallback((frame: LightingFrame) => {
    const dmx = dmxOutputRef.current;
    if (dmx && dmx.isConnected()) {
      dmx.sendUniverse(frame.universe);
    }
  }, []);

  // Initialize audio and start rendering
  const initAudio = useCallback(
    async (deviceId?: string, fixtureList?: FixtureInstance[]) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      // Cleanup previous instances
      if (lightingEngineRef.current) {
        lightingEngineRef.current.stop();
        lightingEngineRef.current = null;
      }
      if (visualEngineRef.current) {
        visualEngineRef.current.stop();
        visualEngineRef.current = null;
      }
      if (audioEngineRef.current) {
        audioEngineRef.current.destroy();
        audioEngineRef.current = null;
      }

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

        // Subscribe to lighting frames for fixture inspector
        lightingEngine.onFrame(handleLightingFrame);

        // Connect lighting to DMX output
        lightingEngine.onFrame(handleDmxFrame);

        // Start engines
        visualEngine.start();
        lightingEngine.start();

        setStatus("running");
        setCurrentSceneId(visualEngine.getCurrentScene().id);

        // Refresh device list
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

  // Handle device selection change
  const handleDeviceChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const deviceId = e.target.value;
      setSelectedDeviceId(deviceId);
      initAudio(deviceId || undefined);
    },
    [initAudio]
  );

  // Handle scene selection change
  const handleSceneChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const sceneId = e.target.value;
      const engine = visualEngineRef.current;
      if (engine && engine.setSceneById(sceneId)) {
        setCurrentSceneId(sceneId);
      }
    },
    []
  );

  // DMX connect
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

  // DMX disconnect
  const handleDmxDisconnect = useCallback(async () => {
    const dmx = dmxOutputRef.current;
    if (dmx) {
      await dmx.disconnect();
      setDmxStatus("disconnected");
    }
  }, []);

  // Handle fixture address change
  const handleFixtureAddressChange = useCallback(
    (fixtureId: string, newAddress: number) => {
      updateFixtureAddress(fixtureId, newAddress);
      lightingEngineRef.current?.updateFixtureAddress(fixtureId, newAddress);
    },
    [updateFixtureAddress]
  );

  // Handle strobe mode change
  const handleStrobeModeChange = useCallback(
    (fixtureId: string, mode: StrobeMode) => {
      updateFixtureStrobeMode(fixtureId, mode);
      lightingEngineRef.current?.updateFixtureStrobeMode(fixtureId, mode);
    },
    [updateFixtureStrobeMode]
  );

  // Handle strobe threshold change
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

  // Handle strobe max change
  const handleStrobeMaxChange = useCallback(
    (fixtureId: string, max: number) => {
      updateFixtureStrobeMax(fixtureId, max);
      lightingEngineRef.current?.updateFixtureStrobeMax(fixtureId, max);
    },
    [updateFixtureStrobeMax]
  );

  // Handle color mode change
  const handleColorModeChange = useCallback(
    (fixtureId: string, mode: ColorMode) => {
      updateFixtureColorMode(fixtureId, mode);
      lightingEngineRef.current?.updateFixtureColorMode(fixtureId, mode);
    },
    [updateFixtureColorMode]
  );

  // Handle solid color change
  const handleSolidColorChange = useCallback(
    (fixtureId: string, color: { r: number; g: number; b: number }) => {
      updateFixtureSolidColor(fixtureId, color);
      lightingEngineRef.current?.updateFixtureSolidColor(fixtureId, color);
    },
    [updateFixtureSolidColor]
  );

  // Handle adding a new fixture
  const handleAddFixture = useCallback(() => {
    addFixture(selectedProfileId);
  }, [addFixture, selectedProfileId]);

  // Handle removing a fixture
  const handleRemoveFixture = useCallback(
    (fixtureId: string) => {
      removeFixture(fixtureId);
    },
    [removeFixture]
  );

  // Handle fixture profile change
  const handleFixtureProfileChange = useCallback(
    (fixtureId: string, profileId: string) => {
      updateFixtureProfile(fixtureId, profileId);
    },
    [updateFixtureProfile]
  );

  // Debug features polling
  useEffect(() => {
    if (!showDebug) return;

    const interval = setInterval(() => {
      const engine = audioEngineRef.current;
      if (engine) {
        const features = engine.getLatestFeatures();
        setDebugFeatures(features);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [showDebug]);

  // Check WebUSB support and auto-connect to paired device
  useEffect(() => {
    const supported = DmxOutput.isSupported();
    setDmxSupported(supported);
    if (!supported) {
      setDmxStatus("unsupported");
      return;
    }

    // Try to auto-connect to a previously paired device
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
      // Reinitialize lighting engine with new fixtures
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
      // Cleanup on unmount
      if (lightingEngineRef.current) {
        lightingEngineRef.current.stop();
      }
      if (visualEngineRef.current) {
        visualEngineRef.current.stop();
      }
      if (audioEngineRef.current) {
        audioEngineRef.current.destroy();
      }
      if (dmxOutputRef.current) {
        dmxOutputRef.current.disconnect();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex flex-col gap-4 w-full max-w-4xl mx-auto p-4">
      {/* Header with status, device selector, and scene selector */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono uppercase tracking-wide text-neutral-400">
            Status:
          </span>
          <span
            className={`text-sm font-mono uppercase tracking-wide ${
              status === "running"
                ? "text-emerald-400"
                : status === "error"
                ? "text-red-400"
                : "text-amber-400"
            }`}
          >
            {status === "idle" && "Initializing"}
            {status === "requesting" && "Requesting audio permission"}
            {status === "running" && "Running"}
            {status === "error" && "Error"}
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* Scene selector */}
          <select
            value={currentSceneId}
            onChange={handleSceneChange}
            className="bg-neutral-900 border border-neutral-700 text-neutral-200 text-sm rounded px-3 py-1.5 focus:outline-none focus:border-emerald-500"
          >
            {SCENES.map((scene) => (
              <option key={scene.id} value={scene.id}>
                {scene.name}
              </option>
            ))}
          </select>

          {/* Device selector */}
          {devices.length > 1 && (
            <select
              value={selectedDeviceId}
              onChange={handleDeviceChange}
              className="bg-neutral-900 border border-neutral-700 text-neutral-200 text-sm rounded px-3 py-1.5 focus:outline-none focus:border-emerald-500"
            >
              <option value="">Default device</option>
              {devices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Error message */}
      {status === "error" && errorMessage && (
        <div className="text-red-400 text-sm font-mono bg-red-950/30 border border-red-900 rounded px-3 py-2">
          {errorMessage}
        </div>
      )}

      {/* Canvas container */}
      <div className="relative w-full aspect-[4/1] bg-neutral-950 rounded-lg overflow-hidden border border-neutral-800">
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

      {/* Debug panel - AudioFeatures display */}
      {showDebug && (
        <div className="bg-neutral-900/80 border border-neutral-700 rounded-lg p-4 font-mono text-xs">
          <div className="text-neutral-400 uppercase tracking-wide mb-3">
            Audio Features
          </div>
          {debugFeatures ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <FeatureBar label="RMS" value={debugFeatures.rms} />
              <FeatureBar label="Peak" value={debugFeatures.peak} />
              <FeatureBar
                label="Low"
                value={debugFeatures.energyLow}
                color="text-red-400"
                barColor="bg-red-500"
              />
              <FeatureBar
                label="Mid"
                value={debugFeatures.energyMid}
                color="text-yellow-400"
                barColor="bg-yellow-500"
              />
              <FeatureBar
                label="High"
                value={debugFeatures.energyHigh}
                color="text-cyan-400"
                barColor="bg-cyan-500"
              />
              <FeatureBar
                label="Centroid"
                value={debugFeatures.spectralCentroid}
                color="text-purple-400"
                barColor="bg-purple-500"
              />
            </div>
          ) : (
            <div className="text-neutral-500">Waiting for audio data...</div>
          )}
        </div>
      )}

      {/* Lighting toggle */}
      <button
        onClick={() => setShowLighting(!showLighting)}
        className="self-start text-xs font-mono text-neutral-500 hover:text-neutral-300 transition-colors"
      >
        {showLighting ? "Hide" : "Show"} Lighting / DMX
      </button>

      {/* Lighting panel */}
      {showLighting && (
        <div className="bg-neutral-900/80 border border-neutral-700 rounded-lg p-4 font-mono text-xs">
          <div className="text-neutral-400 uppercase tracking-wide mb-3">
            Lighting / DMX
          </div>

          {/* DMX Connection controls */}
          <div className="flex items-center gap-4 mb-4">
            <div className="flex items-center gap-2">
              <span className="text-neutral-500">DMX:</span>
              <span
                className={`uppercase ${
                  dmxStatus === "connected"
                    ? "text-emerald-400"
                    : dmxStatus === "connecting"
                    ? "text-amber-400"
                    : dmxStatus === "unsupported"
                    ? "text-red-400"
                    : "text-neutral-400"
                }`}
              >
                {dmxStatus}
              </span>
            </div>

            {dmxSupported && dmxStatus !== "connected" && (
              <button
                onClick={handleDmxConnect}
                disabled={dmxStatus === "connecting"}
                className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-neutral-700 text-white rounded text-xs transition-colors"
              >
                Connect DMX
              </button>
            )}

            {dmxStatus === "connected" && (
              <button
                onClick={handleDmxDisconnect}
                className="px-3 py-1 bg-red-600 hover:bg-red-500 text-white rounded text-xs transition-colors"
              >
                Disconnect
              </button>
            )}

            {!dmxSupported && (
              <span className="text-red-400 text-xs">
                WebUSB not supported in this browser
              </span>
            )}
          </div>

          {/* Add Fixture */}
          <div className="flex items-center gap-3 mb-4 border-b border-neutral-700 pb-4">
            <span className="text-neutral-500 text-xs">Add Fixture:</span>
            <select
              value={selectedProfileId}
              onChange={(e) => setSelectedProfileId(e.target.value)}
              className="flex-1 bg-neutral-900 border border-neutral-600 text-neutral-200 text-xs rounded px-2 py-1.5 focus:outline-none focus:border-emerald-500"
            >
              {FIXTURE_PROFILES.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}
                </option>
              ))}
            </select>
            <button
              onClick={handleAddFixture}
              className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs transition-colors"
            >
              Add
            </button>
          </div>

          {/* Fixture Inspector */}
          <div className="space-y-4">
            {fixtures.length === 0 ? (
              <div className="text-neutral-500 text-center py-4">
                No fixtures configured. Add one above.
              </div>
            ) : (
              fixtures.map((fixture) => (
                <FixtureInspector
                  key={fixture.id}
                  fixture={fixture}
                  values={fixtureValues.get(fixture.id)}
                  onAddressChange={(addr) =>
                    handleFixtureAddressChange(fixture.id, addr)
                  }
                  onStrobeModeChange={(mode) =>
                    handleStrobeModeChange(fixture.id, mode)
                  }
                  onStrobeThresholdChange={(t) =>
                    handleStrobeThresholdChange(fixture.id, t)
                  }
                  onStrobeMaxChange={(m) =>
                    handleStrobeMaxChange(fixture.id, m)
                  }
                  onColorModeChange={(mode) =>
                    handleColorModeChange(fixture.id, mode)
                  }
                  onSolidColorChange={(color) =>
                    handleSolidColorChange(fixture.id, color)
                  }
                  onProfileChange={(profileId) =>
                    handleFixtureProfileChange(fixture.id, profileId)
                  }
                  onRemove={() => handleRemoveFixture(fixture.id)}
                />
              ))
            )}
          </div>
        </div>
      )}

      {/* Footer hint */}
      <p className="text-xs text-neutral-500 font-mono">
        vj0 - Audio-reactive visuals. Select a scene and connect a USB audio
        interface for best results.
      </p>
    </div>
  );
}

// Strobe mode options for the dropdown
const STROBE_MODE_OPTIONS: { value: StrobeMode; label: string }[] = [
  { value: "off", label: "Off" },
  { value: "energyLow", label: "Bass (Low)" },
  { value: "energyMid", label: "Mid" },
  { value: "energyHigh", label: "High" },
  { value: "peak", label: "Peak" },
  { value: "rms", label: "RMS" },
];

/**
 * Fixture inspector component - shows fixture info, address control, and channel values
 */
function FixtureInspector({
  fixture,
  values,
  onAddressChange,
  onStrobeModeChange,
  onStrobeThresholdChange,
  onStrobeMaxChange,
  onColorModeChange,
  onSolidColorChange,
  onProfileChange,
  onRemove,
}: {
  fixture: FixtureInstance;
  values?: Uint8Array;
  onAddressChange: (addr: number) => void;
  onStrobeModeChange: (mode: StrobeMode) => void;
  onStrobeThresholdChange: (threshold: number) => void;
  onStrobeMaxChange: (max: number) => void;
  onColorModeChange: (mode: ColorMode) => void;
  onSolidColorChange: (color: { r: number; g: number; b: number }) => void;
  onProfileChange: (profileId: string) => void;
  onRemove: () => void;
}) {
  const maxAddress = 512 - fixture.profile.channels.length + 1;
  const hasStrobe = fixture.profile.channels.includes("strobe");
  const {
    address,
    strobeMode,
    strobeThreshold,
    strobeMax,
    colorMode,
    solidColor,
  } = fixture;

  // Get RGB values for color preview
  const r = values?.[fixture.profile.channels.indexOf("red")] ?? 0;
  const g = values?.[fixture.profile.channels.indexOf("green")] ?? 0;
  const b = values?.[fixture.profile.channels.indexOf("blue")] ?? 0;

  return (
    <div className="bg-neutral-800/50 rounded p-3">
      {/* Header with profile selector and controls */}
      <div className="flex items-center justify-between mb-3 gap-3">
        <div className="flex items-center gap-3 flex-1">
          <select
            value={fixture.profile.id}
            onChange={(e) => onProfileChange(e.target.value)}
            className="bg-neutral-900 border border-neutral-600 text-neutral-200 text-xs rounded px-2 py-1 focus:outline-none focus:border-emerald-500"
          >
            {FIXTURE_PROFILES.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.name}
              </option>
            ))}
          </select>
          <span className="text-neutral-500 text-xs">ID: {fixture.id}</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-neutral-500 text-xs">DMX:</span>
            <input
              type="number"
              min={1}
              max={maxAddress}
              value={address}
              onChange={(e) =>
                onAddressChange(parseInt(e.target.value, 10) || 1)
              }
              className="w-14 bg-neutral-900 border border-neutral-600 text-neutral-200 text-xs rounded px-2 py-1 text-center focus:outline-none focus:border-emerald-500"
            />
          </div>
          <button
            onClick={onRemove}
            className="px-2 py-1 bg-red-600/20 hover:bg-red-600/40 text-red-400 rounded text-xs transition-colors"
            title="Remove fixture"
          >
            Remove
          </button>
        </div>
      </div>

      {/* Color source controls */}
      <div className="flex items-center gap-4 mb-3">
        <div className="flex items-center gap-2">
          <span className="text-neutral-500 text-xs">Color:</span>
          <button
            onClick={() => onColorModeChange("canvas")}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              colorMode === "canvas"
                ? "bg-emerald-600 text-white"
                : "bg-neutral-700 text-neutral-400 hover:bg-neutral-600"
            }`}
          >
            Canvas
          </button>
          <button
            onClick={() => onColorModeChange("solid")}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              colorMode === "solid"
                ? "bg-emerald-600 text-white"
                : "bg-neutral-700 text-neutral-400 hover:bg-neutral-600"
            }`}
          >
            Solid
          </button>
        </div>

        {colorMode === "solid" && (
          <div className="flex items-center gap-2">
            <input
              type="color"
              value={`#${solidColor.r
                .toString(16)
                .padStart(2, "0")}${solidColor.g
                .toString(16)
                .padStart(2, "0")}${solidColor.b
                .toString(16)
                .padStart(2, "0")}`}
              onChange={(e) => {
                const hex = e.target.value;
                const r = parseInt(hex.slice(1, 3), 16);
                const g = parseInt(hex.slice(3, 5), 16);
                const b = parseInt(hex.slice(5, 7), 16);
                onSolidColorChange({ r, g, b });
              }}
              className="w-8 h-6 rounded border border-neutral-600 cursor-pointer"
            />
            <span className="text-neutral-400 text-xs">
              {solidColor.r}, {solidColor.g}, {solidColor.b}
            </span>
          </div>
        )}
      </div>

      {/* Color preview and channel bars */}
      <div className="flex gap-4 mb-3">
        {/* RGB Color preview */}
        <div
          className="w-12 h-12 rounded border border-neutral-600 flex-shrink-0"
          style={{ backgroundColor: `rgb(${r}, ${g}, ${b})` }}
          title={`RGB: ${r}, ${g}, ${b}`}
        />

        {/* Channel bars */}
        <div className="flex-1 grid grid-cols-3 sm:grid-cols-6 gap-2">
          {fixture.profile.channels.map((kind, i) => {
            const value = values?.[i] ?? 0;
            const label = CHANNEL_LABELS[kind] || kind;
            const barColor = CHANNEL_COLORS[kind] || "bg-neutral-500";

            return (
              <div key={i} className="flex flex-col gap-1">
                <div className="flex justify-between items-baseline">
                  <span className="text-neutral-500">{label}</span>
                  <span className="text-neutral-300">{value}</span>
                </div>
                <div className="h-1.5 bg-neutral-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${barColor} transition-all duration-75`}
                    style={{ width: `${(value / 255) * 100}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Strobe controls - only show if fixture has strobe channel */}
      {hasStrobe && (
        <div className="border-t border-neutral-700 pt-3 mt-3">
          <div className="text-neutral-400 text-xs uppercase tracking-wide mb-2">
            Audio-Reactive Strobe
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* Strobe mode selector */}
            <div className="flex flex-col gap-1">
              <label className="text-neutral-500 text-xs">Source</label>
              <select
                value={strobeMode}
                onChange={(e) =>
                  onStrobeModeChange(e.target.value as StrobeMode)
                }
                className="bg-neutral-900 border border-neutral-600 text-neutral-200 text-xs rounded px-2 py-1 focus:outline-none focus:border-emerald-500"
              >
                {STROBE_MODE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Threshold slider */}
            <div className="flex flex-col gap-1">
              <label className="text-neutral-500 text-xs">
                Threshold: {(strobeThreshold * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min={0}
                max={100}
                value={strobeThreshold * 100}
                onChange={(e) =>
                  onStrobeThresholdChange(parseInt(e.target.value, 10) / 100)
                }
                className="w-full accent-emerald-500"
                disabled={strobeMode === "off"}
              />
            </div>

            {/* Max speed slider */}
            <div className="flex flex-col gap-1">
              <label className="text-neutral-500 text-xs">
                Max Speed: {strobeMax}
              </label>
              <input
                type="range"
                min={0}
                max={255}
                value={strobeMax}
                onChange={(e) =>
                  onStrobeMaxChange(parseInt(e.target.value, 10))
                }
                className="w-full accent-emerald-500"
                disabled={strobeMode === "off"}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Individual feature bar component for debug display.
 */
function FeatureBar({
  label,
  value,
  color = "text-emerald-400",
  barColor = "bg-emerald-500",
}: {
  label: string;
  value: number;
  color?: string;
  barColor?: string;
}) {
  const percentage = Math.min(100, Math.max(0, value * 100));

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-baseline">
        <span className="text-neutral-400">{label}</span>
        <span className={color}>{value.toFixed(3)}</span>
      </div>
      <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} transition-all duration-75`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
