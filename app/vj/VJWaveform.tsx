'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { AudioEngine } from '@/src/lib/audio-engine';
import { VisualEngine, SCENES } from '@/src/lib/scenes';
import type { AudioFeatures } from '@/src/lib/audio-features';

type Status = 'idle' | 'requesting' | 'running' | 'error';

interface AudioDevice {
  deviceId: string;
  label: string;
}

export function VJWaveform() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioEngineRef = useRef<AudioEngine | null>(null);
  const visualEngineRef = useRef<VisualEngine | null>(null);

  const [status, setStatus] = useState<Status>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [currentSceneId, setCurrentSceneId] = useState<string>(SCENES[0].id);

  // Debug panel state - throttled to 10fps to avoid React overhead
  const [debugFeatures, setDebugFeatures] = useState<AudioFeatures | null>(null);
  const [showDebug, setShowDebug] = useState<boolean>(true);

  // Fetch available audio input devices
  const fetchDevices = useCallback(async () => {
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = allDevices
        .filter((d) => d.kind === 'audioinput')
        .map((d) => ({
          deviceId: d.deviceId,
          label: d.label || `Microphone ${d.deviceId.slice(0, 8)}`,
        }));
      setDevices(audioInputs);
    } catch {
      console.error('Failed to enumerate devices');
    }
  }, []);

  // Initialize audio and start rendering
  const initAudio = useCallback(async (deviceId?: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Cleanup previous instances
    if (visualEngineRef.current) {
      visualEngineRef.current.stop();
      visualEngineRef.current = null;
    }
    if (audioEngineRef.current) {
      audioEngineRef.current.destroy();
      audioEngineRef.current = null;
    }

    setStatus('requesting');
    setErrorMessage('');

    try {
      // Create and initialize audio engine
      const audioEngine = new AudioEngine();
      await audioEngine.init(deviceId);
      audioEngineRef.current = audioEngine;

      // Create visual engine with all scenes
      const visualEngine = new VisualEngine(canvas, audioEngine, SCENES);
      visualEngineRef.current = visualEngine;

      // Start render loop
      visualEngine.start();

      setStatus('running');
      setCurrentSceneId(visualEngine.getCurrentScene().id);

      // Refresh device list (labels become available after permission)
      fetchDevices();
    } catch (err) {
      setStatus('error');
      setErrorMessage(err instanceof Error ? err.message : 'Failed to initialize audio');
    }
  }, [fetchDevices]);

  // Handle device selection change
  const handleDeviceChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const deviceId = e.target.value;
    setSelectedDeviceId(deviceId);
    initAudio(deviceId || undefined);
  }, [initAudio]);

  // Handle scene selection change
  const handleSceneChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const sceneId = e.target.value;
    const engine = visualEngineRef.current;
    if (engine && engine.setSceneById(sceneId)) {
      setCurrentSceneId(sceneId);
    }
  }, []);

  // Debug features polling - throttled to ~10fps to minimize React re-renders
  useEffect(() => {
    if (!showDebug) return;

    const interval = setInterval(() => {
      const engine = audioEngineRef.current;
      if (engine) {
        const features = engine.getLatestFeatures();
        setDebugFeatures(features);
      }
    }, 100); // 10fps

    return () => clearInterval(interval);
  }, [showDebug]);

  // Initial setup
  useEffect(() => {
    fetchDevices();
    initAudio();

    return () => {
      // Cleanup on unmount
      if (visualEngineRef.current) {
        visualEngineRef.current.stop();
      }
      if (audioEngineRef.current) {
        audioEngineRef.current.destroy();
      }
    };
  }, [fetchDevices, initAudio]);

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
              status === 'running'
                ? 'text-emerald-400'
                : status === 'error'
                ? 'text-red-400'
                : 'text-amber-400'
            }`}
          >
            {status === 'idle' && 'Initializing'}
            {status === 'requesting' && 'Requesting audio permission'}
            {status === 'running' && 'Running'}
            {status === 'error' && 'Error'}
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
      {status === 'error' && errorMessage && (
        <div className="text-red-400 text-sm font-mono bg-red-950/30 border border-red-900 rounded px-3 py-2">
          {errorMessage}
        </div>
      )}

      {/* Canvas container */}
      <div className="relative w-full aspect-[4/1] bg-neutral-950 rounded-lg overflow-hidden border border-neutral-800">
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ imageRendering: 'pixelated' }}
        />
      </div>

      {/* Debug toggle */}
      <button
        onClick={() => setShowDebug(!showDebug)}
        className="self-start text-xs font-mono text-neutral-500 hover:text-neutral-300 transition-colors"
      >
        {showDebug ? 'Hide' : 'Show'} Audio Features Debug
      </button>

      {/* Debug panel - AudioFeatures display */}
      {showDebug && (
        <div className="bg-neutral-900/80 border border-neutral-700 rounded-lg p-4 font-mono text-xs">
          <div className="text-neutral-400 uppercase tracking-wide mb-3">Audio Features</div>
          {debugFeatures ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <FeatureBar label="RMS" value={debugFeatures.rms} />
              <FeatureBar label="Peak" value={debugFeatures.peak} />
              <FeatureBar label="Low" value={debugFeatures.energyLow} color="text-red-400" barColor="bg-red-500" />
              <FeatureBar label="Mid" value={debugFeatures.energyMid} color="text-yellow-400" barColor="bg-yellow-500" />
              <FeatureBar label="High" value={debugFeatures.energyHigh} color="text-cyan-400" barColor="bg-cyan-500" />
              <FeatureBar label="Centroid" value={debugFeatures.spectralCentroid} color="text-purple-400" barColor="bg-purple-500" />
            </div>
          ) : (
            <div className="text-neutral-500">Waiting for audio data...</div>
          )}
        </div>
      )}

      {/* Footer hint */}
      <p className="text-xs text-neutral-500 font-mono">
        vj0 - Audio-reactive visuals. Select a scene and connect a USB audio interface for best results.
      </p>
    </div>
  );
}

/**
 * Individual feature bar component for debug display.
 * Shows label, numeric value, and visual bar.
 */
function FeatureBar({
  label,
  value,
  color = 'text-emerald-400',
  barColor = 'bg-emerald-500',
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
