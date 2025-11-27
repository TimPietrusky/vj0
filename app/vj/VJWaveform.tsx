'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { AudioEngine } from '@/src/lib/audio-engine';
import { WaveformRenderer } from '@/src/lib/waveform-renderer';

type Status = 'idle' | 'requesting' | 'running' | 'error';

interface AudioDevice {
  deviceId: string;
  label: string;
}

export function VJWaveform() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<AudioEngine | null>(null);
  const rendererRef = useRef<WaveformRenderer | null>(null);
  const bufferRef = useRef<Float32Array | null>(null);

  const [status, setStatus] = useState<Status>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

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
    if (rendererRef.current) {
      rendererRef.current.stop();
      rendererRef.current = null;
    }
    if (engineRef.current) {
      engineRef.current.destroy();
      engineRef.current = null;
    }

    setStatus('requesting');
    setErrorMessage('');

    try {
      // Create and initialize audio engine
      const engine = new AudioEngine();
      await engine.init(deviceId);
      engineRef.current = engine;

      // Create renderer
      const renderer = new WaveformRenderer(canvas);
      rendererRef.current = renderer;

      // Allocate buffer once (reused every frame)
      const buffer = new Float32Array(engine.bufferSize);
      bufferRef.current = buffer;

      // Start render loop - callback fills buffer with audio data
      renderer.start((buf) => {
        engine.getTimeDomainData(buf);
      }, buffer);

      setStatus('running');

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

  // Initial setup
  useEffect(() => {
    fetchDevices();
    initAudio();

    return () => {
      // Cleanup on unmount
      if (rendererRef.current) {
        rendererRef.current.stop();
      }
      if (engineRef.current) {
        engineRef.current.destroy();
      }
    };
  }, [fetchDevices, initAudio]);

  return (
    <div className="flex flex-col gap-4 w-full max-w-4xl mx-auto p-4">
      {/* Header with status and device selector */}
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

      {/* Footer hint */}
      <p className="text-xs text-neutral-500 font-mono">
        vj0 - Audio waveform visualization. Connect a USB audio interface for best results.
      </p>
    </div>
  );
}

