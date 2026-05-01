/**
 * AudioEngine - Framework-agnostic Web Audio API wrapper
 *
 * Handles audio input capture, time-domain analysis, and AudioWorklet-based
 * feature extraction. Uses AnalyserNode for waveform data and a custom
 * AudioWorklet for real-time audio features.
 *
 * Extension points:
 * - Replace message-based worklet communication with SharedArrayBuffer for
 *   zero-copy transfer. Requires COEP/COOP headers and Atomics for sync.
 * - Add more features to AudioFeatures type (beat detection, tempo, etc.)
 * - Visuals/DMX can consume features via getLatestFeatures() in rAF loop
 */

import type { AudioFeatures } from './audio-features';

export class AudioEngine {
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private analyserNode: AnalyserNode | null = null;

  // AudioWorklet nodes
  private workletNode: AudioWorkletNode | null = null;
  private silentGainNode: GainNode | null = null;

  // Latest features received from worklet
  private latestFeatures: AudioFeatures | null = null;

  /**
   * Initialize audio capture from a microphone-style device or from a
   * pre-existing MediaStream (e.g. a `getDisplayMedia` capture for system
   * audio). Sets up both AnalyserNode (waveform) and AudioWorklet (features).
   *
   * @param source - Either an audio input device ID from enumerateDevices(),
   *                 or a MediaStream that already carries an audio track.
   *                 When omitted, the browser picks the default mic.
   *                 When a MediaStream is given the engine takes ownership of
   *                 it — destroy() will stop its tracks like any other.
   */
  async init(source?: string | MediaStream): Promise<void> {
    if (source instanceof MediaStream) {
      // Caller already got the stream (e.g. system audio via getDisplayMedia).
      // Just keep a reference; the audio graph below works on any stream that
      // has at least one audio track.
      this.mediaStream = source;
    } else {
      // Request microphone access with processing disabled for raw signal
      const constraints: MediaStreamConstraints = {
        audio: {
          deviceId: source ? { exact: source } : undefined,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      };
      this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    }

    // Create AudioContext with low latency hint for real-time processing
    this.audioContext = new AudioContext({
      latencyHint: 'interactive',
    });

    // Load the AudioWorklet module
    await this.audioContext.audioWorklet.addModule('/audio-worklet/vj0-audio-processor.js');

    // Create source node from media stream
    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

    // AnalyserNode for waveform (existing functionality)
    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 2048;

    // AudioWorkletNode for feature extraction
    this.workletNode = new AudioWorkletNode(this.audioContext, 'vj0-audio-processor');

    // Silent gain node to keep the graph valid without feedback
    // Connect worklet output here instead of destination to avoid playback
    this.silentGainNode = this.audioContext.createGain();
    this.silentGainNode.gain.value = 0;

    // Connect the audio graph:
    // source -> analyser (for waveform)
    // source -> worklet -> silentGain -> destination (for features, no audio output)
    this.sourceNode.connect(this.analyserNode);
    this.sourceNode.connect(this.workletNode);
    this.workletNode.connect(this.silentGainNode);
    this.silentGainNode.connect(this.audioContext.destination);

    // Listen for features from the worklet
    this.workletNode.port.onmessage = (event: MessageEvent<AudioFeatures>) => {
      this.latestFeatures = event.data;
    };
  }

  /**
   * Fill the provided buffer with time-domain audio samples.
   * Values are normalized floats in range [-1, 1].
   *
   * @param target - Pre-allocated Float32Array to fill (must match bufferSize)
   */
  getTimeDomainData(target: Float32Array): void {
    if (!this.analyserNode) return;
    // Newer lib.dom narrows getFloatTimeDomainData to Float32Array<ArrayBuffer>
    // (excluding SharedArrayBuffer-backed arrays). The buffer here is a
    // plain ArrayBuffer at runtime — assert that to TS so the call type-checks.
    this.analyserNode.getFloatTimeDomainData(
      target as Float32Array<ArrayBuffer>
    );
  }

  /**
   * Returns the size of the time-domain buffer (equals fftSize).
   */
  get bufferSize(): number {
    return this.analyserNode?.fftSize ?? 2048;
  }

  /**
   * Get the most recently computed audio features from the worklet.
   *
   * Returns null if:
   * - Audio engine not initialized
   * - No features received yet from worklet
   *
   * Usage: Poll this in requestAnimationFrame for visuals/DMX mapping.
   * Do NOT put directly into React state at 60fps - use refs or throttling.
   */
  getLatestFeatures(): AudioFeatures | null {
    return this.latestFeatures;
  }

  /**
   * Build a fresh tap on the input audio for downstream consumers (e.g. the
   * RecordingEngine feeding MediaRecorder).
   *
   * Adds a `MediaStreamAudioDestinationNode` as an extra sink on the existing
   * `sourceNode` — this is a parallel branch, so the analyser/worklet paths
   * are unaffected and the audio graph keeps its zero-allocation hot loop.
   * Returns a handle with the live `MediaStream` and a `disconnect` function;
   * calling `disconnect` removes only this branch and stops the stream tracks.
   *
   * Returns null if the engine isn't initialised. Multiple taps are allowed
   * (each is independent), but the typical case is one tap per recording.
   *
   * The destination node is owned by the AudioContext, so it is also torn
   * down automatically when `destroy()` closes the context — callers don't
   * have to disconnect on engine teardown, but they SHOULD when they're
   * just stopping their own work, to avoid accumulating sinks across repeated
   * start/stop cycles.
   */
  createAudioTap(): { stream: MediaStream; disconnect: () => void } | null {
    if (!this.audioContext || !this.sourceNode) return null;
    const destination = this.audioContext.createMediaStreamDestination();
    const source = this.sourceNode;
    source.connect(destination);
    return {
      stream: destination.stream,
      disconnect: () => {
        try {
          source.disconnect(destination);
        } catch {
          // Either we're already disconnected or the context closed under us.
          // Both are fine — the destination becomes unreachable either way.
        }
        destination.stream.getTracks().forEach((t) => t.stop());
      },
    };
  }

  /**
   * Clean up all audio resources.
   * Call this when unmounting the component or switching devices.
   */
  destroy(): void {
    // Stop all media tracks
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    // Disconnect and clean up worklet
    if (this.workletNode) {
      this.workletNode.port.onmessage = null;
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    // Disconnect silent gain
    if (this.silentGainNode) {
      this.silentGainNode.disconnect();
      this.silentGainNode = null;
    }

    // Disconnect source and analyser
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    this.analyserNode = null;
    this.latestFeatures = null;

    // Close audio context
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}
