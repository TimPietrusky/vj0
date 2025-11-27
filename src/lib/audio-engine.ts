/**
 * AudioEngine - Framework-agnostic Web Audio API wrapper
 *
 * Handles audio input capture and time-domain analysis.
 * Uses AnalyserNode for now - designed to be replaceable with AudioWorklet later.
 *
 * Extension point: Replace AnalyserNode with AudioWorklet for advanced
 * frequency analysis or custom DSP processing. The public API (init,
 * getTimeDomainData, destroy) should remain stable.
 */

export class AudioEngine {
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private analyserNode: AnalyserNode | null = null;

  /**
   * Initialize audio capture with optional device selection.
   *
   * @param deviceId - Optional audio input device ID from enumerateDevices()
   */
  async init(deviceId?: string): Promise<void> {
    // Request microphone access with processing disabled for raw signal
    const constraints: MediaStreamConstraints = {
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    };

    this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

    // Create AudioContext with low latency hint for real-time processing
    this.audioContext = new AudioContext({
      latencyHint: 'interactive',
    });

    // Connect source -> analyser
    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.analyserNode = this.audioContext.createAnalyser();

    // Configure analyser for time-domain data
    // 2048 gives good resolution without excessive CPU usage
    this.analyserNode.fftSize = 2048;

    this.sourceNode.connect(this.analyserNode);
    // Note: Not connecting to destination - we only analyze, don't play back
  }

  /**
   * Fill the provided buffer with time-domain audio samples.
   * Values are normalized floats in range [-1, 1].
   *
   * @param target - Pre-allocated Float32Array to fill (must match bufferSize)
   */
  getTimeDomainData(target: Float32Array): void {
    if (!this.analyserNode) return;
    this.analyserNode.getFloatTimeDomainData(target);
  }

  /**
   * Returns the size of the time-domain buffer (equals fftSize).
   */
  get bufferSize(): number {
    return this.analyserNode?.fftSize ?? 2048;
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

    // Disconnect nodes
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    this.analyserNode = null;

    // Close audio context
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

