/**
 * vj0 Audio Processor - AudioWorklet for real-time audio feature extraction
 *
 * Runs on the audio rendering thread for stable, low-latency analysis.
 * Communicates with main thread via MessagePort (postMessage).
 *
 * Future improvements:
 * - Switch to SharedArrayBuffer for zero-copy transfer (requires cross-origin isolation)
 * - Integrate Meyda for advanced features (MFCC, chroma, etc.)
 * - Add beat detection with onset/tempo estimation
 *
 * To use SharedArrayBuffer later:
 * 1. Enable COEP/COOP headers in next.config.ts
 * 2. Replace postMessage with direct SAB writes
 * 3. Use Atomics.notify() for synchronization
 */

class VJ0AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // FFT size for spectrum analysis (must be power of 2)
    this.fftSize = 2048;

    // Pre-allocated arrays to avoid allocations in hot path
    this.timeDomainBuffer = new Float32Array(this.fftSize);
    this.frequencyBuffer = new Float32Array(this.fftSize / 2);
    this.bufferIndex = 0;

    // Simple sliding window for FFT input
    this.windowBuffer = new Float32Array(this.fftSize);

    // Hamming window coefficients (pre-computed)
    this.hammingWindow = new Float32Array(this.fftSize);
    for (let i = 0; i < this.fftSize; i++) {
      this.hammingWindow[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (this.fftSize - 1));
    }

    // FFT buffers (real and imaginary parts)
    this.fftReal = new Float32Array(this.fftSize);
    this.fftImag = new Float32Array(this.fftSize);

    // Sample rate from audio context (set in first process call)
    this.actualSampleRate = 44100;
  }

  /**
   * Simple in-place FFT (Cooley-Tukey radix-2 DIT)
   * Not the fastest, but avoids external dependencies
   */
  computeFFT(real, imag) {
    const n = real.length;
    const levels = Math.log2(n);

    // Bit-reversal permutation
    for (let i = 0; i < n; i++) {
      let j = 0;
      for (let k = 0; k < levels; k++) {
        j = (j << 1) | ((i >> k) & 1);
      }
      if (j > i) {
        [real[i], real[j]] = [real[j], real[i]];
        [imag[i], imag[j]] = [imag[j], imag[i]];
      }
    }

    // Cooley-Tukey FFT
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;
      const angleStep = (-2 * Math.PI) / size;

      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < halfSize; j++) {
          const angle = angleStep * j;
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);

          const evenIdx = i + j;
          const oddIdx = i + j + halfSize;

          const tr = real[oddIdx] * cos - imag[oddIdx] * sin;
          const ti = real[oddIdx] * sin + imag[oddIdx] * cos;

          real[oddIdx] = real[evenIdx] - tr;
          imag[oddIdx] = imag[evenIdx] - ti;
          real[evenIdx] = real[evenIdx] + tr;
          imag[evenIdx] = imag[evenIdx] + ti;
        }
      }
    }
  }

  /**
   * Compute magnitude spectrum from FFT output
   */
  computeMagnitudeSpectrum() {
    const halfSize = this.fftSize / 2;
    for (let i = 0; i < halfSize; i++) {
      const re = this.fftReal[i];
      const im = this.fftImag[i];
      this.frequencyBuffer[i] = Math.sqrt(re * re + im * im) / this.fftSize;
    }
  }

  /**
   * Get frequency bin index for a given frequency
   */
  freqToBin(freq) {
    return Math.round((freq * this.fftSize) / this.actualSampleRate);
  }

  /**
   * Compute energy in a frequency band (sum of squared magnitudes)
   */
  computeBandEnergy(lowFreq, highFreq) {
    const lowBin = Math.max(0, this.freqToBin(lowFreq));
    const highBin = Math.min(this.frequencyBuffer.length - 1, this.freqToBin(highFreq));

    let energy = 0;
    for (let i = lowBin; i <= highBin; i++) {
      energy += this.frequencyBuffer[i] * this.frequencyBuffer[i];
    }
    return energy;
  }

  /**
   * Compute spectral centroid (center of mass of spectrum)
   */
  computeSpectralCentroid() {
    let weightedSum = 0;
    let magnitudeSum = 0;

    for (let i = 0; i < this.frequencyBuffer.length; i++) {
      const freq = (i * this.actualSampleRate) / this.fftSize;
      const mag = this.frequencyBuffer[i];
      weightedSum += freq * mag;
      magnitudeSum += mag;
    }

    if (magnitudeSum === 0) return 0;

    const centroid = weightedSum / magnitudeSum;
    // Normalize to 0-1 range (assuming max ~10kHz for typical content)
    return Math.min(1, centroid / 10000);
  }

  process(inputs, outputs, parameters) {
    // Get first channel of first input
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const samples = input[0];
    this.actualSampleRate = sampleRate; // Global from AudioWorklet scope

    // Accumulate samples into window buffer
    for (let i = 0; i < samples.length; i++) {
      this.windowBuffer[this.bufferIndex] = samples[i];
      this.bufferIndex++;

      // When buffer is full, compute features and send
      if (this.bufferIndex >= this.fftSize) {
        this.analyzeAndSend();
        this.bufferIndex = 0;
      }
    }

    return true;
  }

  analyzeAndSend() {
    // Compute RMS and peak from time domain
    let sumSquares = 0;
    let peak = 0;

    for (let i = 0; i < this.fftSize; i++) {
      const sample = this.windowBuffer[i];
      sumSquares += sample * sample;
      const absSample = Math.abs(sample);
      if (absSample > peak) peak = absSample;
    }

    const rms = Math.sqrt(sumSquares / this.fftSize);

    // Apply Hamming window and prepare for FFT
    for (let i = 0; i < this.fftSize; i++) {
      this.fftReal[i] = this.windowBuffer[i] * this.hammingWindow[i];
      this.fftImag[i] = 0;
    }

    // Compute FFT
    this.computeFFT(this.fftReal, this.fftImag);

    // Get magnitude spectrum
    this.computeMagnitudeSpectrum();

    // Compute band energies
    const energyLowRaw = this.computeBandEnergy(20, 250);
    const energyMidRaw = this.computeBandEnergy(250, 4000);
    const energyHighRaw = this.computeBandEnergy(4000, 20000);

    // Normalize energies (using sqrt for perceptual scaling, clamp to 0-1)
    // Scale factors tuned empirically for typical music signals
    const energyLow = Math.min(1, Math.sqrt(energyLowRaw) * 8);
    const energyMid = Math.min(1, Math.sqrt(energyMidRaw) * 12);
    const energyHigh = Math.min(1, Math.sqrt(energyHighRaw) * 20);

    // Spectral centroid
    const spectralCentroid = this.computeSpectralCentroid();

    // Pack features and send to main thread
    const features = {
      rms: Math.min(1, rms * 3), // Scale up for visibility, clamp to 1
      peak: Math.min(1, peak),
      energyLow,
      energyMid,
      energyHigh,
      spectralCentroid,
    };

    this.port.postMessage(features);
  }
}

registerProcessor('vj0-audio-processor', VJ0AudioProcessor);

