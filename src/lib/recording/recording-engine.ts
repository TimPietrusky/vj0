/**
 * RecordingEngine - capture canvas + live audio into a downloadable video.
 *
 * Uses native browser APIs only:
 *   - HTMLCanvasElement.captureStream(fps) for video
 *   - AudioEngine.createAudioTap() → MediaStream for audio
 *   - MediaRecorder for the encode + container muxing
 *
 * MediaRecorder runs the encoder off the JS main thread and the platform's
 * hardware video encoder (VideoToolbox on macOS, NVENC / Media Foundation on
 * Windows) when available. The render loop sees ~zero overhead — there's no
 * per-frame allocation, no readback, no copy on the JS side.
 *
 * Format: probes for MP4/H.264 first (Chrome ≥ 126, Safari) and falls back
 * to WebM/VP9 → WebM/VP8. The chosen container is reflected in the result so
 * the caller can name the download file correctly.
 *
 * Lifecycle:
 *   const rec = new RecordingEngine({
 *     getCanvas: () => previewRendererRef.current?.getCanvas() ?? null,
 *     createAudioTap: () => audioEngineRef.current?.createAudioTap() ?? null,
 *   });
 *   rec.start();                      // capture begins
 *   const result = await rec.stop();  // returns Blob + metadata
 *   triggerDownload(result);          // up to caller (UI)
 *
 * Notes:
 * - The captured canvas can change content shape any time — captureStream
 *   adapts to the new canvas dimensions on the next frame, so resolution
 *   switches mid-recording don't break the encode (the resulting video just
 *   contains frames at varying resolutions, which all sane players handle).
 * - getElapsedMs() is cheap to poll for a UI timer (uses performance.now()).
 *   Drive it with a 250 ms timer in the React layer; do NOT put it in
 *   requestAnimationFrame React state.
 */

export interface RecordingResult {
  /** Encoded video. MIME type matches `mimeType` below. */
  blob: Blob;
  /** Container + codec MIME, e.g. "video/mp4;codecs=avc1,mp4a.40.2". */
  mimeType: string;
  /** File extension matching the container — "mp4" or "webm". */
  extension: string;
  /** Wall-clock duration captured, in milliseconds. */
  durationMs: number;
  /** Final blob size in bytes. */
  sizeBytes: number;
}

export interface RecordingEngineDeps {
  /**
   * Returns the canvas to capture frames from at start() time. Lazy because
   * the canvas may not exist when the engine is constructed (e.g. the AI
   * preview WebGL canvas mounts after VJApp's first paint).
   */
  getCanvas: () => HTMLCanvasElement | null;
  /**
   * Returns a fresh audio tap for the duration of the recording. Called
   * exactly once per `start()`. The tap's `disconnect()` is invoked when the
   * recording stops to release the destination node back to the GC.
   *
   * May return null if there's no audio source available — the recording
   * proceeds video-only in that case.
   */
  createAudioTap: () => { stream: MediaStream; disconnect: () => void } | null;
}

export interface RecordingStartOptions {
  /**
   * Target capture frame rate. Defaults to 30 — gives a video frame even
   * when the canvas hasn't been drawn (encoder duplicates the last frame),
   * which matters for the AI preview canvas that only paints on AI frame
   * arrival. Pass 60 for 60 fps captures of a constantly-animating canvas.
   *
   * The same value also throttles the offscreen-canvas copy loop so we
   * don't waste GPU bandwidth drawing frames the encoder will discard.
   */
  frameRate?: number;
  /** Video bitrate hint. Default 8 Mbit/s — clean 1080p without bloating files. */
  videoBitsPerSecond?: number;
  /** Audio bitrate hint. Default 192 kbit/s. Ignored when no audio tap. */
  audioBitsPerSecond?: number;
  /**
   * MediaRecorder timeslice in ms. The recorder fires `dataavailable` at
   * this interval — keeps memory growth steady and gives us a flush point
   * if the page crashes mid-recording. Default 1000 ms.
   */
  timesliceMs?: number;
  /**
   * Output video dimensions. Default 1920×1080 (Full HD, 16:9). The source
   * canvas is composited onto this with object-fit:contain math, so any
   * aspect-ratio mismatch is filled with black bars rather than stretching
   * the picture. Pick {1080, 1920} for vertical (TikTok / Reels), or
   * {1080, 1080} for square (Instagram). The output dimensions are the
   * EXACT pixel size of the produced video, regardless of how big the
   * source canvas is or what its aspect ratio is.
   */
  outputWidth?: number;
  outputHeight?: number;
}

/**
 * Probe order: try MP4 (best compatibility for sharing) before falling back
 * to WebM. Each entry pairs the MIME tested against `MediaRecorder.isTypeSupported`
 * with the file extension we'd give the resulting blob.
 */
const PREFERRED_MIME_TYPES: ReadonlyArray<{ mimeType: string; extension: string }> = [
  { mimeType: "video/mp4;codecs=avc1.42E01E,mp4a.40.2", extension: "mp4" }, // H.264 baseline + AAC-LC
  { mimeType: "video/mp4;codecs=avc1,mp4a", extension: "mp4" },
  { mimeType: "video/mp4", extension: "mp4" },
  { mimeType: "video/webm;codecs=vp9,opus", extension: "webm" },
  { mimeType: "video/webm;codecs=vp8,opus", extension: "webm" },
  { mimeType: "video/webm", extension: "webm" },
];

function pickMimeType(): { mimeType: string; extension: string } | null {
  if (typeof MediaRecorder === "undefined") return null;
  for (const candidate of PREFERRED_MIME_TYPES) {
    try {
      if (MediaRecorder.isTypeSupported(candidate.mimeType)) return candidate;
    } catch {
      // Some old browsers throw on unknown types instead of returning false.
      // Treat that as "not supported" and keep probing.
    }
  }
  return null;
}

export class RecordingEngine {
  private deps: RecordingEngineDeps;

  // Live recording state. All null when idle.
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private chosenMime: { mimeType: string; extension: string } | null = null;
  private startTimeMs = 0;
  private audioTap: { stream: MediaStream; disconnect: () => void } | null = null;
  private videoStream: MediaStream | null = null;

  // Offscreen compositing surface. We render into THIS canvas at the user's
  // chosen output resolution (default 1920×1080) and capture from it, so the
  // produced video has clean, predictable pixel dimensions regardless of how
  // big the source canvas's internal buffer happens to be. The source canvas
  // is composited with object-fit:contain math — aspect preserved, black
  // letterbox bars filling any difference.
  private outputCanvas: HTMLCanvasElement | null = null;
  private outputCtx: CanvasRenderingContext2D | null = null;
  private sourceCanvas: HTMLCanvasElement | null = null;
  private rafId: number | null = null;
  private lastDrawAtMs = 0;
  private targetFrameInterval = 1000 / 30;

  // stop() may be called multiple times; the second call should resolve to
  // the same result as the first instead of throwing.
  private stopPromise: Promise<RecordingResult> | null = null;
  private resolveStop: ((result: RecordingResult) => void) | null = null;
  private rejectStop: ((err: Error) => void) | null = null;

  constructor(deps: RecordingEngineDeps) {
    this.deps = deps;
  }

  /** Quick capability probe. Use to gate the UI button's `disabled` state. */
  static isSupported(): boolean {
    return typeof MediaRecorder !== "undefined" && pickMimeType() !== null;
  }

  isRecording(): boolean {
    return this.recorder !== null && this.recorder.state === "recording";
  }

  /** Wall-clock ms since start(). Returns 0 when idle. Cheap to poll. */
  getElapsedMs(): number {
    if (!this.startTimeMs) return 0;
    return performance.now() - this.startTimeMs;
  }

  /** Format chosen for the in-flight recording (or null when idle). */
  getMimeType(): string | null {
    return this.chosenMime?.mimeType ?? null;
  }

  /**
   * Begin recording. Throws synchronously if:
   *   - Already recording
   *   - Canvas is unavailable
   *   - No supported MediaRecorder MIME type
   *
   * Audio is best-effort: if `createAudioTap()` returns null we record
   * video only.
   */
  start(options: RecordingStartOptions = {}): void {
    if (this.recorder) {
      throw new Error("RecordingEngine: already recording");
    }
    const canvas = this.deps.getCanvas();
    if (!canvas) {
      throw new Error("RecordingEngine: canvas unavailable");
    }
    const mime = pickMimeType();
    if (!mime) {
      throw new Error(
        "RecordingEngine: this browser doesn't support MediaRecorder for any of the formats we tried (mp4, webm)"
      );
    }

    const frameRate = options.frameRate ?? 30;
    const outputWidth = Math.max(2, Math.round(options.outputWidth ?? 1920));
    const outputHeight = Math.max(2, Math.round(options.outputHeight ?? 1080));

    // Build the offscreen compositing canvas. NOT appended to the DOM —
    // it lives only as long as the recording does. `alpha: false` lets the
    // 2D context drop blending math; the letterbox bars are opaque black
    // anyway so we never need transparency.
    const outputCanvas = document.createElement("canvas");
    outputCanvas.width = outputWidth;
    outputCanvas.height = outputHeight;
    const outputCtx = outputCanvas.getContext("2d", { alpha: false });
    if (!outputCtx) {
      throw new Error("RecordingEngine: failed to acquire 2D context for output canvas");
    }
    // Pre-fill black so the very first captured frame doesn't flash white
    // (some browsers initialise canvases to transparent-white).
    outputCtx.fillStyle = "#000";
    outputCtx.fillRect(0, 0, outputWidth, outputHeight);

    this.outputCanvas = outputCanvas;
    this.outputCtx = outputCtx;
    this.sourceCanvas = canvas;
    this.targetFrameInterval = 1000 / frameRate;
    this.lastDrawAtMs = 0;

    // Capture from the OUTPUT canvas, not the source. Same captureStream
    // mechanics, but the produced track now has the dimensions we picked
    // (e.g. 1920×1080) instead of the WebGL canvas's internal pixel grid.
    const captureStream = (
      outputCanvas as HTMLCanvasElement & {
        captureStream: (frameRequestRate?: number) => MediaStream;
      }
    ).captureStream(frameRate);
    this.videoStream = captureStream;

    const tap = this.deps.createAudioTap();
    this.audioTap = tap;

    // Combine video + audio tracks. We build a fresh MediaStream rather than
    // mutating the captureStream so that adding/removing audio doesn't affect
    // the canvas track lifecycle.
    const tracks: MediaStreamTrack[] = [...captureStream.getVideoTracks()];
    if (tap) tracks.push(...tap.stream.getAudioTracks());
    const combined = new MediaStream(tracks);

    const recorderOpts: MediaRecorderOptions = {
      mimeType: mime.mimeType,
      videoBitsPerSecond: options.videoBitsPerSecond ?? 8_000_000,
    };
    if (tap) {
      recorderOpts.audioBitsPerSecond = options.audioBitsPerSecond ?? 192_000;
    }

    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(combined, recorderOpts);
    } catch (err) {
      // Bitrate combos sometimes get rejected on Firefox — retry with the
      // bare-minimum config (mime only) before giving up.
      try {
        recorder = new MediaRecorder(combined, { mimeType: mime.mimeType });
      } catch {
        // Cleanup partial state so the next start() can try again. We have
        // to drop the offscreen canvas + audio tap too — they were allocated
        // before MediaRecorder construction and won't be released by the
        // normal `cleanupAfterStop` path because no recorder exists yet.
        captureStream.getTracks().forEach((t) => t.stop());
        tap?.disconnect();
        this.videoStream = null;
        this.audioTap = null;
        this.outputCanvas = null;
        this.outputCtx = null;
        this.sourceCanvas = null;
        throw err instanceof Error
          ? err
          : new Error("RecordingEngine: MediaRecorder rejected the configured options");
      }
    }

    this.recorder = recorder;
    this.chosenMime = mime;
    this.chunks = [];
    this.startTimeMs = performance.now();

    recorder.addEventListener("dataavailable", (event: BlobEvent) => {
      if (event.data && event.data.size > 0) {
        this.chunks.push(event.data);
      }
    });

    recorder.addEventListener("stop", () => {
      const blob = new Blob(this.chunks, { type: mime.mimeType });
      const durationMs = this.startTimeMs
        ? performance.now() - this.startTimeMs
        : 0;
      const result: RecordingResult = {
        blob,
        mimeType: mime.mimeType,
        extension: mime.extension,
        durationMs,
        sizeBytes: blob.size,
      };
      this.cleanupAfterStop();
      this.resolveStop?.(result);
    });

    recorder.addEventListener("error", (event: Event) => {
      // MediaRecorder.onerror payload is browser-specific; fish out a useful
      // message from whichever shape we got, falling back to a generic.
      const message =
        (event as ErrorEvent).message ||
        (event as unknown as { error?: { name?: string; message?: string } })
          .error?.message ||
        "unknown MediaRecorder error";
      const err = new Error(`RecordingEngine: ${message}`);
      this.cleanupAfterStop();
      this.rejectStop?.(err);
    });

    // Emit chunks every second by default so a long recording doesn't sit
    // entirely in one giant Blob until stop().
    recorder.start(options.timesliceMs ?? 1000);

    // Kick off the source → output copy loop. We schedule on rAF so we ride
    // alongside any ongoing render; the actual draw is throttled to
    // `frameRate` so we don't waste GPU bandwidth at 120 Hz when the encoder
    // only takes 30 frames per second.
    this.scheduleNextCopy();
  }

  /**
   * Composite the current source-canvas image into the output canvas with
   * object-fit:contain math. Picks the largest scale that fits the source
   * inside the output without cropping; any leftover space is filled with
   * black, producing letterbox bars for portrait content in a landscape
   * recording (and vice-versa).
   *
   * Called from a throttled rAF loop while recording. Reads from a WebGL
   * canvas works as long as the WebGL canvas has been composited at least
   * once — modern browsers grab the composited image rather than the (now
   * cleared) backbuffer, even with `preserveDrawingBuffer: false`.
   */
  private composite(): void {
    const ctx = this.outputCtx;
    const out = this.outputCanvas;
    const src = this.sourceCanvas;
    if (!ctx || !out || !src) return;

    // Source can be 0×0 briefly during init or after a context loss; skip.
    if (src.width <= 0 || src.height <= 0) return;

    // Black background for letterbox bars. Done with fillRect rather than
    // clearRect so the canvas stays opaque (alpha:false context).
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, out.width, out.height);

    // object-fit: contain — biggest box that fits inside (out.width × out.height)
    // while preserving source aspect.
    const srcRatio = src.width / src.height;
    const outRatio = out.width / out.height;
    let drawW: number;
    let drawH: number;
    if (srcRatio >= outRatio) {
      // Source wider than (or matches) output → fit to width, bars top/bottom.
      drawW = out.width;
      drawH = out.width / srcRatio;
    } else {
      // Source taller → fit to height, bars left/right.
      drawH = out.height;
      drawW = out.height * srcRatio;
    }
    const drawX = (out.width - drawW) / 2;
    const drawY = (out.height - drawH) / 2;

    try {
      ctx.drawImage(src, drawX, drawY, drawW, drawH);
    } catch {
      // drawImage can throw if the source canvas was tainted or its context
      // was lost. Skip this frame; the next tick will try again.
    }
  }

  private scheduleNextCopy(): void {
    this.rafId = requestAnimationFrame((nowMs: number) => {
      // Loop terminates the moment the recorder is gone (stop / cancel /
      // error). cleanupAfterStop also explicitly cancels the rAF, this
      // check is just a belt-and-braces guard.
      if (!this.recorder) return;
      if (nowMs - this.lastDrawAtMs >= this.targetFrameInterval) {
        this.composite();
        this.lastDrawAtMs = nowMs;
      }
      this.scheduleNextCopy();
    });
  }

  /**
   * Stop recording and return the encoded blob. Safe to call when not
   * recording (rejects with a clear error). Calling twice returns the same
   * promise as the first call.
   */
  stop(): Promise<RecordingResult> {
    if (!this.recorder) {
      return Promise.reject(new Error("RecordingEngine: not recording"));
    }
    if (this.stopPromise) return this.stopPromise;
    this.stopPromise = new Promise<RecordingResult>((resolve, reject) => {
      this.resolveStop = resolve;
      this.rejectStop = reject;
    });
    if (this.recorder.state !== "inactive") {
      try {
        this.recorder.stop();
      } catch (err) {
        this.cleanupAfterStop();
        this.rejectStop?.(
          err instanceof Error
            ? err
            : new Error("RecordingEngine: failed to stop")
        );
      }
    }
    return this.stopPromise;
  }

  /**
   * Tear down without producing a result. Used when the host needs to
   * reinitialise the audio graph mid-recording (e.g. device switch) and
   * doesn't want a half-finished blob handed back to the user.
   */
  cancel(): void {
    if (!this.recorder) return;
    try {
      if (this.recorder.state !== "inactive") this.recorder.stop();
    } catch {
      // ignore — we're tearing down anyway
    }
    this.cleanupAfterStop();
    this.rejectStop?.(new Error("RecordingEngine: cancelled"));
    this.stopPromise = null;
    this.resolveStop = null;
    this.rejectStop = null;
  }

  private cleanupAfterStop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.audioTap?.disconnect();
    this.audioTap = null;
    this.videoStream?.getTracks().forEach((t) => t.stop());
    this.videoStream = null;
    this.recorder = null;
    // Drop the offscreen canvas — once captureStream's track is stopped the
    // canvas serves no purpose, and letting it go lets the GC reclaim the
    // backing buffer (~8 MB at 1920×1080×4 bytes).
    this.outputCanvas = null;
    this.outputCtx = null;
    this.sourceCanvas = null;
    this.chunks = [];
    this.startTimeMs = 0;
    this.lastDrawAtMs = 0;
  }
}
