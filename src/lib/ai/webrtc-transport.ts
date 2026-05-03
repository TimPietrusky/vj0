import type {
  AiIncomingFrame,
  AiTransport,
  AiTransportStatus,
} from "./transport";

export type WebRtcAiTransportConfig = {
  signalingUrl: string; // e.g. "/api/webrtc/offer"
  iceServers: RTCIceServer[]; // e.g. [{ urls: "stun:stun.l.google.com:19302" }]
  /**
   * Use "relay" when you want to force TURN (e.g. platforms without UDP ingress).
   * Default: "all"
   */
  iceTransportPolicy?: RTCIceTransportPolicy;
  iceGatheringTimeoutMs?: number;
};

type FrameListener = (frame: AiIncomingFrame) => void;
type StatusListener = (status: AiTransportStatus) => void;

function createListenerSet<T extends (...args: any[]) => void>() {
  const listeners = new Set<T>();
  return {
    add(fn: T) {
      listeners.add(fn);
    },
    remove(fn: T) {
      listeners.delete(fn);
    },
    emit(...args: Parameters<T>) {
      for (const fn of listeners) fn(...args);
    },
    clear() {
      listeners.clear();
    },
  };
}

async function waitForIceGatheringComplete(
  pc: RTCPeerConnection,
  timeoutMs: number
): Promise<void> {
  if (pc.iceGatheringState === "complete") return;

  await new Promise<void>((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      pc.removeEventListener("icegatheringstatechange", onState);
      window.clearTimeout(timer);
      resolve();
    };

    const onState = () => {
      if (pc.iceGatheringState === "complete") finish();
    };

    const timer = window.setTimeout(() => finish(), timeoutMs);
    pc.addEventListener("icegatheringstatechange", onState);
  });
}

/** Live diagnostics — exposed on window.__VJ0_DIAG for console inspection */
export interface WebRtcDiag {
  framesSent: number;
  framesReceived: number;
  bytesSent: number;
  bytesReceived: number;
  settingsSent: number;
  sendSkippedNotConnected: number;
  sendSkippedBackpressure: number;
  sendSkippedPendingEncode: number;
  lastSentAt: number;
  lastReceivedAt: number;
  channelState: string;
  connectionState: string;
  iceState: string;
  bufferedAmount: number;
  status: AiTransportStatus;
  errors: string[];
  stateChanges: Array<{ t: number; from: string; to: string }>;
  uptimeMs: number;
}

export class WebRtcAiTransport implements AiTransport {
  private readonly config: WebRtcAiTransportConfig;

  private status: AiTransportStatus = "idle";

  private pc: RTCPeerConnection | null = null;
  private channel: RTCDataChannel | null = null;

  private readonly frameListeners = createListenerSet<FrameListener>();
  private readonly statusListeners = createListenerSet<StatusListener>();

  private opToken = 0;
  private disconnectTimer: number | null = null;

  // ── diagnostics ──
  readonly diag: WebRtcDiag = {
    framesSent: 0,
    framesReceived: 0,
    bytesSent: 0,
    bytesReceived: 0,
    settingsSent: 0,
    sendSkippedNotConnected: 0,
    sendSkippedBackpressure: 0,
    sendSkippedPendingEncode: 0,
    lastSentAt: 0,
    lastReceivedAt: 0,
    channelState: "none",
    connectionState: "none",
    iceState: "none",
    bufferedAmount: 0,
    status: "idle",
    errors: [],
    stateChanges: [],
    uptimeMs: 0,
  };
  private diagStartedAt = 0;
  private diagTimer: ReturnType<typeof setInterval> | null = null;

  constructor(config: WebRtcAiTransportConfig) {
    this.config = config;
    // Expose on window for live console inspection: window.__VJ0_DIAG
    if (typeof window !== "undefined") {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__VJ0_DIAG = this.diag;
    }
  }

  private diagLog(msg: string) {
    const t = this.diagStartedAt ? ((Date.now() - this.diagStartedAt) / 1000).toFixed(1) : "0";
    console.log(`[VJ0-DIAG t=${t}s] ${msg}`);
  }

  private diagPushError(msg: string) {
    this.diag.errors.push(`${new Date().toISOString()} ${msg}`);
    if (this.diag.errors.length > 50) this.diag.errors.shift();
    this.diagLog(`ERROR: ${msg}`);
  }

  private startDiagLoop() {
    this.diagStartedAt = Date.now();
    this.diagTimer = setInterval(() => {
      const d = this.diag;
      d.uptimeMs = Date.now() - this.diagStartedAt;
      d.channelState = this.channel?.readyState ?? "null";
      d.connectionState = this.pc?.connectionState ?? "null";
      d.iceState = this.pc?.iceConnectionState ?? "null";
      d.bufferedAmount = this.channel?.bufferedAmount ?? 0;
      d.status = this.status;

      const sinceSent = d.lastSentAt ? ((Date.now() - d.lastSentAt) / 1000).toFixed(1) : "never";
      const sinceRecv = d.lastReceivedAt ? ((Date.now() - d.lastReceivedAt) / 1000).toFixed(1) : "never";

      // Alert on stalls
      const alerts: string[] = [];
      if (d.lastSentAt && Date.now() - d.lastSentAt > 5000 && d.framesSent > 0) alerts.push("NO_SEND_5s");
      if (d.lastReceivedAt && Date.now() - d.lastReceivedAt > 10000 && d.framesReceived > 0) alerts.push("NO_RECV_10s");
      if (d.channelState !== "open" && d.framesSent > 0) alerts.push("CH_NOT_OPEN");
      if (d.bufferedAmount > 256 * 1024) alerts.push(`BACKPRESSURE_${(d.bufferedAmount / 1024).toFixed(0)}KB`);

      console.log(
        `[VJ0-DIAG] sent=${d.framesSent} recv=${d.framesReceived} settings=${d.settingsSent} ` +
        `skip:conn=${d.sendSkippedNotConnected} bp=${d.sendSkippedBackpressure} ` +
        `| since: sent=${sinceSent}s recv=${sinceRecv}s ` +
        `| ch=${d.channelState} conn=${d.connectionState} ice=${d.iceState} buf=${d.bufferedAmount} ` +
        (alerts.length > 0 ? `| ${alerts.join(" ")}` : "")
      );
    }, 5000);
  }

  private stopDiagLoop() {
    if (this.diagTimer) {
      clearInterval(this.diagTimer);
      this.diagTimer = null;
    }
  }

  isConnected(): boolean {
    return this.status === "connected" && this.channel?.readyState === "open";
  }

  onFrame(callback: (frame: AiIncomingFrame) => void): void {
    this.frameListeners.add(callback);
  }

  offFrame(callback: (frame: AiIncomingFrame) => void): void {
    this.frameListeners.remove(callback);
  }

  onStatusChange(callback: (status: AiTransportStatus) => void): void {
    this.statusListeners.add(callback);
    callback(this.status);
  }

  offStatusChange(callback: (status: AiTransportStatus) => void): void {
    this.statusListeners.remove(callback);
  }

  private setStatus(status: AiTransportStatus) {
    if (this.status === status) return;
    const prev = this.status;
    this.status = status;
    this.diag.status = status;
    this.diag.stateChanges.push({ t: Date.now(), from: prev, to: status });
    if (this.diag.stateChanges.length > 100) this.diag.stateChanges.shift();
    this.diagLog(`status: ${prev} → ${status}`);
    this.statusListeners.emit(status);
  }

  private closeConnections() {
    if (this.disconnectTimer !== null) {
      window.clearTimeout(this.disconnectTimer);
      this.disconnectTimer = null;
    }

    const channel = this.channel;
    const pc = this.pc;

    this.channel = null;
    this.pc = null;

    if (channel) {
      channel.onopen = null;
      channel.onclose = null;
      channel.onmessage = null;
      channel.onerror = null;
      try {
        channel.close();
      } catch {
        // ignore
      }
    }

    if (pc) {
      pc.onconnectionstatechange = null;
      try {
        pc.close();
      } catch {
        // ignore
      }
    }
  }

  async start(): Promise<void> {
    if (this.status === "connecting" || this.status === "connected") return;

    const myToken = ++this.opToken;
    this.closeConnections();
    this.setStatus("connecting");

    try {
      const pc = new RTCPeerConnection({
        iceServers: this.config.iceServers,
        iceTransportPolicy: this.config.iceTransportPolicy,
      });
      this.pc = pc;

      const channel = pc.createDataChannel("frames");
      channel.binaryType = "arraybuffer";
      this.channel = channel;

      channel.onopen = () => {
        if (myToken !== this.opToken) return;
        this.diagLog("DataChannel OPEN");
        this.setStatus("connected");
        this.startDiagLoop();
      };

      channel.onclose = () => {
        if (myToken !== this.opToken) return;
        this.diagLog(`DataChannel CLOSED (was ${this.status})`);
        this.stopDiagLoop();
        if (this.status !== "error") this.setStatus("disconnected");
      };

      channel.onerror = (ev) => {
        if (myToken !== this.opToken) return;
        const errMsg = (ev as RTCErrorEvent)?.error?.message ?? "unknown";
        this.diagPushError(`DataChannel error: ${errMsg}`);
        this.stopDiagLoop();
        this.setStatus("error");
        this.closeConnections();
      };

      channel.onmessage = (event) => {
        if (myToken !== this.opToken) return;

        if (typeof event.data === "string") {
          try {
            const data = JSON.parse(event.data) as unknown;
            if (
              typeof data === "object" &&
              data !== null &&
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              (data as any).type === "text" &&
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              typeof (data as any).message === "string"
            ) {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              this.frameListeners.emit({
                kind: "text",
                message: (data as any).message,
              });
              return;
            }
          } catch {
            // fall through: treat as plain text
          }
          this.frameListeners.emit({ kind: "text", message: event.data });
          return;
        }

        const raw = event.data instanceof Blob ? event.data : event.data as ArrayBuffer;
        const size = raw instanceof Blob ? raw.size : raw.byteLength;
        this.diag.framesReceived++;
        this.diag.bytesReceived += size;
        this.diag.lastReceivedAt = Date.now();
        const blob = raw instanceof Blob ? raw : new Blob([raw]);
        this.frameListeners.emit({ kind: "image", blob });
      };

      pc.onconnectionstatechange = () => {
        if (myToken !== this.opToken) return;
        const state = pc.connectionState;
        this.diagLog(`PC state: ${state} (ice=${pc.iceConnectionState} ch=${channel.readyState})`);

        if (state === "connected") {
          if (this.disconnectTimer !== null) {
            window.clearTimeout(this.disconnectTimer);
            this.disconnectTimer = null;
          }
          if (channel.readyState === "open") this.setStatus("connected");
          return;
        }

        if (state === "disconnected") {
          // "disconnected" can be transient on lossy networks. Delay downgrade.
          if (this.disconnectTimer !== null) {
            window.clearTimeout(this.disconnectTimer);
          }
          this.disconnectTimer = window.setTimeout(() => {
            if (myToken !== this.opToken) return;
            if (pc.connectionState === "disconnected" && this.status !== "error") {
              this.setStatus("disconnected");
            }
            this.disconnectTimer = null;
          }, 5000);
          return;
        }

        if (state === "failed" || state === "closed") {
          if (this.disconnectTimer !== null) {
            window.clearTimeout(this.disconnectTimer);
            this.disconnectTimer = null;
          }
          if (this.status !== "error") {
            this.setStatus("disconnected");
          }
        }
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      await waitForIceGatheringComplete(
        pc,
        this.config.iceGatheringTimeoutMs ?? 10000
      );
      if (myToken !== this.opToken) return;

      const local = pc.localDescription;
      if (!local) {
        throw new Error("Missing localDescription after createOffer()");
      }

      const res = await fetch(this.config.signalingUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: local }),
      });

      if (!res.ok) {
        throw new Error(`Signaling failed: ${res.status} ${res.statusText}`);
      }

      const json = (await res.json()) as { sdp?: RTCSessionDescriptionInit };
      if (!json?.sdp) {
        throw new Error("Signaling response missing { sdp }");
      }

      await pc.setRemoteDescription(json.sdp);
      if (myToken !== this.opToken) return;
      if (channel.readyState === "open") this.setStatus("connected");
    } catch {
      if (myToken !== this.opToken) return;
      this.setStatus("error");
      this.closeConnections();
    }
  }

  async stop(): Promise<void> {
    this.opToken++;
    this.stopDiagLoop();
    this.closeConnections();

    if (this.status !== "idle") {
      this.setStatus("disconnected");
    }
  }

  sendText(message: string): void {
    const ch = this.channel;
    if (!ch || ch.readyState !== "open") {
      this.diag.sendSkippedNotConnected++;
      return;
    }
    this.diag.settingsSent++;
    ch.send(message);
  }

  sendImageBlob(blob: Blob): void {
    const ch = this.channel;
    if (!ch || ch.readyState !== "open") return;
    void blob.arrayBuffer().then((buffer) => {
      if (!this.channel || this.channel.readyState !== "open") return;
      this.channel.send(buffer);
    });
  }

  /** Send raw binary data directly - no allocation */
  sendBinary(data: ArrayBuffer | ArrayBufferView): void {
    const ch = this.channel;
    if (!ch || ch.readyState !== "open") {
      this.diag.sendSkippedNotConnected++;
      return;
    }
    const size = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
    this.diag.framesSent++;
    this.diag.bytesSent += size;
    this.diag.lastSentAt = Date.now();
    // Newer lib.dom types narrow RTCDataChannel.send to ArrayBufferView<ArrayBuffer>
    // (i.e. typed arrays NOT backed by a SharedArrayBuffer). Narrow by branch
    // so each call site picks the matching overload — runtime accepts either.
    if (data instanceof ArrayBuffer) {
      ch.send(data);
    } else {
      ch.send(data as ArrayBufferView<ArrayBuffer>);
    }
  }

  /** Get current send buffer size in bytes (for backpressure) */
  getBufferedAmount(): number {
    return this.channel?.bufferedAmount ?? 0;
  }

  /** Check if buffer is low enough to send (threshold in bytes) */
  canSend(maxBuffered = 256 * 1024): boolean {
    const ch = this.channel;
    if (!ch || ch.readyState !== "open") return false;
    return ch.bufferedAmount < maxBuffered;
  }
}
