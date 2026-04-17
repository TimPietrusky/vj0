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

export class WebRtcAiTransport implements AiTransport {
  private readonly config: WebRtcAiTransportConfig;

  private status: AiTransportStatus = "idle";

  private pc: RTCPeerConnection | null = null;
  private channel: RTCDataChannel | null = null;

  private readonly frameListeners = createListenerSet<FrameListener>();
  private readonly statusListeners = createListenerSet<StatusListener>();

  private opToken = 0;
  private disconnectTimer: number | null = null;

  constructor(config: WebRtcAiTransportConfig) {
    this.config = config;
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
    this.status = status;
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
        this.setStatus("connected");
      };

      channel.onclose = () => {
        if (myToken !== this.opToken) return;
        if (this.status !== "error") this.setStatus("disconnected");
      };

      channel.onerror = () => {
        if (myToken !== this.opToken) return;
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

        const blob =
          event.data instanceof Blob
            ? event.data
            : new Blob([event.data as ArrayBuffer]);
        this.frameListeners.emit({ kind: "image", blob });
      };

      pc.onconnectionstatechange = () => {
        if (myToken !== this.opToken) return;
        const state = pc.connectionState;

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
    this.closeConnections();

    if (this.status !== "idle") {
      this.setStatus("disconnected");
    }
  }

  sendText(message: string): void {
    const ch = this.channel;
    if (!ch || ch.readyState !== "open") return;
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
    if (!ch || ch.readyState !== "open") return;
    ch.send(data);
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
