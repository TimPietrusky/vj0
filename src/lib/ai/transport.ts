export type AiIncomingFrame =
  | { kind: "text"; message: string }
  | { kind: "image"; blob: Blob };

export type AiTransportStatus =
  | "idle"
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";

export interface AiTransport {
  start(): Promise<void>;
  stop(): Promise<void>;
  isConnected(): boolean;

  sendText(message: string): void;
  sendImageBlob(blob: Blob): void;
  /** Send raw binary data directly - no allocation */
  sendBinary(data: ArrayBuffer | ArrayBufferView): void;
  /** Get current send buffer size in bytes (for backpressure) */
  getBufferedAmount(): number;
  /** Check if buffer is low enough to send (threshold in bytes) */
  canSend(maxBuffered?: number): boolean;

  onFrame(callback: (frame: AiIncomingFrame) => void): void;
  offFrame(callback: (frame: AiIncomingFrame) => void): void;

  onStatusChange(callback: (status: AiTransportStatus) => void): void;
  offStatusChange(callback: (status: AiTransportStatus) => void): void;
}

