type DmxStatus = "disconnected" | "connecting" | "connected" | "unsupported";

interface DmxControlsProps {
  status: DmxStatus;
  supported: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
}

/**
 * Single-button DMX action. Status itself is shown in the top SystemsBar
 * already, so we don't repeat it here — the panel just needs whatever
 * action is appropriate for the current state.
 */
export function DmxControls({
  status,
  supported,
  onConnect,
  onDisconnect,
}: DmxControlsProps) {
  if (!supported) {
    return (
      <div className="text-[10px] uppercase tracking-wider font-mono text-[color:var(--vj-error)]">
        WebUSB unavailable
      </div>
    );
  }

  if (status === "connected") {
    return (
      <button
        onClick={onDisconnect}
        className="vj-btn vj-btn--danger"
        title="Release the USB device"
      >
        ✕ unpair
      </button>
    );
  }

  return (
    <button
      onClick={onConnect}
      disabled={status === "connecting"}
      className="vj-btn vj-btn--live"
      title="Pair with a DMX USB device (browser will prompt)"
    >
      {status === "connecting" ? "scanning…" : "↻ pair dmx"}
    </button>
  );
}
