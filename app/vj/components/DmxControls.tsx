type DmxStatus = "disconnected" | "connecting" | "connected" | "unsupported";

interface DmxControlsProps {
  status: DmxStatus;
  supported: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
  onReconnect: () => void;
}

/**
 * DMX action buttons. Status itself is shown in the top SystemsBar, so we
 * don't repeat it here. While connected we surface two actions:
 *   - ↻ reconnect : close + re-open the same paired device (no browser
 *                   prompt, fixes sleep/tab-conflict hangs without making
 *                   the user re-pair)
 *   - ✕ unpair    : actually release the device (requires re-pair to
 *                   reconnect)
 * While disconnected we show the pair button.
 */
export function DmxControls({
  status,
  supported,
  onConnect,
  onDisconnect,
  onReconnect,
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
      <div className="flex gap-1">
        <button
          onClick={onReconnect}
          className="vj-btn"
          title="Close and re-open the USB connection (no re-pair)"
        >
          ↻ reconnect
        </button>
        <button
          onClick={onDisconnect}
          className="vj-btn vj-btn--danger"
          title="Release the USB device entirely (requires re-pair)"
        >
          ✕ unpair
        </button>
      </div>
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
