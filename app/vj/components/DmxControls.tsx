type DmxStatus = "disconnected" | "connecting" | "connected" | "unsupported";

interface DmxControlsProps {
  status: DmxStatus;
  supported: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
}

/**
 * DMX connection controls - connect/disconnect buttons and status display.
 */
export function DmxControls({
  status,
  supported,
  onConnect,
  onDisconnect,
}: DmxControlsProps) {
  return (
    <div className="flex items-center gap-4 mb-4">
      <div className="flex items-center gap-2">
        <span className="text-neutral-500">DMX:</span>
        <span
          className={`uppercase ${
            status === "connected"
              ? "text-emerald-400"
              : status === "connecting"
              ? "text-amber-400"
              : status === "unsupported"
              ? "text-red-400"
              : "text-neutral-400"
          }`}
        >
          {status}
        </span>
      </div>

      {supported && status !== "connected" && (
        <button
          onClick={onConnect}
          disabled={status === "connecting"}
          className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-neutral-700 text-white rounded text-xs transition-colors"
        >
          Connect DMX
        </button>
      )}

      {status === "connected" && (
        <button
          onClick={onDisconnect}
          className="px-3 py-1 bg-red-600 hover:bg-red-500 text-white rounded text-xs transition-colors"
        >
          Disconnect
        </button>
      )}

      {!supported && (
        <span className="text-red-400 text-xs">
          WebUSB not supported in this browser
        </span>
      )}
    </div>
  );
}
