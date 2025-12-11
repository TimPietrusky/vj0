import type { VjScene } from "@/src/lib/scenes";

type Status = "idle" | "requesting" | "running" | "error";

interface AudioDevice {
  deviceId: string;
  label: string;
}

interface StatusBarProps {
  status: Status;
  scenes: VjScene[];
  currentSceneId: string;
  onSceneChange: (sceneId: string) => void;
  devices: AudioDevice[];
  selectedDeviceId: string;
  onDeviceChange: (deviceId: string) => void;
}

/**
 * Status bar with status indicator, scene selector, and device selector.
 */
export function StatusBar({
  status,
  scenes,
  currentSceneId,
  onSceneChange,
  devices,
  selectedDeviceId,
  onDeviceChange,
}: StatusBarProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-2">
        <span className="text-sm font-mono uppercase tracking-wide text-neutral-400">
          Status:
        </span>
        <span
          className={`text-sm font-mono uppercase tracking-wide ${
            status === "running"
              ? "text-emerald-400"
              : status === "error"
              ? "text-red-400"
              : "text-amber-400"
          }`}
        >
          {status === "idle" && "Initializing"}
          {status === "requesting" && "Requesting audio permission"}
          {status === "running" && "Running"}
          {status === "error" && "Error"}
        </span>
      </div>

      <div className="flex items-center gap-3">
        {/* Scene selector */}
        <select
          value={currentSceneId}
          onChange={(e) => onSceneChange(e.target.value)}
          className="bg-neutral-900 border border-neutral-700 text-neutral-200 text-sm rounded px-3 py-1.5 focus:outline-none focus:border-emerald-500"
        >
          {scenes.map((scene) => (
            <option key={scene.id} value={scene.id}>
              {scene.name}
            </option>
          ))}
        </select>

        {/* Device selector */}
        {devices.length > 1 && (
          <select
            value={selectedDeviceId}
            onChange={(e) => onDeviceChange(e.target.value)}
            className="bg-neutral-900 border border-neutral-700 text-neutral-200 text-sm rounded px-3 py-1.5 focus:outline-none focus:border-emerald-500"
          >
            <option value="">Default device</option>
            {devices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label}
              </option>
            ))}
          </select>
        )}
      </div>
    </div>
  );
}
