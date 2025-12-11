import type { AudioFeatures } from "@/src/lib/audio-features";
import { FeatureBar } from "./FeatureBar";

interface AudioDebugPanelProps {
  features: AudioFeatures | null;
}

/**
 * Audio features debug panel - displays audio analysis values.
 * Receives throttled data via props (parent polls at ~10fps).
 */
export function AudioDebugPanel({ features }: AudioDebugPanelProps) {
  return (
    <div className="bg-neutral-900/80 border border-neutral-700 rounded-lg p-4 font-mono text-xs">
      <div className="text-neutral-400 uppercase tracking-wide mb-3">
        Audio Features
      </div>
      {features ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <FeatureBar label="RMS" value={features.rms} />
          <FeatureBar label="Peak" value={features.peak} />
          <FeatureBar
            label="Low"
            value={features.energyLow}
            color="text-red-400"
            barColor="bg-red-500"
          />
          <FeatureBar
            label="Mid"
            value={features.energyMid}
            color="text-yellow-400"
            barColor="bg-yellow-500"
          />
          <FeatureBar
            label="High"
            value={features.energyHigh}
            color="text-cyan-400"
            barColor="bg-cyan-500"
          />
          <FeatureBar
            label="Centroid"
            value={features.spectralCentroid}
            color="text-purple-400"
            barColor="bg-purple-500"
          />
        </div>
      ) : (
        <div className="text-neutral-500">Waiting for audio data...</div>
      )}
    </div>
  );
}
