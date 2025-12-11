import { FIXTURE_PROFILES } from "@/src/lib/lighting";

interface FixtureSelectorProps {
  selectedProfileId: string;
  onProfileSelect: (profileId: string) => void;
  onAdd: () => void;
}

/**
 * Fixture selector dropdown with add button.
 */
export function FixtureSelector({
  selectedProfileId,
  onProfileSelect,
  onAdd,
}: FixtureSelectorProps) {
  return (
    <div className="flex items-center gap-3 mb-4 border-b border-neutral-700 pb-4">
      <span className="text-neutral-500 text-xs">Add Fixture:</span>
      <select
        value={selectedProfileId}
        onChange={(e) => onProfileSelect(e.target.value)}
        className="flex-1 bg-neutral-900 border border-neutral-600 text-neutral-200 text-xs rounded px-2 py-1.5 focus:outline-none focus:border-emerald-500"
      >
        {FIXTURE_PROFILES.map((profile) => (
          <option key={profile.id} value={profile.id}>
            {profile.name}
          </option>
        ))}
      </select>
      <button
        onClick={onAdd}
        className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs transition-colors"
      >
        Add
      </button>
    </div>
  );
}
