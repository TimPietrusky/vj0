import { FIXTURE_PROFILES } from "@/src/lib/lighting";

interface FixtureSelectorProps {
  selectedProfileId: string;
  onProfileSelect: (profileId: string) => void;
  onAdd: () => void;
}

/**
 * Compact fixture profile selector + add button. Uses vj-* tokens so it
 * blends with the rest of the dashboard.
 */
export function FixtureSelector({
  selectedProfileId,
  onProfileSelect,
  onAdd,
}: FixtureSelectorProps) {
  return (
    <div className="grid grid-cols-[1fr_auto] gap-2 items-end">
      <select
        value={selectedProfileId}
        onChange={(e) => onProfileSelect(e.target.value)}
        className="vj-input"
        title="Profile to use for the next added fixture"
      >
        {FIXTURE_PROFILES.map((profile) => (
          <option key={profile.id} value={profile.id}>
            {profile.name}
          </option>
        ))}
      </select>
      <button
        onClick={onAdd}
        className="vj-btn vj-btn--accent"
        title="Add a fixture using the selected profile"
      >
        + add
      </button>
    </div>
  );
}
