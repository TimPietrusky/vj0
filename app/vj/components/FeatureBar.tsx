/**
 * Individual feature bar component for debug display.
 * Pure presentational - receives value via props.
 */
export function FeatureBar({
  label,
  value,
  color = "text-emerald-400",
  barColor = "bg-emerald-500",
}: {
  label: string;
  value: number;
  color?: string;
  barColor?: string;
}) {
  const percentage = Math.min(100, Math.max(0, value * 100));

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-baseline">
        <span className="text-neutral-400">{label}</span>
        <span className={color}>{value.toFixed(3)}</span>
      </div>
      <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} transition-all duration-75`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
