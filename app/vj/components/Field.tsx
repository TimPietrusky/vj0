import type { ReactNode } from "react";

interface FieldProps {
  label: string;
  children: ReactNode;
  hint?: string;
  tone?: "default" | "warn";
  className?: string;
}

/**
 * Compact label + control wrapper. Used inside clusters of small settings so
 * each control has a tiny uppercase caption without wasting vertical space.
 */
export function Field({
  label,
  children,
  hint,
  tone = "default",
  className = "",
}: FieldProps) {
  const labelColor =
    tone === "warn" ? "text-[color:var(--vj-warn)]" : "text-[color:var(--vj-ink-dim)]";
  return (
    <label
      className={`flex flex-col gap-1 ${className}`}
      title={hint}
    >
      <span
        className={`text-[9px] uppercase tracking-[0.18em] font-semibold ${labelColor}`}
      >
        {label}
      </span>
      {children}
    </label>
  );
}
