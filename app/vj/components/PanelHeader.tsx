import type { ReactNode } from "react";

interface PanelHeaderProps {
  /** Card title — rendered with .vj-panel-title styling. */
  title: string;
  /**
   * Right-side action slot. Keep contents compact — single-row, icon-sized
   * buttons or short kbd hints. Big controls (sliders, multi-button toolbars)
   * belong in a separate row below the header so card titles line up across
   * the dashboard.
   */
  actions?: ReactNode;
}

/**
 * Standardised card title bar. Every dashboard card uses this so titles sit
 * at the same baseline regardless of what's on the right. The fixed-height
 * .vj-panel-header rule keeps the row from growing when a card happens to
 * have a button on the right vs. another card with only a tiny kbd hint.
 */
export function PanelHeader({ title, actions }: PanelHeaderProps) {
  return (
    <div className="vj-panel-header">
      <div className="vj-panel-title">{title}</div>
      {actions && <div className="vj-panel-actions">{actions}</div>}
    </div>
  );
}
