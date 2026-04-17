"use client";

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

interface PopoverProps {
  trigger: (props: { open: boolean; toggle: () => void }) => ReactNode;
  children: ReactNode;
  align?: "left" | "right";
  width?: number;
}

/**
 * Floating popover. The trigger is rendered inline; the panel is portaled to
 * <body> and positioned via getBoundingClientRect of the trigger. Using a
 * portal (instead of an absolutely-positioned sibling) guarantees the panel
 * cannot push, expand, or otherwise affect the layout of its parent — which
 * matters in the dense top bar where chips sit in a flex row and any width
 * leak from the panel would shove the other chips around.
 */
export function Popover({
  trigger,
  children,
  align = "left",
  width = 320,
}: PopoverProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLSpanElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  // Place the panel under the trigger. Re-measure on open + scroll/resize so
  // it tracks the trigger if the page reflows.
  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const t = triggerRef.current;
      if (!t) return;
      const r = t.getBoundingClientRect();
      const left = align === "right" ? r.right - width : r.left;
      // Clamp horizontally so the panel never escapes the viewport.
      const clampedLeft = Math.max(
        8,
        Math.min(window.innerWidth - width - 8, left)
      );
      setPos({ top: r.bottom + 6, left: clampedLeft });
    };
    place();
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open, align, width]);

  // Outside-click + Esc dismiss. Outside means: not in trigger AND not in panel.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (triggerRef.current?.contains(target)) return;
      if (panelRef.current?.contains(target)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <>
      <span ref={triggerRef} style={{ display: "inline-flex" }}>
        {trigger({ open, toggle: () => setOpen((v) => !v) })}
      </span>
      {open &&
        pos &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            ref={panelRef}
            className="vj-panel p-3 shadow-[0_18px_48px_-12px_rgba(0,0,0,0.7)]"
            style={{
              position: "fixed",
              top: pos.top,
              left: pos.left,
              width,
              maxHeight: `calc(100vh - ${pos.top + 16}px)`,
              overflowY: "auto",
              zIndex: 9999,
            }}
          >
            {children}
          </div>,
          document.body
        )}
    </>
  );
}
