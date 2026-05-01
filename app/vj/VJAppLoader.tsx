"use client";

import dynamic from "next/dynamic";

// Load VJApp client-only (ssr: false). Two reasons, both important:
//
// 1. Every meaningful thing in this app uses browser APIs (canvas,
//    AudioContext, getUserMedia, getDisplayMedia, WebUSB). There's nothing
//    to gain from server-rendering a tree that immediately needs the DOM
//    to do anything useful.
//
// 2. UI state (panel toggles, audio device, scene, fixtures, prompt presets,
//    AI backend, output size, ...) is persisted via Zustand + localStorage.
//    With SSR on, the server renders DEFAULT values, the client hydrates
//    with the same defaults, then Zustand reads localStorage and triggers
//    a re-render with the persisted values — visible as a sub-second
//    "flash of wrong UI" on every reload (e.g. Audio Features panel
//    opening then collapsing, fixtures appearing late, etc.).
//    Client-only mount means the first paint already has the persisted
//    state, no flash.
//
// The <main> wrapper in app/page.tsx supplies the dark background, so the
// brief loading window is just an empty dark screen — strictly better than
// a flash of incorrect toggles.
export const VJApp = dynamic(
  () => import("./VJApp").then((m) => m.VJApp),
  { ssr: false }
);
