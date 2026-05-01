import { VJApp } from "./vj/VJAppLoader";

export const metadata = {
  title: "vj0 — live audio-reactive visuals",
  description: "Real-time audio visualization for live visual artists",
};

export default function Home() {
  return (
    <main className="min-h-screen w-screen bg-[#0a0a0f] text-neutral-200">
      <VJApp />
    </main>
  );
}
