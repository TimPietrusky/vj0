import { VJWaveform } from './VJWaveform';

export const metadata = {
  title: 'vj0 - Live Audio Waveform',
  description: 'Real-time audio visualization for live visual artists',
};

export default function VJPage() {
  return (
    <main className="min-h-screen bg-black flex flex-col items-center justify-center">
      <header className="py-8">
        <h1 className="text-3xl font-mono font-bold tracking-tight text-emerald-400">
          vj0
        </h1>
      </header>
      <VJWaveform />
    </main>
  );
}

