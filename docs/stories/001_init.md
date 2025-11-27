Title: vj0 – base app with real time audio waveform

Context

I am building a new web based VJ tool called vj0. The public site and app will live on vj0.live.

vj0 will later:

- Receive audio from a USB audio interface
- Analyze the audio in real time
- Render visuals to canvas (2D, WebGL, WebGPU) at 60 fps
- Use these visuals as input for cloud based AI image to image models
- Control stage lights via a WebUSB DMX512 controller by sampling pixels from the visuals

For this first iteration we focus only on:

- Getting the audio signal into the browser from the audio interface
- Rendering a smooth waveform at 60 fps
- Making architecture choices that will not block future performance work

I am an experienced web developer and core contributor to modV. I want a clean, modern setup that is easy to extend.

User story

As a live visual artist (VJ)
I want to open vj0 in my browser (vj0.live in the future), select my USB audio interface as audio input, and see a responsive audio waveform rendered in real time
So that I can verify my signal chain and use this as the base for future visual and AI powered effects

Tech stack requirements

- Next.js (latest, App Router), TypeScript, React
- Tailwind is fine for basic layout and styling
- No heavy client side state management library
- Audio and rendering logic must be framework agnostic TypeScript modules
- Use modern browser APIs:
  - Web Audio API for audio input and analysis
  - For now use AnalyserNode
  - Design so that we can replace this with AudioWorklet later
- For rendering use Canvas 2D for this first step
  - Design so that we can later swap to WebGL or WebGPU

Functional requirements

1. App setup

   - Create a new Next.js app named vj0 with TypeScript and App Router
   - Add a route (for example /vj or /app/vj) that hosts the VJ view
   - The route should render a single page with:
     - A full width canvas element for the waveform
     - A simple status indicator (for example “waiting for audio permission”, “running”)
     - Optional: a dropdown to select an audio input device if more than one is available
   - Prepare the code so it can later be deployed under the domain vj0.live (no hardcoded localhost URLs)

2. Audio engine module (src/lib/audio-engine.ts)

   - Implement an AudioEngine class with:
     - init(deviceId?) to:
       - Call getUserMedia with audio constraints
       - Prefer a specific deviceId if provided
       - Disable echoCancellation, noiseSuppression, autoGainControl
       - Create an AudioContext with latencyHint set to “interactive”
       - Create a MediaStreamAudioSourceNode from the input
       - Create an AnalyserNode for time domain data
       - Configure fftSize to a reasonable value (for example 2048)
     - getTimeDomainData(target: Float32Array) that fills the passed buffer with float time domain samples
     - bufferSize getter returning the internal fftSize
     - destroy() that closes the AudioContext and stops the audio tracks
   - The class should not depend on React. It is a pure TypeScript utility.

3. Waveform renderer module (src/lib/waveform-renderer.ts)

   - Implement a WaveformRenderer class with:
     - A constructor that receives an HTMLCanvasElement and acquires a 2D context
       - Set up line width, stroke color, fill color once
     - start(loop: (buffer: Float32Array) => void, buffer: Float32Array)
       - Uses requestAnimationFrame internally
       - On each frame calls the loop callback so the caller can fill the buffer with audio data
       - After the loop callback returns, draw the waveform for the current buffer
     - stop() to cancel the animation frame
   - The renderer should:
     - Clear the canvas with a fill rect every frame
     - Draw a polyline waveform centered vertically
     - Use a fixed internal canvas resolution (e.g. width 1024, height 256) and rely on CSS for visual scaling

4. React integration component (app/vj/page.tsx and a VJWaveform component)

   - Create a client component that:
     - Renders a canvas that fills the available width
     - Uses useRef for the canvas ref
     - Uses useEffect to:
       - Create an instance of AudioEngine
       - Await engine.init() to request microphone permission
       - Configure the canvas internal size (for example width 1024 height 256)
       - Create an instance of WaveformRenderer with the canvas
       - Create a Float32Array buffer sized to engine.bufferSize
       - Call renderer.start, passing a callback that calls engine.getTimeDomainData(buffer)
     - On cleanup:
       - Stop the renderer
       - Destroy the audio engine
   - Important: do not put audio samples into React state
     - No useState for the audio buffer
     - There should be no re render on each frame

5. Device selection (optional but nice)
   - Query available audio input devices via enumerateDevices
   - Expose a simple select element above the canvas listing the available audio inputs
   - When the user selects a device, re initialize the AudioEngine with the chosen deviceId

Non functional requirements

- The waveform should feel smooth at 60 fps on a typical laptop when running in production build
- Avoid allocations in the render loop:
  - Reuse the Float32Array buffer
  - No array map or filter in the hot path
- Keep the public interfaces stable and simple:
  - Later we must be able to:
    - Replace AnalyserNode with AudioWorklet based feature extraction
    - Replace Canvas 2D with WebGL or WebGPU or OffscreenCanvas in a Worker
    - Plug a DMX module that reads a small subset of the visual data without changing the AudioEngine API

Out of scope for this iteration

- AI image generation or cloud GPU integration
- WebRTC or WebCodecs
- WebUSB and DMX control
- Mobile layout fine tuning
- Any design work beyond a functional vj0 branded waveform UI (basic black background with a neon waveform is enough)

Deliverables

- A working Next.js project named vj0 that I can run locally with:
  - `pnpm dev` or `npm run dev`
  - Visit /vj
  - Allow microphone input
  - See a real time waveform from my USB audio interface
- Clear code structure:
  - src/lib/audio-engine.ts
  - src/lib/waveform-renderer.ts
  - app/vj/page.tsx (or similar)
- Minimal comments explaining how to extend:
  - Where to plug in AudioWorklet later
  - Where to swap Canvas 2D for WebGL or WebGPU
  - Where we will later hook in DMX and AI modules
