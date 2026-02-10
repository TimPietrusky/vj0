# 005_webrtc_poc – Remote WebRTC PoC to GPU worker

## Context

vj0 currently:

- Runs the main **VJ UI on `/`** (Next.js app).
- Has a working **audio engine** (AudioWorklet-based `AudioEngine`).
- Renders visuals at **60 fps** via a scene system on the main canvas.
- Has a **Lighting / DMX** layer sampling that canvas.
- Has (or will have) space in the layout for an **AI Output** panel.

The long-term goal is:

> Use a remote, high-end GPU machine to generate AI visuals and stream them back into vj0 in near real-time.

Before integrating any AI model, we need to prove the hard part:

> A browser running vj0 can connect to a **remote process on a GPU host** via WebRTC, send frames, and receive frames back.

This story is a **pure transport PoC**: WebRTC + remote worker + frame echo.  
No AI model yet. Just “I can talk to a process in the cloud and see the results in my UI”.

---

## Goal of this story

From the vj0 UI at `/` running in the browser:

- Establish a **WebRTC data channel** to a Node process running on a remote GPU host (e.g. RunPod).
- Send small canvas frames over that channel.
- Receive frames back (echoed or trivially modified).
- Display those returned frames in an **AI Output** panel.

Constraints:

- The remote Node process must run on a **separate machine in the cloud**, not locally.
- The remote Node process must be **Dockerized** so it can be run on **RunPod** as a container image.
- WebRTC data channels typically use **UDP** via ICE. Some platforms (including RunPod) do **not** expose inbound UDP to containers.
  - For RunPod, this PoC must support **TURN over TCP/TLS** (relay) as the practical way to make WebRTC work without UDP ingress.
- Signaling happens via HTTP between:
  - Browser → Next.js backend → GPU worker → Next.js backend → Browser.
- This story must handle both:
  - Simple **text messages** (ping/pong), and
  - Small **image frames**.

---

## User story

> As a VJ using vj0  
> I want to connect vj0 to a remote AI worker running on a GPU host  
> So that I can send frames to the cloud and see frames coming back in my UI as proof that real-time remote AI will be possible

---

## High-level architecture

Three actors:

1. **Browser** – vj0 UI at `/` (Next.js front-end).
2. **Next.js backend** – the existing server part of the app.
3. **Remote GPU worker** – **Dockerized** Node process on a GPU pod / VM (RunPod).

Only **browser ↔ GPU worker** speak **WebRTC**.  
The Next.js backend is used **only for signaling via HTTP**, not for media/data transport.

### Browser (vj0 / Next.js front-end)

- Runs the main VJ UI at `/`.
- Renders the main canvas at 60 fps.
- Creates a **WebRtcAiTransport** that:

  - uses `RTCPeerConnection`,
  - creates a `RTCDataChannel` for frames,
  - performs signaling by calling a Next.js API route (`/api/webrtc/offer`).

- Sends frames (compressed from the canvas) over the data channel at low fps (e.g. 2–5 fps).
- Receives frames back via the data channel and displays them in an **AI Output** panel.

### Next.js backend (signaling proxy only)

- Adds a route: `POST /api/webrtc/offer`.
- Accepts an SDP offer from the browser.
- Forwards that offer to the remote GPU worker’s HTTP endpoint.
- Returns the SDP answer from the GPU worker back to the browser.

The Next.js backend does **no WebRTC** and **never sees frame data**. It only forwards SDP JSON.

### Remote GPU worker (Docker image: Node.js + wrtc on GPU host)

- Runs on a remote GPU machine (pod or VM) as a **Docker container** (RunPod target).
- Uses `wrtc` (or equivalent) to:

  - create a `RTCPeerConnection`,
  - accept SDP offers from vj0 via HTTP (`/webrtc/offer`),
  - create a data channel and answer,
  - return the SDP answer to the Next.js proxy.

- On the WebRTC data channel:
  - Receives text messages and echos them back (e.g. ping/pong).
  - Receives small image frames as binary and echos them back to the browser.

This proves we can do a **browser ↔ remote GPU box WebRTC round-trip**.

---

## Architectural requirements

### 1. AiTransport abstraction (browser)

Create `src/lib/ai/transport.ts` with a generic transport interface:

```ts
export type AiIncomingFrame =
  | { kind: "text"; message: string }
  | { kind: "image"; blob: Blob };

export type AiTransportStatus =
  | "idle"
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";

export interface AiTransport {
  start(): Promise<void>;
  stop(): Promise<void>;
  isConnected(): boolean;

  sendText(message: string): void;
  sendImageBlob(blob: Blob): void;

  onFrame(callback: (frame: AiIncomingFrame) => void): void;
  offFrame(callback: (frame: AiIncomingFrame) => void): void;

  onStatusChange(callback: (status: AiTransportStatus) => void): void;
  offStatusChange(callback: (status: AiTransportStatus) => void): void;
}
```

For this story, we only implement a WebRTC-based transport:

- `WebRtcAiTransport` implementing `AiTransport`.

Later, other transports (HTTP, WebSocket, etc.) can reuse this interface.

### 2. WebRtcAiTransport (browser-side WebRTC)

Create `src/lib/ai/webrtc-transport.ts`.

#### 2.1 Configuration

`WebRtcAiTransport` should be created with:

```ts
type WebRtcAiTransportConfig = {
  signalingUrl: string; // e.g. "/api/webrtc/offer"
  iceServers: RTCIceServer[]; // e.g. [{ urls: "stun:stun.l.google.com:19302" }]
};
```

#### 2.2 Signaling flow (browser ↔ Next.js ↔ GPU worker)

Implement `start()`:

1. Set status to `"connecting"`.
2. Create a new `RTCPeerConnection` with `iceServers`.
3. Create a `RTCDataChannel` (label `"frames"`).
4. Attach event listeners:
   - `dataChannel.onopen`, `dataChannel.onclose`, `dataChannel.onmessage`
   - `peerConnection.onicecandidate`
5. Create an SDP offer:

   ```ts
   const offer = await pc.createOffer();
   await pc.setLocalDescription(offer);
   ```

6. Wait for ICE gathering to complete or a short timeout (gather candidates into `localDescription`).
7. Send the local SDP to the Next.js signaling endpoint:

   ```ts
   const res = await fetch(config.signalingUrl, {
     method: "POST",
     headers: { "Content-Type": "application/json" },
     body: JSON.stringify({ sdp: pc.localDescription }),
   });
   const { sdp: answer } = await res.json();
   await pc.setRemoteDescription(answer);
   ```

8. When the data channel `open` event fires, set status to `"connected"`.

Implement `stop()`:

- Close the data channel and peer connection.
- Set status to `"disconnected"`.

Error handling:

- On any error in signaling or PC, set status to `"error"` and close connections.

> Note: For this PoC, trickle ICE can be skipped. The simple “offer → localDescription → send → answer → remoteDescription” flow is fine.

#### 2.3 Message protocol on the data channel

Use a simple protocol:

- **Text messages**: JSON string payload.

  - Browser sending:

    ```ts
    transport.sendText("ping from vj0");
    ```

    Internally:

    ```ts
    channel.send(JSON.stringify({ type: "text", message }));
    ```

  - On incoming string:

    ```ts
    const data = JSON.parse(event.data);
    if (data.type === "text") {
      emit({ kind: "text", message: data.message });
    }
    ```

- **Image messages**: binary payload.

  - Browser sending:

    ```ts
    sendImageBlob(blob) {
      blob.arrayBuffer().then((buffer) => channel.send(buffer));
    }
    ```

  - On incoming binary (`ArrayBuffer`/`Blob`):

    ```ts
    const blob =
      event.data instanceof Blob ? event.data : new Blob([event.data]);
    emit({ kind: "image", blob });
    ```

No header JSON is required for the PoC; all non-string messages are treated as images.

### 3. Next.js signaling endpoint (proxy only)

Create `app/api/webrtc/offer/route.ts`.

Responsibilities:

- Accept `POST` with body `{ sdp: RTCSessionDescriptionInit }` from the browser.
- Forward this to the GPU worker’s signaling endpoint defined via env:
  - `VJ0_WEBRTC_SIGNALING_ENDPOINT` (e.g. `https://gpu-worker.example.com/webrtc/offer`).
- Return the GPU worker’s response (expected `{ sdp: RTCSessionDescriptionInit }`) back to the browser.

Pseudo implementation:

```ts
import { NextRequest } from "next/server";

const workerEndpoint = process.env.VJ0_WEBRTC_SIGNALING_ENDPOINT;

export async function POST(req: NextRequest) {
  if (!workerEndpoint) {
    return new Response("Signaling endpoint not configured", { status: 500 });
  }

  const offer = await req.json(); // expect { sdp: RTCSessionDescriptionInit }

  const res = await fetch(workerEndpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(offer),
  });

  if (!res.ok) {
    return new Response("Remote signaling failed", { status: 502 });
  }

  const answer = await res.json();
  return Response.json(answer);
}
```

Notes:

- The Next.js backend never touches `RTCPeerConnection` or frames.
- It simply relays SDP JSON between browser and GPU worker.

### 4. Remote GPU worker (Node.js + wrtc)

Implement a standalone Node service (repo/folder) to run on the GPU host **as a Docker image**.

Responsibilities:

- Expose `POST /webrtc/offer` for signaling.
- Use `wrtc` to establish WebRTC with the browser.
- Echo data channel messages (text + binary).

#### 4.1 Signaling handler

Pseudocode:

```ts
import express from "express";
import { RTCPeerConnection } from "wrtc";

const app = express();
app.use(express.json());

app.post("/webrtc/offer", async (req, res) => {
  const { sdp } = req.body;

  const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  pc.ondatachannel = (event) => {
    const channel = event.channel;

    channel.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        // Assume JSON text
        channel.send(ev.data); // echo text
      } else {
        // Binary frame echo
        channel.send(ev.data);
      }
    };
  };

  await pc.setRemoteDescription(sdp);

  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);

  // Wait for ICE gathering to complete (or set a short timeout)
  const localDesc = await new Promise<RTCSessionDescriptionInit>((resolve) => {
    if (pc.iceGatheringState === "complete") {
      resolve(pc.localDescription!);
    } else {
      const checkState = () => {
        if (pc.iceGatheringState === "complete") {
          pc.removeEventListener("icegatheringstatechange", checkState);
          resolve(pc.localDescription!);
        }
      };
      pc.addEventListener("icegatheringstatechange", checkState);
      // Optionally: add timeout fallback
    }
  });

  res.json({ sdp: localDesc });
});

app.listen(3000, () => {
  console.log("GPU WebRTC worker listening on :3000");
});
```

For the PoC it is enough that:

- It accepts one connection at a time.
- It echoes messages back over the data channel.

You can refine/scale connection management later.

#### 4.2 Containerization requirements (RunPod-ready)

The remote worker must ship as a Docker image so we can run it on RunPod without SSHing into a machine to install system deps manually.

Minimum requirements:

- The worker folder includes a `Dockerfile` that produces a runnable image.
- The container listens on `0.0.0.0` and respects `PORT` (default 3000).
- The container image includes any required **system libraries** needed for `wrtc` on Linux (varies by distro; assume Debian/Ubuntu base image).
- The container is runnable with:

  - `docker run -p 3000:3000 <image>`

Networking assumptions for this PoC:

- HTTP signaling to the worker must be reachable from the Next.js backend.
- WebRTC requires UDP for ICE/media/data transport; for this PoC we assume the RunPod deployment provides a public/reachable network path for UDP without TURN.
  - If that assumption is false in a given environment, adding TURN is a future story (explicitly out of scope here).

#### 4.3 Remote deployment output (what “done” means)

When this story is done, we must have:

- A published Docker image on DockerHub (or equivalent registry) for the remote worker.
- Clear env configuration values for:
  - `VJ0_WEBRTC_SIGNALING_ENDPOINT` (vj0 server env; points to the worker’s public URL)
  - `PORT` (worker env)
  - Optional: STUN/TURN env hooks (even if we default to `stun:stun.l.google.com:19302`)

---

## Integration with vj0 UI (`/`)

### 1. Wiring WebRtcAiTransport on `/`

On the main page component (e.g. `app/page.tsx`):

- Create a `WebRtcAiTransport` instance after mount:

```ts
const transport = useMemo(
  () =>
    new WebRtcAiTransport({
      signalingUrl: "/api/webrtc/offer",
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    }),
  []
);
```

- Manage its lifecycle with `useEffect` if needed (e.g. clean up on unmount).
- Track status in state:

```ts
const [status, setStatus] = useState<AiTransportStatus>("idle");

useEffect(() => {
  const handler = (s: AiTransportStatus) => setStatus(s);
  transport.onStatusChange(handler);
  return () => transport.offStatusChange(handler);
}, [transport]);
```

- Provide **Connect / Disconnect** buttons wired to `transport.start()` and `transport.stop()`.

### 2. Text ping/pong debug UI

Add a small debug area in the layout:

- Button: `Send ping` → `transport.sendText("ping from vj0")`.
- Subscribe to incoming frames:

```ts
const [logs, setLogs] = useState<string[]>([]);

useEffect(() => {
  const handler = (frame: AiIncomingFrame) => {
    if (frame.kind === "text") {
      setLogs((prev) => [...prev, `← ${frame.message}`].slice(-20));
    }
  };
  transport.onFrame(handler);
  return () => transport.offFrame(handler);
}, [transport]);
```

- When sending ping, optionally also add `→ ping from vj0` to the log.

This proves simple text round-trip via WebRTC to the remote worker.

### 3. Frame echo into AI Output panel

Add logic to send small canvas frames when connected.

- Create an offscreen `captureCanvas` (e.g. 128×128) and draw the main canvas into it at a low fps (2–5 fps):

```ts
useEffect(() => {
  if (!transport.isConnected()) return;

  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = 128;
  captureCanvas.height = 128;
  const captureCtx = captureCanvas.getContext("2d");
  if (!captureCtx) return;

  let stopped = false;

  const sendFrame = () => {
    if (stopped || !transport.isConnected()) return;

    // mainCanvasRef should point to your main scene canvas
    const mainCanvas = mainCanvasRef.current;
    if (mainCanvas) {
      captureCtx.drawImage(
        mainCanvas,
        0,
        0,
        mainCanvas.width,
        mainCanvas.height,
        0,
        0,
        captureCanvas.width,
        captureCanvas.height
      );

      captureCanvas.toBlob(
        (blob) => {
          if (blob) {
            transport.sendImageBlob(blob);
          }
        },
        "image/webp",
        0.7
      );
    }
  };

  const interval = window.setInterval(sendFrame, 500); // 2 fps

  return () => {
    stopped = true;
    window.clearInterval(interval);
  };
}, [transport.isConnected()]);
```

- Handle incoming `image` frames to display them:

```ts
const [aiImageUrl, setAiImageUrl] = useState<string | null>(null);

useEffect(() => {
  const handler = (frame: AiIncomingFrame) => {
    if (frame.kind === "image") {
      const url = URL.createObjectURL(frame.blob);
      setAiImageUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
    }
  };
  transport.onFrame(handler);
  return () => transport.offFrame(handler);
}, [transport]);
```

- In the UI, add an **AI Output (WebRTC echo)** panel:

```tsx
<div className="ai-output-panel">
  <h2>AI Output (WebRTC echo)</h2>
  {aiImageUrl ? (
    <img
      src={aiImageUrl}
      alt="AI echo output"
      className="w-full h-auto object-contain rounded"
    />
  ) : (
    <div className="placeholder">Connect to see remote echo</div>
  )}
</div>
```

Optionally add a toggle `Send frames to remote` to enable/disable the send loop.

---

## UI requirements (on `/`)

Add to the main UI layout:

1. **Connection controls**

   - Button: `Connect Remote AI Worker`
     - Calls `transport.start()`.
   - Button: `Disconnect`
     - Calls `transport.stop()`.
   - Status label showing `AiTransportStatus`:
     - `AI link: idle / connecting / connected / disconnected / error`.

2. **Text debug panel**

   - Button: `Send ping`.
   - Small scrolling log of text messages:
     - `→ ping from vj0`
     - `← pong from GPU worker`

3. **AI Output (WebRTC echo) panel**

   - Displays the last received image frame as `<img>`.
   - Placeholder when no frame has been received.

4. **Optional frame sending toggle**

   - Checkbox or switch: `Send canvas frames to remote`.
   - When OFF, the frame send interval is disabled even if connected.

---

## Non-functional requirements

- **Remote-only assumption**:

  - The GPU worker runs on a remote host.
  - `VJ0_WEBRTC_SIGNALING_ENDPOINT` is a public or reachable URL from the Next.js backend.
  - The GPU worker is deployed as a **Docker image** (RunPod target).
  - If the deployment environment does not support UDP ingress (e.g. RunPod), the connection must work via **TURN relay** (TCP/TLS).

- **Performance & latency**:

  - Frame resolution for this PoC: ~128×128.
  - Format: WebP or JPEG at ~0.7 quality.
  - Target send rate: 2–5 fps.
  - The main 60 fps render loop must remain smooth (no blocking operations in `requestAnimationFrame`).

- **Resilience**:

  - If signaling fails (e.g. worker down), show status `"error"` and do not crash the page.
  - If the connection drops, set status `"disconnected"` and allow reconnecting.

- **Security & config**:

  - The GPU worker URL (`VJ0_WEBRTC_SIGNALING_ENDPOINT`) must live in env, not in client code.
  - Any auth tokens for the GPU worker should only appear in Next.js server code.
  - The Docker image should not bake in secrets; secrets are provided via env at runtime (RunPod).

- **Extensibility**:
  - `AiTransport` abstraction allows adding more transports later.
  - The same WebRTC transport can later be reused for real img2img / video generation in the worker.
  - The AI Output panel will be reused for actual AI frames once the worker runs a model.

---

## Out of scope

Not part of this story:

- Any AI model integration (Stable Diffusion, Flux, etc.).
- Sending audio features or DMX information to the worker.
- Using TURN servers or solving complex NAT traversal (assume STUN + reasonably reachable GPU host).
- (Exception) For RunPod specifically: supporting TURN **as a relay transport option** is required because UDP ingress is not available.
- Authentication & authorization beyond a simple secret env token if desired.
- Multi-client connection management on the worker side (PoC can handle one client).

---

## Deliverables

- `src/lib/ai/transport.ts`:
  - `AiTransport`, `AiIncomingFrame`, `AiTransportStatus` definitions.
- `src/lib/ai/webrtc-transport.ts`:
  - `WebRtcAiTransport` implementing `AiTransport` with a WebRTC data channel.
- `app/api/webrtc/offer/route.ts`:
  - Signaling proxy endpoint that forwards SDP offers to `VJ0_WEBRTC_SIGNALING_ENDPOINT` and returns answers.
- Remote GPU worker **Dockerized** Node service (within this repo as a subfolder):
  - HTTP endpoint `POST /webrtc/offer` accepting `{ sdp }`.
  - Uses `wrtc` to create a `RTCPeerConnection`, accept the browser offer, create an answer, and return it.
  - Echoes text and binary messages on the data channel back to the browser.
  - Includes:
    - `Dockerfile` (RunPod-ready)
    - runtime env support (`PORT`, and any optional STUN/TURN settings)
    - published Docker image (DockerHub)
  - Includes a GitHub Action workflow that builds & pushes the worker image to DockerHub on changes to the worker folder (using repo secrets for auth).
- Updated main page component (`app/page.tsx`):
  - Connection controls and status display.
  - Text ping/pong debug log.
  - Frame sending loop for small canvas frames when connected.
  - AI Output panel showing echoed frames from the remote GPU worker.

When this story is complete, vj0 will have **hard proof** that:

- The browser UI can connect over WebRTC to a **remote** process on a GPU host.
- Frames from the main canvas can travel to the cloud and come back.
- The UI already has a place to show those remote frames, ready for actual AI models in future stories.
  - The remote worker can be deployed on RunPod via a Docker image (no manual machine setup).
