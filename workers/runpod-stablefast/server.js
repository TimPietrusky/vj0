/**
 * WebRTC server with stable-fast inference integration.
 * Receives frames via WebRTC DataChannel, sends to Python inference, returns generated images.
 */
const express = require("express");
const wrtc = require("@roamhq/wrtc");
const { spawn } = require("child_process");

const PORT = Number(process.env.PORT || 3000);
const INFERENCE_SCRIPT = process.env.INFERENCE_SCRIPT || "./inference_server.py";

// ICE server configuration
function getIceServers() {
  const raw = process.env.ICE_SERVERS_JSON;
  if (!raw) return [{ urls: "stun:stun.l.google.com:19302" }];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed;
  } catch {}
  return [{ urls: "stun:stun.l.google.com:19302" }];
}

// State
let activePc = null;
let activeChannel = null;
let inferenceProcess = null;
let inferenceReady = false;
let pendingRequests = [];

// Start Python inference process
function startInference() {
  console.log("Starting inference process...");
  
  inferenceProcess = spawn("python3", [INFERENCE_SCRIPT], {
    stdio: ["pipe", "pipe", "inherit"],
  });

  inferenceProcess.stdout.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean);
    for (const line of lines) {
      try {
        const msg = JSON.parse(line);
        
        if (msg.log) {
          console.log("[inference]", msg.log);
        }
        
        if (msg.status === "ready") {
          console.log("Inference ready!");
          inferenceReady = true;
          flushPendingRequests();
        }
        
        if (msg.status === "frame" && activeChannel?.readyState === "open") {
          // Send generated image back via WebRTC as binary
          const imgBuffer = Buffer.from(msg.image_base64, "base64");
          activeChannel.send(imgBuffer);
          
          // Also send generation time as text message
          activeChannel.send(JSON.stringify({ 
            type: "stats", 
            gen_time_ms: msg.gen_time_ms,
            width: msg.width,
            height: msg.height
          }));
          console.log(`Frame sent: ${msg.gen_time_ms}ms (${msg.width}x${msg.height})`);
        }
        
        if (msg.status === "error") {
          console.error("[inference error]", msg.message);
        }
      } catch (e) {
        console.log("[inference raw]", line);
      }
    }
  });

  inferenceProcess.on("close", (code) => {
    console.log(`Inference process exited with code ${code}`);
    inferenceReady = false;
  });
}

// Send request to inference process
function sendToInference(request) {
  if (!inferenceProcess || !inferenceReady) {
    console.log("Inference not ready, queuing request");
    pendingRequests.push(request);
    return;
  }
  inferenceProcess.stdin.write(JSON.stringify(request) + "\n");
}

// Flush pending requests when inference becomes ready
function flushPendingRequests() {
  if (pendingRequests.length > 0) {
    console.log(`Flushing ${pendingRequests.length} pending requests`);
    for (const request of pendingRequests) {
      inferenceProcess.stdin.write(JSON.stringify(request) + "\n");
    }
    pendingRequests = [];
  }
}

// Clear queue when new parameters arrive (to avoid old frames with stale settings)
function clearPendingFrames() {
  const oldLength = pendingRequests.length;
  pendingRequests = pendingRequests.filter(req => !req.image_base64); // Keep non-frame requests
  if (oldLength !== pendingRequests.length) {
    console.log(`Cleared ${oldLength - pendingRequests.length} pending frames due to parameter change`);
  }
}

// Close active WebRTC connection
function closeActivePc() {
  if (activeChannel) {
    try {
      activeChannel.close();
    } catch {}
    activeChannel = null;
  }
  if (activePc) {
    try {
      activePc.close();
    } catch {}
    activePc = null;
  }
}

// Wait for ICE gathering
function waitForIceGatheringComplete(pc, timeoutMs) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      pc.removeEventListener("icegatheringstatechange", onState);
      clearTimeout(timer);
      resolve();
    };
    const onState = () => {
      if (pc.iceGatheringState === "complete") finish();
    };
    const timer = setTimeout(finish, timeoutMs);
    pc.addEventListener("icegatheringstatechange", onState);
  });
}

// Express app
const app = express();

// CORS - allow all origins for WebRTC signaling
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    return res.sendStatus(200);
  }
  next();
});

app.use(express.json({ limit: "10mb" }));

app.get("/healthz", (_req, res) => {
  res.json({ ok: true, inferenceReady });
});

app.post("/webrtc/offer", async (req, res) => {
  const body = req.body || {};

  if (!body.sdp || typeof body.sdp.type !== "string" || typeof body.sdp.sdp !== "string") {
    res.status(400).send("Expected body: { sdp: RTCSessionDescriptionInit }");
    return;
  }

  // Single client: replace existing connection
  closeActivePc();

  const pc = new wrtc.RTCPeerConnection({
    iceServers: getIceServers(),
  });
  activePc = pc;

  pc.onconnectionstatechange = () => {
    const s = pc.connectionState;
    console.log("Connection state:", s);
    if (s === "failed" || s === "disconnected" || s === "closed") {
      if (activePc === pc) closeActivePc();
    }
  };

  pc.ondatachannel = (event) => {
    const channel = event.channel;
    channel.binaryType = "arraybuffer";
    activeChannel = channel;
    console.log("DataChannel opened:", channel.label);

    channel.onmessage = (ev) => {
      // Handle incoming data
      if (typeof ev.data === "string") {
        // JSON message (prompt, settings, etc.)
        try {
          const msg = JSON.parse(ev.data);
          // Clear queued frames when parameters change to avoid stale settings
          if (msg.prompt || msg.seed || msg.width || msg.height) {
            clearPendingFrames();
          }
          sendToInference(msg);
        } catch {
          console.log("Invalid JSON from client");
        }
      } else {
        // Binary data (image frame)
        const buffer = Buffer.from(ev.data);
        const base64 = buffer.toString("base64");
        sendToInference({ image_base64: base64 });
      }
    };

    channel.onclose = () => {
      console.log("DataChannel closed");
      if (activeChannel === channel) activeChannel = null;
    };
  };

  try {
    await pc.setRemoteDescription(body.sdp);
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    await waitForIceGatheringComplete(pc, 2000);

    if (!pc.localDescription) {
      res.status(500).send("Missing localDescription");
      return;
    }

    res.json({ sdp: pc.localDescription });
  } catch (err) {
    if (activePc === pc) closeActivePc();
    res.status(500).send(err instanceof Error ? err.message : "WebRTC error");
  }
});

// Start server
startInference();

app.listen(PORT, "0.0.0.0", () => {
  console.log(`VJ WebRTC server listening on 0.0.0.0:${PORT}`);
});
